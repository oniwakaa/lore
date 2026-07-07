"""Orchestrator: decomposes complex tasks, schedules workers, aggregates results.

For simple tasks: delegates to existing _dispatch() unchanged.
For complex tasks: estimate → decompose → schedule → execute → aggregate.

Sits above routing and _dispatch(). Transparent for simple tasks.
"""
import logging
import time
from collections import defaultdict, deque

from lore.complexity import estimate as estimate_complexity
from lore.decomposer import TaskDecomposer, TaskPlan, SubTask
from lore.worker import Worker, WorkerResult
from lore.templates import get_template

logger = logging.getLogger(__name__)


class Orchestrator:
    """Decomposes complex tasks, schedules workers, aggregates results.

    For simple tasks: delegates to existing _dispatch() unchanged.
    For complex tasks: decompose → schedule → execute → aggregate.
    """

    def __init__(self, server, router, memory, config: dict | None = None,
                 ctx=None, req_logger=None, verifier=None):
        self._server = server
        self._router = router
        self._memory = memory
        self._config = config or {}
        self._ctx = ctx
        self._req_logger = req_logger
        self._verifier = verifier

        planning_cfg = self._config.get("planning", {})
        self._decomposer = TaskDecomposer(server, {**planning_cfg, "max_subtasks": self._config.get("max_subtasks", 5)})
        self._complexity_threshold = self._config.get("complexity_threshold", 0.6)
        self._memory_cap_gb = self._config.get("memory_cap_gb", 14)
        self._model_rss = self._config.get("model_rss", {"primary": 5.5, "specialist": 1.1})

        # Dynamic model lifecycle state
        self._specialist_offloaded = False
        dyn_cfg = self._config.get("dynamic_model_lifecycle", {})
        self._dyn_enabled = dyn_cfg.get("enabled", True)
        self._offload_threshold = dyn_cfg.get("offload_threshold", 0.8)

        # Aggregation config
        agg_cfg = self._config.get("aggregation", {})
        self._agg_max_tokens = agg_cfg.get("max_tokens", 4096)
        self._agg_temperature = agg_cfg.get("temperature", 0.5)

    def process(self, query: str, json_mode: bool = False, dispatch_fn=None) -> dict:
        """Main entry point. Routes simple tasks, orchestrates complex ones.

        Returns dict with same shape as _dispatch(): route, confidence, model,
        content, success, latency_ms, plus orchestrated, subtasks_completed.
        """
        t0 = time.time()

        # 1. Route the query
        try:
            route, confidence = self._router.classify(query)
        except Exception:
            route, confidence = "PRIMARY", 0.0

        # 2. TOOL_ONLY fast-path → existing dispatch
        if route == "TOOL_ONLY":
            return self._delegate_dispatch(query, json_mode, dispatch_fn, route, confidence)

        # 3. Estimate complexity
        est = estimate_complexity(query, route)
        logger.info(f"Complexity: {'complex' if est.is_complex else 'simple'} "
                     f"(conf={est.confidence:.2f}) signals={est.signals}")

        # 4. Simple → existing dispatch path
        if not est.is_complex:
            return self._delegate_dispatch(query, json_mode, dispatch_fn, route, confidence)

        # 5. Complex → orchestrate
        try:
            result = self._orchestrate(query, est, route, confidence, json_mode)
            return result
        except Exception as e:
            logger.warning(f"Orchestration failed ({e}), falling back to dispatch")
            return self._delegate_dispatch(query, json_mode, dispatch_fn, route, confidence)

    def _delegate_dispatch(self, query, json_mode, dispatch_fn, route, confidence) -> dict:
        """Call existing _dispatch() for simple tasks via dispatch_fn closure.

        dispatch_fn is always provided by the caller (cli.py). No fallback
        import — that would create a circular dependency (cli imports
        orchestrator, orchestrator imports cli).
        """
        if dispatch_fn is not None:
            r = dispatch_fn(query, json_mode=json_mode)
            r.setdefault("orchestrated", False)
            r.setdefault("subtasks_completed", 0)
            return r

        # No dispatch_fn — cannot delegate
        return {
            "route": route, "confidence": confidence, "model": "primary",
            "content": "Error: no dispatch function provided",
            "success": False, "latency_ms": 0.0,
            "orchestrated": False, "subtasks_completed": 0,
        }

    def _orchestrate(self, query, est, route, confidence, json_mode) -> dict:
        """Full orchestration: decompose → schedule → execute → aggregate."""
        t0 = time.time()

        # 1. Decompose
        plan = self._decomposer.decompose(query)
        logger.info(f"Plan: {len(plan.subtasks)} subtasks, "
                     f"~{plan.total_estimated_tokens} tokens estimated")
        for st in plan.subtasks:
            logger.info(f"  {st.id} ({st.model}, {st.context_budget}tok): {st.description[:80]}")

        # 2. Dynamic model lifecycle: offload specialist if all primary-only
        self._maybe_offload_specialist(plan)

        # 3. Schedule: topological sort → waves
        waves = self._build_waves(plan.subtasks)
        logger.info(f"Schedule: {len(waves)} waves")

        # 4. Execute waves
        results: dict[str, WorkerResult] = {}
        wave_num = 0
        for wave in waves:
            wave_num += 1
            wave_results = self._execute_wave(wave, results)
            results.update(wave_results)
            logger.info(f"Wave {wave_num} done: {len(wave_results)} subtasks completed")

        # 5. Reload specialist if it was offloaded
        self._maybe_reload_specialist()

        # 6. Aggregate
        all_success = all(r.success for r in results.values())
        if not all_success:
            errors = [r.error for r in results.values() if not r.success]
            logger.warning(f"Some subtasks failed: {errors}")

        agg_content = self._aggregate(query, plan, results)

        # 7. Store to memory
        if self._memory is not None:
            try:
                self._memory.episodic.store_summary(
                    f"Orchestrated task: {query[:100]}. "
                    f"Used {len(plan.subtasks)} subtasks. "
                    f"Summary: {agg_content[:200]}"
                )
            except Exception as e:
                logger.debug(f"Memory store failed: {e}")

        latency = (time.time() - t0) * 1000

        # 8. Log request
        if self._req_logger is not None:
            try:
                import hashlib
                self._req_logger.log_request({
                    "input_hash": f"sha256:{hashlib.sha256(query.encode()).hexdigest()[:16]}",
                    "route": route,
                    "confidence": confidence,
                    "model": "orchestrated",
                    "tokens_out": len(agg_content.split()),
                    "latency_ms": int(latency),
                    "success": all_success,
                    "orchestrated": True,
                    "subtasks_completed": len(results),
                    "plan_subtasks": len(plan.subtasks),
                    "waves": len(waves),
                })
            except Exception:
                pass

        # 9. Return result dict (same shape as _dispatch + extra fields)
        return {
            "route": route,
            "confidence": confidence,
            "model": "orchestrated",
            "content": agg_content,
            "success": all_success,
            "latency_ms": latency,
            "orchestrated": True,
            "subtasks_completed": len(results),
            "plan": plan,
        }

    def _build_waves(self, subtasks: list[SubTask]) -> list[list[SubTask]]:
        """Topological sort subtasks into execution waves.

        Each wave contains subtasks with all dependencies satisfied by
        previous waves. Within a wave, subtasks on different models CAN
        run in parallel (but we serialize them here — 1 slot per server
        means parallel only when different models).

        Returns list of waves (each wave is a list of SubTasks).
        """
        # Build dependency graph
        deps: dict[str, list[str]] = {st.id: list(st.dependencies) for st in subtasks}
        by_id: dict[str, SubTask] = {st.id: st for st in subtasks}

        waves: list[list[SubTask]] = []
        completed: set[str] = set()

        remaining = set(by_id.keys())

        while remaining:
            # Find all subtasks whose deps are satisfied
            ready = [sid for sid in remaining if all(d in completed for d in deps[sid])]
            if not ready:
                # Circular dependency or missing dep — just take remaining in order
                logger.warning(f"Circular dependency detected, forcing remaining: {remaining}")
                ready = list(remaining)

            # Group ready subtasks by model to enable parallel detection
            # Within a wave, different models can run in parallel
            wave = [by_id[sid] for sid in ready]
            waves.append(wave)

            for sid in ready:
                remaining.discard(sid)
                completed.add(sid)

        return waves

    def _execute_wave(self, wave: list[SubTask],
                      prior_results: dict[str, WorkerResult]) -> dict[str, WorkerResult]:
        """Execute a wave of subtasks.

        Subtasks on different models can run in parallel (different servers).
        Subtasks on the same model run sequentially (1 slot per server).

        For simplicity and safety, we execute sequentially. The wave grouping
        already ensures independent subtasks are in the same wave — the
        execution order within a wave doesn't affect correctness.
        """
        results: dict[str, WorkerResult] = {}

        for subtask in wave:
            # Collect outputs from dependencies
            prev_outputs: dict[str, str] = {}
            if subtask.depends_on_outputs:
                for dep_id in subtask.dependencies:
                    if dep_id in prior_results:
                        prev_outputs[dep_id] = prior_results[dep_id].content

            worker = Worker(subtask, self._server, memory=None)
            result = worker.run(previous_outputs=prev_outputs if prev_outputs else None)
            results[subtask.id] = result

        return results

    def _aggregate(self, query: str, plan: TaskPlan,
                   results: dict[str, WorkerResult]) -> str:
        """Aggregate subtask results into a coherent final response."""
        # Build formatted results text
        parts = []
        for st in plan.subtasks:
            r = results.get(st.id)
            if r:
                parts.append(f"### {st.id}: {st.description}\n{r.content}")
        results_text = "\n\n".join(parts)

        agg_prompt = plan.aggregation_prompt or get_template("aggregation")
        messages = [
            {"role": "system", "content": agg_prompt},
            {"role": "user", "content": f"Original task: {query}\n\nResults:\n{results_text}"},
        ]

        try:
            result = self._server.chat(
                "primary",
                messages,
                max_tokens=self._agg_max_tokens,
                temperature=self._agg_temperature,
            )
            content = result["choices"][0]["message"]["content"]
            logger.info(f"Aggregation complete: {len(content)} chars")
            return content
        except Exception as e:
            logger.warning(f"Aggregation call failed ({e}), concatenating results")
            # Fallback: just concatenate
            return results_text

    # ─── Dynamic Model Lifecycle ───────────────────────────────────────────

    def _maybe_offload_specialist(self, plan: TaskPlan) -> None:
        """If all subtasks use primary only, offload specialist to free memory.

        Only fires when >80% of subtasks are primary-only (offload_threshold).
        The swap takes 5-10s, so only worth it when specialist is truly unused.
        """
        if not self._dyn_enabled:
            return

        models_needed = {s.model for s in plan.subtasks}
        primary_count = sum(1 for s in plan.subtasks if s.model == "primary")
        total = len(plan.subtasks)
        primary_ratio = primary_count / total if total > 0 else 0

        if primary_ratio >= self._offload_threshold and self._server.is_model_running("specialist"):
            try:
                self._server.stop_model("specialist")
                self._specialist_offloaded = True
                logger.info("Offloaded specialist (all primary-only plan)")
            except Exception as e:
                logger.warning(f"Specialist offload failed: {e}")

    def _maybe_reload_specialist(self) -> None:
        """Reload specialist if it was offloaded during orchestration."""
        if not self._specialist_offloaded:
            return
        try:
            self._server.start_model("specialist")
            logger.info("Reloaded specialist after orchestration")
        except Exception as e:
            logger.warning(f"Specialist reload failed: {e}")
        self._specialist_offloaded = False
