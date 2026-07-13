"""Orchestrator: decomposes complex tasks, schedules workers, aggregates results.

For simple tasks: delegates to existing _dispatch() unchanged.
For complex tasks: estimate → decompose → schedule → execute → aggregate.

Sits above routing and _dispatch(). Transparent for simple tasks.
"""
import concurrent.futures
import hashlib
import json
import logging
import re
import time


from lore.complexity import estimate as estimate_complexity, ComplexityEstimate
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
                 ctx=None, req_logger=None, verifier=None,
                 classifier=None, registry=None):
        self._server = server
        self._router = router
        self._memory = memory
        self._config = config or {}
        self._ctx = ctx
        self._req_logger = req_logger
        self._verifier = verifier
        self._classifier = classifier
        self._registry = registry
        self._classification = None
        self._repo_context = None  # set by process() for SWE-bench tasks

        planning_cfg = self._config.get("planning", {})
        self._decomposer = TaskDecomposer(server, {**planning_cfg, "max_subtasks": self._config.get("max_subtasks", 3)})
        self._complexity_threshold = self._config.get("complexity_threshold", 0.6)
        self._memory_cap_gb = self._config.get("memory_cap_gb", 14)
        self._model_rss = self._config.get("model_rss", {"primary": 5.5, "specialist": 1.1})
        self._parallel_slots = self._config.get("parallel_slots", 3)

        # Dynamic model lifecycle state
        self._specialist_offloaded = False
        dyn_cfg = self._config.get("dynamic_model_lifecycle", {})
        self._dyn_enabled = dyn_cfg.get("enabled", True)
        self._offload_threshold = dyn_cfg.get("offload_threshold", 0.8)

        # Aggregation config
        agg_cfg = self._config.get("aggregation", {})
        self._agg_max_tokens = agg_cfg.get("max_tokens", 4096)
        self._agg_temperature = agg_cfg.get("temperature", 0.5)

    def process(self, query: str, json_mode: bool = False, dispatch_fn=None,
                repo_context=None) -> dict:
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

        self._repo_context = repo_context

        # 2. TOOL_ONLY fast-path → existing dispatch
        if route == "TOOL_ONLY":
            return self._delegate_dispatch(query, json_mode, dispatch_fn, route, confidence)

        # 3. Estimate complexity (skip for SWE-bench — always complex)
        if repo_context is not None:
            est = ComplexityEstimate(
                is_complex=True, confidence=1.0, signals=["swebench"],
                estimated_subtasks=3, suggested_model="primary",
            )
            self._classification = None
        elif self._classifier is not None:
            try:
                classification = self._classifier.classify(query, route)
                est = ComplexityEstimate(
                    is_complex=classification.is_complex,
                    confidence=classification.confidence,
                    signals=(classification.hints or {}).get("signals", []),
                    estimated_subtasks=classification.estimated_subtasks,
                    suggested_model=classification.suggested_model,
                )
                self._classification = classification
                logger.info(f"Classification: {'complex' if classification.is_complex else 'simple'} "
                             f"task={classification.task_type} (conf={classification.confidence:.2f}) "
                             f"source={classification.source}")
            except Exception:
                est = estimate_complexity(query, route)
                self._classification = None
                logger.info(f"Complexity (heuristic fallback): {'complex' if est.is_complex else 'simple'} "
                             f"(conf={est.confidence:.2f}) signals={est.signals}")
        else:
            est = estimate_complexity(query, route)
            self._classification = None
            logger.info(f"Complexity: {'complex' if est.is_complex else 'simple'} "
                         f"(conf={est.confidence:.2f}) signals={est.signals}")

        # 4. Simple → existing dispatch path (skip for SWE-bench tasks)
        if not est.is_complex and repo_context is None:
            return self._delegate_dispatch(query, json_mode, dispatch_fn, route, confidence)

        # 4b. Complex but self-contained → still dispatch direct (skip for SWE-bench)
        if repo_context is None and not self._should_decompose(query, est):
            logger.info("Self-contained task, routing direct despite complex classification")
            return self._delegate_dispatch(query, json_mode, dispatch_fn, route, confidence)

        # 5. Complex → orchestrate
        try:
            result = self._orchestrate(query, est, route, confidence, json_mode, dispatch_fn, repo_context)
            return result
        except Exception as e:
            logger.warning(f"Orchestration failed ({e}), falling back to dispatch")
            return self._delegate_dispatch(query, json_mode, dispatch_fn, route, confidence)
        finally:
            if self._specialist_offloaded:
                try:
                    if not self._server.is_model_running("specialist"):
                        self._server.start_model("specialist")
                        self._specialist_offloaded = False
                        logger.info("Specialist reloaded (finally)")
                except Exception as reload_err:
                    logger.warning(f"Specialist reload failed: {reload_err}")

    @staticmethod
    def _should_decompose(query: str, est: ComplexityEstimate) -> bool:
        """Decide if a complex task actually benefits from decomposition.

        Conservative: when in doubt, decompose. Only skip decomposition for
        clearly self-contained single-function tasks where splitting adds noise.
        """
        stripped = query.strip()

        # HumanEval-style: starts with Python code (imports + def/class)
        # These are self-contained function definitions that don't benefit from decomposition
        if re.match(r'^(from |import )', stripped) and re.search(r'^def |^class ', stripped, re.MULTILINE):
            return False

        # Bare def/class at start = single function implementation
        if re.match(r'^(def |class )', stripped):
            return False

        # "Write/Create/Implement a function/method" — but ONLY if it's a single
        # self-contained function, not "write a function AND test it AND document it"
        if re.match(r'^(Write|Create|Implement|Build)\s+(a|an)\s+(function|method|class)',
                    stripped, re.IGNORECASE):
            # Multi-part indicators mean this is NOT a single function
            if not re.search(r'\b(and then|also|plus|additionally|as well as)\b',
                             stripped, re.IGNORECASE):
                return False

        # Default: trust the complexity estimator
        return est.is_complex

    def reset_state(self) -> None:
        """Clear per-task state. Call between independent tasks (e.g. benchmark).

        Clears accumulated episodic memory entries and stale classification
        to prevent retrieval overhead and state bleed across independent tasks.
        """
        self._classification = None
        if self._memory is not None:
            try:
                self._memory.clear()
            except Exception:
                pass

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

    def _orchestrate(self, query, est, route, confidence, json_mode, dispatch_fn=None,
                     repo_context=None) -> dict:
        """Full orchestration: decompose → schedule → execute → aggregate."""
        t0 = time.time()

        # 1. Decompose (pass classifier hints if available)
        hints = None
        if self._classification is not None:
            hints = {
                "task_type": self._classification.task_type,
                "estimated_subtasks": self._classification.estimated_subtasks,
                "suggested_model": self._classification.suggested_model,
                **self._classification.hints,
            }
        if repo_context is not None:
            hints = hints or {}
            hints["swebench"] = True
        t_decompose_start = time.time()
        plan = self._decomposer.decompose(query, hints=hints)
        decompose_ms = (time.time() - t_decompose_start) * 1000

        # Skip orchestration on fallback plan (planning failed) → delegate to dispatch
        if plan.is_fallback:
            logger.info("Fallback plan (planning failed), delegating to dispatch")
            return self._delegate_dispatch(query, json_mode, dispatch_fn, route, confidence)
        logger.info(f"Plan: {len(plan.subtasks)} subtasks, "
                     f"~{plan.total_estimated_tokens} tokens estimated")
        for st in plan.subtasks:
            logger.info(f"  {st.id} ({st.model}, {st.context_budget}tok): {st.description[:80]}")

        # 2. Dynamic model lifecycle: offload specialist if all primary-only
        self._maybe_offload_specialist(plan)

        # 3. Schedule: topological sort → waves
        waves = self._build_waves(plan)
        if waves is None:
            logger.warning("Invalid plan (cycle or missing dependency), falling back to direct")
            raise RuntimeError("invalid plan topology")
        # Convert ID waves to SubTask waves for execution
        by_id = {st.id: st for st in plan.subtasks}
        subtask_waves = [[by_id[sid] for sid in wave] for wave in waves]
        logger.info(f"Schedule: {len(waves)} waves")

        # 4. Execute waves
        results: dict[str, WorkerResult] = {}
        wave_num = 0
        for wave in subtask_waves:
            wave_num += 1
            wave_results = self._execute_wave(wave, results, repo_context)
            results.update(wave_results)
            logger.info(f"Wave {wave_num} done: {len(wave_results)} subtasks completed")

        if not results:
            logger.warning("No subtasks completed within budget, falling back to direct dispatch")
            return self._delegate_dispatch(query, json_mode, dispatch_fn, route, confidence)

        # 5. Reload specialist if it was offloaded
        self._maybe_reload_specialist()

        # 6. Aggregate
        all_success = all(r.success for r in results.values())
        if not all_success:
            errors = [r.error for r in results.values() if not r.success]
            logger.warning(f"Some subtasks failed: {errors}")

        t_agg_start = time.time()
        agg_content = self._aggregate(query, plan, results)
        aggregate_ms = (time.time() - t_agg_start) * 1000

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

        # 8. Build metrics
        execute_ms = sum(r.latency_ms for r in results.values())
        needs_aggregation = True  # always attempt aggregation
        metrics = {
            "decompose_ms": round(decompose_ms),
            "execute_ms": round(execute_ms),
            "aggregate_ms": round(aggregate_ms),
            "total_ms": round(latency),
            "subtasks": len(plan.subtasks),
            "waves": len(waves),
            "llm_calls": 1 + len(results) + 1,  # decompose + workers + aggregate
            "partial_results": sum(1 for r in results.values() if r.error == "timeout_with_partial_output"),
        }
        logger.info(f"Orchestration metrics: {json.dumps(metrics)}")

        # 9. Log request
        if self._req_logger is not None:
            try:
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

        # 10. Return result dict (same shape as _dispatch + extra fields)
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
            "metrics": metrics,
            "subtask_results": {sid: r.content for sid, r in results.items()},
        }

    def _build_waves(self, plan: TaskPlan) -> list[list[str]] | None:
        """Topologically sort subtasks into waves. Returns None if cyclic.

        Uses Kahn's algorithm. Each wave contains subtask IDs whose
        dependencies are all satisfied by previous waves. Returns None
        when a cycle or missing dependency is detected.
        """
        # Build adjacency and in-degree
        deps: dict[str, set[str]] = {st.id: set(st.dependencies) for st in plan.subtasks}
        all_ids = set(deps.keys())

        # Missing dependency → invalid plan
        for sid, dset in deps.items():
            for d in dset:
                if d not in all_ids:
                    return None

        in_degree = {sid: len(dset) for sid, dset in deps.items()}
        queue = sorted(sid for sid, deg in in_degree.items() if deg == 0)
        waves: list[list[str]] = []
        visited = 0

        while queue:
            waves.append(queue)
            next_queue: list[str] = []
            for sid in queue:
                visited += 1
                for other_sid, dset in deps.items():
                    if sid in dset:
                        in_degree[other_sid] -= 1
                        if in_degree[other_sid] == 0:
                            next_queue.append(other_sid)
            queue = sorted(next_queue)

        if visited != len(deps):
            return None  # cycle detected

        return waves

    def _execute_wave(self, wave: list[SubTask],
                      prior_results: dict[str, WorkerResult],
                      repo_context=None) -> dict[str, WorkerResult]:
        """Execute a wave of subtasks in parallel.

        With -np N on the primary server, multiple subtasks on the same model
        run concurrently through separate slots. All subtasks in a wave launch
        simultaneously via ThreadPoolExecutor.
        """
        # Consult registry for benchmark-driven model selection
        if self._registry is not None and self._classification is not None:
            task_type = self._classification.task_type
            try:
                resolved = self._registry.get_model_for_task(task_type)
                if resolved:
                    for subtask in wave:
                        subtask.model = resolved
            except Exception:
                pass

        results: dict[str, WorkerResult] = {}

        def _run_subtask(subtask: SubTask) -> tuple[str, WorkerResult]:
            prev_outputs = self._collect_prev_outputs(subtask, prior_results)
            worker = Worker(subtask, self._server, memory=None, repo_context=repo_context)
            return subtask.id, worker.run_with_retry(previous_outputs=prev_outputs)

        # Run all subtasks in parallel (up to parallel_slots)
        max_workers = min(len(wave), self._parallel_slots) if len(wave) > 1 else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_subtask, st): st.id for st in wave}
            for future in concurrent.futures.as_completed(futures):
                try:
                    sid, result = future.result()
                except Exception as exc:
                    sid = futures[future]
                    result = WorkerResult(
                        subtask_id=sid, content="", success=False,
                        latency_ms=0.0, tokens_used=0, model="unknown",
                        error=str(exc),
                    )
                    logger.error(f"Subtask {sid} raised: {exc}")
                results[sid] = result

        return results

    def _collect_prev_outputs(self, subtask: SubTask,
                              prior_results: dict[str, WorkerResult]) -> dict[str, str] | None:
        """Collect outputs from dependencies for a subtask."""
        if not subtask.depends_on_outputs:
            return None
        prev: dict[str, str] = {}
        for dep_id in subtask.dependencies:
            if dep_id in prior_results:
                prev[dep_id] = prior_results[dep_id].content
        return prev if prev else None

    def _check_slot_activity(self, model: str) -> list[dict]:
        """Check /slots endpoint for active generation.

        Returns list of active slot dicts (is_processing=True).
        Used for intelligent supervision: if a subtask fails but slots
        show active generation, partial output is usable.
        """
        slots = self._server.get_slots(model)
        active = [s for s in slots if s.get("is_processing", False)]
        return active

    @staticmethod
    def _truncate_output(text: str, max_tokens: int = 500) -> str:
        """Truncate text to ~max_tokens (first half + last half)."""
        # ~4 chars/token approximation
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n...\n[truncated]\n...\n" + text[-half:]

    def _aggregate(self, query: str, plan: TaskPlan,
                   results: dict[str, WorkerResult]) -> str:
        """Aggregate subtask results into a coherent final response.

        For 4+ subtasks with large outputs, uses progressive (tree-based)
        aggregation to reduce context per call. For large outputs, uses
        specialist pre-summarization before aggregation.
        """
        # Fast path: all code, no dependencies → just concatenate (no LLM call)
        all_code = all(st.output_format in ("code_python", "code_bash") for st in plan.subtasks)
        no_deps = all(not st.dependencies for st in plan.subtasks)
        if all_code and no_deps and len(results) > 1:
            logger.info("Fast aggregation: code-only plan, concatenating (no LLM call)")
            parts = []
            for st in plan.subtasks:
                r = results.get(st.id)
                if r and r.success:
                    parts.append(f"# --- {st.description[:80]} ---\n{r.content}")
                elif r and r.content:
                    parts.append(f"# --- {st.description[:80]} (partial) ---\n{r.content}")
            return "\n\n".join(parts)

        # Pre-summarize long outputs using specialist
        summaries = self._pre_summarize_for_aggregation(results)

        # Fast path: 2 subtasks, both short outputs → concatenate (no LLM call)
        if len(plan.subtasks) == 2:
            total_chars = sum(len(summaries.get(st.id, "")) for st in plan.subtasks)
            if total_chars < 1000:
                logger.info("Fast aggregation: 2 short subtasks, concatenating")
                parts = []
                for st in plan.subtasks:
                    content = summaries.get(st.id, "")
                    if content:
                        parts.append(f"### {st.description[:60]}\n{content}")
                return "\n\n".join(parts)

        # Progressive aggregation for 4+ subtasks
        if len(plan.subtasks) >= 4:
            return self._aggregate_progressive(query, plan, summaries)

        # Standard aggregation for small plans
        parts = []
        total_chars = sum(len(summaries.get(st.id, "")) for st in plan.subtasks if st.id in summaries)
        truncate = total_chars > 2000 * 4
        for st in plan.subtasks:
            content = summaries.get(st.id, "")
            if content:
                if truncate:
                    content = self._truncate_output(content)
                parts.append(f"### {st.id}: {st.description}\n{content}")
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
            return results_text

    def _pre_summarize_for_aggregation(self, results: dict[str, WorkerResult]) -> dict[str, str]:
        """Use specialist to summarize long subtask outputs before aggregation.

        Outputs >3000 chars get summarized to ~200 tokens by the specialist.
        Outputs 1000-3000 chars get truncated (cheaper than LLM call).
        Short outputs pass through unchanged. Falls back to truncation on error.
        """
        summaries: dict[str, str] = {}
        for sid, result in results.items():
            if not result.success:
                summaries[sid] = result.content
                continue
            if len(result.content) > 3000:
                try:
                    resp = self._server.chat(
                        "specialist",
                        [
                            {"role": "system", "content": "Summarize the following in 200 tokens or less. Keep all key details, code snippets, and conclusions."},
                            {"role": "user", "content": result.content},
                        ],
                        max_tokens=300,
                        temperature=0.1,
                    )
                    summaries[sid] = resp["choices"][0]["message"]["content"]
                except Exception:
                    summaries[sid] = self._truncate_output(result.content, 200)
            elif len(result.content) > 1000:
                summaries[sid] = self._truncate_output(result.content, 500)
            else:
                summaries[sid] = result.content
        return summaries

    def _aggregate_progressive(self, query: str, plan: TaskPlan,
                               summaries: dict[str, str]) -> str:
        """Tree-based aggregation: reduce N results to 1 via log(N) calls.

        Pairs subtask outputs and aggregates each pair, then aggregates
        the pairs, until one result remains. Reduces per-call context from
        O(N x subtask_size) to O(2 x subtask_size).
        """
        parts = [
            (st.description, summaries.get(st.id, ""))
            for st in plan.subtasks if st.id in summaries
        ]

        agg_prompt = plan.aggregation_prompt or get_template("aggregation")

        while len(parts) > 2:
            next_parts: list[tuple[str, str]] = []
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    combined = self._aggregate_pair(query, parts[i], parts[i + 1])
                    next_parts.append(("Combined result", combined))
                else:
                    next_parts.append(parts[i])
            parts = next_parts

        # Final aggregation
        results_text = "\n\n".join(
            f"### {desc}\n{content}" for desc, content in parts
        )
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
            logger.info(f"Progressive aggregation complete: {len(content)} chars")
            return content
        except Exception as e:
            logger.warning(f"Final aggregation failed ({e}), concatenating")
            return "\n\n".join(content for _, content in parts)

    def _aggregate_pair(self, query: str,
                        part_a: tuple[str, str],
                        part_b: tuple[str, str]) -> str:
        """Aggregate two subtask outputs into a combined summary."""
        desc_a, content_a = part_a
        desc_b, content_b = part_b
        messages = [
            {"role": "system", "content": "You are combining two subtask results. Integrate them coherently. Remove redundancy. Keep all important details."},
            {"role": "user", "content": f"Task: {query}\n\nPart 1 ({desc_a}):\n{content_a}\n\nPart 2 ({desc_b}):\n{content_b}"},
        ]
        try:
            result = self._server.chat(
                "primary",
                messages,
                max_tokens=self._agg_max_tokens,
                temperature=self._agg_temperature,
            )
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Pair aggregation failed ({e}), concatenating")
            return f"{content_a}\n\n{content_b}"

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
            self._specialist_offloaded = False
            logger.info("Reloaded specialist after orchestration")
        except Exception as e:
            logger.warning(f"Specialist reload failed: {e}")

    def set_memory(self, memory) -> None:
        """Update the memory reference (used after session switch in REPL)."""
        self._memory = memory
