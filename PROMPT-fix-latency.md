# Prompt: Fix Orchestration Latency — Timeout, Retry, and Call Reduction

## Identity & Context

You are working on **LORE** (Local Orchestration & Runtime Engine) at `~/projects/lore`. Run with `PYTHONPATH=src`.

LORE's orchestration engine works correctly (100% pass rate on complex tasks) but is catastrophically slow. One task took 1675s (28 minutes). A direct single-model call would take 60-120s. The root cause is a timeout-retry death spiral and excessive sequential LLM calls.

## The Problem

A orchestrated task on Ornith-9B (~15 tok/s) makes 4-6 sequential LLM calls. Each can take 60-120s. That's 4-7 minutes minimum — acceptable. But the retry logic makes it catastrophic:

```
Current death spiral:
  Call → 300s timeout → DOUBLE max_tokens → retry → 300s timeout → DOUBLE again → retry → 300s timeout
  Total: 900s wasted on ONE subtask, still failed
```

**The problem isn't the timeout duration. The problem is retrying with MORE tokens after a timeout.** If a model can't generate 2048 tokens in 300s, asking for 4096 tokens will take even longer.

## 7 Fixes

### Fix 1: No Retry on Timeout — Use Partial Output or Skip (CRITICAL)

In `src/lore/worker.py`, rewrite `run_with_retry()`:

```python
def run_with_retry(self, max_retries: int = 1) -> WorkerResult:
    """Run subtask. Retry on generation errors, NOT on timeouts.
    
    On timeout: the model was too slow. More tokens won't help.
    Take partial output (if server returned any) or mark as failed.
    
    On generation error (crash, parse failure): retry with more context
    and escalate model if needed.
    """
    result = self._run_once()
    
    # Success on first try — done
    if result.success:
        return result
    
    # Check if it was a timeout
    is_timeout = result.error and "timeout" in str(result.error).lower()
    
    if is_timeout:
        # DO NOT retry on timeout. The model was too slow.
        # If we got partial output, use it (better than nothing).
        if result.content and len(result.content) > 100:
            logger.warning(f"Subtask {self._subtask.id} timed out but has partial output ({len(result.content)} chars), using it")
            return WorkerResult(
                subtask_id=result.subtask_id,
                content=result.content,
                success=True,  # treat partial output as success
                latency_ms=result.latency_ms,
                tokens_used=result.tokens_used,
                model=result.model,
                error="timeout_with_partial_output",
            )
        else:
            logger.warning(f"Subtask {self._subtask.id} timed out with no useful output, skipping")
            return result  # return as failed, orchestrator will handle
    
    # Generation error (not timeout) — retry once with escalation
    if max_retries > 0:
        logger.warning(f"Subtask {self._subtask.id} failed ({result.error}), retrying with escalation")
        self._subtask.max_tokens = min(self._subtask.max_tokens * 2, 4096)
        self._subtask.system_prompt += f"\n\nPrevious attempt failed with: {result.error}. Be careful."
        if self._subtask.model == "specialist":
            self._subtask.model = "primary"
        return self._run_once()
    
    return result
```

Key changes:
- **On timeout: do NOT retry.** Use partial output if available, or skip.
- **On generation error: retry once** with escalation (current behavior, capped at 4096 tokens).
- **Partial output handling:** If the server returned 100+ chars before timeout, treat it as success. The aggregation can work with partial results.

### Fix 2: Reasonable Per-Call Timeouts

Keep timeouts generous enough for the model to complete real work. The constraint is the TOTAL budget (Fix 3), not per-call limits.

| Call | Model | max_tokens | Timeout | Rationale |
|------|-------|-----------|---------|-----------|
| Classifier | Falcon-H1-1.5B | 256 | 60s | 1.5B model, 256 tokens = ~5-10s. 60s is generous. |
| Decomposer | Ornith-9B | 1024 | 180s | 9B model, 1024 tokens = ~60-90s. 180s allows for slow starts. |
| Worker | Primary/Specialist | varies | 180s | 9B model, up to 2048 tokens = ~90-120s. 180s is generous. |
| Pre-summarize | Falcon-H1-1.5B | 300 | 60s | 1.5B model, 300 tokens = ~5-10s. |
| Aggregation | Ornith-9B | 4096 | 180s | 9B model, up to 4096 tokens = ~120-180s. |

Add these timeouts explicitly to every `server.chat()` call. Don't rely on the 300s default.

In `src/lore/orchestrator.py`:
```python
# Classifier call
result = self._server.chat("specialist", messages, max_tokens=256, temperature=0.1, timeout=60)

# Aggregation call
result = self._server.chat("primary", messages, max_tokens=self._agg_max_tokens, 
                           temperature=0.1, timeout=180)

# Pre-summarization call
result = self._server.chat("specialist", messages, max_tokens=300, temperature=0.1, timeout=60)
```

In `src/lore/decomposer.py`:
```python
result = self._server.chat("primary", messages, max_tokens=self._max_tokens,
                           temperature=self._temperature, timeout=180,
                           response_format={"type": "json_object"})
```

In `src/lore/worker.py`, pass timeout to `_run_once()`:
```python
def __init__(self, subtask, server, memory=None, timeout=180):
    self._timeout = timeout
    ...

def _run_once(self) -> WorkerResult:
    ...
    result = self._server.chat(model, messages, 
                               max_tokens=self._subtask.max_tokens,
                               temperature=temperature, 
                               timeout=self._timeout)
    ...
```

### Fix 3: Total Orchestration Budget (Circuit Breaker)

This is the REAL constraint. Individual calls can take as long as they need (up to 180s). But the TOTAL orchestration time is capped.

In `src/lore/orchestrator.py`, add to `_orchestrate()`:

```python
def _orchestrate(self, query, est, route, confidence, json_mode, dispatch_fn=None) -> dict:
    t0 = time.time()
    max_orchestration_time = self._config.get("max_orchestration_time_s", 600)  # 10 min
    
    # ... classify, decompose ...
    
    results: dict[str, WorkerResult] = {}
    for wave_num, wave in enumerate(waves, 1):
        elapsed = time.time() - t0
        if elapsed > max_orchestration_time:
            logger.warning(
                f"Orchestration budget exceeded ({elapsed:.0f}s > {max_orchestration_time}s). "
                f"Completed {len(results)}/{len(plan.subtasks)} subtasks. Aggregating partial results."
            )
            break
        
        wave_results = self._execute_wave(wave, results)
        results.update(wave_results)
    
    if not results:
        logger.warning("No subtasks completed, falling back to direct dispatch")
        return self._delegate_dispatch(query, json_mode, dispatch_fn, route, confidence)
    
    # Aggregate whatever we have (even if partial)
    agg_content = self._aggregate(query, plan, results)
    ...
```

The circuit breaker **aggregates partial results** instead of discarding them. If 2/5 subtasks completed, the aggregation works with those 2 outputs.

### Fix 4: Reduce Planning Token Budget

In `src/lore/decomposer.py`:
```python
self._max_tokens = self._config.get("max_tokens", 1024)  # was 2048
```

A plan for 3-5 subtasks is ~500-800 tokens of JSON. 1024 is sufficient.

### Fix 5: Pass Classifier Context to Decomposer

In `src/lore/orchestrator.py`, build richer planning context:

```python
# In _orchestrate(), when building decomposer user content:
if self._classification is not None:
    hints_text = (
        f"Pre-analysis:\n"
        f"- Task type: {self._classification.task_type}\n"
        f"- Estimated subtasks: {self._classification.estimated_subtasks}\n"
        f"- Suggested model for main work: {self._classification.suggested_model}\n"
    )
    plan = self._decomposer.decompose(query, hints={
        "pre_analysis": hints_text,
        **self._classification.hints,
    })
```

This gives the decomposer richer context so it produces better plans on the first try.

### Fix 6: Fast Aggregation for Code Tasks

For tasks where all subtasks produce code and have no dependencies, skip the aggregation LLM call entirely:

```python
def _aggregate(self, query, plan, results):
    """Aggregate results. Fast-path for independent code tasks."""
    
    # Fast path: all code, no dependencies → just concatenate
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
    
    # Full aggregation: pre-summarize if needed, then aggregate
    ...
```

This saves one LLM call (~120s) for code-heavy tasks.

### Fix 7: Trim Planning Prompt

In `src/lore/decomposer.py`, reduce `_PLANNING_SYSTEM` from 3 examples to 2 (remove the "simple" example — simple tasks don't reach the decomposer). Target ~2200 chars instead of ~3500.

Also, the JSON format specification should be shorter:
```python
_PLANNING_SYSTEM = """You are a task planner. Break a complex task into 2-5 subtasks.

Models available:
- PRIMARY (9B): code, reasoning, analysis, planning
- SPECIALIST (1.5B): extraction, formatting, summarization

Output JSON: {"subtasks": [{"id", "description", "model", "context_budget", "system_prompt", "dependencies", "max_tokens", "output_format"}], "aggregation_prompt"}

Rules:
- Max 5 subtasks. Aim for ⌈√(estimated steps)⌉ subtasks.
- First subtask must have no dependencies.
- Specialist only for simple text/data tasks. Everything else → primary.
- context_budget: 1024 for extraction, 4096 for code, 8192 for complex reasoning.
- output_format: code_python, code_bash, json, or free.

Example 1: [moderate coding task example]
Example 2: [complex multi-file task example]
"""
```

## Files to Modify

1. `src/lore/worker.py` — Fix 1 (no retry on timeout, partial output handling), Fix 2 (explicit timeout=180)
2. `src/lore/orchestrator.py` — Fix 2 (explicit timeouts), Fix 3 (circuit breaker), Fix 5 (classifier context), Fix 6 (fast aggregation)
3. `src/lore/decomposer.py` — Fix 2 (timeout=180), Fix 4 (max_tokens=1024), Fix 7 (prompt trim)
4. `tests/test_orchestrator.py` — Update tests for new behavior

## Testing

1. `PYTHONPATH=src python -m pytest tests/ -v --tb=short` — all tests pass
2. `PYTHONPATH=src python scripts/benchmark_orchestration.py --benchmark humaneval --limit 10` — pass@1 ≥87%, avg latency ≤120s
3. `PYTHONPATH=src python scripts/benchmark_orchestration.py --benchmark complex --limit 5` — each task ≤300s, pass rate ≥80%

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Orchestrated task avg latency | 600-1675s | **120-300s** |
| Time wasted on timeout retries | 900s per timeout | **0s (no retry on timeout)** |
| Partial output utilization | 0% (discarded) | **Used when available** |
| Total orchestration budget | Unlimited | **600s hard cap** |
| HumanEval pass@1 | 87% | **≥87%** |
| Complex task pass rate | 100% structural | **≥80%** |

## Key Principle

**Give the model enough time to do real work (180s). But don't waste time retrying timeouts with bigger requests.** The circuit breaker (600s total) is the real constraint. Individual calls are generous; the budget is strict.
