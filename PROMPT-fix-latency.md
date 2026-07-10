# Prompt: Fix Orchestration Latency — Timeout, Retry, and Call Reduction

## Identity & Context

You are working on **LORE** (Local Orchestration & Runtime Engine) at `~/projects/lore`. Run with `PYTHONPATH=src`.

LORE's orchestration engine works correctly (100% pass rate on complex tasks) but is catastrophically slow. One task took 1675s (28 minutes). A direct single-model call would take 60-120s. The root cause is a timeout-retry death spiral and excessive sequential LLM calls.

## The Problem (Read This First)

A orchestrated task on Ornith-9B (~15 tok/s) makes 4-6 sequential LLM calls:
- Classifier (Falcon-H1, ~5s)
- Decomposer (Ornith, ~90s)  
- 3-5 Workers (Ornith, ~60-120s each)
- Aggregation (Ornith, ~120s)

**Total: 4-7 minutes minimum.** With the timeout death spiral, it balloons to 28 minutes.

The death spiral happens in `worker.py`'s `run_with_retry()`:
```
timeout at 300s → max_tokens doubled (2048→4096) → retry → timeout again → doubled (→8192) → retry → timeout AGAIN → total 900s wasted on ONE subtask
```

Doubling tokens on timeout makes the NEXT attempt take LONGER, not shorter. This is backwards.

## 7 Fixes (in priority order)

### Fix 1: Timeout-Aware Retry Logic (CRITICAL)

In `src/lore/worker.py`, rewrite `run_with_retry()` to distinguish timeout from generation errors:

```python
def run_with_retry(self, max_retries: int = 1) -> WorkerResult:
    """Run subtask with retry. Different strategies for timeout vs generation errors."""
    for attempt in range(max_retries + 1):
        # Use shorter timeout on retry attempts
        effective_timeout = self._timeout if attempt == 0 else min(self._timeout, 60)
        
        result = self._run_once(timeout=effective_timeout)
        if result.success:
            return result
        
        if attempt < max_retries:
            if result.error and "timeout" in str(result.error).lower():
                # TIMEOUT: REDUCE output expectations, don't increase
                logger.warning(f"Subtask {self._subtask.id} timed out, reducing max_tokens")
                self._subtask.max_tokens = max(self._subtask.max_tokens // 2, 256)
                self._subtask.system_prompt += (
                    "\n\nCRITICAL: Be extremely concise. Previous attempt timed out. "
                    "Give the shortest possible correct answer. No explanations, just code."
                )
            else:
                # GENERATION ERROR: escalate with more context
                logger.warning(f"Subtask {self._subtask.id} failed: {result.error}")
                self._subtask.max_tokens = min(self._subtask.max_tokens * 2, 4096)
                if self._subtask.model == "specialist":
                    self._subtask.model = "primary"
    
    return result
```

Key changes:
- **max_retries: 2 → 1** (one retry, not two)
- **On timeout: REDUCE max_tokens by half** (don't double it)
- **On timeout: shorter retry timeout** (60s, not 300s)
- **On timeout: add "be concise" instruction** to system prompt
- **On generation error: escalate** (current behavior, but cap at 4096 not 8192)

### Fix 2: Explicit Timeouts on Every LLM Call

Every `server.chat()` call should have an explicit timeout. Currently most use the 300s default.

In `src/lore/orchestrator.py`:
```python
# Classifier call
result = self._server.chat("specialist", messages, max_tokens=256, temperature=0.1, timeout=30)

# Decomposer call  
result = self._server.chat("primary", messages, max_tokens=1024, temperature=0.2, timeout=120)

# Aggregation call
result = self._server.chat("primary", messages, max_tokens=self._agg_max_tokens, temperature=0.1, timeout=120)

# Pre-summarization call
result = self._server.chat("specialist", messages, max_tokens=300, temperature=0.1, timeout=30)
```

In `src/lore/worker.py`:
```python
# Worker call — store timeout as instance variable
def __init__(self, subtask, server, memory=None, timeout=120):
    self._timeout = timeout
    ...
```

In `src/lore/decomposer.py`:
```python
# Decomposer call
result = self._server.chat("primary", messages, max_tokens=self._max_tokens,
                           temperature=self._temperature, timeout=120,
                           response_format={"type": "json_object"})
```

**Timeout summary:**
| Call | Model | Timeout |
|------|-------|---------|
| Classifier | Falcon-H1-1.5B | 30s |
| Decomposer | Ornith-9B | 120s |
| Worker | Primary or Specialist | 120s (60s on retry) |
| Pre-summarize | Falcon-H1-1.5B | 30s |
| Aggregation | Ornith-9B | 120s |

### Fix 3: Total Orchestration Budget (Circuit Breaker)

Add a total time budget to `_orchestrate()`. If elapsed > budget, stop executing and aggregate what we have.

In `src/lore/orchestrator.py`, at the start of `_orchestrate()`:
```python
def _orchestrate(self, query, est, route, confidence, json_mode, dispatch_fn=None) -> dict:
    t0 = time.time()
    max_orchestration_time = self._config.get("max_orchestration_time_s", 600)  # 10 min default
    
    # ... decompose ...
    
    # Execute waves with circuit breaker
    results: dict[str, WorkerResult] = {}
    for wave_num, wave in enumerate(waves, 1):
        # Check budget before each wave
        elapsed = time.time() - t0
        if elapsed > max_orchestration_time:
            logger.warning(f"Orchestration budget exceeded ({elapsed:.0f}s > {max_orchestration_time}s), "
                          f"aggregating partial results ({len(results)}/{len(plan.subtasks)} completed)")
            break
        
        wave_results = self._execute_wave(wave, results)
        results.update(wave_results)
    
    # Aggregate whatever we have (even if incomplete)
    if not results:
        logger.warning("No subtasks completed, falling back to direct dispatch")
        return self._delegate_dispatch(query, json_mode, dispatch_fn, route, confidence)
    
    agg_content = self._aggregate(query, plan, results)
    ...
```

### Fix 4: Reduce Planning Token Budget

In `src/lore/decomposer.py`:
```python
self._max_tokens = self._config.get("max_tokens", 1024)  # was 2048
self._temperature = self._config.get("temperature", 0.2)  # was 0.3
```

A plan for 3-5 subtasks in JSON is ~500-800 tokens. 1024 is sufficient. The `_repair_truncated()` function handles cutoff gracefully.

### Fix 5: Merge Classify + Decompose Into One Call

Currently the pipeline makes two sequential calls to the primary model:
1. Classifier call (specialist, ~5s) → gets task_type, is_complex, estimated_subtasks
2. Decomposer call (primary, ~90s) → gets subtask plan

The classifier hints are passed to the decomposer, but the decomposer re-reads the whole task. We can save one call by having the decomposer include classification in its output.

**Implementation:** In `orchestrator.py`'s `_orchestrate()`, pass classifier results directly into the planning prompt as structured context, so the decomposer doesn't need to re-analyze:

```python
# In _orchestrate(), build the planning user content with classifier results
if self._classification is not None:
    user_content = (
        f"Task to decompose:\n{query}\n\n"
        f"Pre-analysis (from classifier):\n"
        f"- Task type: {self._classification.task_type}\n"
        f"- Complexity: {'high' if self._classification.is_complex else 'low'}\n"
        f"- Estimated subtasks: {self._classification.estimated_subtasks}\n"
        f"- Suggested model: {self._classification.suggested_model}\n"
        f"- Signals: {self._classification.hints}\n\n"
        f"Based on this analysis, decompose into subtasks."
    )
```

This doesn't eliminate the classifier call (it's fast, ~5s on the specialist), but it gives the decomposer richer context so it produces better plans on the first try — reducing the chance of fallback plans that waste time.

**Alternative (bigger change):** Skip the separate classifier call entirely and have the decomposer handle classification + decomposition in one call. The decomposer prompt already asks for model assignment — adding `task_type` and `is_complex` to its output schema is trivial. This saves ~5-10s (the classifier call) but more importantly simplifies the pipeline. Only do this if it's low-risk.

### Fix 6: Skip Aggregation for Code Tasks

For tasks where subtask outputs are code (Python functions, classes, files), aggregation is often just concatenation — the 9B model spends 120s rearranging code that was already correct.

In `src/lore/orchestrator.py`, add a fast-path aggregation:

```python
def _aggregate(self, query, plan, results):
    """Aggregate subtask results. Fast-path for code-only plans."""
    
    # Fast path: if all subtasks are code outputs and have no dependencies,
    # just concatenate (no synthesis needed)
    all_code = all(
        st.output_format in ("code_python", "code_bash")
        for st in plan.subtasks
    )
    no_deps = all(not st.dependencies for st in plan.subtasks)
    
    if all_code and no_deps and len(results) > 1:
        logger.info("Fast aggregation: code-only plan with no deps, concatenating")
        parts = []
        for st in plan.subtasks:
            r = results.get(st.id)
            if r and r.success:
                parts.append(f"# --- {st.description[:80]} ---\n{r.content}")
        return "\n\n".join(parts)
    
    # Full aggregation (existing logic with pre-summarization + progressive)
    ...
```

This eliminates the aggregation LLM call entirely for code tasks that don't need synthesis. Saves ~120s.

### Fix 7: Planning Prompt Trim

The current planning prompt has 3 few-shot examples (~3500 chars). Trim to 2 examples (~2200 chars):

In `src/lore/decomposer.py`, keep the "moderate coding task" and "complex multi-file task" examples. Remove the "simple multi-part task" example (simple tasks won't reach the decomposer — they're routed directly).

Also move the JSON format specification to a separate, shorter section:

```python
_PLANNING_SYSTEM = """You are a task planner for a local AI system with two models:
- PRIMARY (9B): strong at reasoning, coding, planning. Use for complex subtasks.
- SPECIALIST (1.5B): fast, good at extraction, formatting, summarization. Use for simple helper tasks.

Given a complex task, break it into 2-5 subtasks. Output JSON:
{
  "subtasks": [
    {"id": "s1", "description": "...", "model": "primary|specialist",
     "context_budget": 2048, "system_prompt": "...", "dependencies": [],
     "max_tokens": 1024, "output_format": "code_python|json|free"}
  ],
  "aggregation_prompt": "..."
}

Rules:
- Max 5 subtasks. Aim for ⌈√S⌉ where S = estimated steps.
- Specialist handles: text extraction, formatting, summarization, simple transforms.
- Primary handles: code generation, reasoning, analysis, debugging.
- Independent subtasks run in parallel. Dependent ones are sequential.
- First subtask must have no dependencies.

Example 1: Moderate coding task
[Keep the moderate example from current prompt]

Example 2: Complex multi-file task
[Keep the complex example from current prompt]
"""
```

## Files to Modify

1. `src/lore/worker.py` — Fix 1 (retry logic), Fix 2 (explicit timeout)
2. `src/lore/orchestrator.py` — Fix 2 (explicit timeouts), Fix 3 (circuit breaker), Fix 5 (merge classify/decompose context), Fix 6 (fast aggregation)
3. `src/lore/decomposer.py` — Fix 2 (explicit timeout), Fix 4 (token budget), Fix 7 (prompt trim)
4. `tests/test_orchestrator.py` — Update tests for new timeout behavior

## Testing

1. Run existing tests: `PYTHONPATH=src python -m pytest tests/ -v --tb=short`
   - All 237+ tests must pass
2. Smoke test: `PYTHONPATH=src python scripts/benchmark_orchestration.py --benchmark humaneval --limit 10`
   - Pass@1 must be ≥87% (no regression)
   - Avg latency should be ≤120s (down from 157s in v2, way down from orchestrated tasks)
3. Complex task test: `PYTHONPATH=src python scripts/benchmark_orchestration.py --benchmark complex --limit 5`
   - Each task should complete in ≤300s (down from 456-1675s)
   - Pass rate should be ≥80%

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Orchestrated task avg latency | 600-1675s | **120-300s** |
| Timeout rate | 50% | **<10%** |
| Time wasted on retries | 900s per timeout | **60s max** |
| Total orchestration budget | Unlimited | **600s hard cap** |
| HumanEval pass@1 | 87% | **≥87%** |
| Complex task pass rate | 100% (structural) | **≥80%** |

## Constraints

1. All existing tests must pass
2. Don't change the direct dispatch path
3. Don't change the classifier or router
4. The circuit breaker must aggregate partial results (not discard them)
5. Fast aggregation (Fix 6) must only apply to code tasks with no dependencies

## When to Ask

- If Fix 5 (merge classify/decompose) is too risky, skip it and focus on Fixes 1-4 + 6-7
- If the circuit breaker fires on most tasks (budget too tight), increase to 900s
- If fast aggregation produces worse output than full aggregation for code tasks, disable it
