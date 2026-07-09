# Prompt: Investigate Performance Degradation + Improve Orchestration Quality

## Identity & Context

You are working on **LORE** (Local Orchestration & Runtime Engine) at `~/projects/lore`. Read `AGENTS.md` for project context. Run with `PYTHONPATH=src`.

## Two Problems to Solve

### Problem 1: Performance Degradation During Benchmark

During the HumanEval benchmark run (164 tasks), LORE started fast (~30+ tasks/min) but progressively slowed down to ~2 tasks per 20 minutes. The system became almost unusable by the end.

**You already have the evidence.** During the benchmark run, you performed health checks, resource monitoring, and logged timing data. Search your session history for:

- `log_resources()` outputs showing RSS memory growth over time
- `log_server_health()` outputs showing KV cache / slot usage
- Per-task timing breakdowns (process_s, extract_s, test_s, total_s)
- Any psutil metrics, server /slots responses, or health check outputs
- Timestamps between consecutive tasks showing the slowdown pattern

**First step:** Use `session_search` to find the benchmark session and extract the timing/resource data. Look for patterns like:
- RSS memory growing linearly with task count
- Server slot n_past accumulating (not resetting between tasks)
- orchestrator.process() taking progressively longer
- Any error messages, warnings, or retries that increase over time

### Problem 2: Orchestrated Tasks Low Pass Rate

LORE achieved 84% HumanEval overall, but the breakdown was:
- **Direct routed: 89% (125/141)** — classifier correctly identified these as simple
- **Orchestrated: 57% (13/23)** — these were flagged as complex, but orchestration hurt more than it helped

For HumanEval specifically, these are self-contained single functions. Decomposing a single function into subtasks and then aggregating adds noise. The orchestrator should probably recognize "this is a single function, don't decompose" even if it looks complex.

## Phase 1: Extract Evidence from Session History

Search your past sessions for the benchmark run data:

```python
# Look for benchmark execution sessions
session_search(query="benchmark humaneval tasks degradation", limit=5)
session_search(query="log_resources RSS memory tasks", limit=3)
session_search(query="health server slots n_past", limit=3)
session_search(query="task latency timing per_task", limit=3)
```

Extract and tabulate:
1. **Per-task latency** — time per task at task #1, #10, #20, #50, #100, #150
2. **Memory growth** — RSS at each checkpoint
3. **Server state** — KV cache usage, slot n_past at each checkpoint
4. **Any anomalies** — errors, retries, timeout warnings

## Phase 2: Root Cause Analysis

Based on the extracted data, identify the root cause. Most likely suspects:

### Suspect A: Context Accumulation (MOST LIKELY)

The ContextManager (`src/lore/context.py`) adds messages via `add_message()` on every task. If the benchmark doesn't reset context between tasks, after 50 tasks the context contains 100+ messages sent with every new request.

**Evidence to check:** Does `build_prompt()` include all accumulated messages? If yes, the prompt grows linearly → slower inference → eventual timeout.

**Fix:** Add `ContextManager.reset()` that clears accumulated messages but preserves the system prompt. Call it between tasks in the benchmark loop.

### Suspect B: Memory Accumulation

`HierarchicalMemory` (`src/lore/memory.py`) stores embeddings for every task result. After 50 tasks, `retrieve()` might scan 50+ entries with embedding similarity, getting slower.

**Evidence to check:** Does the orchestrator or dispatch function call `memory.store()` on every task? Is `memory.retrieve()` called in `build_prompt()`?

**Fix:** Either disable memory for benchmarks, or add `memory.reset()` that clears episodic entries.

### Suspect C: KV Cache Not Reset

llama-server accumulates KV cache across requests in the same slot. After 50 tasks with 2-4K context each, the cache might be 100-200K tokens — way beyond the 16K budget.

**Evidence to check:** Look at `/slots` endpoint output. Does `n_past` grow across tasks? If yes, the server is accumulating context.

**Fix:** The benchmark should use `cache_prompt: false` or explicitly clear the KV cache between tasks. Check if `server.chat()` passes any cache control parameters.

### Suspect D: Orchestrator State Leak

The orchestrator (`src/lore/orchestrator.py`) carries `self._classification`, `self._plan`, `self._results` between tasks. If these accumulate, subsequent tasks pay the cost of processing stale state.

**Evidence to check:** Does `process()` reset `self._results` at the start? Does it carry over classification from the previous task?

**Fix:** Add `Orchestrator.reset_state()` that clears per-task state at the start of `process()`.

### Suspect E: Server-Side Degradation

llama-server itself might degrade after many requests (memory fragmentation, GC pressure in Python HTTP client).

**Evidence to check:** Does latency increase even for simple tasks that bypass orchestration? If yes, it's server-side.

**Fix:** Restart servers every N tasks (expensive) or investigate server-side memory management.

## Phase 3: Fix Both Issues

### Fix 1: Performance Degradation

After identifying the root cause(s), apply the minimal fix:

1. If context accumulation: add `ctx.reset()` between benchmark tasks
2. If memory accumulation: add `memory.clear_episodic()` between tasks
3. If KV cache: add cache clearing to the benchmark loop
4. If orchestrator state: add `orchestrator.reset_state()` at start of `process()`
5. If multiple causes: fix all, in order of impact

**Critical:** The fix must NOT change pass@1. Run `--limit 10` before and after to verify correctness is identical.

### Fix 2: Orchestrator Quality (57% → higher)

For HumanEval-style tasks (single function, self-contained), the orchestrator should recognize them as "don't decompose" even if the classifier flags them as complex.

**Approach:** Add a **task scope check** before decomposition:

```python
# In orchestrator.py, before calling self._decomposer.decompose()
def _should_decompose(self, query: str, est: ComplexityEstimate) -> bool:
    """Decompose only if the task actually benefits from splitting."""
    # Single function definitions don't benefit from decomposition
    if re.match(r'^(def |class |import |from )', query.strip()):
        return False
    # Very short tasks don't benefit
    if len(query.split()) < 50:
        return False
    # Tasks with "write a function" pattern are self-contained
    if re.match(r'^(Write|Create|Implement|Build)\s+(a|an)\s+(function|class|method)', query, re.IGNORECASE):
        if len(query.split()) < 100:  # short description = single function
            return False
    return est.is_complex
```

This keeps orchestration for genuinely complex multi-step tasks (FastAPI app, CLI tool, full test suite) while routing single-function tasks directly even if the classifier thinks they're complex.

**Expected result:** More tasks routed direct → higher pass@1 on the orchestrated subset → higher overall pass@1.

## Phase 4: Re-Run and Validate

1. Run with `--limit 20` to verify:
   - No performance degradation (consistent per-task latency throughout)
   - Orchestrated pass rate improved (fewer tasks decomposed unnecessarily)
   - Overall pass@1 same or better than 84%

2. Run full 164 tasks:
   - Save to `benchmarks/results/humaneval_lore_v2.json`
   - Print comparison: v1 (84%) vs v2 (XX%)

3. Print degradation analysis:
   ```
   ═══════════════════════════════════════════════════════════
    DEGRADATION FIX REPORT
   ═══════════════════════════════════════════════════════════
    Root cause: [identified cause]
    Fix applied: [what was changed]
    
    Before fix:
      Tasks 1-10:   Xs avg
      Tasks 50-60:  Xs avg
      Tasks 100+:   Xs avg
    
    After fix:
      Tasks 1-10:   Xs avg
      Tasks 50-60:  Xs avg
      Tasks 100+:   Xs avg
    
    Consistency (stddev/mean): before X.XX → after X.XX
    Pass@1: before 84% → after XX%
    Orchestrated tasks: before 23 → after N
    Orchestrated pass rate: before 57% → after XX%
   ═══════════════════════════════════════════════════════════
   ```

## Files to Read First

1. `scripts/benchmark_orchestration.py` — Benchmark loop (where to add resets)
2. `src/lore/orchestrator.py` — process(), state management, decomposition decision
3. `src/lore/context.py` — ContextManager, add_message(), build_prompt()
4. `src/lore/memory.py` — HierarchicalMemory, store(), retrieve()
5. `src/lore/models.py` — ModelServer.chat(), HTTP client
6. `src/lore/decomposer.py` — decompose(), plan parsing

## Constraints

1. **Search session history FIRST** — you have the evidence from the benchmark run. Don't guess.
2. **Don't change pass@1 without the degradation fix** — the two fixes should be independent
3. **Test each fix separately** — degradation fix first, then orchestration quality fix
4. **Save all profiling data** to `benchmarks/results/`
5. **The _should_decompose heuristic must be conservative** — when in doubt, decompose. Only skip decomposition for clearly self-contained single-function tasks.

## When to Ask

- If session search doesn't find the benchmark data, check `benchmarks/results/humaneval_lore.json` for per-task timing
- If the root cause is server-side (llama-server degradation), report it — may need a different approach
- If the _should_decompose heuristic reduces orchestration too aggressively (fewer than 5 tasks orchestrated), relax the rules
