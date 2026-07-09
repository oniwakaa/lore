# Prompt: Investigate LORE Performance Degradation During HumanEval Benchmark

## Identity & Context

You are working on **LORE** (Local Orchestration & Runtime Engine) at `~/projects/lore`. Read `AGENTS.md` for project context. Run with `PYTHONPATH=src`.

## The Problem

During a full HumanEval benchmark run (164 tasks), LORE starts fast (~30+ tasks/min) but progressively slows down to ~2 tasks per 20 minutes. By the end, the system is almost unusable.

The benchmark runs each task through `Orchestrator.process()`. Most tasks (141/164) are classified as "simple" and routed directly. Only 23/164 are orchestrated (decomposed into subtasks).

**Root cause is likely one of:**
1. **Context accumulation** — the ContextManager keeps adding messages and never cleans up, so each subsequent task builds a larger prompt
2. **Memory accumulation** — HierarchicalMemory stores every task result as an episodic memory entry, and retrieval/embedding gets slower as it grows
3. **KV cache pressure** — the llama-server's KV cache fills up across tasks (no reset between tasks), causing memory pressure and slower inference
4. **Orchestrator state leak** — classification results, decomposer plans, or worker results accumulate in the orchestrator object across tasks
5. **Server-side degradation** — llama-server itself slows down after many requests (memory fragmentation, KV cache fragmentation)
6. **Log file growth** — logging to files that grow unbounded, causing I/O slowdown

## Your Mission

**Instrument the benchmark to identify exactly WHERE the time goes, then fix it.**

## Phase 1: Instrument and Profile

### 1.1 Add Per-Task Timing Breakdown

Modify `scripts/benchmark_orchestration.py` to add granular timing for each task:

```python
def run_humaneval_task(problem, orchestrator, dispatch_fn):
    timings = {}
    
    # Time the full pipeline
    t0 = time.time()
    
    # Time: orchestrator.process()
    t_process = time.time()
    result = orchestrator.process(prompt, dispatch_fn=dispatch_fn)
    timings["process_s"] = time.time() - t_process
    
    # Time: code extraction
    t_extract = time.time()
    code = extract_code(result["content"], prompt)
    timings["extract_s"] = time.time() - t_extract
    
    # Time: test execution
    t_test = time.time()
    test_result = run_test_sandboxed(full_code)
    timings["test_s"] = time.time() - t_test
    
    timings["total_s"] = time.time() - t0
    timings["task_number"] = task_counter  # track position in benchmark
    
    return {**result, **timings, **test_result}
```

### 1.2 Add Resource Monitoring

After every N tasks (e.g., every 10), log:

```python
import psutil

def log_resources(task_num):
    proc = psutil.Process()
    mem = proc.memory_info()
    print(f"  [Task {task_num}] RSS={mem.rss/1024/1024:.1f}MB "
          f"VMS={mem.vms/1024/1024:.1f}MB "
          f"Threads={proc.num_threads()} "
          f"OpenFDs={proc.num_fds()}")
```

### 1.3 Log Server Health

Check llama-server health/metrics before and after each batch of 10 tasks:

```python
def log_server_health(port, label):
    try:
        # Check slots usage
        resp = requests.get(f"http://127.0.0.1:{port}/slots", timeout=5)
        if resp.ok:
            slots = resp.json()
            for s in slots:
                print(f"  [{label}] Slot {s.get('id')}: "
                      f"n_past={s.get('n_past',0)} "
                      f"n_ctx={s.get('n_ctx',0)}")
        # Check health
        resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
        print(f"  [{label}] Health: {resp.json() if resp.ok else 'FAIL'}")
    except Exception as e:
        print(f"  [{label}] Health check failed: {e}")
```

## Phase 2: Run the Instrumented Benchmark

Run with `--limit 50` (not all 164 — enough to see the degradation pattern):

```bash
PYTHONPATH=src python scripts/benchmark_orchestration.py --benchmark humaneval --limit 50
```

**Save the instrumented output to `benchmarks/results/degradation_profile.txt`**

## Phase 3: Analyze and Fix

Based on the timing breakdown, identify which component is causing the slowdown. The most likely suspects in order:

### Suspect 1: Context Accumulation (MOST LIKELY)

The ContextManager adds messages via `ctx.add_message()` on every task but never resets between tasks. After 50 tasks, the context contains 100+ messages (user + assistant) that get sent with every new request.

**Fix:** Reset the context between tasks in the benchmark loop:
```python
for task in tasks:
    # Reset context for each task (fresh session)
    orchestrator._ctx = build_fresh_context(orchestrator._server, orchestrator._config)
    result = run_humaneval_task(task, orchestrator, dispatch_fn)
```

Or better: add a `reset()` method to ContextManager that clears accumulated messages while preserving the system prompt.

### Suspect 2: Memory Accumulation

HierarchicalMemory stores embeddings for every task. After 50 tasks, retrieval might be scanning 50+ entries.

**Fix:** Either disable memory for benchmarks, or add a `memory.reset()` that clears episodic entries.

### Suspect 3: KV Cache Not Reset Between Tasks

llama-server's KV cache accumulates across requests. After 50 tasks, the cache might be full, causing eviction/recomputation.

**Fix:** Check if llama-server has a `/completion` endpoint that supports cache clearing. Or restart the server between batches.

### Suspect 4: Orchestrator State

The orchestrator object carries `self._classification`, `self._plan`, `self._results` between tasks.

**Fix:** Add a `reset_state()` method to Orchestrator that clears per-task state:
```python
def reset_state(self):
    """Clear per-task state between benchmark runs."""
    self._classification = None
    self._plan = None
    self._results = {}
```

### Suspect 5: Server-Side Degradation

llama-server might degrade after many requests (memory fragmentation).

**Fix:** Check server health metrics. If degradation is server-side, the fix is restarting the server every N tasks (expensive but definitive).

## Phase 4: Fix and Re-Run

After identifying the root cause:

1. Apply the fix (context reset, memory reset, state cleanup, or server restart)
2. Re-run the full benchmark: `--benchmark humaneval` (all 164 tasks)
3. Verify: task timing should be consistent throughout (no degradation)
4. Compare pass@1: the fix should not change correctness, only speed
5. Save results to `benchmarks/results/humaneval_lore_fixed.json`

## Phase 5: Report

Print a summary:
```
═══════════════════════════════════════════════════════════
 PERFORMANCE DEGRADATION ANALYSIS
═══════════════════════════════════════════════════════════

 Root cause: [identified cause]

 Before fix:
   Tasks 1-10:   Xs avg per task
   Tasks 50-60:  Xs avg per task  
   Tasks 100+:   Xs avg per task

 After fix:
   Tasks 1-10:   Xs avg per task
   Tasks 50-60:  Xs avg per task
   Tasks 100+:   Xs avg per task

 Consistency: X.XX (stddev/mean of per-task latency)
 Pass@1 before: 84% | After: XX%
═══════════════════════════════════════════════════════════
```

## Files to Read First

1. `scripts/benchmark_orchestration.py` — Current benchmark script
2. `src/lore/orchestrator.py` — Orchestrator.process(), state management
3. `src/lore/context.py` — ContextManager, add_message(), build_prompt()
4. `src/lore/memory.py` — HierarchicalMemory, episodic storage
5. `src/lore/models.py` — ModelServer.chat(), HTTP client

## Constraints

1. **Don't change the benchmark logic** — only add instrumentation and fix the degradation
2. **The fix must not change pass@1** — correctness should be identical (or better)
3. **If the fix requires API changes** (adding reset methods), update both the class and all callers
4. **Run the fix with `--limit 50` first** to verify degradation is gone, then full 164
5. **Save all profiling data** to `benchmarks/results/`

## When to Ask

- If profiling shows the slowdown is server-side (llama-server), report it — we may need to restart servers between batches
- If multiple causes contribute, fix them in order of impact (biggest time savings first)
- If the fix requires architectural changes to ContextManager or Orchestrator, document the change and its test impact
