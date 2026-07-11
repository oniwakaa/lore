# A/B Test Results: Orchestrated vs Direct (Full 20-Task Run)

**Date:** 2026-07-11  
**Hardware:** Apple Silicon M4, 16 GB unified memory  
**Models:** Ornith-1.0-9B Q4_K_M (primary), Falcon-H1-1.5B Q4_K_M (specialist)  
**Task set:** `benchmarks/eval_tasks/ab_test_tasks.json` (10 simple + 10 complex)  
**Raw results:** `benchmarks/results/orchestration_ab.json`

## Per-Task Results

| Task    | Category | Direct (s) | Orch (s) | Δ       | Winner  | Direct ok | Orch ok |
|---------|----------|------------|----------|---------|---------|-----------|---------|
| task_01 | simple   | 65.9       | 28.8     | -56%    | Orch    | Yes       | Yes     |
| task_02 | simple   | 32.0       | 13.9     | -57%    | Orch    | Yes       | Yes     |
| task_03 | simple   | 17.5       | 18.0     | +3%     | Direct  | Yes       | Yes     |
| task_04 | simple   | 15.4       | 9.9      | -36%    | Orch    | Yes       | Yes     |
| task_05 | simple   | 31.1       | 19.8     | -36%    | Orch    | Yes       | Yes     |
| task_06 | simple   | 15.3       | 15.8     | +3%     | Direct  | Yes       | Yes     |
| task_07 | simple   | 39.0       | 24.6     | -37%    | Orch    | Yes       | Yes     |
| task_08 | simple   | 24.3       | 19.6     | -19%    | Orch    | Yes       | Yes     |
| task_09 | simple   | 24.2       | 20.2     | -17%    | Orch    | Yes       | Yes     |
| task_10 | simple   | 34.8       | 18.2     | -48%    | Orch    | Yes       | Yes     |
| task_11 | complex  | 94.9       | 300.0†   | +216%   | Direct  | Yes       | No      |
| task_12 | complex  | 300.0†     | N/A‡     | —       | —       | No        | —       |
| task_13 | complex  | 244.4      | N/A‡     | —       | —       | Yes       | —       |
| task_14 | complex  | 300.0†     | N/A‡     | —       | —       | No        | —       |
| task_15 | complex  | 300.0†     | N/A‡     | —       | —       | No        | —       |
| task_16 | complex  | 300.0†     | N/A‡     | —       | —       | No        | —       |
| task_17 | complex  | 300.0†     | N/A‡     | —       | —       | No        | —       |
| task_18 | complex  | 300.0†     | N/A‡     | —       | —       | No        | —       |
| task_19 | complex  | 300.1†     | N/A‡     | —       | —       | No        | —       |
| task_20 | complex  | 300.0†     | N/A‡     | —       | —       | No        | —       |

† Hit 300s circuit breaker timeout.  
‡ Not reached — orchestrated run killed by outer process timeout before reaching complex tasks.

## Summary Statistics

### Simple Tasks (task_01 to task_10)

|              | Direct | Orchestrated |
|--------------|--------|--------------|
| Avg latency  | 29.9s  | 18.8s        |
| Correct      | 10/10  | 10/10        |
| Orch wins    | —      | 8/10 on latency |
| Latency gain | —      | **37% faster** |

Router classified all 10 simple tasks as `simple` (math/classification, conf=0.85). Orchestration routed them to the 1.5B specialist — no decomposition, no planning overhead. Result: 37% faster on average, same correctness.

### Complex Tasks (task_11 to task_20)

|                   | Direct            | Orchestrated         |
|-------------------|-------------------|----------------------|
| Completed         | 2/10 (within 300s)| 0/10 (timed out)     |
| Correct (of those)| 2/2               | 0/0                  |
| Timeout rate      | 8/10 (300s limit) | task_11 stalled; 12-20 not reached |

The complex benchmark exposed two separate problems:

1. **Direct primary model**: hits token generation limits at 300s for 8/10 complex tasks. Only task_11 (94.9s) and task_13 (244.4s) completed. Direct works on medium-complexity tasks; fails on highly complex ones regardless of orchestration.

2. **Orchestration decomposition**: task_11 was decomposed into 5 subtasks. Subtask s1 (primary, 2048 tok) failed at the 180s subtask-level timeout, stalling the wave. The orchestration overhead (decomposition + subtask routing + wave scheduling) adds latency on top of already slow primary model calls. Net result: orchestration was slower, not faster, for task_11.

## Key Findings

### Finding 1: Routing to specialist wins on simple tasks

Routing math/classification queries to Falcon-H1-1.5B instead of Ornith-9B saves 37% latency with no correctness loss. This is the clearest win. The 1.5B specialist is fast; the 9B primary is slow for tasks that don't need it.

**When it applies:** Any factual, classification, or single-step math query the router classifies with high confidence as `simple`.

### Finding 2: Decomposition does not help (yet) on complex tasks

Task_11 (FastAPI app generation) was decomposed into 5 subtasks and timed out. The primary model is already slow enough that 5 sequential/pipelined calls each risk the 180s subtask timeout. Decomposition multiplies the latency risk.

The decomposition planner also scheduled all 4 primary subtasks through a single model endpoint, creating a sequential bottleneck. The theoretical benefit (parallel subtasks) doesn't materialize with one primary model instance.

**When it would help:** True parallelism across 2+ primary model instances, or subtask sizes small enough each completes in <60s.

### Finding 3: Both approaches fail on highly complex tasks

8/10 complex tasks hit the 300s wall for direct. This is a model capability/speed ceiling at Q4_K_M on M4 — not an orchestration problem. The complex task prompts generate very long outputs (thousands of tokens) at ~20 tok/s effective throughput.

The honest diagnosis: these tasks are at the edge of what the hardware can deliver in reasonable time with the current quantization and context window settings.

## Honest Assessment

**Does orchestration beat direct for complex tasks?**  
No, not in this run. Orchestration was slower for task_11 and didn't complete complex tasks 12-20. The routing-only benefit (specialist for simple tasks) is real and valuable. Decomposition, as implemented, adds overhead without delivering the parallelism benefit.

**Does orchestration beat direct for simple tasks?**  
Yes. 37% faster, same correctness. This alone justifies the routing layer.

**Is orchestration worth the complexity?**  
The routing layer (TF-IDF + LogReg) is worth it — it's fast (<1ms), small, and delivers measurable gains. The decomposition/planning layer needs rework before it earns its keep. Specific issues:

- Subtask timeout (180s) is too aggressive for primary model calls generating long outputs
- Sequential subtask waves on a single model endpoint undo parallelism benefits
- Decomposing code generation into 5 subtasks may produce less coherent output than one well-prompted call

## Recommended Next Steps

1. **Ship routing-only mode as default** — classifier + specialist routing is validated. Decomposition is opt-in.
2. **Tune complex task handling** — raise subtask timeout to 240s, reduce decomposition granularity (2-3 subtasks max), or use streaming assembly instead of blocking subtasks.
3. **Re-run complex benchmark with lower complexity tasks** — current complex tasks (full FastAPI apps, CLI tools with edge cases) may exceed what's achievable in a single session at this quantization level.
4. **Measure correctness more granularly** — `code_runs` correctness check is binary. Partial credit (correct structure, wrong detail) would give a more useful signal.

## Run Notes

- Full orchestrated run timed out at 3600s (1 hour) before reaching complex tasks 12-20
- Results for tasks 12-20 (orchestrated) are marked N/A — not a model failure, a process timeout
- Orchestrated timings for tasks 01-10 and direct timings for all 20 tasks are live measurements from inference servers
- Servers remained healthy throughout; no crashes
