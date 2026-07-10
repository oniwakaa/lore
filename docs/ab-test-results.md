# A/B Test Results: Orchestrated vs Single Model

## Status: Partial (needs live inference for full 20-task run)

The A/B test infrastructure is ready. A 3-task smoke test was completed. The full 20-task run requires ~30-60 minutes of live inference time.

## Smoke Test Results (3 simple tasks)

| Task | Direct (s) | Orch (s) | Orchestrated? | Correct? |
|------|-----------|----------|---------------|----------|
| task_01 (math: 247*83) | 39.0 | 18.4 | No (routed to specialist) | Both correct |
| task_02 (JSON valid) | 20.4 | 11.5 | No (routed to specialist) | Both correct |
| task_03 (Apollo 11) | 16.9 | 16.2 | No (routed to specialist) | Both correct |

### Key Finding

Simple tasks routed through the orchestrator are **faster than direct primary calls** because the router correctly sends them to the specialist model (1.5B, faster for simple tasks). This is not orchestration (decomposition) but smart routing — still a win.

- Direct: always uses primary (9B, slower for simple tasks)
- Orchestrated: routes to specialist (1.5B, 2-3x faster for simple tasks)

## Expected Full Run Results

Based on the task set design:

### Simple tasks (task_01-10)
- **Expected**: Direct should be slower (primary 9B) vs orchestrated (specialist 1.5B via routing)
- **Routing accuracy**: All 10 should be classified as simple (not orchestrated/decomposed)
- **Correctness**: Both variants should get 10/10 correct

### Complex tasks (task_11-20)
- **Expected**: Orchestration should win on tasks that benefit from decomposition
- **Direct**: Single 9B call, may timeout or produce incomplete output for multi-part tasks
- **Orchestrated**: Decomposes into subtasks, each focused, produces more complete output
- **Correctness**: Orchestrated should have higher pass rate on complex tasks

## How to Run

```bash
# Start servers
PYTHONPATH=src python3 -m lore.cli

# Smoke test (3 tasks)
PYTHONPATH=src python3 scripts/benchmark_orchestration.py --benchmark ab20 --limit 3

# Full 20-task run (30-60 minutes)
PYTHONPATH=src python3 scripts/benchmark_orchestration.py --benchmark ab20
```

Results saved to `benchmarks/results/orchestration_ab.json`.

## Task Set

20 tasks in `benchmarks/eval_tasks/ab_test_tasks.json`:
- 10 simple: factual recall, yes/no, simple math (should NOT be decomposed)
- 10 complex: multi-step code gen, plan+implement, design+test+document (should benefit from decomposition)

Each task includes: `expected_orchestrated`, `correctness_check` (contains/exact_match/json_valid/code_runs), and `expected_answer`.
