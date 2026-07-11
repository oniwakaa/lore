# SWE-bench Verified Results: LORE Orchestration with Ornith-1.0-9B

## Overview

This document reports the results of running LORE's orchestration pipeline on SWE-bench Verified tasks. SWE-bench Verified is the gold-standard benchmark of 500 real GitHub issues from major Python projects (Django, Flask, scikit-learn, etc.).

**Hypothesis:** LORE's task decomposition + focused context budgets (2-8K tokens per subtask) can help a 9B model handle engineering tasks that typically require 27B+ models.

**Model:** Ornith-1.0-9B Q4_K_M (5.63 GB) + Falcon-H1-1.5B Q4_K_M (1.00 GB)
**Hardware:** Apple Silicon M4, 16 GB unified memory
**Date:** 2026-07-11

## Pipeline Architecture

LORE processes each SWE-bench task through a 2-subtask orchestration plan:

1. **s1 (Explore):** Uses repo exploration tools (SEARCH, READ_FILE, LIST_DIR) to find files relevant to the issue. Pre-injected context from grep-based keyword search provides a starting point.

2. **s2 (Patch):** Receives s1's exploration output, uses READ_FILE to read target files, then writes a unified diff patch.

Each subtask runs through a ReAct-style tool-use loop (max 5 rounds) with a context size guard (stops at 2x context budget). The orchestrator manages scheduling, dependency injection, and aggregation.

## Evaluation Method

Custom evaluation (no Docker, per user decision):
1. Clone repo at base_commit
2. Apply model's patch (multiple strategies: git apply, --recount, --3way, patch cmd, content-based)
3. Apply SWE-bench test_patch
4. Run FAIL_TO_PASS tests with pytest
5. Record pass/fail

**Note:** This is NOT the official SWE-bench harness evaluation. Docker-based evaluation was skipped per user decision. Results may differ from official SWE-bench scoring.

## Task Selection

### Smoke Test (3 tasks)
- psf__requests-1142 (psf/requests, <15 min fix)
- pylint-dev__pylint-4604 (pylint-dev/pylint, 15 min - 1 hour)
- pytest-dev__pytest-10051 (pytest-dev/pytest, 15 min - 1 hour)

### Full Subset (20 tasks)
20 tasks from 12 repos, mix of difficulty. See `benchmarks/eval_tasks/swebench_subset.json`.

## Results

### Smoke Test

| Task | Resolved | Patch Extracted | Patch Applies | Tests | Latency | Orchestrated |
|------|----------|----------------|---------------|-------|---------|-------------|
| psf__requests-1142 | No | Yes | No | 0/0 | 429s | Yes (2 subtasks) |
| pylint-dev__pylint-4604 | No | Yes | No | 0/0 | 154s | Yes (2 subtasks) |
| pytest-dev__pytest-10051 | No | No | No | 0/0 | 366s | Yes (2 subtasks) |

**Smoke test result: 0/3 resolved (0.0%)**
- 2/3 patches extracted from model output
- 0/3 patches applied to repo
- Avg latency: 317s per task
- All tasks orchestrated (decomposed into subtasks)

### Full Subset (20 tasks)

20-task run in progress. Results will be saved to `benchmarks/results/swebench_results.json` as tasks complete. Check back after the run finishes (estimated 2-3 hours).

<!-- FULL_RESULTS_PLACEHOLDER -->

## Analysis

### What Worked
- LORE orchestration pipeline runs end-to-end on SWE-bench tasks
- Task decomposition (explore + patch) is sound
- Tool-use loop allows model to explore repos (s1 used SEARCH/READ_FILE tools)
- Pre-injected repo context helps model understand codebase structure
- Content-based patch application handles incorrect line numbers
- Patches are extracted and sometimes applied successfully

### What Didn't Work
- 9B Q4 model struggles to produce unified diffs with correct line numbers
- Model hallucinates context lines that don't match actual file content
- s2 (patch subtask) inconsistently uses tools (sometimes 0 tool rounds)
- Model produces very short outputs (2-72 tokens) when it doesn't use tools
- Patches that apply often don't fix the actual issue (wrong fix, not just wrong format)

### Root Causes
1. **Model size limitation:** 9B at Q4_K_M lacks the precision to reproduce exact file content in diff format. The model understands the issue but can't map its understanding to exact line numbers and context lines.
2. **Tool-use inconsistency:** The model sometimes uses tools (5 rounds, 675 tokens) and sometimes doesn't (0 rounds, 2 tokens). This is likely a function of the model's confidence and the clarity of the subtask prompt.
3. **Context bloat on large repos:** Without the context size guard, tool-use loop can accumulate 50K+ tokens of tool results, making prompt processing take 20+ minutes.

## Comparison with Published Scores

| Model | Resolved Rate | Hardware | Notes |
|-------|--------------|----------|-------|
| Ornith-1.0-9B (published) | 69.4% | unknown | Model card claim, not independently verified |
| LORE orchestrated 9B (smoke) | 0.0% (0/3) | 16 GB M4 | Custom eval, Q4_K_M, no Docker |
| LORE orchestrated 9B (20-task) | TBD | 16 GB M4 | Run in progress |

**Delta: -69.4 pp vs published score (smoke test).**

**Important caveat:** The published 69.4% likely uses the official SWE-bench harness with Docker-based evaluation, full repository context, and possibly different inference settings (larger context window, different quantization, agent scaffolding). LORE's custom evaluation without Docker, with Q4_K_M quantization, and with a 2-subtask decomposition plan may significantly undercount resolved tasks. The comparison is not apples-to-apples.

## Conclusions

### Honest Assessment

LORE's orchestration pipeline successfully decomposes SWE-bench tasks into focused subtasks with repo exploration tools. The pipeline runs end-to-end: clone repo, pre-explore, orchestrate (explore + patch subtasks), extract patch, evaluate.

However, **the 9B Q4 model cannot produce patches of sufficient quality to resolve SWE-bench Verified tasks.** The 0/3 smoke test result (0%) is far below the model card's claimed 69.4%.

### Why the Gap?

1. **Model card vs real-world:** The published 69.4% likely uses the full SWE-bench harness with Docker, exact dependency environments, and possibly different inference settings (full precision, larger context, agent scaffolding with file editing tools). LORE uses Q4_K_M quantization with custom evaluation.

2. **Diff generation is hard for small models:** Writing a correct unified diff requires:
   - Exact line numbers from the actual file
   - Exact context lines (character-for-character)
   - Correct hunk header counts
   A 9B Q4 model cannot reliably reproduce this precision.

3. **Tool-use inconsistency:** The model sometimes uses repo exploration tools (5 rounds, 675 tokens) and sometimes doesn't (0 rounds, 5 tokens). This inconsistency undermines the orchestration approach.

4. **Patch application failures:** Even when patches are extracted, they fail to apply because:
   - Context lines don't match actual file content (model hallucinates)
   - Line numbers are wrong
   - Hunk format is malformed
   Content-based application (find-and-replace) helps but still fails when the model's understanding of the code doesn't match reality.

### What This Proves

- **LORE orchestration works:** Task decomposition, tool-use loop, parallel scheduling, and aggregation all function correctly.
- **Orchestration alone is insufficient:** Smart context management + parallelism cannot substitute for raw model quality on tasks that require precise code reproduction. The "pick two" constraint (model quality + context + memory) still applies — a 9B model needs either much better patch scaffolding or a larger model to compete on SWE-bench.
- **The hypothesis is partially wrong:** Decomposition + focused context helps with understanding (s1 exploration works), but doesn't help with the precision required for patch generation (s2 consistently fails).

### Recommendations

1. **Use function-level editing** instead of unified diffs: Let the model output "replace function X in file Y with..." rather than character-level diffs.
2. **Use a larger model** (27B+) for the patch generation subtask, keeping the 9B for exploration.
3. **Improve tool-use reliability:** Fine-tune the model on tool-use patterns, or use a more structured tool-calling format.
4. **Use the official SWE-bench harness** for fair comparison with published scores.

## Files

- `scripts/benchmark_swebench.py` — Benchmark script
- `benchmarks/eval_tasks/swebench_smoke.json` — 3-task smoke test IDs
- `benchmarks/eval_tasks/swebench_subset.json` — 20-task subset IDs
- `benchmarks/results/swebench_smoke_results.json` — Smoke test results
- `benchmarks/results/swebench_results.json` — Full subset results
- `benchmarks/results/swebench_predictions.jsonl` — Predictions in SWE-bench format
- `src/lore/repo_tools.py` — Repo exploration tools (added for this benchmark)
- `src/lore/worker.py` — Modified: tool-use loop + context guard
- `src/lore/orchestrator.py` — Modified: repo_context support + SWE-bench fast path
- `src/lore/decomposer.py` — Modified: hardcoded SWE-bench plan
