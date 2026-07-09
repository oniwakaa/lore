# Orchestration Engine Overhaul — Research Findings

Date: 2026-07-09

## 1. Task Decomposition: DGI Scaling Law (√S)

**Source:** ACONIC (arxiv:2510.07772, Columbia University, Oct 2025)

ACONIC introduces a formal complexity framework for LLM task decomposition using constraint satisfaction problems. Key findings:

- **Principled decomposition** (minimizing treewidth) improves task completion by 9-40% over heuristic chain-of-thought decomposition.
- **Complexity-based decomposition** defines "frontiers of difficulty" — tasks beyond a certain complexity threshold benefit from decomposition, while simple tasks don't.
- Even LLaMA-3-70B benefits from decomposition, so 9B models should too, but with **simpler plans** (fewer subtasks, lower treewidth per subtask).

**Verdict on √S scaling:** The ACONIC paper doesn't directly validate the √S (square root of steps) heuristic, but it supports the core principle: decomposition should be **complexity-aware**, not fixed-granularity. For 9B models, the √S guidance is reasonable but should be conservative (2-4 subtasks for most tasks, max 5). The key insight is that **over-decomposition hurts** — too many subtasks increase coordination overhead and error propagation.

**Implementation alignment:** Our `_validate_plan()` marks plans with ≤2 all-primary no-dependency subtasks as fallback (no orchestration benefit). This aligns with ACONIC's finding that simple tasks don't benefit from decomposition.

## 2. Dynamic Context Allocation with Heterogeneous Models

**Source:** DAAO (arxiv:2509.11079, Sep 2025)

DAAO (Difficulty-Aware Agentic Orchestration) dynamically adapts workflow depth and LLM assignment based on query difficulty:

- **Workflow depth scaling:** L = ⌈d · ℓ⌉ where d ∈ [0,1] is predicted difficulty and ℓ is max layers. Harder queries get deeper workflows.
- **Heterogeneous LLM routing:** Different LLMs are assigned to different operators based on task needs. Smaller models handle simple subtasks; larger models handle complex reasoning.
- **Results:** 11.21% accuracy improvement over SOTA multi-agent systems with only 64% inference cost, using heterogeneous models.

**Verdict on water-filling approach:** DAAO validates the principle of dynamic resource allocation based on task difficulty. Our `compute_subtask_budget()` implements a simplified version: base budget by task type, scaled by description length, with extra budget for dependency injection. This is a heuristic approximation of the water-filling principle — allocate more context to harder subtasks, less to simple ones.

**Implementation alignment:** Our `TEMPERATURE_MAP` (0.1 for code/json, 0.7 for free text) and `_estimate_max_tokens()` are consistent with DAAO's approach of tailoring execution parameters to task characteristics.

## 3. LLM Orchestration Error Recovery & Retry

**Sources:**
- Multi-Agent Error Recovery Patterns (blog.naitive.cloud, Jun 2026)
- LLM Error Handling and Fallback Strategies (buildmvpfast.com, Mar 2026)
- Mastering Retry Logic Agents (sparkco.ai, Oct 2025)

Common patterns identified across sources:

1. **Retry with backoff** — Standard pattern. Increase resources on each retry.
2. **Multi-model fallback chains** — If model A fails, try model B. Our specialist→primary escalation matches this.
3. **Error context injection** — Append failure information to the prompt on retry. Our implementation does this: `"A previous attempt failed with error: '{error}'. Learn from this mistake."`
4. **Validation gates** — Check output quality before accepting. Our verifier module handles this.
5. **Circuit breakers** — Stop retrying after N failures. Our `max_retries=2` implements this.

**Verdict:** Our `run_with_retry()` implementation aligns with industry-standard patterns. The escalation strategy (more tokens, error context, model upgrade) is well-supported by the literature.

## 4. GBNF Grammars for Decomposer JSON Output

**Finding:** The decomposer already uses `response_format={"type": "json_object"}` in the `server.chat()` call. This is llama-server's built-in JSON mode, which uses GBNF grammars internally to constrain output to valid JSON.

**Assessment:** A custom GBNF grammar for the specific plan schema (with field names, types, and constraints) would further reduce parsing failures, but:
1. The built-in JSON mode already eliminates most malformed JSON.
2. The decomposer has robust fallback parsing (fence stripping, brace matching, truncated JSON repair).
3. Custom GBNF grammars for complex schemas are hard to maintain and can overly constrain the model's planning flexibility.
4. The `_validate_plan()` step catches semantic issues (bad model assignments, extreme budgets) that GBNF can't prevent.

**Verdict:** Skip custom GBNF grammar for now. The combination of built-in JSON mode + robust parsing + plan validation is sufficient. If parsing failures become a significant issue, a custom grammar can be added later.

## Summary

| Approach | Research Support | Implementation Status |
|----------|-----------------|----------------------|
| √S granularity guidance | Supported (complexity-aware decomposition) | Implemented in planning prompt |
| Dynamic context budget | Supported (DAAO: difficulty-aware allocation) | `compute_subtask_budget()` in decomposer |
| Dynamic temperature | Supported (task-specific parameters) | `TEMPERATURE_MAP` in worker |
| Retry with escalation | Standard pattern (multiple sources) | `run_with_retry()` in worker |
| Specialist pre-summarization | Novel (not in literature, but logical) | `_pre_summarize_for_aggregation()` |
| Progressive aggregation | Standard tree-reduction pattern | `_aggregate_progressive()` |
| GBNF for plan output | Already using built-in JSON mode | Sufficient, no custom grammar needed |
