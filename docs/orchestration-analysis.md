# LORE Orchestration Deep Analysis

## Current Architecture (How It Works Today)

```
User Request
    ↓
Router (TF-IDF + LogReg) → route: PRIMARY/SPECIALIST/TOOL_ONLY
    ↓
Classifier (Falcon-H1-1.5B) → is_complex, task_type, estimated_subtasks
    ↓
_orchestrate()
    ↓
Decomposer (primary 9B, 1 planning call, max_tokens=1024)
    → TaskPlan with 2-5 SubTasks, each with model/context_budget/system_prompt
    ↓
Scheduler (topological sort → waves)
    ↓
Worker × N (each gets own ContextManager with subtask's budget)
    → server.chat(model, messages, max_tokens=2048, temperature=0.7)
    ↓
Aggregator (primary 9B, max_tokens=4096, temperature=0.5, timeout=300)
    → combine all subtask outputs into final response
```

## Identified Flaws (12 Issues)

### CRITICAL — Directly Impacts Quality

**1. Decomposer token budget too small (1024 tokens)**
The planning call to the primary model is limited to 1024 output tokens. A good decomposition plan for a 5-subtask task needs ~600-800 tokens (JSON with descriptions, system prompts, dependencies). At 1024, the model has almost no room for reasoning — it's forced to produce shallow plans. Real-world orchestrators use 2048-4096 for planning.

**2. No decomposition examples in the planning prompt**
The `_PLANNING_SYSTEM` prompt tells the model WHAT to produce (JSON format) but never shows a GOOD example. A 9B model needs examples to produce quality plans. Without them, it produces inconsistent plans — sometimes good, sometimes garbage.

**3. Static temperature for all subtasks (0.7)**
Worker.run() uses temperature=0.7 for everything. Code generation needs 0.0-0.3 (deterministic). JSON extraction needs 0.1 (structured). Creative text can use 0.7-0.9. Wrong temperature = wrong output quality.

**4. Static max_tokens for all subtasks (2048)**
Worker.run() uses max_tokens=2048 regardless of task. A "summarize this in 2 sentences" subtask gets 2048 tokens. A "implement a full class with tests" subtask gets 2048 tokens. One wastes tokens, the other gets truncated.

**5. No error recovery on subtask failure**
When a Worker fails, the error propagates through. No retry, no fallback, no re-decomposition. The orchestrator logs the failure and continues — producing incomplete results.

**6. Aggregation is all-or-nothing**
_aggregate() sends ALL subtask outputs to the primary model in one call. If that call times out (300s), it falls back to string concatenation. No intermediate aggregation, no progressive summarization, no quality control.

### HIGH — Impacts Efficiency

**7. Decomposer doesn't control decomposition granularity**
The planning prompt says "2-5 subtasks" but doesn't scale with task complexity. Research (DGI paper, 2026) shows optimal decomposition scales as √S where S = required steps. A 3-step task should get 2-3 subtasks. A 15-step task should get 5-6. LORE uses the same 2-5 range for everything.

**8. Model assignment is disconnected from benchmark data**
The decomposer assigns "primary" or "specialist" based on vague rules ("specialist for extraction/formatting"). The registry's actual benchmark data (which model scores what per task type) is never consulted during decomposition. It's only checked at execution time (Issue #4 fix) — too late to influence planning.

**9. Context budget is chosen by the decomposer model, not computed**
The 9B model decides context_budget (512-16384) during planning. But the model doesn't know the actual memory constraints, KV cache state, or how much context previous outputs will consume. It's guessing.

**10. Dynamic context sizing uses regex, ignores classifier**
sizing.py uses regex patterns (`_COMPLEX_KEYWORDS`, `_SIMPLE_KEYWORDS`) to estimate context budget. The classifier already identified task_type and complexity — this information is never passed to the sizing function.

### MEDIUM — Architecture Gaps

**11. No topology selection**
LORE always uses the same orchestration pattern: decompose → sequential waves → aggregate. Research (AdaptOrch, 2026) shows that topology selection (parallel vs sequential vs hierarchical vs hybrid) improves performance 12-23%. LORE should choose topology based on task dependency structure.

**12. No replanning on failure**
When a subtask fails, LORE doesn't replan. Research (HTN replanning) shows that localized replanning (re-decompose just the failed subtree, not the whole plan) is significantly more efficient than full re-planning or no replanning.

## Proposed Fixes (Prioritized)

### Phase 1: Decomposer Quality (Issues 1, 2, 7)

**1a. Increase decomposer max_tokens to 2048-4096**
More tokens = better plans. The decomposer is a one-time cost.

**1b. Add few-shot examples to planning prompt**
Show 2-3 good decomposition examples (simple, moderate, complex) in the planning system prompt. Include examples of model assignment (specialist for extraction, primary for coding).

**1c. Implement DGI-aware decomposition**
Before calling the decomposer, estimate the task's required steps S from the classifier's hints. Guide the model toward ~√S subtasks. The planning prompt should say "This task appears to need ~N steps. Create approximately √N subtasks."

**1d. Add structured output format for plans**
Use GBNF grammar to force valid plan JSON output. Eliminates all JSON parsing failures.

### Phase 2: Worker Execution (Issues 3, 4, 5)

**2a. Dynamic temperature based on output_format**
```python
TEMPERATURE_MAP = {
    "code_python": 0.1,
    "code_bash": 0.1,
    "json": 0.1,
    "free": 0.7,
}
```

**2b. Dynamic max_tokens based on task description**
Estimate expected output length from the subtask description:
- "summarize in 2 sentences" → max_tokens=256
- "write a function" → max_tokens=1024
- "implement a full class with tests" → max_tokens=4096

**2c. Add retry with escalation on subtask failure**
```python
def run_with_retry(self, max_retries=2):
    for attempt in range(max_retries + 1):
        result = self.run()
        if result.success:
            return result
        # Escalate: increase tokens, adjust temperature, add error context
        self._subtask.max_tokens *= 2
        self._subtask.system_prompt += f"\nPrevious attempt failed: {result.error}"
    return result  # final attempt
```

### Phase 3: Context Budget (Issues 9, 10)

**3a. Compute context budget from classifier output**
Instead of asking the decomposer model to guess, compute it:
```python
def compute_subtask_budget(subtask, task_type, total_budget):
    base = {
        "extraction": 2048, "summarization": 2048,
        "code_gen": 4096, "testing": 4096,
        "planning": 8192, "review": 4096,
    }.get(task_type, 4096)
    # Scale by subtask complexity signals
    if len(subtask.description.split()) > 100:
        base *= 2
    # Reserve space for previous outputs if dependent
    if subtask.depends_on_outputs:
        base += 2048  # reserve for injected context
    return min(base, total_budget)
```

**3b. Pass classifier task_type to sizing function**
Wire `classification.task_type` and `classification.is_complex` into `estimate_context_budget()` instead of regex.

### Phase 4: Aggregation (Issue 6)

**4a. Tree-based progressive aggregation**
Instead of one giant aggregation call, aggregate in pairs:
```python
def _aggregate_progressive(self, query, results):
    parts = [r.content for r in results.values()]
    while len(parts) > 1:
        next_parts = []
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                combined = self._aggregate_pair(query, parts[i], parts[i+1])
                next_parts.append(combined)
            else:
                next_parts.append(parts[i])
        parts = next_parts
    return parts[0]
```

**4b. Specialist quick-summarize before aggregation**
If total output > 3000 tokens, use the specialist to summarize each subtask to 200 tokens before sending to the primary for final aggregation. This cuts aggregation input from ~3000 to ~400 tokens.

### Phase 5: Topology & Replanning (Issues 11, 12)

**5a. Topology selection based on dependency graph**
After decomposition, analyze the dependency graph:
- No dependencies → all parallel (fastest)
- Linear chain → sequential (necessary)
- Mixed → waves (current behavior, correct for mixed)
- Complex DAG → hierarchical (decompose into parallel groups)

**5b. Localized replanning on failure**
When a subtask fails, replan only that subtree:
```python
def _replan_failed(self, failed_subtask, original_plan):
    # Re-decompose just the failed subtask with error context
    new_subtasks = self._decomposer.decompose_single(
        failed_subtask.description + f"\nPrevious attempt error: {failed_subtask.error}",
        context_budget=failed_subtask.context_budget * 2,
    )
    # Replace in plan, re-schedule
    ...
```

## Research References

1. **DGI (Decomposition Granularity Index)** — Optimal subtasks ≈ √S. Over-decomposition at DGI>5 causes 71% token waste on coordination. (clawRxiv, 2026)
2. **AdaCtx** — Dynamic context allocation via water-filling. 12.8% improvement over uniform. 31% fewer tokens at matched quality. (clawRxiv, 2026)
3. **ZEBRA** — Budget-aware orchestration. Recovers 94.4% of quality at 50% budget. (arXiv, 2026)
4. **AdaptOrch** — Topology selection (parallel/sequential/hierarchical/hybrid) improves 12-23% over static topology. (arXiv, 2026)
5. **CASTER** — Dynamic model routing per subtask. 72.4% cost reduction at matched success rate. (arXiv, 2026)
6. **Orchestrator Pattern** — Context isolation: <5K token subtasks outperform 80K+ accumulated contexts. Spawning overhead is 200-500ms (noise). (PromptEngines, 2026)
7. **HTN Replanning** — Localized replanning (re-decompose failed subtree only) is significantly more efficient than full replanning. (Zylos Research, 2026)
