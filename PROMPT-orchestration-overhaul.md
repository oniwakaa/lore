# Prompt: Overhaul LORE Orchestration Engine

## Identity & Context

You are working on **LORE** (Local Orchestration & Runtime Engine) at `~/projects/lore`. Read `AGENTS.md` for project context. Run with `PYTHONPATH=src`.

LORE orchestrates multiple small models on 16 GB edge devices. The orchestration pipeline: classifier → decomposer → scheduler → workers → aggregator. It recently scored 87% on HumanEval — but that's with most tasks routed directly. The orchestrated path (57% pass on complex tasks) has significant room for improvement.

**Read `docs/orchestration-analysis.md` first.** It contains a detailed analysis of 12 flaws in the current orchestration pipeline, prioritized by impact, with proposed fixes and research references. This prompt implements that analysis.

## The Mission

Overhaul the orchestration engine to produce higher-quality decompositions, smarter resource allocation, and more robust execution. The goal: when LORE DOES orchestrate a complex task, it should produce better results than a single model could — not worse.

## Phase 1: Decomposer Quality (CRITICAL — do this first)

The decomposer is the brain of orchestration. If the plan is bad, everything downstream fails.

### 1.1 Increase Planning Token Budget

In `src/lore/decomposer.py`:
- Change `self._max_tokens = self._config.get("max_tokens", 1024)` to default **2048**
- Change `self._temperature = self._config.get("temperature", 0.3)` to default **0.2** (more deterministic planning)

### 1.2 Rewrite Planning Prompt with Few-Shot Examples

Replace `_PLANNING_SYSTEM` in `src/lore/decomposer.py` with a much better prompt that includes:

1. **Clear role definition** — "You are a task planner for a local AI system with a 9B primary model and a 1.5B specialist model."
2. **2-3 few-shot decomposition examples** showing good plans for:
   - A simple multi-part task (2 subtasks, both primary)
   - A moderate coding task (3 subtasks, mixed primary/specialist)
   - A complex multi-file task (4 subtasks with dependencies)
3. **Granularity guidance** — "For a task that appears to need S sequential steps, create approximately ⌈√S⌉ subtasks. Most tasks need 2-4. Never exceed 5."
4. **Model assignment rules with examples**:
   - Specialist (1.5B): text extraction, formatting, summarization, simple transforms, schema validation
   - Primary (9B): code generation, reasoning, planning, analysis, debugging, multi-step logic
5. **Context budget guidance by task type**:
   - Extraction/formatting: 1024-2048
   - Code generation: 4096-8192
   - Complex reasoning: 8192-16384
   - "Always budget for previous step outputs if this subtask has dependencies"
6. **Output format guidance**: "code_python for code tasks, json for structured data, free for text"

### 1.3 Add Classifier Hints to Decomposition

In `orchestrator.py`'s `_orchestrate()`, the `hints` dict passed to `decompose()` already includes `task_type` and `estimated_subtasks`. But the decomposer doesn't use these to constrain the plan. In the planning prompt, add:

```
Classifier analysis:
- Task type: {task_type} (use this to choose output formats)
- Estimated complexity: {estimated_subtasks} subtasks suggested
- Suggested model: {suggested_model} (for the main work; use specialist for helper steps)
```

### 1.4 Validate Plan Quality After Parsing

After `_parse_plan()`, add a `_validate_plan()` step:

```python
def _validate_plan(self, plan: TaskPlan, query: str) -> TaskPlan:
    """Validate and fix common plan issues."""
    # 1. All subtasks on same model with no deps → likely should just dispatch directly
    models = {s.model for s in plan.subtasks}
    if len(plan.subtasks) <= 2 and models == {"primary"} and not any(s.dependencies for s in plan.subtasks):
        # This plan doesn't benefit from orchestration — mark as fallback
        plan.is_fallback = True
        return plan
    
    # 2. All subtasks depend on the first → linear chain, fine but check if first is too big
    # 3. Context budgets are reasonable
    for st in plan.subtasks:
        if st.context_budget < 512:
            st.context_budget = 2048  # floor
        if st.context_budget > 16384:
            st.context_budget = 16384  # ceiling for 16GB
    
    # 4. Every subtask has a system prompt (not just default)
    for st in plan.subtasks:
        if st.system_prompt == "You are a helpful assistant.":
            # Assign template based on output_format
            st.system_prompt = get_template(st.output_format if st.output_format != "free" else "implementation")
    
    return plan
```

## Phase 2: Dynamic Worker Configuration (HIGH)

### 2.1 Dynamic Temperature

In `worker.py`, change the hardcoded `temperature=0.7` to be driven by output_format:

```python
TEMPERATURE_MAP = {
    "code_python": 0.1,   # deterministic code
    "code_bash": 0.1,     # deterministic scripts
    "json": 0.1,          # structured output
    "free": 0.7,          # general text
}
```

Use `TEMPERATURE_MAP.get(self._subtask.output_format, 0.7)` in the `server.chat()` call.

### 2.2 Dynamic max_tokens

Estimate appropriate max_tokens from the subtask description:

```python
def _estimate_max_tokens(description: str, output_format: str) -> int:
    """Estimate output token budget from task description."""
    words = len(description.split())
    
    if output_format in ("code_python", "code_bash"):
        # Code: scale with description complexity
        if words < 30:
            return 1024    # simple function
        elif words < 80:
            return 2048    # moderate implementation
        else:
            return 4096    # complex implementation with tests
    elif output_format == "json":
        return 1024        # structured data is compact
    else:
        # Free text: scale with expected output
        if any(kw in description.lower() for kw in ["summarize", "brief", "short", "one sentence"]):
            return 256
        elif any(kw in description.lower() for kw in ["explain", "describe", "list"]):
            return 1024
        else:
            return 2048
```

Wire this into `Worker.__init__()` — if `subtask.max_tokens` is the default (2048), override with the estimate.

### 2.3 Retry with Escalation

In `worker.py`, add a `run_with_retry()` method:

```python
def run_with_retry(self, max_retries: int = 2) -> WorkerResult:
    """Run subtask with retry on failure. Escalate on each attempt."""
    for attempt in range(max_retries + 1):
        result = self.run()
        if result.success:
            return result
        
        if attempt < max_retries:
            logger.warning(f"Subtask {self._subtask.id} failed (attempt {attempt+1}): {result.error}")
            # Escalate: more tokens, lower temperature, error context
            self._subtask.max_tokens = min(self._subtask.max_tokens * 2, 8192)
            self._subtask.system_prompt += (
                f"\n\nIMPORTANT: A previous attempt to complete this task failed with error: "
                f"'{result.error}'. Learn from this mistake. "
                f"Be more careful with edge cases and output format."
            )
            # If specialist failed, try primary
            if self._subtask.model == "specialist" and attempt == 0:
                self._subtask.model = "primary"
                logger.info(f"Escalating subtask {self._subtask.id} from specialist to primary")
    
    return result
```

Update `orchestrator.py`'s `_execute_wave()` to call `worker.run_with_retry()` instead of `worker.run()`.

## Phase 3: Context Budget Optimization (HIGH)

### 3.1 Compute Context Budget from Task Type

Replace the decomposer's guessed context_budget with a computed value:

```python
def compute_subtask_budget(subtask: SubTask, task_type: str, 
                           total_memory_budget: int = 16384) -> int:
    """Compute appropriate context budget for a subtask."""
    # Base budget by task type
    type_budgets = {
        "extraction": 2048, "summarization": 2048, "classification": 1024,
        "code_gen": 4096, "testing": 4096, "documentation": 4096,
        "planning": 8192, "review": 4096, "math": 4096,
    }
    base = type_budgets.get(task_type, 4096)
    
    # Scale by description length (longer description = more complex)
    desc_words = len(subtask.description.split())
    if desc_words > 100:
        base = min(base * 2, 16384)
    elif desc_words < 20:
        base = max(base // 2, 1024)
    
    # Reserve space for injected previous outputs
    if subtask.depends_on_outputs:
        base += 2048  # room for previous step context
    
    # Clamp to reasonable range
    return max(1024, min(base, total_memory_budget))
```

After plan parsing, recompute all budgets:
```python
for st in plan.subtasks:
    st.context_budget = compute_subtask_budget(st, task_type)
```

### 3.2 Wire Classifier into Context Sizing

In `src/lore/sizing.py`, update `estimate_context_budget()` to accept optional classifier info:

```python
def estimate_context_budget(route: str, query: str, config: dict,
                            task_type: str = None, is_complex: bool = None) -> int:
    # If classifier provided task_type, use it instead of regex
    if task_type:
        base_budgets = {
            "extraction": 2048, "summarization": 2048, "classification": 2048,
            "code_gen": 8192, "testing": 8192, "documentation": 4096,
            "planning": 16384, "review": 8192, "math": 8192,
        }
        budget = base_budgets.get(task_type, config.get("default_budget", 16384))
        if is_complex:
            budget = min(budget * 2, config.get("max_budget", 32768))
        return budget
    
    # Fall back to current regex-based logic
    ...
```

Update `orchestrator.py` to pass `classification.task_type` and `classification.is_complex` to the sizing function.

## Phase 4: Aggregation Improvements (MEDIUM)

### 4.1 Specialist Pre-Summarization

Before aggregation, if total subtask output exceeds 3000 tokens, use the specialist to summarize each subtask to ~200 tokens:

```python
def _pre_summarize_for_aggregation(self, results: dict[str, WorkerResult]) -> dict[str, str]:
    """Use specialist to summarize long subtask outputs before aggregation."""
    summaries = {}
    for sid, result in results.items():
        if len(result.content) > 1000:  # only summarize long outputs
            try:
                resp = self._server.chat("specialist", [
                    {"role": "system", "content": "Summarize the following in 200 tokens or less. Keep all key details, code snippets, and conclusions."},
                    {"role": "user", "content": result.content},
                ], max_tokens=300, temperature=0.1)
                summaries[sid] = resp["choices"][0]["message"]["content"]
            except Exception:
                summaries[sid] = self._truncate_output(result.content, 200)
        else:
            summaries[sid] = result.content
    return summaries
```

### 4.2 Progressive Aggregation

For plans with 4+ subtasks, aggregate in pairs (tree reduction):

```python
def _aggregate_progressive(self, query: str, plan: TaskPlan,
                           results: dict[str, WorkerResult]) -> str:
    """Tree-based aggregation: reduce N results to 1 via log(N) calls."""
    parts = [(st.description, results[st.id].content) for st in plan.subtasks if st.id in results]
    
    while len(parts) > 2:
        next_parts = []
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                combined = self._aggregate_pair(query, parts[i], parts[i+1])
                next_parts.append(combined)
            else:
                next_parts.append(parts[i])
        parts = next_parts
    
    # Final aggregation
    return self._aggregate_final(query, parts)
```

This reduces the aggregation context from O(N×subtask_size) to O(2×subtask_size) per call.

## Phase 5: Research & Verify

Before implementing, verify these approaches against current research:

1. **Search for "task decomposition best practices LLM 2026"** and verify the DGI scaling law (√S) applies to local models
2. **Search for "dynamic context allocation multi-agent"** and verify the water-filling approach works with heterogeneous models (9B + 1.5B)
3. **Search for "LLM orchestration error recovery retry"** and verify the retry-with-escalation pattern
4. **Check if GBNF grammars can constrain the decomposer's JSON output** — this would eliminate ALL plan parsing failures

Document your findings in `docs/orchestration-research.md`.

## Phase 6: Integration & Testing

### 6.1 Update Tests

All 237 existing tests must pass. Add new tests for:
- `_validate_plan()` — test with bad plans (all-same-model, no system prompts, extreme budgets)
- `_estimate_max_tokens()` — test with different description types
- `compute_subtask_budget()` — test with different task types and dependency patterns
- `run_with_retry()` — test retry logic, escalation from specialist to primary
- Dynamic temperature — verify code tasks get low temperature

### 6.2 Integration Test

Run `--benchmark humaneval --limit 20` and compare:
- v2 (current): 87% pass@1, 101s avg
- v3 (after fixes): should be ≥87% pass@1 (orchestrated tasks should improve)

### 6.3 Document Changes

Update `docs/orchestration-analysis.md` with:
- Which fixes were implemented
- Test results
- Any deviations from the proposed plan

## Files to Modify (in order)

1. `src/lore/decomposer.py` — Planning prompt, max_tokens, validation
2. `src/lore/worker.py` — Dynamic temperature, dynamic max_tokens, retry
3. `src/lore/orchestrator.py` — Budget computation, pre-summarization, progressive aggregation, retry integration
4. `src/lore/sizing.py` — Classifier-aware context sizing
5. `tests/test_orchestrator.py` — New tests
6. `tests/test_worker.py` — New tests (create if doesn't exist)
7. `docs/orchestration-research.md` — Research findings

## Constraints

1. **All 237 existing tests must pass** after every change
2. **No new dependencies** — use what's in pyproject.toml
3. **Don't break the direct dispatch path** — simple tasks must still route directly unchanged
4. **Backward compatible config** — all new behavior should have defaults that work without config changes
5. **Measure before/after** — run `--limit 20` benchmark before and after to verify improvement

## When to Ask

- If a proposed fix requires changing the `SubTask` dataclass (adding new fields), document the change and check all consumers
- If GBNF grammar for plan output is too complex, skip it and focus on better JSON parsing
- If progressive aggregation adds too much latency (extra inference calls), fall back to specialist pre-summarization only
- If the DGI scaling law doesn't hold for9B models (smaller models may need simpler plans), adjust the guidance accordingly

## Success Criteria

After all phases:
1. Orchestrated tasks on HumanEval subset should pass at ≥70% (up from 57%)
2. Overall HumanEval pass@1 should be ≥87% (no regression)
3. Aggregation timeout should be eliminated (pre-summarization + progressive aggregation)
4. Per-subtask latency should be more predictable (dynamic token budgets prevent truncation)
