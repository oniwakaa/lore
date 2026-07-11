"""Task decomposer: breaks a complex query into structured subtasks.

Uses the primary model (one planning call) with constrained JSON output
to produce a TaskPlan with 2-5 subtasks, a dependency graph, and an
aggregation prompt.
"""
import json
import logging
import re
from dataclasses import dataclass, field

from lore.templates import get_template

logger = logging.getLogger(__name__)


@dataclass
class SubTask:
    """A single subtask within a task plan."""
    id: str                       # "s1", "s2", ...
    description: str              # what this subtask does
    model: str                    # "primary" | "specialist"
    context_budget: int           # tokens (512 to 32768)
    system_prompt: str            # tailored system prompt for this subtask
    dependencies: list[str] = field(default_factory=list)  # subtask IDs that must complete first
    max_tokens: int = 2048        # generation limit
    output_format: str = "free"   # "free" | "json" | "code_python" | "code_bash"
    depends_on_outputs: bool = False  # True if needs previous outputs as input


@dataclass
class TaskPlan:
    """A decomposition plan for a complex task."""
    original_query: str
    subtasks: list[SubTask] = field(default_factory=list)
    aggregation_prompt: str = ""
    total_estimated_tokens: int = 0
    is_fallback: bool = False  # True if planning failed and trivial plan was used


# Planning prompt sent to the primary model — few-shot, with granularity +
# model-assignment + context-budget guidance.
_PLANNING_SYSTEM = """Task planner for local AI with two models:
- PRIMARY (9B): reasoning, coding, planning, analysis, debugging.
- SPECIALIST (1.5B): fast, extraction, formatting, summarization.

Break a complex task into 2-3 focused subtasks with a dependency graph.

## Rules
- Max 3 subtasks. Each should produce 500-1500 tokens of output.
- SPECIALIST: extraction/formatting/summarization. PRIMARY: code/reasoning.
- Context: extraction 1024-2048, code 4096-8192, reasoning 8192-16384. +2048 if deps.
- Output: code_python, json, or free.
- Keep subtasks focused and small — they run in parallel, not sequentially.

## Example — Moderate task (3 subtasks, mixed)
Task: "Parse CSV, extract emails, summarize."
{"subtasks": [
  {"id":"s1","description":"Write Python to parse CSV, extract emails","model":"primary","context_budget":4096,"system_prompt":"Programmer.","dependencies":[],"max_tokens":2048,"output_format":"code_python"},
  {"id":"s2","description":"Validate emails from CSV","model":"specialist","context_budget":2048,"system_prompt":"Extract info precisely.","dependencies":["s1"],"max_tokens":1024,"output_format":"json"},
  {"id":"s3","description":"Summarize emails in 2-3 sentences","model":"specialist","context_budget":1024,"system_prompt":"Summarize concisely.","dependencies":["s2"],"max_tokens":256,"output_format":"free"}
], "aggregation_prompt":"Combine code, emails."}

## Example — Complex task (4 subtasks with deps)
Task: "Build REST registration: route, validation, tests, docs."
{"subtasks": [
  {"id":"s1","description":"Write registration route with validation","model":"primary","context_budget":8192,"system_prompt":"Backend engineer.","dependencies":[],"max_tokens":4096,"output_format":"code_python"},
  {"id":"s2","description":"Write pytest tests for endpoint","model":"primary","context_budget":4096,"system_prompt":"Test engineer.","dependencies":["s1"],"max_tokens":2048,"output_format":"code_python"},
  {"id":"s3","description":"Write API docs for endpoint","model":"specialist","context_budget":2048,"system_prompt":"Clear docs.","dependencies":["s1"],"max_tokens":1024,"output_format":"free"},
  {"id":"s4","description":"Review for correctness and security","model":"primary","context_budget":4096,"system_prompt":"Reviewer.","dependencies":["s1","s2"],"max_tokens":2048,"output_format":"free"}
], "aggregation_prompt":"Combine route, tests, docs."}

## Output (JSON)
{"subtasks":[{"id":"s1","description":"...","model":"primary","context_budget":4096,"system_prompt":"...","dependencies":[],"max_tokens":2048,"output_format":"code_python"}],"aggregation_prompt":"Combine."}"""

_VALID_MODELS = {"primary", "specialist"}
_VALID_FORMATS = {"free", "json", "code_python", "code_bash"}


def compute_subtask_budget(subtask: SubTask, task_type: str,
                           total_memory_budget: int = 16384) -> int:
    """Compute appropriate context budget for a subtask.

    Base budget by task type, scaled by description length and dependency
    injection overhead. Clamped to [1024, total_memory_budget].
    """
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
        base += 2048

    return max(1024, min(base, total_memory_budget))


class TaskDecomposer:
    """Breaks a complex query into a TaskPlan with structured subtasks.

    Uses the primary model for planning. Returns a plan with 2-5 subtasks,
    dependency graph, and aggregation prompt.
    """

    def __init__(self, server, config: dict | None = None):
        self._server = server
        self._config = config or {}
        self._max_tokens = self._config.get("max_tokens", 1024)
        self._temperature = self._config.get("temperature", 0.2)
        self._max_subtasks = self._config.get("max_subtasks", 3)

    def decompose(self, query: str, hints: dict | None = None) -> TaskPlan:
        """Break a complex query into a TaskPlan.

        Sends one planning call to the primary model with constrained JSON
        output. Parses the response into SubTask objects with a validated
        dependency graph.

        Args:
            query: The task to decompose.
            hints: Optional classifier hints (task_type, estimated_subtasks,
                   suggested_model, multi_part, needs_code, etc.) that get
                   injected into the planning prompt to guide decomposition.

        Falls back to a trivial 2-subtask plan (do everything on primary,
        then aggregate) if the planning call fails or returns invalid JSON.
        """
        user_content = f"Task to decompose:\n{query}"
        if hints:
            hint_lines = []
            if hints.get("task_type"):
                hint_lines.append(f"- Task type: {hints['task_type']} (use this to choose output formats)")
            if hints.get("estimated_subtasks"):
                hint_lines.append(f"- Estimated complexity: {hints['estimated_subtasks']} subtasks suggested")
            if hints.get("suggested_model"):
                hint_lines.append(
                    f"- Suggested model: {hints['suggested_model']} "
                    f"(for the main work; use specialist for helper steps)"
                )
            # Pass through any extra hints
            for k, v in hints.items():
                if k not in ("task_type", "estimated_subtasks", "suggested_model") and v:
                    hint_lines.append(f"- {k}: {v}")
            if hint_lines:
                user_content += "\n\nClassifier analysis:\n" + "\n".join(hint_lines)

        messages = [
            {"role": "system", "content": _PLANNING_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        try:
            result = self._server.chat(
                "primary",
                messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                response_format={"type": "json_object"},
            )
            raw = result["choices"][0]["message"]["content"]
            plan = self._parse_plan(raw, query)
            if plan is not None:
                plan = self._validate_plan(plan, query, hints or {})
                logger.info(f"Decomposed '{query[:60]}' into {len(plan.subtasks)} subtasks")
                return plan
            logger.warning(f"Planning call returned invalid plan, using fallback")
        except Exception as e:
            logger.warning(f"Planning call failed ({e}), using fallback plan")

        return self._fallback_plan(query)

    def _validate_plan(self, plan: TaskPlan, query: str, hints: dict) -> TaskPlan:
        """Validate and fix common plan issues after parsing."""
        # 1. All subtasks on same model with no deps → doesn't benefit from orchestration
        models = {s.model for s in plan.subtasks}
        has_deps = any(s.dependencies for s in plan.subtasks)
        if len(plan.subtasks) <= 2 and models == {"primary"} and not has_deps:
            plan.is_fallback = True
            logger.info("Plan validated as fallback (all primary, no deps, <=2 subtasks)")
            return plan

        # 2. Recompute context budgets using task type
        task_type = hints.get("task_type", "")
        for st in plan.subtasks:
            st.context_budget = compute_subtask_budget(st, task_type)

        # 3. Clamp context budgets to reasonable range for 16GB device
        for st in plan.subtasks:
            if st.context_budget < 512:
                st.context_budget = 2048  # floor
            if st.context_budget > 16384:
                st.context_budget = 16384  # ceiling

        # 4. Every subtask should have a meaningful system prompt
        for st in plan.subtasks:
            if st.system_prompt == "You are a helpful assistant.":
                fmt = st.output_format
                template_name = fmt if fmt != "free" else "implementation"
                st.system_prompt = get_template(template_name)

        # 5. Recompute total estimated tokens
        plan.total_estimated_tokens = sum(st.context_budget for st in plan.subtasks)

        return plan

    def _parse_plan(self, raw: str, query: str) -> TaskPlan | None:
        """Parse the JSON response into a TaskPlan. None if invalid."""
        from lore.json_utils import parse_json_response

        data = parse_json_response(raw)
        if data is None:
            logger.debug(f"Raw planning response (unparseable): {raw[:300]}")
            return None

        raw_subtasks = data.get("subtasks", [])
        if not raw_subtasks or not isinstance(raw_subtasks, list):
            return None

        subtasks: list[SubTask] = []
        valid_ids: set[str] = set()

        for raw_st in raw_subtasks[:self._max_subtasks]:
            if not isinstance(raw_st, dict):
                # Truncated subtask (incomplete object) — skip it
                continue
            sid = raw_st.get("id", f"s{len(subtasks)+1}")
            model = raw_st.get("model", "primary")
            if model not in _VALID_MODELS:
                model = "primary"
            fmt = raw_st.get("output_format", "free")
            if fmt not in _VALID_FORMATS:
                fmt = "free"
            deps = raw_st.get("dependencies", [])
            if not isinstance(deps, list):
                deps = []

            def _safe_int(val, default):
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return default

            st = SubTask(
                id=sid,
                description=raw_st.get("description", ""),
                model=model,
                context_budget=_safe_int(raw_st.get("context_budget"), 4096),
                system_prompt=raw_st.get("system_prompt", "You are a helpful assistant."),
                dependencies=[d for d in deps if isinstance(d, str)],
                max_tokens=_safe_int(raw_st.get("max_tokens"), 2048),
                output_format=fmt,
                depends_on_outputs=bool(deps),  # if has deps, likely needs outputs
            )
            subtasks.append(st)
            valid_ids.add(sid)

        if not subtasks:
            return None

        # Validate dependency graph: filter deps to only valid IDs
        for st in subtasks:
            st.dependencies = [d for d in st.dependencies if d in valid_ids]
            # Update depends_on_outputs based on validated deps
            st.depends_on_outputs = len(st.dependencies) > 0

        # Ensure at least one subtask has no dependencies (entry point)
        if not any(not st.dependencies for st in subtasks):
            subtasks[0].dependencies = []
            subtasks[0].depends_on_outputs = False

        agg_prompt = data.get(
            "aggregation_prompt",
            "You are combining multiple subtask outputs into a final unified response. "
            "Synthesize the results coherently. Present a clean, complete answer to the original task.",
        )

        total_tokens = sum(st.context_budget for st in subtasks)

        return TaskPlan(
            original_query=query,
            subtasks=subtasks,
            aggregation_prompt=agg_prompt,
            total_estimated_tokens=total_tokens,
        )

    def _fallback_plan(self, query: str) -> TaskPlan:
        """Trivial plan: one primary subtask, then aggregate. Used on planning failure."""
        s1 = SubTask(
            id="s1",
            description=query,
            model="primary",
            context_budget=8192,
            system_prompt="You are a helpful assistant. Answer the task completely and accurately.",
            dependencies=[],
            max_tokens=4096,
            output_format="free",
            depends_on_outputs=False,
        )
        return TaskPlan(
            original_query=query,
            subtasks=[s1],
            aggregation_prompt="Present the following result as a clean, complete answer.",
            total_estimated_tokens=8192,
            is_fallback=True,
        )
