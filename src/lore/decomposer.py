"""Task decomposer: breaks a complex query into structured subtasks.

Uses the primary model (one planning call) with constrained JSON output
to produce a TaskPlan with 2-5 subtasks, a dependency graph, and an
aggregation prompt.
"""
import json
import logging
from dataclasses import dataclass, field

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


# Default planning prompt sent to the primary model
_PLANNING_SYSTEM = """You are a task planner for a local AI system with two models:
- PRIMARY (9B): strong at reasoning, coding, planning. Expensive. Use for complex tasks.
- SPECIALIST (1.5B): fast, good at simple extraction, formatting, summarization. Cheap.

Given a complex user task, break it into 2-5 subtasks. For each subtask specify:
- Which model to use (primary for reasoning/coding, specialist for simple extraction/formatting)
- Context budget in tokens (512=trivial, 2048=simple, 4096=moderate, 8192=complex, 16384+=heavy)
- A tailored system prompt (concise, focused on this subtask only)
- Dependencies: which subtasks must complete before this one
- Output format: free, json, code_python, code_bash

Rules:
- Max 5 subtasks. If the task needs more, merge related steps.
- Specialist handles: text formatting, summarization, extraction, simple transforms.
- Primary handles: reasoning, coding, planning, analysis, multi-step logic.
- Independent subtasks can run in parallel. Dependent ones must be sequential.
- The first subtask should have no dependencies (entry point).
- The last step is always aggregation (combining all outputs).

Output JSON:
{
  "subtasks": [
    {
      "id": "s1",
      "description": "...",
      "model": "primary",
      "context_budget": 4096,
      "system_prompt": "...",
      "dependencies": [],
      "max_tokens": 2048,
      "output_format": "code_python"
    }
  ],
  "aggregation_prompt": "You are combining multiple subtask outputs into a final response..."
}"""

_VALID_MODELS = {"primary", "specialist"}
_VALID_FORMATS = {"free", "json", "code_python", "code_bash"}


class TaskDecomposer:
    """Breaks a complex query into a TaskPlan with structured subtasks.

    Uses the primary model for planning. Returns a plan with 2-5 subtasks,
    dependency graph, and aggregation prompt.
    """

    def __init__(self, server, config: dict | None = None):
        self._server = server
        self._config = config or {}
        self._max_tokens = self._config.get("max_tokens", 1024)
        self._temperature = self._config.get("temperature", 0.3)
        self._max_subtasks = self._config.get("max_subtasks", 5)

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
            hint_str = ", ".join(f"{k}={v}" for k, v in hints.items() if v)
            if hint_str:
                user_content += f"\n\nClassifier hints: {hint_str}"

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
                logger.info(f"Decomposed '{query[:60]}' into {len(plan.subtasks)} subtasks")
                return plan
            logger.warning(f"Planning call returned invalid plan, using fallback")
        except Exception as e:
            logger.warning(f"Planning call failed ({e}), using fallback plan")

        return self._fallback_plan(query)

    def _parse_plan(self, raw: str, query: str) -> TaskPlan | None:
        """Parse the JSON response into a TaskPlan. None if invalid."""
        import re

        # Try direct JSON parse first
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # Strip markdown code fences if present (open or closed).
            # Models wrap JSON in ```json even with response_format=json_object.
            # Non-greedy \{.*?\} misses multi-object JSON; use greedy + optional close.
            data = None
            fence_match = re.search(r"```(?:json)?\s*(\{.*?)(?:```)?\s*$", raw, re.DOTALL)
            if fence_match:
                try:
                    data = json.loads(fence_match.group(1))
                except json.JSONDecodeError:
                    pass
            if data is None:
                # Grab everything between first { and last }
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if not match:
                    logger.debug(f"Raw planning response (no JSON found): {raw[:300]}")
                    return None
                candidate = match.group(0)
                try:
                    data = json.loads(candidate)
                except json.JSONDecodeError:
                    # Fix trailing commas
                    cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        data = json.loads(cleaned)
                    except json.JSONDecodeError:
                        # Repair truncated JSON by closing open brackets/braces
                        data = self._repair_truncated(cleaned)
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

    def _repair_truncated(self, text: str) -> dict | None:
        """Attempt to repair truncated JSON by closing open strings/brackets/braces.

        When max_tokens cuts the model mid-generation, the JSON is incomplete.
        Counts open vs close brackets/braces (respecting strings), detects
        if we're mid-string, then closes everything and tries to parse.
        """
        import re

        in_string = False
        escape = False
        stack: list[str] = []

        for ch in text:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in "}]":
                if stack and stack[-1] == ch:
                    stack.pop()

        repaired = text.rstrip()

        # If we're mid-string, close it and add comma
        if in_string:
            # Trim trailing whitespace/newlines inside the string
            repaired = repaired.rstrip()
            repaired += '",'
        elif repaired and repaired[-1] not in ",{}[":
            # Trailing incomplete key or value — trim back to last safe point
            # Find last occurrence of , or : to cut at
            last_safe = max(repaired.rfind(","), repaired.rfind('": '))
            if last_safe > 0:
                repaired = repaired[:last_safe + 1]
            if not repaired.endswith(","):
                repaired += ","

        # Fix any double commas
        repaired = re.sub(r",\s*,", ",", repaired)

        # Close all open brackets/braces
        for closer in reversed(stack):
            repaired += closer

        # Fix trailing commas before closers
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            logger.debug(f"Repair failed. Repaired text: {repaired[:300]}")
            return None

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
