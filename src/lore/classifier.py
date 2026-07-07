"""Model-based task classifier. Uses specialist model for NLU, falls back to heuristic.

Replaces complexity.estimate() for complex tasks. The specialist model gets
the query + a classification prompt, returns structured JSON with:
- is_complex: bool
- task_type: one of TASK_BENCHMARKS keys
- estimated_subtasks: int
- suggested_model: "primary" | "specialist"
- hints: dict of extra hints for the decomposer

If the specialist call fails or returns invalid output, falls back to the
heuristic estimator (complexity.estimate()).
"""
import json
import logging
from dataclasses import dataclass, field

from lore.complexity import estimate as heuristic_estimate

logger = logging.getLogger(__name__)

_CLASSIFY_SYSTEM = """You are a task classifier for a local AI orchestration system.
Classify the user's task into one of these categories:
- classification: sorting/categorizing text
- extraction: pulling structured data from text
- summarization: condensing text
- code_gen: writing code
- testing: writing tests
- documentation: writing docs
- math: mathematical computation
- planning: multi-step planning
- review: code/text review

Also determine:
- is_complex: true if the task needs decomposition into subtasks
- estimated_subtasks: 1-5, how many subtasks if complex
- suggested_model: "primary" for reasoning/coding, "specialist" for simple tasks

Output JSON:
{"is_complex": bool, "task_type": "code_gen", "estimated_subtasks": 3, "suggested_model": "primary", "hints": {"multi_part": true, "needs_code": true}}"""

_VALID_TASK_TYPES = {
    "classification", "extraction", "summarization", "code_gen",
    "testing", "documentation", "math", "planning", "review",
}


@dataclass
class ClassificationResult:
    """Result of task classification."""
    is_complex: bool
    task_type: str
    estimated_subtasks: int
    suggested_model: str
    confidence: float
    hints: dict = field(default_factory=dict)
    source: str = "model"  # "model" or "heuristic"


class TaskClassifier:
    """Classifies tasks using specialist model, falls back to heuristic."""

    def __init__(self, server, config: dict | None = None):
        self._server = server
        self._config = config or {}
        self._max_tokens = self._config.get("max_tokens", 256)
        self._temperature = self._config.get("temperature", 0.1)
        self._fallback_model = self._config.get("fallback_model", "specialist")

    def classify(self, query: str, router_route: str = "PRIMARY") -> ClassificationResult:
        """Classify a task. Uses specialist model, falls back to heuristic."""
        # TOOL_ONLY always simple — skip model call
        if router_route == "TOOL_ONLY":
            return ClassificationResult(
                is_complex=False, task_type="classification",
                estimated_subtasks=1, suggested_model="primary",
                confidence=1.0, hints={}, source="heuristic",
            )

        try:
            result = self._server.chat(
                self._fallback_model,
                [
                    {"role": "system", "content": _CLASSIFY_SYSTEM},
                    {"role": "user", "content": f"Classify this task:\n{query}"},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                response_format={"type": "json_object"},
            )
            raw = result["choices"][0]["message"]["content"]
            return self._parse(raw, query, router_route)
        except Exception as e:
            logger.debug(f"Model classification failed ({e}), using heuristic")
            return self._heuristic_fallback(query, router_route)

    def _parse(self, raw: str, query: str, router_route: str) -> ClassificationResult:
        """Parse JSON classification response."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return self._heuristic_fallback(query, router_route)
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return self._heuristic_fallback(query, router_route)

        task_type = data.get("task_type", "planning")
        if task_type not in _VALID_TASK_TYPES:
            task_type = "planning"

        model = data.get("suggested_model", "primary")
        if model not in ("primary", "specialist"):
            model = "primary"

        return ClassificationResult(
            is_complex=bool(data.get("is_complex", False)),
            task_type=task_type,
            estimated_subtasks=max(1, min(5, int(data.get("estimated_subtasks", 2)))),
            suggested_model=model,
            confidence=0.85,  # model-based, decent confidence
            hints=data.get("hints", {}),
            source="model",
        )

    def _heuristic_fallback(self, query: str, router_route: str) -> ClassificationResult:
        """Fall back to the heuristic complexity estimator."""
        est = heuristic_estimate(query, router_route)
        # Map router route to task type guess
        task_type = "planning"
        if router_route == "SPECIALIST":
            task_type = "summarization"
        return ClassificationResult(
            is_complex=est.is_complex,
            task_type=task_type,
            estimated_subtasks=est.estimated_subtasks,
            suggested_model=est.suggested_model,
            confidence=est.confidence,
            hints={"signals": est.signals},
            source="heuristic",
        )
