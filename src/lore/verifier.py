"""Validate and repair structured model outputs.

Wire into _dispatch(): after model response, validate if route expects
structured output. Retry with repair hints up to max_repair_attempts.
"""
import ast
import json
import logging
import re

from lore.json_utils import strip_fences, extract_json_object

logger = logging.getLogger(__name__)

# Task types that expect structured output
_STRUCTURED_TASK_TYPES = {"json", "code_python", "code_bash"}


def _task_type_from_route(route: str, json_mode: bool) -> str:
    """Infer task type from route + json_mode flag."""
    if json_mode:
        return "json"
    return "free_form"


class Verifier:
    """Validate and repair structured model outputs."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._enabled = cfg.get("enabled", True)
        self._max_attempts = cfg.get("max_repair_attempts", 2)
        self._json_validation = cfg.get("json_validation", True)
        self._code_syntax_check = cfg.get("code_syntax_check", True)

    def validate(self, output: str, task_type: str) -> dict:
        """Check if output matches expected format.

        Returns {"valid": bool, "errors": list[str], "repaired": str | None}
        """
        if not self._enabled or task_type == "free_form":
            return {"valid": True, "errors": [], "repaired": None}

        errors = []
        if task_type == "json" and self._json_validation:
            errors = self._validate_json(output)
        elif task_type in ("code_python", "code_bash") and self._code_syntax_check:
            errors = self._validate_code(output, task_type)

        repaired = None
        if errors:
            repaired = self.repair(output, task_type)

        return {"valid": not errors, "errors": errors, "repaired": repaired}

    def repair(self, output: str, task_type: str) -> str | None:
        """Attempt to fix common format errors.

        JSON: missing closing brace, trailing comma, unescaped quotes.
        Code: strip markdown fences.
        Returns repaired string or None.
        """
        if task_type == "json":
            return self._repair_json(output)
        if task_type in ("code_python", "code_bash"):
            return self._repair_code(output)
        return None

    def _validate_json(self, text: str) -> list[str]:
        text = text.strip()
        text = strip_fences(text)
        try:
            json.loads(text)
            return []
        except json.JSONDecodeError as e:
            return [str(e)]

    def _validate_code(self, text: str, task_type: str) -> list[str]:
        code = strip_fences(text.strip())
        if task_type == "code_python":
            try:
                ast.parse(code)
                return []
            except SyntaxError as e:
                return [str(e)]
        # Bash: no cheap static check available; just confirm non-empty
        if not code.strip():
            return ["Empty code output"]
        return []

    def _repair_json(self, text: str) -> str | None:
        text = strip_fences(text.strip())
        # Trailing comma before closing brace/bracket
        text = re.sub(r",\s*([}\]])", r"\1", text)
        # Try parsing after comma fix first
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
        # Add missing closing braces — count outside quoted strings only
        opens, brackets = _count_braces_outside_strings(text)
        text = text + "}" * max(0, opens) + "]" * max(0, brackets)
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            return None

    def _repair_code(self, text: str) -> str | None:
        code = strip_fences(text.strip())
        if code:
            return code
        return None


def _count_braces_outside_strings(text: str) -> tuple[int, int]:
    """Count brace/bracket imbalance, ignoring chars inside quoted strings."""
    curly = 0
    square = 0
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            curly += 1
        elif ch == "}":
            curly -= 1
        elif ch == "[":
            square += 1
        elif ch == "]":
            square -= 1
    return curly, square
