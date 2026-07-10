"""Shared JSON parsing utilities for model outputs.

Consolidates fence stripping, JSON extraction, trailing-comma fixing,
and truncated-JSON repair that was independently implemented in
decomposer.py, verifier.py, and classifier.py.
"""
import json
import logging
import re

logger = logging.getLogger(__name__)


def strip_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```)."""
    match = re.match(r"^```[\w]*\n?(.*?)```\s*$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def extract_json_object(text: str) -> str | None:
    """Extract the first JSON object string from text. Returns None if not found."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    return match.group(0)


def repair_truncated_json(text: str) -> dict | None:
    """Repair truncated JSON by closing open strings/brackets/braces.

    When max_tokens cuts the model mid-generation, the JSON is incomplete.
    Counts open vs close brackets/braces (respecting strings), detects
    mid-string state, closes everything, then tries to parse.
    """
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

    if in_string:
        repaired = repaired.rstrip()
        repaired += '",'
    elif repaired and repaired[-1] not in ',{}["':
        repaired += ','

    repaired = re.sub(r",\s*,", ",", repaired)

    for closer in reversed(stack):
        repaired += closer

    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        logger.debug(f"Repair failed. Repaired text: {repaired[:300]}")
        return None


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before closing braces/brackets."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def parse_json_response(raw: str) -> dict | None:
    """Full pipeline: strip fences -> extract JSON -> fix commas -> parse -> repair if needed.

    Returns parsed dict or None if unparseable.
    """
    # Try direct parse first
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strip fences and retry
    stripped = strip_fences(raw)
    if stripped != raw:
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            pass

    # Extract JSON object from surrounding text
    candidate = extract_json_object(stripped)
    if candidate is None:
        # No complete { ... } found — try repairing the raw text directly
        return repair_truncated_json(stripped)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas
    cleaned = _fix_trailing_commas(candidate)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Repair truncated JSON
    return repair_truncated_json(cleaned)
