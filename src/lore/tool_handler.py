"""TOOL_ONLY fast-path. Regex/heuristic handlers, no model call."""
import json
import operator
import re
from datetime import date

_OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}

_UNIT_TO_KM = {"km": 1.0, "kilometer": 1.0, "kilometers": 1.0,
               "mi": 1.60934, "mile": 1.60934, "miles": 1.60934}
_UNIT_TO_KG = {"kg": 1.0, "kilogram": 1.0, "kilograms": 1.0,
               "lb": 0.453592, "lbs": 0.453592, "pound": 0.453592, "pounds": 0.453592}


def _fmt_number(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else f"{n:.4f}".rstrip("0").rstrip(".")


def _handle_math(match: re.Match) -> str:
    a, op, b = match.group(1), match.group(2), match.group(3)
    result = _OPS[op](float(a), float(b))
    return _fmt_number(result)


def _handle_date(match: re.Match) -> str:
    return date.today().isoformat()


def _handle_unit_conversion(match: re.Match) -> str | None:
    value, from_unit, to_unit = float(match.group(1)), match.group(2).lower(), match.group(3).lower()

    if from_unit in _UNIT_TO_KM and to_unit in _UNIT_TO_KM:
        km = value * _UNIT_TO_KM[from_unit]
        return f"{_fmt_number(km / _UNIT_TO_KM[to_unit])} {match.group(3)}"

    if from_unit in _UNIT_TO_KG and to_unit in _UNIT_TO_KG:
        kg = value * _UNIT_TO_KG[from_unit]
        return f"{_fmt_number(kg / _UNIT_TO_KG[to_unit])} {match.group(3)}"

    if {from_unit, to_unit} == {"c", "f"} or {from_unit, to_unit} == {"celsius", "fahrenheit"}:
        if from_unit in ("c", "celsius"):
            result = value * 9 / 5 + 32
        else:
            result = (value - 32) * 5 / 9
        return f"{_fmt_number(result)} {match.group(3)}"

    return None  # unit combo not supported, fall back to specialist


def _handle_json_format(match: re.Match) -> str:
    try:
        json.loads(match.group("payload"))
        return "Valid JSON"
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"


# Ordered (pattern, handler). First match wins.
_REGISTRY: list[tuple[re.Pattern, "callable"]] = [
    (re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*$"), _handle_math),
    (re.compile(r"(?i)\b(today'?s date|current date|what.*date is it|what day is (it|today))\b"), _handle_date),
    (re.compile(
        r"(?i)^\s*(-?\d+(?:\.\d+)?)\s*"
        r"(km|kilometers?|mi|miles?|kg|kilograms?|lbs?|pounds?|c|celsius|f|fahrenheit)\s*"
        r"(?:to|in)\s*"
        r"(km|kilometers?|mi|miles?|kg|kilograms?|lbs?|pounds?|c|celsius|f|fahrenheit)\s*$"
    ), _handle_unit_conversion),
    (re.compile(r"(?i)^is\s+this\s+(?:valid\s+)?json\s*:\s*(?P<payload>.+)$"), _handle_json_format),
]


def handle_tool_only(query: str) -> str | None:
    """Try each registered pattern. Returns result string, or None if no match."""
    for pattern, handler in _REGISTRY:
        match = pattern.search(query.strip())
        if match:
            result = handler(match)
            if result is not None:
                return result
    return None
