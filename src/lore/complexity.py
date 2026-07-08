"""Heuristic task complexity estimator. No LLM call, <1ms."""
import re
from dataclasses import dataclass, field


@dataclass
class ComplexityEstimate:
    is_complex: bool          # True = needs decomposition
    confidence: float         # 0-1, how sure we are
    signals: list[str] = field(default_factory=list)
    estimated_subtasks: int = 2
    suggested_model: str = "primary"  # "primary" if complex (needs reasoning)


# --- Pattern regexes (compiled once at import) ---

# Multi-part request connectors
_MULTI_PART_RE = re.compile(
    r"\b(and then|also|additionally|after that|next,?|then\b|finally,?)\b",
    re.IGNORECASE,
)
# Numbered list in query (1. 2. etc. or 1) 2) etc.)
_NUMBERED_LIST_RE = re.compile(r"^\s*\d+[.)]|[,;]\s*\d+[.)]", re.MULTILINE)
# Complex action verbs
_COMPLEX_KW_RE = re.compile(
    r"\b(refactor|implement|build|create from scratch|design|plan|"
    r"architecture|migrate|debug and fix|review and fix)\b",
    re.IGNORECASE,
)
# Code + instruction pattern ("write X and test it", "implement X then document it")
_CODE_PLUS_RE = re.compile(
    r"(write|implement|create|build).+\b(and|then)\b.+(test|document|explain|review|deploy|refactor)",
    re.IGNORECASE,
)
# File path + action
_FILE_PATH_RE = re.compile(r"[/\\][\w./\\-]+\.\w{1,5}\b")
_FILE_ACTION_RE = re.compile(
    r"(in|inside|within|at)\s+[/\\][\w./\\-]+\.\w{1,5}\b.*\b(add|update|change|modify|fix|remove|delete|insert|append)\b",
    re.IGNORECASE,
)
# Multiple outputs requested
_MULTI_OUTPUT_RE = re.compile(
    r"\b(give me|provide|output|return|generate)\b.+\b(and|also|as well as)\b.+(test|doc|readme|example|usage|specification)\b",
    re.IGNORECASE,
)

# Simple signals
_SIMPLE_KW_RE = re.compile(
    r"\b(what is|explain|translate|convert|format|summarize|list)\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(r"\?\s*$")


def estimate(query: str, router_route: str = "PRIMARY") -> ComplexityEstimate:
    """Estimate task complexity. Returns in <1ms (no LLM call).

    Heuristic scoring:
    - 2+ complex signals → complex (high confidence)
    - 1 complex signal + no simple signals → uncertain → default simple
    - 1+ simple signals → simple
    - TOOL_ONLY route → always simple
    """
    if not query or not query.strip():
        return ComplexityEstimate(is_complex=False, confidence=1.0, signals=["empty query"])

    # TOOL_ONLY always simple — already handled by tool_handler
    if router_route == "TOOL_ONLY":
        return ComplexityEstimate(
            is_complex=False, confidence=1.0,
            signals=["TOOL_ONLY route"], estimated_subtasks=1,
        )

    complex_signals: list[str] = []
    simple_signals: list[str] = []

    # --- Complex signal detection ---

    multi_part = len(_MULTI_PART_RE.findall(query))
    if multi_part >= 1:
        complex_signals.append(f"multi-part connector (x{multi_part})")

    if len(query) > 500:
        complex_signals.append(f"long query ({len(query)} chars)")

    if _CODE_PLUS_RE.search(query):
        complex_signals.append("code + instruction pattern")

    if _COMPLEX_KW_RE.search(query):
        complex_signals.append(f"complex verb: {_COMPLEX_KW_RE.search(query).group(0)}")

    if _FILE_ACTION_RE.search(query):
        complex_signals.append("file path + action")

    if _MULTI_OUTPUT_RE.search(query):
        complex_signals.append("multiple outputs requested")

    numbered = _NUMBERED_LIST_RE.findall(query)
    if len(numbered) >= 2:
        complex_signals.append(f"numbered list ({len(numbered)} items)")

    # --- Simple signal detection ---

    if _SIMPLE_KW_RE.search(query) and len(query) < 200:
        simple_signals.append(f"simple keyword: {_SIMPLE_KW_RE.search(query).group(0)}")

    if _QUESTION_RE.search(query) and len(query) < 200:
        simple_signals.append("short question (<200 chars)")

    if len(query) < 100 and not complex_signals:
        simple_signals.append("very short query")

    # --- Classification ---

    if len(complex_signals) >= 2:
        estimated = min(3 + len(complex_signals), 5)
        return ComplexityEstimate(
            is_complex=True,
            confidence=min(0.6 + 0.1 * len(complex_signals), 0.95),
            signals=complex_signals,
            estimated_subtasks=estimated,
            suggested_model="primary",
        )

    if len(complex_signals) == 1 and not simple_signals:
        # Uncertain (one complex signal, no simple signals) → default simple
        # Orchestration overhead not worth it for ambiguous cases
        return ComplexityEstimate(
            is_complex=False,
            confidence=0.5,
            signals=complex_signals + simple_signals + ["uncertain → default simple"],
            estimated_subtasks=2,
            suggested_model="primary",
        )

    # Simple (0 complex signals, or 1 complex + simple signals)
    return ComplexityEstimate(
        is_complex=False,
        confidence=0.8 if simple_signals else 0.5,
        signals=simple_signals or ["no complex signals"],
        estimated_subtasks=1,
        suggested_model="primary" if router_route == "PRIMARY" else "specialist",
    )
