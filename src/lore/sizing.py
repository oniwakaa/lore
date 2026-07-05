"""Dynamic context sizing. Allocate more budget to harder tasks.

Heuristics on query content + route type. Called per-request in ContextManager
to override the fixed working_context budget.
"""
import logging
import re

logger = logging.getLogger(__name__)

# Complexity keywords → larger budget
_COMPLEX_KEYWORDS = re.compile(
    r"\b(refactor|debug|review|plan|architecture|migrate|implement|analyse|analyze|"
    r"rewrite|redesign|optimize|benchmark|profile|security|audit)\b",
    re.IGNORECASE,
)
_SIMPLE_KEYWORDS = re.compile(
    r"\b(explain|summarize|what is|define|describe|list|translate|convert)\b",
    re.IGNORECASE,
)
_CODE_BLOCK_RE = re.compile(r"```")
_FILE_PATH_RE = re.compile(r"[/\\][\w./\\-]+\.\w{1,5}\b")


def estimate_context_budget(route: str, query: str, config: dict) -> int:
    """Estimate token budget based on task complexity.

    Returns budget in tokens. Logs the sizing decision.
    """
    base = config.get("default_budget", 16384)
    min_budget = config.get("min_budget", 2048)
    max_budget = config.get("max_budget", 32768)

    # Route-based floor
    if route == "TOOL_ONLY":
        budget = min_budget
        reason = "TOOL_ONLY fast-path"
    elif route == "SPECIALIST":
        budget = min(min_budget * 2, base)  # 4096 default
        reason = "SPECIALIST: medium budget"
    elif route == "MULTIMODAL":
        budget = base
        reason = "MULTIMODAL: default budget"
    else:
        # PRIMARY: use query heuristics
        query_tokens = len(query.split())
        has_code_block = bool(_CODE_BLOCK_RE.search(query))
        has_file_path = bool(_FILE_PATH_RE.search(query))
        is_complex = bool(_COMPLEX_KEYWORDS.search(query))
        is_simple = bool(_SIMPLE_KEYWORDS.search(query)) and not is_complex

        if query_tokens > 500 or (has_code_block and has_file_path):
            budget = max_budget
            reason = "large query or code+filepath"
        elif has_code_block or has_file_path or is_complex:
            budget = max(base, 8192)
            reason = "code block / file path / complex keyword"
        elif is_simple:
            budget = max(min_budget * 2, 4096)
            reason = "simple query keyword"
        else:
            budget = base
            reason = "default PRIMARY budget"

    budget = max(min_budget, min(budget, max_budget))
    logger.debug(f"Context sizing: {route} → {budget} tokens ({reason})")
    return budget
