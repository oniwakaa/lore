"""Pre-built prompt templates for common subtask types.

The decomposer can reference these, or the orchestrator can use them as defaults.
"""


TEMPLATES = {
    "planning": (
        "You are a task planner. Break the request into clear, actionable steps. "
        "Be concise. Specify which model should handle each step."
    ),
    "implementation": (
        "You are a skilled programmer. Write clean, correct code. "
        "Include type hints. Follow the specifications exactly. "
        "Do not add unnecessary abstractions."
    ),
    "review": (
        "You are a code reviewer. Check for bugs, style issues, missing edge cases. "
        "Be specific and constructive. Cite line numbers when relevant."
    ),
    "testing": (
        "You are a test engineer. Write comprehensive tests. "
        "Cover happy path, edge cases, and error conditions. "
        "Use pytest style. Include imports."
    ),
    "extraction": (
        "You extract structured information from text. Be precise. "
        "Output only the requested format, nothing else."
    ),
    "summarization": (
        "You summarize text concisely. Focus on key points, decisions, "
        "and actionable items. 2-4 sentences max."
    ),
    "documentation": (
        "You write clear documentation. Include examples. "
        "Be thorough but not verbose. Markdown format."
    ),
    "analysis": (
        "You analyze code or systems. Identify patterns, risks, and opportunities. "
        "Be specific and evidence-based."
    ),
    "aggregation": (
        "You are combining multiple subtask outputs into a final unified response. "
        "Synthesize the results coherently. Do not just concatenate — integrate. "
        "Remove redundancy. Present a clean, complete answer to the original task."
    ),
}

# Map task types to relevant template sets
_TASK_TYPE_MAP = {
    "coding": {
        "implementation": "implementation",
        "testing": "testing",
        "review": "review",
        "documentation": "documentation",
    },
    "analysis": {
        "planning": "planning",
        "analysis": "analysis",
        "summarization": "summarization",
    },
    "extraction": {
        "extraction": "extraction",
        "summarization": "summarization",
    },
    "general": {
        "planning": "planning",
        "implementation": "implementation",
        "review": "review",
        "testing": "testing",
        "documentation": "documentation",
        "extraction": "extraction",
        "summarization": "summarization",
    },
}


def get_template(name: str) -> str:
    """Get a prompt template by name."""
    return TEMPLATES.get(name, TEMPLATES["implementation"])


def get_templates_for_task(task_type: str) -> dict[str, str]:
    """Get a set of templates appropriate for a task type.

    task_type: "coding", "analysis", "extraction", "general"
    Returns dict of {subtask_role: template_text}
    """
    roles = _TASK_TYPE_MAP.get(task_type, _TASK_TYPE_MAP["general"])
    return {role: TEMPLATES[tpl_name] for role, tpl_name in roles.items()}
