# tests/test_router.py
import json
import tempfile
from pathlib import Path
import pytest

@pytest.fixture
def training_file():
    """Create a small training file for testing."""
    data = [
        {"text": "write a function to sort a list", "label": "PRIMARY"},
        {"text": "debug this stack trace", "label": "PRIMARY"},
        {"text": "extract names from this text", "label": "SPECIALIST"},
        {"text": "classify as positive or negative", "label": "SPECIALIST"},
        {"text": "count words in this string", "label": "TOOL_ONLY"},
        {"text": "parse this date format", "label": "TOOL_ONLY"},
        {"text": "implement a binary search tree", "label": "PRIMARY"},
        {"text": "summarize this article", "label": "SPECIALIST"},
        {"text": "replace all foo with bar", "label": "TOOL_ONLY"},
        {"text": "explain quantum computing", "label": "PRIMARY"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        return f.name

def test_router_trains_and_classifies(training_file):
    """Router can train on data and classify new text."""
    from lore.router import Router

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        model_path = f.name

    metrics = Router.train(training_file, model_path)
    assert metrics["accuracy"] > 0.5
    assert "PRIMARY" in metrics["classes"]

    router = Router.load(model_path)
    route, confidence = router.classify("write a Python web scraper")
    assert route == "PRIMARY"
    assert confidence > 0.0

def test_router_confidence_gate(training_file):
    """Low confidence defaults to PRIMARY."""
    from lore.router import Router

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        model_path = f.name

    Router.train(training_file, model_path)
    router = Router.load(model_path, confidence_threshold=0.99)

    # With 0.99 threshold, almost everything should default to PRIMARY
    route, confidence = router.classify("ambiguous query that could be anything")
    assert route == "PRIMARY"  # gate kicks in

def test_router_unknown_text():
    """Router handles text unlike any training example."""
    from lore.router import Router
    # if model file doesn't exist, Router.load should handle gracefully
    router = Router.load("nonexistent.joblib")
    route, confidence = router.classify("something")
    assert route == "PRIMARY"  # fallback when no model
    assert confidence == 0.0
