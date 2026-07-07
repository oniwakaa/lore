"""Tests for the model-based task classifier. Mock server, no real inference."""
import json
from unittest.mock import MagicMock

import pytest

from lore.classifier import TaskClassifier, ClassificationResult


def _mock_classification_response(is_complex=True, task_type="code_gen",
                                  estimated_subtasks=3, suggested_model="primary",
                                  hints=None):
    """Build a mock server.chat() return value for the classifier."""
    result_json = json.dumps({
        "is_complex": is_complex,
        "task_type": task_type,
        "estimated_subtasks": estimated_subtasks,
        "suggested_model": suggested_model,
        "hints": hints or {},
    })
    return {"choices": [{"message": {"content": result_json}}]}


# ─── Model-Based Classification ──────────────────────────────────────────────

def test_classifier_parses_valid_response():
    """Classifier parses valid JSON response from specialist."""
    server = MagicMock()
    server.chat.return_value = _mock_classification_response(
        is_complex=True, task_type="code_gen", estimated_subtasks=3,
        suggested_model="primary", hints={"needs_code": True},
    )

    classifier = TaskClassifier(server)
    result = classifier.classify("Write a CSV parser and tests")

    assert result.is_complex is True
    assert result.task_type == "code_gen"
    assert result.estimated_subtasks == 3
    assert result.suggested_model == "primary"
    assert result.source == "model"
    assert result.hints["needs_code"] is True


def test_classifier_simple_task():
    """Simple task classified as not complex."""
    server = MagicMock()
    server.chat.return_value = _mock_classification_response(
        is_complex=False, task_type="summarization",
        estimated_subtasks=1, suggested_model="specialist",
    )

    classifier = TaskClassifier(server)
    result = classifier.classify("Summarize this paragraph")

    assert result.is_complex is False
    assert result.task_type == "summarization"
    assert result.suggested_model == "specialist"


def test_classifier_invalid_task_type_defaults_to_planning():
    """Invalid task_type in response → defaults to 'planning'."""
    server = MagicMock()
    server.chat.return_value = _mock_classification_response(
        task_type="unknown_type", estimated_subtasks=2,
    )

    classifier = TaskClassifier(server)
    result = classifier.classify("some task")

    assert result.task_type == "planning"


def test_classifier_invalid_model_defaults_to_primary():
    """Invalid suggested_model → defaults to 'primary'."""
    server = MagicMock()
    server.chat.return_value = _mock_classification_response(
        suggested_model="mega_model",
    )

    classifier = TaskClassifier(server)
    result = classifier.classify("some task")

    assert result.suggested_model == "primary"


def test_classifier_clamps_estimated_subtasks():
    """estimated_subtasks clamped to 1-5 range."""
    server = MagicMock()
    server.chat.return_value = _mock_classification_response(
        estimated_subtasks=10,
    )

    classifier = TaskClassifier(server)
    result = classifier.classify("some task")

    assert result.estimated_subtasks == 5


def test_classifier_clamps_subtasks_minimum():
    """estimated_subtasks clamped to minimum 1."""
    server = MagicMock()
    server.chat.return_value = _mock_classification_response(
        estimated_subtasks=0,
    )

    classifier = TaskClassifier(server)
    result = classifier.classify("some task")

    assert result.estimated_subtasks == 1


# ─── Heuristic Fallback ──────────────────────────────────────────────────────

def test_classifier_falls_back_on_server_error():
    """Server error → heuristic fallback."""
    server = MagicMock()
    server.chat.side_effect = Exception("server down")

    classifier = TaskClassifier(server)
    result = classifier.classify("Write a parser and then test it", "PRIMARY")

    assert result.source == "heuristic"
    # This is a complex query (multi-part + code+instruction) per complexity.py
    assert result.is_complex is True


def test_classifier_falls_back_on_invalid_json():
    """Invalid JSON response → heuristic fallback."""
    server = MagicMock()
    server.chat.return_value = {"choices": [{"message": {"content": "not json"}}]}

    classifier = TaskClassifier(server)
    result = classifier.classify("What is 2+2?", "PRIMARY")

    assert result.source == "heuristic"
    assert result.is_complex is False


def test_classifier_falls_back_on_partial_json():
    """JSON with missing fields → still parsed with defaults."""
    server = MagicMock()
    server.chat.return_value = {
        "choices": [{"message": {"content": json.dumps({"is_complex": True})}}]
    }

    classifier = TaskClassifier(server)
    result = classifier.classify("some task")

    # Should parse with defaults for missing fields
    assert result.is_complex is True
    assert result.task_type == "planning"  # default
    assert result.suggested_model == "primary"  # default


def test_classifier_falls_back_on_json_in_text():
    """JSON embedded in text → extracted and parsed."""
    server = MagicMock()
    content = f"Here is the classification:\n```json\n{json.dumps({'is_complex': True, 'task_type': 'math', 'estimated_subtasks': 2, 'suggested_model': 'primary'})}\n```"
    server.chat.return_value = {"choices": [{"message": {"content": content}}]}

    classifier = TaskClassifier(server)
    result = classifier.classify("Solve this equation")

    assert result.is_complex is True
    assert result.task_type == "math"


# ─── TOOL_ONLY Route ─────────────────────────────────────────────────────────

def test_classifier_tool_only_skips_model():
    """TOOL_ONLY route → no model call, returns simple classification."""
    server = MagicMock()
    classifier = TaskClassifier(server)
    result = classifier.classify("2+2", "TOOL_ONLY")

    assert result.is_complex is False
    assert result.source == "heuristic"
    # Server should NOT have been called
    server.chat.assert_not_called()


# ─── Classification Result Dataclass ─────────────────────────────────────────

def test_classification_result_defaults():
    r = ClassificationResult(
        is_complex=True, task_type="code_gen",
        estimated_subtasks=3, suggested_model="primary",
        confidence=0.85,
    )
    assert r.hints == {}
    assert r.source == "model"
