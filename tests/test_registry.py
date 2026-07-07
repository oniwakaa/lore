"""Tests for the model registry. Mock scanner, no HF calls."""
import json
from unittest.mock import MagicMock, patch

import pytest

from lore.registry import ModelRegistry, WorkerAssignment
from lore.leaderboard import ModelCandidate, UpgradeCandidate


def _mock_config(**overrides):
    """Build a minimal registry config."""
    cfg = {
        "orchestrator": {"model": "Ornith-1.0-9B"},
        "auto_select": True,
        "size_class": "medium",
        "task_mapping": {
            "classification": "specialist",
            "code_gen": "primary",
        },
        "leaderboard": {},
    }
    cfg.update(overrides)
    return cfg


# ─── Orchestrator Lock ───────────────────────────────────────────────────────

def test_orchestrator_model_locked():
    """Orchestrator model is user-set and never changes."""
    registry = ModelRegistry(_mock_config(), models_dir="/nonexistent")
    assert registry.orchestrator_model == "Ornith-1.0-9B"


def test_orchestrator_model_property_read_only():
    """orchestrator_model is a read-only property."""
    registry = ModelRegistry(_mock_config(), models_dir="/nonexistent")
    # No setter — attempting to set raises AttributeError
    with pytest.raises(AttributeError):
        registry.orchestrator_model = "OtherModel"


# ─── Auto-Select Workers ─────────────────────────────────────────────────────

def test_select_workers_picks_best_local():
    """Auto-select picks the best-scoring local model for each task."""
    registry = ModelRegistry(_mock_config(), models_dir="/nonexistent/models")

    # Mock scanner: two local models with different scores
    registry._scanner = MagicMock()
    registry._scanner.get_model_scores.side_effect = lambda mid: {
        "ModelA": {"IFEval": 80, "BBH": 70, "MMLU-Pro": 75},
        "ModelB": {"IFEval": 60, "BBH": 50, "MMLU-Pro": 55},
    }.get(mid, {})
    registry._scanner._compute_task_score.side_effect = lambda scores, task: sum(scores.values()) / len(scores)

    registry._scan_local_models = MagicMock(return_value={
        "ModelA": "/models/modelA.gguf",
        "ModelB": "/models/modelB.gguf",
    })

    assignments = registry.select_workers()

    # ModelA should be selected for all task types (higher scores)
    for task_type, assignment in assignments.items():
        assert assignment.model_id == "ModelA"
        assert assignment.auto_selected is True
        assert assignment.benchmark_score > 0


def test_select_workers_falls_back_when_no_scores():
    """No scores available → falls back to config task_mapping."""
    registry = ModelRegistry(_mock_config(), models_dir="/nonexistent/models")

    registry._scanner = MagicMock()
    registry._scanner.get_model_scores.return_value = {}
    registry._scan_local_models = MagicMock(return_value={})

    assignments = registry.select_workers()

    # Should use fallback mapping
    assert assignments["classification"].model_id == "specialist"
    assert assignments["classification"].auto_selected is False


def test_select_workers_falls_back_when_no_local_models():
    """No local models → all tasks use fallback."""
    registry = ModelRegistry(_mock_config(), models_dir="/nonexistent/models")

    registry._scanner = MagicMock()
    registry._scan_local_models = MagicMock(return_value={})

    assignments = registry.select_workers()

    for task_type, assignment in assignments.items():
        assert assignment.auto_selected is False


# ─── Check For Upgrades ──────────────────────────────────────────────────────

def test_check_for_upgrades_delegates_to_scanner():
    """check_for_upgrades calls scanner.scan_for_upgrades with installed models."""
    registry = ModelRegistry(_mock_config(), models_dir="/nonexistent")
    registry._scanner = MagicMock()
    registry._scanner.scan_for_upgrades.return_value = []

    # Set up some assignments
    registry._assignments = {
        "code_gen": WorkerAssignment("code_gen", "OldModel", "/path", 50.0, True),
    }

    upgrades = registry.check_for_upgrades()
    registry._scanner.scan_for_upgrades.assert_called_once()
    call_args = registry._scanner.scan_for_upgrades.call_args
    assert call_args[0][0] == {"code_gen": "OldModel"}


def test_check_for_upgrades_uses_fallback_when_no_assignments():
    """No assignments yet → uses task_mapping as current models."""
    registry = ModelRegistry(_mock_config(), models_dir="/nonexistent")
    registry._scanner = MagicMock()
    registry._scanner.scan_for_upgrades.return_value = []

    upgrades = registry.check_for_upgrades()
    call_args = registry._scanner.scan_for_upgrades.call_args
    # Should pass fallback mapping as installed
    installed = call_args[0][0]
    assert "classification" in installed
    assert "code_gen" in installed


# ─── Approve Upgrade ─────────────────────────────────────────────────────────

def test_approve_upgrade_downloads_and_updates():
    """approve_upgrade downloads GGUF and updates the assignment."""
    registry = ModelRegistry(_mock_config(), models_dir="/tmp/test_models")

    candidate = ModelCandidate(
        model_id="New/Model-9B", params_b=9.0,
        scores={"SWE-bench": 70, "HumanEval": 80},
        gguf_repo="New/Model-9B-GGUF", gguf_quant="Q4_K_M", gguf_size_gb=5.0,
    )
    upgrade = UpgradeCandidate(
        task_type="code_gen", current_model="Old/Model",
        current_score=50.0, better_model=candidate,
        better_score=75.0, improvement_pct=50.0,
    )

    with patch.object(registry, "_download_gguf", return_value=True):
        with patch.object(registry, "_find_local_gguf", return_value="/tmp/test_models/new-model.gguf"):
            result = registry.approve_upgrade(upgrade)

    assert result is True
    assert registry._assignments["code_gen"].model_id == "New/Model-9B"
    assert registry._assignments["code_gen"].benchmark_score == 75.0
    assert registry._assignments["code_gen"].auto_selected is True


def test_approve_upgrade_fails_no_gguf_repo():
    """approve_upgrade returns False when no GGUF repo."""
    registry = ModelRegistry(_mock_config(), models_dir="/nonexistent")

    candidate = ModelCandidate(model_id="NoGGUF/Model", params_b=5.0, gguf_repo="")
    upgrade = UpgradeCandidate(
        task_type="code_gen", current_model="Old",
        current_score=50.0, better_model=candidate,
        better_score=70.0, improvement_pct=40.0,
    )

    result = registry.approve_upgrade(upgrade)
    assert result is False


def test_approve_upgrade_fails_on_download_error():
    """approve_upgrade returns False when download fails."""
    registry = ModelRegistry(_mock_config(), models_dir="/nonexistent")

    candidate = ModelCandidate(
        model_id="Fail/Model", params_b=5.0,
        gguf_repo="Fail/Model-GGUF", gguf_quant="Q4_K_M",
    )
    upgrade = UpgradeCandidate(
        task_type="code_gen", current_model="Old",
        current_score=50.0, better_model=candidate,
        better_score=70.0, improvement_pct=40.0,
    )

    with patch.object(registry, "_download_gguf", return_value=False):
        result = registry.approve_upgrade(upgrade)

    assert result is False
    # Assignment NOT updated
    assert "code_gen" not in registry._assignments


# ─── Get Model For Task ──────────────────────────────────────────────────────

def test_get_model_for_task_with_assignment():
    """get_model_for_task returns assigned model."""
    registry = ModelRegistry(_mock_config(), models_dir="/nonexistent")
    registry._assignments = {
        "code_gen": WorkerAssignment("code_gen", "BestModel", "/path", 80.0, True),
    }
    assert registry.get_model_for_task("code_gen") == "BestModel"


def test_get_model_for_task_without_assignment():
    """get_model_for_task falls back to task_mapping."""
    registry = ModelRegistry(_mock_config(), models_dir="/nonexistent")
    assert registry.get_model_for_task("classification") == "specialist"


def test_get_model_for_task_unknown_task():
    """Unknown task falls back to 'specialist'."""
    registry = ModelRegistry(_mock_config(), models_dir="/nonexistent")
    assert registry.get_model_for_task("unknown_task") == "specialist"


# ─── Prompt Upgrade ──────────────────────────────────────────────────────────

def test_prompt_upgrade_empty_list():
    """No upgrades → empty list, no output."""
    registry = ModelRegistry(_mock_config(), models_dir="/nonexistent")
    result = registry.prompt_upgrade([])
    assert result == []


def test_prompt_upgrade_headless_no_input():
    """Headless mode (EOFError) → no downloads approved."""
    registry = ModelRegistry(_mock_config(), models_dir="/nonexistent")
    candidate = ModelCandidate(model_id="New/Model", params_b=5.0)
    upgrade = UpgradeCandidate(
        task_type="code_gen", current_model="Old",
        current_score=50.0, better_model=candidate,
        better_score=70.0, improvement_pct=40.0,
    )

    with patch("builtins.input", side_effect=EOFError):
        result = registry.prompt_upgrade([upgrade])

    assert result == []


# ─── Scan Local Models ───────────────────────────────────────────────────────

def test_scan_local_models_strips_quant_suffix():
    """_scan_local_models strips quant suffixes from filenames."""
    import tempfile
    import os
    from pathlib import Path

    registry = ModelRegistry(_mock_config(), models_dir="/nonexistent")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create fake GGUF files
        Path(tmpdir, "Falcon-H1-1.5B-Instruct-Q4_K_M.gguf").touch()
        Path(tmpdir, "Some-Model-Q5_K_M.gguf").touch()

        registry._models_dir = Path(tmpdir)
        models = registry._scan_local_models()

        assert "Falcon-H1-1.5B-Instruct" in models
        assert "Some-Model" in models


def test_scan_local_models_empty_dir():
    """Empty models dir → empty dict."""
    registry = ModelRegistry(_mock_config(), models_dir="/nonexistent")
    models = registry._scan_local_models()
    assert models == {}
