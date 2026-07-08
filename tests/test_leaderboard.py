"""Tests for the leaderboard scanner. Mock HF API, no network calls."""
import time
from unittest.mock import MagicMock, patch

import pytest

from lore.leaderboard import (
    LeaderboardScanner,
    ModelCandidate,
    UpgradeCandidate,
    TASK_BENCHMARKS,
    MAX_PARAMS_B,
    PREFERRED_QUANTS,
)


# ─── Task Score Computation ──────────────────────────────────────────────────

def test_compute_task_score_code_gen():
    """Code gen score weights SWE-bench, HumanEval, MBPP, MATH."""
    scanner = LeaderboardScanner()
    scores = {"SWE-bench": 70.0, "HumanEval": 80.0, "MBPP": 75.0, "MATH-Lvl5": 60.0}
    result = scanner.compute_task_score(scores, "code_gen")
    expected = 70 * 0.4 + 80 * 0.3 + 75 * 0.2 + 60 * 0.1
    assert abs(result - expected) < 0.01


def test_compute_task_score_missing_benchmarks():
    """Missing benchmarks only use available ones for weighting."""
    scanner = LeaderboardScanner()
    scores = {"HumanEval": 80.0}
    result = scanner.compute_task_score(scores, "code_gen")
    # Only HumanEval (weight 0.3), so score = 80.0
    assert result == 80.0


def test_compute_task_score_empty_scores():
    """Empty scores → 0.0."""
    scanner = LeaderboardScanner()
    assert scanner.compute_task_score({}, "code_gen") == 0.0


def test_compute_task_score_all_task_types():
    """All task types produce a score with full benchmarks."""
    scanner = LeaderboardScanner()
    scores = {"IFEval": 75.0, "MMLU-Pro": 70.0, "BBH": 65.0,
              "SWE-bench": 60.0, "HumanEval": 80.0, "MBPP": 70.0,
              "MATH-Lvl5": 55.0, "GSM8K": 65.0}
    for task_type in TASK_BENCHMARKS:
        result = scanner.compute_task_score(scores, task_type)
        assert result > 0


# ─── Benchmark Normalization ─────────────────────────────────────────────────

def test_normalize_benchmark_known():
    scanner = LeaderboardScanner()
    assert scanner._normalize_benchmark("google/IFEval") == "IFEval"
    assert scanner._normalize_benchmark("TIGER-Lab/MMLU-Pro") == "MMLU-Pro"
    assert scanner._normalize_benchmark("openai/openai_humaneval") == "HumanEval"
    assert scanner._normalize_benchmark("openai/gsm8k") == "GSM8K"
    assert scanner._normalize_benchmark("SWE-bench/SWE-bench_Verified") == "SWE-bench"


def test_normalize_benchmark_unknown():
    scanner = LeaderboardScanner()
    assert scanner._normalize_benchmark("some/unknown/benchmark") is None


# ─── Filter Viable ───────────────────────────────────────────────────────────

def test_filter_viable_excludes_oversized():
    """Models over MAX_PARAMS_B are excluded."""
    scanner = LeaderboardScanner()
    candidates = [
        ModelCandidate(model_id="A/Small", params_b=3.0, scores={"IFEval": 70}),
        ModelCandidate(model_id="B/Big", params_b=15.0, scores={"IFEval": 80}),
    ]
    with patch.object(scanner, "_find_gguf", return_value=("A/Small-GGUF", "Q4_K_M", 2.0)):
        viable = scanner._filter_viable(candidates, "/nonexistent/models")
    ids = [c.model_id for c in viable]
    assert "A/Small" in ids
    assert "B/Big" not in ids


def test_filter_viable_includes_unknown_size():
    """Models with unknown params_b (0.0) are included."""
    scanner = LeaderboardScanner()
    candidates = [
        ModelCandidate(model_id="C/Unknown", params_b=0.0, scores={"IFEval": 70}),
    ]
    with patch.object(scanner, "_find_gguf", return_value=("C/Unknown-GGUF", "Q4_K_M", 2.0)):
        viable = scanner._filter_viable(candidates, "/nonexistent/models")
    assert any(c.model_id == "C/Unknown" for c in viable)


# ─── Upgrade Detection ───────────────────────────────────────────────────────

def test_scan_for_upgrades_finds_better_model():
    """Scanner detects a model with higher task score than installed."""
    scanner = LeaderboardScanner()

    # Mock leaderboard data
    installed = ModelCandidate(
        model_id="Old/Model-7B", params_b=7.0,
        scores={"SWE-bench": 40.0, "HumanEval": 50.0, "MBPP": 45.0, "MATH-Lvl5": 30.0},
    )
    better = ModelCandidate(
        model_id="New/Model-9B", params_b=9.0,
        scores={"SWE-bench": 70.0, "HumanEval": 80.0, "MBPP": 75.0, "MATH-Lvl5": 60.0},
        gguf_repo="New/Model-9B-GGUF", gguf_quant="Q4_K_M", gguf_size_gb=5.0,
    )

    with patch.object(scanner, "_load_leaderboard_data", return_value=[installed, better]):
        with patch.object(scanner, "_filter_viable", return_value=[installed, better]):
            with patch.object(scanner, "_score_installed_model", return_value=40.0):
                upgrades = scanner.scan_for_upgrades(
                    {"code_gen": "Old/Model-7B"},
                    models_dir="/nonexistent",
                    min_improvement_pct=5.0,
                )

    assert len(upgrades) >= 1
    assert upgrades[0].better_model.model_id == "New/Model-9B"
    assert upgrades[0].improvement_pct > 5.0


def test_scan_for_upgrades_filters_small_improvement():
    """Upgrades below min_improvement_pct are not included."""
    scanner = LeaderboardScanner()

    installed = ModelCandidate(
        model_id="Old/Model-7B", params_b=7.0,
        scores={"IFEval": 70.0},
    )
    slightly_better = ModelCandidate(
        model_id="New/SlightlyBetter", params_b=7.0,
        scores={"IFEval": 72.0},  # ~2.8% improvement
        gguf_repo="New/SlightlyBetter-GGUF", gguf_quant="Q4_K_M",
    )

    with patch.object(scanner, "_load_leaderboard_data", return_value=[installed, slightly_better]):
        with patch.object(scanner, "_filter_viable", return_value=[installed, slightly_better]):
            with patch.object(scanner, "_score_installed_model", return_value=70.0):
                upgrades = scanner.scan_for_upgrades(
                    {"classification": "Old/Model-7B"},
                    min_improvement_pct=5.0,
                )

    assert len(upgrades) == 0  # 2.8% < 5% threshold


def test_scan_for_upgrades_skips_installed_models():
    """Installed models are not flagged as upgrades."""
    scanner = LeaderboardScanner()

    installed = ModelCandidate(
        model_id="Installed/Model", params_b=7.0,
        scores={"IFEval": 70.0}, is_installed=True,
    )

    with patch.object(scanner, "_load_leaderboard_data", return_value=[installed]):
        with patch.object(scanner, "_filter_viable", return_value=[installed]):
            with patch.object(scanner, "_score_installed_model", return_value=70.0):
                upgrades = scanner.scan_for_upgrades(
                    {"classification": "Installed/Model"},
                )

    assert len(upgrades) == 0


def test_scan_for_upgrades_keeps_multi_task_upgrades():
    """Same model best for multiple tasks → all tasks kept (no dedup drop)."""
    scanner = LeaderboardScanner()

    installed1 = ModelCandidate(model_id="Old/A", params_b=7.0, scores={"IFEval": 50.0, "BBH": 50.0})
    better = ModelCandidate(
        model_id="New/B", params_b=7.0,
        scores={"IFEval": 80.0, "BBH": 80.0},
        gguf_repo="New/B-GGUF", gguf_quant="Q4_K_M",
    )

    with patch.object(scanner, "_load_leaderboard_data", return_value=[installed1, better]):
        with patch.object(scanner, "_filter_viable", return_value=[installed1, better]):
            with patch.object(scanner, "_score_installed_model", return_value=50.0):
                upgrades = scanner.scan_for_upgrades(
                    {"classification": "Old/A", "summarization": "Old/A"},
                )

    # Same model is better for both tasks — both should appear
    task_types = {u.task_type for u in upgrades}
    assert "classification" in task_types
    assert "summarization" in task_types
    assert all(u.better_model.model_id == "New/B" for u in upgrades)


def test_scan_for_upgrades_empty_leaderboard():
    """Empty leaderboard data → no upgrades."""
    scanner = LeaderboardScanner()
    with patch.object(scanner, "_load_leaderboard_data", return_value=[]):
        upgrades = scanner.scan_for_upgrades({"code_gen": "Some/Model"})
    assert upgrades == []


def test_scan_for_upgrades_sorted_by_improvement():
    """Upgrades sorted biggest improvement first."""
    scanner = LeaderboardScanner()

    installed = ModelCandidate(model_id="Old/Base", params_b=7.0, scores={"IFEval": 50.0})
    med_upgrade = ModelCandidate(
        model_id="Med/Upgrade", params_b=7.0,
        scores={"IFEval": 60.0},
        gguf_repo="Med/Upgrade-GGUF", gguf_quant="Q4_K_M",
    )
    big_upgrade = ModelCandidate(
        model_id="Big/Upgrade", params_b=7.0,
        scores={"IFEval": 80.0},
        gguf_repo="Big/Upgrade-GGUF", gguf_quant="Q4_K_M",
    )

    with patch.object(scanner, "_load_leaderboard_data", return_value=[installed, med_upgrade, big_upgrade]):
        with patch.object(scanner, "_filter_viable", return_value=[installed, med_upgrade, big_upgrade]):
            with patch.object(scanner, "_score_installed_model", return_value=50.0):
                upgrades = scanner.scan_for_upgrades(
                    {"classification": "Old/Base", "extraction": "Old/Base"},
                )

    assert len(upgrades) >= 1
    # Biggest improvement first
    assert upgrades[0].improvement_pct >= upgrades[-1].improvement_pct


# ─── Get Model Scores ────────────────────────────────────────────────────────

def test_get_model_scores_returns_scores():
    """get_model_scores fetches and normalizes scores from HF."""
    scanner = LeaderboardScanner()

    mock_api = MagicMock()
    mock_info = MagicMock()
    mock_result = MagicMock()
    mock_result.dataset_id = "google/IFEval"
    mock_result.value = 75.5
    mock_info.eval_results = [mock_result]
    mock_api.model_info.return_value = mock_info

    scanner._api = mock_api
    scores = scanner.get_model_scores("Some/Model")
    assert scores == {"IFEval": 75.5}


def test_get_model_scores_handles_error():
    """get_model_scores returns empty dict on error."""
    scanner = LeaderboardScanner()
    mock_api = MagicMock()
    mock_api.model_info.side_effect = Exception("network error")
    scanner._api = mock_api
    scores = scanner.get_model_scores("Some/Model")
    assert scores == {}


def test_get_model_scores_caches_results():
    """Second call for same model_id doesn't hit HF API."""
    scanner = LeaderboardScanner()

    mock_api = MagicMock()
    mock_info = MagicMock()
    mock_result = MagicMock()
    mock_result.dataset_id = "google/IFEval"
    mock_result.value = 75.5
    mock_info.eval_results = [mock_result]
    mock_api.model_info.return_value = mock_info

    scanner._api = mock_api
    # First call hits API
    scores1 = scanner.get_model_scores("Cached/Model")
    assert scores1 == {"IFEval": 75.5}
    assert mock_api.model_info.call_count == 1
    # Second call should use cache, not hit API
    scores2 = scanner.get_model_scores("Cached/Model")
    assert scores2 == {"IFEval": 75.5}
    assert mock_api.model_info.call_count == 1  # still 1, no second call


# ─── Parquet Cache ───────────────────────────────────────────────────────────

def test_parquet_cache_avoids_reload():
    """Cache prevents redundant parquet loads within TTL."""
    scanner = LeaderboardScanner({"cache_ttl_hours": 1})
    # Pre-populate cache
    cached = [ModelCandidate(model_id="Cached/Model", params_b=7.0, scores={"IFEval": 70})]
    scanner._parquet_cache = cached
    scanner._parquet_cache_time = time.time()

    result = scanner._load_leaderboard_data()
    assert result is cached


def test_parquet_cache_expires():
    """Cache expires after TTL."""
    scanner = LeaderboardScanner({"cache_ttl_hours": 1})
    cached = [ModelCandidate(model_id="Cached/Model", params_b=7.0, scores={"IFEval": 70})]
    scanner._parquet_cache = cached
    scanner._parquet_cache_time = time.time() - 7200  # 2 hours ago, TTL is 1 hour

    with patch("lore.leaderboard.pd", create=True) as mock_pd:
        # Will try to reload
        mock_pd.read_parquet.side_effect = Exception("no pandas")
        with patch.object(scanner, "_load_from_individual_leaderboards", return_value=[]):
            result = scanner._load_leaderboard_data()
    # Cache expired, tried to reload, fell back to empty
    assert result == []


# ─── Dataclass Tests ─────────────────────────────────────────────────────────

def test_model_candidate_defaults():
    c = ModelCandidate(model_id="Test/Model", params_b=5.0)
    assert c.scores == {}
    assert c.task_scores == {}
    assert c.gguf_repo == ""
    assert c.is_installed is False


def test_upgrade_candidate_fields():
    c = ModelCandidate(model_id="New/Model", params_b=5.0, scores={"IFEval": 80})
    u = UpgradeCandidate(
        task_type="classification",
        current_model="Old/Model",
        current_score=70.0,
        better_model=c,
        better_score=80.0,
        improvement_pct=14.28,
    )
    assert u.task_type == "classification"
    assert u.better_model.model_id == "New/Model"
