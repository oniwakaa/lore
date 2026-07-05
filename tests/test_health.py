# tests/test_health.py
import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_health_ok_when_context_small():
    """Health report is 'ok' when context utilization is low."""
    from lore.health import ContextHealth
    health = ContextHealth({"warn_threshold": 0.80, "critical_threshold": 0.90})
    context = [{"role": "user", "content": "hello"}]
    report = health.check(context, token_usage=100, budget=4096)
    assert report.action == "ok"
    assert report.context_utilization == pytest.approx(100 / 4096)


def test_health_warn_at_80_percent():
    """Health report warns when utilization crosses warn_threshold."""
    from lore.health import ContextHealth
    health = ContextHealth({"warn_threshold": 0.80, "critical_threshold": 0.90})
    context = [{"role": "user", "content": "x"}] * 20
    report = health.check(context, token_usage=3300, budget=4096)
    assert report.context_utilization > 0.80
    assert report.action in ("compress", "warn_degradation")


def test_health_summarize_at_critical_with_stale():
    """At critical utilization with high staleness, action is 'summarize'."""
    from lore.health import ContextHealth
    health = ContextHealth({"warn_threshold": 0.80, "critical_threshold": 0.90,
                            "stale_after_turns": 5})
    # 30 messages = 15 turns, stale_after=5 → 20 stale messages
    context = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
    report = health.check(context, token_usage=3800, budget=4096)
    assert report.context_utilization > 0.90
    assert report.stale_context_ratio > 0.5
    assert report.action == "summarize"


def test_health_prune_at_critical_without_stale():
    """At critical utilization without much staleness, action is 'prune'."""
    from lore.health import ContextHealth
    health = ContextHealth({"warn_threshold": 0.80, "critical_threshold": 0.90,
                            "stale_after_turns": 100})
    context = [{"role": "user", "content": f"unique {i}"} for i in range(10)]
    report = health.check(context, token_usage=3800, budget=4096)
    assert report.context_utilization > 0.90
    assert report.stale_context_ratio == 0.0
    assert report.action == "prune"


def test_health_repetition_detection():
    """High repetition score triggers warn_degradation."""
    from lore.health import ContextHealth
    health = ContextHealth({"warn_threshold": 0.80, "critical_threshold": 0.90})
    # Lots of near-duplicate messages
    context = [{"role": "user", "content": "hello world foo bar"}] * 10
    report = health.check(context, token_usage=100, budget=4096)
    assert report.repetition_score > 0.5
    assert "warn_degradation" in report.action or report.action == "warn_degradation"


def test_health_stale_context_ratio():
    """Stale context ratio is computed correctly."""
    from lore.health import ContextHealth
    health = ContextHealth({"stale_after_turns": 5})
    # 30 messages = 15 turns. stale_after=5 turns = 10 messages.
    # stale = 30 - 10 = 20. ratio = 20/30 = 0.667
    context = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
    report = health.check(context, token_usage=1000, budget=4096)
    assert report.stale_context_ratio == pytest.approx(20 / 30, rel=0.01)


def test_health_compression_effectiveness():
    """record_compression tracks reduction ratio."""
    from lore.health import ContextHealth
    health = ContextHealth()
    health.record_compression(before_tokens=1000, after_tokens=400)
    # 60% reduction
    assert health._last_compression_ratio == pytest.approx(0.6)


def test_health_logs_to_jsonl(tmp_path):
    """Health metrics are logged to logs/context_health.jsonl."""
    from lore.health import ContextHealth, _HEALTH_LOG
    with patch("lore.health._HEALTH_LOG", tmp_path / "context_health.jsonl"):
        health = ContextHealth()
        context = [{"role": "user", "content": "test"}]
        health.check(context, token_usage=100, budget=4096)
        log_path = tmp_path / "context_health.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert "context_utilization" in entry
        assert "action" in entry


def test_health_empty_context():
    """Empty context returns a default report."""
    from lore.health import ContextHealth
    health = ContextHealth()
    report = health.check([], token_usage=0, budget=4096)
    assert report.action == "ok"
    assert report.context_utilization == 0.0


def test_health_should_check_every_n_turns():
    """should_check() returns True every N turns."""
    from lore.health import ContextHealth
    health = ContextHealth({"check_every_n_turns": 3})
    results = [health.should_check() for _ in range(7)]
    # Turn 3 and 6 should be True (1-indexed)
    assert results == [False, False, True, False, False, True, False]


def test_health_warnings_generated():
    """Warnings are generated for critical utilization."""
    from lore.health import ContextHealth
    health = ContextHealth({"warn_threshold": 0.80, "critical_threshold": 0.90})
    context = [{"role": "user", "content": "x"}] * 20
    report = health.check(context, token_usage=3800, budget=4096)
    assert len(report.warnings) > 0
    assert any("critical" in w for w in report.warnings)
