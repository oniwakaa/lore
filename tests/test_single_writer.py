"""Tests for single-writer orchestration fixes (Task 5).

Covers specialist reload in finally block and cyclic plan rejection.
"""
from unittest.mock import MagicMock

import pytest


def test_specialist_reload_after_orchestration_failure():
    """Specialist is reloaded even when orchestration raises."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    server.is_model_running = MagicMock(return_value=False)
    router = MagicMock()
    router.classify = MagicMock(return_value=("PRIMARY", 0.9))
    orchestrator = Orchestrator(server, router, memory=None, config={})
    orchestrator._specialist_offloaded = True
    # Force _orchestrate to raise
    orchestrator._orchestrate = MagicMock(side_effect=RuntimeError("boom"))
    # _delegate_dispatch returns a valid result
    orchestrator._delegate_dispatch = MagicMock(return_value={
        "route": "PRIMARY", "confidence": 0.9, "model": "primary",
        "content": "ok", "success": True, "latency_ms": 10,
        "orchestrated": False, "subtasks_completed": 0,
    })
    result = orchestrator.process("complex query", repo_context={"task": "test"})
    # Specialist must be reloaded despite failure
    server.start_model.assert_any_call("specialist")


def test_cycle_rejection_falls_back_to_direct():
    """Cyclic dependency plan falls back to direct dispatch."""
    from lore.orchestrator import Orchestrator
    from lore.decomposer import TaskPlan, SubTask

    server = MagicMock()
    router = MagicMock()
    router.classify = MagicMock(return_value=("PRIMARY", 0.9))
    orchestrator = Orchestrator(server, router, memory=None, config={})
    # Create a cyclic plan
    plan = TaskPlan(
        original_query="test",
        subtasks=[
            SubTask(id="a", description="task a", model="primary",
                    context_budget=2048, system_prompt="sp", dependencies=["b"]),
            SubTask(id="b", description="task b", model="primary",
                    context_budget=2048, system_prompt="sp", dependencies=["a"]),
        ],
    )
    waves = orchestrator._build_waves(plan)
    assert waves is None, "Cyclic plan should return None waves"


def test_session_id_rejects_traversal():
    """SessionManager rejects unsafe session IDs."""
    import tempfile

    from lore.session import SessionManager

    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager({"save_dir": tmp})
        bad_ids = ["..", ".", "", "a/b", "a\\b", "/etc/passwd", "foo/../bar"]
        for bad_id in bad_ids:
            with pytest.raises(ValueError):
                sm.save_session(bad_id, server=None, context=None)

