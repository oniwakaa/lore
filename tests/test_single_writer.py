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


def test_model_startup_cleanup_on_failed_health(monkeypatch):
    """Failed health check leaves no tracked process or open log handle."""
    import tempfile

    from lore.models import ModelServer

    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "primary": {"path": tmp + "/fake.gguf", "port": 19999},
            "defaults": {"context_size": 1024},
        }
        # Create fake model file
        with open(tmp + "/fake.gguf", "w") as f:
            f.write("fake")
        server = ModelServer(cfg)
        # Mock Popen to return a fake process
        class FakeProc:
            pid = 99999
            poll = lambda self: None
            def terminate(self): pass
            def wait(self, timeout=None): pass
            def kill(self): pass
        monkeypatch.setattr("lore.models.subprocess.Popen", lambda *a, **kw: FakeProc())
        # Mock health_check to return False
        monkeypatch.setattr(server, "health_check", lambda port: False)
        try:
            server.start_model("primary")
            assert False, "Should have raised"
        except RuntimeError:
            pass
        assert "primary" not in server._processes
        assert "primary" not in server._log_files

