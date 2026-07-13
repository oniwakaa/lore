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


def test_classifier_not_on_critical_path():
    """Orchestrator created without classifier by default."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    router = MagicMock()
    router.classify = MagicMock(return_value=("PRIMARY", 0.9))
    orchestrator = Orchestrator(server, router, memory=None, config={})
    assert orchestrator._classifier is None, "Classifier should not be created by default"


def test_single_routing_decision():
    """Router.classify is called once per request, not twice."""
    from unittest.mock import MagicMock
    from lore.cli import _dispatch
    server = MagicMock()
    server.chat = MagicMock(return_value={
        "choices": [{"message": {"content": "test response"}}]
    })
    router = MagicMock()
    router.classify = MagicMock(return_value=("PRIMARY", 0.95))
    from lore.context import ContextManager
    ctx = ContextManager({"context_budget": 4096}, server, system_prompt="test")
    memory = MagicMock()
    memory.retrieve = MagicMock(return_value=[])
    memory.store = MagicMock()
    from lore.logging import RequestLogger
    req_logger = RequestLogger()
    result = _dispatch("hello", server, router, ctx, memory, req_logger)
    assert router.classify.call_count == 1, f"Expected 1 classify call, got {router.classify.call_count}"


def test_memory_retrieved_once():
    """Memory.retrieve is called once per model request, not twice."""
    from unittest.mock import MagicMock
    from lore.cli import _dispatch
    server = MagicMock()
    server.chat = MagicMock(return_value={
        "choices": [{"message": {"content": "test response"}}]
    })
    router = MagicMock()
    router.classify = MagicMock(return_value=("PRIMARY", 0.95))
    from lore.context import ContextManager
    ctx = ContextManager({"context_budget": 4096}, server, system_prompt="test")
    memory = MagicMock()
    memory.retrieve = MagicMock(return_value=[])
    memory.store = MagicMock()
    from lore.logging import RequestLogger
    req_logger = RequestLogger()
    _dispatch("hello", server, router, ctx, memory, req_logger)
    # build_prompt also calls memory.retrieve if ctx._memory is set, but we pass memory=None to ctx
    # So only _execute_query should call it
    assert memory.retrieve.call_count <= 1, f"Expected <=1 retrieve, got {memory.retrieve.call_count}"


def test_budget_no_drift():
    """Context budget does not drift after a small-budget request."""
    from unittest.mock import MagicMock
    from lore.cli import _dispatch
    from lore.context import ContextManager
    server = MagicMock()
    server.chat = MagicMock(return_value={
        "choices": [{"message": {"content": "response"}}]
    })
    router = MagicMock()
    router.classify = MagicMock(return_value=("PRIMARY", 0.9))
    ctx = ContextManager({"working_context": 8192}, server, system_prompt="test")
    original_budget = ctx._config.get("working_context", 8192)
    memory = MagicMock()
    memory.retrieve = MagicMock(return_value=[])
    memory.store = MagicMock()
    from lore.logging import RequestLogger
    req_logger = RequestLogger()
    _dispatch("hello", server, router, ctx, memory, req_logger)
    assert ctx._config.get("working_context", 0) == original_budget, "Budget drifted"


def test_api_forwards_controls():
    """API forwards max_tokens and temperature to dispatch."""
    from unittest.mock import MagicMock, patch
    from lore.api import LoreHandler, _app_state
    from io import BytesIO
    import json

    server = MagicMock()
    router = MagicMock()
    from lore.context import ContextManager
    ctx = ContextManager({"working_context": 4096}, server, system_prompt="test")
    memory = MagicMock()
    req_logger = MagicMock()
    verifier = MagicMock()

    from lore.orchestrator import Orchestrator
    orchestrator = Orchestrator(server, router, memory=None, config={})
    router.classify.return_value = ("PRIMARY", 0.9)

    _app_state.update({
        "server": server, "router": router, "ctx": ctx,
        "memory": memory, "req_logger": req_logger,
        "verifier": verifier, "orchestrator": orchestrator,
    })

    # Prepare POST request body
    request_body = json.dumps({
        "messages": [{"role": "user", "content": "what is 2+2?"}],
        "stream": False,
        "max_tokens": 1000,
        "temperature": 0.5,
    }).encode()

    mock_wfile = BytesIO()

    class TestHandler(LoreHandler):
        def __init__(self):
            self.path = "/v1/chat/completions"
            self.headers = {"Content-Length": str(len(request_body))}
            self.wfile = mock_wfile
            self.rfile = BytesIO(request_body)
            self._status = None
            self._headers = {}

        def send_response(self, status):
            self._status = status

        def send_header(self, key, value):
            self._headers[key] = value

        def end_headers(self):
            pass

        def log_message(self, *args):
            pass

    # Patch _dispatch in lore.cli
    with patch("lore.cli._dispatch") as mock_dispatch:
        mock_dispatch.return_value = {
            "route": "PRIMARY", "confidence": 0.9, "model": "primary",
            "content": "result", "success": True, "latency_ms": 10,
        }
        handler = TestHandler()
        handler.do_POST()

        # Check if mock_dispatch was called with max_tokens=1000 and temperature=0.5
        mock_dispatch.assert_called_once()
        kwargs = mock_dispatch.call_args[1]
        assert kwargs.get("max_tokens") == 1000
        assert kwargs.get("temperature") == 0.5







