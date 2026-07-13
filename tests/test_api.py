# tests/test_api.py
"""Tests for the OpenAI-compatible API server."""
import json
from unittest.mock import MagicMock, patch
from io import BytesIO

import pytest


def _make_request_body(messages=None, **kwargs):
    """Helper to build a chat completion request body."""
    body = {
        "messages": messages or [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    body.update(kwargs)
    return json.dumps(body).encode()


def _make_handler(path, request_body, mock_wfile=None):
    """Create a LoreHandler subclass instance for testing."""
    from lore.api import LoreHandler
    mock_wfile = mock_wfile or BytesIO()

    class TestHandler(LoreHandler):
        def __init__(self):
            self.path = path
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

    return TestHandler(), mock_wfile


def _setup_app_state(**overrides):
    """Set up minimal _app_state for API tests."""
    from lore.api import _app_state
    defaults = {
        "server": MagicMock(),
        "router": MagicMock(),
        "req_logger": MagicMock(),
        "cfg": MagicMock(),
    }
    defaults.update(overrides)
    _app_state.update(defaults)
    return defaults


# ─── Health and models endpoints ──────────────────────────────────────────────

def test_api_health_endpoint():
    """GET /health returns 200 with status ok."""
    handler, mock_wfile = _make_handler("/health", b"")
    handler.do_GET()
    assert handler._status == 200
    body = json.loads(mock_wfile.getvalue())
    assert body["status"] == "ok"


def test_api_models_endpoint():
    """GET /v1/models returns list of running models."""
    mock_server = MagicMock()
    mock_server.is_model_running.return_value = True
    mock_cfg = MagicMock()
    mock_cfg.models = {
        "primary": {"name": "Ornith-1.0-9B"},
        "specialist": {"name": "Falcon-H1-1.5B"},
    }
    _setup_app_state(server=mock_server, cfg=mock_cfg)

    handler, mock_wfile = _make_handler("/v1/models", b"")
    handler.do_GET()

    assert handler._status == 200
    body = json.loads(mock_wfile.getvalue())
    assert body["object"] == "list"
    # "lore" + primary + specialist = 3
    assert len(body["data"]) == 3
    ids = [m["id"] for m in body["data"]]
    assert "lore" in ids


# ─── Error handling ────────────────────────────────────────────────────────────

def test_api_chat_completions_empty_body():
    """POST with empty body returns 400."""
    _setup_app_state()
    handler, mock_wfile = _make_handler("/v1/chat/completions", b"")
    handler.do_POST()
    assert handler._status == 400
    body = json.loads(mock_wfile.getvalue())
    assert "error" in body


def test_api_chat_completions_no_messages():
    """POST with no messages returns 400."""
    _setup_app_state()
    request_body = json.dumps({"messages": []}).encode()
    handler, mock_wfile = _make_handler("/v1/chat/completions", request_body)
    handler.do_POST()
    assert handler._status == 400


def test_api_chat_completions_stream_supported():
    """POST with stream=true returns SSE-formatted response."""
    mock_server = MagicMock()
    mock_server.chat.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}]
    }
    mock_router = MagicMock()
    mock_router.classify.return_value = ("SPECIALIST", 0.88)
    _setup_app_state(server=mock_server, router=mock_router)

    request_body = _make_request_body(
        messages=[{"role": "user", "content": "say hello"}], stream=True)
    handler, mock_wfile = _make_handler("/v1/chat/completions", request_body)
    handler.do_POST()

    assert handler._status == 200
    raw = mock_wfile.getvalue().decode()
    assert "text/event-stream" in handler._headers.get("Content-Type", "")
    assert "data: " in raw
    assert "[DONE]" in raw
    assert "hello" in raw


# ─── Chat completions ──────────────────────────────────────────────────────────

def test_api_chat_completions_primary():
    """POST with a complex query routes to primary model."""
    mock_server = MagicMock()
    mock_server.chat.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "4"}, "finish_reason": "stop"}]
    }
    mock_router = MagicMock()
    mock_router.classify.return_value = ("PRIMARY", 0.95)
    _setup_app_state(server=mock_server, router=mock_router)

    request_body = _make_request_body(
        messages=[{"role": "user", "content": "write a function to compute fibonacci"}])
    handler, mock_wfile = _make_handler("/v1/chat/completions", request_body)
    handler.do_POST()

    assert handler._status == 200
    body = json.loads(mock_wfile.getvalue())
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "4"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["lore"]["route"] == "PRIMARY"
    # No tools → direct server.chat call
    mock_server.chat.assert_called_once()
    call_args = mock_server.chat.call_args
    assert call_args[0][0] == "primary"


def test_api_chat_completions_specialist():
    """POST with a simple query routes to specialist model (no auto-tools)."""
    mock_server = MagicMock()
    mock_server.chat.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "result"}, "finish_reason": "stop"}]
    }
    mock_router = MagicMock()
    mock_router.classify.return_value = ("SPECIALIST", 0.88)
    _setup_app_state(server=mock_server, router=mock_router)

    request_body = _make_request_body(
        messages=[{"role": "user", "content": "extract names from this text"}])
    handler, mock_wfile = _make_handler("/v1/chat/completions", request_body)
    handler.do_POST()

    assert handler._status == 200
    body = json.loads(mock_wfile.getvalue())
    assert body["lore"]["route"] == "SPECIALIST"
    # No tools auto-injected → direct server.chat call, no run_tool_loop
    mock_server.chat.assert_called_once()
    call_args = mock_server.chat.call_args
    assert call_args[0][0] == "specialist"
    # No tools passed
    assert "tools" not in call_args[1]


def test_api_chat_completions_tool_only():
    """POST with TOOL_ONLY query uses regex fast-path, no LLM call."""
    mock_server = MagicMock()
    mock_router = MagicMock()
    mock_router.classify.return_value = ("TOOL_ONLY", 0.9)
    _setup_app_state(server=mock_server, router=mock_router)

    request_body = _make_request_body(
        messages=[{"role": "user", "content": "2+2"}])
    handler, mock_wfile = _make_handler("/v1/chat/completions", request_body)
    handler.do_POST()

    assert handler._status == 200
    body = json.loads(mock_wfile.getvalue())
    assert body["choices"][0]["message"]["content"] == "4"
    assert body["lore"]["route"] == "TOOL_ONLY"
    mock_server.chat.assert_not_called()


def test_api_chat_completions_json_mode():
    """POST with response_format=json_object passes json_mode to model."""
    mock_server = MagicMock()
    mock_server.chat.return_value = {
        "choices": [{"message": {"role": "assistant", "content": '{"answer": 4}'}, "finish_reason": "stop"}]
    }
    mock_router = MagicMock()
    mock_router.classify.return_value = ("PRIMARY", 0.9)
    _setup_app_state(server=mock_server, router=mock_router)

    request_body = json.dumps({
        "messages": [{"role": "user", "content": "return json"}],
        "response_format": {"type": "json_object"},
        "stream": False,
    }).encode()
    handler, mock_wfile = _make_handler("/v1/chat/completions", request_body)
    handler.do_POST()

    assert handler._status == 200
    body = json.loads(mock_wfile.getvalue())
    assert body["choices"][0]["message"]["content"] == '{"answer": 4}'
    # Verify response_format was passed
    call_kwargs = mock_server.chat.call_args[1]
    assert call_kwargs.get("response_format") == {"type": "json_object"}


def test_api_chat_completions_specialist_fallback():
    """Specialist failure falls back to primary."""
    mock_server = MagicMock()
    mock_server.chat.side_effect = [
        Exception("specialist down"),
        {"choices": [{"message": {"role": "assistant", "content": "primary result"}, "finish_reason": "stop"}]},
    ]
    mock_router = MagicMock()
    mock_router.classify.return_value = ("SPECIALIST", 0.88)
    _setup_app_state(server=mock_server, router=mock_router)

    request_body = _make_request_body(
        messages=[{"role": "user", "content": "summarize this"}],
        tools=[{"type": "function", "function": {"name": "custom_tool", "parameters": {"type": "object", "properties": {}}}}],
    )
    handler, mock_wfile = _make_handler("/v1/chat/completions", request_body)
    handler.do_POST()

    assert handler._status == 200
    body = json.loads(mock_wfile.getvalue())
    assert body["choices"][0]["message"]["content"] == "primary result"


def test_api_chat_completions_primary_error():
    """Primary failure returns 500."""
    mock_server = MagicMock()
    mock_server.chat.side_effect = Exception("server down")
    mock_router = MagicMock()
    mock_router.classify.return_value = ("PRIMARY", 0.9)
    _setup_app_state(server=mock_server, router=mock_router)

    request_body = _make_request_body(
        messages=[{"role": "user", "content": "write a function"}])
    handler, mock_wfile = _make_handler("/v1/chat/completions", request_body)
    handler.do_POST()

    assert handler._status == 500
    body = json.loads(mock_wfile.getvalue())
    assert "error" in body


def test_api_chat_completions_with_tool_calls():
    """Model responds with tool_calls → tool proxy executes them."""
    mock_server = MagicMock()
    mock_server.chat.side_effect = [
        # Round 1: model requests tool call
        {"choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": "test.py"})}}]
        }, "finish_reason": "tool_calls"}]},
        # Round 2: final response
        {"choices": [{"message": {"role": "assistant", "content": "Found the file"}, "finish_reason": "stop"}]},
    ]
    mock_router = MagicMock()
    mock_router.classify.return_value = ("SPECIALIST", 0.88)
    _setup_app_state(server=mock_server, router=mock_router)

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "test.py").write_text("print('hello')\n")

        # Agent must pass tools to trigger run_tool_loop
        request_body = json.dumps({
            "messages": [{"role": "user", "content": "read test.py"}],
            "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
            "stream": False,
            "repo_root": td,
        }).encode()
        handler, mock_wfile = _make_handler("/v1/chat/completions", request_body)
        handler.do_POST()

    assert handler._status == 200
    body = json.loads(mock_wfile.getvalue())
    assert body["choices"][0]["message"]["content"] == "Found the file"
    assert mock_server.chat.call_count == 2


def test_api_chat_completions_multimodal():
    """POST with image reference routes to multimodal."""
    mock_server = MagicMock()
    mock_server.chat.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "image desc"}, "finish_reason": "stop"}]
    }
    mock_router = MagicMock()
    _setup_app_state(server=mock_server, router=mock_router)

    request_body = _make_request_body(
        messages=[{"role": "user", "content": "describe photo.png"}])
    handler, mock_wfile = _make_handler("/v1/chat/completions", request_body)
    handler.do_POST()

    # Multimodal route is detected but model server handles it (may fail since no swap)
    # Just check the route was set
    body = json.loads(mock_wfile.getvalue())
    assert body["lore"]["route"] == "MULTIMODAL"


def test_api_chat_completions_agent_tools_passthrough():
    """Agent-defined tools are passed through to the model."""
    mock_server = MagicMock()
    mock_server.chat.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]
    }
    mock_router = MagicMock()
    mock_router.classify.return_value = ("PRIMARY", 0.9)
    _setup_app_state(server=mock_server, router=mock_router)

    agent_tools = [{"type": "function", "function": {"name": "custom_tool", "parameters": {"type": "object", "properties": {}}}}]
    request_body = json.dumps({
        "messages": [{"role": "user", "content": "use custom tool"}],
        "tools": agent_tools,
        "stream": False,
    }).encode()
    handler, mock_wfile = _make_handler("/v1/chat/completions", request_body)
    handler.do_POST()

    assert handler._status == 200
    # Tools were passed to run_tool_loop → server.chat
    call_kwargs = mock_server.chat.call_args[1]
    assert "tools" in call_kwargs
