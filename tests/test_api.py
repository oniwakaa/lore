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


class MockHTTPRequest:
    """Minimal mock of an HTTP request for testing handlers."""
    def __init__(self, method="GET", path="/health", body=b"", headers=None):
        self.method = method
        self.path = path
        self.body = body
        self.headers = headers or {}
        self._response = None
        self._status = None
        self._headers = {}

    def makefile(self, *args, **kwargs):
        return BytesIO(self.body)

    def send_response(self, status):
        self._status = status

    def send_header(self, key, value):
        self._headers[key] = value

    def end_headers(self):
        pass

    def read(self, size=-1):
        return self.body


def test_api_health_endpoint():
    """GET /health returns 200 with status ok."""
    from lore.api import LoreHandler, _app_state

    # Minimal mock of the handler
    mock_wfile = BytesIO()

    class TestHandler(LoreHandler):
        def __init__(self):
            self.path = "/health"
            self.headers = {}
            self.wfile = mock_wfile
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

    handler = TestHandler()
    handler.do_GET()

    assert handler._status == 200
    body = json.loads(mock_wfile.getvalue())
    assert body["status"] == "ok"


def test_api_models_endpoint():
    """GET /v1/models returns list of running models."""
    from lore.api import LoreHandler, _app_state

    mock_server = MagicMock()
    mock_server.is_model_running.return_value = True
    mock_cfg = MagicMock()
    mock_cfg.models = {
        "primary": {"name": "Ornith-1.0-9B"},
        "specialist": {"name": "Falcon-H1-1.5B"},
    }
    _app_state["server"] = mock_server
    _app_state["cfg"] = mock_cfg

    mock_wfile = BytesIO()

    class TestHandler(LoreHandler):
        def __init__(self):
            self.path = "/v1/models"
            self.headers = {}
            self.wfile = mock_wfile
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

    handler = TestHandler()
    handler.do_GET()

    assert handler._status == 200
    body = json.loads(mock_wfile.getvalue())
    assert body["object"] == "list"
    assert len(body["data"]) == 2


def test_api_chat_completions_empty_body():
    """POST /v1/chat/completions with empty body returns 400."""
    from lore.api import LoreHandler, _app_state

    mock_wfile = BytesIO()

    class TestHandler(LoreHandler):
        def __init__(self):
            self.path = "/v1/chat/completions"
            self.headers = {"Content-Length": "0"}
            self.wfile = mock_wfile
            self.rfile = BytesIO(b"")
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

    handler = TestHandler()
    handler.do_POST()

    assert handler._status == 400
    body = json.loads(mock_wfile.getvalue())
    assert "error" in body


def test_api_chat_completions_no_messages():
    """POST /v1/chat/completions with no messages returns 400."""
    from lore.api import LoreHandler

    mock_wfile = BytesIO()
    request_body = json.dumps({"messages": []}).encode()

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

    handler = TestHandler()
    handler.do_POST()

    assert handler._status == 400


def test_api_chat_completions_stream_rejected():
    """POST /v1/chat/completions with stream=true returns 400."""
    from lore.api import LoreHandler

    mock_wfile = BytesIO()
    request_body = json.dumps({
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }).encode()

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

    handler = TestHandler()
    handler.do_POST()

    assert handler._status == 400
    body = json.loads(mock_wfile.getvalue())
    assert "stream" in body["error"].lower()


def test_api_chat_completions_success():
    """POST /v1/chat/completions with valid request returns OpenAI-format response."""
    from lore.api import LoreHandler, _app_state

    # Mock the dispatch pipeline
    mock_result = {
        "route": "PRIMARY", "confidence": 0.95,
        "model": "primary", "content": "4",
        "success": True, "latency_ms": 1500,
    }

    mock_server = MagicMock()
    mock_router = MagicMock()
    mock_ctx = MagicMock()
    mock_memory = MagicMock()
    mock_logger = MagicMock()
    mock_verifier = MagicMock()

    _app_state.update({
        "server": mock_server, "router": mock_router,
        "ctx": mock_ctx, "memory": mock_memory,
        "req_logger": mock_logger, "verifier": mock_verifier,
    })

    mock_wfile = BytesIO()
    request_body = json.dumps({
        "messages": [{"role": "user", "content": "what is 2+2?"}],
        "stream": False,
    }).encode()

    with patch("lore.api._dispatch" if False else "lore.cli._dispatch") as mock_dispatch:
        mock_dispatch.return_value = mock_result

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

        handler = TestHandler()
        handler.do_POST()

    assert handler._status == 200
    body = json.loads(mock_wfile.getvalue())
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "4"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "id" in body
    assert "usage" in body
    assert body["lore"]["route"] == "PRIMARY"


def test_api_chat_completions_json_mode():
    """POST with response_format=json_object passes json_mode to dispatch."""
    from lore.api import LoreHandler, _app_state

    mock_result = {
        "route": "PRIMARY", "confidence": 0.9,
        "model": "primary", "content": '{"answer": 4}',
        "success": True, "latency_ms": 2000,
    }

    _app_state.update({
        "server": MagicMock(), "router": MagicMock(),
        "ctx": MagicMock(), "memory": MagicMock(),
        "req_logger": MagicMock(), "verifier": MagicMock(),
    })

    mock_wfile = BytesIO()
    request_body = json.dumps({
        "messages": [{"role": "user", "content": "return json"}],
        "response_format": {"type": "json_object"},
        "stream": False,
    }).encode()

    with patch("lore.cli._dispatch") as mock_dispatch:
        mock_dispatch.return_value = mock_result

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

        handler = TestHandler()
        handler.do_POST()

    # Verify json_mode was passed to dispatch (positional arg index 6)
    call_args = mock_dispatch.call_args
    assert call_args[0][6] is True
