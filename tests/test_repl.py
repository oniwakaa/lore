# tests/test_repl.py
"""REPL test coverage for _run_repl() in cli.py."""
import pytest
from unittest.mock import patch, MagicMock, call


def _make_mocks():
    """Create standard mocks for REPL dependencies."""
    mock_server = MagicMock()
    mock_server.chat.return_value = {"choices": [{"message": {"content": "response"}}]}
    mock_router = MagicMock()
    mock_router.classify.return_value = ("PRIMARY", 0.95)
    mock_ctx = MagicMock()
    mock_ctx.was_truncated = False
    mock_ctx.history = []
    mock_ctx.system_prompt = "test"
    mock_memory = MagicMock()
    mock_memory.retrieve.return_value = []
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.session = {"auto_save_every_n_turns": 100}
    return mock_server, mock_router, mock_ctx, mock_memory, mock_logger, mock_cfg


def test_repl_exit():
    """/exit breaks the loop cleanly."""
    from lore.cli import _run_repl
    server, router, ctx, memory, logger, cfg = _make_mocks()
    with patch("builtins.input", side_effect=["/exit"]):
        _run_repl(server, router, ctx, memory, logger, cfg)
    server.stop_all.assert_called_once()


def test_repl_clear():
    """/clear calls ctx.clear() and memory.clear()."""
    from lore.cli import _run_repl
    server, router, ctx, memory, logger, cfg = _make_mocks()
    with patch("builtins.input", side_effect=["/clear", "/exit"]):
        _run_repl(server, router, ctx, memory, logger, cfg)
    ctx.clear.assert_called_once()
    memory.clear.assert_called_once()


def test_repl_route():
    """/route prints the last route."""
    from lore.cli import _run_repl
    server, router, ctx, memory, logger, cfg = _make_mocks()
    with patch("builtins.input", side_effect=["/route", "/exit"]):
        _run_repl(server, router, ctx, memory, logger, cfg)
    # No crash — last_route is None initially


def test_repl_save():
    """/save [name] calls session_mgr.save_session."""
    from lore.cli import _run_repl
    server, router, ctx, memory, logger, cfg = _make_mocks()
    mock_session_mgr = MagicMock()
    with patch("builtins.input", side_effect=["/save test-session", "/exit"]):
        _run_repl(server, router, ctx, memory, logger, cfg, session_mgr=mock_session_mgr)
    mock_session_mgr.save_session.assert_called_once_with("test-session", server, ctx)


def test_repl_resume():
    """/resume <name> calls session_mgr.resume_session."""
    from lore.cli import _run_repl
    server, router, ctx, memory, logger, cfg = _make_mocks()
    mock_session_mgr = MagicMock()
    mock_session_mgr.resume_session.return_value = True
    with patch("builtins.input", side_effect=["/resume test-session", "/exit"]):
        _run_repl(server, router, ctx, memory, logger, cfg, session_mgr=mock_session_mgr)
    mock_session_mgr.resume_session.assert_called_once_with("test-session", server, ctx)


def test_repl_sessions():
    """/sessions lists saved sessions."""
    from lore.cli import _run_repl
    server, router, ctx, memory, logger, cfg = _make_mocks()
    mock_session_mgr = MagicMock()
    mock_session_mgr.list_sessions.return_value = []
    with patch("builtins.input", side_effect=["/sessions", "/exit"]):
        _run_repl(server, router, ctx, memory, logger, cfg, session_mgr=mock_session_mgr)
    mock_session_mgr.list_sessions.assert_called_once()


def test_repl_query_dispatches():
    """Regular query dispatches to _dispatch."""
    from lore.cli import _run_repl
    server, router, ctx, memory, logger, cfg = _make_mocks()
    with patch("builtins.input", side_effect=["hello world", "/exit"]):
        with patch("lore.cli._dispatch") as mock_dispatch:
            mock_dispatch.return_value = {
                "route": "PRIMARY", "confidence": 0.95,
                "model": "primary", "content": "hi there",
                "success": True, "latency_ms": 50.0,
            }
            _run_repl(server, router, ctx, memory, logger, cfg, verifier=None)
    mock_dispatch.assert_called_once()
    args = mock_dispatch.call_args
    assert args[0][0] == "hello world"


def test_repl_error_continues():
    """Exception in dispatch prints error and continues loop."""
    from lore.cli import _run_repl
    server, router, ctx, memory, logger, cfg = _make_mocks()
    with patch("builtins.input", side_effect=["bad query", "/exit"]):
        with patch("lore.cli._dispatch", side_effect=RuntimeError("boom")):
            _run_repl(server, router, ctx, memory, logger, cfg, verifier=None)
    # Should not crash — server.stop_all still called on exit
    server.stop_all.assert_called_once()


def test_repl_eof_breaks():
    """EOFError breaks the loop."""
    from lore.cli import _run_repl
    server, router, ctx, memory, logger, cfg = _make_mocks()
    with patch("builtins.input", side_effect=EOFError()):
        _run_repl(server, router, ctx, memory, logger, cfg)
    server.stop_all.assert_called_once()


def test_repl_empty_input_skipped():
    """Empty input is skipped, not dispatched."""
    from lore.cli import _run_repl
    server, router, ctx, memory, logger, cfg = _make_mocks()
    with patch("builtins.input", side_effect=["", "/exit"]):
        with patch("lore.cli._dispatch") as mock_dispatch:
            _run_repl(server, router, ctx, memory, logger, cfg, verifier=None)
    mock_dispatch.assert_not_called()
