# tests/test_session.py
import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class _FakeContext:
    """Minimal ContextManager stub for session tests."""
    def __init__(self, system_prompt="", history=None):
        self._system_prompt = system_prompt
        self._history = list(history or [])

    @property
    def system_prompt(self):
        return self._system_prompt

    @property
    def history(self):
        return list(self._history)

    def restore(self, system_prompt, history):
        self._system_prompt = system_prompt
        self._history = list(history)

    def build_prompt(self):
        return [{"role": "system", "content": self._system_prompt}]


def test_save_session_writes_files(tmp_path):
    """save_session writes context.json and metadata.json."""
    from lore.session import SessionManager
    sm = SessionManager({"save_dir": str(tmp_path / "sessions")})

    mock_server = MagicMock()
    mock_ctx = _FakeContext(
        system_prompt="You are a helpful assistant.",
        history=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
    )

    sid = sm.save_session("test-001", mock_server, mock_ctx)
    assert sid == "test-001"

    session_dir = tmp_path / "sessions" / "test-001"
    assert (session_dir / "context.json").exists()
    assert (session_dir / "metadata.json").exists()

    ctx_data = json.loads((session_dir / "context.json").read_text())
    assert ctx_data["system_prompt"] == "You are a helpful assistant."
    assert len(ctx_data["history"]) == 2

    meta = json.loads((session_dir / "metadata.json").read_text())
    assert meta["session_id"] == "test-001"
    assert meta["turn_count"] == 1
    assert meta["topic"] == "Hello"


def test_resume_session_restores_context(tmp_path):
    """resume_session loads saved history into the context manager."""
    from lore.session import SessionManager
    sm = SessionManager({"save_dir": str(tmp_path / "sessions")})

    mock_server = MagicMock()
    # Simulate a successful prefix replay warmup
    mock_server.chat.return_value = {"choices": [{"message": {"content": "ok"}}]}

    mock_ctx = _FakeContext(
        system_prompt="Test prompt.",
        history=[
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "6"},
        ],
    )
    sm.save_session("sess-123", mock_server, mock_ctx)

    # Create a fresh context to resume into
    new_ctx = _FakeContext()

    result = sm.resume_session("sess-123", mock_server, new_ctx)
    assert result is True
    assert new_ctx._system_prompt == "Test prompt."
    assert len(new_ctx._history) == 4
    # Prefix replay warmup should have been called
    mock_server.chat.assert_called_once()


def test_resume_nonexistent_session_returns_false(tmp_path):
    """resume_session returns False for a session that doesn't exist."""
    from lore.session import SessionManager
    sm = SessionManager({"save_dir": str(tmp_path / "sessions")})
    mock_server = MagicMock()
    mock_ctx = MagicMock()
    result = sm.resume_session("nonexistent", mock_server, mock_ctx)
    assert result is False


def test_list_sessions_returns_metadata(tmp_path):
    """list_sessions returns metadata for all saved sessions."""
    from lore.session import SessionManager
    sm = SessionManager({"save_dir": str(tmp_path / "sessions")})

    mock_server = MagicMock()
    for i in range(3):
        mock_ctx = _FakeContext("prompt", [{"role": "user", "content": f"session {i}"}])
        sm.save_session(f"sess-{i}", mock_server, mock_ctx)

    sessions = sm.list_sessions()
    assert len(sessions) == 3
    assert all("session_id" in s for s in sessions)
    assert all("turn_count" in s for s in sessions)


def test_cleanup_old_sessions(tmp_path):
    """cleanup_old_sessions deletes sessions older than max_age_days."""
    from lore.session import SessionManager
    sm = SessionManager({"save_dir": str(tmp_path / "sessions")})

    mock_server = MagicMock()
    mock_ctx = _FakeContext("prompt", [{"role": "user", "content": "old"}])
    sm.save_session("old-sess", mock_server, mock_ctx)

    # Manually backdate the metadata timestamp
    meta_path = tmp_path / "sessions" / "old-sess" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["timestamp"] = time.time() - (8 * 86400)  # 8 days ago
    meta_path.write_text(json.dumps(meta))

    deleted = sm.cleanup_old_sessions(max_age_days=7)
    assert deleted == 1
    assert not (tmp_path / "sessions" / "old-sess").exists()


def test_max_sessions_enforced(tmp_path):
    """Saving beyond max_sessions deletes the oldest."""
    from lore.session import SessionManager
    sm = SessionManager({"save_dir": str(tmp_path / "sessions"), "max_sessions": 2})

    mock_server = MagicMock()
    for i in range(4):
        mock_ctx = _FakeContext("prompt", [{"role": "user", "content": f"session {i}"}])
        sm.save_session(f"sess-{i}", mock_server, mock_ctx)
        time.sleep(0.01)  # ensure distinct timestamps

    sessions = sm.list_sessions()
    assert len(sessions) == 2  # only max_sessions kept


def test_infer_topic_from_first_user_message(tmp_path):
    """Session topic is inferred from the first user message."""
    from lore.session import SessionManager
    sm = SessionManager({"save_dir": str(tmp_path / "sessions")})

    mock_server = MagicMock()
    mock_ctx = _FakeContext("prompt", [
        {"role": "assistant", "content": "Welcome!"},
        {"role": "user", "content": "Help me debug a memory leak in Python"},
        {"role": "assistant", "content": "Let's start by profiling..."},
    ])
    sm.save_session("topic-test", mock_server, mock_ctx)

    meta = json.loads((tmp_path / "sessions" / "topic-test" / "metadata.json").read_text())
    assert "memory leak" in meta["topic"]


def test_resume_session_without_server(tmp_path):
    """resume_session works without server (skips prefix replay)."""
    from lore.session import SessionManager
    sm = SessionManager({"save_dir": str(tmp_path / "sessions")})

    mock_ctx = _FakeContext("prompt", [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "ok"},
    ])
    sm.save_session("no-server", None, mock_ctx)

    new_ctx = _FakeContext()
    result = sm.resume_session("no-server", None, new_ctx)
    assert result is True
    assert len(new_ctx._history) == 2
