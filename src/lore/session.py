"""Session save/resume. Persists context + memory state to disk for restart.

llama-server doesn't have built-in KV dump/load. The approach is to save the
full context (messages + memory state) as JSON and replay it on resume. For
SSM models, the recurrent state is per-token, so replaying the prefix is the
only option. Prefix caching makes replay fast (one prefill pass).

Sessions live in sessions/{id}/:
  - context.json:  message history + system prompt
  - metadata.json: timestamp, turn count, topic, session_id
"""
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class SessionManager:
    """Save and restore session state across restarts."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._save_dir = Path(cfg.get("save_dir", "sessions"))
        self._auto_save_every_n = cfg.get("auto_save_every_n_turns", 10)
        self._max_sessions = cfg.get("max_sessions", 50)
        self._save_dir.mkdir(parents=True, exist_ok=True)

    def save_session(self, session_id: str, server, context) -> str:
        """Save current context + memory metadata to disk.

        Args:
            session_id: Unique session identifier.
            server: ModelServer instance (unused for now — KV cache is rebuilt
                    on resume via prefix replay, not dumped to disk).
            context: ContextManager instance with message history.

        Returns:
            session_id for later resume.
        """
        session_dir = self._save_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save context: history + system prompt
        context_data = {
            "system_prompt": context._system_prompt,
            "history": context._history,
        }
        (session_dir / "context.json").write_text(json.dumps(context_data, indent=2))

        # Save metadata
        turn_count = len(context._history) // 2
        metadata = {
            "session_id": session_id,
            "timestamp": time.time(),
            "turn_count": turn_count,
            "message_count": len(context._history),
            "topic": self._infer_topic(context._history),
        }
        (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        logger.info(f"Saved session {session_id} ({turn_count} turns)")
        self._enforce_max_sessions()
        return session_id

    def resume_session(self, session_id: str, server, context) -> bool:
        """Restore context from disk + rebuild KV cache via prefix replay.

        Args:
            session_id: Session to resume.
            server: ModelServer instance (used for prefix replay warmup).
            context: ContextManager instance to populate with saved history.

        Returns:
            True if session was restored, False if session not found.
        """
        session_dir = self._save_dir / session_id
        context_path = session_dir / "context.json"
        if not context_path.exists():
            logger.warning(f"Session {session_id} not found")
            return False

        try:
            data = json.loads(context_path.read_text())
            context._system_prompt = data.get("system_prompt", "")
            context._history = data.get("history", [])

            # Rebuild KV cache by feeding the saved messages as a prefix.
            # This triggers a single prefill pass; subsequent generation
            # benefits from the prefix cache. For SSM models this is the
            # only option since recurrent state is per-token.
            messages = context.build_prompt()
            if messages and server:
                try:
                    # Warmup: send a 1-token generation to prefill the KV cache
                    server.chat("primary", messages, max_tokens=1, temperature=0)
                    logger.info(f"Replayed prefix for session {session_id}")
                except Exception as e:
                    logger.warning(f"Prefix replay warmup failed ({e}), context loaded but KV not warmed")

            logger.info(f"Resumed session {session_id} ({len(context._history)} messages)")
            return True
        except Exception as e:
            logger.error(f"Failed to resume session {session_id}: {e}")
            return False

    def list_sessions(self) -> list[dict]:
        """List saved sessions with metadata (timestamp, turn count, topic)."""
        sessions = []
        for session_dir in sorted(self._save_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            meta_path = session_dir / "metadata.json"
            if meta_path.exists():
                try:
                    sessions.append(json.loads(meta_path.read_text()))
                except Exception:
                    # Corrupt metadata, skip
                    continue
        return sessions

    def cleanup_old_sessions(self, max_age_days: int = 7) -> int:
        """Delete sessions older than max_age_days. Return count deleted."""
        cutoff = time.time() - (max_age_days * 86400)
        deleted = 0
        for session_dir in self._save_dir.iterdir():
            if not session_dir.is_dir():
                continue
            meta_path = session_dir / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    if meta.get("timestamp", 0) < cutoff:
                        self._delete_session_dir(session_dir)
                        deleted += 1
                except Exception:
                    # Corrupt metadata, delete based on dir mtime as fallback
                    if session_dir.stat().st_mtime < cutoff:
                        self._delete_session_dir(session_dir)
                        deleted += 1
        if deleted:
            logger.info(f"Cleaned up {deleted} old sessions")
        return deleted

    def _enforce_max_sessions(self) -> None:
        """If over max_sessions, delete oldest by timestamp."""
        sessions = self.list_sessions()
        if len(sessions) <= self._max_sessions:
            return
        # Sort by timestamp ascending, delete oldest
        sessions.sort(key=lambda s: s.get("timestamp", 0))
        excess = len(sessions) - self._max_sessions
        for meta in sessions[:excess]:
            sid = meta.get("session_id")
            if sid:
                self._delete_session_dir(self._save_dir / sid)

    def _delete_session_dir(self, session_dir: Path) -> None:
        """Recursively delete a session directory."""
        for f in session_dir.iterdir():
            f.unlink()
        session_dir.rmdir()

    def _infer_topic(self, history: list[dict]) -> str:
        """Infer a short topic from the first user message."""
        for msg in history:
            if msg.get("role") == "user" and msg.get("content"):
                # First 60 chars of first user message
                return msg["content"][:60].replace("\n", " ")
        return "untitled"
