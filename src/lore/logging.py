# src/lore/logging.py
"""Request logger. Appends JSON lines to requests.jsonl for Phase 2 A/B testing."""
import json
from datetime import datetime, timezone
from pathlib import Path

class RequestLogger:
    """Append-only JSONL logger for routing decisions, model calls, latency."""

    def __init__(self, log_path: str = "logs/requests.jsonl"):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_request(self, entry: dict) -> None:
        """Append a log entry as one JSON line."""
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_route(self, text_hash: str, route: str, confidence: float) -> None:
        """Convenience method for logging a routing decision."""
        self.log_request({
            "input_hash": text_hash,
            "route": route,
            "confidence": confidence,
        })
