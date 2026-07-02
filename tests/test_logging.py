# tests/test_logging.py
import json
import tempfile
from pathlib import Path

def test_log_request_writes_jsonl():
    """Logger appends JSON object to jsonl file."""
    with tempfile.TemporaryDirectory() as d:
        from lore.logging import RequestLogger
        logger = RequestLogger(log_path=str(Path(d) / "requests.jsonl"))

        logger.log_request({
            "timestamp": "2026-07-02T18:30:00Z",
            "input_hash": "sha256:abc123",
            "route": "SPECIALIST",
            "confidence": 0.92,
            "model": "falcon-h1-1.5b",
            "tokens_in": 145,
            "tokens_out": 32,
            "latency_ms": 1200,
            "success": True,
            "fallback": False,
            "context_truncated": False,
            "cache_hit": None,
            "error": None,
        })

        lines = Path(d, "requests.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["route"] == "SPECIALIST"
        assert entry["confidence"] == 0.92

def test_log_request_appends():
    """Multiple log calls append to same file."""
    with tempfile.TemporaryDirectory() as d:
        from lore.logging import RequestLogger
        logger = RequestLogger(log_path=str(Path(d) / "requests.jsonl"))

        for i in range(3):
            logger.log_request({"route": "PRIMARY", "index": i})

        lines = Path(d, "requests.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3
