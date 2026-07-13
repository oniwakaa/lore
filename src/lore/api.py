# src/lore/api.py
"""OpenAI-compatible API server for LORE.

Exposes /v1/chat/completions and /v1/models so tools like Continue.dev,
Cline, and Open WebUI can use LORE as a drop-in replacement for OpenAI.

Usage:
  PYTHONPATH=src python3 -m lore.api --port 8000
  # or via CLI:
  lore --api --port 8000
"""
import json
import logging
import threading
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import yaml

from lore.config import LoreConfig
from lore.models import ModelServer
from lore.router import Router
from lore.context import ContextManager
from lore.memory import HierarchicalMemory
from lore.health import ContextHealth
from lore.session import SessionManager
from lore.logging import RequestLogger
from lore.tool_handler import handle_tool_only
from lore.tool_attention import ToolAttention
from lore.verifier import Verifier
from lore.sizing import estimate_context_budget
from lore.orchestrator import Orchestrator
from lore.classifier import TaskClassifier
from lore.registry import ModelRegistry
from lore.cli import is_multimodal

logger = logging.getLogger(__name__)

# Global state — initialized once by create_app()
_app_state = {}


def _init_lore():
    """Initialize all LORE components. Called once at startup."""
    cfg = LoreConfig.load()

    server = ModelServer(cfg.models)
    logger.info("Starting model servers...")
    server.start_all()

    router_cfg = cfg.router
    router = Router.load(
        router_cfg.get("model_path", "configs/router_model.joblib"),
        confidence_threshold=router_cfg.get("confidence_threshold", 0.70),
    )

    system_prompt = "You are a helpful assistant. Answer concisely and accurately."
    tokenizer_source = cfg.models.get("defaults", {}).get("tokenizer_source", "local")
    tokenizer_repo = cfg.models.get("primary", {}).get("source", "")
    if tokenizer_repo.endswith("-GGUF"):
        tokenizer_repo = tokenizer_repo[:-len("-GGUF")]
    tool_attention = ToolAttention.from_config(server)
    compression_cfg_path = Path("configs/compression.yaml")
    compression_cfg = yaml.safe_load(compression_cfg_path.read_text()) if compression_cfg_path.exists() else {}
    health_cfg = cfg.memory.get("health", {})
    health = ContextHealth(health_cfg) if health_cfg.get("enabled", False) else None
    memory = HierarchicalMemory(cfg.memory, server)
    ctx = ContextManager(cfg.context, server, system_prompt=system_prompt,
                          tokenizer_source=tokenizer_source, tokenizer_repo=tokenizer_repo or None,
                          tool_attention=tool_attention, compression=compression_cfg,
                          memory=memory, health=health)
    session_mgr = SessionManager(cfg.session if hasattr(cfg, "session") else {})
    verifier_cfg_path = Path("configs/verifier.yaml")
    verifier_cfg = yaml.safe_load(verifier_cfg_path.read_text()) if verifier_cfg_path.exists() else {}
    verifier = Verifier(verifier_cfg)
    req_logger = RequestLogger()

    orch_cfg_path = Path("configs/orchestrator.yaml")
    orch_cfg = yaml.safe_load(orch_cfg_path.read_text()) if orch_cfg_path.exists() else {}
    models_cfg = cfg.models
    registry = None
    if isinstance(models_cfg, dict) and "orchestrator" in models_cfg:
        try:
            registry = ModelRegistry(models_cfg, models_dir="models")
        except Exception as e:
            logger.warning(f"Model registry init failed: {e}")
    # Classifier disabled from normal startup per single-writer design.
    # TF-IDF router alone handles routing; classifier is not on the critical path.
    classifier = None
    orchestrator = Orchestrator(server, router, memory, orch_cfg,
                                ctx=ctx, req_logger=req_logger, verifier=verifier,
                                classifier=classifier, registry=registry)

    cache_active = server.verify_prefix_cache()
    if not cache_active:
        logger.warning("Prefix cache not verified — responses may be slower")

    return {
        "cfg": cfg, "server": server, "router": router, "ctx": ctx,
        "memory": memory, "req_logger": req_logger, "verifier": verifier,
        "orchestrator": orchestrator, "session_mgr": session_mgr,
    }


class LoreHandler(BaseHTTPRequestHandler):
    """HTTP handler implementing OpenAI-compatible API."""

    def log_message(self, format, *args):
        """Suppress default stderr logging — use our logger."""
        logger.debug(format % args)

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            models = []
            server = _app_state["server"]
            for role in ("primary", "specialist"):
                if server.is_model_running(role):
                    cfg = _app_state["cfg"].models.get(role, {})
                    models.append({
                        "id": cfg.get("name", role),
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "lore",
                    })
            self._send_json(200, {"object": "list", "data": models})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        else:
            self._send_json(404, {"error": "Not found"})

    def _handle_chat_completions(self):
        # Parse request body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(400, {"error": "Empty request body"})
            return

        try:
            body = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        messages = body.get("messages", [])
        if not messages:
            self._send_json(400, {"error": "No messages provided"})
            return

        # Extract the user query (last user message)
        user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break
        if not user_msg:
            self._send_json(400, {"error": "No user message found"})
            return

        # Optional params
        max_tokens = body.get("max_tokens", 2048)
        temperature = body.get("temperature", 0.7)
        json_mode = body.get("response_format", {}).get("type") == "json_object"
        stream = body.get("stream", False)

        # Clamp max_tokens: positive int, max 8192, default 2048
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            max_tokens = 2048
        max_tokens = min(max_tokens, 8192)

        # Validate temperature: 0-2, default 0.7
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            temperature = 0.7

        if stream:
            self._send_json(400, {"error": "Streaming not yet supported. Set stream=false."})
            return

        # Use LORE's dispatch pipeline
        server = _app_state["server"]
        router = _app_state["router"]
        ctx = _app_state["ctx"]
        memory = _app_state["memory"]
        req_logger = _app_state["req_logger"]
        verifier = _app_state["verifier"]
        orchestrator = _app_state["orchestrator"]

        from lore.cli import _dispatch
        t0 = time.time()

        # Load prior messages into context (not just the last user message)
        for msg in messages[:-1]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                ctx.add_message(role, content)

        try:
            dispatch_fn = lambda q, json_mode=False: _dispatch(
                q, server, router, ctx, memory, req_logger, json_mode, verifier,
                max_tokens=max_tokens, temperature=temperature)
            result = orchestrator.process(user_msg, json_mode=json_mode, dispatch_fn=dispatch_fn)
        except Exception as e:
            logger.error(f"Dispatch failed: {e}")
            self._send_json(500, {"error": f"Internal error: {e}"})
            return

        latency_ms = int((time.time() - t0) * 1000)

        # Build OpenAI-compatible response
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result.get("model", "lore"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.get("content", ""),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(user_msg.split()),  # rough estimate
                "completion_tokens": len(result.get("content", "").split()),
                "total_tokens": len(user_msg.split()) + len(result.get("content", "").split()),
            },
            "lore": {
                "route": result.get("route"),
                "confidence": result.get("confidence"),
                "orchestrated": result.get("orchestrated", False),
                "subtasks_completed": result.get("subtasks_completed", 0),
                "latency_ms": latency_ms,
            },
        }

        self._send_json(200, response)


def create_server(host: str = "127.0.0.1", port: int = 8000) -> HTTPServer:
    """Create and return the HTTP server (does not start it)."""
    return HTTPServer((host, port), LoreHandler)


def main():
    """Run the LORE API server."""
    import argparse

    parser = argparse.ArgumentParser(prog="lore-api", description="LORE OpenAI-compatible API")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--init-only", action="store_true", help="Init LORE but don't start server (for testing)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    logger.info("Initializing LORE...")
    global _app_state
    _app_state = _init_lore()
    logger.info("LORE initialized.")

    if args.init_only:
        return

    server = create_server(args.host, args.port)
    logger.info(f"LORE API server listening on http://{args.host}:{args.port}")
    logger.info(f"  POST /v1/chat/completions  — OpenAI-compatible chat")
    logger.info(f"  GET  /v1/models            — list available models")
    logger.info(f"  GET  /health               — health check")
    logger.info("")
    logger.info("Use with Continue.dev, Cline, Open WebUI, or any OpenAI-compatible client.")
    logger.info("Set base_url to http://127.0.0.1:{args.port}/v1")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        _app_state["server"].stop_all()
        server.shutdown()


if __name__ == "__main__":
    main()
