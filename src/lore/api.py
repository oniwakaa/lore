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
from lore.tool_proxy import TOOL_DEFINITIONS, run_tool_loop
from lore.verifier import Verifier
from lore.sizing import estimate_context_budget
from lore.cli import is_multimodal

logger = logging.getLogger(__name__)

# Global state — initialized once by create_app()
_app_state = {}

# ponytail: conservative token estimate. Real ratio varies 2.5-4.5 chars/token
# depending on content. Using 3 to be safe — better to over-truncate than 400.
_CHARS_PER_TOKEN = 3


def _truncate_messages(messages: list[dict], max_tokens: int, keep_system: bool = True) -> list[dict]:
    """Truncate message list to fit within max_tokens (rough estimate).

    Keeps the system message (if keep_system) and the most recent messages.
    Drops older messages from the middle. If a single message exceeds the
    budget, truncates its content.
    """
    if not messages:
        return messages

    total_chars = sum(len(m.get("content", "") or "") for m in messages)
    total_tokens = total_chars // _CHARS_PER_TOKEN

    if total_tokens <= max_tokens:
        return messages

    # Split: system messages (keep) + conversation messages (truncate from front)
    system_msgs = []
    conv_msgs = []
    for m in messages:
        if m.get("role") == "system" and keep_system:
            system_msgs.append(m)
        else:
            conv_msgs.append(m)

    system_tokens = sum(len(m.get("content", "") or "") for m in system_msgs) // _CHARS_PER_TOKEN
    budget = max_tokens - system_tokens

    # Keep most recent messages, drop oldest until under budget
    while conv_msgs and budget < 0:
        dropped = conv_msgs.pop(0)
        budget += len(dropped.get("content", "") or "") // _CHARS_PER_TOKEN

    # If still over budget, truncate the largest message
    conv_chars = sum(len(m.get("content", "") or "") for m in conv_msgs)
    conv_tokens = conv_chars // _CHARS_PER_TOKEN
    while conv_tokens > budget and conv_msgs:
        # Find the largest non-system message and truncate it
        largest_idx = 0
        largest_len = 0
        for i, m in enumerate(conv_msgs):
            clen = len(m.get("content", "") or "")
            if clen > largest_len:
                largest_len = clen
                largest_idx = i
        if largest_len <= 100:
            break  # can't truncate further meaningfully
        m = conv_msgs[largest_idx]
        content = m.get("content", "") or ""
        # Cut to half
        m["content"] = content[:len(content) // 2] + "\n[...truncated...]"
        conv_tokens = sum(len(m.get("content", "") or "") for m in conv_msgs) // _CHARS_PER_TOKEN

    result = system_msgs + conv_msgs
    dropped_count = len(messages) - len(result)
    if dropped_count > 0:
        logger.info(f"Truncated {dropped_count} messages to fit {max_tokens} token budget")
    return result


def _truncate_to_recent(messages: list[dict], keep: int = 4) -> list[dict]:
    """Keep system messages + the N most recent conversation messages.

    Used for the specialist model: it handles simple ops (file read, search, QA)
    and doesn't need full conversation history. Keeps it fast regardless of
    how large the Droid's session is.
    """
    if not messages:
        return messages

    system_msgs = [m for m in messages if m.get("role") == "system"]
    conv_msgs = [m for m in messages if m.get("role") != "system"]

    if len(conv_msgs) <= keep:
        return messages

    recent = conv_msgs[-keep:]
    result = system_msgs + recent
    dropped = len(messages) - len(result)
    if dropped > 0:
        logger.info(f"Specialist context: kept {keep} recent messages, dropped {dropped}")
    return result


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

    cache_active = server.verify_prefix_cache()
    if not cache_active:
        logger.warning("Prefix cache not verified — responses may be slower")

    return {
        "cfg": cfg, "server": server, "router": router, "ctx": ctx,
        "memory": memory, "req_logger": req_logger, "verifier": verifier,
        "session_mgr": session_mgr,
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

    def _send_sse(self, content: str, model: str, finish_reason: str = "stop",
                  route: str = None, confidence: float = None, latency_ms: int = None,
                  tool_calls: list | None = None):
        """Send response as SSE stream (OpenAI streaming format).

        Since we get the full response from the model at once, this is
        'fake streaming' — entire content sent in one chunk. The client
        gets the SSE format it expects; content arrives after model finishes.
        """
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def write_chunk(delta: dict, fr=None):
            chunk = {
                "id": chunk_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": fr}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        # 1. Role delta
        write_chunk({"role": "assistant", "content": ""})

        # 2. Content delta (full content in one chunk)
        if content:
            write_chunk({"content": content})

        # 3. Tool calls delta (if any)
        if tool_calls:
            for tc in tool_calls:
                write_chunk({"tool_calls": [tc]})

        # 4. Final chunk with finish_reason
        write_chunk({}, fr=finish_reason)

        # 5. Done sentinel
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            models = []
            server = _app_state["server"]
            # "lore" = the routed endpoint (agent sends this, router decides model)
            if server.is_model_running("primary") or server.is_model_running("specialist"):
                models.append({
                    "id": "lore",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "lore",
                })
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
        """Handle multi-turn chat completions with model-aware routing.

        The agent sends the full conversation history each request. LORE routes
        to specialist (cheap ops) or primary (hard ops) based on the last user
        message. Tool calls are executed locally via the tool proxy.
        """
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
        request_tools = body.get("tools")
        repo_root = body.get("repo_root", ".")

        # Clamp max_tokens
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            max_tokens = 2048
        max_tokens = min(max_tokens, 8192)

        # Validate temperature
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            temperature = 0.7

        server = _app_state["server"]
        router = _app_state["router"]
        req_logger = _app_state["req_logger"]

        t0 = time.time()

        # Route the query
        if is_multimodal(user_msg):
            route, confidence, model = "MULTIMODAL", 1.0, "multimodal"
        else:
            route, confidence = router.classify(user_msg)
            model = "primary" if route == "PRIMARY" else "specialist"

        # TOOL_ONLY fast-path: regex/heuristic, no LLM call
        if route == "TOOL_ONLY":
            tool_result = handle_tool_only(user_msg)
            if tool_result is not None:
                latency_ms = int((time.time() - t0) * 1000)
                if stream:
                    self._send_sse(tool_result, "tool_handler", route=route,
                                   confidence=confidence, latency_ms=latency_ms)
                else:
                    self._send_json(200, self._build_response(
                        tool_result, route, confidence, "tool_handler", latency_ms))
                return

        # Build chat options
        chat_opts = {"max_tokens": max_tokens, "temperature": temperature}
        if json_mode:
            chat_opts["response_format"] = {"type": "json_object"}

        # Determine tools:
        # - Agent-defined tools: pass through, tool proxy executes by name
        # - No auto-injection: built-in tools only used when agent explicitly
        #   requests them. Auto-injecting confuses small models (Falcon-H1
        #   generates tool-call text instead of using OpenAI tool_calls format).
        tools = request_tools

        # Extract system prompt and history from incoming messages
        system_prompt = ""
        history_msgs = []
        for msg in messages:
            if msg.get("role") == "system":
                if system_prompt:
                    system_prompt += "\n" + (msg.get("content", "") or "")
                else:
                    system_prompt = msg.get("content", "") or ""
            else:
                history_msgs.append(msg)

        ctx = _app_state["ctx"]
        ctx.restore(system_prompt, history_msgs)

        # Budget context
        cfg = _app_state.get("cfg")
        if model == "specialist":
            specialist_ctx = cfg.models.get("specialist", {}).get("context", 131072) if cfg else 131072
            ctx.set_budget(int(specialist_ctx))
        else:
            orig_budget = cfg.context.get("working_context", 16384) if cfg else 16384
            sizing_cfg = {
                "default_budget": orig_budget,
                "min_budget": cfg.context.get("min_context_budget", 2048) if cfg else 2048,
                "max_budget": cfg.context.get("max_context_budget", 32768) if cfg else 32768,
            }
            budget = estimate_context_budget(route, user_msg, sizing_cfg)
            ctx.set_budget(budget)

        send_messages = ctx.build_prompt(query=user_msg)

        try:
            if tools:
                result = run_tool_loop(server, model, send_messages, tools=tools,
                                       repo_root=repo_root, ctx=ctx, **chat_opts)
            else:
                result = server.chat(model, send_messages, **chat_opts)
        except Exception as e:
            logger.warning(f"{model} failed: {e}")
            if model == "specialist":
                try:
                    # Fallback to primary
                    orig_budget = cfg.context.get("working_context", 16384) if cfg else 16384
                    sizing_cfg = {
                        "default_budget": orig_budget,
                        "min_budget": cfg.context.get("min_context_budget", 2048) if cfg else 2048,
                        "max_budget": cfg.context.get("max_context_budget", 32768) if cfg else 32768,
                    }
                    budget = estimate_context_budget("PRIMARY", user_msg, sizing_cfg)
                    ctx.set_budget(budget)
                    send_messages = ctx.build_prompt(query=user_msg)
                    
                    result = server.chat("primary", send_messages, **chat_opts)
                    model = "primary"
                except Exception as e2:
                    logger.error(f"Primary fallback failed: {e2}")
                    if stream:
                        self._send_sse(f"Error: {e2}", "primary", finish_reason="stop")
                    else:
                        self._send_json(500, {"error": f"Internal error: {e2}"})
                    return
            else:
                logger.error(f"Primary failed: {e}")
                if stream:
                    self._send_sse(f"Error: {e}", "primary", finish_reason="stop")
                else:
                    self._send_json(500, {"error": f"Internal error: {e}"})
                return

        latency_ms = int((time.time() - t0) * 1000)
        choice = result["choices"][0]
        content = choice["message"].get("content", "")
        finish_reason = choice.get("finish_reason", "stop")

        # Store turn in episodic memory
        if ctx._memory is not None:
            ctx._memory.store(user_msg, "user")
            if content:
                ctx._memory.store(content, "assistant")

        # Log request
        req_logger.log_request({
            "route": route,
            "confidence": confidence,
            "model": model,
            "tokens_out": ctx.token_count(content or ""),
            "latency_ms": latency_ms,
            "success": True,
        })

        tool_calls = choice["message"].get("tool_calls")

        if stream:
            self._send_sse(content or "", model, finish_reason=finish_reason,
                           route=route, confidence=confidence, latency_ms=latency_ms,
                           tool_calls=tool_calls)
        else:
            response = self._build_response(content, route, confidence, model, latency_ms,
                                            finish_reason=finish_reason,
                                            tool_calls=tool_calls)
            self._send_json(200, response)

    def _build_response(self, content: str, route: str, confidence: float,
                        model: str, latency_ms: int,
                        finish_reason: str = "stop",
                        tool_calls: list | None = None) -> dict:
        """Build an OpenAI-compatible chat completion response."""
        message = {"role": "assistant", "content": content or ""}
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        ctx = _app_state.get("ctx")
        if ctx is not None:
            token_count = ctx.token_count(content or "")
        else:
            token_count = len((content or "").split())
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": token_count,
                "completion_tokens": token_count,
                "total_tokens": token_count * 2,
            },
            "lore": {
                "route": route,
                "confidence": confidence,
                "latency_ms": latency_ms,
            },
        }


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
