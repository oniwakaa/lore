# src/lore/cli.py
"""CLI entry point. Single-shot and interactive modes."""
import re
import sys
import time
import hashlib
import logging
from pathlib import Path

import yaml

# Module-level imports so test patches (lore.cli.ModelServer, etc.) work.
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

logger = logging.getLogger(__name__)

# Multimodal detection patterns (structural, not classifier)
_IMAGE_EXTS = r"\.(png|jpg|jpeg|gif|webp|bmp|tiff?|svg)"
_AUDIO_EXTS = r"\.(wav|mp3|flac|ogg|aac|m4a)"
_MULTIMODAL_RE = re.compile(
    rf"(\b\w+{_IMAGE_EXTS}\b)|"  # image file paths
    rf"(https?://\S+{_IMAGE_EXTS})|"  # image URLs
    rf"(\b\w+{_AUDIO_EXTS}\b)|"  # audio file paths
    rf"(^/(image|audio)\b)",  # explicit /image or /audio prefix
    re.IGNORECASE,
)

def is_multimodal(text: str) -> bool:
    """Structural check: does input contain image/audio references?"""
    return bool(_MULTIMODAL_RE.search(text))


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(prog="lore", description="LORE orchestration engine")
    parser.add_argument("query", nargs="?", help="Single-shot query")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--json", action="store_true", help="JSON output mode (GBNF)")
    args = parser.parse_args()

    # Load config
    cfg = LoreConfig.load()

    # Start model servers
    server = ModelServer(cfg.models)
    logger.info("Starting model servers...")
    server.start_all()

    # Load router
    router_cfg = cfg.router
    router = Router.load(
        router_cfg.get("model_path", "configs/router_model.joblib"),
        confidence_threshold=router_cfg.get("confidence_threshold", 0.70),
    )

    # Init context manager + memory + logger
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

    # Verify prefix cache
    cache_active = server.verify_prefix_cache()
    if not cache_active:
        logger.warning("Prefix cache not verified — responses may be slower")

    if args.interactive:
        _run_repl(server, router, ctx, memory, req_logger, cfg, session_mgr, verifier)
    elif args.query:
        _process_single(args.query, server, router, ctx, memory, req_logger, args.json, verifier)
    else:
        parser.print_help()
        server.stop_all()


def _dispatch(query, server, router, ctx, memory, req_logger, json_mode, verifier=None):
    """Route a query, execute it (tool fast-path or model chat), log, store to memory.

    Returns dict: route, confidence, model, content, success, latency_ms.
    Raises whatever server.swap_in() raises on multimodal failure — caller decides
    whether that means aborting (single-shot) or skipping this turn (REPL).
    """
    t0 = time.time()

    if is_multimodal(query):
        server.swap_in("gemma-4-e4b")
        route, confidence, model = "MULTIMODAL", 1.0, "multimodal"
    else:
        route, confidence = router.classify(query)
        model = "primary" if route == "PRIMARY" else "specialist"

    # Dynamic context sizing: per-request budget override
    try:
        sizing_cfg = {
            "default_budget": ctx._config.get("working_context", 16384),
            "min_budget": ctx._config.get("min_context_budget", 2048),
            "max_budget": ctx._config.get("max_context_budget", 32768),
        }
        ctx._config["working_context"] = estimate_context_budget(route, query, sizing_cfg)
    except (TypeError, AttributeError, KeyError):
        pass  # ctx._config not dict-like (e.g. in tests with MagicMock)

    # TOOL_ONLY fast-path: handle with regex/heuristics, skip LLM entirely
    tool_result = handle_tool_only(query) if route == "TOOL_ONLY" else None
    if tool_result is not None:
        content, success, model = tool_result, True, "tool_handler"
        ctx.add_message("user", query)
    else:
        memories = memory.retrieve(query) if model != "multimodal" else []
        ctx.add_message("user", query)
        messages = ctx.build_prompt(memories=memories, query=query)

        opts = {}
        if json_mode:
            opts["response_format"] = {"type": "json_object"}

        try:
            result = server.chat(model, messages, max_tokens=2048, temperature=0.7, **opts)
            content = result["choices"][0]["message"]["content"]
            success = True
        except Exception as e:
            if model == "specialist":
                logger.warning(f"Specialist failed, retrying on primary: {e}")
                result = server.chat("primary", messages, max_tokens=2048, temperature=0.7, **opts)
                content = result["choices"][0]["message"]["content"]
                success = True
            else:
                content = f"Error: {e}"
                success = False

    latency = (time.time() - t0) * 1000
    tokens_out = len(content.split())  # rough estimate

    # Validate and attempt repair for structured outputs
    repair_used = False
    if verifier is not None and tool_result is None:
        task_type = "json" if json_mode else "free_form"
        vresult = verifier.validate(content, task_type)
        if not vresult["valid"]:
            logger.warning(f"Verifier: invalid {task_type} output: {vresult['errors']}")
            if vresult["repaired"]:
                content = vresult["repaired"]
                repair_used = True
                logger.info("Verifier: repaired output")

    # Store in memory
    memory.store(query, "user")
    memory.store(content, "assistant")
    ctx.add_message("assistant", content)

    req_logger.log_request({
        "input_hash": f"sha256:{hashlib.sha256(query.encode()).hexdigest()[:16]}",
        "route": route,
        "confidence": confidence,
        "model": model,
        "tokens_out": tokens_out,
        "latency_ms": int(latency),
        "success": success,
        "fallback": model == "primary" and route == "SPECIALIST",
        "context_truncated": ctx.was_truncated,
        "cache_hit": None,
        "error": None if success else content,
    })

    if route == "MULTIMODAL":
        server.swap_out("gemma-4-e4b")

    return {"route": route, "confidence": confidence, "model": model,
            "content": content, "success": success, "latency_ms": latency}


def _process_single(query, server, router, ctx, memory, req_logger, json_mode, verifier=None):
    """Process a single query."""
    try:
        r = _dispatch(query, server, router, ctx, memory, req_logger, json_mode, verifier)
    except Exception as e:
        print(f"Error: multimodal unavailable ({e})", file=sys.stderr)
        server.stop_all()
        return

    print(f"[route: {r['route']} ({r['confidence']:.2f}) | {r['latency_ms']:.1f}s]", file=sys.stderr)
    print(r["content"])
    server.stop_all()


def _run_repl(server, router, ctx, memory, req_logger, cfg, session_mgr=None, verifier=None):
    """Interactive REPL mode."""
    print("LORE interactive mode. Type /exit to quit, /clear to reset, /route for last decision.")
    print("Session commands: /save [name], /resume <name>, /sessions")
    last_route = None
    turn_count = 0
    auto_save_every = cfg.session.get("auto_save_every_n_turns", 10) if hasattr(cfg, "session") else 10

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query == "/exit":
            break
        if query == "/clear":
            ctx.clear()
            memory.clear()
            print("Cleared.")
            continue
        if query == "/route":
            print(f"Last route: {last_route}")
            continue

        # Session commands
        if session_mgr is not None:
            if query.startswith("/save"):
                parts = query.split(maxsplit=1)
                name = parts[1] if len(parts) > 1 else f"session-{int(__import__('time').time())}"
                session_mgr.save_session(name, server, ctx)
                print(f"Saved session '{name}'.")
                continue
            if query.startswith("/resume"):
                parts = query.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: /resume <name>")
                    continue
                ok = session_mgr.resume_session(parts[1], server, ctx)
                print(f"Resumed '{parts[1]}'." if ok else f"Session '{parts[1]}' not found.")
                continue
            if query == "/sessions":
                sessions = session_mgr.list_sessions()
                if not sessions:
                    print("No saved sessions.")
                else:
                    for s in sessions:
                        print(f"  {s['session_id']:30s}  {s['turn_count']} turns  {s.get('topic', '')[:40]}")
                continue

        # Process query (reuse single-shot dispatch logic but don't stop servers)
        try:
            r = _dispatch(query, server, router, ctx, memory, req_logger, json_mode=False, verifier=verifier)
        except Exception as e:
            print(f"Multimodal unavailable: {e}")
            continue

        last_route = f"{r['route']} ({r['confidence']:.2f})"
        print(f"[{r['route']} ({r['confidence']:.2f}) | {r['latency_ms']:.0f}ms]")
        print(r["content"])
        print()

        turn_count += 1
        # Auto-save in background every N turns
        if session_mgr is not None and turn_count % auto_save_every == 0:
            try:
                import threading
                name = f"auto-{int(__import__('time').time())}"
                threading.Thread(
                    target=session_mgr.save_session,
                    args=(name, server, ctx),
                    daemon=True,
                ).start()
                logger.debug(f"Auto-saved session as '{name}'")
            except Exception as e:
                logger.warning(f"Auto-save failed: {e}")

    server.stop_all()
