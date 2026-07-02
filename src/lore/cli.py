# src/lore/cli.py
"""CLI entry point. Single-shot and interactive modes."""
import re
import sys
import time
import hashlib
import logging
from pathlib import Path

# Module-level imports so test patches (lore.cli.ModelServer, etc.) work.
from lore.config import LoreConfig
from lore.models import ModelServer
from lore.router import Router
from lore.context import ContextManager
from lore.memory import EpisodicMemory
from lore.logging import RequestLogger

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
    ctx = ContextManager(cfg.context, server, system_prompt=system_prompt)
    memory = EpisodicMemory(cfg.memory, server)
    req_logger = RequestLogger()

    # Verify prefix cache
    cache_active = server.verify_prefix_cache()
    if not cache_active:
        logger.warning("Prefix cache not verified — responses may be slower")

    if args.interactive:
        _run_repl(server, router, ctx, memory, req_logger, cfg)
    elif args.query:
        _process_single(args.query, server, router, ctx, memory, req_logger, args.json)
    else:
        parser.print_help()
        server.stop_all()


def _process_single(query, server, router, ctx, memory, req_logger, json_mode):
    """Process a single query."""
    t0 = time.time()

    # Multimodal pre-check
    if is_multimodal(query):
        try:
            server.swap_in("gemma-4-e4b")
            route, confidence = "MULTIMODAL", 1.0
            model = "multimodal"
        except Exception as e:
            print(f"Error: multimodal unavailable ({e})", file=sys.stderr)
            server.stop_all()
            return
    else:
        route, confidence = router.classify(query)
        # TODO Phase 2: TOOL_ONLY should skip LLM entirely (regex/parser fast-path)
        model = "primary" if route == "PRIMARY" else "specialist"

    # Build context with memories
    memories = memory.retrieve(query) if model != "multimodal" else []
    ctx.add_message("user", query)
    messages = ctx.build_prompt(memories=memories)

    # Dispatch
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

    # Store in memory
    memory.store(query, "user")
    memory.store(content, "assistant")
    ctx.add_message("assistant", content)

    # Log
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

    # Output
    print(f"[route: {route} ({confidence:.2f}) | {latency:.1f}s]", file=sys.stderr)
    print(content)

    # Swap out multimodal if loaded
    if route == "MULTIMODAL":
        server.swap_out("gemma-4-e4b")

    server.stop_all()


def _run_repl(server, router, ctx, memory, req_logger, cfg):
    """Interactive REPL mode."""
    print("LORE interactive mode. Type /exit to quit, /clear to reset, /route for last decision.")
    last_route = None

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

        # Process query (reuse single-shot logic but don't stop servers)
        t0 = time.time()

        if is_multimodal(query):
            try:
                server.swap_in("gemma-4-e4b")
                route, confidence = "MULTIMODAL", 1.0
                model = "multimodal"
            except Exception as e:
                print(f"Multimodal unavailable: {e}")
                continue
        else:
            route, confidence = router.classify(query)
            # TODO Phase 2: TOOL_ONLY should skip LLM entirely (regex/parser fast-path)
            model = "primary" if route == "PRIMARY" else "specialist"

        last_route = f"{route} ({confidence:.2f})"

        memories = memory.retrieve(query) if model != "multimodal" else []
        ctx.add_message("user", query)
        messages = ctx.build_prompt(memories=memories)

        try:
            result = server.chat(model, messages, max_tokens=2048, temperature=0.7)
            content = result["choices"][0]["message"]["content"]
        except Exception as e:
            if model == "specialist":
                result = server.chat("primary", messages, max_tokens=2048, temperature=0.7)
                content = result["choices"][0]["message"]["content"]
            else:
                content = f"Error: {e}"

        latency = (time.time() - t0) * 1000

        memory.store(query, "user")
        memory.store(content, "assistant")
        ctx.add_message("assistant", content)

        req_logger.log_request({
            "input_hash": f"sha256:{hashlib.sha256(query.encode()).hexdigest()[:16]}",
            "route": route,
            "confidence": confidence,
            "model": model,
            "latency_ms": int(latency),
            "success": not content.startswith("Error:"),
        })

        print(f"[{route} ({confidence:.2f}) | {latency:.0f}ms]")
        print(content)
        print()

        if route == "MULTIMODAL":
            server.swap_out("gemma-4-e4b")

    server.stop_all()
