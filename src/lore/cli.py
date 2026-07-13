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
from lore.orchestrator import Orchestrator
from lore.classifier import TaskClassifier
from lore.registry import ModelRegistry

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
    parser.add_argument("--api", action="store_true", help="Start OpenAI-compatible API server")
    parser.add_argument("--api-port", type=int, default=8000, help="API server port (default: 8000)")
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

    # Load orchestrator config + create orchestrator
    orch_cfg_path = Path("configs/orchestrator.yaml")
    orch_cfg = yaml.safe_load(orch_cfg_path.read_text()) if orch_cfg_path.exists() else {}

    # Model registry + classifier (Phase 4.2: live benchmark model selection)
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

    # Verify prefix cache
    cache_active = server.verify_prefix_cache()
    if not cache_active:
        logger.warning("Prefix cache not verified — responses may be slower")

    if args.api:
        from lore.api import create_server
        import lore.api as api_module
        api_module._app_state = {
            "cfg": cfg, "server": server, "router": router, "ctx": ctx,
            "memory": memory, "req_logger": req_logger, "verifier": verifier,
            "orchestrator": orchestrator, "session_mgr": session_mgr,
        }
        api_server = create_server("127.0.0.1", args.api_port)
        print(f"LORE API server listening on http://127.0.0.1:{args.api_port}")
        print(f"  POST /v1/chat/completions  — OpenAI-compatible chat")
        print(f"  GET  /v1/models            — list available models")
        print(f"  GET  /health               — health check")
        print(f"")
        print(f"Set base_url to http://127.0.0.1:{args.api_port}/v1 in your client.")
        try:
            api_server.serve_forever()
        except KeyboardInterrupt:
            print("Shutting down...")
            server.stop_all()
            api_server.shutdown()
    elif args.interactive:
        _run_repl(server, router, ctx, memory, req_logger, cfg, session_mgr, verifier, orchestrator, registry)
    elif args.query:
        _process_single(args.query, server, router, ctx, memory, req_logger, args.json, verifier, orchestrator)
    else:
        parser.print_help()
        server.stop_all()


def _resolve_route(query, router):
    """Classify query and determine model. Returns (route, confidence, model)."""
    if is_multimodal(query):
        return "MULTIMODAL", 1.0, "multimodal"
    route, confidence = router.classify(query)
    model = "primary" if route == "PRIMARY" else "specialist"
    return route, confidence, model


def _execute_query(query, model, server, ctx, memory, json_mode):
    """Execute model chat for a query. Returns (content, success).

    Handles specialist→primary fallback on failure. Does NOT handle
    TOOL_ONLY fast-path or multimodal — those are handled by _dispatch().

    Memory retrieval is owned by ContextManager.build_prompt() when
    ctx._memory is set and a query is provided — do not retrieve here
    to avoid duplicate embedding calls.
    """
    ctx.add_message("user", query)
    messages = ctx.build_prompt(query=query)

    opts = {}
    if json_mode:
        opts["response_format"] = {"type": "json_object"}

    try:
        result = server.chat(model, messages, max_tokens=2048, temperature=0.7, **opts)
        content = result["choices"][0]["message"]["content"]
        return content, True
    except Exception as e:
        if model == "specialist":
            logger.warning(f"Specialist failed, retrying on primary: {e}")
            result = server.chat("primary", messages, max_tokens=2048, temperature=0.7, **opts)
            content = result["choices"][0]["message"]["content"]
            return content, True
        return f"Error: {e}", False


def _post_dispatch(query, route, confidence, model, content, success,
                   latency, ctx, memory, req_logger, verifier, json_mode, tool_result):
    """Post-dispatch: verify output, store to memory, log request, cleanup multimodal."""
    tokens_out = len(content.split())

    if verifier is not None and tool_result is None:
        task_type = "json" if json_mode else "free_form"
        vresult = verifier.validate(content, task_type)
        if not vresult["valid"]:
            logger.warning(f"Verifier: invalid {task_type} output: {vresult['errors']}")
            if vresult["repaired"]:
                content = vresult["repaired"]
                logger.info("Verifier: repaired output")

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

    return content


def _dispatch(query, server, router, ctx, memory, req_logger, json_mode, verifier=None,
              route_info=None):
    """Route a query, execute it (tool fast-path or model chat), log, store to memory.

    Returns dict: route, confidence, model, content, success, latency_ms.
    Raises whatever server.swap_in() raises on multimodal failure — caller decides
    whether that means aborting (single-shot) or skipping this turn (REPL).

    If route_info is provided as (route, confidence, model), skip re-classification
    to avoid calling the router twice per request.
    """
    t0 = time.time()

    if route_info is not None:
        route, confidence, model = route_info
    else:
        route, confidence, model = _resolve_route(query, router)

    if route == "MULTIMODAL":
        server.swap_in("gemma-4-e4b")

    # Dynamic context sizing: per-request budget override
    try:
        sizing_cfg = {
            "default_budget": ctx._config.get("working_context", 16384),
            "min_budget": ctx._config.get("min_context_budget", 2048),
            "max_budget": ctx._config.get("max_context_budget", 32768),
        }
        ctx.set_budget(estimate_context_budget(route, query, sizing_cfg))
    except (TypeError, AttributeError, KeyError):
        pass

    # TOOL_ONLY fast-path: handle with regex/heuristics, skip LLM entirely
    tool_result = handle_tool_only(query) if route == "TOOL_ONLY" else None
    if tool_result is not None:
        content, success = tool_result, True
        model = "tool_handler"
        ctx.add_message("user", query)
    else:
        content, success = _execute_query(query, model, server, ctx, memory, json_mode)

    latency = (time.time() - t0) * 1000

    content = _post_dispatch(
        query, route, confidence, model, content, success,
        latency, ctx, memory, req_logger, verifier, json_mode, tool_result,
    )

    return {"route": route, "confidence": confidence, "model": model,
            "content": content, "success": success, "latency_ms": latency}


def _process_single(query, server, router, ctx, memory, req_logger, json_mode, verifier=None, orchestrator=None):
    """Process a single query."""
    try:
        if orchestrator is not None:
            dispatch_fn = lambda q, json_mode=False: _dispatch(q, server, router, ctx, memory, req_logger, json_mode, verifier)
            r = orchestrator.process(query, json_mode=json_mode, dispatch_fn=dispatch_fn)
        else:
            r = _dispatch(query, server, router, ctx, memory, req_logger, json_mode, verifier)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        server.stop_all()
        return

    print(f"[route: {r['route']} ({r['confidence']:.2f}) | {r['latency_ms']/1000:.1f}s]", file=sys.stderr)
    print(r["content"])
    server.stop_all()


class _ContextSnapshot:
    """Thread-safe snapshot of context state for background auto-save.

    Copies history and system_prompt at snapshot time so the background
    thread reads immutable data instead of racing with the main thread.
    """
    __slots__ = ("history", "system_prompt", "was_truncated")

    def __init__(self, history, system_prompt, was_truncated=False):
        self.history = history
        self.system_prompt = system_prompt
        self.was_truncated = was_truncated


def _run_repl(server, router, ctx, memory, req_logger, cfg, session_mgr=None, verifier=None, orchestrator=None, registry=None):
    """Interactive REPL mode."""
    print("LORE interactive mode. Type /exit to quit, /clear to reset, /route for last decision.")
    print("Session commands: /save [name], /resume <name>, /sessions")
    print("Model commands: /upgrades, /models")
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

        # Model commands
        if query == "/upgrades" and registry is not None:
            upgrades = registry.check_for_upgrades()
            if not upgrades:
                print("All installed models are current best for their tasks.")
            else:
                approved = registry.prompt_upgrade(upgrades)
                for u in approved:
                    print(f"Downloading {u.better_model.model_id}...")
                    ok = registry.approve_upgrade(u)
                    print(f"  {'OK' if ok else 'FAILED'}")
            continue
        if query == "/models" and registry is not None:
            print(f"  Orchestrator: {registry.orchestrator_model} (locked)")
            assignments = registry.select_workers()
            for task, assignment in assignments.items():
                marker = "auto" if assignment.auto_selected else "fallback"
                print(f"  {task:20s} -> {assignment.model_id:30s} "
                      f"score={assignment.benchmark_score:.1f} [{marker}]")
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
            if query.startswith("/switch"):
                parts = query.split(maxsplit=1)
                if len(parts) < 2:
                    # List active sessions
                    active = session_mgr.list_active_sessions()
                    if not active:
                        print("No active sessions. Use /save + /resume to create them.")
                    else:
                        for s in active:
                            cur = " (current)" if s["is_current"] else ""
                            print(f"  {s['session_id']:30s}  {s['turn_count']} turns{cur}")
                    continue
                target = session_mgr.switch_session(parts[1])
                if target is None:
                    print(f"Session '{parts[1]}' not active. Use /resume to load it first.")
                else:
                    ctx = target.context
                    memory = target.memory
                    if orchestrator is not None:
                        orchestrator.set_memory(memory)
                    print(f"Switched to session '{parts[1]}' ({len(target.context.history) // 2} turns).")
                continue

        # Process query (reuse single-shot dispatch logic but don't stop servers)
        try:
            if orchestrator is not None:
                dispatch_fn = lambda q, json_mode=False: _dispatch(q, server, router, ctx, memory, req_logger, json_mode, verifier)
                r = orchestrator.process(query, json_mode=False, dispatch_fn=dispatch_fn)
            else:
                r = _dispatch(query, server, router, ctx, memory, req_logger, json_mode=False, verifier=verifier)
        except Exception as e:
            print(f"Error: {e}")
            continue

        last_route = f"{r['route']} ({r['confidence']:.2f})"
        print(f"[{r['route']} ({r['confidence']:.2f}) | {r['latency_ms']:.0f}ms]")
        print(r["content"])
        print()

        turn_count += 1
        # Auto-save in background every N turns (thread-safe: snapshot before spawn)
        if session_mgr is not None and turn_count % auto_save_every == 0:
            try:
                import threading
                name = f"auto-{int(__import__('time').time())}"
                # Snapshot context to avoid race with main thread modifying ctx.history
                snapshot = _ContextSnapshot(
                    history=list(ctx.history),
                    system_prompt=ctx.system_prompt,
                    was_truncated=ctx.was_truncated,
                )
                threading.Thread(
                    target=session_mgr.save_session,
                    args=(name, server, snapshot),
                    daemon=True,
                ).start()
                logger.debug(f"Auto-saved session as '{name}'")
            except Exception as e:
                logger.warning(f"Auto-save failed: {e}")

    server.stop_all()
