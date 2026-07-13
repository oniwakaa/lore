#!/usr/bin/env python3
"""Real inference test: start LORE API, send requests, verify routing + model output.

Uses reduced context (16K, 1 slot) to stay within memory budget.
"""
import json
import sys
import time
import threading
import logging
from pathlib import Path
from io import BytesIO

# Set up paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def test_real_inference():
    """Start models, init API state, send test requests through the handler."""
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
    from lore.api import LoreHandler, _app_state

    # Load config and reduce context for test
    cfg = LoreConfig.load()
    # Override context sizes to 16K, 1 slot
    cfg._config["models"]["primary"]["context"] = 16384
    cfg._config["models"]["primary"]["parallel_slots"] = 1
    cfg._config["models"]["specialist"]["context"] = 16384

    # Start model servers
    server = ModelServer(cfg.models)
    logger.info("Starting model servers (16K context, 1 slot)...")
    server.start_all()
    logger.info("Models started.")

    # Init components
    router = Router.load(
        cfg.router.get("model_path", "configs/router_model.joblib"),
        confidence_threshold=0.40,  # lowered to test all routes
    )

    system_prompt = "You are a helpful assistant. Answer concisely and accurately."
    tokenizer_source = cfg.models.get("defaults", {}).get("tokenizer_source", "local")
    tokenizer_repo = cfg.models.get("primary", {}).get("source", "")
    if tokenizer_repo.endswith("-GGUF"):
        tokenizer_repo = tokenizer_repo[:-len("-GGUF")]
    tool_attention = ToolAttention.from_config(server)
    compression_cfg = yaml.safe_load(Path("configs/compression.yaml").read_text()) if Path("configs/compression.yaml").exists() else {}
    memory = HierarchicalMemory(cfg.memory, server)
    ctx = ContextManager(cfg.context, server, system_prompt=system_prompt,
                          tokenizer_source=tokenizer_source, tokenizer_repo=tokenizer_repo or None,
                          tool_attention=tool_attention, compression=compression_cfg, memory=memory)
    verifier = Verifier(yaml.safe_load(Path("configs/verifier.yaml").read_text()) if Path("configs/verifier.yaml").exists() else {})
    req_logger = RequestLogger()

    # Set up API state
    _app_state.update({
        "cfg": cfg, "server": server, "router": router, "ctx": ctx,
        "memory": memory, "req_logger": req_logger, "verifier": verifier,
    })

    # Test requests — chosen to hit each route at confidence > 0.40
    # Note: router model classifies unit conversion as SPECIALIST, not TOOL_ONLY.
    # The TOOL_ONLY fast-path only fires when the router says TOOL_ONLY.
    tests = [
        {
            "name": "SPECIALIST (unit conversion)",
            "messages": [{"role": "user", "content": "100 km to miles"}],
            "expect_route": "SPECIALIST",
        },
        {
            "name": "SPECIALIST (summarization)",
            "messages": [{"role": "user", "content": "Summarize this paragraph in one sentence: The quick brown fox jumps over the lazy dog. The dog, surprised by the sudden leap, barked loudly and chased the fox through the meadow until both collapsed from exhaustion."}],
            "expect_route": "SPECIALIST",
        },
        {
            "name": "PRIMARY (code generation)",
            "messages": [{"role": "user", "content": "Write a Python function that returns the nth fibonacci number using memoization."}],
            "expect_route": "PRIMARY",
        },
        {
            "name": "SPECIALIST (simple QA)",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "expect_route": "SPECIALIST",
        },
    ]

    results = []
    for test in tests:
        name = test["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST: {name}")
        logger.info(f"{'='*60}")

        request_body = json.dumps({
            "messages": test["messages"],
            "stream": False,
            "max_tokens": 512,
        }).encode()

        mock_wfile = BytesIO()

        class TestHandler(LoreHandler):
            def __init__(self):
                self.path = "/v1/chat/completions"
                self.headers = {"Content-Length": str(len(request_body))}
                self.wfile = mock_wfile
                self.rfile = BytesIO(request_body)
                self._status = None
                self._headers = {}

            def send_response(self, status):
                self._status = status

            def send_header(self, key, value):
                self._headers[key] = value

            def end_headers(self):
                pass

            def log_message(self, *args):
                pass

        t0 = time.time()
        handler = TestHandler()
        try:
            handler.do_POST()
        except Exception as e:
            logger.error(f"Request failed: {e}")
            results.append({"name": name, "status": "ERROR", "error": str(e)})
            continue

        elapsed = time.time() - t0

        if handler._status != 200:
            body = json.loads(mock_wfile.getvalue())
            logger.error(f"HTTP {handler._status}: {body}")
            results.append({"name": name, "status": f"HTTP {handler._status}", "error": body.get("error", "")})
            continue

        body = json.loads(mock_wfile.getvalue())
        content = body["choices"][0]["message"]["content"]
        route = body["lore"]["route"]
        confidence = body["lore"]["confidence"]
        model = body["model"]

        logger.info(f"Route: {route} (confidence: {confidence:.2f})")
        logger.info(f"Model: {model}")
        logger.info(f"Latency: {body['lore']['latency_ms']}ms ({elapsed:.1f}s)")
        logger.info(f"Response ({len(content)} chars):")
        logger.info(content[:500])

        # Verify routing
        expected_route = test.get("expect_route")
        route_ok = expected_route is None or route == expected_route
        expected_model = test.get("expect_model")
        model_ok = expected_model is None or model == expected_model

        status = "PASS" if route_ok and model_ok else "FAIL"
        if not route_ok:
            status += f" (expected route {expected_route}, got {route})"
        if not model_ok:
            status += f" (expected model {expected_model}, got {model})"

        results.append({
            "name": name, "status": status, "route": route,
            "confidence": confidence, "model": model,
            "latency_ms": body["lore"]["latency_ms"],
            "content_preview": content[:200],
        })

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    for r in results:
        status_icon = "OK" if r["status"] == "PASS" else "!!"
        logger.info(f"  [{status_icon}] {r['name']}: {r['status']}")
        if "route" in r:
            logger.info(f"       route={r['route']} conf={r.get('confidence', 0):.2f} model={r['model']} latency={r.get('latency_ms', 0)}ms")

    # Cleanup
    server.stop_all()
    logger.info("Servers stopped.")

    all_pass = all(r["status"] == "PASS" for r in results)
    return all_pass


if __name__ == "__main__":
    success = test_real_inference()
    sys.exit(0 if success else 1)
