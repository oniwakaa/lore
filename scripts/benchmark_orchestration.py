#!/usr/bin/env python3
"""Benchmark: orchestrated vs single-model on identical tasks.

Runs 10 tasks of varying complexity through two paths:
1. Orchestrated (complexity → decompose → execute → aggregate)
2. Single-model (direct _dispatch, one model call)

Measures: latency, token count, success rate, routing accuracy.

Self-contained: starts its own servers, waits for health, runs benchmark, stops servers.

Usage:
    PYTHONPATH=src python3 scripts/benchmark_orchestration.py
"""
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import requests
import yaml
from lore.config import LoreConfig
from lore.models import ModelServer
from lore.router import Router
from lore.context import ContextManager
from lore.memory import HierarchicalMemory
from lore.health import ContextHealth
from lore.logging import RequestLogger
from lore.tool_attention import ToolAttention
from lore.verifier import Verifier
from lore.orchestrator import Orchestrator
from lore.complexity import estimate as estimate_complexity

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


TASKS = [
    # Simple tasks (should NOT be orchestrated)
    {"query": "What is 2+2?", "expected_orchestrated": False, "category": "simple"},
    {"query": "Explain what a hash map is in one sentence", "expected_orchestrated": False, "category": "simple"},
    {"query": "Convert this to uppercase: hello world", "expected_orchestrated": False, "category": "simple"},
    {"query": "What is the time complexity of binary search?", "expected_orchestrated": False, "category": "simple"},
    {"query": "List 3 Python best practices", "expected_orchestrated": False, "category": "simple"},
    # Complex tasks (SHOULD be orchestrated)
    {"query": "Write a Python function to parse CSV files, add unit tests, and write a README", "expected_orchestrated": True, "category": "complex"},
    {"query": "Implement a REST API with authentication and then write integration tests and also document the endpoints", "expected_orchestrated": True, "category": "complex"},
    {"query": "Refactor this code to use the strategy pattern and add comprehensive tests and update the documentation", "expected_orchestrated": True, "category": "complex"},
    {"query": "Build a CLI tool that reads JSON config and validates it and generates a markdown report and also handles errors gracefully", "expected_orchestrated": True, "category": "complex"},
    {"query": "Design a database schema for a blog platform and write the migration scripts and create the API endpoints and add unit tests", "expected_orchestrated": True, "category": "complex"},
]


def setup():
    """Start servers and init all components. Self-contained."""
    cfg = LoreConfig.load()
    # ponytail: 262K/131K native context → OOM on 16 GB. 16K enough for benchmark.
    cfg.models["primary"]["context"] = 16384
    cfg.models["specialist"]["context"] = 16384
    cfg.models["defaults"]["context_size"] = 16384

    server = ModelServer(cfg.models)
    print("Starting model servers (Ornith-9B + Falcon-H1 + nomic-embed)...")
    server.start_all()

    for role, port in [("embeddings", 19002), ("specialist", 19001), ("primary", 19000)]:
        for attempt in range(90):
            try:
                r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            print(f"  [FAIL] {role} on port {port} did not start (90s timeout)")
            server.stop_all()
            sys.exit(1)
        print(f"  [OK] {role} healthy on port {port} ({attempt+1}s)")

    router = Router.load(
        cfg.router.get("model_path", "configs/router_model.joblib"),
        confidence_threshold=cfg.router.get("confidence_threshold", 0.70),
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
    verifier = Verifier()
    req_logger = RequestLogger()

    orch_cfg_path = Path("configs/orchestrator.yaml")
    orch_cfg = yaml.safe_load(orch_cfg_path.read_text()) if orch_cfg_path.exists() else {}
    orchestrator = Orchestrator(server, router, memory, orch_cfg,
                                ctx=ctx, req_logger=req_logger, verifier=verifier)

    return server, router, ctx, memory, orchestrator, verifier, req_logger


def run_orchestrated(orchestrator, query, dispatch_fn):
    """Run a query through the orchestrator. Returns metrics dict."""
    t0 = time.time()
    r = orchestrator.process(query, dispatch_fn=dispatch_fn)
    latency = (time.time() - t0) * 1000

    plan_subtasks = 0
    waves = 0
    if r.get("plan"):
        plan_subtasks = len(r["plan"].subtasks)
        # Reconstruct wave count from the plan's dependency graph
        waves = len(orchestrator._build_waves(r["plan"].subtasks))

    return {
        "orchestrated": r["orchestrated"],
        "subtasks_completed": r.get("subtasks_completed", 0),
        "total_latency_ms": latency,
        "content_length": len(r["content"]),
        "success": r["success"],
        "model": r["model"],
        "plan_subtasks": plan_subtasks,
        "waves": waves,
    }


def run_single_model(server, router, ctx, memory, req_logger, verifier, query):
    """Run a query through direct _dispatch (single-model path). Returns metrics dict."""
    from lore.cli import _dispatch
    t0 = time.time()
    r = _dispatch(query, server, router, ctx, memory, req_logger, json_mode=False, verifier=verifier)
    latency = (time.time() - t0) * 1000

    return {
        "orchestrated": False,
        "subtasks_completed": 0,
        "total_latency_ms": latency,
        "content_length": len(r["content"]),
        "success": r["success"],
        "model": r["model"],
        "plan_subtasks": 0,
        "waves": 0,
    }


def main():
    print("=" * 70)
    print("LORE Orchestration Benchmark — Orchestrated vs Single-Model")
    print("=" * 70)

    try:
        server, router, ctx, memory, orchestrator, verifier, req_logger = setup()
    except Exception as e:
        print(f"FAILED to start: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # dispatch_fn closure for orchestrator's simple-task path
    from lore.cli import _dispatch
    dispatch_fn = lambda q, json_mode=False: _dispatch(
        q, server, router, ctx, memory, req_logger, json_mode, verifier)

    results = []

    try:
        for i, task in enumerate(TASKS):
            query = task["query"]
            category = task["category"]
            expected = task["expected_orchestrated"]

            print(f"\n[{i+1}/{len(TASKS)}] ({category}) {query[:70]}...")

            # Complexity estimate (no LLM call, <1ms)
            route, conf = router.classify(query)
            est = estimate_complexity(query, route)
            print(f"  Route: {route} ({conf:.2f}) | Complex: {est.is_complex} (conf={est.confidence:.2f})")

            # Path 1: Orchestrated
            print(f"  Running orchestrated path...")
            orch_metrics = run_orchestrated(orchestrator, query, dispatch_fn)
            print(f"  → orchestrated={orch_metrics['orchestrated']} "
                  f"subtasks={orch_metrics['subtasks_completed']} "
                  f"latency={orch_metrics['total_latency_ms']:.0f}ms "
                  f"model={orch_metrics['model']}")

            # Path 2: Single-model (direct dispatch, fresh context per task)
            # ponytail: reset context for fair comparison — each task independent.
            ctx2 = ContextManager(
                config=ctx._config if hasattr(ctx, '_config') else {},
                model_server=server,
                system_prompt=ctx.system_prompt,
                memory=None, health=None,
            )
            print(f"  Running single-model path...")
            single_metrics = run_single_model(server, router, ctx2, memory, req_logger, verifier, query)
            print(f"  → model={single_metrics['model']} "
                  f"latency={single_metrics['total_latency_ms']:.0f}ms "
                  f"len={single_metrics['content_length']}")

            # Routing accuracy: did orchestration decision match expected?
            routing_correct = orch_metrics["orchestrated"] == expected

            results.append({
                "task_id": i + 1,
                "category": category,
                "query": query,
                "expected_orchestrated": expected,
                "route": route,
                "route_confidence": conf,
                "complexity_is_complex": est.is_complex,
                "complexity_confidence": est.confidence,
                "complexity_signals": est.signals,
                "orchestrated": orch_metrics,
                "single_model": single_metrics,
                "routing_correct": routing_correct,
            })

    except KeyboardInterrupt:
        print("\n\nInterrupted. Saving partial results...")
    finally:
        server.stop_all()

    # --- Summary ---

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Markdown table
    print(f"\n| # | Cat | Query (truncated) | Orch? | Subs | Waves | Orch ms | Single ms | Model | Route OK |")
    print(f"|---|-----|------------------|-------|------|-------|---------|-----------|-------|----------|")
    for r in results:
        q = r["query"][:40].replace("|", "\\|")
        o = r["orchestrated"]
        s = r["single_model"]
        ok = "✓" if r["routing_correct"] else "✗"
        print(f"| {r['task_id']} | {r['category']} | {q} | {o['orchestrated']} | "
              f"{o['subtasks_completed']} | {o['waves']} | "
              f"{o['total_latency_ms']:.0f} | {s['total_latency_ms']:.0f} | "
              f"{o['model']} | {ok} |")

    # Summary statistics
    simple = [r for r in results if r["category"] == "simple"]
    complex_ = [r for r in results if r["category"] == "complex"]

    print(f"\n--- Summary ---")
    if simple:
        orch_lat = [r["orchestrated"]["total_latency_ms"] for r in simple]
        single_lat = [r["single_model"]["total_latency_ms"] for r in simple]
        orch_count = sum(1 for r in simple if r["orchestrated"]["orchestrated"])
        print(f"Simple tasks ({len(simple)}):")
        print(f"  Avg orchestrated latency: {sum(orch_lat)/len(orch_lat):.0f}ms")
        print(f"  Avg single-model latency: {sum(single_lat)/len(single_lat):.0f}ms")
        print(f"  Orchestrated (should be 0): {orch_count}")

    if complex_:
        orch_lat = [r["orchestrated"]["total_latency_ms"] for r in complex_]
        single_lat = [r["single_model"]["total_latency_ms"] for r in complex_]
        sub_counts = [r["orchestrated"]["subtasks_completed"] for r in complex_]
        wave_counts = [r["orchestrated"]["waves"] for r in complex_]
        orch_count = sum(1 for r in complex_ if r["orchestrated"]["orchestrated"])
        print(f"Complex tasks ({len(complex_)}):")
        print(f"  Orchestrated: {orch_count}/{len(complex_)}")
        print(f"  Avg orchestrated latency: {sum(orch_lat)/len(orch_lat):.0f}ms")
        print(f"  Avg single-model latency: {sum(single_lat)/len(single_lat):.0f}ms")
        print(f"  Avg subtasks: {sum(sub_counts)/len(sub_counts):.1f}")
        print(f"  Avg waves: {sum(wave_counts)/len(wave_counts):.1f}")

    routing_accuracy = sum(1 for r in results if r["routing_correct"]) / len(results) if results else 0
    print(f"\nRouting accuracy: {routing_accuracy:.0%} ({sum(1 for r in results if r['routing_correct'])}/{len(results)})")

    # Save JSON results
    Path("benchmarks/results").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"benchmarks/results/orchestration_benchmark_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "task_count": len(results),
            "routing_accuracy": routing_accuracy,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
