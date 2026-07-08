#!/usr/bin/env python3
"""Orchestration A/B benchmark: direct (single 9B call) vs orchestrated.

Runs 10 tasks through both paths, measures latency/tokens/correctness,
saves JSON to benchmarks/results/orchestration_ab.json.

Usage: PYTHONPATH=src python scripts/benchmark_orchestration.py
       PYTHONPATH=src python scripts/benchmark_orchestration.py --quick
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lore.config import LoreConfig
from lore.context import ContextManager
from lore.memory import HierarchicalMemory
from lore.models import ModelServer
from lore.classifier import TaskClassifier
from lore.orchestrator import Orchestrator
from lore.router import Router

TASKS_PATH = ROOT / "benchmarks/eval_tasks/orchestration_ab.json"
RESULTS_PATH = ROOT / "benchmarks/results/orchestration_ab.json"

PRIMARY_PORT = 19000
SPECIALIST_PORT = 19001
EMBED_PORT = 19002
TIMEOUT_S = 120

logger = logging.getLogger("bench_orch")

# Embedded fallback if file missing
_FALLBACK_TASKS = [
    {"id": "simple-1", "complexity": "simple",
     "prompt": "What is 247 * 83? Reply with just the number."},
    {"id": "simple-2", "complexity": "simple",
     "prompt": "Is the string '{\"a\": 1}' valid JSON? Answer yes or no."},
    {"id": "simple-3", "complexity": "simple",
     "prompt": "What year did the Apollo 11 moon landing happen?"},
    {"id": "complex-1", "complexity": "complex",
     "prompt": "Write a Python FastAPI app with: 1) a POST /users endpoint that validates email and returns 201, 2) a GET /users/{id} endpoint, 3) an in-memory store, 4) input validation with Pydantic. Include all imports and make it runnable."},
    {"id": "complex-2", "complexity": "complex",
     "prompt": "Create a Python CLI tool that: 1) reads a CSV file, 2) validates each row against a schema, 3) writes valid rows to output.csv, 4) writes invalid rows to errors.jsonl with row number and error details, 5) prints a summary. Use argparse, handle edge cases."},
    {"id": "complex-3", "complexity": "complex",
     "prompt": "Write a complete Python test suite for a Stack class that has push, pop, peek, is_empty, and size methods. Include: 1) basic operations, 2) edge cases (empty stack, single element), 3) LIFO ordering, 4) type hints, 5) at least 10 test methods using pytest."},
    {"id": "complex-4", "complexity": "complex",
     "prompt": "Design and implement a Python rate limiter with: 1) sliding window algorithm, 2) per-user tracking, 3) configurable max requests and window size, 4) a decorator interface, 5) thread safety, 6) cleanup of expired windows. Include docstrings and type hints."},
    {"id": "complex-5", "complexity": "complex",
     "prompt": "Write a Python script that: 1) connects to a SQLite database, 2) creates tables for users, posts, and comments with proper foreign keys, 3) seeds sample data, 4) implements functions for: get user with post count, get posts with comment count, search posts by keyword, 5) includes a simple CLI interface."},
    {"id": "complex-6", "complexity": "complex",
     "prompt": "Build a Python context manager that: 1) manages database transactions with commit/rollback, 2) supports nested savepoints, 3) logs all SQL operations, 4) measures execution time, 5) raises a custom TransactionError on rollback. Include full implementation with type hints and usage example."},
    {"id": "complex-7", "complexity": "complex",
     "prompt": "Create a Python module for a simple task queue: 1) Task dataclass with id, payload, status, created_at, 2) TaskQueue class with enqueue, dequeue, peek, size, 3) retry logic with configurable max retries and backoff, 4) dead letter queue for permanently failed tasks, 5) persistence to JSON file, 6) thread safety. Include comprehensive docstrings."},
]

# Expected content checks
_CORRECTNESS = {
    "simple-1": lambda c: "20501" in c,
    "simple-2": lambda c: "yes" in c.lower(),
    "simple-3": lambda c: "1969" in c,
}


def load_tasks() -> list[dict]:
    if TASKS_PATH.exists():
        data = json.loads(TASKS_PATH.read_text())
        return data.get("tasks", _FALLBACK_TASKS)
    return _FALLBACK_TASKS


def is_healthy(port: int) -> bool:
    try:
        return requests.get(f"http://127.0.0.1:{port}/health", timeout=3).status_code == 200
    except Exception:
        return False


def ensure_servers(server: ModelServer) -> dict[str, bool]:
    """Check existing servers; start any that aren't running. Returns status dict."""
    status = {
        "primary": is_healthy(PRIMARY_PORT),
        "specialist": is_healthy(SPECIALIST_PORT),
        "embeddings": is_healthy(EMBED_PORT),
    }
    for role in ("primary", "specialist", "embeddings"):
        if not status[role]:
            try:
                logger.info(f"Starting {role} server (not detected on port)...")
                server.start_model(role)
                port = {"primary": PRIMARY_PORT, "specialist": SPECIALIST_PORT, "embeddings": EMBED_PORT}[role]
                status[role] = is_healthy(port)
            except Exception as e:
                logger.error(f"Failed to start {role}: {e}")
                status[role] = False
    return status


def run_direct(task: dict, server: ModelServer) -> dict:
    """Variant A: single chat completion to primary model. Baseline."""
    t0 = time.time()
    try:
        resp = server.chat("primary",
                           [{"role": "user", "content": task["prompt"]}],
                           max_tokens=4096, temperature=0.3, timeout=TIMEOUT_S)
        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        return {
            "wall_clock_s": time.time() - t0,
            "tokens_in": usage.get("prompt_tokens", 0),
            "tokens_out": usage.get("completion_tokens", len(content.split())),
            "success": True,
            "orchestrated": False,
            "subtasks_count": 0,
            "model_used": "primary",
            "content": content,
        }
    except Exception as e:
        return {
            "wall_clock_s": time.time() - t0,
            "tokens_in": 0, "tokens_out": 0,
            "success": False, "orchestrated": False,
            "subtasks_count": 0, "model_used": "primary",
            "content": f"ERROR: {e}",
        }


def run_orchestrated(task: dict, orchestrator: Orchestrator, dispatch_fn) -> dict:
    """Variant B: full orchestration pipeline (decompose/schedule/aggregate)."""
    t0 = time.time()
    try:
        r = orchestrator.process(task["prompt"], json_mode=False, dispatch_fn=dispatch_fn)
        content = r.get("content", "")
        return {
            "wall_clock_s": time.time() - t0,
            "tokens_in": 0,  # orchestrator doesn't aggregate usage; estimate from content
            "tokens_out": len(content.split()),
            "success": bool(r.get("success", False)),
            "orchestrated": bool(r.get("orchestrated", False)),
            "subtasks_count": int(r.get("subtasks_completed", 0)),
            "model_used": r.get("model", "unknown"),
            "content": content,
        }
    except Exception as e:
        return {
            "wall_clock_s": time.time() - t0,
            "tokens_in": 0, "tokens_out": 0,
            "success": False, "orchestrated": False,
            "subtasks_count": 0, "model_used": "unknown",
            "content": f"ERROR: {e}",
        }


def check_correctness(task_id: str, content: str) -> bool | None:
    fn = _CORRECTNESS.get(task_id)
    if fn is not None:
        return fn(content)
    if task_id.startswith("complex-"):
        return "```" in content and len(content) > 200
    return None


def build_orchestrator(server: ModelServer) -> Orchestrator:
    """Wire Orchestrator like cli.py does — minimal, no verifier/registry extras."""
    cfg = LoreConfig.load()
    router = Router.load(
        cfg.router.get("model_path", "configs/router_model.joblib"),
        confidence_threshold=cfg.router.get("confidence_threshold", 0.70),
    )
    system_prompt = "You are a helpful assistant. Answer concisely and accurately."
    tokenizer_source = cfg.models.get("defaults", {}).get("tokenizer_source", "local")
    tokenizer_repo = cfg.models.get("primary", {}).get("source", "")
    if tokenizer_repo.endswith("-GGUF"):
        tokenizer_repo = tokenizer_repo[:-len("-GGUF")]
    memory = HierarchicalMemory(cfg.memory, server)
    ctx = ContextManager(cfg.context, server, system_prompt=system_prompt,
                         tokenizer_source=tokenizer_source,
                         tokenizer_repo=tokenizer_repo or None,
                         memory=memory)
    orch_cfg_path = ROOT / "configs/orchestrator.yaml"
    orch_cfg = yaml.safe_load(orch_cfg_path.read_text()) if orch_cfg_path.exists() else {}
    classifier_cfg = orch_cfg.get("classifier", {})
    classifier = TaskClassifier(server, classifier_cfg) if classifier_cfg.get("enabled", False) else None
    return Orchestrator(server, router, memory, orch_cfg, ctx=ctx,
                        classifier=classifier)


def make_dispatch_fn(server, router, ctx, memory):
    """Closure matching cli.py _dispatch signature (minimal subset)."""
    def dispatch_fn(query, json_mode=False):
        t0 = time.time()
        try:
            route, confidence = router.classify(query)
            model = "primary" if route == "PRIMARY" else "specialist"
        except Exception:
            route, confidence, model = "PRIMARY", 0.0, "primary"
        ctx.add_message("user", query)
        messages = ctx.build_prompt(query=query)
        try:
            result = server.chat(model, messages, max_tokens=2048, temperature=0.7,
                                 timeout=TIMEOUT_S)
            content = result["choices"][0]["message"]["content"]
            success = True
        except Exception as e:
            if model == "specialist":
                result = server.chat("primary", messages, max_tokens=2048,
                                     temperature=0.7, timeout=TIMEOUT_S)
                content = result["choices"][0]["message"]["content"]
                success = True
            else:
                content = f"Error: {e}"
                success = False
        ctx.add_message("assistant", content)
        return {"route": route, "confidence": confidence, "model": model,
                "content": content, "success": success,
                "latency_ms": (time.time() - t0) * 1000}
    return dispatch_fn


def fmt_pct(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.0f}%"


def print_table(results: dict) -> None:
    print("\n════════════════════════════════════════════════════════════════")
    print(" ORCHESTRATION A/B BENCHMARK")
    print("════════════════════════════════════════════════════════════════")
    print(f"\n{'Task':<18}│{'Direct (s)':>12}│{'Orch (s)':>12}│{'Δ (%)':>10}│ Winner")
    print("─" * 18 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 10 + "┼" + "─" * 12)

    direct_wins_simple = 0
    orch_wins_simple = 0
    direct_wins_complex = 0
    orch_wins_complex = 0
    direct_correct = 0
    orch_correct = 0
    complex_direct_total_s = 0.0
    complex_orch_total_s = 0.0
    complex_count = 0

    for tid in [t["id"] for t in load_tasks()]:
        d = results["direct"].get(tid, {})
        o = results["orchestrated"].get(tid, {})
        dt = d.get("wall_clock_s", 0)
        ot = o.get("wall_clock_s", 0)
        if dt > 0:
            delta = (ot - dt) / dt * 100
        else:
            delta = 0
        is_complex = tid.startswith("complex-")
        winner = "—"
        if d.get("success") and o.get("success"):
            if ot < dt:
                winner = "Orch ✓"
                if is_complex:
                    orch_wins_complex += 1
                else:
                    orch_wins_simple += 1
            else:
                winner = "Direct"
                if is_complex:
                    direct_wins_complex += 1
                else:
                    direct_wins_simple += 1
        if d.get("correct") is True:
            direct_correct += 1
        if o.get("correct") is True:
            orch_correct += 1
        if is_complex:
            complex_direct_total_s += dt
            complex_orch_total_s += ot
            complex_count += 1
        print(f"{tid:<18}│{dt:>12.1f}│{ot:>12.1f}│{fmt_pct(delta):>10}│ {winner}")

    print("\n─── SUMMARY ────────────────────────────────────────────────────")
    simple_total = sum(1 for t in load_tasks() if t["complexity"] == "simple")
    complex_total = sum(1 for t in load_tasks() if t["complexity"] == "complex")
    print(f"Simple tasks:  Direct wins {direct_wins_simple}/{simple_total} (orch overhead wasted)")
    print(f"Complex tasks: Orch wins {orch_wins_complex}/{complex_total} (decomposition pays off)")
    if complex_count and complex_direct_total_s > 0:
        avg_saved = (complex_direct_total_s - complex_orch_total_s) / complex_direct_total_s * 100
        print(f"Overall:       Orchestration saves {avg_saved:.0f}% avg latency on complex tasks")
    else:
        print("Overall:       (no complex tasks completed)")
    print(f"Correctness:   Direct {direct_correct}/10  |  Orchestrated {orch_correct}/10")
    print("═════════════════════════════════════════════════════════════════\n")


def main():
    parser = argparse.ArgumentParser(description="Orchestration A/B benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Run 3 tasks only (1 simple + 2 complex) for smoke test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    tasks = load_tasks()
    if args.quick:
        simple = [t for t in tasks if t["complexity"] == "simple"][:1]
        complex_ = [t for t in tasks if t["complexity"] == "complex"][:2]
        tasks = simple + complex_

    cfg = LoreConfig.load()
    server = ModelServer(cfg.models)

    print("Checking servers...")
    status = ensure_servers(server)
    for role, ok in status.items():
        tag = "OK" if ok else "MISSING"
        print(f"  {role}: {tag}")
    if not status["primary"]:
        print("Primary server unavailable — cannot run benchmark.", file=sys.stderr)
        sys.exit(1)

    orchestrator = build_orchestrator(server)
    router = Router.load(
        cfg.router.get("model_path", "configs/router_model.joblib"),
        confidence_threshold=cfg.router.get("confidence_threshold", 0.70),
    )
    system_prompt = "You are a helpful assistant. Answer concisely and accurately."
    tokenizer_source = cfg.models.get("defaults", {}).get("tokenizer_source", "local")
    tokenizer_repo = cfg.models.get("primary", {}).get("source", "")
    if tokenizer_repo.endswith("-GGUF"):
        tokenizer_repo = tokenizer_repo[:-len("-GGUF")]
    memory = HierarchicalMemory(cfg.memory, server)
    ctx = ContextManager(cfg.context, server, system_prompt=system_prompt,
                         tokenizer_source=tokenizer_source,
                         tokenizer_repo=tokenizer_repo or None,
                         memory=memory)
    dispatch_fn = make_dispatch_fn(server, router, ctx, memory)

    results = {"direct": {}, "orchestrated": {}, "tasks": [t["id"] for t in tasks]}

    print(f"\nRunning {len(tasks)} tasks × 2 variants = {len(tasks) * 2} runs...\n")

    # Variant A: direct
    print("=== Variant A: Direct (single 9B call) ===")
    for t in tasks:
        print(f"  [{t['id']}] ", end="", flush=True)
        r = run_direct(t, server)
        r["correct"] = check_correctness(t["id"], r["content"])
        results["direct"][t["id"]] = r
        print(f"{r['wall_clock_s']:.1f}s ok={r['success']} correct={r['correct']}")

    # Variant B: orchestrated
    print("\n=== Variant B: Orchestrated ===")
    for t in tasks:
        print(f"  [{t['id']}] ", end="", flush=True)
        r = run_orchestrated(t, orchestrator, dispatch_fn)
        r["correct"] = check_correctness(t["id"], r["content"])
        results["orchestrated"][t["id"]] = r
        print(f"{r['wall_clock_s']:.1f}s ok={r['success']} orch={r['orchestrated']} "
              f"subtasks={r['subtasks_count']} correct={r['correct']}")

    # Save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {RESULTS_PATH}")

    print_table(results)
    print("Servers left running. Kill manually if done:")
    for role, port in [("primary", PRIMARY_PORT), ("specialist", SPECIALIST_PORT),
                       ("embeddings", EMBED_PORT)]:
        print(f"  {role}: port {port}")


if __name__ == "__main__":
    main()
