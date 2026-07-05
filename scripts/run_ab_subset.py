#!/usr/bin/env python3
"""Fast Phase 3 evaluation: 5-task subset, 4 variants, ~5 min total.

Tests the same Phase 3 features as run_ab_realistic.py but with a
representative 5-task subset instead of 50, so results come back in
minutes, not hours.

Usage: .venv/bin/python scripts/run_ab_subset.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lore.ab_test import ABTest  # noqa: E402
from lore.context import ContextManager  # noqa: E402
from lore.tool_attention import ToolAttention  # noqa: E402

SERVER_BIN = ROOT / "external/llama-cpp-turboquant/build/bin/llama-server"
PRIMARY_GGUF = ROOT / "models/ornith-1.0-9b-Q4_K_M.gguf"
EMBED_GGUF = ROOT / "models/nomic-embed-text-v1.5.f16.gguf"
PRIMARY_PORT = 19200
EMBED_PORT = 19201
TASKS_PATH = ROOT / "benchmarks/eval_tasks/agentic_subset.json"
RESULTS_PATH = ROOT / "benchmarks/results/ab_subset_report.json"

WORKING_CONTEXT_BUDGET = 16384
MAX_TOKENS = 128  # enough for real answers, fast enough for 5x4=20 calls
COMPRESSION_MIN_TURNS = 10


def _derive_tokenizer_repo() -> str:
    models_path = ROOT / "configs" / "models.yaml"
    if models_path.exists():
        data = yaml.safe_load(models_path.read_text()) or {}
        source = data.get("primary", {}).get("source", "")
        if source.endswith("-GGUF"):
            source = source[: -len("-GGUF")]
        return source
    return ""


class DirectServer:
    def chat(self, model, messages, **opts):
        resp = requests.post(
            f"http://127.0.0.1:{PRIMARY_PORT}/v1/chat/completions",
            json={"messages": messages, "stream": False, **opts}, timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    def embed(self, text):
        resp = requests.post(
            f"http://127.0.0.1:{EMBED_PORT}/v1/embeddings",
            json={"input": text, "model": "nomic-embed"}, timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def tokenize(self, model, text):
        return len(text) // 4


def start_server(gguf, port, ctx, extra_args=None):
    args = [
        str(SERVER_BIN), "-m", str(gguf),
        "-c", str(ctx), "-ngl", "999",
        "-np", "1", "--port", str(port), "--host", "127.0.0.1",
    ] + (extra_args or [])
    Path("logs").mkdir(exist_ok=True)
    log = open(ROOT / f"logs/ab_subset_{port}.log", "w")
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=log)


def wait_healthy(port, timeout=120):
    for _ in range(timeout):
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def make_ctx(server, compression_enabled, tool_attention_enabled):
    tool_attention = ToolAttention.from_config(server) if tool_attention_enabled else None
    compression = {
        "enabled": compression_enabled,
        "ratio": 0.5,
        "min_turns": COMPRESSION_MIN_TURNS,
        "preserve_recent_turns": 3,
    }
    return ContextManager(
        {"working_context": WORKING_CONTEXT_BUDGET}, server,
        system_prompt="You are a helpful senior software engineering assistant. Answer concisely and accurately.",
        tokenizer_source="local",
        tokenizer_repo=_derive_tokenizer_repo(),
        tool_attention=tool_attention, compression=compression,
    )


def make_run_fn(server, ctx):
    def run(task, config):
        t0 = time.time()
        try:
            ctx.add_message("user", task["prompt"])
            messages = ctx.build_prompt(query=task["prompt"])
            prompt_tokens = sum(len(m["content"]) // 4 for m in messages)
            result = server.chat("primary", messages, max_tokens=MAX_TOKENS, temperature=0)
            content = result["choices"][0]["message"]["content"]
            ctx.add_message("assistant", content)
            latency = time.time() - t0
            tokens_out = result.get("usage", {}).get("completion_tokens", len(content.split()))
            return {
                "latency_s": round(latency, 2),
                "tokens_out": tokens_out,
                "success": True,
                "prompt_tokens_est": prompt_tokens,
                "context_msgs": len(messages),
            }
        except Exception as e:
            return {
                "latency_s": round(time.time() - t0, 2),
                "tokens_out": 0,
                "success": False,
                "error": str(e),
            }
    return run


def main():
    if not PRIMARY_GGUF.exists():
        print("Primary model missing, aborting.", file=sys.stderr)
        sys.exit(1)

    tasks = json.loads(TASKS_PATH.read_text())["tasks"]
    server = DirectServer()

    print(f"Starting servers (5 tasks x 4 variants = 20 calls, {MAX_TOKENS} max tokens)...")
    primary_proc = start_server(
        PRIMARY_GGUF, PRIMARY_PORT, ctx=WORKING_CONTEXT_BUDGET + 4096,
        extra_args=["-fa", "on", "-ctk", "turbo4", "-ctv", "turbo4"],
    )
    embed_proc = start_server(
        EMBED_GGUF, EMBED_PORT, ctx=2048,
        extra_args=["--embedding", "--pooling", "mean"],
    )
    if not wait_healthy(PRIMARY_PORT) or not wait_healthy(EMBED_PORT):
        print("Server(s) failed to start", file=sys.stderr)
        stop_server(primary_proc)
        stop_server(embed_proc)
        sys.exit(1)

    variants = {
        "baseline": {"compression": False, "tool_attention": False},
        "plus_compression": {"compression": True, "tool_attention": False},
        "plus_tool_attention": {"compression": False, "tool_attention": True},
        "plus_all_combined": {"compression": True, "tool_attention": True},
    }

    report = {}
    for name, flags in variants.items():
        print(f"\n=== {name} ===")
        ctx = make_ctx(server, flags["compression"], flags["tool_attention"])
        run_fn = make_run_fn(server, ctx)

        # Run tasks one by one so we can see per-task progress
        results = []
        for i, task in enumerate(tasks):
            r = run_fn(task, {})
            results.append({"task_id": task["id"], **r})
            status = "OK" if r["success"] else "FAIL"
            print(f"  [{i+1}/5] {task['id']}: {status} {r['latency_s']}s "
                  f"{r.get('tokens_out', 0)} tok, {r.get('context_msgs', 0)} msgs")
            sys.stdout.flush()

        latencies = [r["latency_s"] for r in results]
        tok_rates = [r["tokens_out"] / r["latency_s"] for r in results
                     if r["latency_s"] > 0 and r["tokens_out"] > 0]
        successes = [r["success"] for r in results]

        metrics = {
            "label": name,
            "n_tasks": len(tasks),
            "p50_latency_s": sorted(latencies)[len(latencies) // 2],
            "p95_latency_s": sorted(latencies)[-1],
            "avg_tokens_per_sec": round(sum(tok_rates) / len(tok_rates), 2) if tok_rates else 0,
            "completion_rate": round(sum(successes) / len(tasks), 4),
            "per_task": results,
        }
        report[name] = metrics
        print(json.dumps({k: v for k, v in metrics.items() if k != "per_task"}, indent=2))

    stop_server(primary_proc)
    stop_server(embed_proc)

    report["_params"] = {
        "n_tasks": len(tasks),
        "working_context_budget": WORKING_CONTEXT_BUDGET,
        "max_tokens": MAX_TOKENS,
        "tool_registry_size": 50,
        "compression_min_turns": COMPRESSION_MIN_TURNS,
        "note": "5-task representative subset for fast Phase 3 evaluation.",
    }
    ABTest.save_report(report, str(RESULTS_PATH))
    print(f"\nSaved report to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
