#!/usr/bin/env python3
"""Benchmark KV cache strategies: memory, quality, latency.

Usage:
  PYTHONPATH=src python scripts/benchmark_kv_cache.py --model primary --context-sizes 4096,8192,16384
  PYTHONPATH=src python scripts/benchmark_kv_cache.py --model specialist --kv-types turbo4,q8_0,q4_0,fp16

Requires a running llama-server. The script:
1. Queries /health to verify server is up
2. Sends prompts at different context sizes
3. Measures RSS via /metrics endpoint (if available) or psutil
4. Measures latency per token
5. Saves results to benchmarks/results/kv_cache_benchmark.json
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

logger = logging.getLogger("bench_kv")

PORTS = {"primary": 19000, "specialist": 19001, "embeddings": 19002}
RESULTS_PATH = ROOT / "benchmarks/results/kv_cache_benchmark.json"


def is_healthy(port: int) -> bool:
    try:
        return requests.get(f"http://127.0.0.1:{port}/health", timeout=3).status_code == 200
    except Exception:
        return False


def get_rss_mb(pid: int | None = None) -> float:
    """Get RSS in MB for a process. Falls back to 0 if unavailable."""
    try:
        import psutil
        if pid:
            return psutil.Process(pid).memory_info().rss / 1024 / 1024
        return psutil.virtual_memory().used / 1024 / 1024
    except ImportError:
        return 0.0


def find_server_pid(port: int) -> int | None:
    """Try to find the llama-server PID by port (lsof)."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip():
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def measure_latency(port: int, prompt: str, max_tokens: int = 128) -> dict:
    """Send a chat completion and measure latency + token throughput."""
    t0 = time.time()
    try:
        resp = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        wall_s = time.time() - t0
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        content = data["choices"][0]["message"]["content"]
        return {
            "wall_s": round(wall_s, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_s": round(completion_tokens / wall_s, 2) if wall_s > 0 else 0,
            "content_length": len(content),
            "success": True,
        }
    except Exception as e:
        return {"wall_s": time.time() - t0, "success": False, "error": str(e)[:200]}


def build_context_prompt(target_tokens: int) -> str:
    """Build a prompt that fills approximately target_tokens of context."""
    # ~4 chars/token, so target_tokens * 4 = target chars
    # Use a repeated passage to fill context
    base_text = (
        "The quick brown fox jumps over the lazy dog. "
        "In machine learning, attention mechanisms allow models to focus on "
        "relevant parts of the input sequence. The KV cache stores key and "
        "value tensors for each attention head across all layers, enabling "
        "efficient autoregressive generation without recomputing attention. "
    )
    target_chars = target_tokens * 4
    repetitions = max(1, target_chars // len(base_text))
    filled = base_text * repetitions
    # Add a question at the end so the model generates output
    return filled[:target_chars] + "\n\nQuestion: What is the KV cache used for? Answer in one sentence."


def run_benchmark(model: str, kv_types: list[str], context_sizes: list[int]) -> list[dict]:
    """Run benchmark for each KV type × context size combination."""
    port = PORTS.get(model, 19000)
    if not is_healthy(port):
        print(f"Server for {model} not running on port {port}. Start it first.")
        print(f"  PYTHONPATH=src python3 -c \"from lore.models import ModelServer; from lore.config import LoreConfig; s=ModelServer(LoreConfig().models); s.start_model('{model}')\"")
        return []

    pid = find_server_pid(port)
    results = []

    for ctx_size in context_sizes:
        prompt = build_context_prompt(ctx_size)
        print(f"\n  Context: {ctx_size} tokens")

        # Measure baseline RSS before prompt
        rss_before = get_rss_mb(pid) if pid else 0

        # Send prompt
        print(f"    Sending prompt (~{len(prompt)} chars)...", end="", flush=True)
        latency_result = measure_latency(port, prompt, max_tokens=64)
        print(f" {latency_result.get('tokens_per_s', 0):.1f} tok/s")

        # Measure RSS after
        rss_after = get_rss_mb(pid) if pid else 0

        results.append({
            "model": model,
            "context_size": ctx_size,
            "kv_cache_type": "current",  # server is already running with its config
            "rss_before_mb": round(rss_before, 1),
            "rss_after_mb": round(rss_after, 1),
            "rss_delta_mb": round(rss_after - rss_before, 1),
            **latency_result,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="KV cache benchmark")
    parser.add_argument("--model", choices=["primary", "specialist"], default="primary",
                        help="Model to benchmark")
    parser.add_argument("--context-sizes", type=str, default="4096,8192,16384",
                        help="Comma-separated context sizes in tokens")
    parser.add_argument("--kv-types", type=str, default="turbo4",
                        help="Comma-separated KV types (note: server must be restarted per type)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    context_sizes = [int(x) for x in args.context_sizes.split(",")]
    kv_types = [x.strip() for x in args.kv_types.split(",")]

    print(f"\n{'='*60}")
    print(f" KV Cache Benchmark: {args.model}")
    print(f" Context sizes: {context_sizes}")
    print(f" KV types: {kv_types} (note: server uses current config)")
    print(f"{'='*60}")

    results = run_benchmark(args.model, kv_types, context_sizes)

    if results:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "model": args.model,
            "context_sizes": context_sizes,
            "results": results,
            "note": "Server was running with its configured KV type. To test "
                    "different KV types, restart server with different config.",
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved → {RESULTS_PATH}")

        # Print summary table
        print(f"\n{'='*60}")
        print(f" {'Context':>10}│{'RSS (MB)':>10}│{'Tok/s':>8}│{'Wall (s)':>10}")
        print(" " + "─" * 9 + "┼" + "─" * 9 + "┼" + "─" * 7 + "┼" + "─" * 9)
        for r in results:
            rss = r.get("rss_after_mb", 0)
            tps = r.get("tokens_per_s", 0)
            wall = r.get("wall_s", 0)
            print(f" {r['context_size']:>10}│{rss:>10.1f}│{tps:>8.1f}│{wall:>10.3f}")
        print(f"{'='*60}\n")
    else:
        print("\nNo results. Start the model server first.")


if __name__ == "__main__":
    main()
