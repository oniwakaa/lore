#!/usr/bin/env python3
"""Benchmark speculative decoding: Falcon-H1 (draft) + Ornith-9B (target).

Usage: .venv/bin/python scripts/benchmark_spec_decode.py

Starts llama-server twice (baseline, then with -md/--spec-type draft-simple),
runs a fixed prompt set against each, and reports tokens/sec + time-to-first-token.
If the two models' vocabs are incompatible, llama-server logs a warning and
silently runs without draft acceleration -- this script detects that case from
the server log and skips the (meaningless) "with spec decode" run.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
SERVER_BIN = ROOT / "external/llama-cpp-turboquant/build/bin/llama-server"
PRIMARY_GGUF = ROOT / "models/ornith-1.0-9b-Q4_K_M.gguf"
DRAFT_GGUF = ROOT / "models/Falcon-H1-1.5B-Instruct-Q4_K_M.gguf"
PORT = 19096
LOG_PATH = ROOT / "logs/spec_decode_bench.log"

PROMPTS = [
    "Write a Python function that reverses a string.",
    "What is the capital of France?",
    "Explain what a hash table is in one paragraph.",
    "Write a function to check if a number is prime.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What is 15 multiplied by 23?",
    "Write a SQL query to select all users older than 30.",
    "Explain the difference between a list and a tuple in Python.",
    "Write a function that computes the factorial of a number.",
    "What does HTTP stand for?",
    "Write a regex to match a valid email address.",
    "Explain what a binary search algorithm does.",
    "Write a function to find the maximum value in a list.",
    "What is the time complexity of quicksort on average?",
    "Write a Python class representing a simple bank account.",
    "Explain the difference between TCP and UDP.",
    "Write a function that checks if a string is a palindrome.",
    "What is a race condition in concurrent programming?",
    "Write a function to compute the nth Fibonacci number.",
    "Explain what garbage collection does in Python.",
]


def start_server(extra_args: list[str]) -> subprocess.Popen:
    args = [
        str(SERVER_BIN), "-m", str(PRIMARY_GGUF),
        "-c", "8192", "-ngl", "999", "-fa", "on",
        "-ctk", "turbo4", "-ctv", "turbo4",
        "-np", "1", "--port", str(PORT), "--host", "127.0.0.1",
    ] + extra_args
    log = open(LOG_PATH, "w")
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=log)


def wait_healthy(timeout: int = 60) -> bool:
    for _ in range(timeout):
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def vocab_incompatible() -> bool:
    """Check server log for the vocab-mismatch warning."""
    if not LOG_PATH.exists():
        return False
    return "vocabs are not compatible" in LOG_PATH.read_text()


def run_prompts(prompts: list[str]) -> dict:
    latencies_ttft = []
    tokens_per_sec = []
    for p in prompts:
        t0 = time.time()
        resp = requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": p}], "max_tokens": 64, "temperature": 0},
            timeout=120,
        )
        elapsed = time.time() - t0
        data = resp.json()
        completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
        if completion_tokens > 0:
            tokens_per_sec.append(completion_tokens / elapsed)
        latencies_ttft.append(elapsed)
    return {
        "n": len(prompts),
        "avg_latency_s": sum(latencies_ttft) / len(latencies_ttft) if latencies_ttft else 0,
        "avg_tokens_per_sec": sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0,
    }


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def main():
    if not PRIMARY_GGUF.exists() or not DRAFT_GGUF.exists():
        print("Model files missing, aborting.", file=sys.stderr)
        sys.exit(1)

    print("=== Baseline (no speculative decoding) ===")
    proc = start_server([])
    if not wait_healthy():
        print("Server failed to start", file=sys.stderr)
        stop_server(proc)
        sys.exit(1)
    baseline = run_prompts(PROMPTS)
    stop_server(proc)
    print(json.dumps(baseline, indent=2))

    print("\n=== Speculative decoding (Falcon-H1 draft) ===")
    proc = start_server(["-md", str(DRAFT_GGUF), "--spec-type", "draft-simple"])
    if not wait_healthy():
        print("Server failed to start", file=sys.stderr)
        stop_server(proc)
        sys.exit(1)

    if vocab_incompatible():
        stop_server(proc)
        print("DECISION GATE: draft and target vocabs are incompatible "
              "(Falcon-H1 tokenizer != Ornith tokenizer). llama-server falls back "
              "to running without draft acceleration -- a spec-decode benchmark run "
              "would only remeasure the baseline. Skipping. See docs/optimization-log.md.")
        return

    spec = run_prompts(PROMPTS)
    stop_server(proc)
    print(json.dumps(spec, indent=2))

    speedup = (spec["avg_tokens_per_sec"] / baseline["avg_tokens_per_sec"] - 1) * 100
    print(f"\nSpeedup: {speedup:.1f}%")
    print("PASS" if speedup >= 10 else "DECISION GATE: speedup < 10%, do not integrate")


if __name__ == "__main__":
    main()
