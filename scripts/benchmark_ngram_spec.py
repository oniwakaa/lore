#!/usr/bin/env python3
"""Benchmark ngram-simple speculative decoding on the specialist (Falcon-H1-1.5B).

Usage: .venv/bin/python scripts/benchmark_ngram_spec.py

Starts llama-server twice with the specialist model (baseline, then with
--spec-type ngram-simple), runs the real TaskClassifier workload plus an
extraction set against each at temperature 0, and reports tokens/sec.
ngram-simple is prompt lookup decoding: it drafts tokens by matching n-gram
patterns already in the KV cache, so it needs NO draft model. Greedy decoding
must produce identical output with and without it, so the script also reports
an exact-match rate between the two arms as the quality check.

Env overrides:
  LORE_LLAMA_SERVER  path to llama-server (default: bundled turboquant build)
  LORE_BENCH_KV      KV cache type for both arms (default: turbo4)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
SERVER_BIN = Path(os.environ.get(
    "LORE_LLAMA_SERVER",
    ROOT / "external/llama-cpp-turboquant/build/bin/llama-server"))
SPECIALIST_GGUF = ROOT / "models/Falcon-H1-1.5B-Instruct-Q4_K_M.gguf"
KV_TYPE = os.environ.get("LORE_BENCH_KV", "turbo4")
PORT = 19097
LOG_PATH = ROOT / "logs/ngram_spec_bench.log"

# Real specialist workload: the TaskClassifier system prompt. The JSON output
# repeats key names verbatim from this prompt — the pattern ngram lookup drafts.
CLASSIFY_SYSTEM = """You are a task classifier for a local AI orchestration system.
Classify the user's task into one of these categories:
- classification: sorting/categorizing text
- extraction: pulling structured data from text
- summarization: condensing text
- code_gen: writing code
- testing: writing tests
- documentation: writing docs
- math: mathematical computation
- planning: multi-step planning
- review: code/text review

Also determine:
- is_complex: true if the task needs decomposition into subtasks
- estimated_subtasks: 1-5, how many subtasks if complex
- suggested_model: "primary" for reasoning/coding, "specialist" for simple tasks

Output JSON:
{"is_complex": bool, "task_type": "code_gen", "estimated_subtasks": 3, "suggested_model": "primary", "hints": {"multi_part": true, "needs_code": true}}"""

CLASSIFY_QUERIES = [
    "Sort these support tickets by urgency: server down, typo on homepage, billing question.",
    "Write a Python script that scrapes product prices and stores them in SQLite.",
    "Summarize this quarterly report into three bullet points for the exec team.",
    "Extract the invoice number, date, and total from this email.",
    "Plan the migration of our monolith to microservices over two quarters.",
    "Is this review positive or negative: 'The battery died after two days.'",
    "Write unit tests for the payment retry logic.",
    "Document the REST endpoints of the users service.",
    "What is the integral of x^2 * sin(x)?",
    "Review this pull request for security issues.",
    "Categorize these expenses as travel, meals, or office supplies.",
    "Pull all email addresses and phone numbers out of this contact page.",
    "Condense this 5-page design doc into a one-paragraph abstract.",
    "Build a CLI tool that converts CSV files to JSON.",
    "Label each sentence as fact or opinion.",
    "Break down 'launch the new onboarding flow' into engineering subtasks.",
    "Extract all function names and their arguments from this source file.",
    "Is this email spam or not: 'You won a free cruise, click here.'",
    "Compute the compound interest on $10,000 at 4% over 10 years.",
    "Summarize the changelog since v2.3 for the release notes.",
]

# Extraction/summarization set: output copies long spans from the input,
# the other high-repetition pattern in the specialist path.
EXTRACT_DOC = (
    "Order #A-4471 was placed on 2026-03-14 by Dana Whitfield "
    "(dana.whitfield@example.com, +1-555-0142). Items: 2x USB-C cable at $9.99, "
    "1x mechanical keyboard at $84.50, 3x HDMI adapter at $12.25. Shipping to "
    "88 Alder Street, Portland, OR 97205. Total charged: $141.23 to Visa ending 4402."
)
EXTRACT_PROMPTS = [
    f"Extract the order number, customer name, email, and total as JSON:\n{EXTRACT_DOC}",
    f"List every item with quantity and unit price, one per line:\n{EXTRACT_DOC}",
    f"Extract the full shipping address:\n{EXTRACT_DOC}",
    f"Summarize this order in one sentence:\n{EXTRACT_DOC}",
    f"Extract all monetary amounts in the order they appear:\n{EXTRACT_DOC}",
]


def start_server(extra_args: list[str]) -> subprocess.Popen:
    args = [
        str(SERVER_BIN), "-m", str(SPECIALIST_GGUF),
        "-c", "8192", "-ngl", "999", "-fa", "on",
        "-ctk", KV_TYPE, "-ctv", KV_TYPE,
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


def run_workload() -> dict:
    """Run classification + extraction prompts, return metrics and raw outputs."""
    requests_spec = [
        {"messages": [{"role": "system", "content": CLASSIFY_SYSTEM},
                      {"role": "user", "content": f"Classify this task:\n{q}"}],
         "max_tokens": 128}
        for q in CLASSIFY_QUERIES
    ] + [
        {"messages": [{"role": "user", "content": p}], "max_tokens": 160}
        for p in EXTRACT_PROMPTS
    ]

    outputs, latencies, tokens_per_sec = [], [], []
    for body in requests_spec:
        t0 = time.time()
        resp = requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json={**body, "stream": False, "temperature": 0},
            timeout=120,
        )
        elapsed = time.time() - t0
        data = resp.json()
        outputs.append(data["choices"][0]["message"]["content"])
        completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
        if completion_tokens > 0:
            tokens_per_sec.append(completion_tokens / elapsed)
        latencies.append(elapsed)
    return {
        "n": len(requests_spec),
        "avg_latency_s": sum(latencies) / len(latencies),
        "avg_tokens_per_sec": sum(tokens_per_sec) / len(tokens_per_sec),
        "outputs": outputs,
    }


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_arm(name: str, extra_args: list[str]) -> dict:
    print(f"=== {name} ===")
    proc = start_server(extra_args)
    if not wait_healthy():
        stop_server(proc)
        print(f"Server failed to start for '{name}', see {LOG_PATH}", file=sys.stderr)
        sys.exit(1)
    result = run_workload()
    stop_server(proc)
    print(json.dumps({k: v for k, v in result.items() if k != "outputs"}, indent=2))
    return result


def main():
    if not SPECIALIST_GGUF.exists():
        print(f"Model file missing: {SPECIALIST_GGUF}", file=sys.stderr)
        sys.exit(1)
    if not SERVER_BIN.exists():
        print(f"llama-server not found: {SERVER_BIN}", file=sys.stderr)
        sys.exit(1)

    baseline = run_arm("Baseline (no speculative decoding)", [])
    print()
    ngram = run_arm("ngram-simple speculative decoding",
                    ["--spec-type", "ngram-simple"])

    matches = sum(a == b for a, b in zip(baseline["outputs"], ngram["outputs"]))
    speedup = (ngram["avg_tokens_per_sec"] / baseline["avg_tokens_per_sec"] - 1) * 100
    print(f"\nExact-match outputs (quality, greedy must be lossless): "
          f"{matches}/{baseline['n']}")
    print(f"Speedup: {speedup:.1f}%")
    print("PASS" if speedup >= 10 else "DECISION GATE: speedup < 10%, do not integrate")


if __name__ == "__main__":
    main()
