#!/usr/bin/env python3
"""Run benchmarks/eval_tasks/agentic.json at agentic-realistic scale:
  - 50-tool registry (configs/tools.yaml)
  - 16K context budget
  - 50-task suite with session accumulation
  - 256 max_tokens per generation
  - 4 variants: baseline, +compression, +tool_attention, +all_combined

Compression auto-enables after turn 10 (gate in ContextManager).
Tool Attention auto-enables with 50 tools (gate in ToolAttention).

Usage: .venv/bin/python scripts/run_ab_realistic.py
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


def _derive_tokenizer_repo() -> str:
    """Read primary model source from configs/models.yaml and strip -GGUF suffix."""
    models_path = ROOT / "configs" / "models.yaml"
    if models_path.exists():
        data = yaml.safe_load(models_path.read_text()) or {}
        source = data.get("primary", {}).get("source", "")
        if source.endswith("-GGUF"):
            source = source[: -len("-GGUF")]
        return source
    return ""

from lore.ab_test import ABTest  # noqa: E402
from lore.context import ContextManager  # noqa: E402
from lore.tool_attention import ToolAttention  # noqa: E402

SERVER_BIN = ROOT / "external/llama-cpp-turboquant/build/bin/llama-server"
PRIMARY_GGUF = ROOT / "models/ornith-1.0-9b-Q4_K_M.gguf"
EMBED_GGUF = ROOT / "models/nomic-embed-text-v1.5.f16.gguf"
PRIMARY_PORT = 19200
EMBED_PORT = 19201
TASKS_PATH = ROOT / "benchmarks/eval_tasks/agentic.json"
RESULTS_PATH = ROOT / "benchmarks/results/ab_realistic_report.json"

# Agentic-realistic parameters
WORKING_CONTEXT_BUDGET = 16384  # 16K, not 800
MAX_TOKENS = 256  # not 32
COMPRESSION_MIN_TURNS = 10  # auto-enable compression after turn 10


class DirectServer:
    """Minimal ModelServer-compatible client (chat + embed + tokenize stub)."""

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


def start_server(gguf: Path, port: int, ctx: int = 16384,
                  extra_args: list[str] | None = None) -> subprocess.Popen:
    args = [
        str(SERVER_BIN), "-m", str(gguf),
        "-c", str(ctx), "-ngl", "999",
        "-np", "1", "--port", str(port), "--host", "127.0.0.1",
    ] + (extra_args or [])
    Path("logs").mkdir(exist_ok=True)
    log = open(ROOT / f"logs/ab_realistic_{port}.log", "w")
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=log)


def wait_healthy(port: int, timeout: int = 120) -> bool:
    for _ in range(timeout):
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def make_ctx(server, compression_enabled: bool, tool_attention_enabled: bool) -> ContextManager:
    # ToolAttention.from_config loads all 50 tools + gate params from configs/tools.yaml.
    # With 50 tools > min_tools_for_attention(15), embedding-based selection is active.
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


def make_run_fn(server, ctx: ContextManager):
    def run(task, config):
        t0 = time.time()
        try:
            ctx.add_message("user", task["prompt"])
            messages = ctx.build_prompt(query=task["prompt"])
            result = server.chat("primary", messages, max_tokens=MAX_TOKENS, temperature=0)
            content = result["choices"][0]["message"]["content"]
            ctx.add_message("assistant", content)
            latency = time.time() - t0
            tokens_out = result.get("usage", {}).get("completion_tokens") or len(content.split())
            return {"latency_s": latency, "tokens_out": tokens_out, "success": True}
        except Exception as e:
            return {"latency_s": time.time() - t0, "tokens_out": 0, "success": False, "error": str(e)}
    return run


def main():
    if not PRIMARY_GGUF.exists():
        print("Primary model missing, aborting.", file=sys.stderr)
        sys.exit(1)

    tasks = json.loads(TASKS_PATH.read_text())["tasks"]
    server = DirectServer()

    print(f"Starting primary + embeddings servers (50 tasks, {WORKING_CONTEXT_BUDGET}-token budget, {MAX_TOKENS} max tokens)...")
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
        ab = ABTest(tasks, run_fn)
        metrics = ab.run_variant({}, label=name)
        report[name] = metrics
        print(json.dumps(metrics, indent=2))

    stop_server(primary_proc)
    stop_server(embed_proc)

    report["_params"] = {
        "n_tasks": len(tasks),
        "working_context_budget": WORKING_CONTEXT_BUDGET,
        "max_tokens": MAX_TOKENS,
        "tool_registry_size": 50,
        "compression_min_turns": COMPRESSION_MIN_TURNS,
        "note": "Agentic-realistic scale: 50 tools, 16K budget, 256 max_tokens, session accumulation across 50 tasks.",
    }
    ABTest.save_report(report, str(RESULTS_PATH))
    print(f"\nSaved report to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
