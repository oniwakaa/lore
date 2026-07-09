#!/usr/bin/env python3
"""Orchestration benchmark: A/B mode or HumanEval pass@1.

A/B mode: direct (single 9B call) vs orchestrated, on custom tasks.
HumanEval mode: LORE full pipeline on OpenAI HumanEval (164 coding tasks).

Usage:
  PYTHONPATH=src python scripts/benchmark_orchestration.py                 # A/B
  PYTHONPATH=src python scripts/benchmark_orchestration.py --quick          # A/B 3 tasks
  PYTHONPATH=src python scripts/benchmark_orchestration.py --benchmark humaneval --limit 10
  PYTHONPATH=src python scripts/benchmark_orchestration.py --benchmark humaneval          # all 164
"""
import argparse
import json
import logging
import os
import re
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
HUMANEVAL_PATH = ROOT / "benchmarks/eval_tasks/humaneval.jsonl"
HUMANEVAL_RESULTS_PATH = ROOT / "benchmarks/results/humaneval_lore.json"

PRIMARY_PORT = 19000
SPECIALIST_PORT = 19001
EMBED_PORT = 19002
TIMEOUT_S = 300

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


def build_orchestrator(server: ModelServer) -> tuple[Orchestrator, callable]:
    """Wire Orchestrator like cli.py does. Returns (orchestrator, dispatch_fn)."""
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
    orchestrator = Orchestrator(server, router, memory, orch_cfg, ctx=ctx,
                                classifier=classifier)
    dispatch_fn = make_dispatch_fn(server, router, ctx, memory)
    return orchestrator, dispatch_fn


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
            result = server.chat(model, messages, max_tokens=2048, temperature=0.0,
                                 timeout=TIMEOUT_S)
            content = result["choices"][0]["message"]["content"]
            success = True
        except Exception as e:
            if model == "specialist":
                result = server.chat("primary", messages, max_tokens=2048,
                                     temperature=0.0, timeout=TIMEOUT_S)
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


# ═══ HumanEval Functions ═════════════════════════════════════════════

def load_humaneval(limit: int | None = None) -> list[dict]:
    """Load HumanEval problems from JSONL."""
    problems = []
    with open(HUMANEVAL_PATH) as f:
        for line in f:
            problems.append(json.loads(line))
    if limit:
        problems = problems[:limit]
    return problems


def extract_code(response: str, prompt: str) -> str:
    """Extract Python function implementation from LORE's response."""
    # Strategy 1: markdown code block (longest block = most likely implementation)
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    if blocks:
        code = max(blocks, key=len).strip()
        if "def " in code or "class " in code or "import " in code:
            return code

    # Strategy 2: find code starting from first def/class/import
    lines = response.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith(('def ', 'class ', 'import ', 'from ')):
            return '\n'.join(lines[i:]).strip()

    # Strategy 3: return raw (might be terse)
    return response.strip()


def build_test_program(prompt: str, generated_code: str,
                       test_code: str, entry_point: str) -> str:
    """Build executable test program from prompt + generated code + tests."""
    # Extract import lines from prompt (needed regardless)
    prompt_lines = prompt.strip().split('\n')
    import_lines = [l for l in prompt_lines if l.strip().startswith(('import ', 'from '))]

    if f'def {entry_point}' in generated_code:
        # Model output the full function — use it, add imports from prompt
        return '\n'.join(import_lines) + '\n\n' + generated_code + '\n\n' + test_code + f'\ncheck({entry_point})\n'
    else:
        # Model output just the body — prepend full prompt (signature + docstring + imports)
        return prompt.rstrip() + '\n' + generated_code + '\n\n' + test_code + f'\ncheck({entry_point})\n'


def run_test_sandboxed(code: str, timeout: float = 10.0) -> dict:
    """Execute code in a subprocess with timeout."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
        f.write(code)
        f.flush()
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            timeout=timeout, capture_output=True, text=True,
        )
        return {
            "passed": result.returncode == 0,
            "stdout": result.stdout[-500:],
            "stderr": result.stderr[-500:],
            "timeout": False,
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "stdout": "", "stderr": "TIMEOUT", "timeout": True}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def run_humaneval_task(problem: dict, orchestrator: Orchestrator, dispatch_fn) -> dict:
    """Run LORE on one HumanEval problem and test the output."""
    prompt = problem["prompt"]
    test_code = problem["test"]
    entry_point = problem["entry_point"]

    # 1. Send through LORE
    t0 = time.time()
    result = orchestrator.process(prompt, json_mode=False, dispatch_fn=dispatch_fn)
    latency = time.time() - t0
    content = result.get("content", "")

    # 2. Extract code
    generated_code = extract_code(content, prompt)
    code_extracted = bool(generated_code.strip()) and (
        "def " in generated_code or "class " in generated_code
        or "import " in generated_code or "from " in generated_code
        or generated_code.strip()[0] not in '#\n'
    )

    # 3. Build full test program
    full_code = build_test_program(prompt, generated_code, test_code, entry_point)

    # 4. Execute in sandbox
    test_result = run_test_sandboxed(full_code, timeout=10.0)

    return {
        "task_id": problem["task_id"],
        "entry_point": entry_point,
        "passed": test_result["passed"],
        "latency_s": round(latency, 2),
        "orchestrated": bool(result.get("orchestrated", False)),
        "subtasks_count": int(result.get("subtasks_completed", 0)),
        "model_used": result.get("model", "unknown"),
        "code_extracted": code_extracted,
        "test_timeout": test_result["timeout"],
        "error": test_result.get("stderr", "")[:200],
    }


def save_humaneval_incremental(results: list[dict], path: Path) -> None:
    """Save results after each task (partial runs still produce data)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "orchestrated": sum(1 for r in results if r["orchestrated"]),
        "code_extracted": sum(1 for r in results if r.get("code_extracted")),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / max(len(results), 1), 2),
    }
    output = {"summary": summary, "results": results}
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def print_humaneval_table(results: list[dict]) -> None:
    """Print comparison table with published scores."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    attempted = total
    orchestrated = sum(1 for r in results if r["orchestrated"])
    code_extracted = sum(1 for r in results if r.get("code_extracted"))
    avg_latency = sum(r["latency_s"] for r in results) / max(total, 1)
    pass_pct = passed / max(attempted, 1) * 100

    # Published baselines
    baselines = [
        ("Qwen3.6-27B (published)", "~90%", "24 GB", "262K"),
        ("Qwen2.5-Coder-14B Q4 (publ.)", "~73%", "24 GB", "128K"),
        ("Qwen3.5-9B Q4 (published)", "~75%", "16 GB", "262K (theoretical)"),
        ("Ornith-1.0-9B Q4 (published)", "~75%", "16 GB", "262K (theoretical)"),
    ]
    lore_pct = f"{pass_pct:.0f}%"

    print("\n═══════════════════════════════════════════════════════════════════")
    print(" LORE BENCHMARK: HumanEval (pass@1)")
    print("═══════════════════════════════════════════════════════════════════\n")
    print(f" {'Model':<32}│{'pass@1':>8}│{'Hardware':>10}│{'Context':>20}")
    print(" ─" * 32 + "┼" + "─" * 8 + "┼" + "─" * 10 + "┼" + "─" * 20)
    for name, pct, hw, ctx in baselines:
        print(f" {name:<32}│{pct:>8}│{hw:>10}│{ctx:>20}")
    print(" ─" * 32 + "┼" + "─" * 8 + "┼" + "─" * 10 + "┼" + "─" * 20)
    print(f" {'LORE (Orchestrated 9B+1.5B)':<32}│{lore_pct:>8}│{'16 GB':>10}│{'2-4K/task':>20}")
    print(" ─" * 32 + "┼" + "─" * 8 + "┼" + "─" * 10 + "┼" + "─" * 20)

    # Deltas vs baselines (parse numeric from published)
    lore_num = pass_pct
    for name, pct, _, _ in baselines:
        m = re.search(r'(\d+)', pct)
        if m:
            base_num = int(m.group(1))
            delta = lore_num - base_num
            sign = "+" if delta >= 0 else ""
            label = name.split("(")[0].strip()
            print(f" Δ LORE vs {label:<26}│{sign}{delta:.0f} pp")
    print(" ═" * 33 + "══" * 8 + "══" * 10 + "══" * 20)

    print(f"\n Tasks: 164 | Attempted: {attempted} | Passed: {passed} | Failed: {attempted - passed}")
    print(f" Avg latency: {avg_latency:.1f}s | Orchestrated: {orchestrated}/{total} | Routed direct: {total - orchestrated}/{total}")
    print(f" Code extraction success: {code_extracted}/{total}")
    print("═══════════════════════════════════════════════════════════════════\n")


# ═══ A/B Table ═══════════════════════════════════════════════════════

def fmt_pct(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.0f}%"


def print_ab_table(results: dict) -> None:
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


# ═══ Main ═════════════════════════════════════════════════════════════

def run_ab_benchmark(server: ModelServer, tasks: list[dict]) -> None:
    """Run the original A/B benchmark (direct vs orchestrated)."""
    orchestrator, dispatch_fn = build_orchestrator(server)

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

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {RESULTS_PATH}")

    print_ab_table(results)


def run_humaneval_benchmark(server: ModelServer, limit: int | None) -> None:
    """Run LORE on HumanEval pass@1 benchmark."""
    problems = load_humaneval(limit=limit)
    print(f"\nLoaded {len(problems)} HumanEval problems")

    orchestrator, dispatch_fn = build_orchestrator(server)

    # Fresh context per task — don't let history accumulate across problems
    fresh_dispatch = make_fresh_dispatch(server, dispatch_fn)

    results: list[dict] = []
    print(f"\nRunning LORE on {len(problems)} HumanEval tasks...\n")

    for i, problem in enumerate(problems):
        tid = problem["task_id"]
        print(f"  [{i+1}/{len(problems)}] {tid} ({problem['entry_point']}) ", end="", flush=True)
        r = run_humaneval_task(problem, orchestrator, fresh_dispatch)
        results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        orch = "orch" if r["orchestrated"] else "direct"
        print(f"{status} {r['latency_s']:.1f}s {orch} subtasks={r['subtasks_count']}"
              + (f" err={r['error'][:60]}" if r["error"] else ""))

        # Save incrementally after each task
        save_humaneval_incremental(results, HUMANEVAL_RESULTS_PATH)

    print(f"\nSaved → {HUMANEVAL_RESULTS_PATH}")
    print_humaneval_table(results)


def make_fresh_dispatch(server, dispatch_fn):
    """Wrap dispatch_fn to reset context between HumanEval problems.

    Each HumanEval task is independent — accumulated chat history
    from previous problems would pollute context and waste tokens.
    Router is loaded once (from disk) and reused; only context is fresh.
    """
    from lore.config import LoreConfig
    cfg = LoreConfig.load()
    system_prompt = "You are a helpful assistant. Answer concisely and accurately."
    tokenizer_source = cfg.models.get("defaults", {}).get("tokenizer_source", "local")
    tokenizer_repo = cfg.models.get("primary", {}).get("source", "")
    if tokenizer_repo.endswith("-GGUF"):
        tokenizer_repo = tokenizer_repo[:-len("-GGUF")]
    memory = HierarchicalMemory(cfg.memory, server)
    router = Router.load(
        cfg.router.get("model_path", "configs/router_model.joblib"),
        confidence_threshold=cfg.router.get("confidence_threshold", 0.70),
    )

    def fresh_dispatch(query, json_mode=False):
        ctx = ContextManager(cfg.context, server, system_prompt=system_prompt,
                             tokenizer_source=tokenizer_source,
                             tokenizer_repo=tokenizer_repo or None,
                             memory=memory)
        fn = make_dispatch_fn(server, router, ctx, memory)
        return fn(query, json_mode=json_mode)

    return fresh_dispatch


def main():
    parser = argparse.ArgumentParser(description="Orchestration benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="A/B mode: run 3 tasks only for smoke test")
    parser.add_argument("--benchmark", choices=["ab", "humaneval"], default="ab",
                        help="Benchmark mode: 'ab' (default) or 'humaneval'")
    parser.add_argument("--limit", type=int, default=None,
                        help="HumanEval: limit number of tasks (e.g. --limit 10)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

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

    if args.benchmark == "humaneval":
        run_humaneval_benchmark(server, args.limit)
    else:
        tasks = load_tasks()
        if args.quick:
            simple = [t for t in tasks if t["complexity"] == "simple"][:1]
            complex_ = [t for t in tasks if t["complexity"] == "complex"][:2]
            tasks = simple + complex_
        run_ab_benchmark(server, tasks)

    print("Servers left running. Kill manually if done:")
    for role, port in [("primary", PRIMARY_PORT), ("specialist", SPECIALIST_PORT),
                       ("embeddings", EMBED_PORT)]:
        print(f"  {role}: port {port}")


if __name__ == "__main__":
    main()
