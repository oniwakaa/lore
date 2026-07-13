#!/usr/bin/env python3
"""Quality comparison: direct vs orchestrated on a single task.

Runs the same prompt two ways with the current primary model:
1. Direct: straight to primary model, no orchestration
2. Orchestrated: full LORE pipeline (router, decomposer, workers, aggregator)

Saves both outputs + timing to docs/orchestration-quality-comparison.md
"""
import sys
import time
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import requests
from lore.config import LoreConfig
from lore.models import ModelServer
from lore.router import Router
from lore.context import ContextManager
from lore.memory import HierarchicalMemory
from lore.health import ContextHealth
from lore.tool_attention import ToolAttention
from lore.verifier import Verifier
from lore.logging import RequestLogger
from lore.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TASK = (
    "Refactor the authentication module to use JWT tokens and then "
    "update all the tests and also write migration documentation"
)


def setup():
    cfg = LoreConfig.load()
    cfg.models["primary"]["context"] = 16384
    cfg.models["specialist"]["context"] = 16384
    cfg.models["defaults"]["context_size"] = 16384

    server = ModelServer(cfg.models)
    logger.info("Starting model servers...")
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
            print(f"  [FAIL] {role} on port {port} did not start")
            server.stop_all()
            sys.exit(1)
        print(f"  [OK] {role} healthy ({attempt+1}s)")

    router = Router.load(
        cfg.router.get("model_path", "configs/router_model.joblib"),
        confidence_threshold=cfg.router.get("confidence_threshold", 0.70),
    )
    system_prompt = "You are a helpful assistant. Answer concisely and accurately. /no_think"
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

    return server, router, ctx, memory, orchestrator


def run_direct(server, task):
    """Send task straight to primary model, no orchestration."""
    print("\n--- Direct (no orchestration) ---")
    t0 = time.time()
    resp = server.chat("primary", [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely and accurately."},
        {"role": "user", "content": task},
    ], max_tokens=4096, temperature=0.6)
    latency = time.time() - t0
    content = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    tokens_gen = usage.get("completion_tokens", 0)
    tok_s = tokens_gen / latency if latency > 0 else 0
    print(f"  Latency: {latency:.1f}s")
    print(f"  Tokens: {tokens_gen}")
    print(f"  Tok/s: {tok_s:.1f}")
    print(f"  Content preview: {content[:200]}...")
    return {"content": content, "latency_s": latency, "tokens": tokens_gen, "tok_s": tok_s,
            "usage": usage}


def run_orchestrated(orchestrator, dispatch_fn, task):
    """Run task through full LORE orchestration pipeline."""
    print("\n--- Orchestrated (full pipeline) ---")
    t0 = time.time()
    r = orchestrator.process(task, dispatch_fn=dispatch_fn)
    latency = time.time() - t0
    print(f"  Orchestrated: {r['orchestrated']}")
    print(f"  Subtasks: {r.get('subtasks_completed', 0)}")
    print(f"  Latency: {latency:.1f}s")
    print(f"  Content preview: {r['content'][:200]}...")
    return r


def main():
    print("=" * 60)
    print("LORE Quality Comparison: Direct vs Orchestrated")
    print("=" * 60)
    print(f"Task: {TASK[:80]}...")

    try:
        server, router, ctx, memory, orchestrator = setup()
    except Exception as e:
        print(f"FAILED to start: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    from lore.cli import _dispatch
    req_logger = RequestLogger()
    verifier = Verifier()
    dispatch_fn = lambda q, json_mode=False: _dispatch(
        q, server, router, ctx, memory, req_logger, json_mode, verifier)

    model_name = LoreConfig.load().models.get("primary", {}).get("name", "unknown")
    print(f"\nPrimary model: {model_name}")

    try:
        direct = run_direct(server, TASK)
        orchestrated = run_orchestrated(orchestrator, dispatch_fn, TASK)

        # Write comparison doc
        doc_path = Path("docs/orchestration-quality-comparison.md")
        doc = f"""# Orchestration Quality Comparison: Direct vs Orchestrated

**Model:** {model_name}
**Date:** 2026-07-13
**Task:** {TASK}

## Results

| Metric | Direct | Orchestrated |
|--------|--------|-------------|
| Latency | {direct['latency_s']:.1f}s | {orchestrated.get('latency_ms', 0)/1000:.1f}s |
| Tokens generated | {direct['tokens']} | N/A |
| Tok/s | {direct['tok_s']:.1f} | N/A |
| Orchestrated | No | {orchestrated.get('orchestrated', False)} |
| Subtasks | 0 | {orchestrated.get('subtasks_completed', 0)} |

## Direct Output

```
{direct['content'][:3000]}
```

## Orchestrated Output

```
{orchestrated.get('content', '')[:3000]}
```

## Analysis

- **Latency:** Orchestrated adds overhead from routing, decomposition, and aggregation.
- **Quality:** Compare completeness, structure, and correctness of the two outputs.
- **Verdict:** See docs/qwythos-eval.md for the final decision.
"""
        doc_path.write_text(doc)
        print(f"\nComparison written to {doc_path}")

        # Also save raw JSON
        results = {"model": model_name, "task": TASK, "direct": direct, "orchestrated": {
            "orchestrated": orchestrated.get("orchestrated", False),
            "subtasks_completed": orchestrated.get("subtasks_completed", 0),
            "content": orchestrated.get("content", ""),
            "latency_ms": orchestrated.get("latency_ms", 0),
        }}
        Path("benchmarks/results/quality_comparison.json").write_text(json.dumps(results, indent=2))

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        server.stop_all()


if __name__ == "__main__":
    main()
