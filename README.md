<p align="center">
  <strong>⚔️ LORE</strong><br>
  <em>Local Orchestration & Runtime Engine</em><br><br>
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/Inference-llama.cpp-green?logo=cplusplus" alt="llama.cpp">
  <img src="https://img.shields.io/badge/Models-2%20loaded-9B59B6" alt="2 models">
  <img src="https://img.shields.io/badge/Memory-6.6%20GB%20%2F%2016%20GB-orange" alt="Memory">
  <img src="https://img.shields.io/badge/Tests-222-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/Phases-0--4.2%20complete-brightgreen" alt="Phases">
</p>

---

LORE orchestrates multiple specialized small language models on edge devices with 16 GB RAM. Instead of picking one big model and hoping for the best, LORE loads a **primary model** for reasoning and a **specialist model** for lightweight tasks — simultaneously, within budget.

**The key insight:** With TurboQuant KV cache compression (4.57×) and SSM/hybrid specialist models (near-zero KV cache), you no longer have to choose between model quality, context length, and memory headroom. You get all three.

## Architecture

```
                         USER REQUEST
                              │
                    ┌─────────▼──────────┐
                    │    TOOL ATTENTION    │  Lazy schema loading (93.5% fewer tokens)
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │   CONTEXT MANAGER   │  Budget, compression, memory, health
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │       ROUTER        │  TF-IDF + LogReg (<1ms, no LLM cost)
                    └───┬─────────┬───┘
                        │         │
            ┌───────────▼──┐  ┌──▼───────────┐  ┌──────────────────┐
            │  PRIMARY 9B  │  │ SPECIALIST 1.5B│  │   TOOL-ONLY      │
            │  Ornith-1.0  │  │  Falcon-H1     │  │  (no LLM needed) │
            │  5.63 GB     │  │  1.00 GB       │  │                  │
            └──────┬───────┘  └──────┬────────┘  └────────┬─────────┘
                   └────────────────┴─────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │    ORCHESTRATOR     │  Decompose → schedule → execute → aggregate
                    └────────────────────┘  (only for complex multi-step tasks)
```

## Memory Budget (measured)

| Component | Size |
|-----------|------|
| Ornith-1.0-9B Q4_K_M (primary) | 5.50 GB |
| Falcon-H1-1.5B Q4_K_M (specialist) | 1.09 GB |
| KV cache (both models, turbo4, 16K context) | included above |
| **Total** | **6.59 GB** |
| **Headroom** | **7.41 GB** |

All features enabled (compression, memory, health): **6.99 GB**. Never exceeds 14 GB.

## What's Inside

| Module | Purpose |
|--------|---------|
| `orchestrator.py` | Task decomposition, wave-based scheduling, parallel execution, aggregation |
| `router.py` | TF-IDF + LogReg classifier (<1ms, >85% accuracy) |
| `context.py` | Token budget management, compression gating, prefix cache |
| `memory.py` | Hierarchical memory — episodic (embeddings) + semantic (facts) |
| `tool_attention.py` | Lazy tool schema selection via embeddings (NTILC pattern) |
| `verifier.py` | JSON/code validation and auto-repair |
| `leaderboard.py` | Live HuggingFace benchmark scanning for model upgrades |
| `registry.py` | Auto-select best local model per task type from benchmarks |
| `classifier.py` | Model-based task complexity estimation (replaces regex heuristics) |
| `session.py` | KV cache disk persistence for instant session resume |
| `health.py` | Context utilization monitoring, staleness detection |
| `models.py` | llama-server lifecycle, model swapping via llama-swap |

## Optimizations (measured, not theoretical)

| Technique | Result | Default |
|-----------|--------|---------|
| TurboQuant KV compression | 0% PPL degradation on hybrid SSM models | Always on |
| LLMLingua-2 compression | 56.5% token reduction | Off (opt-in, activates at 10+ turns) |
| Tool Attention | 93.5% fewer tool tokens (3200→207 at 50 tools) | On (gated at 15+ tools) |
| Parallel wave execution | Cross-model subtasks run simultaneously | On |
| Speculative decoding | **Skipped** — vocab mismatch between models | N/A |
| TIDE early exit | **Skipped** — SSM-incompatible | N/A |
| MiniCache | **Skipped** — TurboQuant conflict | N/A |

Every optimization was tested against real inference. Techniques that don't work on hybrid SSM architectures were skipped with evidence, not hoped away.

## Quick Start

```bash
# Clone
git clone https://github.com/oniwakaa/lore.git
cd lore

# Install
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Setup (on Apple Silicon M4 with 16 GB)
bash scripts/setup.sh          # builds llama.cpp, downloads models, generates imatrix

# Launch
lore                           # starts interactive REPL
```

## Configuration

All config lives in `configs/`:

- **`models.yaml`** — Model paths, quantization types, engine settings, server path
- **`router.yaml`** — Router training parameters
- **`memory.yaml`** — Memory system settings (episodic/semantic)
- **`compression.yaml`** — LLMLingua-2 settings
- **`llama-swap.yaml`** — Model hot-swap configuration

## REPL Commands

| Command | Action |
|---------|--------|
| `<message>` | Process a task (auto-routes or orchestrates) |
| `/switch <session>` | Switch to a different session |
| `/save` | Save current session (KV cache + state) |
| `/resume <id>` | Resume a saved session |
| `/sessions` | List all saved sessions |
| `/models` | Show loaded models and memory usage |
| `/upgrades` | Check HuggingFace for better model options |

## Project Status

| Phase | Status | Key Result |
|-------|--------|------------|
| 0: Foundation | ✅ | Dual model @ 6.59 GB, TurboQuant validated |
| 1: Core Stack | ✅ | Router, context manager, GBNF structured output |
| 2: Optimizations | ✅ | 6 techniques measured, 3 shipped, 3 skipped with evidence |
| 3: Agentic | ✅ | Memory, health, sessions, multi-session management |
| 3.5: Wire + Verify | ✅ | End-to-end integration, verifier, dynamic sizing |
| 4: Orchestration | ✅ | Decomposer, workers, wave scheduling, aggregation |
| 4.1: Hardening | ✅ | Public APIs, parallel execution, deduped memory |
| 4.2: Live Benchmarks | ✅ | HF leaderboard scanning, registry, model-based classifier |
| 5: Benchmark & Harden | 🔜 | Full A/B: orchestrated vs single model |

## Design Philosophy

1. **Measure before stacking.** Every optimization must prove its value at the scale it's designed for. No feature runs unconditionally.
2. **Skip with evidence.** If a technique doesn't work on hybrid SSM architectures, it gets skipped with a table showing why — not a TODO.
3. **Conditional gating.** Optimizations activate only when their crossover point is reached (compression at 10+ turns, tool attention at 15+ tools).
4. **Honest benchmarks.** Toy-scale tests that don't reach crossover points don't count. Real inference, real hardware, real numbers.
5. **The orchestration question.** Orchestration adds complexity. It must measurably beat single-model dispatch to justify itself. Phase 5 answers this.

## Hardware Requirements

- **Minimum:** Apple Silicon M4 (or equivalent), 16 GB unified memory
- **Storage:** ~15 GB for model files
- **OS:** macOS (primary), Linux (supported)
- **Backend:** Metal (macOS), CPU (fallback)

## References

- [TurboQuant](https://arxiv.org/abs/2504.19874) — KV cache compression (ICLR 2026)
- [Ornith-1.0](https://github.com/deepreinforce-ai/Ornith-1) — Primary model
- [Falcon-H1](https://huggingface.co/tiiuae/Falcon-H1-1.5B-Instruct-GGUF) — Specialist model
- [Tool Attention](https://arxiv.org/abs/2604.21816) — Lazy schema loading
- [Sakana Fugu](https://sakana.ai/fugu) — Orchestration inspiration

---

<p align="center">
  <em>Built for the edge. Measured on real hardware. No hand-waving.</em>
</p>
