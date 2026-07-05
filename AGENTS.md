# LORE — Local Orchestration & Runtime Engine

## Project Overview

LORE is a local AI orchestration layer for edge devices with 16 GB RAM. It coordinates multiple specialized small language models to maximize real-world performance under strict memory and compute constraints. The system is inspired by Sakana Fugu's orchestration principles but adapted for local inference with quantized open-source models.

**Core thesis:** With TurboQuant KV cache compression (4.57×), SSM/hybrid specialist models (near-zero KV cache), and smart routing, you CAN have a strong primary model AND a specialist AND long context simultaneously in 16 GB. The previous "pick two" constraint no longer applies.

## Hardware Baseline

- **Device:** Apple Silicon M4, 16 GB unified memory
- **Memory model:** Unified — RAM = VRAM. Model weights, KV cache, and OS share 16 GB.
- **GPU backend:** Metal (NOT CUDA). All GPU kernels must be Metal-compatible.
- **Storage:** SSD required for model files (~15 GB) and KV cache disk persistence
- **OS:** macOS (primary target)

## Tech Stack

- **Inference engine:** llama.cpp via `TheTom/llama-cpp-turboquant` (TurboQuant+ with Metal kernels)
- **Model format:** GGUF (quantized with imatrix calibration)
- **Primary model:** Ornith-1.0-9B Q4_K_M (5.63 GB) — SOTA agentic coding
- **Specialist model:** Falcon-H1-1.5B Q4_K_M (1.00 GB) — hybrid SSM, near-zero KV cache
- **Embeddings:** nomic-embed-text-v1.5 (0.30 GB)
- **Router:** TF-IDF + Logistic Regression (scikit-learn, <50 MB)
- **Constrained output:** GBNF grammars via llama.cpp, XGrammar
- **Model management:** llama-swap (hot-swap between models)
- **Language:** Python 3.11+ for orchestration layer, C/C++ for inference

## Memory Budget

```
Ornith-1.0-9B Q4_K_M (primary)         5.63 GB
Falcon-H1-1.5B Q4_K_M (specialist)     1.00 GB
nomic-embed-text-v1.5                   0.30 GB
KV cache (Ornith @ 16K, turbo4_0)       0.61 GB
KV cache (Falcon-H1 @ 16K)              0.04 GB
OS + llama.cpp + buffers                1.50 GB
Working memory                          1.00 GB
────────────────────────────────────────────────
TOTAL                                  10.08 GB   (5.92 GB headroom)
```

**Actual measured (Phase 0):** 6.59 GB total (both models turbo4 16K). 7.41 GB headroom.

**With Phase 2+3 optional components (when enabled):**

```
Dual model baseline (turbo4, 16K)       6.59 GB
LLMLingua-2 compression (CPU, opt-in)  +0.34 GB  (758 MB process RSS, lazy singleton)
Session state (JSON, in-memory)         ~0.01 GB  (negligible)
Hierarchical memory (embeddings)        ~0.05 GB  (200 episodes x 768-dim float32 + 100 facts)
Health monitor (stateless)              ~0.00 GB  (logs to disk only)
────────────────────────────────────────────────
MAX (all features enabled)              6.99 GB   (7.01 GB headroom)
```

**HARD RULE:** Total memory must stay under 14 GB at all times. 2 GB reserved for OS spikes.

## Conditional Gating Strategy

Phase 2 showed optimizations have real overhead that dominates at small scale. Each is now conditionally gated:

| Feature | Gate Condition | Default | When It Helps |
|---------|---------------|---------|---------------|
| LLMLingua-2 compression | session >= 10 turns AND usage > 70% of budget AND old messages exist | disabled | 50+ turn sessions with context pressure |
| Tool Attention embed() | registry > 15 tools | enabled (50-tool registry) | Large tool registries where full injection would bloat prompt |
| Hierarchical memory | Opt-in via config | disabled | Multi-session workflows needing persistent context |
| Health monitor | Runs every 5 turns when enabled | disabled | Long sessions where context degradation is a risk |
| Session persistence | Opt-in via config | disabled | Session resume across restarts |

**Key insight:** "Measure before stacking" means each optimization must prove its value at the scale it's designed for, not at toy scale. The gates enforce this automatically.

## Project Structure

```
lore/
├── AGENTS.md                 ← This file (agent instructions)
├── PLAN.md                   ← Full consolidated plan (all research, techniques, roadmap)
├── docs/                     ← Research papers, analysis documents
│   ├── feasibility.md        ← Original feasibility analysis
│   ├── revised-analysis.md   ← TurboQuant + new models revision
│   ├── ssm-specialists.md    ← SSM/Mamba specialist deep dive
│   ├── optimization-log.md   ← Per-technique measurement results
│   └── architecture.md       ← System architecture + data flow (Phase 3)
├── src/                      ← Python orchestration layer
│   ├── lore/                 ← Main package
│   │   ├── __init__.py
│   │   ├── router.py         ← TF-IDF + LogReg task classifier
│   │   ├── context.py        ← Context manager (budget, compression, memory, health)
│   │   ├── memory.py         ← Hierarchical memory: episodic + semantic tiers
│   │   ├── health.py         ← Context health monitor (utilization, staleness, actions)
│   │   ├── session.py        ← Session save/resume (KV cache replay)
│   │   ├── tool_attention.py ← Lazy tool schema loading (NTILC pattern, size-gated)
│   │   ├── models.py         ← Model lifecycle (load, swap, health check)
│   │   └── config.py         ← Configuration management
│   └── cli.py                ← CLI entry point
├── configs/                  ← Model and system configurations
│   ├── models.yaml           ← Model registry (paths, quant types, roles)
│   ├── router.yaml           ← Router training config
│   ├── memory.yaml           ← Memory system config
│   └── llama-swap.yaml       ← llama-swap model management config
├── scripts/                  ← Setup, benchmarking, maintenance
│   ├── setup.sh              ← Build llama.cpp, download models, generate imatrix
│   ├── benchmark.sh          ← Run full benchmark suite
│   ├── train_router.py       ← Train the TF-IDF router on labeled data
│   └── health_check.sh       ← System health check (memory, models, latency)
├── benchmarks/               ← Benchmark results and evaluation
│   ├── results/              ← JSON result files per benchmark run
│   └── eval_tasks/           ← Custom evaluation task definitions
└── models/                   ← Model files (GGUF, embeddings) — gitignored
```

## Architecture Decisions

### Decision 1: Primary Model — Ornith-1.0-9B
- **Why:** Best agentic coding benchmarks at 9B. SWE-bench Verified 69.4 (beats Gemma4-31B). Self-improving RL training.
- **Quantization:** Q4_K_M with imatrix calibration on mixed corpus (code + chat + math + multilingual)
- **Fallback:** Qwen3.5-9B if Ornith has issues with non-coding tasks

### Decision 2: Specialist Model — Falcon-H1-1.5B
- **Why:** Beats every 1.5B transformer on every benchmark. Hybrid SSM (only 2 attention heads) = near-zero KV cache. 128K native context. Fully supported in llama.cpp.
- **Key advantage:** At 128K context, KV cache is ~0.32 GB vs ~1.34 GB for a pure transformer
- **Fallback:** Qwen2.5-1.5B if Falcon-H1 has issues with few-shot ICL tasks

### Decision 3: KV Cache Compression — TurboQuant (turbo4_0)
- **Why:** 3.6× compression with near-lossless quality. Single biggest memory lever.
- **Risk:** +5-8% PPL on Qwen-based architectures (Ornith is Qwen-based). Must validate.
- **Fallback:** Mixed-precision KV (FP16 hot + Q4 cold) if TurboQuant quality is unacceptable

### Decision 4: Router — Non-LLM (TF-IDF + LogReg)
- **Why:** <1ms latency, <50 MB memory, >85% accuracy. No LLM inference cost for routing.
- **Training data:** 200+ labeled examples of task types (code, chat, extraction, classification)
- **Fallback:** If accuracy <80%, add a confidence gate that defaults to primary model

### Decision 5: Structured Output — GBNF + XGrammar
- **Why:** 100% valid JSON/code output. Eliminates all parsing failures. 5-15% throughput cost.
- **Implementation:** `json-schema-to-grammar.py` converts JSON schemas to GBNF grammars
- **Always use for:** tool calls, structured extraction, code generation

## Implementation Phases

### Phase 0: Foundation (Days 1-3)
- [x] Build llama.cpp with TurboQuant support (TheTom/llama-cpp-turboquant fork, Metal kernels)
- [x] Download Ornith-1.0-9B + Falcon-H1-1.5B GGUF models
- [x] Generate imatrix for Ornith-9B on mixed calibration corpus (embedded, 401 chunks)
- [x] Validate: both models loaded simultaneously, measure actual memory (6.59 GB, 7.41 GB headroom)
- [ ] Baseline benchmarks: GSM8K, HumanEval, IFEval — deferred (need eval frameworks)

### Phase 1: MVP Orchestration (Days 4-10)
- [x] Implement TF-IDF router (src/lore/router.py) — 96.1% accuracy
- [x] Configure prefix KV cache reuse (system prompt cached once)
- [x] Set up GBNF constrained decoding for JSON output
- [ ] Configure llama-swap for model management
- [x] Implement basic context manager with token-aware budgets
- [x] Basic memory: embed last 5 turns, retrieve by cosine similarity

### Phase 1.5: Polish
- [x] TOOL_ONLY fast-path, local tokenizer cache (0.19ms vs 5-20ms HTTP), shared dispatch

### Phase 2: Optimization Stack (Days 11-21)
- [x] Integrate LLMLingua-2 for prompt compression — shipped opt-in, 56.5% token reduction, +118MB RSS. **Gated**: only fires when session >= 10 turns AND usage > 70% of budget. See `docs/optimization-log.md`.
- [x] Configure speculative decoding (Falcon-H1 as draft for Ornith) — **SKIP**: Falcon-H1/Ornith-9B vocabs are incompatible (real llama-server test), a fixed architectural constraint.
- [ ] Set up n-gram speculative decoding for code tasks — not attempted this round; `--spec-type ngram-simple` remains untested.
- [x] Evaluate TIDE early exit on specialist model — **SKIP**: Falcon-H1's hybrid SSM/Mamba layers carry recurrent state that TIDE's per-token early exit would corrupt; also HF-transformers/CUDA-only, no GGUF/Metal support.
- [x] Configure host-memory caching (--cram) — shipped opt-in; measured only ~3% RSS reduction on Falcon-H1 (KV cache already near-zero for hybrid SSM), smaller than the ~0.5GB originally planned.
- [x] Implement Tool Attention for lazy schema loading — shipped; 93.5% token reduction on 50-tool registry. **Gated**: skips embed() entirely when registry <= 15 tools. See `docs/optimization-log.md`.
- [x] A/B test each technique independently — `src/lore/ab_test.py` + `scripts/run_ab_suite.py` (20-task) + `scripts/run_ab_subset.py` (5-task). Also surfaced a real chat-template bug (multiple system messages) invisible to mocked tests.
- [x] Gate optimizations for real scale — compression gated on min_turns + usage ratio; tool attention gated on registry size. Validated by 5-task subset A/B.

### Phase 3: Advanced Agentic (Days 22-35)
- [x] Implement hierarchical memory (working → episodic → semantic) — `src/lore/memory.py`: EpisodicMemory (summarize via specialist), SemanticMemory (extract facts), HierarchicalMemory (3-tier orchestrator). 13 tests.
- [x] Context health monitoring (token utilization, quality proxy) — `src/lore/health.py`: ContextHealth.check() returns HealthReport with utilization, age, repetition, staleness. Actions: ok/compress/summarize/prune/warn. Logs to `logs/context_health.jsonl`. 12 tests.
- [x] KV cache disk persistence for session resume — `src/lore/session.py`: SessionManager save/resume/list/cleanup. Saves context + metadata as JSON, replays prefix on resume (only option for SSM models). 8 tests.
- [x] Wire hierarchical memory into ContextManager — `build_prompt()` retrieves top-3 episodic + top-5 semantic by query similarity. Health check triggers summarize/compress actions.
- [ ] Evaluate MiniCache cross-layer KV merging
- [ ] Evaluate PoLar/BUDDY dynamic layer routing
- [ ] Multi-session management with shared prefix

### Phase 4: Benchmark & Harden (Days 36-42)
- [ ] Full benchmark suite (standard + custom agent tasks)
- [ ] Memory and latency profiling
- [ ] A/B: orchestrated vs single model, same memory budget
- [ ] Failure mode catalog
- [ ] Final documentation

## Coding Conventions

- **Python style:** Type hints on all functions. Docstrings on public methods. PEP 8.
- **Error handling:** Every model call wrapped in try/except. Never crash on model failure — fall back to primary model.
- **Logging:** Use Python `logging` module. Log level configurable. Log all routing decisions.
- **Testing:** pytest. Test each component in isolation. Integration tests for full pipeline.
- **Configuration:** YAML files in configs/. Environment variables override YAML.
- **No hardcoded paths:** All model paths from config. Relative to project root or absolute.

## Key Constraints

1. **Memory is sacred.** Every component must report its memory footprint. Total must stay under 14 GB.
2. **Measure before stacking.** Each optimization must be A/B tested independently before combining.
3. **Default to single model.** Orchestration should be opt-in for tasks that benefit, not the default path.
4. **Fail gracefully.** If any orchestration component fails, fall back to raw primary model inference.
5. **No cloud dependencies.** Everything runs locally. No API calls to external services.
6. **Reproducible.** All benchmarks run with fixed seeds, documented configs, saved results.

## Pitfalls to Watch

### Mac M4 / Apple Silicon Specific

1. **TurboQuant fork matters:** Use `TheTom/llama-cpp-turboquant` (TurboQuant+), NOT the animehacker CUDA fork. TurboQuant+ has production Metal kernels. The animehacker fork is CUDA-only.
2. **Speculative decoding on Metal is NOT like CUDA.** The critical factor is draft-to-target speed ratio (need >2.5×), not acceptance rate. A 0.8B draft at 140 tok/s → 9B target at 42 tok/s = 3.31× ratio = 25.7% throughput gain even at 2-4% acceptance. Self-speculative MTP is SLOWER on Metal (known bug: github.com/ggml-org/llama.cpp/issues/23752). MoE models get zero benefit from spec decode.
3. **Quantization sweet spot on Apple Silicon:** Q6_K is Pareto-optimal (0.54% PPL, 1.68× speedup, 59% size reduction). Q4_K_M for memory-constrained. Sub-4-bit (Q2_K) degrades catastrophically (267% PPL increase).
4. **Unified memory pressure:** macOS will start killing processes above ~14 GB RSS. Monitor with `memory_pressure` command. Leave 2 GB headroom.
5. **Metal Flash Attention:** Requires `-fa 1` flag. Essential for long context. Verify it's enabled in your build.
6. **imatrix calibration matters more on Apple Silicon:** At Q4, the weight sensitivity is higher due to unified memory bandwidth constraints. Always calibrate with mixed corpus.

### General

7. **TurboQuant on Qwen architectures:** +5-8% PPL sensitivity. Test on Ornith specifically before committing.
8. **Prefix cache fragility:** One space/character difference in system prompt = full cache miss. Keep system prompt static.
9. **Router accuracy:** If <80%, the system is worse than single model. Always have a confidence gate.
10. **Context bloat from orchestration:** Router output + specialist output + plan + verification all consume tokens. Strict per-component budgets.
11. **Falcon-H1 license:** Uses Falcon License (not Apache 2.0). Verify compatibility with your use case.

## External References

- **TurboQuant paper:** arxiv:2504.19874 (ICLR 2026)
- **TurboQuant llama.cpp fork (CUDA):** github.com/animehacker/llama-turboquant
- **TurboQuant+ (Metal/Apple Silicon, production):** github.com/TheTom/llama-cpp-turboquant
- **Ornith-1.0:** github.com/deepreinforce-ai/Ornith-1
- **Falcon-H1:** huggingface.co/tiiuae/Falcon-H1-1.5B-Instruct-GGUF
- **TIDE early exit:** github.com/RightNow-AI/TIDE
- **PoLar (ICML 2026):** github.com/tianyi-lab/PoLar
- **Tool Attention:** github.com/asadani/tool-attention
- **NTILC:** arxiv:2606.06566
- **llama-swap:** github.com/mostlygeek/llama-swap
- **XGrammar:** github.com/mlc-ai/xgrammar
- **LLMLingua-2:** github.com/microsoft/LLMLingua
- **MiniCache:** arxiv:2405.14366
- **Fugu (Sakana):** arxiv:2606.21228
- **Mamba-3:** arxiv:2603.15569
- **Fugu orchestration principles:** sakana.ai/fugu

## When to Ask the User

- Before downloading any model >2 GB (confirm disk space)
- Before building llama.cpp from a non-mainline fork
- Before changing the primary or specialist model selection
- Before adding a new optimization that increases memory usage
- If a benchmark shows orchestration is worse than single model
- If Falcon-H1 license is incompatible with the use case
