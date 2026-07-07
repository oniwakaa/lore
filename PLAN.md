# LORE — Full Technical Plan & Research Reference

**Project:** LORE (Local Orchestration & Runtime Engine)  
**Date:** 2026-07-02  
**Baseline:** 16 GB RAM, Linux, llama.cpp-based inference  
**Reference:** Sakana Fugu orchestration principles adapted for edge devices

---

## Table of Contents

1. [Feasibility Assessment](#1-feasibility-assessment)
2. [System Architecture](#2-system-architecture)
3. [Model Strategy](#3-model-strategy)
4. [SSM & Hybrid Specialist Models](#4-ssm--hybrid-specialist-models)
5. [KV Cache Compression & TurboQuant](#5-kv-cache-compression--turboquant)
6. [Complete Optimization Inventory](#6-complete-optimization-inventory)
7. [Memory & Context Design](#7-memory--context-design)
8. [Benchmarking & Evaluation](#8-benchmarking--evaluation)
9. [Risks & Failure Modes](#9-risks--failure-modes)
10. [Phased Build Roadmap](#10-phased-build-roadmap)
11. [Top Design Decisions](#11-top-design-decisions)
12. [References](#12-references)

---

## 1. Feasibility Assessment

### The Core Tension

Sakana Fugu achieves frontier performance by orchestrating frontier models (GPT-5.5, Claude Opus 4.8, Gemini 3.1 Pro). Its 5–6% gains over the best single model come from learned routing and multi-step delegation between models that are each individually near-SOTA.

Locally, the situation is inverted. Your worker pool is 3B–14B quantized models. Each orchestration step costs real latency (1–10s), real memory (GBs), and real tokens from a limited context window.

### Honest Assessment

| Claim | Verdict | Evidence |
|-------|---------|----------|
| "Match frontier cloud quality" | **No.** A 7B Q4 model cannot match Claude Opus regardless of orchestration. | Fugu's gains are 5–6% over SOTA baselines — with SOTA workers, not 7B workers. |
| "Outperform a single larger local model" | **Conditional yes.** Well-orchestrated 7B+1.5B can beat a single 14B on some tasks. | Benchmark data shows 7B Q4 ≈ 14B Q4 when 14B is aggressively quantized. |
| "Improve long-horizon agentic tasks" | **Yes, significantly.** Context management alone provides major gains. | Context bloat is the #1 killer of long sessions. |
| "Reduce hallucination through verification" | **Moderate yes.** Second-pass verification catches obvious errors. | Verifier patterns work but add latency. Quality gain is task-dependent. |
| "Enable useful agentic work locally" | **Yes.** The orchestration layer's biggest value is infrastructure, not intelligence. | Routing, memory management, tool use, and context control are the real wins. |

### Where Local Orchestration Is Genuinely Competitive

1. **Structured tasks with verifiable outputs** (code, JSON, math, data extraction)
2. **Long-running agentic workflows** (context management prevents quality collapse)
3. **Mixed-difficulty pipelines** (trivial steps use 1B, saving 7B for reasoning)
4. **Privacy-sensitive applications** (the only option where data cannot leave the device)

### Where It Is Not

1. **Open-ended creative/reasoning tasks** — orchestration adds latency without improving fundamental reasoning capacity
2. **Tasks requiring broad world knowledge** — small models hallucinate more, orchestration doesn't fix this
3. **Real-time interactive use** — multi-model latency (5–30s per response) is unacceptable for chat

---

## 2. System Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                                    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  TOOL ATTENTION LAYER (lazy schema loading)                              │
│  - Embedding-based tool selection (NTILC pattern)                        │
│  - Only inject top-k tool schemas, not full registry                     │
│  - Reduces tool tokens from ~50K to ~2.5K per turn                       │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  CONTEXT MANAGER                                                         │
│  - Tokenizer-aware ingestion (Qwen tiktoken)                             │
│  - Dynamic context sizing (task complexity → token budget)               │
│  - LLMLingua-2 compression (2–5× for QA/summarization)                  │
│  - Prefix KV cache reuse (system prompt computed once, reused)           │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  ROUTER (non-LLM, <1ms)                                                  │
│  - TF-IDF + LogReg classifier                                            │
│  - Routes: PRIMARY / SPECIALIST / TOOL_ONLY / ESCALATE                   │
│  - Confidence gate: <70% → use primary model                             │
└──────────┬─────────────────┬─────────────────┬───────────────────────────┘
           ▼                 ▼                 ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  PRIMARY MODEL   │ │  SPECIALIST MODEL│ │  TOOL-ONLY       │
│  Ornith-1.0-9B   │ │  Falcon-H1-1.5B  │ │  (no LLM)        │
│  Q4_K_M 5.63 GB  │ │  Q4_K_M 1.00 GB  │ │  regex/parser/    │
│  262K context     │ │  128K context     │ │  search           │
│  turbo4_0 KV      │ │  SSM (near-zero   │ │                  │
│  Spec decode with │ │  KV cache)        │ │                  │
│  specialist draft │ │                  │ │                  │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         └────────────────────┴────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  VERIFIER (conditional)                                                  │
│  - GBNF/XGrammar constrained decoding (100% valid JSON/code)            │
│  - Syntax check + test execution for code                                │
│  - 1B model or heuristic for format validation                           │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  MEMORY MANAGER                                                          │
│  - Episodic: summarize + embed (nomic-embed-text)                        │
│  - Semantic: durable facts extracted from episodes                       │
│  - Host-memory prompt caching (--cram for offloaded prefixes)            │
│  - KV cache disk persistence for session resume                          │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  MODEL HOT-SWAP (llama-swap or llama.cpp multi-model API)                │
│  - Primary + specialist loaded simultaneously (6.63 GB)                  │
│  - Gemma 4 E4B hot-swapped for multimodal tasks                          │
│  - Auto-unload after TTL                                                 │
└──────────────────────────────────────────────────────────────────────────┘
```

### Component Roles

**Router** — The critical decision point. Must be extremely fast (<100ms) and cheap (<100 tokens).

| Approach | Latency | Accuracy | Memory | Recommendation |
|----------|---------|----------|--------|----------------|
| TF-IDF + Logistic Regression | <1ms | 80–85% | <50 MB | **Best for MVP** |
| Fine-tuned 0.5B classifier | ~200ms | 85–90% | ~1 GB | Good for complex taxonomies |
| Rule-based keyword matching | <1ms | 70–75% | <1 MB | Fallback. Too brittle. |
| LLM-based routing | ~1s | 90–95% | shared | Too slow for routing step |

**Context Manager** — Highest-value component:
- Dynamic context sizing: simple task = 512 tokens, complex = 4K+, long-horizon = 8K+
- Tokenizer-aware preprocessing: use actual tokenizer for precise counting
- Compression: LLMLingua-2 (BERT-class encoder, 2–5× compression)
- Memory injection: retrieve relevant episodic memories as compressed context

**Verifier** — Conditional second pass:
- Structured output (JSON, code): always verify format
- Factual claims: verify against retrieved context (RAG)
- Mathematical reasoning: re-derive or check with symbolic tool
- Creative/open-ended: skip verification (adds latency, no clear criteria)

### Context Budget Allocation

```
Total context: 16384 tokens (Ornith-9B with turbo4_0 KV)
├── System prompt:        512 tokens (fixed, cached)
├── Retrieved memories:  1024 tokens (top-3 relevant episodes)
├── Running summary:      512 tokens (compressed history)
├── Working context:     2048 tokens (current task + tools)
├── Current input:       2048 tokens (user message + attachments)
├── Tool Attention pool:  768 tokens (top-k tool summaries)
└── Generation headroom: 2048 tokens (reserved for output)
```

---

## 3. Model Strategy

### Primary Model: Ornith-1.0-9B

**Why this model:**
- SOTA agentic coding at 9B: SWE-bench Verified 69.4, Terminal-Bench 43.1
- Beats Gemma4-31B on coding benchmarks at 9B size
- Self-improving RL training (jointly optimizes scaffold and solutions)
- 262K native context window
- MIT licensed, globally accessible
- Q4_K_M = 5.63 GB (well within budget)

| Benchmark | Ornith-9B | Qwen3.5-9B | Gemma4-12B | Gemma4-31B |
|-----------|-----------|------------|------------|------------|
| Terminal-Bench 2.1 | **43.1** | 21.3 | 21 | 42.1 |
| SWE-bench Verified | **69.4** | 53.2 | 44.2 | 52 |
| NL2Repo | **27.2** | 16.2 | 10.3 | 15.5 |

**GGUF sizes:**

| Quant | Size | Recommended |
|-------|------|-------------|
| IQ4_XS | 5.2 GB | Low-RAM option |
| Q4_K_M | 5.63 GB | **Default** |
| Q5_K_M | 6.5 GB | 24 GB systems |
| Q8_0 | 9.5 GB | 32 GB+ |

### Specialist Model: Falcon-H1-1.5B

**Why this model:**
- Beats every 1.5B transformer on every benchmark
- Hybrid Transformer + Mamba-2 (only 2 attention heads)
- Near-zero KV cache (SSM layers use fixed-size state)
- 128K native context
- Fully supported in llama.cpp
- Q4_K_M ≈ 1.00 GB

| Benchmark | Falcon-H1-1.5B | Qwen2.5-1.5B | Qwen3-1.7B | Gemma3-1B |
|-----------|----------------|--------------|------------|-----------|
| MMLU | **62.03** | 59.76 | 57.04 | 40.87 |
| GSM8K | **74.98** | 57.47 | 69.83 | 42.38 |
| HumanEval | **68.29** | 56.10 | 67.68 | 40.85 |
| IFEval | **80.66** | 45.33 | 70.77 | 61.48 |
| BBH | **46.47** | 42.41 | 35.18 | 35.86 |

### Alternative Models

| Model | Size Q4 | Use Case | Notes |
|-------|---------|----------|-------|
| Qwen3.5-9B | ~5.5 GB | Primary (general reasoning) | Apache 2.0. Matches Sonnet 4.5 on many benchmarks. |
| Gemma 4 E4B | 4.98 GB | Multimodal specialist | Text + Image + Audio. 256K context. Hot-swappable. |
| AI-Flow-Ruyi-7B | ~4.4 GB | Adaptive depth | 5 early-exit points (3B/4B/5B/6B/7B) from one model. |
| Gemma 4 26B A4B | 13.3 GB Q3 | High-knowledge tasks | MoE: 26B total, 3.8B active. Tight on 16 GB. |
| RWKV-7 2.9B | ~1.5 GB | Zero-KV specialist | Pure SSM, constant memory. Separate engine needed. |

### When 1B-Class Models Are Useful vs Harmful

**Useful for:**
- Text classification, named entity extraction, format conversion
- Keyword extraction, title generation, simple yes/no decisions
- Token-level importance scoring (LLMLingua-2)

**Harmful for:**
- Open-ended reasoning, mathematical problem-solving
- Code generation beyond trivial snippets
- Multi-step planning, factual Q&A requiring world knowledge

---

## 4. SSM & Hybrid Specialist Models

### The SSM Advantage for Specialists

SSM models have constant per-token memory regardless of context length. The KV cache problem vanishes:

| Property | Transformer (Qwen2.5-1.5B) | SSM (Falcon-H1-1.5B) |
|----------|---------------------------|----------------------|
| KV cache @ 4K context | ~0.15 GB | ~0.01 GB |
| KV cache @ 16K context | ~0.60 GB | ~0.04 GB |
| KV cache @ 128K context | ~4.80 GB | ~0.32 GB |
| Per-token latency at long context | Degrades | Constant |

### The Hybrid Spectrum

```
Pure SSM ←───────────────────────────────────────────────→ Pure Transformer
  │                                                             │
  RWKV-7         BitMamba-2       Falcon-H1    Zamba2    Qwen2.5
  (0 attn)       (0 attn)         (2 attn)    (shared)   (all attn)
  Constant mem   614 MB           ~1 GB        ~0.9 GB    ~1.1 GB
  
  ↑ BEST memory efficiency           ↑ BEST quality/speed balance
  ↑ Weakest at retrieval/ICL         ↑ Strong at ICL/retrieval
```

### SSM Model Comparison

| Model | Size | KV Cache | Quality | llama.cpp | Best For |
|-------|------|----------|---------|-----------|----------|
| Falcon-H1-1.5B | ~1.0 GB | ~0.04 GB @ 16K | ★★★★★ | ✅ | **Default specialist** |
| RWKV-7 2.9B | ~1.5 GB | 0 GB | ★★★★☆ | ❌ | Zero-KV (separate engine) |
| Zamba2-1.2B | ~0.9 GB | Minimal | ★★★☆☆ | ⚠️ PR | Smallest hybrid. 2K context. |
| Nemotron-H-4B | ~2.5 GB | Minimal | ★★★★☆ | ⚠️ PR | Strong but large for specialist. |
| BitMamba-2 1B | 0.6 GB | 0 GB | ★★☆☆☆ | ❌ | Ultra-lightweight router only. |
| DF-SSM 1.3B | 0.28 GB | 0 GB | ★★☆☆☆ | ❌ | Format checking, trivial tasks. |

---

## 5. KV Cache Compression & TurboQuant

### TurboQuant (turbo4_0) — The Breakthrough

TurboQuant (Zandieh et al., ICLR 2026) compresses KV cache to ~3.5 bits per element using Walsh-Hadamard rotation + Lloyd-Max quantization.

**Performance:**

| Model | FP16 PPL | turbo4_0 PPL | Delta | KV Memory |
|-------|----------|-------------|-------|-----------|
| Llama-3.2-3B | 9.77 | 9.82 | +0.4% | 224→63 MiB |
| Qwen2.5-3B | 9.14 | 9.84 | +7.7% | 72→20 MiB |
| Qwen3VL-8B | 8.15 | 8.57 | +5.2% | 288→81 MiB |

**3.6× compression.** Qwen models show higher sensitivity due to K/V norm disparity.

**Implementation status:** PR #21307 (closed — AI policy violation). Fork available: `animehacker/llama-turboquant`. Production-ready with CUDA + CPU support.

### Other KV Compression Techniques

| Technique | Compression | Quality Impact | Status |
|-----------|------------|----------------|--------|
| TurboQuant (turbo4_0) | 3.6× | +0.4–8% PPL | Fork available |
| MiniCache (cross-layer) | 1.53× additional | Near-lossless | Research paper |
| Mixed-precision KV | ~4–8× for cold tokens | Minimal | Commit in llama.cpp |
| Dynamic KV resize | Variable | None | PR #21757 |
| **Combined (TQ + MiniCache)** | **~5×** | TBD | Stacking needs validation |

### The "All Three" Configuration (Revised)

Previous constraint: "Can't have strong model + specialist + long context. Pick two."

With TurboQuant:

```
Ornith-1.0-9B Q4_K_M (primary)         5.63 GB
Falcon-H1-1.5B Q4_K_M (specialist)     1.00 GB
Embeddings                              0.30 GB
KV (Ornith @ 16K, turbo4_0)            0.61 GB
KV (Falcon-H1 @ 16K)                   0.04 GB
OS + buffers                            2.50 GB
────────────────────────────────────────────────
TOTAL                                  10.08 GB   ✓ 5.92 GB headroom
Push to 32K:                           10.83 GB   ✓ fits
Push to 64K:                           12.05 GB   ✓ tight but fits
```

---

## 6. Complete Optimization Inventory

### Tier 1: Core (Must Implement)

| # | Technique | Impact | Source |
|---|-----------|--------|--------|
| 1 | TurboQuant KV cache (turbo4_0) | 3.6× KV compression | Google Research, llama.cpp forks |
| 2 | Q4_K_M quantization + imatrix | ~55% size, best quality at 4-bit | llama.cpp tools |
| 3 | Falcon-H1-1.5B specialist | Better quality + near-zero KV | TII (hybrid SSM) |
| 4 | Ornith-1.0-9B primary | SOTA coding at 9B | DeepReinforce |
| 5 | Prefix KV cache reuse | 5–93% TTFT reduction | llama.cpp mainline |
| 6 | GBNF/XGrammar constrained output | 100% valid structured output | llama.cpp / XGrammar |
| 7 | Dynamic KV resize | Avoid upfront over-allocation | llama.cpp PR #21757 |
| 8 | Speculative decoding | 1.5–2× speedup on slow targets | llama.cpp mainline |
| 9 | TF-IDF + LogReg router | <1ms routing, >85% accuracy | scikit-learn |
| 10 | llama-swap model management | On-demand model loading | mostlygeek/llama-swap |

### Tier 2: High Value (Phase 2)

| # | Technique | Impact | Source |
|---|-----------|--------|--------|
| 11 | LLMLingua-2 prompt compression | 2–5× prompt reduction | Microsoft Research |
| 12 | MiniCache cross-layer KV merging | 1.53× additional compression | arxiv:2405.14366 |
| 13 | Host-memory prompt caching | Offload prefix to RAM | llama.cpp --cram |
| 14 | Tool Attention / NTILC | 95% reduction in tool tokens | arxiv:2604.21816 |
| 15 | TIDE early exit | 5–8% throughput gain | RightNow-AI/TIDE |
| 16 | N-gram speculative decoding | 1.5–2× on repetitive/code | llama.cpp mainline |
| 17 | HyFunc dynamic templating | Reduce boilerplate generation | arxiv:2602.13665 |
| 18 | Advanced GGUF quantizer | Mixed-precision per tensor | github: michaelw9999 |

### Tier 3: Experimental (Phase 3)

| # | Technique | Impact | Source |
|---|-----------|--------|--------|
| 19 | PoLar (Program-of-Layers) | Skip/loop layers per input | tianyi-lab/PoLar (ICML 2026) |
| 20 | BUDDY dynamic depth routing | Budget-driven layer selection | arxiv:2606.09514 |
| 21 | Token Sparse Attention | 3.23× attention at 128K | arxiv:2602.03216 |
| 22 | SparDA forecast-based sparse | 1.7× decode speedup | arxiv:2606.04511 |
| 23 | Lethe KV cache pruning | 2.56× throughput for reasoning | arxiv:2511.06029 |
| 24 | Dr.LLM per-layer routing | Skip/execute/repeat blocks | parameterlab/dr-llm |
| 25 | LayerSkip (Meta) | Self-speculative decoding | facebookresearch/LayerSkip |
| 26 | Mamba-3 | +1.8 pts over Mamba-2 at 1.5B | arxiv:2603.15569 |
| 27 | RWKV-7 specialist | Zero KV, constant memory | RWKV-LM |

### Tier 4: System-Level (Infrastructure)

| # | Technique | Impact | Source |
|---|-----------|--------|--------|
| 28 | Prefix cache prompt structure | 78% hit rate vs 3% | SGLang RadixAttention patterns |
| 29 | Multi-slot KV cache sharing | Amortize prefix across slots | llama.cpp server |
| 30 | KV cache disk persistence | Session resume without re-prefill | ds4 pattern |
| 31 | QAT quantized models | Better quality at same bpw | Google Gemma 4 QAT |

---

## 7. Memory & Context Design

### Memory Architecture

```
WORKING MEMORY (1-2K tokens)
  Current task context, last 2-5 turns, active tool outputs
  ↓ summarize when full
EPISODIC MEMORY (50-200 entries, ~200 tokens each)
  Summarized conversation history, key decisions, outcomes
  Retrieved via embedding similarity
  ↓ compress periodically
SEMANTIC MEMORY (20-100 entries, ~100 tokens each)
  User preferences, learned facts, project state
  Durable knowledge extracted from episodes
```

### Avoiding Quality Collapse

1. Never fill context window to 100%. Reserve 20–25% for generation.
2. Prune aggressively. Extract key information before dropping old turns.
3. Use the specialist (1.5B) for summarization — constrained task, good quality.
4. Embedding-based retrieval beats keyword search for episodic memory.
5. Track context health: token utilization, compression ratio, retrieval relevance.

### Prefix Caching Strategy

The single biggest latency win for multi-turn conversations:

```
Turn 1: [system_prompt | user_msg_1] → compute full, cache KV
Turn 2: [system_prompt | user_msg_1 | assistant_1 | user_msg_2]
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         cached (skip prefill)            ^^^^^^^^^^^^^^^
                                          only new tokens prefilled

TTFT reduction: 3-5 seconds → 300 milliseconds for turn 2+
```

**Critical:** Keep system prompt static. One character difference = full cache miss.

---

## 8. Benchmarking & Evaluation

### Benchmarks to Run

| Benchmark | Measures | Why It Matters Locally |
|-----------|----------|----------------------|
| GSM8K | Math reasoning | Core capability gap vs frontier |
| HumanEval / MBPP | Code generation | Key local use case, verifiable |
| MMLU | World knowledge | Hallucination rate under pressure |
| MT-Bench | Multi-turn quality | Context management effectiveness |
| IFEval | Format compliance | Router + constrained output accuracy |
| Custom: 50-turn agent task | Long-horizon reliability | The killer benchmark |

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Task completion rate | >80% structured, >60% open-ended | Correct task completion |
| Latency P50 | <5s simple, <15s complex | Time to complete response |
| Memory peak | <14 GB | Max RSS during task |
| Hallucination rate | <20% factual Q&A | Fact-check against ground truth |
| Context efficiency | <10% quality degradation over 20 turns | Quality vs context tokens |
| Routing accuracy | >85% | Correct model selection |
| Orchestration overhead | <15% of total tokens | Routing + verification cost |

### Fair Comparison Protocol

- Same total memory budget (orchestrated vs single model)
- Same latency budget
- Same task set (not cherry-picked)
- Report both wins and losses

---

## 9. Risks & Failure Modes

### When Orchestration Is Worse Than One Model

| Failure Mode | Cause | Mitigation | Severity |
|-------------|-------|------------|----------|
| Routing mistakes | Classifier sends hard task to 1B model | Confidence gate: <70% → primary | High |
| Cascading errors | Specialist produces bad output | Verify structured outputs | High |
| Coordination overhead | Multiple model calls + parsing | Latency budget; skip for simple tasks | Medium |
| Context bloat from orchestration | Router + specialist + plan consume tokens | Strict per-component budgets | Medium |
| Quality collapse under load | Summarization loses details | Embedding retrieval to pull back details | High |
| Over-engineering | Complex orchestration for simple tasks | Default to single model | Low |

### The Biggest Risk

**The orchestration layer becomes the project, not the product.** If a single 7B Q4 model completes >80% of your tasks adequately, orchestration may not be worth it. Always benchmark against the single-model baseline.

---

## 10. Phased Build Roadmap

### Phase 0: Foundation (Days 1–3)

**Goal:** Establish inference stack, validate memory budget, baseline benchmarks.

| Task | Verification |
|------|-------------|
| Build llama.cpp with TurboQuant (animehacker fork) | `llama-cli --version` works |
| Download Ornith-9B + Falcon-H1-1.5B GGUF | Files present in models/ |
| Generate imatrix for Ornith-9B on mixed corpus | imatrix.gguf generated |
| Quantize Ornith-9B with imatrix | PPL within 1% of reference |
| Load both models, measure memory | Total RSS < 8 GB |
| Baseline benchmarks | GSM8K, HumanEval, IFEval recorded |

**Exit criteria:** Both models loaded at <8 GB. TurboQuant quality acceptable.

### Phase 1: MVP Orchestration (Days 4–10)

**Goal:** Working two-model system with routing, constrained output, prefix caching.

| Task | Verification |
|------|-------------|
| TF-IDF router trained on 200 labeled examples | >85% accuracy on held-out set |
| Prefix caching configured | TTFT turn 2+ <500ms |
| GBNF constrained decoding for JSON | 100% valid output, zero parse failures |
| llama-swap configured | Swap completes in <5s |
| Context manager with token budgets | No context overflow errors |
| Basic memory (embed + retrieve last 5 turns) | Relevant context injected |

**Exit criteria:** Router >85% accuracy. 100% valid JSON. Prefix cache hit >70%.

### Phase 2: Optimization Stack (Days 11–21)

**Goal:** Stack compression and optimization techniques. Measure each independently.

| Task | Expected Gain |
|------|---------------|
| LLMLingua-2 integration | 2–3× fewer input tokens |
| Speculative decoding (specialist as draft) | 1.5× speedup if acceptance >65% |
| N-gram spec decoding | 1.5–2× on code completion |
| TIDE early exit on specialist | 5–8% throughput gain |
| Host-memory caching (--cram) | Save ~0.5 GB VRAM |
| Tool Attention (lazy schema loading) | Reduce tool tokens from ~10K to ~500 |
| Dynamic KV resize | Avoid wasting memory on short interactions |
| A/B test each technique | Quantify per-technique gain |

**Exit criteria:** Each technique measured. Combined >30% latency reduction.

#### Phase 2 Results (measured 2026-07-05)

| Task | Outcome | Measurement |
|------|---------|-------------|
| LLMLingua-2 integration | **Shipped, opt-in** | 56.5% token reduction (138→60 tok) on 5 sample tasks, 233ms/call, +118MB RSS (+339MB incl. torch). Wired into `ContextManager._truncate_to_budget()`, compresses history older than the last 2 turns before hard-dropping. Default `enabled: false` in `configs/compression.yaml`. |
| Speculative decoding (Falcon-H1 draft) | **SKIP** | Real llama-server run: Falcon-H1 and Ornith-9B have incompatible vocabs ("draft model bos tokens must match target model"), a fixed architectural constraint. Baseline Ornith-9B standalone measured: 15.24 tok/s, 4.20s avg latency (20 prompts). See `scripts/benchmark_spec_decode.py`. |
| N-gram spec decoding | **Not attempted** | Out of scope for this round. `--spec-type ngram-simple` (no draft model needed) remains an untested option for code-heavy tasks. |
| TIDE early exit on specialist | **SKIP** | Falcon-H1 is hybrid SSM/Mamba; TIDE's per-token early exit assumes stateless attention layers and would corrupt SSM recurrent state (correctness issue, not a tradeoff). Also HF-transformers-only (no GGUF/llama.cpp) and CUDA-only (no Metal/MPS). See `scripts/benchmark_tide.py` and the optimization log. |
| Host-memory caching (--cram) | **Shipped, opt-in, smaller gain than planned** | Real measurement on Falcon-H1: ~3% (~40MB) RSS reduction after idle, far below the planned ~0.5GB — Falcon-H1's KV cache is already near-zero due to its hybrid SSM architecture, so there's little to offload. `defaults.host_cache` in `configs/models.yaml`, default `false`. |
| Tool Attention (lazy schema loading) | **Shipped, real-world caveat found** | 93.5% token reduction on a simulated 50-tool registry (3200→207 tokens). Real end-to-end A/B run showed it as the *slowest* variant (9.28s vs 3.08s baseline p50) for a small 5-tool registry + 32-token generations — the fixed per-call `embed()` round-trip outweighs the token savings at this scale. Net benefit depends on registry size and generation length. |
| A/B test each technique | **Shipped** | `src/lore/ab_test.py` (`ABTest` class) + `scripts/run_ab_suite.py`, run against a real 20-task suite (`benchmarks/eval_tasks/standard.json`) across `baseline` / `plus_compression` / `plus_tool_attention` / `plus_all_combined`. Also surfaced a real bug: Ornith's chat template rejects multiple `system`-role messages, invisible to mocked unit tests. Full results and caveats in `docs/optimization-log.md`. |

**Exit criteria check:** Every technique was measured (or definitively ruled out with evidence). Combined (`plus_all_combined`) did not clear a clean >30% latency reduction over baseline in this run — the two "wins" (compression, tool attention) each carried real per-call overhead that partly offset their token savings at this task suite's scale (short generations, small tool registry, 800-token context budget). See the A/B results and caveats in `docs/optimization-log.md` for the full breakdown.

### Phase 3: Advanced Agentic (Days 22–35)

**Goal:** Long-horizon agentic tasks. Context management. Memory persistence.

| Task | Expected Gain | Status |
|------|---------------|--------|
| Hierarchical memory (working → episodic → semantic) | No quality collapse over 50+ turns | DONE |
| Context health monitoring | Detect degradation early | DONE |
| KV cache disk persistence | Instant session resume | DONE |
| MiniCache evaluation | 1.53× additional KV compression | SKIP — TurboQuant conflict |
| PoLar / BUDDY evaluation | 10–20% compute savings | SKIP — SSM incompatible, HF-only |
| HyFunc dynamic templating | 50% tool token reduction | PARTIAL — Tool Attention covers it |
| Multi-session management | Parallel task handling | DONE |

**Exit criteria:** 50+ turn sessions without quality collapse. Memory <14 GB.

### Phase 3.5: Wire + Verify + Evaluate (2026-07-07)

**Goal:** Close wiring gap + evidence-based evaluation of deferred techniques.

| Task | Result |
|------|--------|
| Wire HierarchicalMemory + ContextHealth into CLI | DONE |
| Session REPL commands (/save, /resume, /sessions, /switch) | DONE |
| Fix private attribute access (public API for context/health/session) | DONE |
| Verifier module (JSON/code validation + repair) | DONE |
| Dynamic context sizing (per-request budget heuristics) | DONE |
| MiniCache evaluation | SKIP — architectural conflict with TurboQuant + sparse attention |
| PoLar/BUDDY evaluation | SKIP — SSM recurrent state + HF-only runtime |
| HyFunc evaluation | PARTIAL_ADOPT — Tool Attention covers portable half |
| Multi-session (ActiveSession, /switch REPL command) | DONE |
| End-to-end integration test (30-turn pipeline) | DONE — 23 new tests |
| Documentation updates | DONE |

**Test count:** 122 passing (99 original + 23 new e2e).

### Phase 4: Orchestration Engine (2026-07-07)

**Goal:** Real task orchestration — decompose complex tasks, schedule workers, aggregate results.

| Task | Result |
|------|--------|
| Complexity estimator (heuristic, <1ms) | DONE — `src/lore/complexity.py` |
| Task decomposer (LLM-based planning) | DONE — `src/lore/decomposer.py` |
| Worker abstraction (scoped context per subtask) | DONE — `src/lore/worker.py` |
| Orchestrator (schedule, execute, aggregate) | DONE — `src/lore/orchestrator.py` |
| Prompt templates for subtask types | DONE — `src/lore/templates.py` |
| Dynamic model lifecycle (offload/reload) | DONE — in orchestrator |
| Wave-based scheduling with topological sort | DONE — sequential, parallel TBD |
| Verifier (JSON/code validation + repair) | DONE — `src/lore/verifier.py` |
| Dynamic context sizing | DONE — `src/lore/sizing.py` |
| CLI wired with orchestrator as entry point | DONE |
| Orchestrator unit tests (33 tests) | DONE |
| Real inference smoke tests | DONE — scripts exist, need M4 to run |

**Test count:** 155 passing (122 + 33 new orchestrator).

**Known issues (Phase 4.1):**
- Orchestrator accesses `server._processes` directly (needs public API)
- Specialist reload duplicates `start_all()` logic
- Circular import in `orchestrator._delegate_dispatch()` (deferred `from lore.cli`)
- Sequential wave execution (parallel structure ready, not exploited)
- N+1 memory stores per orchestrated task (workers + orchestrator both store)

### Phase 4.1: Hardening & Parallel Execution (pending)

**Goal:** Fix architectural issues from Phase 4. Enable parallel wave execution.

| Task | Status |
|------|--------|
| Public API for ModelServer (is_model_running, start_model, stop_model) | DONE |
| Remove circular import in orchestrator | DONE |
| ContextManager.set_budget() method | DONE |
| Deduplicate memory storage in orchestrated path | DONE |
| Parallel wave execution (ThreadPoolExecutor) | DONE |

### Phase 4.2: Live Benchmark Model Selection (pending)

**Goal:** LORE proactively discovers better models on HuggingFace. User approves downloads. Only orchestrator model is user-locked.

| Task | Status |
|------|--------|
| Leaderboard scanner (HF parquet + individual leaderboards) | PENDING |
| Upgrade detection (compare installed vs available, filter by size+GGUF) | PENDING |
| Upgrade notifier (show comparison table, ask user approval) | PENDING |
| Registry with auto-select from local models | PENDING |
| Auto-download approved upgrades | PENDING |
| Model-based classifier (specialist as NLU, replaces regex) | PENDING |
| Wire classifier into orchestrator + decomposer hints | PENDING |
| Skip orchestration on fallback plan | PENDING |
| CLI /upgrades, /models commands | PENDING |
| auto_select_models.py discovery script | PENDING |
| Config: orchestrator lock, auto_select, leaderboard | PENDING |
| Tests | PENDING |

### Phase 5: Benchmark & Harden (future)

**Goal:** Comprehensive evaluation. Cut dead weight. Document everything.

| Task | Deliverable |
|------|-------------|
| Full benchmark suite | Results table with all metrics |
| Memory + latency profiling | Peak memory, fragmentation, waterfall chart |
| A/B: orchestrated vs single model | Win/loss/neutral per task type |
| Failure mode catalog | Decision rules for bypass |
| Final documentation | Architecture, config, runbook |

**Exit criteria:** Clear win/loss map. Production-ready configuration.

---

## 11. Top Design Decisions

| # | Decision | Why It Matters Most |
|---|----------|-------------------|
| **1** | **Build llama.cpp with TurboQuant** | Unlocks "all three" config. 4.57× KV compression is the single biggest lever. |
| **2** | **Ornith-1.0-9B as primary** | Beats Gemma4-31B on coding at 9B. Purpose-built for agentic coding. |
| **3** | **Falcon-H1-1.5B as specialist** | Best quality at 1.5B. Hybrid SSM = near-zero KV. Fully supported in llama.cpp. |
| **4** | **Validate TurboQuant on Qwen architectures** | +5–8% PPL sensitivity is real. Must test on Ornith. Mixed-precision KV is fallback. |
| **5** | **Default to single model** | Orchestration only for tasks that measurably benefit. Don't add complexity for its own sake. |

---

## 12. References

### Models
- Ornith-1.0: github.com/deepreinforce-ai/Ornith-1 | huggingface.co/deepreinforce-ai/Ornith-1.0-9B-GGUF
- Falcon-H1: huggingface.co/tiiuae/Falcon-H1-1.5B-Instruct-GGUF
- Gemma 4: ai.google.dev/gemma/docs/core/model_card_4
- Qwen3.5: github.com/QwenLM/Qwen3.5
- RWKV-7: arxiv:2503.14456 | github.com/RWKV/RWKV-LM
- Zamba2: github.com/Zyphra/Zamba2
- AI-Flow-Ruyi-7B: github.com/TeleAI-AI-Flow/AI-Flow-Ruyi
- Nemotron-H: research.nvidia.com/labs/adlr/nemotronh/

### KV Cache Compression
- TurboQuant: arxiv:2504.19874 (ICLR 2026) | github.com/animehacker/llama-turboquant
- MiniCache: arxiv:2405.14366
- Mixed-precision KV: github.com/ggml-org/llama.cpp commit e889fbd
- Dynamic KV: github.com/ggml-org/llama.cpp PR #21757

### Inference Optimization
- TIDE early exit: github.com/RightNow-AI/TIDE
- PoLar (ICML 2026 Oral): github.com/tianyi-lab/PoLar
- BUDDY: arxiv:2606.09514
- LayerSkip: github.com/facebookresearch/LayerSkip
- Dr.LLM: github.com/parameterlab/dr-llm
- Token Sparse Attention: arxiv:2602.03216
- SparDA: arxiv:2606.04511
- Lethe: arxiv:2511.06029

### Agentic Optimization
- Tool Attention: arxiv:2604.21816 | github.com/asadani/tool-attention
- NTILC: arxiv:2606.06566
- HyFunc: arxiv:2602.13665 | github.com/MrBlankness/HyFunc
- XGrammar-2: arxiv:2601.04426

### Prompt Compression
- LLMLingua: github.com/microsoft/LLMLingua
- LLMLingua-2: aclanthology.org/2024.findings-acl.57

### Serving & Infrastructure
- llama-swap: github.com/mostlygeek/llama-swap
- SGLang RadixAttention: github.com/sgl-project/sglang
- Host-memory caching: github.com/ggml-org/llama.cpp discussion #20574
- Prefix KV reuse: github.com/ggml-org/llama.cpp discussion #13606
- Advanced GGUF quantizer: github.com/michaelw9999/advanced-gguf-quantizer

### Orchestration Reference
- Sakana Fugu: arxiv:2606.21228 | sakana.ai/fugu
- Fugu Ultra: sakana.ai/fugu-release
- Mamba-3: arxiv:2603.15569
- BitMamba-2: huggingface.co/rasatavohary/BitMamba-2-1B
- DF-SSM: github.com/cs-cmyk/df-ssm
