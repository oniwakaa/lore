# KV Cache Research Report: KVzip, AsymKV, ChunkKV

## Executive Summary

Three KV cache compression techniques evaluated against LORE's current TurboQuant baseline. **None are available in llama.cpp** — all are PyTorch/Python research implementations. All three operate on a different compression axis (eviction/merging) than TurboQuant (quantization) and should theoretically compose. Practical integration requires custom C/C++ implementation or waiting for upstream llama.cpp adoption.

## Current Baseline: TurboQuant (turbo4_0)

| Property | Value |
|----------|-------|
| Compression | 3.6× (3-bit keys, 2-bit values) |
| PPL impact | +5-8% on Qwen architectures |
| llama.cpp | Yes (TheTom/llama-cpp-turboquant fork, Metal kernels) |
| Memory at 16K | 0.61 GB (Ornith-9B), 0.04 GB (Falcon-H1) |
| Status | **Production in LORE** |

TurboQuant compresses each KV entry's precision. It does not reduce the number of entries.

---

## 1. KVzip (NeurIPS 2025 Oral)

**Paper:** Query-agnostic KV Cache Eviction
**Code:** [github.com/snu-mllab/kvzip](https://github.com/snu-mllab/kvzip)

### Technique
Evicts KV cache entries that are irrelevant to future queries, using context reconstruction. Query-agnostic: eviction decisions made without knowing future queries. Achieves 3-4× memory reduction and 2× latency improvement.

### Supported Models
Qwen3, Qwen2.5, Gemma3, LLaMA3 — includes Qwen3.5 architecture family (Ornith's base).

### llama.cpp Integration
**NOT available.** KVzip is a PyTorch/Python implementation using HuggingFace transformers. No llama.cpp patch, PR, or fork exists. Would require:
1. Porting eviction logic to C/C++
2. Integrating with llama.cpp's KV cache management (`llama_kv_cache`)
3. Adding context reconstruction support for eviction decisions
4. Metal kernel support for Apple Silicon

### TurboQuant Compatibility
**Should compose.** KVzip reduces the number of KV entries (eviction). TurboQuant reduces the size of each entry (quantization). They operate on orthogonal axes:
- KVzip: N entries → N' entries (where N' < N)
- TurboQuant: 16 bytes/entry → 4.4 bytes/entry

Combined: N × 16 bytes → N' × 4.4 bytes = (N'/N) × (4.4/16) = dual compression.

### LORE Applicability
Ideal for session persistence — compressed KV cache could be reused across different queries without re-computation. However, implementation effort is high (C++ port needed).

### Action: Document for future. Monitor for llama.cpp PR.

---

## 2. AsymKV (NeurIPS 2025)

**Paper:** Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry
**Code:** [github.com/the-scale-lab/Asymkv](https://github.com/the-scale-lab/Asymkv)

### Technique
Exploits the observation that adjacent keys receive similar attention weights (local homogeneity) while adjacent values have distinct distributions (heterogeneity). Merges homogeneous keys (reducing key count) while preserving heterogeneous values. Training-free, mathematically proven lossless.

### Compression
50%+ cache reduction with no quality loss.

### llama.cpp Integration
**NOT available.** Python/PyTorch implementation. No llama.cpp support. Would require:
1. Key similarity detection in C/C++
2. Dynamic key merging in KV cache
3. Value preservation logic
4. Metal kernel modifications

### TurboQuant Compatibility
**Should compose.** AsymKV merges adjacent similar keys (reduces entry count). TurboQuant quantizes entries (reduces per-entry size). Different axes:
- AsymKV: N keys → N/2 keys (merge similar)
- TurboQuant: 16 bytes/key → 4.4 bytes/key

Combined: N × 16 → (N/2) × 4.4 = 7.3× total compression (vs 3.6× TurboQuant alone).

### LORE Applicability
Particularly relevant for Ornith-9B's hybrid SSM architecture — only 8 attention layers carry KV cache, so merging within those layers could significantly reduce the already-small cache. However, the hybrid SSM layers (Gated DeltaNet) carry recurrent state, not KV pairs, so AsymKV only applies to the 8 attention layers.

### Action: Document for future. Monitor for llama.cpp PR.

---

## 3. ChunkKV (2025)

**Paper:** Semantic-Preserving KV Cache Compression (arxiv:2502.00299)
**Code:** Not publicly available

### Technique
Groups tokens into semantic chunks, then compresses each chunk while preserving semantic information. Unlike token-level eviction, operates on chunk level to maintain coherence.

### Compression
Targets up to 70% memory reduction (the KV cache's share of total memory).

### llama.cpp Integration
**NOT available.** Paper-only, no public code. No llama.cpp support.

### TurboQuant Compatibility
**Potentially conflicts.** ChunkKV may restructure KV entries (chunking), which could interfere with TurboQuant's per-entry quantization. Needs investigation if code becomes available.

### LORE Applicability
Low priority — no public implementation, and chunk-based restructuring is complex for hybrid SSM architectures where attention layers are sparse (every 4th layer in Ornith).

### Action: Low priority. Monitor for code release.

---

## Comparison Matrix

| Technique | Axis | Compression | Quality Loss | llama.cpp | Composes with TurboQuant | Implementation Effort |
|-----------|------|-------------|-------------|-----------|-------------------------|---------------------|
| TurboQuant | Quantization | 3.6× | +5-8% PPL | Yes (fork) | — (baseline) | Done |
| KVzip | Eviction | 3-4× | Minimal | No | Yes (orthogonal) | High (C++ port) |
| AsymKV | Merging | 2× | Lossless | No | Yes (orthogonal) | Medium-High |
| ChunkKV | Chunking | ~70% | Minimal | No | Uncertain | High (no code) |

## Recommendation

1. **Stay on TurboQuant** as the production KV cache strategy. It works, it's integrated, it's tested.
2. **Monitor KVzip** for llama.cpp integration. If a PR appears, evaluate immediately — it's the most impactful addition (eviction + quantization = maximum compression).
3. **Track AsymKV** for lossless compression potential. The 2× lossless + 3.6× TurboQuant = 7.3× total is very attractive for 16 GB devices.
4. **Configurable KV cache** (3B): Enable users to switch between turbo4, q8_0, q4_0, fp16 via config. Already implemented in `configs/models.yaml` and `src/lore/models.py`.

## Configurable KV Cache Strategy

LORE now supports configurable KV cache types via `configs/models.yaml`:

```yaml
defaults:
  kv_cache_type: turbo4  # options: turbo4, q8_0, q4_0, fp16
```

| Type | Compression | Quality | Use Case |
|------|-------------|---------|----------|
| turbo4 | 3.6× | +5-8% PPL | Default, maximum compression |
| q8_0 | 2× | <1% PPL | Quality-sensitive tasks |
| q4_0 | 4× | +2-3% PPL | Maximum llama.cpp native compression |
| fp16 | 1× (baseline) | None | Quality validation, debugging |

The `start_model()` method in `src/lore/models.py` reads this config and passes it as `-ctk` and `-ctv` args to llama-server.
