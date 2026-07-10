# Speculative Decoding Research Report

## Summary

LORE uses two speculative decoding strategies:
- **ngram-simple** for specialist (Falcon-H1-1.5B) — no draft model needed
- **EAGLE-3** for primary (Ornith-1.0-9B) — requires trained draft head checkpoint

## ngram-simple (Specialist)

Implemented in `src/lore/models.py`. Enabled by default via `configs/models.yaml`:
```yaml
defaults:
  speculative_decoding: true
```

Passes `--spec-type ngram-simple` to llama-server for specialist role. No draft model required. Uses n-gram lookup from prior context to predict likely next tokens. Low overhead, modest speedup for repetitive patterns (code, structured output).

## EAGLE-3 (Primary)

### Status: Config-ready, awaiting llama.cpp merge

EAGLE-3 is the current SOTA speculative decoding algorithm (NeurIPS 2025). It uses a trained draft head that predicts next-token features rather than tokens directly, enabling tree-based verification.

### Ornith-1.0-9B Compatibility

Ornith-1.0-9B is based on **Qwen3.5 architecture** (hybrid SSM: 8 attention + 24 Gated DeltaNet layers).

A compatible EAGLE-3 checkpoint exists:
- **`BLR2/Qwen3.5-9B-Eagle3-ShareGPT`** on HuggingFace
- Trained specifically for Qwen3.5-9B, matching Ornith's architecture
- Must be converted to GGUF format using `convert_hf_to_gguf.py`

### llama.cpp Integration

PR [#24593](https://github.com/ggml-org/llama.cpp/pull/24593) adds EAGLE-3 support for Qwen3.5 & Qwen3.6 in llama.cpp. Key changes:
- `common/speculative.cpp`: EAGLE-3 implementation with deferred-boundary stash for hybrid models
- `src/models/qwen35.cpp`: Layer input tracking for EAGLE-3 embedding extraction
- `tools/server/server-context.cpp`: Checkpoint state serialization for speculative state

The PR handles a critical hybrid-model issue: EAGLE-3's draft trails the target by one position, and on recurrent/hybrid architectures (like Qwen3.5's Gated DeltaNet), the deferred boundary `g_embd` must be stashed in checkpoints (~20 KB/checkpoint overhead).

**As of 2026-07-10: PR is closed/merging. Not yet in mainline llama.cpp.**

### LORE Configuration

When the PR merges and LORE's TurboQuant fork incorporates it:

1. Convert the draft checkpoint:
```bash
python convert_hf_to_gguf.py \
    "BLR2/Qwen3.5-9B-Eagle3-ShareGPT" \
    --outtype q4_k_m \
    --target-model-dir "deepreinforce-ai/Ornith-1.0-9B" \
    --outfile models/ornith-9b-eagle3-draft-Q4_K_M.gguf
```

2. Add to `configs/models.yaml`:
```yaml
primary:
  eagle3_draft_path: models/ornith-9b-eagle3-draft-Q4_K_M.gguf
```

3. LORE automatically passes `--spec-type draft-eagle3 -md <path>` to llama-server.

### Expected Performance

Based on Qwen3.6-27B benchmarks from PR #24593 (DGX Spark, Q4_K_M):
- **1.78x decode speedup** (12.56 → 22.31 tok/s overall)
- **1.71x latency reduction** (25.4s → 14.9s overall)
- ~50% acceptance rate across categories
- Coding: 1.85x decode speedup (12.59 → 23.30 tok/s)

On Apple Silicon M4 with Ornith-9B, expect lower speedup due to:
- Metal vs CUDA differences in speculative verification
- Smaller model (9B vs 27B) means less compute gap between draft and target
- Hybrid SSM layers reduce KV cache pressure but add recurrent state complexity

Conservative estimate: **1.3-1.5x decode speedup** on M4 Metal.

### Memory Impact

EAGLE-3 draft head is a small model (~1-2 GB at Q4_K_M for a 9B-class draft). This fits within LORE's memory budget:
```
Current budget:          6.59 GB (both models, turbo4, 16K)
EAGLE-3 draft head:    + ~1.0 GB (Q4_K_M, loaded only for primary)
                        ─────────
With EAGLE-3:           7.59 GB  (6.41 GB headroom — still safe)
```

### TurboQuant Compatibility

TurboQuant (`turbo4_0`) compresses KV cache. EAGLE-3 operates on token embeddings, not KV cache directly. They should compose: TurboQuant on the target context, EAGLE-3 draft head uses its own small context. No known conflict, but needs validation testing once both are active.

### Action Items

1. **Wait for PR #24593 merge** into llama.cpp mainline
2. **Cherry-pick into TurboQuant fork** (`TheTom/llama-cpp-turboquant`)
3. **Convert `BLR2/Qwen3.5-9B-Eagle3-ShareGPT`** to GGUF Q4_K_M
4. **Benchmark**: measure actual speedup on M4 Metal with Ornith-9B
5. **Validate TurboQuant + EAGLE-3 coexistence**

## Alternative: DFlash for Qwen3.5

A competing approach, DFlash, has been mentioned for Qwen3.5 speculative decoding. Less mature than EAGLE-3. Monitor as backup option if EAGLE-3 integration proves problematic.
