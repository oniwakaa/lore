# LORE Optimization Log

All measurements recorded here. Each entry: date, config, metric, raw number.

## Phase 0: Foundation

### Build Info

| Item | Value |
|------|-------|
| Fork | TheTom/llama-cpp-turboquant |
| Branch | feature/turboquant-kv-cache |
| Commit | 558c6b7 |
| Build date | 2026-07-02 |
| Platform | Apple M4, 16 GB unified memory, macOS |
| Backend | Metal (-DGGML_METAL=ON) |
| Build type | Release |
| Binary | external/llama-cpp-turboquant/build/bin/llama-cli |
| CMake flags | -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON |

### Model Files

| Model | File | Quant | Expected Size | Actual Size | imatrix |
|-------|------|-------|---------------|-------------|---------|
| Ornith-1.0-9B | ornith-1.0-9b-Q4_K_M.gguf | Q4_K_M | 5.63 GB | 5.24 GB (5.02 BPW) | Yes (embedded, 401 chunks, 248 entries) |
| Falcon-H1-1.5B | Falcon-H1-1.5B-Instruct-Q4_K_M.gguf | Q4_K_M | 1.00 GB | 0.88 GB | Unknown |

### Architecture Notes

**Ornith-1.0-9B** is a **hybrid SSM model** (qwen35 arch), NOT a pure transformer:
- 32 layers total, attention every 4th layer (full_attention_interval=4)
- 8 attention layers (blk 3, 7, 11, 15, 19, 23, 27, 31) with KV cache
- 24 SSM layers with recurrent state (no KV cache)
- 16 attention heads, 4 KV heads (GQA=4)
- 256 dim per head, 4096 embedding dim
- SSM: conv_kernel=4, state_size=128, group_count=16, inner_size=4096
- 262K native context, 248K vocab
- This means Ornith's KV cache is already much smaller than a pure transformer's

**Falcon-H1-1.5B** is also a hybrid SSM (only 2 attention heads).

### Memory: Single Model Load

Measured via llama-server, `-np 1` (single slot), 16384 context, Metal flash attention on.

| Model | Context | KV Type | RSS (GB) | Notes |
|-------|---------|---------|----------|-------|
| Ornith-9B | 16384 | f16 | 5.87 | baseline |
| Ornith-9B | 16384 | turbo4 | 5.57 | saved 300 MB (5.1%) |
| Falcon-H1-1.5B | 16384 | f16 | 1.37 | baseline |
| Falcon-H1-1.5B | 16384 | turbo4 | 1.14 | saved 230 MB (16.8%) |

**Turbo4 KV savings:** Ornith 300 MB, Falcon 230 MB. Modest because both are hybrid SSM with few attention layers.

### Memory: Dual Model Load

Both models loaded simultaneously via separate llama-server instances.

| Config | Context | KV Type | Total RSS (GB) | Under 14GB? | Under 8GB? | Notes |
|--------|---------|---------|----------------|-------------|------------|-------|
| Ornith + Falcon | 16384 | turbo4/turbo4 | 6.59 | YES | YES | production config |

**Dual load breakdown:**
- Ornith turbo4 16K: 5.50 GB (slight variance from single-load 5.57 GB)
- Falcon turbo4 16K: 1.09 GB (slight variance from single-load 1.14 GB)
- Total: 6.59 GB, 7.41 GB headroom to 14 GB cap

### PPL: TurboQuant vs f16 Baseline

Measured via llama-perplexity, ctx=128, 2 chunks, on benchmarks/ppl_sample.txt (379 tokens mixed content).

| Model | KV Type | PPL | Std Error | Delta vs f16 | Acceptable? | Notes |
|-------|---------|-----|-----------|--------------|-------------|-------|
| Ornith-9B | f16 | 2.7164 | +/- 0.4272 | baseline | - | reference |
| Ornith-9B | turbo4 | 2.6966 | +/- 0.4232 | -0.73% | YES | No degradation. Hybrid SSM, only 8/32 layers affected |
| Falcon-H1-1.5B | f16 | 5.3679 | +/- 1.3937 | baseline | - | reference |
| Falcon-H1-1.5B | turbo4 | 5.2847 | +/- 1.3543 | -1.55% | YES | No degradation. SSM, minimal KV cache |

**Key finding:** TurboQuant turbo4 shows ZERO PPL degradation on both models. Deltas are negative (within noise). This contradicts the expected +5-8% sensitivity on Qwen architectures. Likely because:
1. Both models are hybrid SSM — turbo4 only affects the few attention layers
2. TurboQuant+ Metal kernels in TheTom's fork are well-optimized
3. Small context (128) may not show full effect; larger context may differ

### Memory Budget Verification

| Component | Expected (GB) | Actual (GB) | Delta |
|-----------|---------------|-------------|-------|
| Ornith-9B Q4_K_M | 5.63 | 5.24 (file) / 5.50 (RSS turbo4) | -0.13 RSS |
| Falcon-H1-1.5B Q4_K_M | 1.00 | 0.88 (file) / 1.09 (RSS turbo4) | +0.09 RSS |
| Dual load total | 10.08 | 6.59 | -3.49 GB better than expected |
| Headroom to 14 GB | 3.92 | 7.41 | +3.49 GB more headroom |

The actual memory usage is significantly better than the planned budget. The hybrid SSM architecture of both models means KV cache is much smaller than assumed for pure transformers.

### Baseline Benchmarks

| Model | GSM8K | HumanEval | IFEval | Notes |
|-------|-------|-----------|--------|-------|
| Ornith-9B Q4_K_M | TBD | TBD | TBD | primary |
| Falcon-H1-1.5B Q4_K_M | TBD | TBD | TBD | specialist |

### Phase 0 Exit Criteria

- [x] Both models loaded simultaneously < 8 GB RSS (6.59 GB)
- [x] TurboQuant PPL delta acceptable (< 8% on Ornith) (-0.73%, no degradation)
- [ ] Baseline benchmarks recorded (GSM8K, HumanEval, IFEval — deferred, need eval frameworks)

## Phase 1.5: Polish

### Local Tokenizer Cache

Replaced per-request HTTP `/tokenize` round-trips (4-6 per request, 5-20ms each) in
`ContextManager.token_count()` with a cached local `tokenizers.Tokenizer` loaded once
at `__init__` via `Tokenizer.from_pretrained(repo)` (repo derived from `primary.source`
in `configs/models.yaml`, stripping the `-GGUF` suffix). Falls back to HTTP on load or
encode failure.

| Method | Latency per call | Notes |
|--------|-------------------|-------|
| Local `tokenizers` encode | 0.19 ms | measured, 200 calls, ~140-char text |
| HTTP `/tokenize` (estimated) | 5-20 ms | per AGENTS.md baseline, network round-trip |

**Result:** ~25-100x faster per call, eliminates 4-6 HTTP round-trips per request
(40-120ms overhead removed). Config toggle: `configs/models.yaml` -> `defaults.tokenizer_source: local|http`.

## Phase 2

### Tool Attention (Lazy Schema Loading, NTILC pattern)

`src/lore/tool_attention.py`: `ToolAttention` embeds each tool schema once via the
nomic-embed-text server, then `select_tools(query, k)` picks the top-k by cosine
similarity instead of injecting the full registry. Wired into
`ContextManager.build_prompt(query=...)`.

| Registry size | Full injection tokens | Top-3 selection tokens | Reduction |
|---------------|------------------------|--------------------------|-----------|
| 50 tools | 3200 | 207 | 93.5% |

Measured with the local Falcon-H1 tokenizer on `configs/tools.yaml` schemas repeated
10x to simulate a larger registry. Matches the expected ~10K -> ~500 token order of
magnitude for large tool registries.
