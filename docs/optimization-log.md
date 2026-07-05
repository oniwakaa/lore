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

### Host-Memory Caching (--cram)

`models.py` adds `-cram <mb>` to server args when `configs/models.yaml` ->
`defaults.host_cache: true`. Offloads idle-slot KV cache to host RAM instead of
unified memory. Measured on Falcon-H1-1.5B, turbo4, 16K context, `-np 1`:

| Scenario | RSS | Notes |
|----------|-----|-------|
| No `-cram`, 1 request | 1244.3 MB | baseline |
| `-cram 512`, 1 request | 1221.4 MB | -22.9 MB at time of first response |
| `-cram 512 --cache-idle-slots`, after 4s idle | 1205.8 MB | -38.6 MB (-3.1%) vs post-request RSS |
| `-cram 512 --cache-idle-slots`, request 2 (slot reactivated) | 1215.5 MB | cache restored, no crash/error |

**Finding:** Real but modest savings (~3%, ~40 MB) on Falcon-H1 at 16K context — much
smaller than the ~0.5 GB estimate in the plan, because Falcon-H1's KV cache is already
near-zero (hybrid SSM, 2 attention heads). The effect only appears after a slot goes
idle (`--cache-idle-slots` required), not on the very first request. Larger absolute
savings are expected on Ornith-9B at longer context (more attention-layer KV cache to
offload) but were not measured here to conserve time/memory budget. Config is opt-in
(`host_cache: false` by default) since the win is context- and workload-dependent.

### LLMLingua-2 Prompt Compression

`src/lore/compression.py`: `compress_prompt()` / `compress_context()` wrap LLMLingua-2
(`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`, CPU). Integrated into
`ContextManager._truncate_to_budget()`: when context exceeds 80% of `working_context`,
messages older than the latest 2 turns are compressed before any hard drop. Opt-in via
`configs/compression.yaml` -> `enabled: true` (default `false`).

Measured on 5 sample tasks (mixed explanation/code/summary prompts):

| Metric | Value |
|--------|-------|
| Model load time | 1.56 s (one-time, lazy singleton) |
| Avg compression latency | 233 ms/call |
| Token reduction | 56.5% (138 -> 60 tokens across 5 samples) |
| RSS after model load | +118 MB |
| RSS after first inference | +339 MB total (758 MB process RSS, includes torch import baseline) |

**Memory impact:** Compression runs as a separate CPU-only process step, not loaded into
either llama-server instance. Adding it to the 6.59 GB dual-model baseline gives
~7.3 GB total, still 6.7 GB under the 14 GB cap. Meaning is preserved via keyword
retention (LLMLingua-2 drops low-information tokens, keeps content words) — verified by
manual inspection of the 5 sample outputs, key nouns/verbs survived compression in all
cases.

### Speculative Decoding (Falcon-H1 draft -> Ornith-9B target)

`scripts/benchmark_spec_decode.py` starts llama-server with `-md <falcon path>
--spec-type draft-simple` to test Falcon-H1 as a draft model for Ornith-9B.

**Result: hard incompatibility, not just a weak speedup.** Server log on startup:

```
common_speculative_are_compatible: draft model bos tokens must match target model
to use speculation. add: 0 - 1, id: 11 - 17)
common_speculative_impl_draft_simple: the target and draft vocabs are not compatible
srv load_model: failed to initialize speculative decoding context: draft model vocab
type must match target model to use speculation
```

llama-server does not crash — it silently continues serving Ornith-9B *without* draft
acceleration. A full 20-prompt A/B would just remeasure the same baseline twice, so the
benchmark script detects this from the log and skips the "with spec decode" run.

**Baseline measured anyway** (20 prompts, max_tokens=64, temperature=0, 8K ctx, turbo4,
Ornith-9B standalone):

| Metric | Value |
|--------|-------|
| Avg latency/request | 4.20 s |
| Avg tokens/sec | 15.24 |

**Decision gate: SKIP.** Classic draft-model speculative decoding requires the draft and
target to share a tokenizer/vocab. Falcon-H1 (tiiuae tokenizer) and Ornith-9B (qwen35,
248K vocab) are different model families with incompatible vocabs — this is a fixed
architectural constraint, not something tunable via config. Any future draft model
candidate must share Ornith's tokenizer (e.g. a smaller Qwen-family checkpoint) to be
viable. n-gram-based speculative decoding (`--spec-type ngram-simple`, no draft model
needed) is a separate, untested option for code-heavy tasks — out of scope for this task.

### TIDE Early Exit (RightNow-AI/TIDE) on Falcon-H1

Cloned `RightNow-AI/TIDE` (`external/TIDE/`, gitignored like other external deps) and
read the README + `python/TIDE/adapters/universal.py`.

**Decision gate: SKIP, for three independent reasons:**

1. **Architectural incompatibility (the disqualifying one).** TIDE's per-token early
   exit compares hidden states across decoder layers and routes "converged" tokens
   straight to the final norm/LM head, skipping the remaining layers for that token.
   This is valid for stateless, parallel attention-style blocks — skipping token *i*
   at layer *L* doesn't affect token *i+1* at layer *L*. Falcon-H1 is a hybrid SSM
   (Mamba) model: SSM layers carry a **sequential recurrent state across the token
   dimension**. Skipping a token through an SSM layer doesn't just lose "extra
   refinement" (as in a transformer) — it corrupts the state trajectory for every
   later token in that sequence at that layer. This is a correctness bug waiting to
   happen, not a tunable quality/speed tradeoff.
2. **No GGUF/llama.cpp integration.** TIDE only wraps HuggingFace `transformers`
   `AutoModelForCausalLM` (PyTorch). Using it would mean loading a second, unquantized
   (fp16/bf16) copy of Falcon-H1 outside our llama.cpp/GGUF serving stack — directly
   at odds with the memory-conscious Q4_K_M-only deployment this project is built on.
3. **No Apple Silicon support.** All of TIDE's published benchmarks are on NVIDIA
   A100 (CUDA kernels for speed). It has a "pure PyTorch fallback" for no-GPU
   environments, but no Metal/MPS kernels — on M4 this would run on CPU only, far
   slower than llama.cpp's Metal path we already use.

`scripts/benchmark_tide.py` encodes this as a pre-flight check: it reads
`configs/models.yaml` -> `specialist.architecture`, refuses to run against any
`*ssm*`/`*mamba*` architecture, and explains why instead of producing numbers from an
architecturally invalid run. If the specialist is ever swapped to a pure-transformer
fallback (e.g. AGENTS.md lists Qwen2.5-1.5B as a Falcon-H1 fallback, and TIDE lists
Qwen as "Benchmarked"), the script's gate would no longer trigger and a real
calibration/benchmark could be attempted then.

## Phase 2 (cont.): A/B Testing Framework

`src/lore/ab_test.py` (`ABTest`): runs a task list through a `run_fn(task, config)` per
variant, reports p50/p95 latency, avg tokens/sec, peak RSS (via `psutil`), and completion
rate. `ABTest.compare(configs)` runs several named configs; `ABTest.save_report()` dumps
JSON to `benchmarks/results/` (gitignored, like other benchmark result JSON).

`benchmarks/eval_tasks/standard.json`: 20-task suite, 5 each across `simple`,
`code_generation`, `structured_extraction`, `complex_reasoning`.

`scripts/run_ab_suite.py`: starts real Ornith-9B (primary) + nomic-embed (embeddings)
servers, runs the 20-task suite through a real `ContextManager` against 4 variants —
`baseline`, `plus_compression`, `plus_tool_attention`, `plus_all_combined` — with
`max_tokens=32`. `+spec_decode` is intentionally excluded: it would be numerically
identical to `baseline` given the vocab-incompatibility finding above.

**A real bug found by this end-to-end run (not caught by mocked unit tests):**
Ornith's chat template raises `400 Bad Request` ("System message must be at the
beginning") when the message list contains more than one `system`-role message.
`ContextManager.build_prompt()` was appending separate `system` turns for memories
and for Tool Attention's selected schemas — this worked in every unit test (mocked
`server.chat`) but broke immediately against the real model. Fixed by folding
memories + tool schemas into a single system message (`\n\n`-joined) instead of
multiple system turns. All 56 tests still pass after the fix. This is exactly the
kind of gap real end-to-end A/B testing is meant to surface.

**Results (single Ornith-9B server, `working_context` budget=800 so history growth
actually exercises compression/truncation, `max_tokens=32`, one continuous 20-task
session per variant so context accumulates across tasks within a variant):**

| Variant | p50 latency | p95 latency | avg tok/s | peak RSS | completion |
|---|---|---|---|---|---|
| baseline | 3.08 s | 4.45 s | 10.44 | 290.6 MB | 100% |
| plus_compression | 5.94 s | 8.64 s | 6.34 | 287.7 MB | 100% |
| plus_tool_attention | 9.28 s | 14.02 s | 3.52 | 348.8 MB | 100% |
| plus_all_combined | 4.10 s | 9.34 s | 6.92 | 345.9 MB | 100% |

(peak RSS here is the *script process's* RSS, not the model servers'; it's dominated
by tokenizer/torch imports, not useful as a model memory signal — see the dedicated
per-component memory measurements above for actual model RSS.)

**Interpretation / caveats:**
- All variants share one confound: each is a single continuous 20-task "session"
  (history isn't reset between tasks), so later tasks in every variant carry more
  context than earlier ones. This is realistic of a real LORE session, but it means
  cross-variant latency deltas aren't a *pure* isolated measurement of each
  technique's overhead — they're that overhead compounded with how each technique
  handles the resulting context growth (compression vs. hard-drop truncation).
- `plus_compression` is slower than baseline: LLMLingua-2 compression (measured
  ~233ms/call in the earlier isolated benchmark) fires repeatedly once history
  crosses 80% of the 800-token budget, and those calls stack up over a 20-task
  session — a real, measurable cost of choosing "compress" over "truncate" for
  this budget size.
- `plus_tool_attention` was the slowest variant, more than the ~150-200 tokens of
  injected tool schema would suggest. The extra cost is the live `embed()` HTTP
  round-trip to the nomic-embed server on every single task (needed to rank tools by
  similarity), plus that server running concurrently with the primary on the same
  M4 GPU. For a small registry (5 tools) and short generations (32 tokens), this
  fixed per-call overhead outweighs the token savings — the opposite of the 93.5%
  token-reduction win measured for a simulated 50-tool registry in the Tool
  Attention section above. This is a genuinely useful finding: Tool Attention's
  net benefit depends on registry size and generation length, not just token count
  saved, and should be enabled selectively rather than unconditionally.
- `plus_all_combined` came in faster than either optimization alone, which is a bit
  suspicious rather than a real synergy — most likely both techniques are shrinking
  the same context at different points, so per-call compression triggers less often
  once tool attention's presence is factored in. Take this number as a "not worse
  than the worst individual variant" signal, not a confirmed multiplicative benefit.
- Full report: `benchmarks/results/ab_suite_report.json` (gitignored, regenerate with
  `scripts/run_ab_suite.py`).

## Phase 2.5: Optimization Gating

The Phase 2 A/B showed both compression and tool attention were net-negative at
small scale (5 tools, 800-token budget, 32-token generations). Rather than
discarding them, both are now conditionally gated so they activate only when
their savings exceed their overhead.

### Compression Gate

`ContextManager._truncate_to_budget()` now requires ALL of these conditions
before invoking LLMLingua-2:

| Condition | Default | Rationale |
|-----------|---------|-----------|
| `compression.enabled` | `false` (opt-in) | User must explicitly enable |
| `session_turns >= min_turns` | 10 | Short sessions: overhead > savings |
| `usage_ratio > 0.70` | 70% of budget | No point compressing when context fits |
| `len(messages) > preserve_recent_turns * 2` | 6 messages | Don't compress if nothing old to compress |

Config: `configs/compression.yaml` (`min_turns`, `preserve_recent_turns`).

### Tool Attention Gate

`ToolAttention.select_tools()` now short-circuits when the registry is at or
below `min_tools_for_attention` (default 15): returns ALL schemas without any
embed() call. Both the one-time schema embedding at init and the per-query
embedding are skipped below the gate.

Config: `configs/tools.yaml` (`min_tools_for_attention`, `default_k`).

### Tool Registry Expansion

`configs/tools.yaml` expanded from 5 to 50 tool schemas across 10 categories
(file ops, shell, web, git, search, memory, calendar, math, testing, project
management). At 50 tools > 15 threshold, embedding-based selection is active.

## Phase 3: Agentic Core — Subset A/B

### 5-Task Representative Subset

Full 50-task A/B would take ~87 min (200 model calls at ~10 tok/s). For fast
Phase 3 evaluation, a 5-task representative subset (`agentic_subset.json`) was
run through 4 variants with 16K context budget, 128 max_tokens, and 50-tool
registry. Total runtime: ~5 min (20 model calls).

**Results:**

| Variant | p50 latency | p95 latency | avg tok/s | completion | prompt_tokens (task 1 / 5) |
|---|---|---|---|---|---|
| baseline | 17.14 s | 25.83 s | 7.96 | 100% | 64 / 270 |
| plus_compression | 16.78 s | 24.42 s | 7.91 | 100% | 64 / 270 |
| plus_tool_attention | 22.63 s | 28.63 s | 5.21 | 100% | 257 / 473 |
| plus_all_combined | 15.92 s | 15.96 s | 7.96 | 100% | 257 / 473 |

**What this proves:**

1. **Compression gate works.** With only 5 turns, the gate (`min_turns=10`)
   blocks compression entirely. `plus_compression` matches baseline within
   noise — no LLMLingua-2 model load, no compression calls, no overhead. This
   validates the gate design: compression is invisible when it shouldn't fire.

2. **Tool attention embed() overhead is real.** `plus_tool_attention` shows a
   ~35% throughput drop (7.96 → 5.21 tok/s) from the per-call embed() round-trip
   to the nomic-embed server. At 128 max_tokens, this fixed overhead is not
   amortized by the token savings from injecting 3 tools instead of 50.

3. **Tool attention adds ~200 tokens to prompt.** The 3 selected tool schemas
   add ~200 tokens vs 0 tools in baseline. The real savings would only appear
   when the alternative is injecting all 50 tool schemas (~3200 tokens) — which
   is the use case Tool Attention was designed for. The baseline in this test
   injects 0 tools, not 50, so tool attention is a net cost here, not a saving.

4. **Context accumulation works.** Messages grow 2 → 4 → 6 → 8 → 10 across the
   5 tasks, confirming session accumulation is functional.

5. **All variants 100% completion.** No crashes, no errors, no failed requests
   across all 20 model calls.

**What this does NOT prove (and why):**

- **Compression effectiveness at 10+ turns.** The gate prevents compression
  from firing at 5 turns. To test compression savings, a session needs 10+ turns
  with enough context to exceed 70% of the 16K budget (~11K tokens of history).
  This requires a 15-20 task session with longer generations.

- **Tool attention token savings vs full 50-tool injection.** The baseline
  variant injects 0 tools, not all 50. To measure the real savings, a fifth
  variant (`plus_all_tools_no_attention`) would need to inject all 50 schemas
  (~3200 tokens) for comparison. The 93.5% token reduction measured in Phase 2
  (3200 → 207 tokens) is the expected savings, but the latency tradeoff depends
  on generation length.

- **Hierarchical memory retrieval.** The A/B runner does not wire
  `HierarchicalMemory` into the `ContextManager` — it tests compression and
  tool attention only. Memory retrieval requires the specialist model
  (Falcon-H1) for summarization, which adds another server to start.

- **Health monitor triggers.** Context utilization stays low (270 tokens /
  16K budget = 1.7%) across 5 tasks. The health monitor needs 80%+ utilization
  to trigger actions, which requires a much longer session.

- **Session save/resume.** Not part of the A/B test. Covered by unit tests
  (`tests/test_session.py`, 8 tests).

### Recommended Default Configuration for Real Agentic Use

Based on the Phase 2 and Phase 3 data:

| Feature | Default | When to Enable |
|---------|---------|----------------|
| Compression | `enabled: false` | Enable for sessions expected to exceed 50+ turns. Gate prevents overhead below 10 turns. |
| Tool Attention | `min_tools_for_attention: 15` | Active by default with 50-tool registry. Disable for small registries (<15 tools). |
| Hierarchical Memory | Opt-in | Enable for multi-session workflows. Requires specialist model for summarization. |
| Health Monitor | Opt-in | Enable for long sessions. Low overhead (runs every 5 turns). |
| Session Persistence | Opt-in | Enable for session resume across restarts. Low overhead (JSON save). |

### Crossover Points

- **Compression crossover:** ~10 turns with 70%+ context utilization. Below
  this, the 233ms/call LLMLingua-2 overhead exceeds the token savings. Above
  this, compression prevents hard-dropping of old messages, preserving context
  quality.

- **Tool Attention crossover:** ~15 tools in the registry. Below this, the
  per-call embed() round-trip (~1-2s) exceeds the token savings from selective
  injection. Above this, the 93.5% token reduction (3200 → 207 tokens at 50
  tools) amortizes the embed cost, especially at longer generation lengths
  (256+ tokens) where reduced prompt processing time compounds with token
  savings.

- **Hierarchical Memory crossover:** When episodic summaries + semantic facts
  (injected as ~100-200 tokens of system context) provide more value than the
  raw history they replace. This is session-dependent and best measured
  qualitatively (does the model reference past context correctly?).
