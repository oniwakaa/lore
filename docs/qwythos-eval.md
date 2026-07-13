# Qwythos-9B-Claude-Mythos-5-1M Evaluation

**Date:** 2026-07-13
**Model:** Qwythos-9B-Claude-Mythos-5-1M Q4_K_M (5.63 GB) + MTP variant (5.89 GB)
**Hardware:** Apple Silicon M4, 16 GB unified memory
**Comparison baseline:** Ornith-1.0-9B Q4_K_M (5.63 GB)

## Summary

**Decision: Qwythos should NOT replace Ornith as primary model for LORE orchestration.**

Qwythos generates higher-quality output per response, but its chain-of-thought (CoT) reasoning overhead makes it 4-10x slower than Ornith for orchestration workloads. The CoT overhead multiplies across orchestration calls (decompose + subtasks + aggregate), making the full pipeline unacceptably slow.

## Step 1: Download

Both GGUF variants downloaded successfully:
- `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf` — 5.63 GB (standard)
- `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf` — 5.89 GB (MTP-enabled)

Files stored in `models/` (gitignored). Not committed.

## Step 2: Latency and Speed Comparison

### Raw Generation Speed

| Model | Generation Speed | Notes |
|-------|-----------------|-------|
| Ornith-1.0-9B | 6.3 t/s | No CoT overhead, direct output |
| Qwythos (standard, no MTP) | 5.3-5.8 t/s | 15% slower than Ornith |
| Qwythos (MTP, draft-mtp) | 5.35 t/s | 41% draft acceptance, 2.4x token overhead = net SLOWER |

### MTP Speculative Decoding

MTP was tested first with `--spec-type draft-mtp --spec-draft-n-max 6`. Results from server logs:

```
draft-mtp: #calls(b,g,a) = 3  413  413, #gen drafts = 413, #acc drafts = 331,
#gen tokens = 2478, #acc tokens = 1029
```

- 2478 tokens generated, only 1029 accepted (41.5% acceptance rate)
- Draft overhead: 2.4x more tokens generated than accepted
- Net effect: SLOWER than no speculation (draft generation dominates)
- Also caused cache issues: "forcing full prompt re-processing due to lack of cache data"
- **Conclusion: MTP does not help with the TurboQuant fork. Use standard version.**

### CoT Reasoning Overhead

Qwythos is a reasoning model — every response opens with a `<think>...</think>` block. Key findings:

| Scenario | Total Tokens | Useful Content Tokens | Latency | Effective Useful t/s |
|----------|-------------|----------------------|---------|---------------------|
| Simple query (no /no_think) | ~120 | ~5 | 17s | 0.3 |
| Simple query (/no_think) | ~80 | ~5 | 14s | 0.4 |
| Code task (no /no_think, 512 max) | 512 | ~50 | 71s | 0.7 |
| Code task (/no_think, 2048 max) | 225 | ~55 | 31s | 1.8 |
| Auth refactor (/no_think, 8192 max) | 1049 | ~950 | 172s | 5.5 |
| Auth refactor (no /no_think, 4096 max) | 4096 | 0 (all thinking) | 711s | 0.0 |

**Critical finding:** Without `/no_think`, the model can use ALL max_tokens for thinking and produce ZERO content. With `/no_think`, thinking is reduced but not eliminated.

### Orchestration Test Results (5-test suite)

| Test | Ornith Latency | Qwythos Latency | Ratio |
|------|---------------|-----------------|-------|
| Test 1 (simple query) | ~3s | 14s | 4.7x |
| Test 2 (complex, 3 subtasks) | ~60s | 540s+ (s1=186s, s2=184s, s3=15s) | 9x |
| Tests 3-5 | ~120s each | TIMEOUT (>900s each) | N/A |

- Test 1 output: "calculator(expression=\"2+2\")" — odd format, not "4"
- Test 2 s2: only 16 useful tokens in 184s (model spent most time thinking)
- Tests 3, 4, 5 could not complete within 900s timeout
- Full 5-test suite estimated: 30+ minutes (vs ~10 min with Ornith)

### Sampling Configuration

Qwythos requires specific sampling params per model card:
- `temperature: 0.6` (lower causes repetition loops)
- `top_p: 0.95, top_k: 20`
- `repetition_penalty: 1.05`
- `max_new_tokens: 16384` recommended

LORE's worker defaults to `temperature=0.1` for code tasks, which causes repetition loops in Qwythos. A `sampling` override was added to `ModelServer.chat()` to apply model-specific sampling from config.

## Step 3: Quality Comparison (Direct vs Orchestrated)

### Direct Call (Qwythos, /no_think, 8192 max_tokens)

Task: "Refactor the authentication module to use JWT tokens..."

- **Latency:** 172.4s
- **Tokens:** 1049
- **Content:** 3807 chars
- **Quality:** Excellent — complete implementation with:
  - `TokenManager` class with `generate_token()`, `verify_token()`
  - Flask middleware for JWT verification
  - Migration notes
  - Security reminders
  - Type hints, docstrings, proper error handling
  - Well-structured, production-ready code

### Orchestrated Call (Qwythos)

The orchestrated pipeline was significantly slower:
- Decomposition alone took 3.5 minutes (vs ~30s with Ornith)
- Subtask s1: 186s for 162 tokens (0.87 useful tok/s)
- Subtask s2: 184s for 16 tokens (0.087 useful tok/s — nearly all thinking)
- Total for test 2: 540s+ (vs ~60s with Ornith)

### Verdict

- **Quality:** Qwythos output is more structured and complete than Ornith's typical output
- **Latency:** 4-10x slower, making orchestration impractical
- **The quality advantage does not justify the latency cost** for LORE's multi-call orchestration pattern

## Step 4: SWE-bench SEARCH/REPLACE Test

Task: `django__django-16082` (add `Combinable.MOD` to a list — 1-line change)

### Results

| Metric | Qwythos | Ornith (previous run) |
|--------|---------|----------------------|
| Resolved | No (0/1) | No (0/1) |
| Patch extracted | No | No |
| Patch applies | No | No |
| Tests passed | 0/0 | 0/0 |
| Latency | 397.3s | 583.0s |
| Orchestrated | Yes (2 subtasks) | Yes (2 subtasks) |
| s1 content tokens | 62 | N/A |
| s2 content tokens | 49 | N/A |
| LLM calls | 4 | N/A |

### What Happened

1. **s1 (explore):** 200s, 5 tool-use rounds, 62 content tokens. The model used READ_FILE and SEARCH tools to explore the Django repo, but produced only 62 tokens of content (the rest was thinking tokens).

2. **s2 (patch):** 194s, 4 tool-use rounds, 49 content tokens. The model was supposed to produce a SEARCH/REPLACE block but generated only 49 tokens of content — not enough for a meaningful patch.

3. **Patch extraction:** No SEARCH/REPLACE blocks found in the output. The `extract_patch()` function tried SEARCH/REPLACE first, then unified diff, then code blocks — all failed.

4. **No patch to apply:** Empty patch string, no evaluation possible.

### Did SEARCH/REPLACE Format Help?

**No.** The model did not output SEARCH/REPLACE blocks. The 49 content tokens from s2 were insufficient for any code patch format. The CoT overhead consumed the token budget before the model could produce structured output.

### Comparison with Ornith

Qwythos was actually **faster** than Ornith for this task (397s vs 583s), possibly because Qwythos's tool-use loop terminated earlier (4-5 rounds) vs Ornith's longer exploration. But both models failed to produce a patch — the fundamental problem is that 9B Q4 models cannot produce precise code patches regardless of format.

## Step 5: Decision

### Should Qwythos replace Ornith as primary?

**No.** Reasons:

1. **CoT overhead is fundamental:** Qwythos generates 200-2000 thinking tokens per response, even with `/no_think`. This is baked into the model training and cannot be fully disabled.

2. **Orchestration multiplies the overhead:** LORE's pipeline requires 3-5 model calls per task (decompose + 2-3 subtasks + aggregate). Each call pays the CoT tax.

3. **MTP does not help:** The MTP speculative decoding has 41% acceptance rate with 2.4x token overhead, making it net slower.

4. **`/no_think` is insufficient:** It reduces but does not eliminate thinking. The model still generates 200-1000 tokens of thinking per response.

5. **Quality advantage doesn't justify latency:** While Qwythos produces better-structured output, the 4-10x latency increase makes the orchestration pipeline impractical for interactive use.

### When Qwythos Would Be Better

- **Direct (non-orchestrated) use:** For single-call tasks where quality matters more than speed (e.g., complex code generation, security analysis)
- **With more compute:** If running on hardware with more memory (32+ GB), higher max_tokens (16384) could allow the model to complete thinking + output without token limits
- **Qwythos v2:** The model card mentions v2 with fixed looping behavior and restored MTP head. Worth evaluating separately.

### Recommendation

Keep Ornith-1.0-9B as primary. Consider Qwythos as an optional "deep reasoning" model for specific high-quality single-call tasks, but not for the orchestration pipeline.
