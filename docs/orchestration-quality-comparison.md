# Orchestration Quality Comparison: Direct vs Orchestrated

**Model:** Qwythos-9B-Claude-Mythos-5-1M Q4_K_M
**Date:** 2026-07-13
**Task:** Refactor the authentication module to use JWT tokens, update tests, write migration docs

## Results

| Metric | Direct | Orchestrated |
|--------|--------|-------------|
| Latency | 172.4s | 540s+ (partial, tests 3-5 timed out) |
| Tokens generated | 1049 | ~180 (s1) + ~16 (s2) + ~69 (s3) |
| Useful content | 3807 chars | ~300 chars (s1) + ~50 chars (s2) + ~200 chars (s3) |
| Orchestrated | No | Yes (3 subtasks) |
| Quality | Complete, production-ready | Fragmented, incomplete |

## Direct Output

The direct call (with `/no_think`, max_tokens=8192, temperature=0.6) produced a complete JWT authentication implementation:

- `TokenManager` class with `generate_token()` and `verify_token()` methods
- Flask middleware for JWT verification
- Migration notes
- Security reminders
- Type hints, docstrings, proper error handling
- Well-structured, production-ready code (3807 chars)

**Quality assessment:** Excellent. The code is well-organized, properly documented, and covers all three requested components (token generation, verification middleware, migration notes).

## Orchestrated Output

The orchestrated pipeline decomposed the task into 3 subtasks:
1. s1 (primary): Write JWT auth implementation — 186s, 162 tokens
2. s2 (primary): Write tests for JWT auth — 184s, 16 tokens
3. s3 (specialist): Write migration docs — 15s, 69 tokens

**Quality assessment:** Poor. The orchestrated output was fragmented and incomplete:
- s1 produced only 162 tokens (vs 1049 direct) — the model spent most tokens on thinking
- s2 produced only 16 useful tokens — nearly all tokens went to thinking
- s3 (specialist/Falcon-H1) was fast but produced minimal content
- The aggregated output was a summary of fragments, not a coherent implementation

## Analysis

### Why Direct Outperformed Orchestrated

1. **CoT overhead multiplies:** Each orchestration call pays the CoT tax. 3 subtask calls = 3x thinking overhead. The direct call pays it once.

2. **Token budget fragmentation:** The orchestrator allocates token budgets per subtask (2048-4096). Qwythos needs 500-1000 tokens for thinking before producing content. With 2048 max_tokens, only 1000-1500 tokens are available for actual content. With 1024 max_tokens (short code tasks), the model may produce ZERO content.

3. **Context loss between subtasks:** The direct call has the full context in one prompt. The orchestrated pipeline splits context across subtasks, losing coherence. Each subtask only sees its own description + previous outputs (truncated to 2000 chars).

4. **Aggregation truncation:** The orchestrator truncates subtask outputs during aggregation, further reducing quality.

### When Orchestration Helps vs Hurts

| Scenario | Direct Better | Orchestrated Better |
|----------|--------------|-------------------|
| Single coherent task (code gen) | Yes — full context, one CoT pass | No — fragmentation hurts |
| Multi-domain task (code + docs + tests) | Maybe — if model handles all | Maybe — if specialist is faster for docs |
| Task requiring exploration (SWE-bench) | No — needs tool-use loop | Yes — structured exploration + patch |
| Speed-critical tasks | Yes — 3-10x faster | No — orchestration overhead |

### Verdict

For Qwythos, **direct calls produce better quality than orchestration** because:
1. The CoT reasoning is most effective with full context in a single pass
2. Token budget fragmentation causes the model to run out of tokens during thinking
3. The 4-10x latency overhead of orchestration is not justified by quality

For Ornith (non-reasoning model), orchestration still makes sense because:
1. No CoT overhead — token budget goes entirely to content
2. Faster generation (6.3 t/s) makes multiple calls affordable
3. Decomposition helps focus the model on specific subtasks

## Recommendation

- **Qwythos:** Use direct calls, not orchestration. The model's CoT reasoning is most effective with full context and generous token budget (8192+).
- **Ornith:** Keep orchestration. The non-reasoning model benefits from task decomposition and focused context budgets.
- **LORE:** Consider a "reasoning model mode" that bypasses orchestration for reasoning models and sends tasks directly with high max_tokens.
