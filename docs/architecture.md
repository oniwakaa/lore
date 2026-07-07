# LORE Architecture

System architecture with Phase 3+4 components: hierarchical memory, context health
monitoring, session persistence, and orchestration engine with parallel execution.

## Orchestration Engine (Phase 4)

The orchestrator sits above routing and `_dispatch()`. For simple tasks, it
delegates to the existing single-model path unchanged. For complex tasks, it
decomposes, schedules, executes (potentially in parallel), and aggregates.

```
USER REQUEST
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR (src/lore/orchestrator.py)                          │
│                                                                   │
│  1. Route query (Router)                                          │
│  2. Estimate complexity (heuristic, <1ms, no LLM call)            │
│     - Simple → _dispatch() (existing path, unchanged)             │
│     - Complex → orchestrate                                       │
│                                                                   │
│  3. Decompose task (one planning call to primary model)           │
│     - 2-5 SubTasks with model, budget, system_prompt, deps        │
│                                                                   │
│  4. Schedule: topological sort → waves                            │
│     - Independent subtasks in same wave                           │
│     - Dependent subtasks in later waves                           │
│                                                                   │
│  5. Execute waves (Worker per subtask)                            │
│     - Same wave + different models → PARALLEL (ThreadPoolExecutor)│
│     - Same wave + same model → sequential (1 slot per server)     │
│     - Each Worker: own ContextManager, scoped budget              │
│                                                                   │
│  6. Aggregate (one call to primary)                               │
│     - Feed all subtask outputs + original task                    │
│     - Return coherent unified response                            │
│                                                                   │
│  7. Store aggregate summary to episodic memory (1 store, not N+1) │
│                                                                   │
│  Dynamic Model Lifecycle:                                         │
│    - If all subtasks are primary-only → offload specialist        │
│      (frees ~1.1 GB for KV cache headroom)                       │
│    - Reload specialist after orchestration completes              │
└──────────────────────────────────────────────────────────────────┘
```

### Parallel Wave Execution

When a wave contains subtasks on different models (e.g., one on primary,
one on specialist), they execute in parallel via `ThreadPoolExecutor`.
Threads share the same process memory — no extra model copies. Each
thread sends HTTP to a different llama-server port (19000 for primary,
19001 for specialist), so there is no slot contention.

Same-model subtasks within a wave run sequentially because each
llama-server has 1 slot (`-np 1`).

## Data Flow (Simple Tasks)

```
USER REQUEST
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  ROUTER (TF-IDF + LogReg, <1ms)                              │
│  Routes: PRIMARY / SPECIALIST / TOOL_ONLY / MULTIMODAL       │
│  Confidence gate: <70% → PRIMARY                             │
└──────────┬──────────────────────────────────────────────────┘
           │
     ┌─────┼─────┬──────────────┐
     │     │     │              │
     ▼     ▼     ▼              ▼
  TOOL   PRIMARY SPECIALIST   MULTIMODAL
  ONLY   (9B)   (1.5B)        (swap-in)
  (regex)       │              │
     │          │              │
     │          ▼              │
     │  ┌──────────────────┐   │
     │  │ CONTEXT MANAGER  │   │
     │  │                  │   │
     │  │ 1. Health check  │   │
     │  │   (every 5 turns)│   │
     │  │   ↓              │   │
     │  │   ok / compress  │   │
     │  │   / summarize    │   │
     │  │   / prune        │   │
     │  │                  │   │
     │  │ 2. Memory        │   │
     │  │   retrieval      │   │
     │  │   (top-3 episodic│   │
     │  │    + top-5 facts)│   │
     │  │                  │   │
     │  │ 3. Tool Attention│   │
     │  │   (if >15 tools: │   │
     │  │    embed+select  │   │
     │  │    top-k)        │   │
     │  │                  │   │
     │  │ 4. Compression   │   │
     │  │   (if >10 turns  │   │
     │  │    AND >70% full)│   │
     │  │                  │   │
     │  │ 5. Truncate to   │   │
     │  │   budget         │   │
     │  └────────┬─────────┘   │
     │           │              │
     │           ▼              │
     │     MODEL CALL           │
     │           │              │
     │           ▼              │
     │     RESPONSE             │
     │           │              │
     │           ▼              │
     │  ┌──────────────────┐   │
     │  │ POST-PROCESSING  │   │
     │  │                  │   │
     │  │ • Verify output  │   │
     │  │   (Verifier)     │   │
     │  │ • Add to history │   │
     │  │ • Store to       │   │
     │  │   memory         │   │
     │  │ • Maybe summarize│   │
     │  │   (every 10 turns)│  │
     │  │ • Log request    │   │
     │  │ • Maybe save     │   │
     │  │   session        │   │
     │  └──────────────────┘   │
     │                         │
     ▼                         ▼
  OUTPUT                    SWAP-OUT
```

## Hierarchical Memory Flow

```
WORKING MEMORY (ContextManager._history)
  Last 5-10 turns, raw messages
        │
        │ When context pressure OR every 10 turns:
        │ HierarchicalMemory.maybe_summarize()
        ▼
EPISODIC MEMORY (EpisodicMemory)
  Compressed summaries of old conversations
  50-200 entries, each 2-3 sentences
  Summarization via specialist model (Falcon-H1)
  Fallback: extractive (first 300 chars)
        │
        │ Every 5 episodes:
        │ SemanticMemory.extract_facts()
        ▼
SEMANTIC MEMORY (SemanticMemory)
  Durable facts: preferences, project state, conventions
  20-100 entries, flat key-value with source references
  Extraction via specialist model (Falcon-H1)
  Fallback: sentence-split heuristic
        │
        │ On query:
        │ retrieve top-5 by embedding similarity
        ▼
INJECTED INTO SYSTEM MESSAGE
  "Relevant context:
   - [episodic summary 1]
   - [episodic summary 2]
   - [semantic fact 1]
   - [semantic fact 2]
   ..."
```

## Optimization Decision Tree

```
When building a prompt:

1. HEALTH CHECK (every 5 turns)
   │
   ├─ utilization < 80% → ok
   ├─ utilization 80-90% + stale > 30% → compress
   ├─ utilization 80-90% + stale <= 30% → warn_degradation
   ├─ utilization > 90% + stale > 50% → summarize (episodic)
   ├─ utilization > 90% + stale <= 50% → prune (hard drop)
   └─ repetition > 50% → warn_degradation

2. MEMORY RETRIEVAL (if HierarchicalMemory enabled)
   │
   ├─ Retrieve top-3 episodic summaries by query similarity
   └─ Retrieve top-5 semantic facts by query similarity
   → Inject as "Relevant context:" in system message

3. TOOL ATTENTION (if ToolAttention enabled)
   │
   ├─ Registry <= 15 tools → inject ALL (no embed cost)
   └─ Registry > 15 tools → embed query, select top-k by similarity
   → Inject as "Available tools:" in system message

4. COMPRESSION (in _truncate_to_budget)
   │
   ├─ NOT enabled → skip
   ├─ session < 10 turns → skip (overhead > savings)
   ├─ usage < 70% of budget → skip (nothing to compress)
   ├─ no old messages (all within preserve_recent) → skip
   └─ ALL conditions met → compress old messages via LLMLingua-2
   → Keep last 3 turns (6 messages) uncompressed

5. TRUNCATION (fallback)
   │
   └─ If still over budget after compression → drop oldest
   → Always keep last 3 turns (6 messages)

After model response:

6. OUTPUT VERIFICATION (Verifier, src/lore/verifier.py)
   │
   ├─ task_type == "free_form" → skip (always valid)
   ├─ task_type == "json" → validate JSON syntax
   │   ├─ valid → pass through
   │   └─ invalid → attempt repair (trailing comma fix, missing brace close)
   └─ task_type == "code_python" → validate with ast.parse()
   → Log repair attempts; max_repair_attempts=2

Per-request context budget (Dynamic Sizing, src/lore/sizing.py):

7. DYNAMIC CONTEXT SIZING (called per-request in _dispatch)
   │
   ├─ TOOL_ONLY → 2048 tokens (min_budget)
   ├─ SPECIALIST → 4096 tokens
   ├─ PRIMARY + code block or file path → 8192+
   ├─ PRIMARY + complex keyword (refactor/debug/review/plan) → 8192+
   ├─ PRIMARY + simple keyword (explain/summarize/what is) → 4096
   ├─ PRIMARY + query > 500 tokens → 32768 (max_budget)
   └─ PRIMARY default → 16384
   → Calls ctx.set_budget() to override working_context for this request only
```

## Memory Budget (All Components)

```
Ornith-1.0-9B Q4_K_M (primary)              5.50 GB  (measured RSS, turbo4 16K)
Falcon-H1-1.5B Q4_K_M (specialist)          1.09 GB  (measured RSS, turbo4 16K)
nomic-embed-text-v1.5 (embeddings)          0.30 GB  (estimated, loaded on demand)
LLMLingua-2 (compression, opt-in, CPU)     +0.34 GB  (lazy singleton, 758 MB process)
Session state (JSON, in-memory)             ~0.01 GB
Hierarchical memory (200 ep + 100 facts)    ~0.05 GB  (embeddings in memory)
Health monitor (stateless, logs to disk)    ~0.00 GB
OS + llama.cpp + buffers                    1.50 GB
────────────────────────────────────────────────────
MAX (all features enabled)                  8.79 GB  (5.21 GB headroom)
TYPICAL (dual model, no opt-ins)            6.59 GB  (7.41 GB headroom)
```

Well under the 14 GB hard cap. Even with all Phase 3 features enabled, the system
has 5+ GB of headroom for working memory and OS spikes.

## Component Toggles

All Phase 3 features are config-driven and opt-in:

| Component | Config File | Default | Key |
|-----------|------------|---------|-----|
| Compression | `configs/compression.yaml` | `enabled: false` | `enabled`, `min_turns` |
| Tool Attention | `configs/tools.yaml` | `min_tools_for_attention: 15` | `min_tools_for_attention` |
| Hierarchical Memory | `configs/memory.yaml` | opt-in | `summarize_after_turns`, `max_facts` |
| Health Monitor | `configs/memory.yaml` | opt-in | `health.warn_threshold` |
| Session Persistence | `configs/sessions.yaml` | opt-in | `auto_save_every_n_turns` |

No Phase 3 feature is hard-depended upon. If any component fails, the system
falls back to raw primary model inference (per the "fail gracefully" constraint).
