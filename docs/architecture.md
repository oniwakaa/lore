# LORE Architecture

System architecture with Phase 3 components: hierarchical memory, context health
monitoring, and session persistence.

## Data Flow

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
