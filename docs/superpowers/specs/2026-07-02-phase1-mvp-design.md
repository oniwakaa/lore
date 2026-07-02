# Phase 1 MVP Orchestration Design

**Date:** 2026-07-02
**Status:** Approved
**Phase:** 1 — MVP Orchestration
**Prerequisites:** Phase 0 complete (build, models, memory validated)

## Goal

Working two-model orchestration system with routing, context management, episodic memory, constrained output, prefix caching, llama-swap for multimodal, and request logging.

## Architecture

```
User Input
    │
    ▼
┌──────────────────┐
│   CLI (cli.py)    │  single-shot or interactive REPL
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Router (router)  │  TF-IDF + LogReg, 4-way: PRIMARY/SPECIALIST/MULTIMODAL/TOOL_ONLY
│  confidence gate  │  < threshold → PRIMARY
└────────┬─────────┘
         │ route decision
         ▼
┌──────────────────┐
│ Context Manager   │  32K ctx, parameterized budget, token counting via /tokenize
│ + Memory Retrieve │  episodic: embed last 5 turns, cosine similarity, inject top-3
└────────┬─────────┘
         │ assembled prompt
         ▼
┌──────────────────┐
│  Model Server     │  HTTP client → llama-server instances
│  (models.py)      │  GBNF via response_format, prefix cache, llama-swap for Gemma
└────────┬─────────┘
         │ response
         ▼
┌──────────────────┐
│  Logger           │  logs/requests.jsonl: route, confidence, latency, tokens
└────────┬─────────┘
         │
         ▼
    User Output
```

## Components

### 1. config.py — Central Config Loader

**Purpose:** Load all YAML configs, apply env var overrides, provide typed access.

**Config files:**
- `configs/models.yaml` — model paths, ports, context size, KV cache type, ports
- `configs/router.yaml` — confidence_threshold, training_data_path, model_path, class labels
- `configs/llama-swap.yaml` — swap config for Gemma 4 E4B
- `configs/memory.yaml` — embed model, top_k, max_turns, similarity threshold

**Env var overrides:** `LORE_CTX_SIZE`, `LORE_PRIMARY_PORT`, `LORE_SPECIALIST_PORT`, `LORE_EMBED_PORT`, `LORE_LOG_LEVEL`, `LORE_CONFIDENCE_THRESHOLD`

**Interface:**
```python
class LoreConfig:
    def load() -> LoreConfig  # load all YAML + env overrides
    @property
    def models: dict          # model configs
    @property
    def router: dict          # router config
    @property
    def memory: dict          # memory config
    @property
    def context: dict         # context budget config
```

### 2. models.py — Model Lifecycle + HTTP Client + llama-swap

**Purpose:** Manage llama-server instances, dispatch requests, handle GBNF, prefix caching, swapping.

**Persistent servers (always running):**
- Ornith-9B turbo4 32K — port 19000 (primary)
- Falcon-H1-1.5B turbo4 32K — port 19001 (specialist)
- nomic-embed-text-v1.5 — port 19002 (embeddings)

**Hot-swap (via llama-swap):**
- Gemma 4 E4B — swapped in on MULTIMODAL route, swapped out after TTL (default 120s idle)

**HTTP client methods:**
```python
class ModelServer:
    def start_all() -> None           # start all persistent servers, health check
    def stop_all() -> None            # graceful shutdown
    def health_check(port) -> bool    # GET /health, retry 3x with 1s backoff
    
    def chat(model: str, messages: list, **opts) -> dict
        # POST /v1/chat/completions
        # opts: max_tokens, temperature, response_format (for GBNF/JSON)
        # cache_prompt=True in request body for prefix cache
        # static system prompt ensures prefix cache hit
    
    def tokenize(model: str, text: str) -> int
        # POST /tokenize, return token count
        # ~5-20ms per call, fine for Phase 1
    
    def embed(text: str) -> list[float]
        # POST /embeddings on port 19002 (nomic-embed)
    
    def swap_in(model_name: str) -> None
        # Trigger llama-swap to load Gemma 4 E4B
        # Health check after swap completes
    
    def swap_out(model_name: str) -> None
        # Trigger llama-swap to unload Gemma 4 E4B
```

**Error handling:**
- Server fails to start: retry once with halved context size, then fall back to primary-only mode
- Specialist request fails: catch exception, retry on primary, log fallback
- Swap fails: log error, return error to user (multimodal unavailable)
- Port conflict: increment port by 1, retry
- Health check timeout: 30s per server, 3 retries with 2s backoff

**GBNF constrained decoding:**
- `chat()` accepts `response_format={"type": "json_object"}` for JSON mode
- `chat()` accepts `response_format={"type": "json_schema", "json_schema": {...}}` for structured output
- llama-server translates this to GBNF grammar internally
- Always use for tool calls and structured extraction

**Prefix KV cache:**
- System prompt: static string from config, never modified between turns
- All requests to server endpoint include `cache_prompt: true` in body (llama-server specific, not OpenAI standard)
- For `/v1/chat/completions`: prefix caching is automatic when system prompt is static, no explicit flag needed
- Verification: check server logs for cache hit ratio (logged to requests.jsonl)

### 3. router.py — TF-IDF + LogReg 4-Way Classifier

**Purpose:** Classify user input into one of 4 routes in <1ms.

**Routes:**
- `PRIMARY` — coding, multi-step reasoning, complex Q&A, planning
- `SPECIALIST` — classification, extraction, formatting, simple yes/no, summarization
- `MULTIMODAL` — image understanding, audio transcription, vision tasks
- `TOOL_ONLY` — regex matching, simple parsing, no LLM needed

**Pipeline:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
])
```

**Confidence gate:**
- `confidence_threshold` from `configs/router.yaml` (default 0.70)
- If `max(predicted_probabilities) < threshold` → route to PRIMARY
- Threshold configurable because optimal value depends on actual accuracy distribution

**Training:**
- Data: `configs/router_training_data.jsonl` (one JSON per line: `{"text": "...", "label": "PRIMARY"}`)
- 200+ examples covering all 4 classes
- Script: `scripts/train_router.py` — trains, evaluates on 20% held-out, saves joblib
- Model: `configs/router_model.joblib` (~50 KB)

**Interface:**
```python
class Router:
    def load(path: str) -> Router     # load trained model
    def classify(text: str) -> tuple[str, float]  # (route, confidence)
    def train(data_path: str, model_path: str) -> dict  # train + eval, return metrics
```

### 4. context.py — Token-Aware Context Manager

**Purpose:** Manage conversation context within configurable token budget.

**Default budget (32K context, all values in configs/models.yaml):**
```
system_prompt:       512 tokens (static, prefix-cached)
retrieved_memories: 1024 tokens (top-3 from episodic memory)
working_context:    4096 tokens (conversation history)
user_input:         2048 tokens (current message + attachments)
generation_headroom:4096 tokens (reserved for output)
────────────────────────────────────
allocated:         11744 tokens
remaining:         20256 tokens (spare for long sessions)
```

**Token counting:**
- Uses `ModelServer.tokenize()` (llama-server `/tokenize` endpoint)
- ~5-20ms per call, acceptable for Phase 1
- Phase 1.5 optimization: local tokenizer cache

**Truncation strategy:**
- When working_context exceeds budget: drop oldest messages first
- Keep system prompt + last 3 turns always
- Log truncation events to requests.jsonl

**Interface:**
```python
class ContextManager:
    def __init__(self, config: dict, model_server: ModelServer)
    def add_message(role: str, content: str) -> None
    def build_prompt(memories: list[str] = None) -> list[dict]  # returns messages list
    def token_count(text: str) -> int
    def truncate_to_budget(messages: list[dict]) -> list[dict]
```

### 5. memory.py — Episodic Memory with Embeddings

**Purpose:** Embed conversation turns, retrieve relevant context by similarity.

**Model:** nomic-embed-text-v1.5 on port 19002 (0.30 GB, 768-dim embeddings)

**Behavior:**
- Embed each assistant response + user turn pair (summarized to ~100 tokens by Falcon-H1 specialist before embedding)
- Store in-memory: `list[(text, embedding, timestamp)]`
- Max 200 entries (circular buffer)
- On new user input: embed input, compute cosine similarity against all entries, retrieve top-3
- Inject retrieved memories into context budget (1024 tokens max)

**Config (`configs/memory.yaml`):**
```yaml
embed_model: nomic-embed-text-v1.5
embed_port: 19002
top_k: 3
max_entries: 200
max_turns_to_embed: 5
similarity_threshold: 0.5
memory_token_budget: 1024
```

**Interface:**
```python
class EpisodicMemory:
    def __init__(self, config: dict, model_server: ModelServer)
    def store(text: str, role: str) -> None        # embed + store
    def retrieve(query: str, top_k: int = 3) -> list[str]  # cosine similarity search
    def clear() -> None
```

### 6. logging.py — Request Logger

**Purpose:** Log every routing decision, model call, latency for Phase 2 A/B testing.

**Output:** `logs/requests.jsonl` (one JSON object per line)

**Log entry schema:**
```json
{
  "timestamp": "2026-07-02T18:30:00Z",
  "input_hash": "sha256:abc123...",
  "route": "SPECIALIST",
  "confidence": 0.92,
  "model": "falcon-h1-1.5b",
  "tokens_in": 145,
  "tokens_out": 32,
  "latency_ms": 1200,
  "success": true,
  "fallback": false,
  "context_truncated": false,
  "cache_hit": null,
  "error": null
}
```

**No PII:** Input is hashed (SHA-256), never stored raw.

**Interface:**
```python
class RequestLogger:
    def log_request(entry: dict) -> None  # append to jsonl
    def log_route(text_hash: str, route: str, confidence: float) -> None
```

### 7. cli.py — CLI Entry Point

**Purpose:** User-facing interface for the orchestration system.

**Modes:**
- Single-shot: `python -m lore.cli "write a function to reverse a list"`
- Interactive: `python -m lore.cli -i` (REPL)
- JSON mode: `python -m lore.cli --json "extract names from: John, Mary, Bob"` (triggers GBNF)

**Interactive commands:**
- `/exit` or Ctrl+C — stop
- `/clear` — clear conversation history + memory
- `/route` — show last routing decision
- `/stats` — show session stats (tokens used, routes, latency)

**Output format:**
```
[route: SPECIALIST (0.92) | 145 tok in | 1.2s]
The names are: John, Mary, Bob.
```

**Startup sequence:**
1. Load config
2. Start all persistent llama-server instances
3. Health check all servers (retry 3x, 2s backoff)
4. Load trained router model
5. Initialize context manager + episodic memory
6. Enter REPL or process single-shot input

## Config Files

### configs/models.yaml (updates from Phase 0)

```yaml
defaults:
  context_size: 32768        # 32K default (was 16K)
  kv_cache_type: turbo4
  flash_attention: true
  gpu_layers: 999
  threads: 4

primary:
  port: 19000
  # ... existing fields, context updated to 32768

specialist:
  port: 19001
  # ... existing fields

embeddings:
  name: nomic-embed-text-v1.5
  port: 19002
  path: models/nomic-embed-text-v1.5.f16.gguf  # download from HuggingFace, verify exact filename
  context: 8192
  embedding_dim: 768

multimodal:
  name: gemma-4-e4b
  port: 19003
  path: models/gemma-4-e4b-Q4_K_M.gguf  # needs download
  quant: Q4_K_M
  expected_size_gb: 5.0
  context: 262144
  swap_ttl_seconds: 120
```

### configs/router.yaml

```yaml
confidence_threshold: 0.70
training_data_path: configs/router_training_data.jsonl
model_path: configs/router_model.joblib
classes:
  - PRIMARY
  - SPECIALIST
  - MULTIMODAL
  - TOOL_ONLY
tfidf:
  ngram_range: [1, 2]
  max_features: 5000
classifier:
  max_iter: 1000
  class_weight: balanced
```

### configs/llama-swap.yaml

```yaml
# llama-swap config for Gemma 4 E4B hot-swap
# When MULTIMODAL route triggers, swap in Gemma 4 E4B
# Swap out after TTL idle
swap_models:
  - name: gemma-4-e4b
    model_path: models/gemma-4-e4b-Q4_K_M.gguf
    port: 19003
    ctx_size: 16384
    kv_cache_type: turbo4
    flash_attention: true
    gpu_layers: 999
    ttl_seconds: 120       # unload after 120s idle
    swap_in_on_route: MULTIMODAL
```

### configs/memory.yaml

```yaml
embed_model: nomic-embed-text-v1.5
embed_port: 19002
top_k: 3
max_entries: 200
max_turns_to_embed: 5
similarity_threshold: 0.5
memory_token_budget: 1024
```

## Memory Layout

```
PERSISTENT (always loaded):
  Ornith-9B turbo4 32K       ~5.7 GB   (port 19000)
  Falcon-H1 turbo4 32K       ~1.2 GB   (port 19001)
  nomic-embed-text-v1.5       0.3 GB   (port 19002)
  ─────────────────────────────────────
  Total persistent            ~7.2 GB   (6.8 GB headroom to 14 GB)

HOT-SWAP (loaded on demand):
  Gemma 4 E4B Q4_K_M          ~5.0 GB   (port 19003, swapped via llama-swap)
  When loaded: total ~12.2 GB (1.8 GB headroom — tight but fits)
```

## Dependencies

```
scikit-learn>=1.4
pyyaml>=6.0
requests>=2.31
numpy>=1.26
joblib>=1.3
```

All lightweight, no GPU/cuda deps. Python 3.11+ (3.14 on this machine).

## Error Handling Summary

| Failure | Action | Log |
|---------|--------|-----|
| Server fails to start | Retry with halved context, then primary-only mode | error + fallback |
| Specialist request fails | Retry on primary | fallback: true |
| Swap fails | Return error to user | error: "multimodal unavailable" |
| Port conflict | Increment port, retry | warning |
| Health check timeout | 3 retries, 2s backoff, then degrade | error |
| Router model missing | Default all to PRIMARY | warning |
| Embeddings server down | Skip memory retrieval, continue | warning |
| Tokenize endpoint fails | Estimate tokens as len(text)/4 | warning |

## Testing

- `tests/test_router.py` — unit test classification accuracy on held-out data
- `tests/test_context.py` — unit test token budget truncation
- `tests/test_models.py` — integration test with mock HTTP server
- `tests/test_memory.py` — unit test embedding storage + retrieval
- `assert`-based `__main__` self-check in each module (ponytail pattern)

## Phase 1 Exit Criteria

- [ ] Router >85% accuracy on held-out test set
- [ ] 100% valid JSON output in JSON mode (GBNF)
- [ ] Prefix cache hit >70% on turn 2+ (verify in server logs)
- [ ] Episodic memory retrieves relevant context (cosine sim >0.5)
- [ ] llama-swap successfully loads/unloads Gemma 4 E4B
- [ ] All routing decisions logged to requests.jsonl
- [ ] Error handling: no crash on server failure, graceful fallback
- [ ] Memory <14 GB at all times (including during swap)
