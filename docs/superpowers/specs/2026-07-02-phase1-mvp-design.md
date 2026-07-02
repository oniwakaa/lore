# Phase 1 MVP Orchestration Design

**Date:** 2026-07-02
**Status:** Approved (revised after review)
**Phase:** 1 — MVP Orchestration
**Prerequisites:** Phase 0 complete (build, models, memory validated)

## Goal

Working two-model orchestration system with routing, context management, episodic memory, constrained output, prefix caching, direct process management for multimodal, and request logging.

## Architecture

```
User Input
    │
    ▼
┌──────────────────┐
│   CLI (cli.py)    │  single-shot or interactive REPL
│  multimodal check │  structural: image/audio refs → MULTIMODAL (pre-router)
└────────┬─────────┘
         │
    ┌────┴────┐
    │ MULTIMODAL? │
    └────┬────┘
     yes │     no
         │      │
         ▼      ▼
┌──────────┐  ┌──────────────────┐
│ swap_in  │  │  Router (router)  │  TF-IDF + LogReg, 3-way
│ Gemma 4  │  │  PRIMARY/SPECIALIST/TOOL_ONLY
│ dispatch │  │  confidence gate  │  < threshold → PRIMARY
└──────────┘  └────────┬─────────┘
                       │ route decision
                       ▼
              ┌──────────────────┐
              │ Context Manager   │  32K ctx, parameterized budget
              │ + Memory Retrieve │  episodic: embed raw text, cosine sim, inject top-3
              └────────┬─────────┘
                       │ assembled prompt
                       ▼
              ┌──────────────────┐
              │  Model Server     │  HTTP client → llama-server instances
              │  (models.py)      │  GBNF via response_format, prefix cache
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
- `configs/models.yaml` — model paths, ports, context size, KV cache type
- `configs/router.yaml` — confidence_threshold, training_data_path, model_path, class labels
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

### 2. models.py — Model Lifecycle + HTTP Client + Direct Process Swap

**Purpose:** Manage llama-server instances, dispatch requests, handle GBNF, prefix caching, swap Gemma 4 E4B via direct process management.

**Persistent servers (always running):**
- Ornith-9B turbo4 32K — port 19000 (primary)
- Falcon-H1-1.5B turbo4 32K — port 19001 (specialist)
- nomic-embed-text-v1.5 — port 19002 (embeddings)

**Hot-swap (direct process management, no llama-swap dependency):**
- Gemma 4 E4B — started as a new llama-server process on port 19003 when multimodal input detected
- Killed after TTL idle (default 120s) via a background timer thread
- Simpler than llama-swap for a single swap model: start process, health check, use, kill when idle

**HTTP client methods:**
```python
class ModelServer:
    def start_all() -> None           # start all persistent servers, health check
    def stop_all() -> None            # graceful shutdown
    def health_check(port) -> bool    # GET /health, retry 3x with 1s backoff

    def chat(model: str, messages: list, **opts) -> dict
        # POST /v1/chat/completions
        # opts: max_tokens, temperature, response_format (for GBNF/JSON)

    def tokenize(model: str, text: str) -> int
        # POST /tokenize, return token count
        # ~5-20ms per call, fine for Phase 1
        # ponytail: first thing to optimize in Phase 1.5 — local tokenizer cache
        #           4-6 calls per request = 40-120ms overhead before model starts

    def embed(text: str) -> list[float]
        # POST /v1/embeddings on port 19002 (nomic-embed)
        # OpenAI-compatible endpoint

    def swap_in(model_name: str) -> None
        # Start llama-server process for Gemma 4 E4B on port 19003
        # Health check after process starts (retry 3x, 2s backoff)
        # Start idle timer thread

    def swap_out(model_name: str) -> None
        # Kill Gemma 4 E4B process, free memory
        # Cancel idle timer

    def verify_prefix_cache() -> bool
        # Startup check: send identical prompt twice, measure TTFT delta
        # Log result to requests.jsonl
        # Verifies prefix caching is active in TurboQuant+ fork
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
- Prefix caching is automatic in recent llama.cpp builds when system prompt is static
- Startup check: `verify_prefix_cache()` sends identical prompt twice, confirms TTFT reduction
- Log cache hit status to requests.jsonl (null if unverified, true/false after check)

### 3. router.py — TF-IDF + LogReg 3-Way Classifier

**Purpose:** Classify user input into one of 3 routes in <1ms.

**Routes:**
- `PRIMARY` — coding, multi-step reasoning, complex Q&A, planning
- `SPECIALIST` — classification, extraction, formatting, simple yes/no, summarization
- `TOOL_ONLY` — regex matching, simple parsing, no LLM needed

**Note:** MULTIMODAL is NOT a router class. Multimodal detection is a structural pre-check in cli.py (image URL, file path, or audio reference in input). This is more reliable than text classification for detecting vision/audio needs.

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
- 200+ examples covering 3 classes (no MULTIMODAL examples needed)
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

**Default budget (32K context, all values parameterized in configs/models.yaml):**
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
- ~5-20ms per call, 4-6 calls per request = 40-120ms total overhead
- ponytail: first optimization target for Phase 1.5 — cache local tokenizer (tiktoken/gpt2 BPE) to eliminate HTTP round-trips
- Acceptable for Phase 1, must not block

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
- Embed raw text of each turn pair (first 500 chars of user + assistant exchange)
- No summarization step — embedding model handles raw text fine
- Summarization before embedding is a Phase 2 optimization, not needed for 200 entries
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
    def store(text: str, role: str) -> None        # embed raw text + store
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

**Multimodal pre-check (structural, not classifier):**
- Before routing, check input for: image file paths (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp`), image URLs (`.png`/`.jpg` in URL), audio file paths (`.wav`, `.mp3`, `.flac`, `.ogg`), or explicit `/image` or `/audio` prefix commands
- If detected → MULTIMODAL path: swap in Gemma 4 E4B, dispatch to it
- If not detected → proceed to router for 3-way classification
- This is more reliable than text classification for detecting multimodal needs

**Modes:**
- Single-shot: `python -m lore "write a function to reverse a list"`
- Interactive: `python -m lore -i` (REPL)
- JSON mode: `python -m lore --json "extract names from: John, Mary, Bob"` (triggers GBNF)

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
4. Verify prefix cache active (`verify_prefix_cache()`)
5. Load trained router model
6. Initialize context manager + episodic memory
7. Enter REPL or process single-shot input

### 8. __main__.py — Module Entry Point

**Purpose:** Enable `python -m lore "query"` syntax (cleaner than `python -m lore.cli`).

```python
# src/lore/__main__.py
from lore.cli import main
main()
```

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
  path: models/  # exact GGUF filename to verify on huggingface.co/nomic-ai/nomic-embed-text-v1.5
  context: 8192
  embedding_dim: 768

multimodal:
  name: gemma-4-e4b
  port: 19003
  path: models/gemma-4-e4b-Q4_K_M.gguf  # needs download
  quant: Q4_K_M
  expected_size_gb: 5.0
  context: 16384           # reduced context for swap model to save memory
  swap_ttl_seconds: 120    # unload after 120s idle

# Context budget allocation (parameterized, not hardcoded)
context_budget:
  system_prompt: 512
  retrieved_memories: 1024
  working_context: 4096
  user_input: 2048
  generation_headroom: 4096
```

### configs/router.yaml

```yaml
confidence_threshold: 0.70
training_data_path: configs/router_training_data.jsonl
model_path: configs/router_model.joblib
classes:
  - PRIMARY
  - SPECIALIST
  - TOOL_ONLY
tfidf:
  ngram_range: [1, 2]
  max_features: 5000
classifier:
  max_iter: 1000
  class_weight: balanced
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
max_text_chars: 500  # embed first 500 chars of raw text, no summarization
```

## Memory Layout

```
PERSISTENT (always loaded):
  Ornith-9B turbo4 32K       ~5.7 GB   (port 19000)
  Falcon-H1 turbo4 32K       ~1.2 GB   (port 19001)
  nomic-embed-text-v1.5       0.3 GB   (port 19002)
  ─────────────────────────────────────
  Total persistent            ~7.2 GB   (6.8 GB headroom to 14 GB)

HOT-SWAP (loaded on demand via direct process management):
  Gemma 4 E4B Q4_K_M          ~5.0 GB   (port 19003, started/killed as needed)
  When loaded: total ~12.2 GB (1.8 GB headroom — tight but fits)
```

## Dependencies

```
scikit-learn>=1.4
pyyaml>=6.0
requests>=2.31
numpy>=1.26
joblib>=1.3
pytest>=8.0
```

All lightweight, no GPU/cuda deps. Python 3.11+ required.
**Note:** This machine has Python 3.14. scikit-learn, numpy, and joblib should work on 3.14 but verify on first install. If any issues, Python 3.12 is the safe fallback (use `python3.12 -m venv`).

## Error Handling Summary

| Failure | Action | Log |
|---------|--------|-----|
| Server fails to start | Retry with halved context, then primary-only mode | error + fallback |
| Specialist request fails | Retry on primary | fallback: true |
| Swap fails (Gemma won't start) | Return error to user | error: "multimodal unavailable" |
| Port conflict | Increment port, retry | warning |
| Health check timeout | 3 retries, 2s backoff, then degrade | error |
| Router model missing | Default all to PRIMARY | warning |
| Embeddings server down | Skip memory retrieval, continue | warning |
| Tokenize endpoint fails | Estimate tokens as len(text)/4 | warning |
| Prefix cache not active | Log warning, continue (still works, just slower) | warning |

## Testing

- `tests/test_router.py` — unit test classification accuracy on held-out data
- `tests/test_context.py` — unit test token budget truncation
- `tests/test_models.py` — integration test with mock HTTP server
- `tests/test_memory.py` — unit test embedding storage + retrieval
- `assert`-based `__main__` self-check in each module (ponytail pattern)

## Phase 1 Exit Criteria

- [ ] Router >85% accuracy on held-out test set (3-way classification)
- [ ] 100% valid JSON output in JSON mode (GBNF)
- [ ] Prefix cache verified active (verify_prefix_cache startup check passes)
- [ ] Episodic memory retrieves relevant context (cosine sim >0.5)
- [ ] Gemma 4 E4B swap-in/swap-out works via direct process management
- [ ] All routing decisions logged to requests.jsonl
- [ ] Error handling: no crash on server failure, graceful fallback
- [ ] Memory <14 GB at all times (including during swap)
