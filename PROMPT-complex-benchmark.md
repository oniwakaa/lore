# Prompt: Build Complex Task Benchmark to Validate Orchestration

## Identity & Context

You are working on **LORE** (Local Orchestration & Runtime Engine) at `~/projects/lore`. Read `AGENTS.md` for project context. Run with `PYTHONPATH=src`.

LORE recently scored 87% on HumanEval — but that benchmark is all single-function tasks. The classifier correctly routes 94% of them directly. Orchestration barely activates. We need a benchmark where orchestration SHOULD activate and SHOULD produce better results than direct dispatch.

## The Mission

Create a benchmark of 20-30 complex coding tasks that **genuinely require decomposition** — multi-file outputs, multi-step reasoning, tasks too large for a single context window. Run LORE on these tasks and measure whether orchestration produces better results than a single model call.

## Task Design Principles

Each task should be:
1. **Multi-step** — requires 3+ distinct operations (create, modify, test, etc.)
2. **Multi-output** — produces 2+ files or artifacts
3. **Verifiable** — has programmatic tests or checks
4. **Too large for comfortable single-pass generation** — would require 3000+ tokens of output
5. **Natural decomposition** — has obvious subtask boundaries

## The 25 Tasks

```python
COMPLEX_TASKS = [
    # --- API BUILDING (5 tasks) ---
    {
        "id": "api-1",
        "category": "api",
        "prompt": "Build a Python FastAPI app with: 1) User model with Pydantic (id, email, name, created_at), 2) POST /users with email validation and duplicate check, 3) GET /users/{id} with 404 handling, 4) GET /users with pagination (offset/limit), 5) In-memory store with thread-safe locking. Include all imports. Make it runnable with `uvicorn app:app`.",
        "test": """
import subprocess, sys, time, requests, threading
# Start server
proc = subprocess.Popen([sys.executable, '-c', '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional
import threading
# ... (the generated code should define app)
'''], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# Basic checks: file exists, has correct structure
code = open('api_1.py').read() if __import__('os').path.exists('api_1.py') else ''
assert 'FastAPI' in code or 'app' in code.lower(), 'Missing FastAPI app'
assert 'POST' in code or 'post' in code.lower(), 'Missing POST endpoint'
assert 'GET' in code or 'get' in code.lower(), 'Missing GET endpoint'
assert 'pydantic' in code.lower() or 'BaseModel' in code, 'Missing Pydantic model'
print('PASS')
""",
        "verification": "structural",  # check code structure, not runtime
    },
    {
        "id": "api-2",
        "category": "api",
        "prompt": "Create a REST API client library in Python with: 1) BaseClient class with configurable base_url, timeout, retries, 2) Automatic JSON serialization/deserialization, 3) Retry with exponential backoff on 5xx errors, 4) Request/response logging, 5) Type hints on all public methods, 6) Context manager support (`async with`). Include docstrings.",
        "test": "structural",
    },
    {
        "id": "api-3",
        "category": "api",
        "prompt": "Build a SQLite-backed CRUD API with: 1) SQLAlchemy models for Author (id, name) and Book (id, title, author_id FK), 2) Alembic migration, 3) Pydantic schemas for request/response, 4) FastAPI endpoints for both resources, 5) Relationship queries (author with books count). Include the migration file content.",
        "test": "structural",
    },
    {
        "id": "api-4",
        "category": "api",
        "prompt": "Create a webhook receiver service: 1) POST /webhooks endpoint that validates HMAC-SHA256 signatures, 2) Stores payloads in SQLite, 3) GET /webhooks to list recent payloads, 4) Async processing with a background task queue, 5) Retry failed webhooks 3 times. Include the signature verification logic.",
        "test": "structural",
    },
    {
        "id": "api-5",
        "category": "api",
        "prompt": "Build a rate-limited API gateway middleware: 1) Sliding window rate limiter per API key, 2) Redis or in-memory backend, 3) Configurable limits per endpoint, 4) Proper HTTP 429 responses with Retry-After header, 5) Rate limit headers (X-RateLimit-*). Include FastAPI middleware integration.",
        "test": "structural",
    },

    # --- CLI TOOLS (5 tasks) ---
    {
        "id": "cli-1",
        "category": "cli",
        "prompt": "Create a Python CLI tool (argparse) that: 1) Reads a CSV file, 2) Validates each row against a configurable schema (column types, required fields, value ranges), 3) Writes valid rows to output.csv, 4) Writes invalid rows to errors.jsonl with row number and error details, 5) Prints summary (total, valid, invalid, error types). Handle edge cases: empty files, missing columns, encoding issues.",
        "test": "functional",  # can run with test CSV
    },
    {
        "id": "cli-2",
        "category": "cli",
        "prompt": "Build a file organizer CLI: 1) Scans a directory recursively, 2) Groups files by type (images, documents, code, data, etc.), 3) Optionally renames files with date prefix from EXIF/mtime, 4) Creates organized directory structure, 5) Dry-run mode that shows what would happen, 6) Undo capability via operation log. Use pathlib, handle symlinks.",
        "test": "structural",
    },
    {
        "id": "cli-3",
        "category": "cli",
        "prompt": "Create a log analyzer CLI that: 1) Parses common log formats (Apache, Nginx, JSON), 2) Aggregates stats: requests/sec, error rate, p50/p95/p99 latency, 3) Detects anomalies (spike in errors, latency outliers), 4) Outputs report as markdown table, 5) Supports time-window filtering (--since, --until). Handle gzipped logs.",
        "test": "structural",
    },
    {
        "id": "cli-4",
        "category": "cli",
        "prompt": "Build a database migration CLI: 1) Connects to PostgreSQL or SQLite, 2) Reads migration files (SQL or Python), 3) Tracks applied migrations in a schema_migrations table, 4) Supports up/down migrations, 5) Shows migration status, 6) Dry-run mode. Include example migration files.",
        "test": "structural",
    },
    {
        "id": "cli-5",
        "category": "cli",
        "prompt": "Create an API mock server CLI: 1) Reads OpenAPI/Swagger spec, 2) Generates mock responses matching the schema, 3) Supports configurable delays and error rates, 4) Records actual requests for replay, 5) Hot-reload when spec file changes. Use FastAPI under the hood.",
        "test": "structural",
    },

    # --- TEST SUITES (5 tasks) ---
    {
        "id": "test-1",
        "category": "testing",
        "prompt": "Write a comprehensive pytest test suite for a Stack class with push, pop, peek, is_empty, size methods. Include: 1) Basic operations (10+ tests), 2) Edge cases (empty stack, single element), 3) LIFO ordering verification, 4) Type error handling, 5) Size consistency after operations, 6) Fixture for pre-populated stack, 7) Parametrized tests for multiple data types, 8) Property-based test using Hypothesis.",
        "test": "functional",  # can run the tests
    },
    {
        "id": "test-2",
        "category": "testing",
        "prompt": "Write integration tests for a SQLite repository pattern. The repository has: save(entity), find_by_id(id), find_all(filters), delete(id), count(). Tests should cover: 1) CRUD operations, 2) Filter combinations, 3) Non-existent records, 4) Duplicate handling, 5) Transaction rollback on error, 6) Concurrent access, 7) Large dataset performance. Use pytest fixtures for DB setup/teardown.",
        "test": "functional",
    },
    {
        "id": "test-3",
        "category": "testing",
        "prompt": "Create a test harness for a REST API client. Mock the HTTP layer and test: 1) Successful requests, 2) Retry on 5xx, 3) Timeout handling, 4) Authentication header injection, 5) Request/response logging, 6) JSON decode errors, 7) Connection errors, 8) Rate limit (429) handling. Use unittest.mock and pytest.",
        "test": "functional",
    },
    {
        "id": "test-4",
        "category": "testing",
        "prompt": "Write end-to-end tests for a FastAPI app with auth: 1) Registration flow (create user, verify email), 2) Login/logout with JWT, 3) Protected endpoint access, 4) Token refresh, 5) Invalid credentials, 6) Expired tokens, 7) Role-based access (admin vs user). Use httpx.AsyncClient and pytest-asyncio.",
        "test": "functional",
    },
    {
        "id": "test-5",
        "category": "testing",
        "prompt": "Create a benchmark test suite that measures: 1) Function execution time, 2) Memory usage via tracemalloc, 3) Comparison against baseline, 4) Regression detection (fail if >10% slower), 5) Report generation (table with p50, p95, p99). Decorator-based: @benchmark(name='test_name', baseline_ms=100).",
        "test": "structural",
    },

    # --- DATA PIPELINES (5 tasks) ---
    {
        "id": "data-1",
        "category": "data",
        "prompt": "Build a CSV-to-JSON transformation pipeline: 1) Read CSV with auto-detected encoding and delimiter, 2) Apply column mappings and type conversions, 3) Handle nested JSON structures from flat CSV (dot notation: address.city), 4) Validate against JSON schema, 5) Output valid records to JSON, invalid to error log, 6) Stream processing for large files (no full load into memory).",
        "test": "structural",
    },
    {
        "id": "data-2",
        "category": "data",
        "prompt": "Create a data deduplication engine: 1) Reads records from JSON/CSV, 2) Computes similarity using multiple strategies (exact match, fuzzy string, phonetic), 3) Configurable similarity threshold, 4) Groups duplicates, 5) Outputs deduplication report with confidence scores, 6) Preserves original records with merge suggestions.",
        "test": "structural",
    },
    {
        "id": "data-3",
        "category": "data",
        "prompt": "Build a time-series data aggregator: 1) Reads timestamped events (JSON lines), 2) Groups by configurable window (minute, hour, day), 3) Computes aggregations (count, sum, avg, p50, p95, p99), 4) Handles out-of-order events with watermark, 5) Outputs to CSV with proper formatting, 6) Memory-efficient streaming for large datasets.",
        "test": "structural",
    },
    {
        "id": "data-4",
        "category": "data",
        "prompt": "Create a schema migration tool for JSON data: 1) Defines schema versions (v1, v2, v3), 2) Auto-detects current version, 3) Applies incremental transforms (rename field, split field, merge fields, add default), 4) Validates after each transform, 5) Supports rollback, 6) Logs all changes. Include 3 example schema versions and transforms.",
        "test": "structural",
    },
    {
        "id": "data-5",
        "category": "data",
        "prompt": "Build an ETL pipeline framework: 1) Define extractors (CSV, JSON, API, DB), 2) Define transforms (filter, map, aggregate, join), 3) Define loaders (CSV, JSON, SQLite), 4) Pipeline builder with chaining, 5) Error handling per record (skip, retry, dead letter), 6) Progress reporting. Functional API style: pipeline = extract(csv).filter(...).transform(...).load(sqlite).",
        "test": "structural",
    },

    # --- MULTI-FILE PROJECTS (5 tasks) ---
    {
        "id": "proj-1",
        "category": "project",
        "prompt": "Create a Python package structure for a task queue library: 1) __init__.py with public API, 2) task.py with Task dataclass (id, payload, status, retries, created_at), 3) queue.py with TaskQueue class (enqueue, dequeue, ack, nack, size), 4) worker.py with Worker class (process loop, error handling, graceful shutdown), 5) storage.py with pluggable backends (memory, JSON file), 6) tests/ with pytest tests for each module. Include pyproject.toml.",
        "test": "structural",
    },
    {
        "id": "proj-2",
        "category": "project",
        "prompt": "Build a plugin system: 1) Base Plugin ABC with lifecycle hooks (on_load, on_enable, on_disable, on_unload), 2) PluginManager that discovers plugins from a directory, 3) Plugin dependency resolution (topological sort), 4) Event bus for inter-plugin communication, 5) Configuration per plugin (YAML), 6) Example plugins (logger plugin, metrics plugin). Include type hints and docstrings.",
        "test": "structural",
    },
    {
        "id": "proj-3",
        "category": "project",
        "prompt": "Create a caching library with: 1) Cache ABC, 2) MemoryCache with LRU eviction, 3) FileCache with TTL, 4) Cache decorator with configurable TTL and key generation, 5) Cache invalidation patterns (tag-based, pattern-based), 6) Statistics (hit rate, miss rate, eviction count), 7) Thread-safe implementation. Include benchmarks comparing implementations.",
        "test": "structural",
    },
    {
        "id": "proj-4",
        "category": "project",
        "prompt": "Build a configuration management library: 1) Support YAML, JSON, TOML, ENV files, 2) Layered config (defaults → file → env vars → CLI args), 3) Schema validation with Pydantic, 4) Hot-reload on file change (watchdog), 5) Secrets management (env vars, not in files), 6) Type-safe access with dot notation. Include example config files for each format.",
        "test": "structural",
    },
    {
        "id": "proj-5",
        "category": "project",
        "prompt": "Create a CLI application framework: 1) Command registration via decorators, 2) Argument parsing with type validation, 3) Subcommand support, 4) Help generation, 5) Output formatting (table, JSON, plain text), 6) Interactive prompts (confirm, select, input), 7) Progress bars for long operations. Build it as a reusable library, then implement a sample CLI (todo app) using it.",
        "test": "structural",
    },
]
```

## Implementation

### Step 1: Save tasks to `benchmarks/eval_tasks/complex_tasks.json`

Convert the Python structure above to JSON format compatible with the benchmark script.

### Step 2: Add `--benchmark complex` mode

Extend `scripts/benchmark_orchestration.py` to support `--benchmark complex`:

```python
if args.benchmark == "complex":
    tasks = load_complex_tasks()  # from complex_tasks.json
    # Run ONLY the orchestrated path — no direct comparison needed
    # The question is: does orchestration produce correct output?
```

### Step 3: Correctness Verification

For each task, verify the output:

- **Structural checks** (for "structural" tasks): Does the code contain expected keywords, class names, method signatures?
- **Functional checks** (for "functional" tasks): Can the code be imported without errors? Can test functions be run?

```python
def verify_complex_task(task_id: str, content: str, verification_type: str) -> dict:
    """Verify complex task output."""
    if verification_type == "structural":
        # Check code structure
        has_code = "```" in content or "def " in content or "class " in content
        has_imports = "import " in content or "from " in content
        return {"passed": has_code and has_imports, "method": "structural"}
    
    elif verification_type == "functional":
        # Extract code and try to import it
        code = extract_code(content, "")
        try:
            compile(code, f"{task_id}.py", "exec")
            return {"passed": True, "method": "compile"}
        except SyntaxError as e:
            return {"passed": False, "method": "compile", "error": str(e)}
```

### Step 4: Measure Orchestration Quality

For each task, record:
- Did the classifier route it as complex? (should be yes for all)
- How many subtasks were created?
- How many waves?
- Did all subtasks succeed?
- How long did aggregation take?
- What was the total latency?

### Step 5: Output Report

```
═══════════════════════════════════════════════════════════════
 LORE Complex Task Benchmark
═══════════════════════════════════════════════════════════════

 Task        │ Category │ Orch? │ Subtasks │ Passed │ Latency
 ────────────┼──────────┼───────┼──────────┼────────┼─────────
 api-1       │ api      │ yes   │ 3        │ ✓      │ 120s
 cli-1       │ cli      │ yes   │ 4        │ ✓      │ 180s
 test-1      │ testing  │ yes   │ 2        │ ✓      │ 90s
 ...

 ─── SUMMARY ─────────────────────────────────────────────────
 Tasks: 25 | Orchestrated: N/25 | Passed: N/25
 Avg subtasks per task: X.X | Avg latency: Xs
 Orchestrated pass rate: XX%
 ══════════════════════════════════════════════════════════════
```

## Files to Read First

1. `scripts/benchmark_orchestration.py` — Current benchmark script (extend, don't replace)
2. `src/lore/orchestrator.py` — Orchestrator.process() API
3. `src/lore/decomposer.py` — TaskPlan, SubTask structures
4. `benchmarks/eval_tasks/orchestration_ab.json` — Reference format

## Constraints

1. **No new dependencies**
2. **Extend existing benchmark script** — add `--benchmark complex` alongside `--benchmark humaneval`
3. **Save results incrementally** to `benchmarks/results/complex_tasks_lore.json`
4. **Run with `--limit 5` first** to validate pipeline before full 25 tasks
5. **All tasks should be genuinely complex** — if you find a task that the classifier routes as simple, that's a data point, not a failure

## When to Ask

- If a task's verification is ambiguous (structural vs functional), use structural (more lenient)
- If the classifier routes most tasks as simple, document it — that means the classifier needs improvement for multi-step tasks
- If orchestration pass rate is below 50%, check decomposition quality before assuming the tasks are wrong
