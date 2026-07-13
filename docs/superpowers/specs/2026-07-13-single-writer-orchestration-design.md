# Single-writer orchestration design

**Date:** 2026-07-13  
**Status:** Approved  
**Scope:** Replace harmful coding decomposition with a direct, tool-aware primary path while retaining bounded specialist routing and safe independent-deliverable orchestration.

## Verdict

The measured dual-model memory and latency thesis is viable. Ornith-1.0-9B, Falcon-H1-1.5B, embeddings, compressed KV state, and runtime overhead fit the 16 GB target with required headroom. Deterministic routing and a small specialist can also avoid unnecessary primary-model calls.

Quality uplift from decomposition is not established. Current coding decomposition is harmful because exploration, diagnosis, patch generation, and aggregation occur in separate contexts. Repository evidence is truncated between workers, several model calls can independently invent assumptions, and the final aggregator can alter otherwise valid code. The system must not claim a quality gain until a controlled comparison demonstrates one.

The approved design therefore uses Ornith-1.0-9B as the only coding writer. A repository coding task stays in one primary-model conversation from exploration through patch generation and validation. Falcon-H1 remains loaded because its memory cost is acceptable, but its role is narrow and never includes code generation, code review, patch aggregation, or summarizing code or patches.

## Goals

1. Keep one continuous repository-tool transcript for every coding task.
2. Route uncertain, stateful, or coding work directly to the primary model.
3. Retain the deterministic `TOOL_ONLY` path without an LLM call.
4. Use the specialist only for high-confidence, bounded natural-language work.
5. Permit decomposition only when deliverables are independent and share no files, mutable state, ordered reasoning, or output dependencies.
6. Preserve CLI and OpenAI-compatible API entry points and response shapes.
7. Repair correctness, lifecycle, path, and request-control defects without running long benchmarks.

## Non-goals

- Proving that orchestration improves benchmark quality.
- Running HumanEval, SWE-bench, complex-task, latency, memory, or leaderboard benchmarks.
- Adding an always-on LLM classifier or a new routing model.
- Letting the specialist generate, review, transform, truncate, or summarize code or patches.
- Supporting streaming API responses.
- Changing model selection, quantization, model files, ports, public CLI commands, or HTTP endpoint paths.
- Replacing SEARCH/REPLACE with unified diffs.
- Rewriting hierarchical memory, Tool Attention, LLMLingua-2, or the model registry beyond the listed repairs.

## Architecture

```text
CLI query or API request
        |
        v
request normalization
  full API message history
  bounded max_tokens and temperature
  request-scoped context budget
        |
        v
single structural and TF-IDF routing decision
        |
        +--> TOOL_ONLY
        |      deterministic handler
        |
        +--> approved SPECIALIST
        |      bounded extraction, classification,
        |      or short natural-language summary
        |      failure falls back to primary
        |
        +--> repository coding task
        |      one primary tool loop
        |      read/search/list in one transcript
        |      SEARCH/REPLACE output
        |      deterministic patch validation
        |
        +--> independent deliverables
        |      optional bounded parallel execution
        |      only when no shared files, state, or dependencies
        |
        +--> all other work
               direct primary generation
        |
        v
verification, memory update, request log, response
```

`Orchestrator.process()` remains the compatibility entry point, but becomes a conservative coordinator. It consumes one routing decision, recognizes the repository coding path, optionally schedules demonstrably independent deliverables, and otherwise delegates to direct dispatch. It no longer performs general LLM decomposition before ordinary work.

## Routing decision table

| Input condition | Execution path | Model | Constraints |
|---|---|---|---|
| Structural multimodal reference | Existing multimodal path | Multimodal model | Swap lifecycle and cleanup remain required |
| TF-IDF route is `TOOL_ONLY` and deterministic handler supports the query | Tool fast path | None | No model call, unchanged behavior |
| TF-IDF route is `SPECIALIST`, confidence meets the configured gate, and task is bounded extraction or classification | Specialist direct | Falcon-H1 | Fixed output bound, structured validation where applicable |
| TF-IDF route is `SPECIALIST`, confidence meets the gate, and task requests a short natural-language summary of prose | Specialist direct | Falcon-H1 | Source must be prose, output must be short |
| Request contains code, a patch, repository work, debugging, tests, implementation, or file mutation | Single-writer repository or direct path | Ornith | Specialist forbidden, no decomposition, no code summarization |
| Deliverables are independent and have no shared files, state, dependencies, or ordered reasoning | Independent-deliverable path | Ornith and bounded specialist where eligible | Every deliverable is self-contained; no LLM aggregation that rewrites code |
| Route is uncertain, unsupported, or below confidence threshold | Direct path | Ornith | Safe default |
| Specialist call fails or violates output validation | Direct fallback | Ornith | Reuse the original source, not a specialist-generated summary |

The TF-IDF router is evaluated once per request. The model-based `TaskClassifier` is not created during normal CLI or API startup and does not participate in routing. Existing heuristic complexity signals may be used only to reject decomposition, never to force it.

## Data flows

### Direct primary flow

1. Normalize the request and resolve one `RouteDecision`.
2. Add the user turn once.
3. Retrieve memory once through `ContextManager.build_prompt()`.
4. Apply a request-scoped budget without mutating the configured default.
5. Call the primary with the caller's validated generation controls.
6. Validate output, then store the user and assistant turns once.
7. Log the original route, selected model, fallback state, and latency.

This removes the current duplicate router call between `Orchestrator.process()` and `_dispatch()`, and the duplicate memory retrieval between `_execute_query()` and `ContextManager.build_prompt()`.

### Repository coding flow

1. Construct `RepoContext` from a contained repository root.
2. Start one Ornith conversation with the issue, repository tool instructions, API or CLI history, and selected relevant memory.
3. Keep every `LIST_DIR`, `SEARCH`, and `READ_FILE` result in that same conversation. Tool output may be bounded, but code is never summarized by Falcon-H1.
4. Require the primary to emit SEARCH/REPLACE blocks for mutations.
5. Parse blocks with `search_replace.parse_edit_blocks()`.
6. Reject absolute paths, traversal, symlink escape, malformed blocks, duplicate conflicting edits, and files outside the repository root.
7. Apply each block to an in-memory copy with `apply_edit_blocks()`. A block that cannot match is invalid.
8. For changed Python content, run deterministic syntax validation. Use existing verifier checks for other supported structured outputs.
9. Return the validated patch. If validation fails, give the same primary conversation the exact deterministic error and allow one bounded repair. If repair fails, return failure without presenting an unvalidated patch as successful.

There is no explorer worker, diagnosis worker, patch worker, specialist pre-summary, or code aggregation call. This preserves repository observations and generated patch text in one writer context.

### Specialist flow

The specialist receives the original bounded source and an explicit output limit. Accepted work is limited to:

- TF-IDF-approved simple classification.
- Bounded extraction into validated JSON or a similarly constrained format.
- A short natural-language summary of prose.

The specialist is not used for complexity classification at startup, planning, repository exploration, code, patches, tests, technical code summaries, or aggregation. If the input contains code fences, SEARCH/REPLACE markers, a diff, or repository-file content for mutation, route to primary.

### Independent-deliverable flow

Decomposition is deterministic and conservative. It is permitted only when the request explicitly names multiple deliverables that can each be completed and returned unchanged without consuming another deliverable's output. The scheduler rejects any dependency edge, shared target path, shared mutable state, or ambiguous ownership.

Eligible examples include extracting two unrelated fields from two separate prose documents or producing separate prose summaries for unrelated inputs. A repository change plus tests is not eligible because both depend on the same implementation state. Multiple changes in one repository are not eligible even when they mention different files, because imports, interfaces, and tests can couple them.

Results are returned in stable request order. Code output is never passed through an LLM aggregator. If independence cannot be proven, execute the original request directly on the primary.

## Request and context controls

Existing CLI calls continue to work with their current defaults. Existing API clients continue using `POST /v1/chat/completions`, and response fields remain compatible.

The API must honor:

- `messages`: preserve the supplied ordered history and use the final user message as the current turn. Do not discard prior system, user, or assistant messages.
- `max_tokens`: validate as a positive integer, clamp to the configured server-safe range, and pass it to the selected generation call.
- `temperature`: validate as a finite number in the supported range and pass it to generation.
- `response_format`: preserve current JSON mode behavior.
- `stream`: continue returning the current explicit unsupported error when true.

Optional parameters are carried through request-scoped execution options rather than written into shared configuration. CLI defaults remain `max_tokens=2048` and `temperature=0.7` unless an existing mode selects stricter values.

`ContextManager.set_budget()` must not cause budget drift across later requests. Budget selection becomes scoped to one prompt build or is restored in `finally`. The budget includes the system prompt, supplied history, retrieved memory, current input, tool results, and generation headroom. The configured default remains unchanged after every success or failure.

## Configuration and path handling

`LoreConfig.load(config_dir)` resolves `config_dir` once and retains its absolute directory. Relative paths in loaded configuration are resolved against the project or configuration root, not the process working directory. CLI and API initialization use the same loader for models, router, sessions, compression, verifier, and orchestrator settings. Existing string values and environment override precedence remain compatible.

Containment rules apply before filesystem access:

- A session ID is a single safe path component. Empty IDs, `.`, `..`, separators, absolute paths, and resolved paths outside `save_dir` are rejected.
- Every repository file or directory argument is resolved against `RepoContext.path` and must remain beneath that root after symlink resolution.
- Search globs are treated as patterns, not paths or command fragments. Search continues to use argument-vector subprocess calls without a shell.
- Model and server paths may be absolute or root-relative, but their resolution must not depend on the caller's current directory.

## Failure handling and lifecycle

### Startup and shutdown

Model startup is transactional per role. If `Popen` fails or health checking fails, terminate and wait for the child, close the log handle, and remove both process and handle entries. If application initialization fails after any role started, `stop_all()` runs before the error escapes.

CLI single-shot, REPL, and API server paths own one shutdown boundary. `stop_all()` runs exactly once in a top-level `finally`, including parser exits, initialization errors, request errors, keyboard interruption, and API shutdown. Existing public startup and shutdown methods remain available.

### Specialist offload and reload

If specialist offload remains enabled for an eligible path, reload is paired with offload in `try/finally`. Reload is attempted after worker failure, validation failure, aggregation failure, fallback, or cancellation. Reload failure is logged and reflected in health state, but must not replace a valid primary result.

### Dependency cycles

Dependency cycles and missing dependencies invalidate a plan. `_build_waves()` must not force cyclic nodes into a wave. Invalid plans fall back to the original direct primary request. This prevents execution with unsatisfied inputs.

### Model and validation failures

- Specialist failure falls back once to the primary using the original request and history.
- Primary generation failure returns the existing error-shaped result and does not write an assistant success turn.
- Invalid JSON may use existing deterministic repair.
- Invalid SEARCH/REPLACE output receives one primary repair attempt with exact errors.
- Memory, health, logging, or cache-verification failure remains non-fatal and is logged.
- Multimodal swap failure preserves the existing explicit failure behavior and cleans up partial processes.

## Repairs and simplifications

| Area | Approved change |
|---|---|
| Routing | Compute one structural and TF-IDF decision, then pass it through orchestration and dispatch |
| Retrieval | Let `ContextManager` own retrieval; remove caller-side duplicate retrieval |
| Coding | Replace explorer, analyzer, writer, and aggregator stages with one primary tool loop |
| Classifier | Disable normal startup construction and remove it from the request critical path |
| Decomposition | Restrict to provably independent deliverables; direct primary otherwise |
| Aggregation | Remove specialist pre-summary for code and patches; never rewrite code through aggregation |
| Context | Make budgets and API controls request-scoped and account for full prompt use |
| Configuration | Resolve all configured paths from one retained root |
| Lifecycle | Centralize startup cleanup and top-level shutdown; reload specialist in `finally` |
| Plans | Reject cycles and missing dependencies instead of forcing execution |
| Filesystem | Enforce session and repository root containment |
| Leaderboard baseline | Fix the local-import shadowing bug so the optional pandas loader or injected test double is resolved consistently without requiring pandas at package import |

The existing `TaskDecomposer`, `Worker`, and aggregation APIs may remain as compatibility surfaces, but normal coding and ordinary request paths do not invoke their general decomposition behavior.

## Compatibility

- Preserve `lore` single-shot and interactive commands.
- Preserve `/save`, `/resume`, `/sessions`, `/switch`, `/models`, and `/upgrades`.
- Preserve `GET /health`, `GET /v1/models`, and `POST /v1/chat/completions`.
- Preserve the current response dictionary keys, including `route`, `confidence`, `model`, `content`, `success`, `latency_ms`, `orchestrated`, and `subtasks_completed`.
- Preserve JSON mode, deterministic tool handling, multimodal structural detection, specialist-to-primary fallback, request logging, and memory storage.
- Add only optional internal parameters to dispatch and orchestration methods so existing positional and keyword callers remain valid.
- Continue accepting existing YAML files. New defaults must make the approved direct behavior active without requiring users to rewrite configuration.

## Test plan

Add focused tests to existing test modules or narrowly named new test modules:

1. One router classification per request, including direct, specialist, and tool-only paths.
2. One memory retrieval per model request.
3. A coding request always selects primary and never invokes the classifier, specialist, decomposer, pre-summary, or LLM aggregator.
4. Repository tool calls and the final patch use one continuous primary message transcript.
5. Valid SEARCH/REPLACE blocks match current file content; malformed, unmatched, escaping, and conflicting blocks fail validation.
6. Python syntax validation runs on the in-memory patched result.
7. Specialist accepts bounded extraction and short prose summarization, but code fences, diffs, and SEARCH/REPLACE input route to primary.
8. Only independent deliverables are parallelized; shared paths, dependencies, ambiguous state, missing dependencies, and cycles fall back to direct primary.
9. `TOOL_ONLY` remains deterministic and makes no model call.
10. API forwards full ordered history, clamped `max_tokens`, validated `temperature`, and JSON mode while preserving the response schema.
11. Request budget changes do not affect the next request, including exception paths.
12. Config and model paths resolve correctly when the process starts outside the repository.
13. Failed spawn and failed health check leave no tracked process or open log handle.
14. Partial application initialization calls `stop_all()`.
15. Specialist reload is attempted from `finally` after execution and aggregation failures.
16. Session IDs and repository paths reject traversal, absolute paths, separators where forbidden, and symlink escape.
17. Leaderboard fallback works with the optional local pandas import absent and with an injected test double.
18. Existing SEARCH/REPLACE parser and fuzzy-application tests continue to pass.

No long benchmark is part of verification. The only validator command is:

```bash
PYTHONPATH=src python -m pytest tests/ -q --tb=short
```

## Migration

1. Land focused regression tests for routing, retrieval, request controls, lifecycle, cycles, and containment.
2. Introduce the single route decision and request-scoped execution controls while retaining existing method signatures.
3. Move repository coding requests to the continuous primary tool loop and deterministic patch validator.
4. Disable the always-on classifier and general coding decomposition in default configuration.
5. Restrict the remaining scheduler to deterministic independent deliverables.
6. Remove code and patch pre-summarization from reachable paths.
7. Run the single approved validator command.

The final implementation must preserve the current uncommitted SEARCH/REPLACE work in `src/lore/decomposer.py` and `src/lore/worker.py`. It must also preserve the generated changes in `benchmarks/results/swebench_predictions.jsonl` and `benchmarks/results/swebench_smoke_results.json`. Those files are existing user-selected work, not cleanup targets. Changes may be integrated with focused tests, but must not be reverted, replaced with older versions, or discarded.

## Rollback

Rollback is configuration-first. Keep compatibility methods and the old decomposition classes available during migration. A single internal feature flag may restore the previous general orchestration path for diagnosis, but the approved default remains single-writer direct execution. Rolling back must not alter session data, generated benchmark outputs, SEARCH/REPLACE WIP, model files, or public interfaces.

If a regression requires code rollback, revert only the coordinator selection and request-scoping changes, then run the same validator command. Do not re-enable the specialist for code or patch summarization as a partial rollback.
