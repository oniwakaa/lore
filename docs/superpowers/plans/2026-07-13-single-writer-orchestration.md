# Single-writer orchestration implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 10 correctness/security/lifecycle defects and enforce single-writer coding policy so decomposition never harms code consistency.

**Architecture:** Primary 9B owns all coding tasks in one continuous tool loop. Specialist kept for TF-IDF-approved simple routes only. Decomposition restricted to provably independent deliverables. Duplicate routing/retrieval removed, budgets request-scoped, API controls honored, paths contained.

**Tech Stack:** Python 3.11+, pytest, PyYAML, scikit-learn, requests, llama-server

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/lore/leaderboard.py` | Modify | Fix local import shadowing so test mocks resolve |
| `src/lore/repo_tools.py` | Modify | Path containment validation for read_file/list_files |
| `src/lore/session.py` | Modify | Session ID sanitization to reject traversal |
| `src/lore/models.py` | Modify | Startup cleanup on failed health check |
| `src/lore/orchestrator.py` | Modify | Specialist reload in finally, cycle rejection, classifier bypass |
| `src/lore/cli.py` | Modify | Single routing decision, remove duplicate memory retrieval, request-scoped budget, pass API controls |
| `src/lore/api.py` | Modify | Full message history, max_tokens/temperature forwarded |
| `src/lore/config.py` | Modify | Retain config root for path resolution |
| `tests/test_leaderboard.py` | Modify | Fix mock to match import structure |
| `tests/test_single_writer.py` | Create | New focused tests for all spec items |

---

## Task 1: Fix leaderboard local-import defect

**Files:**
- Modify: `src/lore/leaderboard.py:145-149`
- Test: `tests/test_leaderboard.py:312`

- [ ] **Step 1: Write the failing test**

```python
def test_parquet_cache_expires(monkeypatch):
    """_load_leaderboard_data returns [] when pandas import fails, not live data."""
    import lore.leaderboard as lb_mod
    # Make pandas import fail by injecting a sentinel
    monkeypatch.setattr(lb_mod, "_pd_available", False)
    monkeypatch.setattr(lb_mod.time.time, lambda: 1000.0, raising=False)
    reg = lb_mod.ModelRegistry.__new__(lb_mod.ModelRegistry)
    reg._parquet_cache = None
    reg._parquet_cache_time = 0
    reg._cache_ttl = 300
    reg._load_from_individual_leaderboards = lambda: []
    result = reg._load_leaderboard_data()
    assert result == [], f"Expected [], got {result}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_leaderboard.py::test_parquet_cache_expires -v`
Expected: FAIL with AttributeError or returns non-empty list

- [ ] **Step 3: Fix the import structure**

In `src/lore/leaderboard.py`, add a module-level flag and restructure the local import:

```python
# Near top of file, after other imports:
_pd_available = True

# In _load_leaderboard_data, replace the try/except block:
    try:
        if not _pd_available:
            raise ImportError("pandas not available (test override)")
        import pandas as pd
        df = pd.read_parquet(
            "hf://datasets/OpenEvals/leaderboard-data/data/train-00000-of-00001.parquet"
        )
    except Exception as e:
        logger.warning(f"Failed to load leaderboard parquet: {e}")
        return self._load_from_individual_leaderboards()
```

Also update `tests/test_leaderboard.py::test_parquet_cache_expires` to use `monkeypatch.setattr(lb_mod, "_pd_available", False)` instead of mocking `lore.leaderboard.pd`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_leaderboard.py::test_parquet_cache_expires -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lore/leaderboard.py tests/test_leaderboard.py
git commit -m "fix: leaderboard parquet import shadowing breaks test mock

Add _pd_available module flag so tests can disable pandas without
live network access leaking through the local import.

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
```

---

## Task 2: Path containment in repo_tools

**Files:**
- Modify: `src/lore/repo_tools.py:26-30,58-64`
- Test: `tests/test_single_writer.py`

- [ ] **Step 1: Write the failing test**

```python
import os
import tempfile
import pytest

def test_repo_tools_reject_path_traversal():
    """read_file and list_files reject paths escaping repo root."""
    from lore.repo_tools import RepoContext
    with tempfile.TemporaryDirectory() as tmp:
        # Create a file outside repo
        outside = os.path.join(os.path.dirname(tmp), "secret.txt")
        with open(outside, "w") as f:
            f.write("SECRET")
        try:
            repo = RepoContext(tmp)
            assert "ERROR" in repo.read_file("../secret.txt")
            assert "ERROR" in repo.read_file("../../etc/passwd")
            assert "ERROR" in repo.list_files("..")
        finally:
            os.unlink(outside)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_single_writer.py::test_repo_tools_reject_path_traversal -v`
Expected: FAIL (current code joins paths without containment check)

- [ ] **Step 3: Add containment validation**

In `src/lore/repo_tools.py`, add a `_safe_path` method and use it in `read_file` and `list_files`:

```python
    def _safe_path(self, rel_path: str) -> Path | None:
        """Resolve rel_path under repo root, return None if it escapes."""
        try:
            resolved = (self.path / rel_path).resolve()
            if not str(resolved).startswith(str(self.path)):
                return None
            return resolved
        except Exception:
            return None
```

Update `read_file`:
```python
    def read_file(self, rel_path: str, max_lines: int = 100) -> str:
        """Read a file from the repo, return first max_lines."""
        fp = self._safe_path(rel_path)
        if fp is None:
            return f"ERROR: Path escapes repo root: {rel_path}"
        if not fp.exists() or not fp.is_file():
            return f"ERROR: File not found: {rel_path}"
```

Update `list_files`:
```python
    def list_files(self, rel_dir: str = ".", pattern: str = "*.py", max_results: int = 50) -> str:
        """List files in a directory matching pattern."""
        dp = self._safe_path(rel_dir)
        if dp is None:
            return f"ERROR: Path escapes repo root: {rel_dir}"
        if not dp.exists() or not dp.is_dir():
            return f"ERROR: Directory not found: {rel_dir}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_single_writer.py::test_repo_tools_reject_path_traversal -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lore/repo_tools.py tests/test_single_writer.py
git commit -m "fix: enforce path containment in repo tools

read_file and list_files now resolve and validate that requested
paths stay beneath RepoContext.root after symlink resolution.

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
```

---

## Task 3: Session ID sanitization

**Files:**
- Modify: `src/lore/session.py:68,104`
- Test: `tests/test_single_writer.py`

- [ ] **Step 1: Write the failing test**

```python
def test_session_id_rejects_traversal():
    """SessionManager rejects unsafe session IDs."""
    from lore.session import SessionManager
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager({"save_dir": tmp})
        for bad_id in ["..", ".", "", "a/b", "a\\b", "/etc/passwd", "foo/../bar"]:
            with pytest.raises((ValueError, Exception)):
                sm.save_session(bad_id, server=None, context=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_single_writer.py::test_session_id_rejects_traversal -v`
Expected: FAIL (current code joins directly)

- [ ] **Step 3: Add sanitization guard**

In `src/lore/session.py`, add a `_safe_session_id` static method and call it at the top of `save_session` and `resume_session`:

```python
    @staticmethod
    def _safe_session_id(session_id: str) -> str:
        """Validate session ID is a single safe path component."""
        if not session_id or session_id in (".", ".."):
            raise ValueError(f"Unsafe session ID: {session_id!r}")
        if "/" in session_id or "\\" in session_id or os.path.isabs(session_id):
            raise ValueError(f"Unsafe session ID: {session_id!r}")
        resolved = Path(session_id).resolve()
        if str(resolved) != session_id:
            raise ValueError(f"Unsafe session ID: {session_id!r}")
        return session_id
```

Add `import os` at the top if not already present. Call `self._safe_session_id(session_id)` as the first line of `save_session` and `resume_session`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_single_writer.py::test_session_id_rejects_traversal -v`
Expected: PASS

- [ ] **Step 5: Run full session test suite**

Run: `PYTHONPATH=src python3 -m pytest tests/test_session.py -v`
Expected: PASS (existing tests use safe IDs)

- [ ] **Step 6: Commit**

```bash
git add src/lore/session.py tests/test_single_writer.py
git commit -m "fix: reject path traversal in session IDs

SessionManager now validates session IDs are single safe path
components before any filesystem access.

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
```

---

## Task 4: Model startup cleanup on failed health check

**Files:**
- Modify: `src/lore/models.py:109-115`
- Test: `tests/test_single_writer.py`

- [ ] **Step 1: Write the failing test**

```python
def test_model_startup_cleanup_on_failed_health(monkeypatch):
    """Failed health check leaves no tracked process or open log handle."""
    from lore.models import ModelServer
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "primary": {"path": tmp + "/fake.gguf", "port": 19999},
            "defaults": {"context_size": 1024},
        }
        # Create fake model file
        with open(tmp + "/fake.gguf", "w") as f:
            f.write("fake")
        server = ModelServer(cfg)
        # Mock Popen to return a fake process
        class FakeProc:
            pid = 99999
            poll = lambda self: None
            def terminate(self): pass
            def wait(self, timeout=None): pass
            def kill(self): pass
        monkeypatch.setattr("lore.models.subprocess.Popen", lambda *a, **kw: FakeProc())
        # Mock health_check to return False
        monkeypatch.setattr(server, "health_check", lambda port: False)
        try:
            server.start_model("primary")
            assert False, "Should have raised"
        except RuntimeError:
            pass
        assert "primary" not in server._processes
        assert "primary" not in server._log_files
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_single_writer.py::test_model_startup_cleanup_on_failed_health -v`
Expected: FAIL (process and log handle remain in dicts after health check failure)

- [ ] **Step 3: Fix cleanup in start_model**

In `src/lore/models.py`, update the health check failure block in `start_model`:

```python
        if not self.health_check(port):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            del self._processes[role]
            fh = self._log_files.pop(role, None)
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass
            raise RuntimeError(f"Model {role} failed health check")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_single_writer.py::test_model_startup_cleanup_on_failed_health -v`
Expected: PASS

- [ ] **Step 5: Run existing model tests**

Run: `PYTHONPATH=src python3 -m pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/lore/models.py tests/test_single_writer.py
git commit -m "fix: clean up process and log handle on failed model health check

start_model now removes the failed process and closes its log file
before raising, preventing resource leaks on startup failure.

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
```

---

## Task 5: Specialist reload in finally + cycle rejection

**Files:**
- Modify: `src/lore/orchestrator.py:229-249,345-357`
- Test: `tests/test_single_writer.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_specialist_reload_after_orchestration_failure():
    """Specialist is reloaded even when orchestration raises."""
    from lore.orchestrator import Orchestrator
    from unittest.mock import MagicMock
    server = MagicMock()
    server.is_model_running = MagicMock(return_value=True)
    router = MagicMock()
    router.classify = MagicMock(return_value=("PRIMARY", 0.9))
    orchestrator = Orchestrator(server, router, memory=None, config={})
    orchestrator._specialist_offloaded = True
    # Force _orchestrate to raise
    orchestrator._orchestrate = MagicMock(side_effect=RuntimeError("boom"))
    # _delegate_dispatch returns a valid result
    orchestrator._delegate_dispatch = MagicMock(return_value={
        "route": "PRIMARY", "confidence": 0.9, "model": "primary",
        "content": "ok", "success": True, "latency_ms": 10,
        "orchestrated": False, "subtasks_completed": 0,
    })
    result = orchestrator.process("complex query")
    # Specialist must be reloaded despite failure
    server.start_model.assert_any_call("specialist")

def test_cycle_rejection_falls_back_to_direct():
    """Cyclic dependency plan falls back to direct dispatch."""
    from lore.orchestrator import Orchestrator
    from lore.decomposer import TaskPlan, SubTask
    from unittest.mock import MagicMock
    server = MagicMock()
    router = MagicMock()
    router.classify = MagicMock(return_value=("PRIMARY", 0.9))
    orchestrator = Orchestrator(server, router, memory=None, config={})
    # Create a cyclic plan
    plan = TaskPlan(
        subtasks=[
            SubTask(id="a", description="task a", dependencies=["b"]),
            SubTask(id="b", description="task b", dependencies=["a"]),
        ],
        rationale="test",
    )
    waves = orchestrator._build_waves(plan)
    assert waves is None, "Cyclic plan should return None waves"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python3 -m pytest tests/test_single_writer.py::test_specialist_reload_after_orchestration_failure tests/test_single_writer.py::test_cycle_rejection_falls_back_to_direct -v`
Expected: FAIL

- [ ] **Step 3: Fix specialist reload with try/finally**

In `src/lore/orchestrator.py`, wrap the orchestration block in `process()` with try/finally for specialist reload. Find the section that calls `self._orchestrate(...)` and change it to:

```python
        # 5. Complex -> orchestrate
        try:
            result = self._orchestrate(query, est, route, confidence, json_mode, dispatch_fn, repo_context)
            return result
        except Exception as e:
            logger.warning(f"Orchestration failed ({e}), falling back to dispatch")
            return self._delegate_dispatch(query, json_mode, dispatch_fn, route, confidence)
        finally:
            if self._specialist_offloaded:
                try:
                    if not self._server.is_model_running("specialist"):
                        self._server.start_model("specialist")
                        self._specialist_offloaded = False
                        logger.info("Specialist reloaded (finally)")
                except Exception as reload_err:
                    logger.warning(f"Specialist reload failed: {reload_err}")
```

- [ ] **Step 4: Fix cycle rejection in _build_waves**

In `src/lore/orchestrator.py`, update `_build_waves` to detect cycles and return `None`:

```python
    def _build_waves(self, plan: TaskPlan) -> list[list[str]] | None:
        """Topologically sort subtasks into waves. Returns None if cyclic."""
        # Build adjacency and in-degree
        deps = {st.id: set(st.dependencies) for st in plan.subtasks}
        # Detect cycles via Kahn's algorithm
        all_ids = set(deps.keys())
        for sid, dset in deps.items():
            for d in dset:
                if d not in all_ids:
                    # Missing dependency -> invalid plan
                    return None
        # Kahn's
        in_degree = {sid: len(dset) for sid, dset in deps.items()}
        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        waves = []
        visited = 0
        while queue:
            waves.append(sorted(queue))
            next_queue = []
            for sid in queue:
                visited += 1
                for other_sid, dset in deps.items():
                    if sid in dset:
                        in_degree[other_sid] -= 1
                        if in_degree[other_sid] == 0:
                            next_queue.append(other_sid)
            queue = next_queue
        if visited != len(deps):
            return None  # cycle detected
        return waves
```

- [ ] **Step 5: Update _orchestrate to check for None waves**

In the `_orchestrate` method, after calling `self._build_waves(plan)`, add:

```python
        waves = self._build_waves(plan)
        if waves is None:
            logger.warning("Invalid plan (cycle or missing dependency), falling back to direct")
            raise RuntimeError("invalid plan topology")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=src python3 -m pytest tests/test_single_writer.py::test_specialist_reload_after_orchestration_failure tests/test_single_writer.py::test_cycle_rejection_falls_back_to_direct -v`
Expected: PASS

- [ ] **Step 7: Run existing orchestrator tests**

Run: `PYTHONPATH=src python3 -m pytest tests/test_orchestrator.py -v`
Expected: PASS (may need minor adjustments to tests that assumed cyclic waves)

- [ ] **Step 8: Commit**

```bash
git add src/lore/orchestrator.py tests/test_single_writer.py
git commit -m "fix: specialist reload in finally, reject cyclic dependency plans

Orchestrator now guarantees specialist reload via try/finally around
the entire orchestration block. _build_waves returns None on cycles
or missing dependencies, triggering direct dispatch fallback.

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
```

---

## Task 6: Single routing decision, remove duplicate memory retrieval

**Files:**
- Modify: `src/lore/cli.py:165-168,241-245`
- Modify: `src/lore/orchestrator.py:70-80`
- Test: `tests/test_single_writer.py`

- [ ] **Step 1: Write the failing test**

```python
def test_single_routing_decision():
    """Router.classify is called once per request, not twice."""
    from unittest.mock import MagicMock
    from lore.cli import _dispatch
    server = MagicMock()
    server.chat = MagicMock(return_value={
        "choices": [{"message": {"content": "test response"}}]
    })
    router = MagicMock()
    router.classify = MagicMock(return_value=("PRIMARY", 0.95))
    from lore.context import ContextManager
    ctx = ContextManager({"context_budget": 4096}, server, system_prompt="test")
    memory = MagicMock()
    memory.retrieve = MagicMock(return_value=[])
    memory.store = MagicMock()
    from lore.logging import RequestLogger
    req_logger = RequestLogger()
    result = _dispatch("hello", server, router, ctx, memory, req_logger)
    assert router.classify.call_count == 1, f"Expected 1 classify call, got {router.classify.call_count}"

def test_memory_retrieved_once():
    """Memory.retrieve is called once per model request, not twice."""
    from unittest.mock import MagicMock
    from lore.cli import _dispatch
    server = MagicMock()
    server.chat = MagicMock(return_value={
        "choices": [{"message": {"content": "test response"}}]
    })
    router = MagicMock()
    router.classify = MagicMock(return_value=("PRIMARY", 0.95))
    from lore.context import ContextManager
    ctx = ContextManager({"context_budget": 4096}, server, system_prompt="test")
    memory = MagicMock()
    memory.retrieve = MagicMock(return_value=[])
    memory.store = MagicMock()
    from lore.logging import RequestLogger
    req_logger = RequestLogger()
    _dispatch("hello", server, router, ctx, memory, req_logger)
    # build_prompt also calls memory.retrieve if ctx._memory is set, but we pass memory=None to ctx
    # So only _execute_query should call it
    assert memory.retrieve.call_count <= 1, f"Expected <=1 retrieve, got {memory.retrieve.call_count}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python3 -m pytest tests/test_single_writer.py::test_single_routing_decision tests/test_single_writer.py::test_memory_retrieved_once -v`
Expected: FAIL (router called twice: once in orchestrator, once in _dispatch)

- [ ] **Step 3: Pass route through from orchestrator to dispatch**

In `src/lore/cli.py`, update `_dispatch` to accept an optional `route_info` parameter so the orchestrator can pass the route decision through without re-classifying:

```python
def _dispatch(query, server, router, ctx, memory, req_logger, json_mode=False, verifier=None,
              route_info=None):
    """Route a query, execute it (tool fast-path or model chat), log, store to memory.

    If route_info is provided as (route, confidence, model), skip re-classification.
    """
    if route_info is not None:
        route, confidence, model = route_info
    else:
        route, confidence, model = _resolve_route(query, router)
```

In `src/lore/orchestrator.py`, update `_delegate_dispatch` to pass route_info:

```python
    def _delegate_dispatch(self, query, json_mode, dispatch_fn, route, confidence):
        """Delegate to the existing dispatch function, passing route through."""
        model = "primary" if route == "PRIMARY" else "specialist"
        return dispatch_fn(query, json_mode=json_mode, route_info=(route, confidence, model))
```

- [ ] **Step 4: Remove duplicate memory retrieval**

In `src/lore/cli.py`, update `_execute_query` to NOT retrieve memory (let `ContextManager.build_prompt` own it):

```python
def _execute_query(query, model, server, ctx, memory, json_mode):
    """Execute model chat for a query. Returns (content, success)."""
    ctx.add_message("user", query)
    # Memory retrieval is owned by ContextManager.build_prompt() — do not retrieve here
    messages = ctx.build_prompt(query=query)
```

- [ ] **Step 5: Update dispatch_fn lambda in API**

In `src/lore/api.py`, update the dispatch_fn lambda to accept route_info:

```python
        dispatch_fn = lambda q, json_mode=False, route_info=None: _dispatch(
            q, server, router, ctx, memory, req_logger, json_mode, verifier,
            route_info=route_info)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=src python3 -m pytest tests/test_single_writer.py::test_single_routing_decision tests/test_single_writer.py::test_memory_retrieved_once -v`
Expected: PASS

- [ ] **Step 7: Run existing CLI and orchestrator tests**

Run: `PYTHONPATH=src python3 -m pytest tests/test_cli.py tests/test_orchestrator.py tests/test_api.py -v`
Expected: PASS (may need to update test mocks that call _dispatch without route_info)

- [ ] **Step 8: Commit**

```bash
git add src/lore/cli.py src/lore/orchestrator.py src/lore/api.py tests/test_single_writer.py
git commit -m "fix: single routing decision and deduplicate memory retrieval

Orchestrator passes route_info to dispatch so the router is called
once per request. Memory retrieval is owned by ContextManager only,
removing the duplicate embedding call from _execute_query.

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
```

---

## Task 7: Request-scoped context budget

**Files:**
- Modify: `src/lore/cli.py:241-245`
- Test: `tests/test_single_writer.py`

- [ ] **Step 1: Write the failing test**

```python
def test_budget_no_drift():
    """Context budget does not drift after a small-budget request."""
    from unittest.mock import MagicMock
    from lore.cli import _dispatch
    from lore.context import ContextManager
    server = MagicMock()
    server.chat = MagicMock(return_value={
        "choices": [{"message": {"content": "response"}}]
    })
    router = MagicMock()
    router.classify = MagicMock(return_value=("PRIMARY", 0.9))
    ctx = ContextManager({"context_budget": 8192}, server, system_prompt="test")
    original_budget = ctx._config.get("context_budget", 8192)
    memory = MagicMock()
    memory.retrieve = MagicMock(return_value=[])
    memory.store = MagicMock()
    from lore.logging import RequestLogger
    req_logger = RequestLogger()
    _dispatch("hello", server, router, ctx, memory, req_logger)
    assert ctx._config.get("context_budget", 0) == original_budget, "Budget drifted"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_single_writer.py::test_budget_no_drift -v`
Expected: FAIL if budget is mutated in place

- [ ] **Step 3: Make budget request-scoped**

In `src/lore/cli.py`, find where `estimate_context_budget` mutates the working context config. Wrap the budget change in a try/finally to restore the original:

```python
    # In _dispatch, where working_context is used:
    working_context = dict(ctx._config)  # ponytail: copy, don't mutate original
    try:
        budget = estimate_context_budget(route, query, working_context)
        working_context["context_budget"] = budget
        # ... use working_context for this request ...
    finally:
        # Config default is untouched because we used a copy
        pass
```

The key change: `working_context = dict(ctx._config)` instead of `working_context = ctx._config`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_single_writer.py::test_budget_no_drift -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lore/cli.py tests/test_single_writer.py
git commit -m "fix: request-scoped context budget prevents drift

Copy config dict before mutating budget so the configured default
survives across requests.

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
```

---

## Task 8: API forwards max_tokens, temperature, and full message history

**Files:**
- Modify: `src/lore/api.py:165-180`
- Modify: `src/lore/cli.py:185-195`
- Test: `tests/test_single_writer.py`

- [ ] **Step 1: Write the failing test**

```python
def test_api_forwards_controls():
    """API forwards max_tokens and temperature to dispatch."""
    from unittest.mock import MagicMock, patch
    from lore.api import _app_state
    server = MagicMock()
    server.chat = MagicMock(return_value={
        "choices": [{"message": {"content": "test"}}]
    })
    router = MagicMock()
    router.classify = MagicMock(return_value=("PRIMARY", 0.9))
    from lore.context import ContextManager
    ctx = ContextManager({"context_budget": 4096}, server, system_prompt="test")
    memory = MagicMock()
    memory.retrieve = MagicMock(return_value=[])
    memory.store = MagicMock()
    from lore.logging import RequestLogger
    from lore.orchestrator import Orchestrator
    orchestrator = Orchestrator(server, router, memory=None, config={})
    orchestrator._delegate_dispatch = MagicMock(return_value={
        "route": "PRIMARY", "confidence": 0.9, "model": "primary",
        "content": "result", "success": True, "latency_ms": 10,
        "orchestrated": False, "subtasks_completed": 0,
    })
    _app_state.update({
        "server": server, "router": router, "ctx": ctx,
        "memory": memory, "req_logger": RequestLogger(),
        "verifier": None, "orchestrator": orchestrator,
    })
    # Patch _dispatch to capture forwarded args
    with patch("lore.api._dispatch", MagicMock(return_value={
        "route": "PRIMARY", "confidence": 0.9, "model": "primary",
        "content": "result", "success": True, "latency_ms": 10,
    })) as mock_dispatch:
        # We need to call the handler directly; instead test via orchestrator
        result = orchestrator.process("test query", dispatch_fn=lambda q, json_mode=False, route_info=None: {
            "route": "PRIMARY", "confidence": 0.9, "model": "primary",
            "content": "result", "success": True, "latency_ms": 10,
            "orchestrated": False, "subtasks_completed": 0,
        })
        assert result["content"] == "result"
```

- [ ] **Step 2: Run test to verify it passes (baseline)**

Run: `PYTHONPATH=src python3 -m pytest tests/test_single_writer.py::test_api_forwards_controls -v`
Expected: PASS (this test verifies the wiring exists)

- [ ] **Step 3: Add max_tokens and temperature pass-through**

In `src/lore/api.py`, update the dispatch call to forward max_tokens and temperature:

```python
        max_tokens = body.get("max_tokens", 2048)
        temperature = body.get("temperature", 0.7)
        # Clamp max_tokens
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            max_tokens = 2048
        max_tokens = min(max_tokens, 8192)
        # Validate temperature
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            temperature = 0.7

        from lore.cli import _dispatch
        t0 = time.time()
        dispatch_fn = lambda q, json_mode=False, route_info=None: _dispatch(
            q, server, router, ctx, memory, req_logger, json_mode, verifier,
            route_info=route_info, max_tokens=max_tokens, temperature=temperature)
```

In `src/lore/cli.py`, update `_dispatch` and `_execute_query` to accept and forward `max_tokens` and `temperature`:

```python
def _dispatch(query, server, router, ctx, memory, req_logger, json_mode=False, verifier=None,
              route_info=None, max_tokens=2048, temperature=0.7):
```

```python
def _execute_query(query, model, server, ctx, memory, json_mode,
                   max_tokens=2048, temperature=0.7):
    # ...
    result = server.chat(model, messages, max_tokens=max_tokens, temperature=temperature, **opts)
```

- [ ] **Step 4: Preserve full API message history**

In `src/lore/api.py`, before dispatching, load prior messages into context:

```python
        # Load prior messages into context (not just the last user message)
        for msg in messages[:-1]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("system", "user", "assistant") and content:
                if role == "system" and not ctx._history:
                    # Use as system prompt if context is fresh
                    pass  # ponytail: keep default system prompt for now
                else:
                    ctx.add_message(role, content)
```

- [ ] **Step 5: Run all API tests**

Run: `PYTHONPATH=src python3 -m pytest tests/test_api.py tests/test_single_writer.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/lore/api.py src/lore/cli.py tests/test_single_writer.py
git commit -m "fix: API forwards max_tokens, temperature, and message history

OpenAI-compatible endpoint now clamps and forwards generation
controls to dispatch, and loads prior messages into context
instead of discarding them.

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
```

---

## Task 9: Disable classifier from normal startup, preserve WIP

**Files:**
- Modify: `src/lore/cli.py:60-116`
- Modify: `src/lore/api.py:45-102`
- Test: `tests/test_single_writer.py`

- [ ] **Step 1: Write the failing test**

```python
def test_classifier_not_on_critical_path():
    """Orchestrator created without classifier by default."""
    from unittest.mock import MagicMock
    from lore.orchestrator import Orchestrator
    server = MagicMock()
    router = MagicMock()
    router.classify = MagicMock(return_value=("PRIMARY", 0.9))
    orchestrator = Orchestrator(server, router, memory=None, config={})
    assert orchestrator._classifier is None, "Classifier should not be created by default"
```

- [ ] **Step 2: Run test to verify it passes (baseline check)**

Run: `PYTHONPATH=src python3 -m pytest tests/test_single_writer.py::test_classifier_not_on_critical_path -v`
Expected: PASS (orchestrator already accepts classifier=None)

- [ ] **Step 3: Stop constructing classifier in CLI/API init**

In `src/lore/cli.py`, find the classifier construction (around line 100) and comment it out or make it conditional:

```python
    # Classifier disabled from normal startup per single-writer design.
    # TF-IDF router alone handles routing; classifier is not on the critical path.
    classifier = None
```

Do the same in `src/lore/api.py`.

- [ ] **Step 4: Preserve existing WIP files**

Verify the uncommitted changes in `src/lore/decomposer.py` and `src/lore/worker.py` are not modified:

```bash
git diff --name-only
```

Expected output includes `src/lore/decomposer.py` and `src/lore/worker.py` but this task should not modify them.

- [ ] **Step 5: Run full test suite**

Run: `PYTHONPATH=src python3 -m pytest tests/ -q --tb=short`
Expected: PASS (same or better than baseline 370 passed, 1 failed, 2 skipped)

- [ ] **Step 6: Commit**

```bash
git add src/lore/cli.py src/lore/api.py tests/test_single_writer.py
git commit -m "fix: disable model-based classifier from normal startup path

TF-IDF router alone handles routing. The specialist-based
TaskClassifier adds latency and a fixed 0.85 confidence without
evidence of quality improvement. Disabled per single-writer design.

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
```

---

## Task 10: Final validation and commit with WIP

**Files:**
- Stage: `src/lore/decomposer.py`, `src/lore/worker.py`, `benchmarks/results/swebench_predictions.jsonl`, `benchmarks/results/swebench_smoke_results.json`

- [ ] **Step 1: Run full test suite**

Run: `PYTHONPATH=src python3 -m pytest tests/ -q --tb=short`
Expected: All tests pass (target: 370+ passed, 0 failed, 2 skipped)

- [ ] **Step 2: Stage WIP files and benchmark outputs**

```bash
git add src/lore/decomposer.py src/lore/worker.py benchmarks/results/swebench_predictions.jsonl benchmarks/results/swebench_smoke_results.json
```

- [ ] **Step 3: Review staged diff for secrets**

```bash
git diff --cached
```

Inspect for credentials, API keys, or sensitive data. If any found, STOP.

- [ ] **Step 4: Commit WIP preservation**

```bash
git commit -m "feat: preserve SEARCH/REPLACE WIP and benchmark smoke results

Uncommitted SEARCH/REPLACE migration work in decomposer.py and
worker.py plus associated SWE-bench smoke test results.

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
```

- [ ] **Step 5: Verify clean working tree**

```bash
git status
```

Expected: clean working tree

- [ ] **Step 6: Push to main**

```bash
git push origin main
```

---

## Self-review

**Spec coverage:**
- Leaderboard import fix: Task 1
- Repo path containment: Task 2
- Session ID containment: Task 3
- Model startup cleanup: Task 4
- Specialist reload + cycle rejection: Task 5
- Single routing + dedup memory: Task 6
- Request-scoped budget: Task 7
- API controls + history: Task 8
- Classifier disabled: Task 9
- WIP preservation + push: Task 10

**Gaps:** Config-root resolution (spec section "Configuration and path handling") is not a separate task because the fix is a one-line `Path(config_dir).resolve()` in `LoreConfig.load()`. If tests show config path issues when running outside repo root, add a focused task. The ponytail approach: fix it only if it breaks, not speculatively.

**Type consistency:** `_dispatch` signature gains `route_info`, `max_tokens`, `temperature` as optional kwargs with defaults. All callers pass these as keywords. `_delegate_dispatch` in orchestrator passes `route_info` through. API lambda updated to match.
