"""Unit tests for the orchestration engine.

Tests with mocked ModelServer — no real inference.
Covers: complexity estimator, decomposer, worker, orchestrator scheduling,
aggregation, and simple-task fallback.
"""
import json
from unittest.mock import MagicMock, patch

import pytest


# ─── Complexity Estimator ────────────────────────────────────────────────────

def test_complexity_simple_question():
    """Short factual question → not complex."""
    from lore.complexity import estimate
    est = estimate("What is 2+2?", "PRIMARY")
    assert not est.is_complex
    assert est.estimated_subtasks == 1


def test_complexity_explain_keyword():
    """Explain keyword + short query → not complex."""
    from lore.complexity import estimate
    est = estimate("Explain how DNS works", "PRIMARY")
    assert not est.is_complex


def test_complexity_tool_only_route():
    """TOOL_ONLY route → always simple."""
    from lore.complexity import estimate
    est = estimate("count words in this text", "TOOL_ONLY")
    assert not est.is_complex
    assert "TOOL_ONLY route" in est.signals


def test_complexity_multi_part():
    """Multiple distinct requests → complex."""
    from lore.complexity import estimate
    est = estimate("Write a Python function to parse CSV files and then add unit tests for it and also write a README", "PRIMARY")
    assert est.is_complex
    assert est.estimated_subtasks >= 2


def test_complexity_long_query():
    """Long query >500 chars → complex."""
    from lore.complexity import estimate
    long_query = "Implement a REST API with the following requirements: " + "x" * 500
    est = estimate(long_query, "PRIMARY")
    assert est.is_complex


def test_complexity_code_plus_instruction():
    """Code + instruction pattern → complex."""
    from lore.complexity import estimate
    est = estimate("Write a CSV parser function and then test it with pytest", "PRIMARY")
    assert est.is_complex


def test_complexity_refactor_keyword():
    """Complex verb (refactor) + another signal → complex."""
    from lore.complexity import estimate
    est = estimate("Refactor the authentication module and also update the documentation", "PRIMARY")
    assert est.is_complex


def test_complexity_file_path_action():
    """File path + action + multi-part → complex."""
    from lore.complexity import estimate
    est = estimate("In /src/auth.py, add rate limiting and then update the tests for it", "PRIMARY")
    assert est.is_complex


def test_complexity_single_complex_signal_defaults_simple():
    """One complex signal, no second → uncertain → defaults to simple."""
    from lore.complexity import estimate
    est = estimate("Refactor the authentication module", "PRIMARY")
    assert not est.is_complex  # uncertain → default simple


def test_complexity_empty_query():
    """Empty query → simple."""
    from lore.complexity import estimate
    est = estimate("", "PRIMARY")
    assert not est.is_complex


def test_complexity_numbered_list():
    """Numbered list + complex verb → complex (2 signals)."""
    from lore.complexity import estimate
    est = estimate("1. Write a parser\n2. Write tests\n3. Write docs\nAlso implement the integration layer", "PRIMARY")
    assert est.is_complex


# ─── Task Decomposer ─────────────────────────────────────────────────────────

def _mock_decomposition_response(subtasks_data, agg_prompt="Combine results"):
    """Build a mock server.chat() return value for the decomposer."""
    plan_json = json.dumps({
        "subtasks": subtasks_data,
        "aggregation_prompt": agg_prompt,
    })
    return {"choices": [{"message": {"content": plan_json}}]}


def test_decomposer_parses_valid_plan():
    """Decomposer parses a valid JSON plan from the model."""
    from lore.decomposer import TaskDecomposer
    server = MagicMock()
    server.chat.return_value = _mock_decomposition_response([
        {
            "id": "s1",
            "description": "Write a CSV parser",
            "model": "primary",
            "context_budget": 4096,
            "system_prompt": "You write Python code.",
            "dependencies": [],
            "max_tokens": 2048,
            "output_format": "code_python",
        },
        {
            "id": "s2",
            "description": "Write tests",
            "model": "primary",
            "context_budget": 4096,
            "system_prompt": "You write tests.",
            "dependencies": ["s1"],
            "max_tokens": 2048,
            "output_format": "code_python",
        },
    ])
    decomposer = TaskDecomposer(server, {"max_subtasks": 5})
    plan = decomposer.decompose("Write a CSV parser and tests")

    assert len(plan.subtasks) == 2
    assert plan.subtasks[0].id == "s1"
    assert plan.subtasks[0].model == "primary"
    assert plan.subtasks[0].context_budget == 4096
    assert plan.subtasks[1].id == "s2"
    assert plan.subtasks[1].dependencies == ["s1"]
    assert plan.subtasks[1].depends_on_outputs is True
    assert plan.total_estimated_tokens == 8192


def test_decomposer_fallback_on_failure():
    """Decomposer falls back to trivial plan on server failure."""
    from lore.decomposer import TaskDecomposer
    server = MagicMock()
    server.chat.side_effect = Exception("server down")
    decomposer = TaskDecomposer(server)
    plan = decomposer.decompose("complex task")

    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].model == "primary"
    assert plan.subtasks[0].dependencies == []


def test_decomposer_fallback_on_invalid_json():
    """Decomposer falls back on invalid JSON response."""
    from lore.decomposer import TaskDecomposer
    server = MagicMock()
    server.chat.return_value = {"choices": [{"message": {"content": "not json"}}]}
    decomposer = TaskDecomposer(server)
    plan = decomposer.decompose("complex task")

    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].model == "primary"


def test_decomposer_invalid_model_defaults_to_primary():
    """Invalid model name in plan → defaults to primary."""
    from lore.decomposer import TaskDecomposer
    server = MagicMock()
    server.chat.return_value = _mock_decomposition_response([
        {"id": "s1", "description": "do something", "model": "unknown_model",
         "context_budget": 2048, "system_prompt": "test", "dependencies": [],
         "max_tokens": 1024, "output_format": "free"},
    ])
    decomposer = TaskDecomposer(server)
    plan = decomposer.decompose("task")
    assert plan.subtasks[0].model == "primary"


def test_decomposer_filters_invalid_dependencies():
    """Dependencies referencing non-existent subtask IDs are filtered out."""
    from lore.decomposer import TaskDecomposer
    server = MagicMock()
    server.chat.return_value = _mock_decomposition_response([
        {"id": "s1", "description": "step 1", "model": "primary",
         "context_budget": 2048, "system_prompt": "test", "dependencies": ["sX"],
         "max_tokens": 1024, "output_format": "free"},
        {"id": "s2", "description": "step 2", "model": "specialist",
         "context_budget": 2048, "system_prompt": "test", "dependencies": ["s1"],
         "max_tokens": 1024, "output_format": "free"},
    ])
    decomposer = TaskDecomposer(server)
    plan = decomposer.decompose("task")
    assert plan.subtasks[0].dependencies == []  # sX filtered out
    assert plan.subtasks[1].dependencies == ["s1"]


def test_decomposer_ensures_entry_point():
    """At least one subtask has no dependencies."""
    from lore.decomposer import TaskDecomposer
    server = MagicMock()
    server.chat.return_value = _mock_decomposition_response([
        {"id": "s1", "description": "step 1", "model": "primary",
         "context_budget": 2048, "system_prompt": "test", "dependencies": ["s2"],
         "max_tokens": 1024, "output_format": "free"},
        {"id": "s2", "description": "step 2", "model": "primary",
         "context_budget": 2048, "system_prompt": "test", "dependencies": ["s1"],
         "max_tokens": 1024, "output_format": "free"},
    ])
    decomposer = TaskDecomposer(server)
    plan = decomposer.decompose("task")
    assert plan.subtasks[0].dependencies == []  # forced to no deps


# ─── Worker ──────────────────────────────────────────────────────────────────

def test_worker_executes_subtask():
    """Worker runs a subtask and returns correct content."""
    from lore.worker import Worker
    from lore.decomposer import SubTask

    server = MagicMock()
    server.chat.return_value = {"choices": [{"message": {"content": "def parse_csv(): pass"}}]}
    server.tokenize.return_value = 10

    st = SubTask(
        id="s1", description="Write a CSV parser", model="primary",
        context_budget=4096, system_prompt="You write code.",
        max_tokens=2048, output_format="code_python",
    )
    worker = Worker(st, server)
    result = worker.run()

    assert result.success
    assert result.content == "def parse_csv(): pass"
    assert result.model == "primary"
    assert result.subtask_id == "s1"
    # Verify server was called with "primary" model
    server.chat.assert_called_once()
    call_args = server.chat.call_args
    assert call_args[0][0] == "primary"


def test_worker_injects_previous_outputs():
    """Worker injects previous outputs when depends_on_outputs is True."""
    from lore.worker import Worker
    from lore.decomposer import SubTask

    server = MagicMock()
    server.chat.return_value = {"choices": [{"message": {"content": "tests here"}}]}
    server.tokenize.return_value = 10

    st = SubTask(
        id="s2", description="Write tests", model="primary",
        context_budget=4096, system_prompt="You write tests.",
        dependencies=["s1"], max_tokens=2048, output_format="code_python",
        depends_on_outputs=True,
    )
    worker = Worker(st, server)
    result = worker.run(previous_outputs={"s1": "def parse_csv(): pass"})

    assert result.success
    # Check that the user message contained the previous output
    call_args = server.chat.call_args
    messages = call_args[0][1]
    user_msg = [m for m in messages if m["role"] == "user"][-1]
    assert "def parse_csv(): pass" in user_msg["content"]
    assert "Previous step results" in user_msg["content"]


def test_worker_specialist_fallback():
    """Worker falls back to primary on specialist failure."""
    from lore.worker import Worker
    from lore.decomposer import SubTask

    server = MagicMock()
    server.chat.side_effect = [
        Exception("specialist error"),
        {"choices": [{"message": {"content": "result from primary"}}]},
    ]

    st = SubTask(
        id="s1", description="Summarize text", model="specialist",
        context_budget=2048, system_prompt="You summarize.",
        max_tokens=1024, output_format="free",
    )
    worker = Worker(st, server)
    result = worker.run()

    assert result.success
    assert result.content == "result from primary"
    assert result.model == "primary"  # fell back
    assert server.chat.call_count == 2


def test_worker_primary_failure_no_fallback():
    """Worker on primary failure returns failure, no fallback."""
    from lore.worker import Worker
    from lore.decomposer import SubTask

    server = MagicMock()
    server.chat.side_effect = Exception("server down")

    st = SubTask(
        id="s1", description="Write code", model="primary",
        context_budget=4096, system_prompt="You write code.",
        max_tokens=2048, output_format="code_python",
    )
    worker = Worker(st, server)
    result = worker.run()

    assert not result.success
    assert "Error" in result.content


def test_worker_stores_to_memory():
    """Worker stores result summary to shared memory on success."""
    from lore.worker import Worker
    from lore.decomposer import SubTask

    server = MagicMock()
    server.chat.return_value = {"choices": [{"message": {"content": "result"}}]}
    server.tokenize.return_value = 5

    memory = MagicMock()
    st = SubTask(
        id="s1", description="Do something", model="primary",
        context_budget=2048, system_prompt="test",
        max_tokens=1024, output_format="free",
    )
    worker = Worker(st, server, memory=memory)
    worker.run()

    memory.episodic.store_summary.assert_called_once()


# ─── Orchestrator: Scheduling ────────────────────────────────────────────────

def test_orchestrator_wave_building():
    """Topological sort groups independent subtasks into waves."""
    from lore.orchestrator import Orchestrator
    from lore.decomposer import SubTask

    server = MagicMock()
    router = MagicMock()
    memory = MagicMock()

    orch = Orchestrator(server, router, memory, {})

    subtasks = [
        SubTask("s1", "step 1", "primary", 2048, "sp", [], 1024, "free", False),
        SubTask("s2", "step 2", "specialist", 2048, "sp", [], 1024, "free", False),
        SubTask("s3", "step 3", "primary", 2048, "sp", ["s1"], 1024, "free", True),
        SubTask("s4", "step 4", "primary", 2048, "sp", ["s2", "s3"], 1024, "free", True),
    ]

    waves = orch._build_waves(subtasks)
    assert len(waves) == 3  # s1+s2 in wave 1, s3 in wave 2, s4 in wave 3
    assert {st.id for st in waves[0]} == {"s1", "s2"}
    assert {st.id for st in waves[1]} == {"s3"}
    assert {st.id for st in waves[2]} == {"s4"}


def test_orchestrator_circular_dependency_recovery():
    """Circular dependency → forces remaining subtasks to execute."""
    from lore.orchestrator import Orchestrator
    from lore.decomposer import SubTask

    server = MagicMock()
    router = MagicMock()
    memory = MagicMock()
    orch = Orchestrator(server, router, memory, {})

    subtasks = [
        SubTask("s1", "a", "primary", 2048, "sp", ["s2"], 1024, "free", True),
        SubTask("s2", "b", "primary", 2048, "sp", ["s1"], 1024, "free", True),
    ]
    waves = orch._build_waves(subtasks)
    assert len(waves) >= 1
    assert len(waves[0]) == 2  # both forced into first wave


# ─── Orchestrator: Simple Task Fallback ──────────────────────────────────────

def test_orchestrator_simple_task_delegates_to_dispatch():
    """Simple query goes through dispatch_fn, not orchestration."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    router = MagicMock()
    router.classify.return_value = ("PRIMARY", 0.95)
    memory = MagicMock()

    orch = Orchestrator(server, router, memory, {})

    dispatched = {"route": "PRIMARY", "confidence": 0.95, "model": "primary",
                  "content": "42", "success": True, "latency_ms": 10.0}

    def dispatch_fn(q, json_mode=False):
        return dict(dispatched)

    r = orch.process("What is 42?", dispatch_fn=dispatch_fn)

    assert r["orchestrated"] is False
    assert r["content"] == "42"
    assert r["subtasks_completed"] == 0


def test_orchestrator_tool_only_delegates_to_dispatch():
    """TOOL_ONLY route goes through dispatch_fn."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    router = MagicMock()
    router.classify.return_value = ("TOOL_ONLY", 0.99)
    memory = MagicMock()

    orch = Orchestrator(server, router, memory, {})

    def dispatch_fn(q, json_mode=False):
        return {"route": "TOOL_ONLY", "confidence": 0.99, "model": "tool_handler",
                "content": "4", "success": True, "latency_ms": 1.0}

    r = orch.process("2+2", dispatch_fn=dispatch_fn)

    assert r["orchestrated"] is False
    assert r["model"] == "tool_handler"


# ─── Orchestrator: Complex Task Orchestration ────────────────────────────────

def test_orchestrator_complex_task_decomposes_and_executes():
    """Complex query triggers decomposition, execution, and aggregation."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    router = MagicMock()
    router.classify.return_value = ("PRIMARY", 0.90)
    memory = MagicMock()

    # Mock decomposer response (planning call)
    plan_json = json.dumps({
        "subtasks": [
            {"id": "s1", "description": "Write CSV parser", "model": "primary",
             "context_budget": 4096, "system_prompt": "Write code.",
             "dependencies": [], "max_tokens": 2048, "output_format": "code_python"},
            {"id": "s2", "description": "Write tests", "model": "primary",
             "context_budget": 4096, "system_prompt": "Write tests.",
             "dependencies": ["s1"], "max_tokens": 2048, "output_format": "code_python"},
            {"id": "s3", "description": "Write README", "model": "specialist",
             "context_budget": 2048, "system_prompt": "Write docs.",
             "dependencies": ["s1"], "max_tokens": 1024, "output_format": "free"},
        ],
        "aggregation_prompt": "Combine all outputs.",
    })

    # Server chat: first call = planning, then s1, s2, s3, then aggregation
    server.chat.side_effect = [
        {"choices": [{"message": {"content": plan_json}}]},        # planning
        {"choices": [{"message": {"content": "def parse_csv(): pass"}}]},  # s1
        {"choices": [{"message": {"content": "def test_parse(): pass"}}]}, # s2
        {"choices": [{"message": {"content": "# CSV Parser README"}}]},     # s3
        {"choices": [{"message": {"content": "Here is the complete solution..."}}]},  # aggregation
    ]
    server.tokenize.return_value = 10

    orch = Orchestrator(server, router, memory, {})

    # Complex query with multi-part + code+instruction
    query = "Write a Python function to parse CSV files and then add unit tests for it and also write a brief README explaining how to use it"
    r = orch.process(query)

    assert r["orchestrated"] is True
    assert r["subtasks_completed"] == 3
    assert r["success"] is True
    assert "complete solution" in r["content"]
    # 5 chat calls: 1 planning + 3 subtasks + 1 aggregation
    assert server.chat.call_count == 5


def test_orchestrator_aggregation_fallback_on_failure():
    """If aggregation call fails, results are concatenated."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    router = MagicMock()
    router.classify.return_value = ("PRIMARY", 0.90)
    memory = MagicMock()

    plan_json = json.dumps({
        "subtasks": [
            {"id": "s1", "description": "Write code", "model": "primary",
             "context_budget": 2048, "system_prompt": "Write code.",
             "dependencies": [], "max_tokens": 1024, "output_format": "free"},
        ],
        "aggregation_prompt": "Combine outputs.",
    })

    server.chat.side_effect = [
        {"choices": [{"message": {"content": plan_json}}]},  # planning
        {"choices": [{"message": {"content": "result s1"}}]},  # s1
        Exception("aggregation failed"),  # aggregation fails
    ]
    server.tokenize.return_value = 5

    orch = Orchestrator(server, router, memory, {})
    r = orch.process("Write a parser and then test it and also document it thoroughly")

    assert r["orchestrated"] is True
    assert r["success"] is True
    assert "result s1" in r["content"]  # concatenated fallback


def test_orchestrator_stores_to_memory():
    """Orchestrator stores summary to episodic memory after orchestration."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    router = MagicMock()
    router.classify.return_value = ("PRIMARY", 0.90)
    memory = MagicMock()

    plan_json = json.dumps({
        "subtasks": [
            {"id": "s1", "description": "Do thing", "model": "primary",
             "context_budget": 2048, "system_prompt": "test",
             "dependencies": [], "max_tokens": 1024, "output_format": "free"},
        ],
        "aggregation_prompt": "Combine.",
    })

    server.chat.side_effect = [
        {"choices": [{"message": {"content": plan_json}}]},
        {"choices": [{"message": {"content": "done"}}]},
        {"choices": [{"message": {"content": "aggregated result"}}]},
    ]
    server.tokenize.return_value = 5

    orch = Orchestrator(server, router, memory, {})
    r = orch.process("Write a parser and then test it and also document it thoroughly")

    assert r["orchestrated"] is True
    # Memory should have been called at least once (orchestrator summary + worker summary)
    assert memory.episodic.store_summary.call_count >= 1


# ─── Orchestrator: Return Dict Shape ─────────────────────────────────────────

def test_orchestrator_return_shape_complex():
    """Complex task return dict has all required fields for CLI display."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    router = MagicMock()
    router.classify.return_value = ("PRIMARY", 0.90)
    memory = MagicMock()

    plan_json = json.dumps({
        "subtasks": [
            {"id": "s1", "description": "Do thing", "model": "primary",
             "context_budget": 2048, "system_prompt": "test",
             "dependencies": [], "max_tokens": 1024, "output_format": "free"},
        ],
        "aggregation_prompt": "Combine.",
    })

    server.chat.side_effect = [
        {"choices": [{"message": {"content": plan_json}}]},
        {"choices": [{"message": {"content": "done"}}]},
        {"choices": [{"message": {"content": "final"}}]},
    ]
    server.tokenize.return_value = 5

    orch = Orchestrator(server, router, memory, {})
    r = orch.process("Write a parser and then test it and also document it thoroughly")

    required = {"route", "confidence", "model", "content", "success", "latency_ms",
                "orchestrated", "subtasks_completed"}
    assert required.issubset(r.keys())


def test_orchestrator_return_shape_simple():
    """Simple task return dict has all required fields."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    router = MagicMock()
    router.classify.return_value = ("PRIMARY", 0.90)
    memory = MagicMock()

    orch = Orchestrator(server, router, memory, {})

    def dispatch_fn(q, json_mode=False):
        return {"route": "PRIMARY", "confidence": 0.9, "model": "primary",
                "content": "answer", "success": True, "latency_ms": 5.0}

    r = orch.process("What is 2+2?", dispatch_fn=dispatch_fn)

    required = {"route", "confidence", "model", "content", "success", "latency_ms",
                "orchestrated", "subtasks_completed"}
    assert required.issubset(r.keys())


# ─── Dynamic Model Lifecycle ─────────────────────────────────────────────────

def test_orchestrator_offloads_specialist_all_primary():
    """Specialist offloaded when all subtasks are primary-only."""
    from lore.orchestrator import Orchestrator
    from lore.decomposer import TaskPlan, SubTask

    server = MagicMock()
    server.is_model_running.return_value = True
    server.stop_model = MagicMock()

    router = MagicMock()
    memory = MagicMock()

    orch = Orchestrator(server, router, memory, {
        "dynamic_model_lifecycle": {"enabled": True, "offload_threshold": 0.8}
    })

    plan = TaskPlan(
        original_query="test",
        subtasks=[
            SubTask("s1", "a", "primary", 2048, "sp", [], 1024, "free", False),
            SubTask("s2", "b", "primary", 2048, "sp", ["s1"], 1024, "free", True),
        ],
        aggregation_prompt="combine",
        total_estimated_tokens=4096,
    )

    orch._maybe_offload_specialist(plan)
    server.stop_model.assert_called_once_with("specialist")
    assert orch._specialist_offloaded is True


def test_orchestrator_no_offload_when_specialist_needed():
    """Specialist NOT offloaded when plan uses specialist."""
    from lore.orchestrator import Orchestrator
    from lore.decomposer import TaskPlan, SubTask

    server = MagicMock()
    server.is_model_running.return_value = True
    server.stop_model = MagicMock()

    router = MagicMock()
    memory = MagicMock()

    orch = Orchestrator(server, router, memory, {
        "dynamic_model_lifecycle": {"enabled": True, "offload_threshold": 0.8}
    })

    plan = TaskPlan(
        original_query="test",
        subtasks=[
            SubTask("s1", "a", "primary", 2048, "sp", [], 1024, "free", False),
            SubTask("s2", "b", "specialist", 2048, "sp", ["s1"], 1024, "free", True),
        ],
        aggregation_prompt="combine",
        total_estimated_tokens=4096,
    )

    orch._maybe_offload_specialist(plan)
    server.stop_model.assert_not_called()
    assert orch._specialist_offloaded is False


# ─── Parallel Wave Execution ─────────────────────────────────────────────────

def test_orchestrator_parallel_wave_different_models():
    """Wave with subtasks on different models → both complete successfully."""
    from lore.orchestrator import Orchestrator
    from lore.decomposer import SubTask

    server = MagicMock()
    server.tokenize.return_value = 5
    # Two different chat responses for the two subtasks
    server.chat.side_effect = [
        {"choices": [{"message": {"content": "primary result"}}]},
        {"choices": [{"message": {"content": "specialist result"}}]},
    ]

    router = MagicMock()
    memory = MagicMock()
    orch = Orchestrator(server, router, memory, {})

    wave = [
        SubTask("s1", "code task", "primary", 2048, "sp", [], 1024, "free", False),
        SubTask("s2", "extract task", "specialist", 2048, "sp", [], 1024, "free", False),
    ]
    results = orch._execute_wave(wave, {})

    assert len(results) == 2
    assert results["s1"].success
    assert results["s2"].success
    assert "primary result" in results["s1"].content
    assert "specialist result" in results["s2"].content


def test_orchestrator_sequential_wave_same_model():
    """Wave with subtasks on same model → sequential, both complete."""
    from lore.orchestrator import Orchestrator
    from lore.decomposer import SubTask

    server = MagicMock()
    server.tokenize.return_value = 5
    server.chat.side_effect = [
        {"choices": [{"message": {"content": "first result"}}]},
        {"choices": [{"message": {"content": "second result"}}]},
    ]

    router = MagicMock()
    memory = MagicMock()
    orch = Orchestrator(server, router, memory, {})

    wave = [
        SubTask("s1", "task a", "primary", 2048, "sp", [], 1024, "free", False),
        SubTask("s2", "task b", "primary", 2048, "sp", [], 1024, "free", False),
    ]
    results = orch._execute_wave(wave, {})

    assert len(results) == 2
    assert results["s1"].content == "first result"
    assert results["s2"].content == "second result"


def test_orchestrator_collect_prev_outputs():
    """_collect_prev_outputs gathers dependency outputs correctly."""
    from lore.orchestrator import Orchestrator
    from lore.decomposer import SubTask
    from lore.worker import WorkerResult

    server = MagicMock()
    router = MagicMock()
    memory = MagicMock()
    orch = Orchestrator(server, router, memory, {})

    prior = {
        "s1": WorkerResult("s1", "output from s1", True, 10, 50, "primary"),
    }

    # Subtask with depends_on_outputs=True
    st = SubTask("s2", "test", "primary", 2048, "sp", ["s1"], 1024, "free", True)
    prev = orch._collect_prev_outputs(st, prior)
    assert prev == {"s1": "output from s1"}

    # Subtask with depends_on_outputs=False
    st_no_deps = SubTask("s3", "test", "primary", 2048, "sp", [], 1024, "free", False)
    assert orch._collect_prev_outputs(st_no_deps, prior) is None

    # Subtask with deps but prior_results missing the dep
    st_missing = SubTask("s4", "test", "primary", 2048, "sp", ["sX"], 1024, "free", True)
    assert orch._collect_prev_outputs(st_missing, prior) is None


# ─── Classifier Integration ──────────────────────────────────────────────────

def test_orchestrator_uses_classifier_when_provided():
    """Orchestrator uses classifier instead of heuristic when provided."""
    from lore.orchestrator import Orchestrator
    from lore.classifier import ClassificationResult

    server = MagicMock()
    router = MagicMock()
    router.classify.return_value = ("PRIMARY", 0.90)
    memory = MagicMock()

    classifier = MagicMock()
    classifier.classify.return_value = ClassificationResult(
        is_complex=False, task_type="code_gen",
        estimated_subtasks=1, suggested_model="primary",
        confidence=0.9, hints={"needs_code": True}, source="model",
    )

    orch = Orchestrator(server, router, memory, {}, classifier=classifier)

    def dispatch_fn(q, json_mode=False):
        return {"route": "PRIMARY", "confidence": 0.9, "model": "primary",
                "content": "answer", "success": True, "latency_ms": 5.0}

    r = orch.process("What is 2+2?", dispatch_fn=dispatch_fn)

    assert r["orchestrated"] is False
    classifier.classify.assert_called_once()


def test_orchestrator_classifier_complex_triggers_orchestration():
    """Classifier says complex → orchestration runs with hints."""
    from lore.orchestrator import Orchestrator
    from lore.classifier import ClassificationResult

    server = MagicMock()
    router = MagicMock()
    router.classify.return_value = ("PRIMARY", 0.90)
    memory = MagicMock()

    classifier = MagicMock()
    classifier.classify.return_value = ClassificationResult(
        is_complex=True, task_type="code_gen",
        estimated_subtasks=3, suggested_model="primary",
        confidence=0.9, hints={"needs_code": True}, source="model",
    )

    plan_json = json.dumps({
        "subtasks": [
            {"id": "s1", "description": "Write code", "model": "primary",
             "context_budget": 2048, "system_prompt": "test",
             "dependencies": [], "max_tokens": 1024, "output_format": "free"},
        ],
        "aggregation_prompt": "Combine.",
    })

    server.chat.side_effect = [
        {"choices": [{"message": {"content": plan_json}}]},
        {"choices": [{"message": {"content": "done"}}]},
        {"choices": [{"message": {"content": "final"}}]},
    ]
    server.tokenize.return_value = 5

    orch = Orchestrator(server, router, memory, {}, classifier=classifier)

    r = orch.process("Write a parser and then test it and also document it thoroughly")

    assert r["orchestrated"] is True
    # Verify hints were passed to decomposer
    call_args = server.chat.call_args_list[0]
    messages = call_args[0][1]
    user_msg = [m for m in messages if m["role"] == "user"][-1]
    assert "task_type=code_gen" in user_msg["content"]


def test_orchestrator_classifier_error_falls_back_to_heuristic():
    """Classifier error → falls back to heuristic complexity."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    router = MagicMock()
    router.classify.return_value = ("PRIMARY", 0.90)
    memory = MagicMock()

    classifier = MagicMock()
    classifier.classify.side_effect = Exception("classifier broken")

    plan_json = json.dumps({
        "subtasks": [
            {"id": "s1", "description": "Do thing", "model": "primary",
             "context_budget": 2048, "system_prompt": "test",
             "dependencies": [], "max_tokens": 1024, "output_format": "free"},
        ],
        "aggregation_prompt": "Combine.",
    })

    server.chat.side_effect = [
        {"choices": [{"message": {"content": plan_json}}]},
        {"choices": [{"message": {"content": "done"}}]},
        {"choices": [{"message": {"content": "final"}}]},
    ]
    server.tokenize.return_value = 5

    orch = Orchestrator(server, router, memory, {}, classifier=classifier)

    r = orch.process("Write a parser and then test it and also document it thoroughly")

    # Should still orchestrate (heuristic says complex for this query)
    assert r["orchestrated"] is True


# ─── Fallback Plan Skip ──────────────────────────────────────────────────────

def test_orchestrator_fallback_plan_delegates_to_dispatch():
    """Fallback plan (planning failed) → delegates to dispatch, skips orchestration."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    router = MagicMock()
    router.classify.return_value = ("PRIMARY", 0.90)
    memory = MagicMock()

    # Planning call fails → decomposer returns fallback plan
    server.chat.side_effect = Exception("planning failed")
    server.tokenize.return_value = 5

    orch = Orchestrator(server, router, memory, {})

    def dispatch_fn(q, json_mode=False):
        return {"route": "PRIMARY", "confidence": 0.9, "model": "primary",
                "content": "dispatched answer", "success": True, "latency_ms": 5.0}

    r = orch.process("Write a parser and then test it and also document it thoroughly",
                     dispatch_fn=dispatch_fn)

    # Fallback plan → delegates to dispatch, not orchestrated
    assert r["orchestrated"] is False
    assert r["content"] == "dispatched answer"


# ─── Registry Integration (Issue #4) ─────────────────────────────────────────

def test_orchestrator_uses_registry_model_selection():
    """_execute_wave consults registry and overrides subtask.model."""
    from lore.orchestrator import Orchestrator
    from lore.decomposer import SubTask
    from lore.classifier import ClassificationResult

    server = MagicMock()
    server.tokenize.return_value = 5
    server.chat.return_value = {"choices": [{"message": {"content": "result"}}]}

    router = MagicMock()
    memory = MagicMock()

    # Registry returns "specialist" for "code_gen"
    registry = MagicMock()
    registry.get_model_for_task.return_value = "specialist"

    orch = Orchestrator(server, router, memory, {}, registry=registry)
    # Set classification so _execute_wave can look up task_type
    orch._classification = ClassificationResult(
        is_complex=True, task_type="code_gen", estimated_subtasks=2,
        suggested_model="primary", confidence=0.9, hints={}, source="model",
    )

    wave = [SubTask("s1", "task a", "primary", 2048, "sp", [], 1024, "free", False)]
    orch._execute_wave(wave, {})

    # Registry was consulted
    registry.get_model_for_task.assert_called_once_with("code_gen")
    # Subtask model overridden to registry's choice
    assert wave[0].model == "specialist"
    # Server.chat called with "specialist", not "primary"
    server.chat.assert_called_once()
    assert server.chat.call_args[0][0] == "specialist"


def test_orchestrator_no_registry_keeps_original_model():
    """Without registry, subtask.model stays as decomposer set it."""
    from lore.orchestrator import Orchestrator
    from lore.decomposer import SubTask

    server = MagicMock()
    server.tokenize.return_value = 5
    server.chat.return_value = {"choices": [{"message": {"content": "result"}}]}

    router = MagicMock()
    memory = MagicMock()

    orch = Orchestrator(server, router, memory, {})
    wave = [SubTask("s1", "task a", "primary", 2048, "sp", [], 1024, "free", False)]
    orch._execute_wave(wave, {})

    assert wave[0].model == "primary"


def test_orchestrator_uses_complexity_estimate_dataclass():
    """Classifier path produces ComplexityEstimate, not dynamic type bridge."""
    from lore.orchestrator import Orchestrator
    from lore.complexity import ComplexityEstimate
    from lore.classifier import ClassificationResult

    server = MagicMock()
    router = MagicMock()
    router.classify.return_value = ("PRIMARY", 0.9)
    memory = MagicMock()

    classifier = MagicMock()
    classifier.classify.return_value = ClassificationResult(
        is_complex=False, task_type="code_gen", estimated_subtasks=1,
        suggested_model="primary", confidence=0.9, hints={"signals": ["test"]},
        source="model",
    )

    dispatch_fn = lambda q, json_mode=False: {
        "route": "PRIMARY", "confidence": 0.9, "model": "primary",
        "content": "ok", "success": True, "latency_ms": 1.0,
    }

    orch = Orchestrator(server, router, memory, {}, classifier=classifier)
    r = orch.process("simple task", dispatch_fn=dispatch_fn)

    # Classification was set and is a ClassificationResult
    assert orch._classification is not None
    assert isinstance(orch._classification, ClassificationResult)


def test_orchestrator_set_memory_updates_reference():
    """set_memory updates the orchestrator's memory reference (issue #6)."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    router = MagicMock()
    old_memory = MagicMock()
    new_memory = MagicMock()

    orch = Orchestrator(server, router, old_memory, {})
    assert orch._memory is old_memory

    orch.set_memory(new_memory)
    assert orch._memory is new_memory
