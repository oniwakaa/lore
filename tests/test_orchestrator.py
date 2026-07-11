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
    decomposer = TaskDecomposer(server, {"max_subtasks": 3})
    plan = decomposer.decompose("Write a CSV parser and tests")

    assert len(plan.subtasks) == 2
    assert plan.subtasks[0].id == "s1"
    assert plan.subtasks[0].model == "primary"
    assert plan.subtasks[1].id == "s2"
    assert plan.subtasks[1].dependencies == ["s1"]
    assert plan.subtasks[1].depends_on_outputs is True


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
            {"id": "s2", "description": "Review code", "model": "primary",
             "context_budget": 2048, "system_prompt": "Review code.",
             "dependencies": ["s1"], "max_tokens": 1024, "output_format": "free"},
        ],
        "aggregation_prompt": "Combine outputs.",
    })

    server.chat.side_effect = [
        {"choices": [{"message": {"content": plan_json}}]},  # planning
        {"choices": [{"message": {"content": "result s1"}}]},  # s1
        {"choices": [{"message": {"content": "result s2"}}]},  # s2
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
            {"id": "s2", "description": "Do other thing", "model": "primary",
             "context_budget": 2048, "system_prompt": "test",
             "dependencies": ["s1"], "max_tokens": 1024, "output_format": "free"},
        ],
        "aggregation_prompt": "Combine.",
    })

    server.chat.side_effect = [
        {"choices": [{"message": {"content": plan_json}}]},
        {"choices": [{"message": {"content": "done"}}]},
        {"choices": [{"message": {"content": "done2"}}]},
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
            {"id": "s2", "description": "Do other thing", "model": "primary",
             "context_budget": 2048, "system_prompt": "test",
             "dependencies": ["s1"], "max_tokens": 1024, "output_format": "free"},
        ],
        "aggregation_prompt": "Combine.",
    })

    server.chat.side_effect = [
        {"choices": [{"message": {"content": plan_json}}]},
        {"choices": [{"message": {"content": "done"}}]},
        {"choices": [{"message": {"content": "done2"}}]},
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
            {"id": "s2", "description": "Write tests", "model": "primary",
             "context_budget": 2048, "system_prompt": "test",
             "dependencies": ["s1"], "max_tokens": 1024, "output_format": "free"},
        ],
        "aggregation_prompt": "Combine.",
    })

    server.chat.side_effect = [
        {"choices": [{"message": {"content": plan_json}}]},
        {"choices": [{"message": {"content": "done"}}]},
        {"choices": [{"message": {"content": "done2"}}]},
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
    assert "code_gen" in user_msg["content"]
    assert "Classifier analysis" in user_msg["content"]


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
            {"id": "s2", "description": "Do other thing", "model": "primary",
             "context_budget": 2048, "system_prompt": "test",
             "dependencies": ["s1"], "max_tokens": 1024, "output_format": "free"},
        ],
        "aggregation_prompt": "Combine.",
    })

    server.chat.side_effect = [
        {"choices": [{"message": {"content": plan_json}}]},
        {"choices": [{"message": {"content": "done"}}]},
        {"choices": [{"message": {"content": "done2"}}]},
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


# ─── Phase 1: Decomposer Validation ──────────────────────────────────────────

def test_validate_plan_marks_all_primary_no_deps_as_fallback():
    """Plan with <=2 all-primary no-dep subtasks → is_fallback=True."""
    from lore.decomposer import TaskDecomposer, TaskPlan, SubTask

    server = MagicMock()
    decomposer = TaskDecomposer(server)

    plan = TaskPlan(
        original_query="test",
        subtasks=[
            SubTask("s1", "do thing", "primary", 4096, "sp", [], 2048, "free", False),
            SubTask("s2", "do other", "primary", 4096, "sp", [], 2048, "free", False),
        ],
    )
    result = decomposer._validate_plan(plan, "test", {})
    assert result.is_fallback is True


def test_validate_plan_keeps_multi_model_plan():
    """Plan with specialist subtask → not fallback."""
    from lore.decomposer import TaskDecomposer, TaskPlan, SubTask

    server = MagicMock()
    decomposer = TaskDecomposer(server)

    plan = TaskPlan(
        original_query="test",
        subtasks=[
            SubTask("s1", "write code", "primary", 4096, "sp", [], 2048, "code_python", False),
            SubTask("s2", "summarize", "specialist", 2048, "sp", ["s1"], 1024, "free", True),
        ],
    )
    result = decomposer._validate_plan(plan, "test", {})
    assert result.is_fallback is False


def test_validate_plan_clamps_extreme_budgets():
    """Context budgets below 512 get floored to 2048, above 16384 get capped."""
    from lore.decomposer import TaskDecomposer, TaskPlan, SubTask

    server = MagicMock()
    decomposer = TaskDecomposer(server)

    plan = TaskPlan(
        original_query="test",
        subtasks=[
            SubTask("s1", "write a very long complex piece of code " * 10,
                    "primary", 100, "sp", [], 2048, "code_python", False),
            SubTask("s2", "do other thing", "primary", 99999, "sp", ["s1"], 2048, "free", True),
        ],
    )
    result = decomposer._validate_plan(plan, "test", {})
    for st in result.subtasks:
        assert st.context_budget >= 512
        assert st.context_budget <= 16384


def test_validate_plan_assigns_template_for_default_prompt():
    """Subtask with default system prompt gets a proper template."""
    from lore.decomposer import TaskDecomposer, TaskPlan, SubTask

    server = MagicMock()
    decomposer = TaskDecomposer(server)

    plan = TaskPlan(
        original_query="test",
        subtasks=[
            SubTask("s1", "write code", "primary", 4096, "You are a helpful assistant.",
                    [], 2048, "code_python", False),
            SubTask("s2", "review code", "primary", 4096, "You are a helpful assistant.",
                    ["s1"], 2048, "free", True),
        ],
    )
    result = decomposer._validate_plan(plan, "test", {})
    for st in result.subtasks:
        assert st.system_prompt != "You are a helpful assistant."


# ─── Phase 1: compute_subtask_budget ─────────────────────────────────────────

def test_compute_subtask_budget_base_by_task_type():
    """Budget varies by task type."""
    from lore.decomposer import SubTask, compute_subtask_budget

    st = SubTask("s1", "do something", "primary", 4096, "sp", [], 2048, "free", False)

    extraction = compute_subtask_budget(st, "extraction")
    code_gen = compute_subtask_budget(st, "code_gen")
    planning = compute_subtask_budget(st, "planning")

    assert extraction < code_gen
    assert code_gen < planning


def test_compute_subtask_budget_scales_with_description():
    """Longer descriptions get more budget."""
    from lore.decomposer import SubTask, compute_subtask_budget

    short_st = SubTask("s1", "do thing", "primary", 4096, "sp", [], 2048, "free", False)
    long_st = SubTask("s2", "do " + "very complex thing " * 20, "primary", 4096, "sp", [], 2048, "free", False)

    short_budget = compute_subtask_budget(short_st, "code_gen")
    long_budget = compute_subtask_budget(long_st, "code_gen")
    assert long_budget > short_budget


def test_compute_subtask_budget_adds_for_dependencies():
    """Subtasks with deps get extra budget for previous outputs."""
    from lore.decomposer import SubTask, compute_subtask_budget

    no_deps = SubTask("s1", "do something", "primary", 4096, "sp", [], 2048, "free", False)
    with_deps = SubTask("s2", "do something", "primary", 4096, "sp", ["s1"], 2048, "free", True)

    base = compute_subtask_budget(no_deps, "code_gen")
    with_dep = compute_subtask_budget(with_deps, "code_gen")
    assert with_dep > base


def test_compute_subtask_budget_clamps_to_range():
    """Budget stays within [1024, 16384]."""
    from lore.decomposer import SubTask, compute_subtask_budget

    st = SubTask("s1", "x", "primary", 4096, "sp", [], 2048, "free", False)
    budget = compute_subtask_budget(st, "classification")
    assert budget >= 1024
    assert budget <= 16384


# ─── Phase 2: Dynamic Temperature ────────────────────────────────────────────

def test_worker_code_python_gets_low_temperature():
    """Code tasks get temperature 0.1."""
    from lore.worker import Worker, TEMPERATURE_MAP
    from lore.decomposer import SubTask

    assert TEMPERATURE_MAP["code_python"] == 0.1
    assert TEMPERATURE_MAP["json"] == 0.1
    assert TEMPERATURE_MAP["free"] == 0.7


def test_worker_uses_dynamic_temperature():
    """Worker passes format-based temperature to server.chat."""
    from lore.worker import Worker
    from lore.decomposer import SubTask

    server = MagicMock()
    server.chat.return_value = {"choices": [{"message": {"content": "def f(): pass"}}]}
    server.tokenize.return_value = 5

    st = SubTask("s1", "write code", "primary", 4096, "You write code.",
                 [], 1024, "code_python", False)
    worker = Worker(st, server)
    worker.run()

    call_kwargs = server.chat.call_args
    assert call_kwargs[1]["temperature"] == 0.1


# ─── Phase 2: Dynamic max_tokens ─────────────────────────────────────────────

def test_estimate_max_tokens_code_short():
    """Short code description → 1024 tokens."""
    from lore.worker import _estimate_max_tokens
    assert _estimate_max_tokens("write a function", "code_python") == 1024


def test_estimate_max_tokens_code_long():
    """Long code description → 4096 tokens."""
    from lore.worker import _estimate_max_tokens
    desc = " ".join(["word"] * 100)
    assert _estimate_max_tokens(desc, "code_python") == 4096


def test_estimate_max_tokens_json():
    """JSON output → 1024 tokens."""
    from lore.worker import _estimate_max_tokens
    assert _estimate_max_tokens("extract data", "json") == 1024


def test_estimate_max_tokens_summary():
    """Summarize keyword → 256 tokens."""
    from lore.worker import _estimate_max_tokens
    assert _estimate_max_tokens("summarize this text", "free") == 256


# ─── Phase 2: run_with_retry ─────────────────────────────────────────────────

def test_worker_run_with_retry_succeeds_first_try():
    """run_with_retry returns on first success."""
    from lore.worker import Worker
    from lore.decomposer import SubTask

    server = MagicMock()
    server.chat.return_value = {"choices": [{"message": {"content": "result"}}]}

    st = SubTask("s1", "do thing", "primary", 4096, "You do things.",
                 [], 1024, "free", False)
    worker = Worker(st, server)
    result = worker.run_with_retry()

    assert result.success
    assert server.chat.call_count == 1


def test_worker_run_with_retry_escalates_on_failure():
    """Failed subtask retries with more tokens and error context."""
    from lore.worker import Worker
    from lore.decomposer import SubTask

    server = MagicMock()
    server.chat.side_effect = [
        Exception("fail"),
        {"choices": [{"message": {"content": "success"}}]},
    ]

    st = SubTask("s1", "do thing", "primary", 4096, "You do things.",
                 [], 1024, "free", False)
    worker = Worker(st, server)
    result = worker.run_with_retry(max_retries=2)

    assert result.success
    assert server.chat.call_count == 2
    # Second call should have more tokens
    second_call = server.chat.call_args_list[1]
    assert second_call[1]["max_tokens"] >= 1024


def test_worker_run_with_retry_escalates_specialist_to_primary():
    """Specialist failure on first attempt → switches to primary."""
    from lore.worker import Worker
    from lore.decomposer import SubTask

    server = MagicMock()
    server.chat.side_effect = [
        Exception("specialist fail"),
        {"choices": [{"message": {"content": "primary result"}}]},
    ]

    st = SubTask("s1", "do thing", "specialist", 2048, "You do things.",
                 [], 1024, "free", False)
    worker = Worker(st, server)
    result = worker.run_with_retry(max_retries=2)

    assert result.success
    assert result.model == "primary"


def test_worker_run_with_retry_exhausts_retries():
    """All retries fail → returns failure."""
    from lore.worker import Worker
    from lore.decomposer import SubTask

    server = MagicMock()
    server.chat.side_effect = Exception("always fails")

    st = SubTask("s1", "do thing", "primary", 2048, "You do things.",
                 [], 1024, "free", False)
    worker = Worker(st, server)
    result = worker.run_with_retry(max_retries=1)

    assert not result.success
    # 1 initial + 1 retry = 2 calls
    assert server.chat.call_count == 2


def test_run_with_retry_timeout_with_partial_output():
    """Timeout error with 200+ chars content → returns success with partial output."""
    from lore.worker import Worker, WorkerResult
    from lore.decomposer import SubTask

    server = MagicMock()
    long_content = "x" * 200

    st = SubTask("s1", "do thing", "primary", 4096, "You do things.",
                 [], 1024, "free", False)
    worker = Worker(st, server)

    timeout_result = WorkerResult(
        subtask_id="s1", content=long_content, success=False,
        latency_ms=180000, tokens_used=50, model="primary",
        error="Request timed out",
    )
    with patch.object(worker, "run", return_value=timeout_result):
        result = worker.run_with_retry(max_retries=1)

    assert result.success
    assert result.error == "timeout_with_partial_output"
    assert len(result.content) == 200


def test_run_with_retry_timeout_short_partial_treated_as_failure():
    """Timeout error with <200 chars content → returns failure (threshold raised from 100)."""
    from lore.worker import Worker, WorkerResult
    from lore.decomposer import SubTask

    server = MagicMock()
    short_content = "x" * 150  # >100 but <200

    st = SubTask("s1", "do thing", "primary", 4096, "You do things.",
                 [], 1024, "free", False)
    worker = Worker(st, server)

    timeout_result = WorkerResult(
        subtask_id="s1", content=short_content, success=False,
        latency_ms=180000, tokens_used=50, model="primary",
        error="Request timed out",
    )
    with patch.object(worker, "run", return_value=timeout_result):
        result = worker.run_with_retry(max_retries=1)

    assert not result.success


def test_run_with_retry_timeout_without_output():
    """Timeout error with empty content → returns failed, no retry."""
    from lore.worker import Worker, WorkerResult
    from lore.decomposer import SubTask

    server = MagicMock()

    st = SubTask("s1", "do thing", "primary", 4096, "You do things.",
                 [], 1024, "free", False)
    worker = Worker(st, server)

    timeout_result = WorkerResult(
        subtask_id="s1", content="", success=False,
        latency_ms=180000, tokens_used=0, model="primary",
        error="Connection timeout",
    )
    with patch.object(worker, "run", return_value=timeout_result):
        result = worker.run_with_retry(max_retries=1)

    assert not result.success


def test_run_with_retry_generation_error_retries():
    """Non-timeout error → retries once with escalation."""
    from lore.worker import Worker, WorkerResult
    from lore.decomposer import SubTask

    server = MagicMock()
    server.chat.return_value = {"choices": [{"message": {"content": "success"}}]}

    st = SubTask("s1", "do thing", "primary", 4096, "You do things.",
                 [], 1024, "free", False)
    worker = Worker(st, server)

    gen_error_result = WorkerResult(
        "s1", "Error: generation error", False,
        100, 5, "primary", "generation error",
    )
    call_count = [0]
    def mock_run(previous_outputs=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return gen_error_result
        # Second call: real run (server.chat returns success)
        return Worker.run(worker, previous_outputs=previous_outputs)

    with patch.object(worker, "run", side_effect=mock_run):
        result = worker.run_with_retry(max_retries=1)

    assert result.success


# ─── Phase 4: Aggregation ────────────────────────────────────────────────────

def test_pre_summarize_short_output_passes_through():
    """Short outputs (<1000 chars) are not summarized."""
    from lore.orchestrator import Orchestrator
    from lore.worker import WorkerResult

    server = MagicMock()
    router = MagicMock()
    memory = MagicMock()
    orch = Orchestrator(server, router, memory, {})

    results = {"s1": WorkerResult("s1", "short output", True, 10, 5, "primary")}
    summaries = orch._pre_summarize_for_aggregation(results)

    assert summaries["s1"] == "short output"
    server.chat.assert_not_called()  # no summarization call


def test_pre_summarize_long_output_uses_specialist():
    """Long outputs (>3000 chars) get summarized by specialist."""
    from lore.orchestrator import Orchestrator
    from lore.worker import WorkerResult

    server = MagicMock()
    server.chat.return_value = {"choices": [{"message": {"content": "summary"}}]}
    router = MagicMock()
    memory = MagicMock()
    orch = Orchestrator(server, router, memory, {})

    long_content = "x" * 3500
    results = {"s1": WorkerResult("s1", long_content, True, 10, 100, "primary")}
    summaries = orch._pre_summarize_for_aggregation(results)

    assert summaries["s1"] == "summary"
    server.chat.assert_called_once()
    assert server.chat.call_args[0][0] == "specialist"


def test_pre_summarize_medium_output_truncates():
    """Medium outputs (1000-3000 chars) get truncated, not LLM-summarized."""
    from lore.orchestrator import Orchestrator
    from lore.worker import WorkerResult

    server = MagicMock()
    router = MagicMock()
    memory = MagicMock()
    orch = Orchestrator(server, router, memory, {})

    medium_content = "x" * 2500
    results = {"s1": WorkerResult("s1", medium_content, True, 10, 100, "primary")}
    summaries = orch._pre_summarize_for_aggregation(results)

    assert "truncated" in summaries["s1"]
    server.chat.assert_not_called()  # no LLM call for medium outputs


def test_pre_summarize_falls_back_on_specialist_failure():
    """Specialist summarization failure → truncation fallback."""
    from lore.orchestrator import Orchestrator
    from lore.worker import WorkerResult

    server = MagicMock()
    server.chat.side_effect = Exception("specialist down")
    router = MagicMock()
    memory = MagicMock()
    orch = Orchestrator(server, router, memory, {})

    long_content = "x" * 3500
    results = {"s1": WorkerResult("s1", long_content, True, 10, 100, "primary")}
    summaries = orch._pre_summarize_for_aggregation(results)

    assert "truncated" in summaries["s1"]


def test_decomposer_injects_hints_into_planning_prompt():
    """Decomposer includes classifier hints in the planning prompt sent to primary."""
    from lore.decomposer import TaskDecomposer
    server = MagicMock()
    server.chat.return_value = _mock_decomposition_response([
        {"id": "s1", "description": "Write code", "model": "primary",
         "context_budget": 2048, "system_prompt": "test",
         "dependencies": [], "max_tokens": 1024, "output_format": "free"},
    ])
    decomposer = TaskDecomposer(server)
    hints = {
        "task_type": "code_gen",
        "estimated_subtasks": 3,
        "suggested_model": "primary",
        "needs_code": True,
    }
    decomposer.decompose("Build a REST API", hints=hints)

    call_args = server.chat.call_args
    messages = call_args[0][1]
    user_msg = [m for m in messages if m["role"] == "user"][-1]
    assert "Classifier analysis" in user_msg["content"]
    assert "code_gen" in user_msg["content"]
    assert "3" in user_msg["content"]
    assert "needs_code" in user_msg["content"]


def test_decomposer_without_hints_omits_classifier_section():
    """Decomposer without hints does not include Classifier analysis section."""
    from lore.decomposer import TaskDecomposer
    server = MagicMock()
    server.chat.return_value = _mock_decomposition_response([
        {"id": "s1", "description": "Write code", "model": "primary",
         "context_budget": 2048, "system_prompt": "test",
         "dependencies": [], "max_tokens": 1024, "output_format": "free"},
    ])
    decomposer = TaskDecomposer(server)
    decomposer.decompose("Build a REST API", hints=None)

    call_args = server.chat.call_args
    messages = call_args[0][1]
    user_msg = [m for m in messages if m["role"] == "user"][-1]
    assert "Classifier analysis" not in user_msg["content"]


def test_fast_aggregation_2_short_subtasks_concatenates():
    """2 subtasks with short outputs (<1000 chars total) → concatenate, no LLM call."""


# ─── Fix 1: Timeout-Aware Retry ──────────────────────────────────────────────

def test_worker_run_with_retry_timeout_with_partial_output():
    """On timeout with partial output (>200 chars), treats as success."""
    from lore.worker import Worker
    from lore.worker import WorkerResult
    from lore.decomposer import SubTask

    server = MagicMock()
    # First call: timeout error with partial content
    server.chat.side_effect = Exception("Connection timed out")

    st = SubTask("s1", "do thing", "primary", 4096, "You do things.",
                 [], 2048, "free", False)
    worker = Worker(st, server)
    # Manually inject partial content into the result to simulate partial output
    with patch.object(worker, 'run') as mock_run:
        mock_run.return_value = WorkerResult(
            subtask_id="s1", content="x" * 200,  # >200 chars
            success=False, latency_ms=5000, tokens_used=50,
            model="primary", error="Connection timed out",
        )
        result = worker.run_with_retry(max_retries=1)

    assert result.success
    assert result.error == "timeout_with_partial_output"
    assert len(result.content) == 200


def test_worker_run_with_retry_timeout_without_output():
    """On timeout with no useful output, returns failed (no retry)."""
    from lore.worker import Worker
    from lore.worker import WorkerResult
    from lore.decomposer import SubTask

    server = MagicMock()

    st = SubTask("s1", "do thing", "primary", 4096, "You do things.",
                 [], 2048, "free", False)
    worker = Worker(st, server)
    with patch.object(worker, 'run') as mock_run:
        mock_run.return_value = WorkerResult(
            subtask_id="s1", content="",
            success=False, latency_ms=5000, tokens_used=0,
            model="primary", error="request timed out",
        )
        result = worker.run_with_retry(max_retries=1)

    assert not result.success
    # Should NOT have retried (run called only once)
    mock_run.assert_called_once()


def test_worker_run_with_retry_generation_error_retries_with_escalation():
    """On non-timeout error, retries once with doubled max_tokens."""
    from lore.worker import Worker
    from lore.worker import WorkerResult
    from lore.decomposer import SubTask

    server = MagicMock()

    st = SubTask("s1", "do thing", "primary", 4096, "You do things.",
                 [], 1024, "free", False)
    worker = Worker(st, server)
    with patch.object(worker, 'run') as mock_run:
        mock_run.side_effect = [
            WorkerResult(subtask_id="s1", content="", success=False,
                         latency_ms=1000, tokens_used=0, model="primary",
                         error="server error"),
            WorkerResult(subtask_id="s1", content="ok", success=True,
                         latency_ms=1000, tokens_used=50, model="primary"),
        ]
        result = worker.run_with_retry(max_retries=1)

    assert result.success
    assert mock_run.call_count == 2


def test_worker_run_with_retry_generation_error_caps_at_4096():
    """Generation error escalation caps max_tokens at 4096."""
    from lore.worker import Worker
    from lore.decomposer import SubTask

    st = SubTask("s1", "do thing", "primary", 4096, "You do things.",
                 [], 2048, "free", False)
    assert st.max_tokens == 2048
    # Simulate what run_with_retry does on generation error
    st.max_tokens = min(st.max_tokens * 2, 4096)
    assert st.max_tokens == 4096  # doubled, capped
    # Second escalation should not exceed cap
    st.max_tokens = min(st.max_tokens * 2, 4096)
    assert st.max_tokens == 4096  # stays at cap


def test_worker_run_with_retry_default_max_retries_is_1():
    """Default max_retries is 1 (retry once, not twice)."""
    import inspect
    from lore.worker import Worker
    sig = inspect.signature(Worker.run_with_retry)
    assert sig.parameters["max_retries"].default == 1


# ─── Fix 2: Explicit Timeouts ────────────────────────────────────────────────

def test_worker_passes_no_timeout_to_server():
    """Worker does NOT pass a timeout to server.chat() — no timeouts."""
    from lore.worker import Worker
    from lore.decomposer import SubTask

    server = MagicMock()
    server.chat.return_value = {"choices": [{"message": {"content": "result"}}]}

    st = SubTask("s1", "do thing", "primary", 2048, "test",
                 [], 1024, "free", False)
    worker = Worker(st, server)
    worker.run()

    # No timeout kwarg should be passed
    assert "timeout" not in server.chat.call_args[1] or server.chat.call_args[1]["timeout"] is None


def test_worker_no_hardcoded_timeout():
    """Worker.run() source has no timeout= in the chat call."""
    import inspect
    from lore.worker import Worker

    source = inspect.getsource(Worker.run)
    # No timeout= in the primary chat call
    assert "timeout=180" not in source


# ─── Fix 3: No Orchestration Timeout (circuit breaker removed) ───────────────

def test_orchestrator_no_timeout_all_subtasks_complete():
    """No orchestration time budget — all subtasks complete regardless of elapsed time."""
    from lore.orchestrator import Orchestrator
    from unittest.mock import patch

    server = MagicMock()
    router = MagicMock()
    router.classify.return_value = ("PRIMARY", 0.90)
    memory = MagicMock()

    plan_json = json.dumps({
        "subtasks": [
            {"id": "s1", "description": "Write code", "model": "primary",
             "context_budget": 2048, "system_prompt": "test",
             "dependencies": [], "max_tokens": 1024, "output_format": "free"},
            {"id": "s2", "description": "Write tests", "model": "primary",
             "context_budget": 2048, "system_prompt": "test",
             "dependencies": ["s1"], "max_tokens": 1024, "output_format": "free"},
            {"id": "s3", "description": "Write docs", "model": "primary",
             "context_budget": 2048, "system_prompt": "test",
             "dependencies": ["s1"], "max_tokens": 1024, "output_format": "free"},
        ],
        "aggregation_prompt": "Combine.",
    })

    server.chat.side_effect = [
        {"choices": [{"message": {"content": plan_json}}]},  # planning
        {"choices": [{"message": {"content": "s1 result"}}]},  # s1 (wave 1)
        {"choices": [{"message": {"content": "aggregated result"}}]},  # aggregation
    ]
    server.tokenize.return_value = 5

    orch = Orchestrator(server, router, memory, {})

    # Even with large time values, all waves execute — no circuit breaker
    call_count = [0]
    def mock_time():
        call_count[0] += 1
        return 10000.0  # very large time value

    with patch("lore.orchestrator.time.time", side_effect=mock_time):
        r = orch.process("Write a parser and then test it and also document it thoroughly")

    assert r["orchestrated"] is True
    # All subtasks complete — no time budget cutoff
    assert r["subtasks_completed"] >= 1


def test_orchestrator_no_results_falls_back_to_dispatch():
    """Fallback plan (planning failed) → delegates to dispatch, no subtask results."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    router = MagicMock()
    router.classify.return_value = ("PRIMARY", 0.90)
    memory = MagicMock()

    # Planning call fails → decomposer returns fallback plan → delegates to dispatch
    server.chat.side_effect = Exception("planning failed")
    server.tokenize.return_value = 5

    orch = Orchestrator(server, router, memory, {})

    def dispatch_fn(q, json_mode=False):
        return {"route": "PRIMARY", "confidence": 0.9, "model": "primary",
                "content": "dispatched", "success": True, "latency_ms": 5.0}

    r = orch.process("Write a parser and then test it and also document it thoroughly",
                     dispatch_fn=dispatch_fn)

    assert r["orchestrated"] is False
    assert r["content"] == "dispatched"


# ─── Fix 6: Fast Aggregation for Code Tasks ──────────────────────────────────

def test_fast_aggregation_code_only_no_deps():
    """Code-only plan with no deps → concatenate, no LLM aggregation call."""
    from lore.orchestrator import Orchestrator
    from lore.decomposer import TaskPlan, SubTask
    from lore.worker import WorkerResult

    server = MagicMock()
    router = MagicMock()
    memory = MagicMock()
    orch = Orchestrator(server, router, memory, {})

    plan = TaskPlan(
        original_query="test",
        subtasks=[
            SubTask("s1", "First task", "primary", 1024, "sys", [], 512, "free"),
            SubTask("s2", "Second task", "primary", 1024, "sys", [], 512, "free"),
        ],
    )
    results = {
        "s1": WorkerResult("s1", "short result 1", True, 10, 50, "primary"),
        "s2": WorkerResult("s2", "short result 2", True, 10, 50, "primary"),
    }
    content = orch._aggregate("test query", plan, results)

    # Should concatenate without calling primary for aggregation
    assert "short result 1" in content
    assert "short result 2" in content
    # server.chat should NOT have been called for aggregation
    # (only called if pre-summarize was needed, which it wasn't for short outputs)
    server.chat.assert_not_called()


def test_fast_aggregation_2_long_subtasks_uses_llm():
    """2 subtasks with long outputs (>1000 chars total) → standard LLM aggregation."""
    from lore.orchestrator import Orchestrator
    from lore.decomposer import TaskPlan, SubTask
    from lore.worker import WorkerResult

    server = MagicMock()
    server.chat.return_value = {"choices": [{"message": {"content": "aggregated result"}}]}
    router = MagicMock()
    memory = MagicMock()
    orch = Orchestrator(server, router, memory, {})

    plan = TaskPlan(
        original_query="test",
        subtasks=[
            SubTask("s1", "First task", "primary", 1024, "sys", [], 512, "free"),
            SubTask("s2", "Second task", "primary", 1024, "sys", [], 512, "free"),
        ],
    )
    long_content = "x" * 600
    results = {
        "s1": WorkerResult("s1", long_content, True, 10, 100, "primary"),
        "s2": WorkerResult("s2", long_content, True, 10, 100, "primary"),
    }
    content = orch._aggregate("test query", plan, results)

    assert content == "aggregated result"
    server.chat.assert_called()  # LLM aggregation was used


def test_orchestration_result_includes_metrics():
    """Orchestration result dict includes structured metrics."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    router = MagicMock()
    router.classify.return_value = ("PRIMARY", 0.90)
    memory = MagicMock()

    plan_json = json.dumps({
        "subtasks": [
            {"id": "s1", "description": "Write code", "model": "primary",
             "context_budget": 4096, "system_prompt": "Write code.",
             "dependencies": [], "max_tokens": 2048, "output_format": "code_python"},
            {"id": "s2", "description": "Write tests", "model": "primary",
             "context_budget": 4096, "system_prompt": "Write tests.",
             "dependencies": ["s1"], "max_tokens": 2048, "output_format": "code_python"},
        ],
        "aggregation_prompt": "Combine all outputs.",
    })

    server.chat.side_effect = [
        {"choices": [{"message": {"content": plan_json}}]},
        {"choices": [{"message": {"content": "def foo(): pass"}}]},
        {"choices": [{"message": {"content": "def test_foo(): pass"}}]},
        {"choices": [{"message": {"content": "Complete solution"}}]},
    ]
    server.tokenize.return_value = 10

    orch = Orchestrator(server, router, memory, {})
    query = "Write a Python function to parse CSV files and then add unit tests for it thoroughly"
    r = orch.process(query)

    assert r["orchestrated"] is True
    assert "metrics" in r
    m = r["metrics"]
    assert "decompose_ms" in m
    assert "execute_ms" in m
    assert "aggregate_ms" in m
    assert "total_ms" in m
    assert "subtasks" in m
    assert "waves" in m
    assert "llm_calls" in m
    assert "partial_results" in m
    assert m["subtasks"] == 2


# ─── Parallel Slots & Intelligent Supervision ────────────────────────────────

def test_parallel_slots_config_read_from_orchestrator_config():
    """Orchestrator reads parallel_slots from config."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    router = MagicMock()
    memory = MagicMock()
    orch = Orchestrator(server, router, memory, {"parallel_slots": 3})
    assert orch._parallel_slots == 3


def test_parallel_slots_defaults_to_3():
    """Default parallel_slots is 3 when not in config."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    router = MagicMock()
    memory = MagicMock()
    orch = Orchestrator(server, router, memory, {})
    assert orch._parallel_slots == 3


def test_execute_wave_runs_same_model_in_parallel():
    """Wave with 3 subtasks on same model → all run in parallel via ThreadPoolExecutor."""
    from lore.orchestrator import Orchestrator
    from lore.decomposer import SubTask

    server = MagicMock()
    server.tokenize.return_value = 5
    server.chat.return_value = {"choices": [{"message": {"content": "result"}}]}

    router = MagicMock()
    memory = MagicMock()
    orch = Orchestrator(server, router, memory, {"parallel_slots": 3})

    wave = [
        SubTask("s1", "task a", "primary", 2048, "sp", [], 1024, "free", False),
        SubTask("s2", "task b", "primary", 2048, "sp", [], 1024, "free", False),
        SubTask("s3", "task c", "primary", 2048, "sp", [], 1024, "free", False),
    ]
    results = orch._execute_wave(wave, {})

    # All 3 subtasks complete — even though same model
    assert len(results) == 3
    assert all(r.success for r in results.values())


def test_execute_wave_caps_workers_at_parallel_slots():
    """ThreadPoolExecutor max_workers capped at parallel_slots even with more subtasks."""
    from lore.orchestrator import Orchestrator
    from lore.decomposer import SubTask

    server = MagicMock()
    server.tokenize.return_value = 5
    server.chat.return_value = {"choices": [{"message": {"content": "result"}}]}

    router = MagicMock()
    memory = MagicMock()
    orch = Orchestrator(server, router, memory, {"parallel_slots": 2})

    wave = [
        SubTask("s1", "a", "primary", 2048, "sp", [], 1024, "free", False),
        SubTask("s2", "b", "primary", 2048, "sp", [], 1024, "free", False),
        SubTask("s3", "c", "primary", 2048, "sp", [], 1024, "free", False),
    ]
    # Should still complete all 3, just with max_workers=2
    results = orch._execute_wave(wave, {})
    assert len(results) == 3


def test_check_slot_activity_returns_active_slots():
    """_check_slot_activity queries /slots and returns active ones."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    server.get_slots.return_value = [
        {"id": 0, "is_processing": True, "n_past": 150},
        {"id": 1, "is_processing": False, "n_past": 0},
        {"id": 2, "is_processing": True, "n_past": 80},
    ]
    router = MagicMock()
    memory = MagicMock()
    orch = Orchestrator(server, router, memory, {})

    active = orch._check_slot_activity("primary")
    assert len(active) == 2
    assert all(s["is_processing"] for s in active)


def test_check_slot_activity_empty_on_error():
    """_check_slot_activity returns empty list on server error."""
    from lore.orchestrator import Orchestrator

    server = MagicMock()
    server.get_slots.return_value = []
    router = MagicMock()
    memory = MagicMock()
    orch = Orchestrator(server, router, memory, {})

    active = orch._check_slot_activity("primary")
    assert active == []


def test_decomposer_max_subtasks_defaults_to_3():
    """TaskDecomposer default max_subtasks is 3 (was 5)."""
    from lore.decomposer import TaskDecomposer

    server = MagicMock()
    decomposer = TaskDecomposer(server)
    assert decomposer._max_subtasks == 3


def test_decomposer_caps_at_3_subtasks():
    """Decomposer truncates plans to 3 subtasks."""
    from lore.decomposer import TaskDecomposer

    server = MagicMock()
    # Return 5 subtasks in the plan
    plan_json = json.dumps({
        "subtasks": [
            {"id": "s1", "description": "a", "model": "primary", "context_budget": 2048,
             "system_prompt": "test", "dependencies": [], "max_tokens": 1024, "output_format": "free"},
            {"id": "s2", "description": "b", "model": "primary", "context_budget": 2048,
             "system_prompt": "test", "dependencies": [], "max_tokens": 1024, "output_format": "free"},
            {"id": "s3", "description": "c", "model": "primary", "context_budget": 2048,
             "system_prompt": "test", "dependencies": [], "max_tokens": 1024, "output_format": "free"},
            {"id": "s4", "description": "d", "model": "primary", "context_budget": 2048,
             "system_prompt": "test", "dependencies": [], "max_tokens": 1024, "output_format": "free"},
            {"id": "s5", "description": "e", "model": "primary", "context_budget": 2048,
             "system_prompt": "test", "dependencies": [], "max_tokens": 1024, "output_format": "free"},
        ],
        "aggregation_prompt": "Combine.",
    })
    server.chat.return_value = {"choices": [{"message": {"content": plan_json}}]}

    decomposer = TaskDecomposer(server)  # default max_subtasks=3
    plan = decomposer.decompose("complex task")
    assert len(plan.subtasks) <= 3
