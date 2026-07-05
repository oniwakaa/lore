import json
import tempfile
from pathlib import Path


def test_run_variant_computes_metrics():
    from lore.ab_test import ABTest

    tasks = [{"id": i} for i in range(4)]
    latencies = [0.1, 0.2, 0.3, 0.4]

    def run_fn(task, config):
        return {"latency_s": latencies[task["id"]], "tokens_out": 10, "success": True}

    ab = ABTest(tasks, run_fn)
    result = ab.run_variant({}, label="baseline")

    assert result["label"] == "baseline"
    assert result["n_tasks"] == 4
    assert result["completion_rate"] == 1.0
    assert result["p50_latency_s"] > 0
    assert result["avg_tokens_per_sec"] > 0


def test_run_variant_tracks_failures():
    from lore.ab_test import ABTest

    tasks = [{"id": 0}, {"id": 1}, {"id": 2}]

    def run_fn(task, config):
        success = task["id"] != 1
        return {"latency_s": 0.1, "tokens_out": 5 if success else 0, "success": success}

    ab = ABTest(tasks, run_fn)
    result = ab.run_variant({})
    assert result["completion_rate"] == round(2 / 3, 4)


def test_run_variant_handles_empty_task_list():
    from lore.ab_test import ABTest

    ab = ABTest([], lambda task, config: {"latency_s": 0, "tokens_out": 0, "success": True})
    result = ab.run_variant({})
    assert result["n_tasks"] == 0
    assert result["completion_rate"] == 0.0
    assert result["avg_tokens_per_sec"] == 0.0


def test_compare_runs_all_named_configs():
    from lore.ab_test import ABTest

    tasks = [{"id": 0}]
    calls = []

    def run_fn(task, config):
        calls.append(config.get("name"))
        return {"latency_s": 0.1, "tokens_out": 10, "success": True}

    ab = ABTest(tasks, run_fn)
    report = ab.compare({"baseline": {"name": "baseline"}, "plus_x": {"name": "plus_x"}})

    assert set(report.keys()) == {"baseline", "plus_x"}
    assert report["baseline"]["label"] == "baseline"
    assert calls == ["baseline", "plus_x"]


def test_save_report_writes_json():
    from lore.ab_test import ABTest

    report = {"baseline": {"n_tasks": 1}}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "results" / "run.json"
        ABTest.save_report(report, str(path))
        assert path.exists()
        assert json.loads(path.read_text()) == report
