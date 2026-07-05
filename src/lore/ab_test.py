"""A/B testing framework. Run a task list under two configs, compare metrics."""
import json
import statistics
from pathlib import Path

import psutil


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(int(p * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[idx]


class ABTest:
    """Runs tasks under a config via run_fn, reports latency/throughput/memory/completion."""

    def __init__(self, tasks: list[dict], run_fn):
        """run_fn(task: dict, config: dict) -> dict with keys:
        latency_s (float), tokens_out (int), success (bool).
        """
        self._tasks = tasks
        self._run_fn = run_fn

    def run_variant(self, config: dict, label: str = "variant") -> dict:
        """Run all tasks once under config, return aggregated metrics."""
        proc = psutil.Process()
        latencies, tok_rates, successes = [], [], []
        peak_rss = proc.memory_info().rss

        for task in self._tasks:
            result = self._run_fn(task, config)
            latencies.append(result.get("latency_s", 0.0))
            tokens_out = result.get("tokens_out", 0)
            if result.get("latency_s", 0) > 0 and tokens_out > 0:
                tok_rates.append(tokens_out / result["latency_s"])
            successes.append(bool(result.get("success", False)))
            peak_rss = max(peak_rss, proc.memory_info().rss)

        sorted_latencies = sorted(latencies)
        n = len(self._tasks)
        return {
            "label": label,
            "n_tasks": n,
            "p50_latency_s": round(_percentile(sorted_latencies, 0.50), 4),
            "p95_latency_s": round(_percentile(sorted_latencies, 0.95), 4),
            "avg_tokens_per_sec": round(statistics.mean(tok_rates), 2) if tok_rates else 0.0,
            "peak_memory_mb": round(peak_rss / 1024**2, 1),
            "completion_rate": round(sum(successes) / n, 4) if n else 0.0,
        }

    def compare(self, configs: dict[str, dict]) -> dict:
        """Run every named config, return {name: metrics} report."""
        return {name: self.run_variant(config, label=name) for name, config in configs.items()}

    @staticmethod
    def save_report(report: dict, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2))
