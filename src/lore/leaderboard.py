"""Live benchmark scanner. Discovers and ranks models from HuggingFace.

Proactively scans the leaderboard to find better models than what's
currently installed. Notifies the user when upgrades are available.

huggingface_hub and pandas are lazy-imported inside methods so this module
imports cleanly without them (tests mock the HF API).
"""
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Benchmarks -> LORE task type relevance (weight)
TASK_BENCHMARKS: dict[str, list[tuple[str, float]]] = {
    "classification": [("IFEval", 0.6), ("MMLU-Pro", 0.3), ("BBH", 0.1)],
    "extraction":     [("IFEval", 0.7), ("BBH", 0.2), ("MMLU-Pro", 0.1)],
    "summarization":  [("BBH", 0.5), ("MMLU-Pro", 0.3), ("IFEval", 0.2)],
    "code_gen":       [("SWE-bench", 0.4), ("HumanEval", 0.3), ("MBPP", 0.2), ("MATH-Lvl5", 0.1)],
    "testing":        [("HumanEval", 0.5), ("MBPP", 0.4), ("IFEval", 0.1)],
    "documentation":  [("IFEval", 0.5), ("BBH", 0.3), ("MMLU-Pro", 0.2)],
    "math":           [("MATH-Lvl5", 0.6), ("GSM8K", 0.3), ("BBH", 0.1)],
    "planning":       [("BBH", 0.4), ("MMLU-Pro", 0.3), ("SWE-bench", 0.2), ("IFEval", 0.1)],
    "review":         [("BBH", 0.4), ("SWE-bench", 0.3), ("MMLU-Pro", 0.2), ("IFEval", 0.1)],
}

MAX_PARAMS_B = 10.0  # must fit in 16GB at Q4

PREFERRED_QUANTS = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q4_0"]


@dataclass
class ModelCandidate:
    """A model found on the leaderboard."""
    model_id: str                    # "Qwen/Qwen3.5-9B"
    params_b: float                  # 9.0
    scores: dict[str, float] = field(default_factory=dict)         # {"MMLU-Pro": 72.5, ...}
    task_scores: dict[str, float] = field(default_factory=dict)    # {"code_gen": 78.3, ...}
    gguf_repo: str = ""
    gguf_quant: str = ""
    gguf_size_gb: float = 0.0
    is_installed: bool = False
    installed_quant: str = ""


@dataclass
class UpgradeCandidate:
    """A model that's better than what's currently installed for a task type."""
    task_type: str
    current_model: str               # what we have now
    current_score: float             # its score for this task
    better_model: ModelCandidate     # the upgrade
    better_score: float              # its score
    improvement_pct: float           # percentage improvement


class LeaderboardScanner:
    """Scans HuggingFace leaderboard and finds upgrade candidates."""

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._hf_token = self._config.get("hf_token")
        self._api = None
        self._parquet_cache: list[ModelCandidate] | None = None
        self._parquet_cache_time = 0.0
        self._cache_ttl = self._config.get("cache_ttl_hours", 24) * 3600

    def _get_api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self._hf_token)
        return self._api

    def scan_for_upgrades(self, installed_models: dict[str, str],
                          models_dir: str = "models",
                          min_improvement_pct: float = 5.0) -> list[UpgradeCandidate]:
        """Scan leaderboard for models better than what's installed.

        Args:
            installed_models: {task_type: model_id} current assignments
            models_dir: path to local models directory
            min_improvement_pct: only flag upgrades with >X% improvement

        Returns:
            List of UpgradeCandidate, sorted by improvement (biggest first).
        """
        all_candidates = self._load_leaderboard_data()
        if not all_candidates:
            logger.warning("No leaderboard data available")
            return []

        viable = self._filter_viable(all_candidates, models_dir)
        logger.info(f"Found {len(viable)} viable models (size + GGUF check)")

        for candidate in viable:
            for task_type in TASK_BENCHMARKS:
                candidate.task_scores[task_type] = self._compute_task_score(
                    candidate.scores, task_type
                )

        upgrades: list[UpgradeCandidate] = []
        for task_type, current_model_id in installed_models.items():
            current_score = self._score_installed_model(
                current_model_id, task_type, viable
            )

            for candidate in viable:
                if candidate.is_installed:
                    continue
                if candidate.model_id == current_model_id:
                    continue

                cand_score = candidate.task_scores.get(task_type, 0)
                if cand_score <= current_score:
                    continue

                improvement = ((cand_score - current_score) / max(current_score, 1)) * 100
                if improvement < min_improvement_pct:
                    continue

                upgrades.append(UpgradeCandidate(
                    task_type=task_type,
                    current_model=current_model_id,
                    current_score=current_score,
                    better_model=candidate,
                    better_score=cand_score,
                    improvement_pct=improvement,
                ))

        upgrades.sort(key=lambda u: -u.improvement_pct)

        # Group by model_id: keep all tasks a model improves (not just first)
        return upgrades

    def _load_leaderboard_data(self) -> list[ModelCandidate]:
        """Load model data from the pre-aggregated parquet + individual lookups."""
        # Check cache
        if self._parquet_cache is not None:
            if time.time() - self._parquet_cache_time < self._cache_ttl:
                return self._parquet_cache

        try:
            import pandas as pd
            df = pd.read_parquet(
                "hf://datasets/OpenEvals/leaderboard-data/data/train-00000-of-00001.parquet"
            )
        except Exception as e:
            logger.warning(f"Failed to load leaderboard parquet: {e}")
            return self._load_from_individual_leaderboards()

        candidates: list[ModelCandidate] = []
        for _, row in df.iterrows():
            model_id = row.get("model_name", row.get("model_id", ""))
            if not model_id:
                continue

            scores: dict[str, float] = {}
            params_b = 0.0
            # Map parquet column names to TASK_BENCHMARKS keys
            _COL_TO_BENCH = {
                "ifeval": "IFEval",
                "mmlu_pro": "MMLU-Pro",
                "swe_bench": "SWE-bench",
                "math_lvl5": "MATH-Lvl5",
                "humaneval": "HumanEval",
            }
            for col in df.columns:
                if col.endswith("_score"):
                    raw = col.replace("_score", "").lower()
                    bench = _COL_TO_BENCH.get(raw, col.replace("_score", "").upper())
                    val = row[col]
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        scores[bench] = float(val)
                if col in ("params", "num_params", "parameter_count"):
                    val = row[col]
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        params_b = val / 1e9 if val > 1e8 else val

            if not scores:
                continue

            candidates.append(ModelCandidate(
                model_id=str(model_id),
                params_b=params_b,
                scores=scores,
            ))

        logger.info(f"Loaded {len(candidates)} models from leaderboard parquet")
        self._parquet_cache = candidates
        self._parquet_cache_time = time.time()
        return candidates

    def _load_from_individual_leaderboards(self) -> list[ModelCandidate]:
        """Fallback: fetch from individual benchmark leaderboards."""
        api = self._get_api()
        all_models: dict[str, dict] = {}

        benchmark_datasets = {
            "IFEval": "google/IFEval",
            "MMLU-Pro": "TIGER-Lab/MMLU-Pro",
            "BBH": "lukaemon/bbh",
            "HumanEval": "openai/openai_humaneval",
            "MBPP": "google-research/mbpp",
            "MATH-Lvl5": "HuggingFaceH4/MATH-500",
            "GSM8K": "openai/gsm8k",
            "SWE-bench": "SWE-bench/SWE-bench_Verified",
        }

        for bench_name, dataset_id in benchmark_datasets.items():
            try:
                entries = api.get_dataset_leaderboard(dataset_id)
                for entry in entries[:100]:
                    mid = entry.model_id
                    if mid not in all_models:
                        all_models[mid] = {"model_id": mid, "scores": {}}
                    all_models[mid]["scores"][bench_name] = float(entry.value)
            except Exception as e:
                logger.debug(f"Leaderboard fetch failed for {bench_name}: {e}")

        candidates: list[ModelCandidate] = []
        for data in all_models.values():
            if len(data["scores"]) >= 2:
                candidates.append(ModelCandidate(
                    model_id=data["model_id"],
                    params_b=0,
                    scores=data["scores"],
                ))

        return candidates

    def _filter_viable(self, candidates: list[ModelCandidate],
                       models_dir: str) -> list[ModelCandidate]:
        """Filter models by size class and GGUF availability.
        
        Two-pass approach to avoid API calls for all candidates:
        1. Filter by size + check local installs (no API calls)
        2. Check GGUF availability only for top N uninstalled candidates
        """
        from pathlib import Path

        # First pass: filter by size and check local installs
        sized: list[ModelCandidate] = []
        local_files = list(Path(models_dir).glob("*.gguf")) if Path(models_dir).exists() else []
        
        for c in candidates:
            if c.params_b > 0 and c.params_b > MAX_PARAMS_B:
                continue

            model_name_short = c.model_id.split("/")[-1].lower()
            for f in local_files:
                if model_name_short in f.name.lower():
                    c.is_installed = True
                    for q in PREFERRED_QUANTS:
                        if q.lower() in f.name.lower():
                            c.installed_quant = q
                            break
                    break

            sized.append(c)

        # Installed models are always viable
        viable = [c for c in sized if c.is_installed]
        uninstalled = [c for c in sized if not c.is_installed]

        # Second pass: check GGUF only for top candidates (avoid rate limits)
        MAX_GGUF_CHECKS = 30
        for c in uninstalled[:MAX_GGUF_CHECKS]:
            gguf_repo, gguf_quant, gguf_size = self._find_gguf(c.model_id)
            if gguf_repo:
                c.gguf_repo = gguf_repo
                c.gguf_quant = gguf_quant
                c.gguf_size_gb = gguf_size
                viable.append(c)

        return viable

    def _find_gguf(self, model_id: str) -> tuple[str, str, float]:
        """Find GGUF version on HuggingFace."""
        try:
            api = self._get_api()

            info = api.model_info(model_id)
            if info.tags and any("gguf" in t.lower() for t in info.tags):
                for sib in (info.siblings or []):
                    for quant in PREFERRED_QUANTS:
                        if quant.lower() in sib.rfilename.lower():
                            return model_id, quant, (sib.size or 0) / 1e9

            base = model_id.split("/")[-1]
            org = model_id.split("/")[0] if "/" in model_id else ""
            for suffix in ["-GGUF", "-gguf"]:
                candidate = f"{org}/{base}{suffix}" if org else f"{base}{suffix}"
                try:
                    cinfo = api.model_info(candidate)
                    for sib in (cinfo.siblings or []):
                        for quant in PREFERRED_QUANTS:
                            if quant.lower() in sib.rfilename.lower():
                                return candidate, quant, (sib.size or 0) / 1e9
                except Exception:
                    continue

        except Exception:
            pass
        return "", "", 0.0

    def _compute_task_score(self, scores: dict[str, float], task_type: str) -> float:
        """Compute weighted score for a task type."""
        benchmarks = TASK_BENCHMARKS.get(task_type, TASK_BENCHMARKS.get("planning", []))
        total_weight = 0.0
        weighted_sum = 0.0
        for bench_name, weight in benchmarks:
            if bench_name in scores:
                weighted_sum += scores[bench_name] * weight
                total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _score_installed_model(self, model_id: str, task_type: str,
                               candidates: list[ModelCandidate]) -> float:
        """Get the score of an installed model for a task type."""
        for c in candidates:
            if c.model_id == model_id or c.model_id.endswith(model_id.split("/")[-1]):
                return c.task_scores.get(task_type, 0)

        try:
            api = self._get_api()
            info = api.model_info(model_id, expand=["evalResults"])
            scores: dict[str, float] = {}
            if info.eval_results:
                for r in info.eval_results:
                    bench = self._normalize_benchmark(r.dataset_id)
                    if bench:
                        scores[bench] = float(r.value)
            return self._compute_task_score(scores, task_type)
        except Exception:
            return 0.0

    def _normalize_benchmark(self, dataset_id: str) -> str | None:
        lower = dataset_id.lower()
        mapping = {
            "mmlu-pro": "MMLU-Pro", "mmlu": "MMLU-Pro",
            "humaneval": "HumanEval", "mbpp": "MBPP",
            "gsm8k": "GSM8K", "math": "MATH-Lvl5",
            "ifeval": "IFEval", "instruction-following": "IFEval",
            "bbh": "BBH", "swe-bench": "SWE-bench",
            "gpqa": "GPQA", "arc": "ARC-Challenge",
        }
        for key, name in mapping.items():
            if key in lower:
                return name
        return None

    def get_model_scores(self, model_id: str) -> dict[str, float]:
        """Get all benchmark scores for a specific model."""
        try:
            api = self._get_api()
            info = api.model_info(model_id, expand=["evalResults"])
            scores: dict[str, float] = {}
            if info.eval_results:
                for r in info.eval_results:
                    bench = self._normalize_benchmark(r.dataset_id)
                    if bench:
                        scores[bench] = float(r.value)
            return scores
        except Exception:
            return {}
