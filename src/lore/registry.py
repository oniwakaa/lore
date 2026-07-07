"""Model registry with live benchmark auto-selection.

Orchestrator model: user-set, NEVER auto-changed.
Worker models: auto-selected from leaderboard, user approves downloads.

huggingface_hub is lazy-imported inside download method so this module
imports cleanly without it (tests mock the HF API).
"""
import logging
from pathlib import Path
from dataclasses import dataclass

from lore.leaderboard import (
    LeaderboardScanner,
    UpgradeCandidate,
    PREFERRED_QUANTS,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkerAssignment:
    """Current model assignment for a task type."""
    task_type: str
    model_id: str
    model_path: str           # local GGUF path
    benchmark_score: float    # score for this task type
    auto_selected: bool       # True if picked by leaderboard, False if manual fallback


class ModelRegistry:
    """Model registry. Orchestrator is locked. Workers are benchmark-driven."""

    def __init__(self, config: dict, models_dir: str = "models"):
        self._config = config
        self._models_dir = Path(models_dir)
        orchestrator_cfg = config.get("orchestrator", {})
        self._orchestrator_model = orchestrator_cfg.get("model", "primary")
        self._auto_select = config.get("auto_select", True)
        self._size_class = config.get("size_class", "medium")
        self._scanner = LeaderboardScanner(config.get("leaderboard", {}))
        self._assignments: dict[str, WorkerAssignment] = {}
        self._task_mapping_fallback = config.get("task_mapping", {})

    def select_workers(self) -> dict[str, WorkerAssignment]:
        """Auto-select the best installed model for each task type.

        Only picks from models already downloaded locally.
        Use check_for_upgrades() + approve_upgrade() to get new models.
        """
        from lore.leaderboard import TASK_BENCHMARKS
        task_types = list(TASK_BENCHMARKS.keys())

        local_models = self._scan_local_models()

        for task_type in task_types:
            best_model: tuple[str, str] | None = None
            best_score = -1.0

            for model_id, model_path in local_models.items():
                scores = self._scanner.get_model_scores(model_id)
                if not scores:
                    continue
                task_score = self._scanner._compute_task_score(scores, task_type)
                if task_score > best_score:
                    best_score = task_score
                    best_model = (model_id, model_path)

            if best_model:
                self._assignments[task_type] = WorkerAssignment(
                    task_type=task_type,
                    model_id=best_model[0],
                    model_path=best_model[1],
                    benchmark_score=best_score,
                    auto_selected=True,
                )
            else:
                fallback = self._task_mapping_fallback.get(task_type, "specialist")
                self._assignments[task_type] = WorkerAssignment(
                    task_type=task_type,
                    model_id=fallback,
                    model_path="",
                    benchmark_score=0.0,
                    auto_selected=False,
                )

        return self._assignments

    def check_for_upgrades(self) -> list[UpgradeCandidate]:
        """Scan HuggingFace for models better than what's installed.

        Does NOT download anything. Returns list of upgrade candidates.
        """
        installed = {a.task_type: a.model_id for a in self._assignments.values()}
        if not installed:
            # No assignments yet — use fallback mapping as current
            installed = dict(self._task_mapping_fallback)
        return self._scanner.scan_for_upgrades(
            installed, str(self._models_dir), min_improvement_pct=5.0
        )

    def prompt_upgrade(self, upgrades: list[UpgradeCandidate]) -> list[UpgradeCandidate]:
        """Show upgrades to user and get approval.

        In REPL mode: prints table, asks y/n per upgrade.
        In headless mode: logs recommendations, returns empty (no download).
        """
        if not upgrades:
            return []

        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║  MODEL UPGRADES AVAILABLE                                     ║")
        print("╠══════════════════════════════════════════════════════════════╣")

        for i, u in enumerate(upgrades):
            print(f"║  {i+1}. {u.task_type:15s}                                        ║")
            print(f"║     Current: {u.current_model:40s} score={u.current_score:.1f}   ║")
            print(f"║     Better:  {u.better_model.model_id:40s} score={u.better_score:.1f}   ║")
            print(f"║     Improvement: +{u.improvement_pct:.1f}%   Size: {u.better_model.gguf_size_gb:.1f} GB          ║")
            print(f"║                                                              ║")

        print("╚══════════════════════════════════════════════════════════════╝")

        approved: list[UpgradeCandidate] = []
        for i, u in enumerate(upgrades):
            try:
                response = input(f"Download {u.better_model.model_id} for {u.task_type}? [y/N/all]: ").strip().lower()
            except EOFError:
                # Headless mode — no input, don't download
                break
            if response == "y":
                approved.append(u)
            elif response == "all":
                approved.extend(upgrades[i:])
                break

        return approved

    def approve_upgrade(self, upgrade: UpgradeCandidate) -> bool:
        """Download an approved upgrade and update assignment.

        Returns True if download succeeded.
        """
        model = upgrade.better_model

        if not model.gguf_repo:
            logger.error(f"No GGUF repo for {model.model_id}")
            return False

        success = self._download_gguf(model)
        if success:
            local_path = self._find_local_gguf(model.model_id, model.gguf_quant)
            self._assignments[upgrade.task_type] = WorkerAssignment(
                task_type=upgrade.task_type,
                model_id=model.model_id,
                model_path=local_path,
                benchmark_score=upgrade.better_score,
                auto_selected=True,
            )
            logger.info(f"Upgraded {upgrade.task_type}: {model.model_id}")
            return True

        return False

    def _scan_local_models(self) -> dict[str, str]:
        """Find all locally installed GGUF models.

        Returns {model_id_guess: file_path}
        """
        models: dict[str, str] = {}
        if not self._models_dir.exists():
            return models
        for f in self._models_dir.glob("*.gguf"):
            name = f.stem
            for q in PREFERRED_QUANTS:
                name = name.replace(f"-{q}", "").replace(f"_{q}", "")
            models[name] = str(f)
        return models

    def _download_gguf(self, model) -> bool:
        """Download GGUF from HuggingFace."""
        self._models_dir.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=model.gguf_repo,
                filename=f"*{model.gguf_quant}*gguf",
                local_dir=str(self._models_dir),
            )
            logger.info(f"Downloaded {model.model_id}: {path}")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def _find_local_gguf(self, model_id: str, quant: str) -> str:
        name = model_id.split("/")[-1]
        for f in self._models_dir.glob(f"*{name}*{quant}*.gguf"):
            return str(f)
        for f in self._models_dir.glob(f"*{name}*.gguf"):
            return str(f)
        return ""

    @property
    def orchestrator_model(self) -> str:
        return self._orchestrator_model

    def get_model_for_task(self, task_type: str) -> str:
        """Get the assigned model for a task type."""
        assignment = self._assignments.get(task_type)
        if assignment:
            return assignment.model_id
        return self._task_mapping_fallback.get(task_type, "specialist")
