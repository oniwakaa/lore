# LORE — Phase 4.2: Live Benchmark Model Selection

You are working on LORE (Local Orchestration & Runtime Engine). Read `AGENTS.md` first.

## The Principle

**LORE proactively discovers better models. The user only sets the orchestrator model.**

The flow:
1. LORE checks HuggingFace leaderboard for all models in the target size range
2. Compares against currently installed models for each task type
3. If a better model exists that isn't installed → notifies the user with the comparison
4. User approves → LORE downloads the GGUF and swaps it in
5. User declines → LORE keeps the current model

LORE does NOT wait for someone to manually add models. It scans the leaderboard, finds the best options, and brings them to the user.

## HuggingFace APIs (verified working)

```python
from huggingface_hub import HfApi
api = HfApi()

# Per-model eval results
info = api.model_info("Qwen/Qwen3.5-9B", expand=["evalResults"])

# Per-benchmark ranked leaderboard
entries = api.get_dataset_leaderboard("SWE-bench/SWE-bench_Verified")

# Pre-aggregated cross-benchmark parquet (fastest for bulk discovery)
import pandas as pd
df = pd.read_parquet("hf://datasets/OpenEvals/leaderboard-data/data/train-00000-of-00001.parquet")
```

## Tasks

### Task 1: Leaderboard Scanner (`src/lore/leaderboard.py`)

Scans HuggingFace for all models in a size class, fetches their benchmark scores, and ranks them per task type.

```python
"""Live benchmark scanner. Discovers and ranks models from HuggingFace.

Proactively scans the leaderboard to find better models than what's
currently installed. Notifies the user when upgrades are available.
"""
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Benchmarks → LORE task type relevance (weight)
TASK_BENCHMARKS = {
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
    scores: dict[str, float]         # {"MMLU-Pro": 72.5, ...}
    task_scores: dict[str, float] = field(default_factory=dict)  # {"code_gen": 78.3, ...}
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
    
    def __init__(self, config: dict = None):
        self._config = config or {}
        self._hf_token = self._config.get("hf_token")
        self._api = None
        self._parquet_cache = None
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
        # 1. Load all candidate models from leaderboard
        all_candidates = self._load_leaderboard_data()
        if not all_candidates:
            logger.warning("No leaderboard data available")
            return []
        
        # 2. Filter by size class and GGUF availability
        viable = self._filter_viable(all_candidates, models_dir)
        logger.info(f"Found {len(viable)} viable models (size + GGUF check)")
        
        # 3. Score each candidate per task type
        for candidate in viable:
            for task_type in TASK_BENCHMARKS:
                candidate.task_scores[task_type] = self._compute_task_score(
                    candidate.scores, task_type
                )
        
        # 4. Compare against installed models
        upgrades = []
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
        
        # Sort by improvement, biggest first
        upgrades.sort(key=lambda u: -u.improvement_pct)
        
        # Deduplicate: if same model is best for multiple tasks, keep top task
        seen_models = set()
        deduped = []
        for u in upgrades:
            if u.better_model.model_id not in seen_models:
                seen_models.add(u.better_model.model_id)
                deduped.append(u)
        
        return deduped
    
    def _load_leaderboard_data(self) -> list[ModelCandidate]:
        """Load model data from the pre-aggregated parquet + individual lookups."""
        import pandas as pd
        
        # Check cache
        if self._parquet_cache is not None:
            if time.time() - self._parquet_cache_time < self._cache_ttl:
                return self._parquet_cache
        
        try:
            df = pd.read_parquet(
                "hf://datasets/OpenEvals/leaderboard-data/data/train-00000-of-00001.parquet"
            )
        except Exception as e:
            logger.warning(f"Failed to load leaderboard parquet: {e}")
            return self._load_from_individual_leaderboards()
        
        candidates = []
        for _, row in df.iterrows():
            model_id = row.get("model_name", row.get("model_id", ""))
            if not model_id:
                continue
            
            scores = {}
            params_b = 0.0
            for col in df.columns:
                if col.endswith("_score"):
                    bench = col.replace("_score", "").upper()
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
                model_id=model_id,
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
        
        candidates = []
        for data in all_models.values():
            if len(data["scores"]) >= 2:  # need at least 2 benchmarks
                candidates.append(ModelCandidate(
                    model_id=data["model_id"],
                    params_b=0,  # unknown, will check later
                    scores=data["scores"],
                ))
        
        return candidates
    
    def _filter_viable(self, candidates: list[ModelCandidate],
                       models_dir: str) -> list[ModelCandidate]:
        """Filter models by size class and GGUF availability."""
        from pathlib import Path
        
        viable = []
        for c in candidates:
            # Check parameter count (skip if unknown and no way to check)
            if c.params_b > 0 and c.params_b > MAX_PARAMS_B:
                continue
            
            # Check if already installed locally
            local_files = list(Path(models_dir).glob("*.gguf"))
            model_name_short = c.model_id.split("/")[-1].lower()
            for f in local_files:
                if model_name_short in f.name.lower():
                    c.is_installed = True
                    # Try to detect quant from filename
                    for q in PREFERRED_QUANTS:
                        if q.lower() in f.name.lower():
                            c.installed_quant = q
                            break
                    break
            
            # If not installed, check if GGUF exists on HF
            if not c.is_installed:
                gguf_repo, gguf_quant, gguf_size = self._find_gguf(c.model_id)
                if gguf_repo:
                    c.gguf_repo = gguf_repo
                    c.gguf_quant = gguf_quant
                    c.gguf_size_gb = gguf_size
            
            # Include if installed OR has GGUF available
            if c.is_installed or c.gguf_repo:
                viable.append(c)
        
        return viable
    
    def _find_gguf(self, model_id: str) -> tuple[str, str, float]:
        """Find GGUF version on HuggingFace."""
        try:
            api = self._get_api()
            
            # Check model itself
            info = api.model_info(model_id)
            if info.tags and any("gguf" in t.lower() for t in info.tags):
                for sib in (info.siblings or []):
                    for quant in PREFERRED_QUANTS:
                        if quant.lower() in sib.rfilename.lower():
                            return model_id, quant, (sib.size or 0) / 1e9
            
            # Check common GGUF repo patterns
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
        # Look in candidates first
        for c in candidates:
            if c.model_id == model_id or c.model_id.endswith(model_id.split("/")[-1]):
                return c.task_scores.get(task_type, 0)
        
        # Try fetching directly
        try:
            api = self._get_api()
            info = api.model_info(model_id, expand=["evalResults"])
            scores = {}
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
            scores = {}
            if info.eval_results:
                for r in info.eval_results:
                    bench = self._normalize_benchmark(r.dataset_id)
                    if bench:
                        scores[bench] = float(r.value)
            return scores
        except Exception:
            return {}
```

### Task 2: Upgrade Notifier

When the scanner finds better models, notify the user and ask for approval before downloading.

In the REPL (or on startup if configured):

```python
def check_for_upgrades(self) -> list[UpgradeCandidate]:
    """Check HuggingFace for models better than what's installed.
    
    Returns list of upgrade candidates. Does NOT download anything.
    The user decides what to do.
    """
    installed = self._get_installed_assignments()  # {task_type: model_id}
    return self._scanner.scan_for_upgrades(
        installed, self._models_dir, min_improvement_pct=5.0
    )

def prompt_upgrade(self, upgrades: list[UpgradeCandidate]) -> list[UpgradeCandidate]:
    """Show upgrades to user and get approval.
    
    In REPL mode: prints table, asks y/n per upgrade.
    In headless/cron mode: logs recommendations, doesn't download.
    """
    if not upgrades:
        return []
    
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  MODEL UPGRADES AVAILABLE                                  ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    
    for i, u in enumerate(upgrades):
        print(f"║  {i+1}. {u.task_type:15s}                                     ║")
        print(f"║     Current: {u.current_model:40s} score={u.current_score:.1f}  ║")
        print(f"║     Better:  {u.better_model.model_id:40s} score={u.better_score:.1f}  ║")
        print(f"║     Improvement: +{u.improvement_pct:.1f}%   Size: {u.better_model.gguf_size_gb:.1f} GB        ║")
        print(f"║                                                             ║")
    
    print("╚══════════════════════════════════════════════════════════════╝")
    
    approved = []
    for i, u in enumerate(upgrades):
        response = input(f"Download {u.better_model.model_id} for {u.task_type}? [y/N/all]: ").strip().lower()
        if response == "y":
            approved.append(u)
        elif response == "all":
            approved.extend(upgrades[i:])
            break
    
    return approved
```

### Task 3: Model Registry (`src/lore/registry.py`)

```python
"""Model registry with live benchmark auto-selection.

Orchestrator model: user-set, NEVER auto-changed.
Worker models: auto-selected from leaderboard, user approves downloads.
"""
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, field

from lore.leaderboard import LeaderboardScanner, UpgradeCandidate

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
        self._orchestrator_model = config["orchestrator"]["model"]
        self._auto_select = config.get("auto_select", True)
        self._size_class = config.get("size_class", "medium")
        self._scanner = LeaderboardScanner(config.get("leaderboard", {}))
        self._assignments: dict[str, WorkerAssignment] = {}
        self._task_mapping_fallback = config.get("task_mapping", {})
    
    def select_workers(self) -> dict[str, WorkerAssignment]:
        """Auto-select the best installed model for each task type.
        
        Only picks from models already downloaded locally.
        Use check_for_upgrades() + approve_upgrades() to get new models.
        """
        task_types = list(self._scanner.TASK_BENCHMARKS.keys()) if hasattr(self._scanner, 'TASK_BENCHMARKS') else [
            "classification", "extraction", "summarization", "code_gen",
            "testing", "documentation", "math", "planning", "review"
        ]
        
        # Find all locally installed GGUF files
        local_models = self._scan_local_models()
        
        # Score each local model for each task type
        for task_type in task_types:
            best_model = None
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
                # Fallback to config default
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
        return self._scanner.scan_for_upgrades(
            installed, str(self._models_dir), min_improvement_pct=5.0
        )
    
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
            # Update assignment
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
        models = {}
        for f in self._models_dir.glob("*.gguf"):
            # Try to extract model ID from filename
            name = f.stem  # e.g., "Falcon-H1-1.5B-Instruct-Q4_K_M"
            # Remove quant suffix
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
```

### Task 4: CLI Commands

```python
# /upgrades — check for better models on HuggingFace
if query == "/upgrades":
    upgrades = registry.check_for_upgrades()
    if not upgrades:
        print("All installed models are current best for their tasks.")
    else:
        approved = registry.prompt_upgrade(upgrades)
        for u in approved:
            print(f"Downloading {u.better_model.model_id}...")
            ok = registry.approve_upgrade(u)
            print(f"  {'OK' if ok else 'FAILED'}")
    continue

# /models — show current assignments
if query == "/models":
    print(f"  Orchestrator: {registry.orchestrator_model} (locked)")
    for task, assignment in registry.select_workers().items():
        marker = "auto" if assignment.auto_selected else "fallback"
        print(f"  {task:20s} → {assignment.model_id:30s} "
              f"score={assignment.benchmark_score:.1f} [{marker}]")
    continue
```

### Task 5: Startup Check (Optional)

On startup, optionally check for upgrades (configurable):

```yaml
# configs/models.yaml
auto_select: true
check_upgrades_on_start: false  # set true to check every time LORE starts
upgrade_check_interval_hours: 168  # default: weekly
```

If enabled, LORE prints upgrade recommendations on startup but doesn't download without approval.

### Task 6: Classifier (`src/lore/classifier.py`)

Model-based task classifier using specialist model. Same as previous design — uses Falcon-H1 for NLU classification, falls back to heuristic.

### Task 7: Wire into Orchestrator

Same as previous design:
- Replace `complexity.estimate()` with `TaskClassifier.classify()`
- Pass classifier hints to decomposer
- Skip orchestration on fallback plan
- Use registry for model resolution per subtask

### Task 8: Config (`configs/models.yaml`)

```yaml
# The orchestrator model: user-set, NEVER auto-changed
orchestrator:
  model: Ornith-1.0-9B
  role: primary
  port: 19000
  path: models/ornith-1.0-9b-Q4_K_M.gguf
  source: deepreinforce-ai/Ornith-1.0-9B-GGUF
  locked: true

# Auto-selection: LORE picks worker models from live benchmarks
auto_select: true
size_class: medium  # "small" (<3B) or "medium" (<10B)
check_upgrades_on_start: false
upgrade_check_interval_hours: 168  # weekly

# Leaderboard scanner config
leaderboard:
  cache_ttl_hours: 24
  hf_token: null
  min_improvement_pct: 5.0  # only flag >5% improvements

# Fallback task mapping (used when auto_select=false or no scores available)
task_mapping:
  classification: specialist
  extraction: specialist
  summarization: specialist
  code_gen: primary
  testing: primary
  documentation: specialist
  math: primary
  planning: primary
  review: primary
```

### Task 9: Auto-Select Script

```python
#!/usr/bin/env python3
"""Discover best models for each LORE task type from HuggingFace.

Scans leaderboard, scores models per task, shows rankings + GGUF availability.

Usage:
    PYTHONPATH=src python scripts/auto_select_models.py [--size small|medium]
"""
```

### Task 10: Tests

- `tests/test_leaderboard.py` — mock HF API, test scoring, test upgrade detection
- `tests/test_registry.py` — test auto-selection, test orchestrator lock, test approve/download
- `tests/test_classifier.py` — mock server, test classification, test fallback
- Update `tests/test_orchestrator.py` — mock registry + classifier

## Files

| File | Action |
|------|--------|
| `src/lore/leaderboard.py` | Create — HF scanner, upgrade detection |
| `src/lore/registry.py` | Rewrite — auto-select from local + upgrade flow |
| `src/lore/classifier.py` | Create — model-based classifier |
| `src/lore/orchestrator.py` | Modify — use registry + classifier |
| `src/lore/decomposer.py` | Modify — accept hints |
| `configs/models.yaml` | Rewrite — orchestrator lock, auto-select, leaderboard |
| `scripts/auto_select_models.py` | Create — standalone discovery |
| `tests/test_leaderboard.py` | Create |
| `tests/test_registry.py` | Create |
| `tests/test_classifier.py` | Create |
| `tests/test_orchestrator.py` | Modify |

## The Full Flow

```
1. LORE starts
2. Registry selects best LOCAL models for each task type (from installed GGUFs)
3. If check_upgrades_on_start=true:
   a. Scanner queries HF leaderboard parquet
   b. Filters: size < 10B, GGUF available, not installed
   c. Scores per task type
   d. Compares against installed models
   e. Prints table: "Better model found for code_gen: Qwen3.5-9B (+12%)"
   f. Asks user: "Download? [y/N]"
   g. If yes → download GGUF → update assignment
4. User works normally
5. /upgrades command → re-check anytime
6. /models command → show current assignments
```

## Constraints

- **Orchestrator model NEVER auto-changed.** `locked: true`.
- **No download without user approval.** Always prompt.
- **Only install models with GGUF.** No point showing a model we can't run.
- **169 tests must pass.**
- **HF API may be slow/rate-limited.** Cache parquet for 24h.
