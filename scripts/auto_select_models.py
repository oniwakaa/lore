#!/usr/bin/env python3
"""Discover best models for each LORE task type from HuggingFace.

Scans leaderboard, scores models per task, shows rankings + GGUF availability.

Usage:
    PYTHONPATH=src python3 scripts/auto_select_models.py [--size small|medium]
"""
import argparse
import sys
import os

# Ensure src is on path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lore.leaderboard import LeaderboardScanner, TASK_BENCHMARKS, MAX_PARAMS_B


def main():
    parser = argparse.ArgumentParser(description="Discover best models from HF leaderboard")
    parser.add_argument("--size", choices=["small", "medium"], default="medium",
                        help="Size class: small (<3B) or medium (<10B)")
    parser.add_argument("--models-dir", default="models", help="Local models directory")
    parser.add_argument("--min-improvement", type=float, default=5.0,
                        help="Min improvement %% to flag as upgrade")
    args = parser.parse_args()

    max_params = 3.0 if args.size == "small" else MAX_PARAMS_B

    scanner = LeaderboardScanner({"cache_ttl_hours": 1})
    print(f"Scanning HuggingFace leaderboard (size class: {args.size}, max {max_params}B params)...")
    print()

    candidates = scanner._load_leaderboard_data()
    if not candidates:
        print("ERROR: No leaderboard data available. Check network/HF access.")
        sys.exit(1)

    # Filter by size
    viable = [c for c in candidates if c.params_b == 0 or c.params_b <= max_params]
    print(f"Found {len(viable)} models under {max_params}B params")

    # Score each per task type
    for c in viable:
        for task_type in TASK_BENCHMARKS:
            c.task_scores[task_type] = scanner._compute_task_score(c.scores, task_type)

    # Show top 3 per task type
    for task_type in TASK_BENCHMARKS:
        print(f"\n=== {task_type.upper()} ===")
        scored = [(c, c.task_scores[task_type]) for c in viable if c.task_scores[task_type] > 0]
        scored.sort(key=lambda x: -x[1])
        for c, score in scored[:3]:
            gguf = "GGUF available" if c.gguf_repo else ("installed" if c.is_installed else "no GGUF")
            print(f"  {c.model_id:45s} score={score:.1f}  [{gguf}]")

    # Check for upgrades against installed models
    print("\n=== UPGRADE CHECK ===")
    # Guess installed from local files
    from pathlib import Path
    installed: dict[str, str] = {}
    if Path(args.models_dir).exists():
        for f in Path(args.models_dir).glob("*.gguf"):
            name = f.stem
            for q in ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q4_0"]:
                name = name.replace(f"-{q}", "").replace(f"_{q}", "")
            installed["code_gen"] = name  # placeholder

    if installed:
        upgrades = scanner.scan_for_upgrades(installed, args.models_dir, args.min_improvement)
        if not upgrades:
            print("All installed models are current best for their tasks.")
        else:
            for u in upgrades:
                print(f"  {u.task_type}: {u.current_model} → {u.better_model.model_id} "
                      f"(+{u.improvement_pct:.1f}%, {u.better_model.gguf_size_gb:.1f} GB)")
    else:
        print("No installed models detected. Run /upgrades in LORE REPL after downloading models.")


if __name__ == "__main__":
    main()
