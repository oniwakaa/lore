#!/usr/bin/env python3
"""Train the TF-IDF router on labeled data."""
import sys
from pathlib import Path
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from lore.router import Router
from lore.config import LoreConfig

def main():
    cfg = LoreConfig.load()
    router_cfg = cfg.router
    metrics = Router.train(
        router_cfg["training_data_path"],
        router_cfg["model_path"],
    )
    print(f"Router trained: accuracy={metrics['accuracy']:.3f}")
    print(f"  classes: {metrics['classes']}")
    print(f"  train: {metrics['train_size']}, test: {metrics['test_size']}")
    if metrics["accuracy"] < 0.80:
        print("  WARNING: accuracy below 80% target")
        sys.exit(1)

if __name__ == "__main__":
    main()
