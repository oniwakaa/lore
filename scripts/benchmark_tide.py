#!/usr/bin/env python3
"""Evaluate TIDE (RightNow-AI/TIDE) early exit on the specialist model.

Usage: .venv/bin/python scripts/benchmark_tide.py

TIDE does per-token early exit by comparing hidden states across transformer
decoder layers and skipping ahead when a token's representation has "converged".
This assumes every layer is a stateless, parallelizable attention-style block —
skipping one token at one layer doesn't affect any other token at that layer.

Falcon-H1 (our specialist) is a hybrid SSM/Mamba model: most layers carry a
sequential recurrent state across the token dimension. Skipping a token through
an SSM layer doesn't just lose "extra refinement" (as it would in a transformer) —
it corrupts the recurrent state trajectory for every later token in that
sequence at that layer. This is a correctness issue, not a quality/speed
tradeoff, so this script refuses to run TIDE against hybrid_ssm architectures
and explains why instead of producing meaningless (or wrong) numbers.

Also: TIDE only wraps HuggingFace `transformers` AutoModelForCausalLM (PyTorch),
not GGUF/llama.cpp models, and its speed benchmarks are all CUDA (falls back to
slow pure-PyTorch CPU on Apple Silicon, no Metal/MPS kernels). Even for an
architecture TIDE does support, running it here would mean loading a second,
unquantized copy of the model outside our llama.cpp/GGUF serving stack.
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


def load_specialist_config() -> dict:
    cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    return cfg.get("specialist", {})


def main():
    specialist = load_specialist_config()
    arch = specialist.get("architecture", "unknown")
    name = specialist.get("name", "specialist")

    print(f"Specialist model: {name} (architecture: {arch})")

    if "ssm" in arch.lower() or "mamba" in arch.lower():
        print(
            "\nDECISION GATE: SKIP.\n"
            f"{name} is a hybrid SSM architecture. TIDE's per-token early exit\n"
            "assumes stateless attention-style layers; SSM layers carry sequential\n"
            "recurrent state that cannot be selectively skipped per-token without\n"
            "corrupting the state trajectory for the rest of the sequence. This is\n"
            "an architectural incompatibility, not a tunable quality/speed tradeoff.\n"
            "See docs/optimization-log.md for the full writeup."
        )
        sys.exit(0)

    try:
        import TIDE  # noqa: F401
    except ImportError:
        print(
            "\ntide-inference not installed and specialist is not SSM-based, but no\n"
            "GGUF/llama.cpp integration exists for TIDE (HF transformers only).\n"
            "Install with `pip install tide-inference` and load the specialist via\n"
            "transformers.AutoModelForCausalLM to proceed manually."
        )
        sys.exit(0)

    print("Specialist architecture is not SSM — manual TIDE calibration/benchmark "
          "would be needed here (not implemented, no current specialist requires it).")


if __name__ == "__main__":
    main()
