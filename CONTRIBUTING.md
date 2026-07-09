# Contributing to LORE

Thanks for your interest in LORE! This project is building local AI orchestration for edge devices — making multiple small models work together to match the quality of much larger models, all within 16 GB of RAM.

## Quick Start

```bash
# Clone
git clone https://github.com/oniwakaa/lore.git
cd lore

# Install
pip install -e ".[dev]"

# Run tests
PYTHONPATH=src python -m pytest tests/ -v
```

You do **not** need a Mac or GPU to contribute code. The test suite runs with mocks — real inference is only needed for benchmarks.

## What We Need Help With

### 🔥 High Priority

- **EAGLE-3 speculative decoding** — Test if Qwen3 EAGLE-3 checkpoints work on Ornith (Qwen3.5 arch). Could give 2-3× speedup. See `docs/` for architecture notes.
- **SWE-bench integration** — Run LORE against SWE-bench Verified subset. See `scripts/benchmark_orchestration.py` for the benchmark harness.
- **HumanEval benchmark** — Extend the benchmark script to run LORE against all 164 HumanEval tasks with programmatic test execution.

### 🟡 Medium Priority

- **Model support** — Add support for new GGUF models (Mamba-3 when available, Qwen3.6 variants)
- **Windows/Linux testing** — Primary target is macOS (Metal), but CPU inference should work everywhere
- **Documentation** — Architecture diagrams, setup guides, config explanations
- **Test coverage** — More edge cases in orchestrator, decomposer, and verifier

### 🟢 Good First Issues

Look for the [`good first issue`](https://github.com/oniwakaa/lore/labels/good%20first%20issue) label.

## Architecture Overview

```
User Request
    → Tool Attention (lazy schema loading)
    → Context Manager (budget, compression, memory)
    → Router (TF-IDF + LogReg, <1ms)
    → Orchestrator (classify → decompose → schedule → execute → aggregate)
    → Model Server (llama-server, primary 9B + specialist 1.5B)
```

Key modules:
- `src/lore/orchestrator.py` — Task decomposition and parallel execution
- `src/lore/router.py` — Non-LLM task routing
- `src/lore/context.py` — Dynamic context management
- `src/lore/leaderboard.py` — HuggingFace benchmark scanning for model upgrades
- `src/lore/registry.py` — Auto-select best local model per task type
- `src/lore/verifier.py` — JSON/code output validation and repair

## Development Workflow

1. **Fork** the repo
2. **Create a branch** from `main`: `git checkout -b feat/your-feature`
3. **Write tests first** (we practice TDD where practical)
4. **Implement** your change
5. **Run the full suite**: `PYTHONPATH=src python -m pytest tests/ -v`
6. **Commit** with conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `perf:`
7. **Open a PR** against `main`

### Commit Convention

```
feat: add Mamba-3 specialist model support
fix: decomposer JSON parsing fails on truncated output
docs: update architecture diagram with Phase 4 components
test: add integration tests for parallel wave execution
perf: cache leaderboard scores to reduce HF API calls
```

### Test Requirements

- All 237+ existing tests must pass
- New features need tests (aim for >80% coverage on new code)
- Run `PYTHONPATH=src python -m pytest tests/ -v --tb=short` before pushing

## Configuration

All config lives in `configs/`:

| File | Purpose |
|------|---------|
| `models.yaml` | Model paths, ports, engine settings |
| `orchestrator.yaml` | Classifier, decomposer, aggregation settings |
| `router.yaml` | Router training parameters |
| `memory.yaml` | Memory system settings |
| `compression.yaml` | LLMLingua-2 settings |

## Running Benchmarks

```bash
# Quick benchmark (10 tasks, ~5 min)
PYTHONPATH=src python scripts/benchmark_orchestration.py --quick

# HumanEval benchmark (164 tasks, ~1 hour)
PYTHONPATH=src python scripts/benchmark_orchestration.py --benchmark humaneval

# Smoke test (10 HumanEval tasks, ~10 min)
PYTHONPATH=src python scripts/benchmark_orchestration.py --benchmark humaneval --limit 10
```

## Code Style

- Python 3.11+ (type hints, dataclasses, `match` statements)
- No formatter enforced (be consistent with surrounding code)
- Docstrings on public methods
- Type hints on function signatures

## Questions?

Open a [Discussion](https://github.com/oniwakaa/lore/discussions) or file an [Issue](https://github.com/oniwakaa/lore/issues).

## License

MIT
