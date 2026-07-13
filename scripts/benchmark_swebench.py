#!/usr/bin/env python3
"""SWE-bench Verified benchmark: run LORE orchestration on real GitHub issues.

Usage:
  PYTHONPATH=src python scripts/benchmark_swebench.py --smoke          # 3-task smoke test
  PYTHONPATH=src python scripts/benchmark_swebench.py --limit 20       # 20 diverse tasks
  PYTHONPATH=src python scripts/benchmark_swebench.py --task psf__requests-1142  # single task
"""
import argparse
import difflib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lore.config import LoreConfig
from lore.context import ContextManager
from lore.memory import HierarchicalMemory
from lore.models import ModelServer
from lore.classifier import TaskClassifier
from lore.orchestrator import Orchestrator
from lore.router import Router
from lore.repo_tools import RepoContext

REPOS_DIR = ROOT / "benchmarks" / "repos"
RESULTS_DIR = ROOT / "benchmarks" / "results"
PREDICTIONS_PATH = RESULTS_DIR / "swebench_predictions.jsonl"
RESULTS_PATH = RESULTS_DIR / "swebench_results.json"
SMOKE_TASKS_PATH = ROOT / "benchmarks" / "eval_tasks" / "swebench_smoke.json"
SUBSET_PATH = ROOT / "benchmarks" / "eval_tasks" / "swebench_subset.json"

PRIMARY_PORT = 19000
SPECIALIST_PORT = 19001
EMBED_PORT = 19002

logger = logging.getLogger("swebench_bench")

# ─── Task Loading ──────────────────────────────────────────────────────

def load_tasks_from_hf(n: int = 20, diverse: bool = True) -> list[dict]:
    """Load n tasks from SWE-bench Verified, picking diverse repos."""
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    if not diverse:
        tasks = []
        for row in ds:
            if len(tasks) >= n:
                break
            tasks.append(_row_to_task(row))
        return tasks

    # Pick diverse: spread across repos, mix of difficulty
    by_repo: dict[str, list] = {}
    for row in ds:
        by_repo.setdefault(row["repo"], []).append(row)

    # Sort repos by count (most first), pick from each round-robin
    sorted_repos = sorted(by_repo.keys(), key=lambda r: len(by_repo[r]), reverse=True)
    tasks = []
    # Round 1: pick one from each repo (prefer simpler patches)
    for repo in sorted_repos:
        if len(tasks) >= n:
            break
        # Pick simplest task from this repo (shortest patch)
        candidates = sorted(by_repo[repo], key=lambda r: len(r["patch"]))
        if candidates:
            tasks.append(_row_to_task(candidates[0]))

    # If not enough, round 2: pick more from larger repos
    if len(tasks) < n:
        for repo in sorted_repos:
            if len(tasks) >= n:
                break
            for row in by_repo[repo][1:]:  # skip first (already picked)
                if len(tasks) >= n:
                    break
                tasks.append(_row_to_task(row))

    return tasks[:n]


def _row_to_task(row) -> dict:
    return {
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "problem_statement": row["problem_statement"],
        "patch": row["patch"],
        "test_patch": row["test_patch"],
        "FAIL_TO_PASS": row["FAIL_TO_PASS"],
        "PASS_TO_PASS": row["PASS_TO_PASS"][:10] if row["PASS_TO_PASS"] else [],
        "version": row["version"],
        "difficulty": row.get("difficulty", "unknown"),
        "hints_text": row.get("hints_text", ""),
        "environment_setup_commit": row.get("environment_setup_commit", ""),
    }


def load_smoke_tasks() -> list[dict]:
    """Load 3-task smoke test subset."""
    if SMOKE_TASKS_PATH.exists():
        data = json.loads(SMOKE_TASKS_PATH.read_text())
        return data.get("tasks", [])
    return []


# ─── Repo Management ───────────────────────────────────────────────────

def clone_repo(repo: str, base_commit: str) -> Path | None:
    """Clone repo and checkout base_commit. Returns path or None on failure."""
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    repo_dir = REPOS_DIR / repo.replace("/", "__")

    if not repo_dir.exists():
        url = f"https://github.com/{repo}.git"
        logger.info(f"Cloning {repo} from {url}...")
        try:
            subprocess.run(
                ["git", "clone", "--quiet", url, str(repo_dir)],
                timeout=300, check=True, capture_output=True,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.error(f"Clone failed for {repo}: {e}")
            return None

    # Checkout base_commit
    try:
        subprocess.run(
            ["git", "-C", str(repo_dir), "checkout", "--quiet", "-f", base_commit],
            timeout=60, check=True, capture_output=True,
        )
        # Clean any untracked files from previous runs
        subprocess.run(
            ["git", "-C", str(repo_dir), "clean", "-fdq"],
            timeout=30, check=True, capture_output=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.error(f"Checkout failed for {repo}@{base_commit[:12]}: {e}")
        return None

    return repo_dir


# ─── LORE Pipeline ─────────────────────────────────────────────────────

def is_healthy(port: int) -> bool:
    try:
        return requests.get(f"http://127.0.0.1:{port}/health", timeout=3).status_code == 200
    except Exception:
        return False


def ensure_servers(server: ModelServer) -> bool:
    """Ensure all servers are running. Start if needed."""
    # Start embeddings + specialist + primary (order matters for memory)
    for role, port in [("embeddings", EMBED_PORT), ("specialist", SPECIALIST_PORT), ("primary", PRIMARY_PORT)]:
        if is_healthy(port):
            logger.info(f"{role} server already running")
            continue
        try:
            logger.info(f"Starting {role} server...")
            server.start_model(role)
        except Exception as e:
            logger.error(f"Failed to start {role}: {e}")
            if role == "primary":
                return False
    return is_healthy(PRIMARY_PORT)


def build_orchestrator(server: ModelServer) -> tuple[Orchestrator, callable]:
    """Wire Orchestrator like benchmark_orchestration.py does."""
    cfg = LoreConfig.load()
    router = Router.load(
        cfg.router.get("model_path", "configs/router_model.joblib"),
        confidence_threshold=cfg.router.get("confidence_threshold", 0.70),
    )
    system_prompt = "You are a helpful assistant. Answer concisely and accurately. /no_think"
    tokenizer_source = cfg.models.get("defaults", {}).get("tokenizer_source", "local")
    tokenizer_repo = cfg.models.get("primary", {}).get("source", "")
    if tokenizer_repo.endswith("-GGUF"):
        tokenizer_repo = tokenizer_repo[:-len("-GGUF")]
    memory = HierarchicalMemory(cfg.memory, server)
    ctx = ContextManager(cfg.context, server, system_prompt=system_prompt,
                         tokenizer_source=tokenizer_source,
                         tokenizer_repo=tokenizer_repo or None,
                         memory=memory)
    orch_cfg_path = ROOT / "configs/orchestrator.yaml"
    orch_cfg = yaml.safe_load(orch_cfg_path.read_text()) if orch_cfg_path.exists() else {}
    classifier_cfg = orch_cfg.get("classifier", {})
    classifier = TaskClassifier(server, classifier_cfg) if classifier_cfg.get("enabled", False) else None
    orchestrator = Orchestrator(server, router, memory, orch_cfg, ctx=ctx,
                                classifier=classifier)
    dispatch_fn = make_dispatch_fn(server, router, ctx, memory)
    return orchestrator, dispatch_fn


def make_dispatch_fn(server, router, ctx, memory):
    """Minimal dispatch closure matching cli.py _dispatch."""
    def dispatch_fn(query, json_mode=False):
        t0 = time.time()
        try:
            route, confidence = router.classify(query)
            model = "primary" if route == "PRIMARY" else "specialist"
        except Exception:
            route, confidence, model = "PRIMARY", 0.0, "primary"
        ctx.add_message("user", query)
        messages = ctx.build_prompt(query=query)
        try:
            result = server.chat(model, messages, max_tokens=2048, temperature=0.0)
            content = result["choices"][0]["message"]["content"]
            success = True
        except Exception as e:
            if model == "specialist":
                result = server.chat("primary", messages, max_tokens=2048, temperature=0.0)
                content = result["choices"][0]["message"]["content"]
                success = True
            else:
                content = f"Error: {e}"
                success = False
        ctx.add_message("assistant", content)
        return {"route": route, "confidence": confidence, "model": model,
                "content": content, "success": success,
                "latency_ms": (time.time() - t0) * 1000}
    return dispatch_fn


# ─── SWE-bench Task Execution ──────────────────────────────────────────

def build_swebench_prompt(task: dict, repo_path: Path) -> str:
    """Build the prompt LORE receives for a SWE-bench task.

    Pre-injects relevant file context (grep + read) so the model has
    context even if it doesn't use tools during orchestration.
    """
    # Pre-explore: extract keywords from issue, search repo, read top files
    context = _pre_explore_repo(task, repo_path)

    return (
        f"/no_think\n"
        f"Fix the following issue in the {task['repo']} codebase.\n"
        f"The repository is cloned at: {repo_path}\n\n"
        f"## Issue\n\n{task['problem_statement']}\n\n"
        f"{context}\n\n"
        f"## Instructions\n\n"
        f"1. Use READ_FILE and SEARCH tools to explore the codebase further if needed.\n"
        f"2. Read the relevant files to get the EXACT code you need to change.\n"
        f"3. Write your fix using SEARCH/REPLACE blocks (NOT unified diffs).\n\n"
        f"For each file you need to change, output:\n"
        f"path/to/file.py\n<<<<<<< SEARCH\n"
        f"exact lines from the file that need changing\n"
        f"=======\n"
        f"replacement lines\n>>>>>>> REPLACE\n\n"
        f"The SEARCH section must exactly match the file content (copy it from READ_FILE output).\n"
        f"Include enough context lines to uniquely identify the location.\n"
        f"Output the blocks directly — no ```diff fences needed."
    )


def _pre_explore_repo(task: dict, repo_path: Path, max_files: int = 3,
                      max_lines_per_file: int = 80) -> str:
    """Pre-explore repo: grep for keywords from issue, read top matching files.

    Returns a context string with file contents and line numbers.
    """
    issue = task["problem_statement"]
    # Extract potential keywords: class names, function names, file paths
    keywords = set()
    # File paths mentioned in issue
    for m in re.finditer(r'(\w+\.py)', issue):
        keywords.add(m.group(1))
    # Class/function names (CamelCase or snake_case identifiers)
    for m in re.finditer(r'\b([A-Z][a-zA-Z]+[A-Z][a-zA-Z]+)\b', issue):
        keywords.add(m.group(1))
    for m in re.finditer(r'\b(def\s+(\w+)|class\s+(\w+))\b', issue):
        keywords.add(m.group(2) or m.group(3))
    # Error-related keywords
    for m in re.finditer(r'(\w+Error|\w+Exception)', issue):
        keywords.add(m.group(1))

    if not keywords:
        # Fallback: use repo name as keyword
        keywords.add(task["repo"].split("/")[-1])

    # Search repo for keywords
    found_files = {}  # file_path -> match_count
    for kw in list(keywords)[:5]:
        try:
            result = subprocess.run(
                ["grep", "-rn", "--include", "*.py", "-l", kw, str(repo_path)],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.strip().split("\n"):
                if line and ".git" not in line:
                    rel = line.replace(str(repo_path) + "/", "")
                    found_files[rel] = found_files.get(rel, 0) + 1
        except Exception:
            pass

    # Sort by match count, take top files
    top_files = sorted(found_files.items(), key=lambda x: -x[1])[:max_files]

    if not top_files:
        # Fallback: list main package directory
        try:
            pkg_name = task["repo"].split("/")[-1].replace("-", "_")
            pkg_dir = repo_path / pkg_name
            if pkg_dir.exists():
                result = subprocess.run(
                    ["find", str(pkg_dir), "-name", "*.py", "-not", "-path", "*/test*"],
                    capture_output=True, text=True, timeout=10,
                )
                files = result.stdout.strip().split("\n")[:max_files]
                top_files = [(f.replace(str(repo_path) + "/", ""), 1) for f in files if f]
        except Exception:
            pass

    # Read top files with line numbers
    context_parts = ["## Pre-explored Files (with line numbers)\n"]
    for file_path, _ in top_files:
        fp = repo_path / file_path
        if not fp.exists() or not fp.is_file():
            continue
        try:
            lines = fp.read_text(errors="replace").split("\n")[:max_lines_per_file]
            numbered = "\n".join(f"{i+1:4d}: {line}" for i, line in enumerate(lines))
            context_parts.append(f"### {file_path}\n```python\n{numbered}\n```")
        except Exception:
            pass

    return "\n\n".join(context_parts) if len(context_parts) > 1 else ""


def extract_patch(content: str, repo_path: Path | None = None) -> str:
    """Extract patch from model output. Tries SEARCH/REPLACE first, then unified diff.

    If repo_path is provided, SEARCH/REPLACE blocks are applied to actual files
    to generate a correct unified diff with proper line numbers for git apply.
    """
    # Strategy 1: SEARCH/REPLACE blocks
    sr_patch = _extract_search_replace(content, repo_path)
    if sr_patch:
        return sr_patch

    # Strategy 2: look for ```diff block
    blocks = re.findall(r'```(?:diff|patch)?\s*\n(.*?)```', content, re.DOTALL)
    for block in blocks:
        if "---" in block and "+++" in block and "@@" in block:
            return _clean_patch(block.strip())

    # Strategy 3: look for diff-style content (--- / +++ / @@)
    lines = content.split("\n")
    diff_start = None
    for i, line in enumerate(lines):
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            diff_start = i
            break
    if diff_start is not None:
        diff_lines = lines[diff_start:]
        return _clean_patch("\n".join(diff_lines).strip())

    # Strategy 4: look for any code block with file paths
    for block in blocks:
        if "def " in block or "class " in block or "import " in block:
            return block.strip()

    return ""


def _extract_search_replace(content: str, repo_path: Path | None = None) -> str:
    """Extract SEARCH/REPLACE blocks and convert to unified diff format.

    If repo_path is provided, blocks are applied to actual file content and
    a proper unified diff is generated with correct line numbers for git apply.
    Otherwise, falls back to diffing search vs replace text (line numbers may
    be wrong — only useful for content-based application).
    """
    from lore.search_replace import parse_edit_blocks, apply_edit_blocks
    from collections import defaultdict

    blocks = parse_edit_blocks(content)
    if not blocks:
        return ""

    # Group blocks by file (multiple edits to same file must be applied sequentially)
    by_file: dict[str, list] = defaultdict(list)
    for filepath, search_text, replace_text in blocks:
        by_file[filepath].append((filepath, search_text, replace_text))

    patches = []
    for filepath, file_blocks in by_file.items():
        if repo_path is not None:
            # Apply blocks to actual file, diff original vs modified
            fp = repo_path / filepath
            if not fp.exists():
                continue
            original = fp.read_text()
            modified = apply_edit_blocks(original, file_blocks)
            if modified is None:
                continue
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile=f"a/{filepath}",
                tofile=f"b/{filepath}",
            )
        else:
            # Fallback: diff search vs replace text (line numbers at 1)
            search_text = file_blocks[0][1]
            replace_text = file_blocks[0][2]
            diff = difflib.unified_diff(
                search_text.splitlines(keepends=True),
                replace_text.splitlines(keepends=True),
                fromfile=f"a/{filepath}",
                tofile=f"b/{filepath}",
            )
        diff_str = "".join(diff)
        if diff_str:
            patches.append(diff_str)

    return "\n".join(patches) if patches else ""


def _clean_patch(patch: str) -> str:
    """Fix common LLM patch issues: strip git metadata, fix line counts."""
    lines = patch.split("\n")
    # Strip diff --git and index lines (they cause git apply to fail with wrong hashes)
    cleaned = []
    for line in lines:
        if line.startswith("diff --git") or line.startswith("index "):
            continue
        cleaned.append(line)
    patch = "\n".join(cleaned)

    # Fix hunk headers to match actual content
    lines = patch.split("\n")
    fixed = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@"):
            m = re.match(r'^@@ -(\d+),(\d+) \+(\d+),(\d+) @@', line)
            if m:
                old_start, old_count, new_start, new_count = map(int, m.groups())
                hunk_body = []
                i += 1
                actual_old = 0
                actual_new = 0
                while i < len(lines) and not lines[i].startswith("@@") and not lines[i].startswith("--- "):
                    body_line = lines[i]
                    if body_line.startswith(" "):
                        hunk_body.append(body_line)
                        actual_old += 1
                        actual_new += 1
                    elif body_line.startswith("-"):
                        hunk_body.append(body_line)
                        actual_old += 1
                    elif body_line.startswith("+"):
                        hunk_body.append(body_line)
                        actual_new += 1
                    elif body_line == "":
                        hunk_body.append(" ")
                        actual_old += 1
                        actual_new += 1
                    else:
                        break
                    i += 1
                fixed_header = f"@@ -{old_start},{actual_old} +{new_start},{actual_new} @@"
                fixed.append(fixed_header)
                fixed.extend(hunk_body)
                continue
        fixed.append(line)
        i += 1
    return "\n".join(fixed)


def run_swebench_task(task: dict, orchestrator: Orchestrator,
                      dispatch_fn, server: ModelServer) -> dict:
    """Run one SWE-bench task through LORE orchestration."""
    t0 = time.time()
    instance_id = task["instance_id"]

    # 1. Clone repo at base_commit
    repo_path = clone_repo(task["repo"], task["base_commit"])
    if repo_path is None:
        return {"instance_id": instance_id, "resolved": False,
                "error": "repo clone failed", "latency_s": 0}

    # 2. Build prompt
    prompt = build_swebench_prompt(task, repo_path)

    # 3. Create RepoContext
    try:
        repo_ctx = RepoContext(str(repo_path))
    except Exception as e:
        return {"instance_id": instance_id, "resolved": False,
                "error": f"repo context failed: {e}", "latency_s": 0}

    # 4. Run through orchestrator with repo_context
    orchestrator.reset_state()
    try:
        result = orchestrator.process(prompt, json_mode=False,
                                      dispatch_fn=dispatch_fn,
                                      repo_context=repo_ctx)
        content = result.get("content", "")
        orchestrated = bool(result.get("orchestrated", False))
        subtasks = int(result.get("subtasks_completed", 0))
        success = bool(result.get("success", False))
    except Exception as e:
        logger.error(f"Orchestration failed for {instance_id}: {e}")
        content = ""
        orchestrated = False
        subtasks = 0
        success = False

    latency = time.time() - t0

    # 5. Extract patch — prefer last subtask output over aggregated content
    #    (aggregation may truncate or mangle the diff format)
    subtask_results = result.get("subtask_results", {})
    # Try s2 (patch subtask in 2-task plan), then s3 (in 3-task plan), then aggregated
    swebench_content = subtask_results.get("s2", "") or subtask_results.get("s3", "")
    patch = extract_patch(swebench_content, repo_path) if swebench_content else ""
    if not patch:
        patch = extract_patch(content, repo_path)
    patch_extracted = bool(patch.strip())

    # 6. Evaluate patch
    eval_result = evaluate_patch(task, repo_path, patch)

    return {
        "instance_id": instance_id,
        "repo": task["repo"],
        "difficulty": task.get("difficulty", "unknown"),
        "resolved": eval_result["resolved"],
        "patch_extracted": patch_extracted,
        "patch_applies": eval_result["patch_applies"],
        "tests_passed": eval_result["tests_passed"],
        "tests_run": eval_result["tests_run"],
        "orchestrated": orchestrated,
        "subtasks_completed": subtasks,
        "latency_s": round(latency, 1),
        "content_length": len(content),
        "patch_length": len(patch),
        "patch": patch,
        "error": eval_result.get("error", ""),
        "metrics": result.get("metrics", {}) if success else {},
    }


# ─── Patch Evaluation (no Docker) ──────────────────────────────────────

def _apply_by_content(repo_path: Path, patch: str) -> bool:
    """Apply patch by finding old lines in file and replacing with new lines.

    Bypasses line number issues by searching for context+removed lines
    in the actual file content. Falls back if exact match not found.
    """
    # Parse diff: extract file paths and hunks
    hunks = _parse_diff_hunks(patch)
    if not hunks:
        return False

    for file_path, old_lines, new_lines in hunks:
        # Normalize: strip leading space (context), - (removed), + (added)
        # old_lines = list of lines that should be in the file (context + removed)
        # new_lines = list of lines to replace them with (context + added)
        old_text = "\n".join(l[1:] if l and l[0] in " -+" else l for l in old_lines)
        new_text = "\n".join(l[1:] if l and l[0] in " -+" else l for l in new_lines)

        fp = repo_path / file_path
        if not fp.exists():
            # Try without a/ prefix
            fp = repo_path / file_path.removeprefix("a/").removeprefix("b/")
        if not fp.exists():
            return False

        content = fp.read_text()
        # Try exact match first
        if old_text in content:
            content = content.replace(old_text, new_text, 1)
            fp.write_text(content)
            continue

        # Try fuzzy: remove extra whitespace and try again
        old_stripped = _strip_ws(old_text)
        content_stripped = _strip_ws(content)
        if old_stripped in content_stripped:
            # Find the position and replace in original content
            pos = content_stripped.find(old_stripped)
            # This is approximate — may not be perfect
            # Find the actual position in original content
            lines = content.split("\n")
            old_lines_stripped = [l.strip() for l in old_text.split("\n")]
            for i in range(len(lines) - len(old_lines_stripped) + 1):
                if all(lines[i + j].strip() == old_lines_stripped[j] for j in range(len(old_lines_stripped))):
                    lines[i:i + len(old_lines_stripped)] = new_text.split("\n")
                    fp.write_text("\n".join(lines))
                    break
            else:
                return False
            continue

        return False  # couldn't find context in file

    return True


def _parse_diff_hunks(patch: str) -> list[tuple[str, list[str], list[str]]]:
    """Parse diff into (file_path, old_lines, new_lines) tuples."""
    hunks = []
    lines = patch.split("\n")
    i = 0
    current_file = None
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- a/"):
            current_file = line[6:].strip()
            i += 1
            if i < len(lines) and lines[i].startswith("+++ b/"):
                i += 1
            continue
        if line.startswith("@@") and current_file:
            # Collect hunk body
            old_lines = []
            new_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("@@") and not lines[i].startswith("--- "):
                body = lines[i]
                if body.startswith(" "):
                    old_lines.append(body)
                    new_lines.append(body)
                elif body.startswith("-"):
                    old_lines.append(body)
                elif body.startswith("+"):
                    new_lines.append(body)
                elif body == "":
                    old_lines.append(" ")
                    new_lines.append(" ")
                i += 1
            hunks.append((current_file, old_lines, new_lines))
            continue
        i += 1
    return hunks


def _strip_ws(text: str) -> str:
    """Strip whitespace from each line for fuzzy matching."""
    return "\n".join(l.strip() for l in text.split("\n"))


def evaluate_patch(task: dict, repo_path: Path, patch: str) -> dict:
    """Evaluate a generated patch: apply + run tests.

    Custom evaluation without Docker:
    1. Apply model patch to repo
    2. Apply test_patch (from SWE-bench) to get the test changes
    3. Run FAIL_TO_PASS tests
    4. Check if they pass
    """
    if not patch.strip():
        return {"resolved": False, "patch_applies": False, "tests_passed": 0,
                "tests_run": 0, "error": "no patch extracted"}

    # 1. Try to apply model patch (try multiple strategies for lenient application)
    patch_file = repo_path / "_model_patch.diff"
    patch_file.write_text(patch + "\n")
    applied = False
    apply_error = ""

    for apply_flags in [[], ["--recount"], ["--ignore-whitespace"], ["--3way"]]:
        try:
            check_args = ["git", "-C", str(repo_path), "apply", "--check"] + apply_flags + [str(patch_file)]
            result = subprocess.run(check_args, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                actual_args = ["git", "-C", str(repo_path), "apply"] + apply_flags + [str(patch_file)]
                subprocess.run(actual_args, capture_output=True, text=True, timeout=30, check=True)
                applied = True
                break
            apply_error = result.stderr[:200]
        except Exception as e:
            apply_error = str(e)[:200]

    if not applied:
        # Last resort: try patch command (most lenient)
        try:
            result = subprocess.run(
                ["patch", "-p1", "--fuzz=3", "--forward", "-i", str(patch_file)],
                cwd=str(repo_path), capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                applied = True
            else:
                apply_error = (result.stdout + result.stderr)[-200:]
        except Exception as e:
            apply_error = str(e)[:200]

    if not applied:
        # Content-based application: find old lines in file, replace with new
        try:
            if _apply_by_content(repo_path, patch):
                applied = True
        except Exception as e:
            apply_error = f"content apply failed: {e}"

    if not applied:
        return {"resolved": False, "patch_applies": False, "tests_passed": 0,
                "tests_run": 0, "error": f"patch apply failed: {apply_error}"}

    # 2. Apply test patch
    test_patch_file = repo_path / "_test_patch.diff"
    test_patch_file.write_text(task["test_patch"] + "\n")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "apply", "--check", str(test_patch_file)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return {"resolved": False, "patch_applies": True, "tests_passed": 0,
                    "tests_run": 0, "error": f"test patch check failed: {result.stderr[:200]}"}
        subprocess.run(
            ["git", "-C", str(repo_path), "apply", str(test_patch_file)],
            capture_output=True, text=True, timeout=30, check=True,
        )
    except Exception as e:
        return {"resolved": False, "patch_applies": True, "tests_passed": 0,
                "tests_run": 0, "error": f"test patch apply failed: {e}"}

    # 3. Run FAIL_TO_PASS tests
    fail_to_pass = task["FAIL_TO_PASS"]
    if isinstance(fail_to_pass, str):
        fail_to_pass = json.loads(fail_to_pass)

    if not fail_to_pass:
        return {"resolved": False, "patch_applies": True, "tests_passed": 0,
                "tests_run": 0, "error": "no FAIL_TO_PASS tests specified"}

    # Try to install the package in a venv, then run tests
    venv_dir = repo_path / ".swebench_venv"
    tests_passed = 0
    tests_run = 0
    errors = []

    for test_spec in fail_to_pass:
        tests_run += 1
        test_result = run_single_test(repo_path, venv_dir, test_spec, task)
        if test_result["passed"]:
            tests_passed += 1
        else:
            errors.append(f"{test_spec}: {test_result['error'][:100]}")

    resolved = tests_passed == tests_run and tests_run > 0

    return {
        "resolved": resolved,
        "patch_applies": True,
        "tests_passed": tests_passed,
        "tests_run": tests_run,
        "error": "; ".join(errors[:3]) if errors else "",
    }


def run_single_test(repo_path: Path, venv_dir: Path, test_spec: str,
                    task: dict) -> dict:
    """Run a single test. Returns {passed, error}."""
    # Parse test spec: "path/to/test.py::TestClass::test_method"
    # or "path/to/test.py::test_function"
    parts = test_spec.split("::")

    # ponytail: skip venv setup, use system python directly with PYTHONPATH
    # This is a rough evaluation — real SWE-bench uses Docker with exact deps.
    # But it catches obvious patch correctness for simple repos.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_path)
    # Some repos need their package installed
    env["PYTHONUNBUFFERED"] = "1"

    # Build pytest args
    pytest_args = [sys.executable, "-m", "pytest", test_spec, "-x", "-q",
                   "--no-header", "--tb=short", "-p", "no:cacheprovider"]

    try:
        result = subprocess.run(
            pytest_args,
            cwd=str(repo_path),
            env=env,
            capture_output=True, text=True,
            timeout=60,
        )
        passed = result.returncode == 0
        error = result.stderr[-300:] if not passed else ""
        if not passed and "No module named" in result.stderr:
            # Try pip install -e . first
            install = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet", "--no-build-isolation"],
                cwd=str(repo_path),
                capture_output=True, text=True,
                timeout=120,
            )
            if install.returncode == 0:
                # Retry test
                result = subprocess.run(
                    pytest_args,
                    cwd=str(repo_path),
                    env=env,
                    capture_output=True, text=True,
                    timeout=60,
                )
                passed = result.returncode == 0
                error = result.stderr[-300:] if not passed else ""
            else:
                error = f"install failed: {install.stderr[-200:]}"
        return {"passed": passed, "error": error}
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "test timed out"}
    except Exception as e:
        return {"passed": False, "error": str(e)}


# ─── Prediction Saving ─────────────────────────────────────────────────

def save_prediction(task: dict, result: dict, patch: str):
    """Save prediction in SWE-bench format."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred = {
        "instance_id": task["instance_id"],
        "model_patch": patch,
        "model_name_or_path": "LORE-orchestrated-Ornith-9B",
    }
    with open(PREDICTIONS_PATH, "a") as f:
        f.write(json.dumps(pred) + "\n")


def save_results(results: list[dict], path: Path):
    """Save full results JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(results)
    resolved = sum(1 for r in results if r["resolved"])
    patches_extracted = sum(1 for r in results if r["patch_extracted"])
    patches_applied = sum(1 for r in results if r["patch_applies"])
    summary = {
        "total": total,
        "resolved": resolved,
        "resolved_rate": round(resolved / max(total, 1) * 100, 1),
        "patches_extracted": patches_extracted,
        "patches_applied": patches_applied,
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / max(total, 1), 1),
    }
    output = {"summary": summary, "results": results}
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SWE-bench Verified benchmark")
    parser.add_argument("--smoke", action="store_true",
                        help="Run 3-task smoke test")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of tasks to run (e.g. 20)")
    parser.add_argument("--task", type=str, default=None,
                        help="Run single task by instance_id")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Load tasks
    if args.smoke:
        tasks = load_smoke_tasks()
        if not tasks:
            print("No smoke tasks found. Run task selection first.", file=sys.stderr)
            sys.exit(1)
        results_path = RESULTS_DIR / "swebench_smoke_results.json"
    elif args.task:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        tasks = []
        for row in ds:
            if row["instance_id"] == args.task:
                tasks.append(_row_to_task(row))
                break
        if not tasks:
            print(f"Task {args.task} not found in SWE-bench Verified", file=sys.stderr)
            sys.exit(1)
        results_path = RESULTS_DIR / f"swebench_{args.task}.json"
    elif args.limit:
        tasks = load_tasks_from_hf(n=args.limit, diverse=True)
        # Save subset
        SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUBSET_PATH.write_text(json.dumps({"tasks": tasks}, indent=2))
        results_path = RESULTS_PATH
    else:
        parser.print_help()
        sys.exit(1)

    print(f"\nLoaded {len(tasks)} SWE-bench Verified tasks")
    for t in tasks:
        print(f"  {t['instance_id']} | {t['repo']} | {t.get('difficulty', '?')}")

    # Setup LORE
    cfg = LoreConfig.load()
    # ponytail: config has large native context → OOM on 16 GB. Override to 16K.
    cfg.models["primary"]["context"] = 16384
    cfg.models["specialist"]["context"] = 16384
    cfg.models["defaults"]["context_size"] = 16384
    server = ModelServer(cfg.models)

    print("\nChecking primary server...")
    if not ensure_servers(server):
        print("Primary server unavailable.", file=sys.stderr)
        sys.exit(1)
    print("  Primary: OK")

    orchestrator, dispatch_fn = build_orchestrator(server)

    # Clear previous predictions for this run
    if PREDICTIONS_PATH.exists():
        PREDICTIONS_PATH.unlink()

    # Run tasks
    results: list[dict] = []
    print(f"\nRunning {len(tasks)} tasks through LORE orchestration...\n")

    for i, task in enumerate(tasks):
        iid = task["instance_id"]
        print(f"  [{i+1}/{len(tasks)}] {iid} ", end="", flush=True)
        t0 = time.time()

        result = run_swebench_task(task, orchestrator, dispatch_fn, server)

        # Save prediction
        save_prediction(task, result, result.get("patch", ""))

        results.append(result)
        status = "RESOLVED" if result["resolved"] else "FAIL"
        print(f"{status} {result['latency_s']:.0f}s "
              f"patch={'Y' if result['patch_extracted'] else 'N'} "
              f"applies={'Y' if result['patch_applies'] else 'N'} "
              f"tests={result['tests_passed']}/{result['tests_run']}"
              + (f" err={result['error'][:60]}" if result.get("error") else ""))

        # Save incrementally
        save_results(results, results_path)

    # Print summary
    total = len(results)
    resolved = sum(1 for r in results if r["resolved"])
    print(f"\n{'='*60}")
    print(f"SWE-bench Verified Results")
    print(f"{'='*60}")
    print(f"Tasks: {total}")
    print(f"Resolved: {resolved}/{total} ({resolved/max(total,1)*100:.1f}%)")
    print(f"Patches extracted: {sum(1 for r in results if r['patch_extracted'])}/{total}")
    print(f"Patches applied: {sum(1 for r in results if r['patch_applies'])}/{total}")
    print(f"Avg latency: {sum(r['latency_s'] for r in results)/max(total,1):.1f}s")
    print(f"\nResults saved to: {results_path}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")

    # Comparison table
    print(f"\n{'='*60}")
    print(f"Comparison vs Published Scores")
    print(f"{'='*60}")
    print(f"  Ornith-1.0-9B (published): 69.4%")
    print(f"  LORE orchestrated 9B:      {resolved/max(total,1)*100:.1f}%")
    delta = resolved / max(total, 1) * 100 - 69.4
    sign = "+" if delta >= 0 else ""
    print(f"  Delta:                      {sign}{delta:.1f} pp")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
