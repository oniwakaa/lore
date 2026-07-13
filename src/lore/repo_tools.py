"""Repository exploration tools for SWE-bench tasks.

Gives workers the ability to read files, search code, and list directories
in a cloned repository during subtask execution. Used by the tool-use loop
in Worker.run_with_tools().
"""
import os
import re
import subprocess
from pathlib import Path


class RepoContext:
    """Lightweight repo exploration tools. No external deps, stdlib + grep only."""

    def __init__(self, repo_path: str):
        self.path = Path(repo_path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Repo not found: {repo_path}")

    @property
    def root(self) -> str:
        return str(self.path)

    def read_file(self, rel_path: str, max_lines: int = 100) -> str:
        """Read a file from the repo, return first max_lines."""
        fp = self.path / rel_path
        if not fp.exists() or not fp.is_file():
            return f"ERROR: File not found: {rel_path}"
        try:
            text = fp.read_text(errors="replace")
            lines = text.split("\n")
            if len(lines) > max_lines:
                return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines truncated)"
            return text
        except Exception as e:
            return f"ERROR reading {rel_path}: {e}"

    def search_files(self, pattern: str, file_glob: str = "*.py", max_results: int = 20) -> str:
        """Search for pattern in files matching glob. Returns matching lines with file:line."""
        try:
            cmd = ["grep", "-rn", "--include", file_glob, "-m", "5", pattern, str(self.path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
            # Strip the repo path prefix for readability
            cleaned = []
            for line in lines[:max_results]:
                # grep output: /path/to/file:line:content
                if line.startswith(str(self.path) + "/"):
                    line = line[len(str(self.path)) + 1:]
                cleaned.append(line)
            return "\n".join(cleaned) if cleaned else "No matches found."
        except subprocess.TimeoutExpired:
            return "ERROR: search timed out"
        except Exception as e:
            return f"ERROR: {e}"

    def list_files(self, rel_dir: str = ".", pattern: str = "*.py", max_results: int = 50) -> str:
        """List files in a directory matching pattern."""
        dp = self.path / rel_dir
        if not dp.exists() or not dp.is_dir():
            return f"ERROR: Directory not found: {rel_dir}"
        try:
            files = sorted(dp.rglob(pattern)) if pattern != "*" else sorted(dp.rglob("*"))
            files = [f for f in files if f.is_file() and ".git" not in str(f)]
            rel_files = [str(f.relative_to(self.path)) for f in files[:max_results]]
            return "\n".join(rel_files) if rel_files else "No files found."
        except Exception as e:
            return f"ERROR: {e}"

    def get_structure(self, max_depth: int = 2) -> str:
        """Get a tree view of the repo structure (top N levels)."""
        try:
            result = subprocess.run(
                ["find", str(self.path), "-maxdepth", str(max_depth),
                 "-not", "-path", "*/.git/*", "-not", "-path", "*/__pycache__/*",
                 "-not", "-path", "*/node_modules/*", "-type", "f"],
                capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
            cleaned = [l[len(str(self.path)) + 1:] for l in lines if l.startswith(str(self.path) + "/")]
            return "\n".join(cleaned[:100]) if cleaned else "Empty repo."
        except Exception as e:
            return f"ERROR: {e}"

    def execute_tool(self, tool_call: str) -> str:
        """Parse and execute a tool call string. Returns tool result.

        Supported formats:
            READ_FILE: path/to/file.py
            SEARCH: pattern
            SEARCH: pattern in *.py
            LIST_DIR: path/to/dir
            LIST_DIR: path/to/dir *.py
            REPO_STRUCTURE
        """
        tool_call = tool_call.strip()
        if tool_call.startswith("READ_FILE:"):
            return self.read_file(tool_call[len("READ_FILE:"):].strip())
        elif tool_call.startswith("SEARCH:"):
            rest = tool_call[len("SEARCH:"):].strip()
            # Check for " in " pattern: "pattern in *.py"
            m = re.match(r'^(.+?)\s+in\s+(\*\.\w+)$', rest)
            if m:
                return self.search_files(m.group(1), file_glob=m.group(2))
            return self.search_files(rest)
        elif tool_call.startswith("LIST_DIR:"):
            rest = tool_call[len("LIST_DIR:"):].strip()
            m = re.match(r'^(.+?)\s+(\*\.\w+)$', rest)
            if m:
                return self.list_files(m.group(1), pattern=m.group(2))
            return self.list_files(rest)
        elif tool_call.startswith("REPO_STRUCTURE"):
            return self.get_structure()
        else:
            return f"ERROR: Unknown tool call: {tool_call}"


TOOL_SYSTEM_PROMPT = """You have access to repository exploration tools. Use them to explore the codebase before writing your patch.

## Available Tools

Output any of these on a new line to call a tool. The result will be provided in the next message.

READ_FILE: path/to/file.py
  - Read a file from the repository (relative path, first 200 lines)

SEARCH: pattern
  - Search for a regex pattern across .py files. Returns matching lines with file:line.

SEARCH: pattern in *.txt
  - Search in specific file type

LIST_DIR: path/to/dir
  - List files in a directory

LIST_DIR: path/to/dir *.py
  - List .py files in a directory (recursive)

REPO_STRUCTURE
  - Get top-level repo file tree (2 levels deep)

## How to Use

1. Call REPO_STRUCTURE or LIST_DIR to understand the codebase layout.
2. Call SEARCH to find relevant code based on the issue description.
3. Call READ_FILE to read the relevant files.
4. Once you understand the code, produce your answer.
   For code changes, use SEARCH/REPLACE blocks (NOT unified diffs):
   path/to/file.py
   <<<<<<< SEARCH
   original code to find in the file
   =======
   replacement code
   >>>>>>> REPLACE

You can make up to 5 tool calls. After exploring, write your final answer.
If you already have enough context, skip tool calls and answer directly."""
