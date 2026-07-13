"""Tool proxy: executes tool calls from the coding agent's model locally.

When a model responds with tool_calls (OpenAI function-calling format),
the tool proxy executes them (read_file, search, list_dir) and returns
results as tool response messages. This keeps file exploration fast and
local, without the agent needing to handle tool execution itself.

Tool definitions use the OpenAI function-calling schema so any compatible
client (Cline, Continue, Aider) can use them transparently.
"""
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# OpenAI-compatible tool definitions
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns the file content as text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read."},
                    "max_lines": {"type": "integer", "description": "Maximum lines to return (default 200).", "default": 200},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for a pattern in files. Returns matching lines with file:line prefixes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for."},
                    "file_glob": {"type": "string", "description": "File glob to search in (default *.py).", "default": "*.py"},
                    "max_results": {"type": "integer", "description": "Maximum results to return (default 20).", "default": 20},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files in a directory. Returns file paths relative to the directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list (default '.').", "default": "."},
                    "pattern": {"type": "string", "description": "File glob pattern (default '*).", "default": "*"},
                    "max_results": {"type": "integer", "description": "Maximum files to return (default 50).", "default": 50},
                },
            },
        },
    },
]


def _safe_path(base: str, rel_path: str) -> Path | None:
    """Resolve rel_path under base, return None if it escapes."""
    try:
        base_p = Path(base).resolve()
        resolved = (base_p / rel_path).resolve()
        if not str(resolved).startswith(str(base_p)):
            return None
        return resolved
    except Exception:
        return None


def execute_tool(name: str, arguments: dict, repo_root: str = ".") -> str:
    """Execute a single tool call. Returns result as string.

    Supported tools: read_file, search_files, list_dir.
    """
    try:
        if name == "read_file":
            return _read_file(arguments, repo_root)
        elif name == "search_files":
            return _search_files(arguments, repo_root)
        elif name == "list_dir":
            return _list_dir(arguments, repo_root)
        else:
            return f"ERROR: Unknown tool: {name}"
    except Exception as e:
        return f"ERROR: {e}"


def _read_file(args: dict, repo_root: str) -> str:
    path = args.get("path", "")
    max_lines = args.get("max_lines", 200)
    fp = _safe_path(repo_root, path)
    if fp is None:
        return f"ERROR: Path escapes repo root: {path}"
    if not fp.exists() or not fp.is_file():
        return f"ERROR: File not found: {path}"
    text = fp.read_text(errors="replace")
    lines = text.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines truncated)"
    return text


def _search_files(args: dict, repo_root: str) -> str:
    pattern = args.get("pattern", "")
    file_glob = args.get("file_glob", "*.py")
    max_results = args.get("max_results", 20)
    try:
        cmd = ["grep", "-rn", "--include", file_glob, "-m", "5", pattern, str(repo_root)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        root = str(Path(repo_root).resolve())
        cleaned = []
        for line in lines[:max_results]:
            if line.startswith(root + "/"):
                line = line[len(root) + 1:]
            cleaned.append(line)
        return "\n".join(cleaned) if cleaned else "No matches found."
    except subprocess.TimeoutExpired:
        return "ERROR: search timed out"
    except Exception as e:
        return f"ERROR: {e}"


def _list_dir(args: dict, repo_root: str) -> str:
    rel_dir = args.get("path", ".")
    pattern = args.get("pattern", "*")
    max_results = args.get("max_results", 50)
    dp = _safe_path(repo_root, rel_dir)
    if dp is None:
        return f"ERROR: Path escapes repo root: {rel_dir}"
    if not dp.exists() or not dp.is_dir():
        return f"ERROR: Directory not found: {rel_dir}"
    files = sorted(dp.rglob(pattern)) if pattern != "*" else sorted(dp.rglob("*"))
    files = [f for f in files if f.is_file() and ".git" not in str(f)]
    root = Path(repo_root).resolve()
    rel_files = [str(f.relative_to(root)) for f in files[:max_results]]
    return "\n".join(rel_files) if rel_files else "No files found."


def execute_tool_calls(tool_calls: list[dict], repo_root: str = ".") -> list[dict]:
    """Execute a list of OpenAI-format tool_calls. Returns tool response messages.

    Each tool_call has: id, type, function: {name, arguments (JSON string)}.
    Returns list of {"role": "tool", "tool_call_id": ..., "content": ...}.
    """
    results = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        try:
            args = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        content = execute_tool(name, args, repo_root)
        results.append({
            "role": "tool",
            "tool_call_id": tc.get("id", ""),
            "content": content,
        })
        logger.debug(f"Tool {name} executed: {content[:80]}...")
    return results


def run_tool_loop(server, model: str, messages: list[dict],
                  tools: list[dict] | None = None, repo_root: str = ".",
                  max_rounds: int = 5, **chat_opts) -> dict:
    """Run a tool-use loop: model → tool_calls → execute → feed back → repeat.

    Returns the final chat completion response (OpenAI format) when the model
    stops requesting tools or max_rounds is reached.
    """
    tools = tools or TOOL_DEFINITIONS
    msgs = list(messages)

    for round_num in range(max_rounds):
        resp = server.chat(model, msgs, tools=tools, **chat_opts)
        choice = resp["choices"][0]
        msg = choice["message"]

        # If no tool_calls, return the final response
        if not msg.get("tool_calls"):
            return resp

        # Add the assistant message with tool_calls to the conversation
        msgs.append(msg)

        # Execute all tool calls
        tool_results = execute_tool_calls(msg["tool_calls"], repo_root)
        msgs.extend(tool_results)

        logger.info(f"Tool loop round {round_num + 1}: {len(msg['tool_calls'])} tools executed")

    # Max rounds reached — force a final response without tools
    msgs.append({"role": "user", "content": "Please provide your final answer based on the information gathered."})
    return server.chat(model, msgs, **chat_opts)


if __name__ == "__main__":
    # ponytail: self-check
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        # Create test files
        (Path(td) / "hello.py").write_text("def hello():\n    print('hello')\n")
        (Path(td) / "sub").mkdir()
        (Path(td) / "sub" / "world.py").write_text("x = 42\n")

        # Test read_file
        result = execute_tool("read_file", {"path": "hello.py"}, repo_root=td)
        assert "def hello" in result, f"read_file failed: {result}"

        # Test search_files
        result = execute_tool("search_files", {"pattern": "hello"}, repo_root=td)
        assert "hello.py" in result, f"search failed: {result}"

        # Test list_dir
        result = execute_tool("list_dir", {"path": "."}, repo_root=td)
        assert "hello.py" in result, f"list_dir failed: {result}"

        # Test tool_calls format
        results = execute_tool_calls([
            {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "hello.py"})}},
        ], repo_root=td)
        assert results[0]["role"] == "tool"
        assert "def hello" in results[0]["content"]

        print("tool_proxy self-check OK")
