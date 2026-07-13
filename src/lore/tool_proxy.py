"""Tool proxy: executes tool calls from the coding agent's model locally.

When a model responds with tool_calls (OpenAI function-calling format),
the tool proxy executes them (read_file, search, list_dir) and returns
results as tool response messages. This keeps file exploration fast and
local, without the agent needing to handle tool execution itself.

Tool definitions use the OpenAI function-calling schema so any compatible
client (Cline, Continue, Aider) can use them transparently.

Large file reads are compressed to structured summaries (imports, signatures,
relevant snippets) instead of raw dumps, keeping model context focused.
"""
import ast
import json
import logging
import re
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


def execute_tool(name: str, arguments: dict, repo_root: str = ".", ctx = None) -> str:
    """Execute a single tool call. Returns result as string.

    Supported tools: read_file, search_files, list_dir.
    """
    try:
        if name == "read_file":
            return _read_file(arguments, repo_root, ctx)
        elif name == "search_files":
            return _search_files(arguments, repo_root)
        elif name == "list_dir":
            return _list_dir(arguments, repo_root)
        else:
            return f"ERROR: Unknown tool: {name}"
    except Exception as e:
        return f"ERROR: {e}"


def _read_file(args: dict, repo_root: str, ctx = None) -> str:
    path = args.get("path", "")
    max_lines = args.get("max_lines", 200)
    fp = _safe_path(repo_root, path)
    if fp is None:
        return f"ERROR: Path escapes repo root: {path}"
    if not fp.exists() or not fp.is_file():
        return f"ERROR: File not found: {path}"
    text = fp.read_text(errors="replace")
    lines = text.split("\n")
    if ctx is not None:
        num_tokens = ctx.token_count(text)
        token_budget = max_lines * 10
        if num_tokens > token_budget:
            truncated_lines = []
            current_tokens = 0
            for line in lines:
                line_tokens = ctx.token_count(line + "\n")
                if current_tokens + line_tokens <= token_budget:
                    truncated_lines.append(line)
                    current_tokens += line_tokens
                else:
                    break
            truncated_count = len(lines) - len(truncated_lines)
            return "\n".join(truncated_lines) + f"\n... ({truncated_count} more lines truncated)"
    else:
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


def compress_tool_result(content: str, max_lines: int = 80, is_code: bool = False, ctx = None) -> str:
    """Compress a tool result to keep model context focused.

    For code files: extract imports, function/class signatures, and docstrings
    as a structured summary. Falls back to truncation for non-code or parse errors.

    For non-code: truncate to max_lines with a truncation notice.
    """
    lines = content.split("\n")
    if ctx is not None:
        num_tokens = ctx.token_count(content)
        token_threshold = max_lines * 10
        if num_tokens <= token_threshold:
            return content
    else:
        if len(lines) <= max_lines:
            return content

    if is_code:
        # Strip any truncation notice from read_file before parsing
        clean = re.sub(r"\n\.\.\..*truncated\)$", "", content)
        summary = _summarize_code(clean)
        if summary:
            return summary

    # Fallback: truncate
    if ctx is not None:
        token_threshold = max_lines * 10
        if ctx.token_count(content) > token_threshold:
            truncated_lines = []
            current_tokens = 0
            for line in lines:
                line_tokens = ctx.token_count(line + "\n")
                if current_tokens + line_tokens <= token_threshold:
                    truncated_lines.append(line)
                    current_tokens += line_tokens
                else:
                    break
            truncated_count = len(lines) - len(truncated_lines)
            return "\n".join(truncated_lines) + f"\n... ({truncated_count} more lines, use read_file with specific path for details)"

    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines, use read_file with specific path for details)"


def _summarize_code(source: str) -> str | None:
    """Extract structured summary from Python source: imports, signatures, docstrings.

    Returns None if parsing fails (caller falls back to truncation).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    lines = source.split("\n")
    parts = []

    # Module docstring
    if (isinstance(tree.body[0], ast.Expr) and
            isinstance(tree.body[0].value, (ast.Constant, ast.Str))):
        doc = ast.get_docstring(tree)
        if doc:
            parts.append(f'"""{doc}"""')

    # Imports
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(lines[node.lineno - 1].strip())
    if imports:
        parts.append("# Imports\n" + "\n".join(imports))

    # Function/class signatures with first few lines
    definitions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _extract_signature(node, lines)
            doc = ast.get_docstring(node)
            if doc:
                sig += f"\n    \"\"\"{doc[:120]}\"\"\"" if len(doc) > 120 else f'\n    """{doc}"""'
            definitions.append(sig)
        elif isinstance(node, ast.ClassDef):
            sig = _extract_signature(node, lines)
            doc = ast.get_docstring(node)
            if doc:
                sig += f'\n    """{doc[:120]}..."""' if len(doc) > 120 else f'\n    """{doc}"""'
            # List methods
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(f"  {item.name}({', '.join(a.arg for a in item.args.args)})")
            if methods:
                sig += "\n  Methods:\n" + "\n".join(methods)
            definitions.append(sig)

    if definitions:
        parts.append("# Definitions\n" + "\n\n".join(definitions))

    return "\n\n".join(parts) if parts else None


def _extract_signature(node, lines: list[str]) -> str:
    """Extract the def/class signature line(s) from source."""
    start = node.lineno - 1
    # Take up to 3 lines for the signature (handles multi-line defs)
    sig_lines = []
    for i in range(start, min(start + 3, len(lines))):
        line = lines[i]
        sig_lines.append(line)
        if ":" in line and not line.strip().startswith("#"):
            break
    return "\n".join(sig_lines)


def execute_tool_calls(tool_calls: list[dict], repo_root: str = ".",
                       compress: bool = True, max_result_lines: int = 80,
                       ctx = None, tool_call_start_idx: int = 0) -> list[dict]:
    """Execute a list of OpenAI-format tool_calls. Returns tool response messages.

    Each tool_call has: id, type, function: {name, arguments (JSON string)}.
    Returns list of {"role": "tool", "tool_call_id": ..., "content": ...}.
    """
    results = []
    for i, tc in enumerate(tool_calls):
        idx = tool_call_start_idx + i
        func = tc.get("function", {})
        name = func.get("name", "")
        try:
            args = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        content = execute_tool(name, args, repo_root, ctx=ctx)
        if compress and name == "read_file":
            is_code = args.get("path", "").endswith((".py", ".js", ".ts", ".java", ".go", ".rs"))
            content = compress_tool_result(content, max_lines=max_result_lines, is_code=is_code, ctx=ctx)
        
        # Wire semantic memory fact extraction
        if ctx is not None and getattr(ctx, "_memory", None) is not None:
            # Gating strategies:
            # 1. Limit to the first 3 tool calls globally per run_tool_loop (idx < 3)
            # 2. Limit to files/search results with more than 50 lines
            # 3. Avoid if episodic memory is already full
            mem = ctx._memory
            is_mem_full = False
            if getattr(mem, "episodic", None) is not None:
                count = getattr(mem.episodic, "count", 0)
                max_entries = getattr(mem.episodic, "_max_entries", 200)
                if count >= max_entries:
                    is_mem_full = True

            if (idx < 3 and 
                not is_mem_full and 
                len(content.splitlines()) > 50 and 
                name in ("read_file", "search_files") and 
                not content.startswith("ERROR:")):
                try:
                    facts = ctx._memory.semantic.extract_facts(content)
                    source_ref = args.get("path") or args.get("pattern") or name
                    for fact in facts:
                        ctx._memory.semantic.add_fact(fact, source=str(source_ref))
                except Exception as e:
                    logger.warning(f"Failed to extract semantic facts from tool result: {e}")

        results.append({
            "role": "tool",
            "tool_call_id": tc.get("id", ""),
            "content": content,
        })
        logger.debug(f"Tool {name} executed: {content[:80]}...")
    return results


def run_tool_loop(server, model: str, messages: list[dict],
                  tools: list[dict] | None = None, repo_root: str = ".",
                  max_rounds: int = 5, ctx = None, **chat_opts) -> dict:
    """Run a tool-use loop: model → tool_calls → execute → feed back → repeat.

    Returns the final chat completion response (OpenAI format) when the model
    stops requesting tools or max_rounds is reached.
    """
    tools = tools or TOOL_DEFINITIONS
    msgs = list(messages)
    tools_executed = 0

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
        tool_results = execute_tool_calls(msg["tool_calls"], repo_root, ctx=ctx, tool_call_start_idx=tools_executed)
        tools_executed += len(msg["tool_calls"])
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

        # Test compression: large Python file → structured summary
        big_code = "\n".join([
            "import os",
            "import sys",
            "",
            "def hello():",
            "    print('hello')",
            "",
            "def world(x):",
            "    return x * 2",
            "",
            "class Foo:",
            "    def bar(self):",
            "        pass",
        ] + [f"line_{i} = {i}" for i in range(100)])
        (Path(td) / "big.py").write_text(big_code)
        result = execute_tool("read_file", {"path": "big.py"}, repo_root=td)
        compressed = compress_tool_result(result, max_lines=20, is_code=True)
        assert "# Imports" in compressed
        assert "def hello" in compressed
        assert "class Foo" in compressed
        assert len(compressed.split("\n")) < 20

        # Test non-code truncation
        (Path(td) / "big.txt").write_text("\n".join(f"line {i}" for i in range(200)))
        result = execute_tool("read_file", {"path": "big.txt"}, repo_root=td)
        compressed = compress_tool_result(result, max_lines=10, is_code=False)
        assert "more lines" in compressed
        assert len(compressed.split("\n")) <= 12

        print("tool_proxy self-check OK")
