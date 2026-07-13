"""Tests for the tool proxy module."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _make_repo(tmpdir):
    """Create a minimal repo structure in tmpdir."""
    (Path(tmpdir) / "main.py").write_text("def main():\n    print('hello')\n")
    (Path(tmpdir) / "utils.py").write_text("x = 42\n# TODO: refactor\n")
    (Path(tmpdir) / "sub").mkdir()
    (Path(tmpdir) / "sub" / "mod.py").write_text("y = 99\n")
    return str(tmpdir)


def test_read_file():
    from lore.tool_proxy import execute_tool
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(td)
        result = execute_tool("read_file", {"path": "main.py"}, repo_root=repo)
        assert "def main" in result
        assert "print('hello')" in result


def test_read_file_not_found():
    from lore.tool_proxy import execute_tool
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(td)
        result = execute_tool("read_file", {"path": "nonexistent.py"}, repo_root=repo)
        assert "ERROR" in result


def test_read_file_path_escape():
    from lore.tool_proxy import execute_tool
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(td)
        result = execute_tool("read_file", {"path": "../../../etc/passwd"}, repo_root=repo)
        assert "ERROR" in result


def test_read_file_truncation():
    from lore.tool_proxy import execute_tool
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(td)
        # Write a long file
        (Path(td) / "long.py").write_text("\n".join(f"line {i}" for i in range(300)))
        result = execute_tool("read_file", {"path": "long.py", "max_lines": 10}, repo_root=repo)
        assert "truncated" in result
        assert "line 0" in result
        assert "line 9" in result
        assert "line 100" not in result


def test_search_files():
    from lore.tool_proxy import execute_tool
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(td)
        result = execute_tool("search_files", {"pattern": "hello"}, repo_root=repo)
        assert "main.py" in result


def test_search_files_no_match():
    from lore.tool_proxy import execute_tool
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(td)
        result = execute_tool("search_files", {"pattern": "nonexistent_pattern_xyz"}, repo_root=repo)
        assert "No matches" in result


def test_list_dir():
    from lore.tool_proxy import execute_tool
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(td)
        result = execute_tool("list_dir", {"path": "."}, repo_root=repo)
        assert "main.py" in result
        assert "utils.py" in result


def test_list_dir_with_pattern():
    from lore.tool_proxy import execute_tool
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(td)
        result = execute_tool("list_dir", {"path": ".", "pattern": "*.py"}, repo_root=repo)
        assert "main.py" in result
        assert "sub/mod.py" in result


def test_unknown_tool():
    from lore.tool_proxy import execute_tool
    result = execute_tool("frobnicate", {}, repo_root=".")
    assert "ERROR" in result
    assert "Unknown tool" in result


def test_execute_tool_calls():
    from lore.tool_proxy import execute_tool_calls
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(td)
        tool_calls = [
            {"id": "call_1", "type": "function",
             "function": {"name": "read_file", "arguments": json.dumps({"path": "main.py"})}},
            {"id": "call_2", "type": "function",
             "function": {"name": "search_files", "arguments": json.dumps({"pattern": "hello"})}},
        ]
        results = execute_tool_calls(tool_calls, repo_root=repo)
        assert len(results) == 2
        assert results[0]["role"] == "tool"
        assert results[0]["tool_call_id"] == "call_1"
        assert "def main" in results[0]["content"]
        assert results[1]["tool_call_id"] == "call_2"
        assert "main.py" in results[1]["content"]


def test_execute_tool_calls_bad_json():
    from lore.tool_proxy import execute_tool_calls
    tool_calls = [
        {"id": "call_1", "type": "function",
         "function": {"name": "read_file", "arguments": "not json"}},
    ]
    results = execute_tool_calls(tool_calls, repo_root=".")
    assert len(results) == 1
    # Should still return a tool message, just with empty args
    assert results[0]["role"] == "tool"


def test_run_tool_loop_no_tools():
    """Model returns no tool_calls → return response immediately."""
    from lore.tool_proxy import run_tool_loop
    mock_server = MagicMock()
    mock_server.chat.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "hello"}}]
    }
    result = run_tool_loop(mock_server, "specialist", [{"role": "user", "content": "hi"}])
    assert result["choices"][0]["message"]["content"] == "hello"
    assert mock_server.chat.call_count == 1


def test_run_tool_loop_with_tools():
    """Model returns tool_calls → execute → feed back → final response."""
    from lore.tool_proxy import run_tool_loop
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(td)
        mock_server = MagicMock()
        # Round 1: tool call, Round 2: final response
        mock_server.chat.side_effect = [
            {"choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": "main.py"})}}]
            }}]},
            {"choices": [{"message": {"role": "assistant", "content": "Found main function"}}]},
        ]
        result = run_tool_loop(mock_server, "specialist",
                               [{"role": "user", "content": "read main.py"}],
                               repo_root=repo)
        assert result["choices"][0]["message"]["content"] == "Found main function"
        assert mock_server.chat.call_count == 2


def test_run_tool_loop_max_rounds():
    """Model keeps requesting tools → stop at max_rounds → force final response."""
    from lore.tool_proxy import run_tool_loop
    mock_server = MagicMock()
    tool_call_msg = {"message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "list_dir", "arguments": json.dumps({"path": "."})}}]
    }}
    final_msg = {"message": {"role": "assistant", "content": "done"}}
    mock_server.chat.side_effect = [
        {"choices": [tool_call_msg]},
        {"choices": [tool_call_msg]},
        {"choices": [final_msg]},  # after max_rounds force
    ]
    result = run_tool_loop(mock_server, "specialist",
                           [{"role": "user", "content": "explore"}],
                           max_rounds=2)
    assert result["choices"][0]["message"]["content"] == "done"


def test_tool_definitions_format():
    """Tool definitions are valid OpenAI function-calling format."""
    from lore.tool_proxy import TOOL_DEFINITIONS
    for td in TOOL_DEFINITIONS:
        assert td["type"] == "function"
        assert "function" in td
        func = td["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func
        assert func["parameters"]["type"] == "object"


# ─── Context compression tests ────────────────────────────────────────────────

def test_compress_small_file_passthrough():
    """Small files pass through unchanged."""
    from lore.tool_proxy import compress_tool_result
    content = "print('hello')\n"
    result = compress_tool_result(content, max_lines=80, is_code=True)
    assert result == content


def test_compress_large_code_file():
    """Large Python files are summarized to imports + signatures."""
    from lore.tool_proxy import compress_tool_result
    code = "\n".join([
        "import os",
        "import sys",
        "",
        "def hello():",
        "    print('hello')",
        "",
        "class Foo:",
        "    def bar(self):",
        "        pass",
    ] + [f"line_{i} = {i}" for i in range(100)])
    result = compress_tool_result(code, max_lines=20, is_code=True)
    assert "# Imports" in result
    assert "def hello" in result
    assert "class Foo" in result
    assert len(result.split("\n")) < 20


def test_compress_non_code_truncation():
    """Non-code files are truncated with a notice."""
    from lore.tool_proxy import compress_tool_result
    content = "\n".join(f"line {i}" for i in range(200))
    result = compress_tool_result(content, max_lines=10, is_code=False)
    assert "more lines" in result
    assert len(result.split("\n")) <= 12


def test_compress_code_with_syntax_error_falls_back():
    """Code with syntax errors falls back to truncation."""
    from lore.tool_proxy import compress_tool_result
    code = "def broken(:\n" + "\n".join(f"line {i}" for i in range(100))
    result = compress_tool_result(code, max_lines=10, is_code=True)
    # Should have truncated, not crashed
    assert "more lines" in result


def test_compress_preserves_class_methods():
    """Class methods are listed in the summary."""
    from lore.tool_proxy import compress_tool_result
    code = "\n".join([
        "class MyClass:",
        "    def method_a(self, x):",
        "        return x",
        "    def method_b(self, y, z):",
        "        return y + z",
    ] + [f"# filler {i}" for i in range(100)])
    result = compress_tool_result(code, max_lines=15, is_code=True)
    assert "method_a" in result
    assert "method_b" in result


def test_compress_preserves_docstrings():
    """Function/class docstrings are included in summary."""
    from lore.tool_proxy import compress_tool_result
    code = '\n'.join([
        'def hello():',
        '    """Greet the user."""',
        '    print("hello")',
        '',
        'class Foo:',
        '    """A foo class."""',
        '    pass',
    ] + [f'# filler {i}' for i in range(100)])
    result = compress_tool_result(code, max_lines=15, is_code=True)
    assert "Greet the user" in result
    assert "A foo class" in result


def test_execute_tool_calls_compresses_read_file():
    """execute_tool_calls compresses large read_file results."""
    from lore.tool_proxy import execute_tool_calls
    with tempfile.TemporaryDirectory() as td:
        # Create a large Python file
        code = "\n".join([
            "import os",
            "def hello():",
            "    print('hello')",
        ] + [f"line_{i} = {i}" for i in range(200)])
        (Path(td) / "big.py").write_text(code)

        results = execute_tool_calls([
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": json.dumps({"path": "big.py"})}},
        ], repo_root=td, compress=True, max_result_lines=20)

        content = results[0]["content"]
        assert "# Imports" in content
        assert "def hello" in content
        assert len(content.split("\n")) < 30
def test_execute_tool_calls_no_compress_search():
    """execute_tool_calls does not compress search results."""
    from lore.tool_proxy import execute_tool_calls
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "test.py").write_text("hello = 1\n")
        results = execute_tool_calls([
            {"id": "c1", "type": "function",
             "function": {"name": "search_files", "arguments": json.dumps({"pattern": "hello"})}},
        ], repo_root=td, compress=True)
        # Search results are not compressed
        assert "test.py" in results[0]["content"]


def test_gate_semantic_extraction():
    """execute_tool_calls limits semantic fact extraction by index and line count."""
    from lore.tool_proxy import execute_tool_calls
    from unittest.mock import MagicMock, patch
    import json
    
    mock_ctx = MagicMock()
    mock_ctx.token_count.side_effect = lambda x: len(str(x)) // 4
    mock_memory = MagicMock()
    mock_ctx._memory = mock_memory
    # Count < max entries
    mock_memory.episodic.count = 5
    mock_memory.episodic._max_entries = 10
    
    # We will simulate 5 read_file tool calls
    tool_calls = [
        {"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": f"file_{i}.py"})}}
        for i in range(5)
    ]
    
    # We mock execute_tool to return a 60-line string for all calls
    large_content = "line\n" * 60
    
    with patch("lore.tool_proxy.execute_tool", return_value=large_content):
        execute_tool_calls(tool_calls, ctx=mock_ctx)
        
    # extract_facts should be called at most 3 times (due to index < 3 limit)
    assert mock_memory.semantic.extract_facts.call_count <= 3


def test_gate_semantic_extraction_short_file():
    """execute_tool_calls does not extract facts from short files (< 50 lines)."""
    from lore.tool_proxy import execute_tool_calls
    from unittest.mock import MagicMock, patch
    import json
    
    mock_ctx = MagicMock()
    mock_ctx.token_count.side_effect = lambda x: len(str(x)) // 4
    mock_memory = MagicMock()
    mock_ctx._memory = mock_memory
    mock_memory.episodic.count = 5
    mock_memory.episodic._max_entries = 10
    
    tool_calls = [
        {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "file.py"})}}
    ]
    
    # 10 lines
    short_content = "line\n" * 10
    
    with patch("lore.tool_proxy.execute_tool", return_value=short_content):
        execute_tool_calls(tool_calls, ctx=mock_ctx)
        
    # Should not call extract_facts
    mock_memory.semantic.extract_facts.assert_not_called()


def test_gate_semantic_extraction_memory_full():
    """execute_tool_calls does not extract facts if episodic memory is full."""
    from lore.tool_proxy import execute_tool_calls
    from unittest.mock import MagicMock, patch
    import json
    
    mock_ctx = MagicMock()
    mock_ctx.token_count.side_effect = lambda x: len(str(x)) // 4
    mock_memory = MagicMock()
    mock_ctx._memory = mock_memory
    # count >= max entries
    mock_memory.episodic.count = 10
    mock_memory.episodic._max_entries = 10
    
    tool_calls = [
        {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "file.py"})}}
    ]
    
    large_content = "line\n" * 60
    
    with patch("lore.tool_proxy.execute_tool", return_value=large_content):
        execute_tool_calls(tool_calls, ctx=mock_ctx)
        
    mock_memory.semantic.extract_facts.assert_not_called()

