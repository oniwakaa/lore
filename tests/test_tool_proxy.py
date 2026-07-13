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
