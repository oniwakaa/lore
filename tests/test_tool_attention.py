import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np


def _mock_server_with_embeddings(text_to_vec: dict):
    """Return a MagicMock server whose embed() looks up a fixed vector per text."""
    server = MagicMock()

    def embed(text):
        for key, vec in text_to_vec.items():
            if key in text:
                return vec
        return [0.0, 0.0, 0.0]

    server.embed.side_effect = embed
    return server


def test_select_tools_returns_top_k_by_similarity():
    from lore.tool_attention import ToolAttention

    schemas = [
        {"name": "read_file", "description": "read a file from disk"},
        {"name": "web_search", "description": "search the web"},
        {"name": "calculator", "description": "do arithmetic"},
    ]
    text_to_vec = {
        "read_file": [1.0, 0.0, 0.0],
        "web_search": [0.0, 1.0, 0.0],
        "calculator": [0.0, 0.0, 1.0],
        "search the internet for news": [0.0, 0.9, 0.1],
    }
    server = _mock_server_with_embeddings(text_to_vec)

    ta = ToolAttention(server, schemas, min_tools_for_attention=0)
    selected = ta.select_tools("search the internet for news", k=1)

    assert len(selected) == 1
    assert selected[0]["name"] == "web_search"


def test_select_tools_respects_k():
    from lore.tool_attention import ToolAttention

    schemas = [{"name": f"tool{i}", "description": f"desc {i}"} for i in range(5)]
    server = MagicMock()
    server.embed.return_value = [1.0, 0.0]

    ta = ToolAttention(server, schemas, min_tools_for_attention=0)
    selected = ta.select_tools("anything", k=3)
    assert len(selected) == 3


def test_select_tools_empty_registry():
    from lore.tool_attention import ToolAttention

    server = MagicMock()
    ta = ToolAttention(server, [])
    assert ta.select_tools("anything") == []


def test_select_tools_falls_back_when_query_embed_fails():
    from lore.tool_attention import ToolAttention

    schemas = [{"name": "read_file", "description": "read a file"}]
    server = MagicMock()
    server.embed.side_effect = [[1.0, 0.0], Exception("embed down")]

    ta = ToolAttention(server, schemas, min_tools_for_attention=0)
    selected = ta.select_tools("query text", k=1)
    assert selected == schemas


def test_from_config_loads_yaml():
    from lore.tool_attention import ToolAttention

    server = MagicMock()
    server.embed.return_value = [1.0, 0.0]

    with tempfile.TemporaryDirectory() as d:
        config_path = Path(d) / "tools.yaml"
        config_path.write_text(
            "tools:\n  - name: shell\n    description: run a shell command\n"
        )
        ta = ToolAttention.from_config(server, str(config_path))
        assert len(ta.select_tools("run ls", k=5)) == 1


def test_from_config_missing_file_returns_empty():
    from lore.tool_attention import ToolAttention

    server = MagicMock()
    ta = ToolAttention.from_config(server, "nonexistent-tools.yaml")
    assert ta.select_tools("anything") == []


def test_size_gate_returns_all_tools_below_threshold():
    """When registry <= min_tools_for_attention, return ALL tools without embed()."""
    from lore.tool_attention import ToolAttention

    schemas = [{"name": f"tool{i}", "description": f"desc {i}"} for i in range(5)]
    server = MagicMock()
    # If embed() is called, fail the test
    server.embed.side_effect = AssertionError("embed() should not be called below gate")

    ta = ToolAttention(server, schemas, min_tools_for_attention=15)
    selected = ta.select_tools("anything", k=2)
    assert len(selected) == 5  # ALL tools, not k=2
    assert server.embed.call_count == 0  # no per-query embed


def test_size_gate_uses_embedding_above_threshold():
    """When registry > min_tools_for_attention, use embedding-based selection."""
    from lore.tool_attention import ToolAttention

    schemas = [{"name": f"tool{i}", "description": f"desc {i}"} for i in range(20)]
    server = MagicMock()
    server.embed.return_value = [1.0, 0.0]

    ta = ToolAttention(server, schemas, min_tools_for_attention=15)
    selected = ta.select_tools("query", k=3)
    assert len(selected) == 3
    # embed() was called: once per schema at init (20) + once for query
    assert server.embed.call_count == 21


def test_from_config_reads_gate_params():
    """from_config loads min_tools_for_attention and default_k from YAML."""
    from lore.tool_attention import ToolAttention

    server = MagicMock()
    server.embed.return_value = [1.0, 0.0]

    with tempfile.TemporaryDirectory() as d:
        config_path = Path(d) / "tools.yaml"
        config_path.write_text(
            "min_tools_for_attention: 2\n"
            "default_k: 1\n"
            "tools:\n"
            "  - name: shell\n    description: run a shell command\n"
        )
        ta = ToolAttention.from_config(server, str(config_path))
        # 1 tool <= min_tools_for_attention(2) → returns all without embed
        selected = ta.select_tools("run ls")
        assert len(selected) == 1
        assert ta._default_k == 1
