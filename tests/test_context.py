# tests/test_context.py
import pytest
from unittest.mock import MagicMock

def test_context_add_and_build_prompt():
    """Context manager builds message list from history."""
    from lore.context import ContextManager
    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5  # 5 tokens per call

    cfg = {"system_prompt": 512, "retrieved_memories": 1024,
           "working_context": 4096, "user_input": 2048,
           "generation_headroom": 4096}
    cm = ContextManager(cfg, mock_server, system_prompt="You are a helpful assistant.")

    cm.add_message("user", "Hello")
    cm.add_message("assistant", "Hi there!")
    cm.add_message("user", "How are you?")

    prompt = cm.build_prompt()
    assert prompt[0]["role"] == "system"
    assert prompt[0]["content"] == "You are a helpful assistant."
    assert len(prompt) == 4  # system + 3 messages

def test_context_truncation_drops_oldest():
    """When working_context budget exceeded, oldest messages dropped."""
    from lore.context import ContextManager
    mock_server = MagicMock()
    mock_server.tokenize.return_value = 100  # 100 tokens per message

    cfg = {"system_prompt": 512, "retrieved_memories": 1024,
           "working_context": 250,  # low budget to force truncation
           "user_input": 2048, "generation_headroom": 4096}
    cm = ContextManager(cfg, mock_server, system_prompt="sys")

    for i in range(10):
        cm.add_message("user", f"message {i}")
        cm.add_message("assistant", f"reply {i}")

    prompt = cm.build_prompt()
    # System prompt + truncated history
    assert prompt[0]["role"] == "system"
    # Should be truncated — fewer than 20 messages + system
    assert len(prompt) < 21

def test_context_token_count_uses_server():
    """token_count delegates to model server /tokenize."""
    from lore.context import ContextManager
    mock_server = MagicMock()
    mock_server.tokenize.return_value = 42
    cm = ContextManager({}, mock_server)
    assert cm.token_count("hello world") == 42
    mock_server.tokenize.assert_called_once_with("primary", "hello world")
