# tests/test_context.py
import pytest
from unittest.mock import MagicMock, patch

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


def test_context_uses_local_tokenizer_when_available():
    """token_count prefers the cached local tokenizer over HTTP."""
    from lore.context import ContextManager
    mock_server = MagicMock()
    mock_server.tokenize.return_value = 999  # should never be used

    with patch("lore.context.Tokenizer") as mock_tok_cls:
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value.ids = [1, 2, 3]
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        cm = ContextManager({}, mock_server, tokenizer_source="local", tokenizer_repo="org/model")
        assert cm.token_count("hi") == 3
        mock_server.tokenize.assert_not_called()


def test_context_falls_back_to_http_when_local_tokenizer_fails():
    """If local tokenizer load raises, fall back to HTTP /tokenize."""
    from lore.context import ContextManager
    mock_server = MagicMock()
    mock_server.tokenize.return_value = 7

    with patch("lore.context.Tokenizer") as mock_tok_cls:
        mock_tok_cls.from_pretrained.side_effect = Exception("network down")

        cm = ContextManager({}, mock_server, tokenizer_source="local", tokenizer_repo="org/model")
        assert cm.token_count("hi") == 7
        mock_server.tokenize.assert_called_once()


def test_context_tokenizer_source_http_skips_local():
    """tokenizer_source='http' never attempts to load a local tokenizer."""
    from lore.context import ContextManager
    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5

    with patch("lore.context.Tokenizer") as mock_tok_cls:
        cm = ContextManager({}, mock_server, tokenizer_source="http", tokenizer_repo="org/model")
        assert cm.token_count("hi") == 5
        mock_tok_cls.from_pretrained.assert_not_called()


def test_context_compresses_old_messages_before_truncating():
    """When compression enabled and budget exceeded, old messages get compressed first."""
    from lore.context import ContextManager

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 100  # 100 tokens per message pre-compression

    cfg = {"working_context": 250}
    compression_cfg = {"enabled": True, "ratio": 0.5}

    cm = ContextManager(cfg, mock_server, system_prompt="sys", compression=compression_cfg)
    for i in range(6):
        cm.add_message("user", f"message {i}")
        cm.add_message("assistant", f"reply {i}")

    with patch("lore.context.compress_context") as mock_compress:
        mock_compress.return_value = [{"role": "user", "content": "c"}] * 8
        cm.build_prompt()
        mock_compress.assert_called_once()
        # only messages beyond the latest 2 turns (4 messages) should be passed in
        compressed_input = mock_compress.call_args[0][0]
        assert len(compressed_input) == len(cm._history) - 4


def test_context_compression_disabled_by_default():
    """Compression is off unless explicitly enabled in config."""
    from lore.context import ContextManager

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 100

    cm = ContextManager({"working_context": 250}, mock_server, system_prompt="sys")
    for i in range(6):
        cm.add_message("user", f"message {i}")
        cm.add_message("assistant", f"reply {i}")

    with patch("lore.context.compress_context") as mock_compress:
        cm.build_prompt()
        mock_compress.assert_not_called()
