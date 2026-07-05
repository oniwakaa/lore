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
    """When compression enabled and all gate conditions met, old messages get compressed first."""
    from lore.context import ContextManager

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 100  # 100 tokens per message pre-compression

    cfg = {"working_context": 250}
    # min_turns=0 + preserve_recent_turns=2 so the gate fires with 6 turns / 12 msgs
    compression_cfg = {"enabled": True, "ratio": 0.5, "min_turns": 0, "preserve_recent_turns": 2}

    cm = ContextManager(cfg, mock_server, system_prompt="sys", compression=compression_cfg)
    for i in range(6):
        cm.add_message("user", f"message {i}")
        cm.add_message("assistant", f"reply {i}")

    with patch("lore.context.compress_context") as mock_compress:
        mock_compress.return_value = [{"role": "user", "content": "c"}] * 8
        cm.build_prompt()
        mock_compress.assert_called_once()
        # only messages beyond the latest preserve_recent_turns*2 should be passed in
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


# --- Phase 3: hierarchical memory + health integration ---

def test_context_injects_hierarchical_memory():
    """build_prompt retrieves memories from HierarchicalMemory when query is provided."""
    from lore.context import ContextManager

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    mock_memory = MagicMock()
    mock_memory.retrieve.return_value = ["User prefers Python 3.12", "Project uses FastAPI"]

    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys",
                        memory=mock_memory)
    cm.add_message("user", "How do I set up the API?")
    prompt = cm.build_prompt(query="How do I set up the API?")
    # Memory should have been retrieved
    mock_memory.retrieve.assert_called_once_with("How do I set up the API?")
    # Memories should be in the system message
    assert "Python 3.12" in prompt[0]["content"]
    assert "FastAPI" in prompt[0]["content"]


def test_context_memory_merged_with_explicit_memories():
    """HierarchicalMemory results are merged with explicitly passed memories."""
    from lore.context import ContextManager

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    mock_memory = MagicMock()
    mock_memory.retrieve.return_value = ["Fact from semantic memory"]

    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys",
                        memory=mock_memory)
    cm.add_message("user", "query")
    prompt = cm.build_prompt(memories=["Explicit memory"], query="query")
    assert "Explicit memory" in prompt[0]["content"]
    assert "Fact from semantic memory" in prompt[0]["content"]


def test_context_health_check_runs_every_n_turns():
    """ContextHealth.check is called every check_every_n_turns."""
    from lore.context import ContextManager
    from lore.health import ContextHealth

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    mock_health = ContextHealth({"check_every_n_turns": 2})

    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys",
                        health=mock_health)
    # Turn 1: should not check
    cm.add_message("user", "msg 1")
    cm.build_prompt()
    assert cm.last_health_report is None
    # Turn 2: should check
    cm.add_message("assistant", "reply 1")
    cm.build_prompt()
    assert cm.last_health_report is not None


def test_context_health_summarize_triggers_memory_summarization():
    """When health says 'summarize', episodic memory summarization is triggered."""
    from lore.context import ContextManager
    from lore.health import ContextHealth, HealthReport

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 100  # high token count to trigger critical
    mock_memory = MagicMock()
    mock_memory.maybe_summarize.return_value = "Summary of old conversation"

    # Create a health instance that will return 'summarize' action
    mock_health = ContextHealth({"check_every_n_turns": 1, "critical_threshold": 0.90,
                                 "stale_after_turns": 2})

    cm = ContextManager({"working_context": 500}, mock_server, system_prompt="sys",
                        memory=mock_memory, health=mock_health)
    # Add enough messages to trigger stale + critical
    for i in range(10):
        cm.add_message("user", f"message {i} with some content")
        cm.add_message("assistant", f"reply {i} with some content")

    cm.build_prompt()
    # maybe_summarize should have been called
    mock_memory.maybe_summarize.assert_called_once()


def test_context_no_memory_no_error():
    """ContextManager works fine without hierarchical memory."""
    from lore.context import ContextManager
    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys")
    cm.add_message("user", "hello")
    prompt = cm.build_prompt(query="hello")
    assert prompt[0]["role"] == "system"


def test_context_no_health_no_error():
    """ContextManager works fine without health monitoring."""
    from lore.context import ContextManager
    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys")
    cm.add_message("user", "hello")
    prompt = cm.build_prompt()
    assert cm.last_health_report is None


def test_context_compression_skipped_below_min_turns():
    """Compression does not fire when session has fewer than min_turns turns."""
    from lore.context import ContextManager

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 100

    cfg = {"working_context": 250}
    # min_turns=10 but we only add 6 turns — gate should block compression
    compression_cfg = {"enabled": True, "ratio": 0.5, "min_turns": 10, "preserve_recent_turns": 2}

    cm = ContextManager(cfg, mock_server, system_prompt="sys", compression=compression_cfg)
    for i in range(6):
        cm.add_message("user", f"message {i}")
        cm.add_message("assistant", f"reply {i}")

    with patch("lore.context.compress_context") as mock_compress:
        cm.build_prompt()
        mock_compress.assert_not_called()


# --- Phase 3: hierarchical memory + health integration ---

def test_context_injects_hierarchical_memory():
    """build_prompt retrieves memories from HierarchicalMemory when query is provided."""
    from lore.context import ContextManager

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    mock_memory = MagicMock()
    mock_memory.retrieve.return_value = ["User prefers Python 3.12", "Project uses FastAPI"]

    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys",
                        memory=mock_memory)
    cm.add_message("user", "How do I set up the API?")
    prompt = cm.build_prompt(query="How do I set up the API?")
    # Memory should have been retrieved
    mock_memory.retrieve.assert_called_once_with("How do I set up the API?")
    # Memories should be in the system message
    assert "Python 3.12" in prompt[0]["content"]
    assert "FastAPI" in prompt[0]["content"]


def test_context_memory_merged_with_explicit_memories():
    """HierarchicalMemory results are merged with explicitly passed memories."""
    from lore.context import ContextManager

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    mock_memory = MagicMock()
    mock_memory.retrieve.return_value = ["Fact from semantic memory"]

    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys",
                        memory=mock_memory)
    cm.add_message("user", "query")
    prompt = cm.build_prompt(memories=["Explicit memory"], query="query")
    assert "Explicit memory" in prompt[0]["content"]
    assert "Fact from semantic memory" in prompt[0]["content"]


def test_context_health_check_runs_every_n_turns():
    """ContextHealth.check is called every check_every_n_turns."""
    from lore.context import ContextManager
    from lore.health import ContextHealth

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    mock_health = ContextHealth({"check_every_n_turns": 2})

    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys",
                        health=mock_health)
    # Turn 1: should not check
    cm.add_message("user", "msg 1")
    cm.build_prompt()
    assert cm.last_health_report is None
    # Turn 2: should check
    cm.add_message("assistant", "reply 1")
    cm.build_prompt()
    assert cm.last_health_report is not None


def test_context_health_summarize_triggers_memory_summarization():
    """When health says 'summarize', episodic memory summarization is triggered."""
    from lore.context import ContextManager
    from lore.health import ContextHealth, HealthReport

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 100  # high token count to trigger critical
    mock_memory = MagicMock()
    mock_memory.maybe_summarize.return_value = "Summary of old conversation"

    # Create a health instance that will return 'summarize' action
    mock_health = ContextHealth({"check_every_n_turns": 1, "critical_threshold": 0.90,
                                 "stale_after_turns": 2})

    cm = ContextManager({"working_context": 500}, mock_server, system_prompt="sys",
                        memory=mock_memory, health=mock_health)
    # Add enough messages to trigger stale + critical
    for i in range(10):
        cm.add_message("user", f"message {i} with some content")
        cm.add_message("assistant", f"reply {i} with some content")

    cm.build_prompt()
    # maybe_summarize should have been called
    mock_memory.maybe_summarize.assert_called_once()


def test_context_no_memory_no_error():
    """ContextManager works fine without hierarchical memory."""
    from lore.context import ContextManager
    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys")
    cm.add_message("user", "hello")
    prompt = cm.build_prompt(query="hello")
    assert prompt[0]["role"] == "system"


def test_context_no_health_no_error():
    """ContextManager works fine without health monitoring."""
    from lore.context import ContextManager
    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys")
    cm.add_message("user", "hello")
    prompt = cm.build_prompt()
    assert cm.last_health_report is None


def test_context_compression_skipped_under_low_usage():
    """Compression does not fire when context usage is below 70% of budget."""
    from lore.context import ContextManager

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5  # 5 tokens/msg, 12 msgs = 60 tokens, budget=250 → 24%

    cfg = {"working_context": 250}
    compression_cfg = {"enabled": True, "ratio": 0.5, "min_turns": 0, "preserve_recent_turns": 2}

    cm = ContextManager(cfg, mock_server, system_prompt="sys", compression=compression_cfg)
    for i in range(6):
        cm.add_message("user", f"message {i}")
        cm.add_message("assistant", f"reply {i}")

    with patch("lore.context.compress_context") as mock_compress:
        cm.build_prompt()
        mock_compress.assert_not_called()


# --- Phase 3: hierarchical memory + health integration ---

def test_context_injects_hierarchical_memory():
    """build_prompt retrieves memories from HierarchicalMemory when query is provided."""
    from lore.context import ContextManager

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    mock_memory = MagicMock()
    mock_memory.retrieve.return_value = ["User prefers Python 3.12", "Project uses FastAPI"]

    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys",
                        memory=mock_memory)
    cm.add_message("user", "How do I set up the API?")
    prompt = cm.build_prompt(query="How do I set up the API?")
    # Memory should have been retrieved
    mock_memory.retrieve.assert_called_once_with("How do I set up the API?")
    # Memories should be in the system message
    assert "Python 3.12" in prompt[0]["content"]
    assert "FastAPI" in prompt[0]["content"]


def test_context_memory_merged_with_explicit_memories():
    """HierarchicalMemory results are merged with explicitly passed memories."""
    from lore.context import ContextManager

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    mock_memory = MagicMock()
    mock_memory.retrieve.return_value = ["Fact from semantic memory"]

    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys",
                        memory=mock_memory)
    cm.add_message("user", "query")
    prompt = cm.build_prompt(memories=["Explicit memory"], query="query")
    assert "Explicit memory" in prompt[0]["content"]
    assert "Fact from semantic memory" in prompt[0]["content"]


def test_context_health_check_runs_every_n_turns():
    """ContextHealth.check is called every check_every_n_turns."""
    from lore.context import ContextManager
    from lore.health import ContextHealth

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    mock_health = ContextHealth({"check_every_n_turns": 2})

    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys",
                        health=mock_health)
    # Turn 1: should not check
    cm.add_message("user", "msg 1")
    cm.build_prompt()
    assert cm.last_health_report is None
    # Turn 2: should check
    cm.add_message("assistant", "reply 1")
    cm.build_prompt()
    assert cm.last_health_report is not None


def test_context_health_summarize_triggers_memory_summarization():
    """When health says 'summarize', episodic memory summarization is triggered."""
    from lore.context import ContextManager
    from lore.health import ContextHealth, HealthReport

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 100  # high token count to trigger critical
    mock_memory = MagicMock()
    mock_memory.maybe_summarize.return_value = "Summary of old conversation"

    # Create a health instance that will return 'summarize' action
    mock_health = ContextHealth({"check_every_n_turns": 1, "critical_threshold": 0.90,
                                 "stale_after_turns": 2})

    cm = ContextManager({"working_context": 500}, mock_server, system_prompt="sys",
                        memory=mock_memory, health=mock_health)
    # Add enough messages to trigger stale + critical
    for i in range(10):
        cm.add_message("user", f"message {i} with some content")
        cm.add_message("assistant", f"reply {i} with some content")

    cm.build_prompt()
    # maybe_summarize should have been called
    mock_memory.maybe_summarize.assert_called_once()


def test_context_no_memory_no_error():
    """ContextManager works fine without hierarchical memory."""
    from lore.context import ContextManager
    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys")
    cm.add_message("user", "hello")
    prompt = cm.build_prompt(query="hello")
    assert prompt[0]["role"] == "system"


def test_context_no_health_no_error():
    """ContextManager works fine without health monitoring."""
    from lore.context import ContextManager
    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys")
    cm.add_message("user", "hello")
    prompt = cm.build_prompt()
    assert cm.last_health_report is None


def test_context_compression_skipped_when_no_old_messages():
    """Compression does not fire when all messages are within preserve_recent_turns."""
    from lore.context import ContextManager

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 100

    cfg = {"working_context": 250}
    # preserve_recent_turns=10 → keep last 20 msgs; we only have 12 → no old messages
    compression_cfg = {"enabled": True, "ratio": 0.5, "min_turns": 0, "preserve_recent_turns": 10}

    cm = ContextManager(cfg, mock_server, system_prompt="sys", compression=compression_cfg)
    for i in range(6):
        cm.add_message("user", f"message {i}")
        cm.add_message("assistant", f"reply {i}")

    with patch("lore.context.compress_context") as mock_compress:
        cm.build_prompt()
        mock_compress.assert_not_called()


# --- Phase 3: hierarchical memory + health integration ---

def test_context_injects_hierarchical_memory():
    """build_prompt retrieves memories from HierarchicalMemory when query is provided."""
    from lore.context import ContextManager

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    mock_memory = MagicMock()
    mock_memory.retrieve.return_value = ["User prefers Python 3.12", "Project uses FastAPI"]

    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys",
                        memory=mock_memory)
    cm.add_message("user", "How do I set up the API?")
    prompt = cm.build_prompt(query="How do I set up the API?")
    # Memory should have been retrieved
    mock_memory.retrieve.assert_called_once_with("How do I set up the API?")
    # Memories should be in the system message
    assert "Python 3.12" in prompt[0]["content"]
    assert "FastAPI" in prompt[0]["content"]


def test_context_memory_merged_with_explicit_memories():
    """HierarchicalMemory results are merged with explicitly passed memories."""
    from lore.context import ContextManager

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    mock_memory = MagicMock()
    mock_memory.retrieve.return_value = ["Fact from semantic memory"]

    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys",
                        memory=mock_memory)
    cm.add_message("user", "query")
    prompt = cm.build_prompt(memories=["Explicit memory"], query="query")
    assert "Explicit memory" in prompt[0]["content"]
    assert "Fact from semantic memory" in prompt[0]["content"]


def test_context_health_check_runs_every_n_turns():
    """ContextHealth.check is called every check_every_n_turns."""
    from lore.context import ContextManager
    from lore.health import ContextHealth

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    mock_health = ContextHealth({"check_every_n_turns": 2})

    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys",
                        health=mock_health)
    # Turn 1: should not check
    cm.add_message("user", "msg 1")
    cm.build_prompt()
    assert cm.last_health_report is None
    # Turn 2: should check
    cm.add_message("assistant", "reply 1")
    cm.build_prompt()
    assert cm.last_health_report is not None


def test_context_health_summarize_triggers_memory_summarization():
    """When health says 'summarize', episodic memory summarization is triggered."""
    from lore.context import ContextManager
    from lore.health import ContextHealth, HealthReport

    mock_server = MagicMock()
    mock_server.tokenize.return_value = 100  # high token count to trigger critical
    mock_memory = MagicMock()
    mock_memory.maybe_summarize.return_value = "Summary of old conversation"

    # Create a health instance that will return 'summarize' action
    mock_health = ContextHealth({"check_every_n_turns": 1, "critical_threshold": 0.90,
                                 "stale_after_turns": 2})

    cm = ContextManager({"working_context": 500}, mock_server, system_prompt="sys",
                        memory=mock_memory, health=mock_health)
    # Add enough messages to trigger stale + critical
    for i in range(10):
        cm.add_message("user", f"message {i} with some content")
        cm.add_message("assistant", f"reply {i} with some content")

    cm.build_prompt()
    # maybe_summarize should have been called
    mock_memory.maybe_summarize.assert_called_once()


def test_context_no_memory_no_error():
    """ContextManager works fine without hierarchical memory."""
    from lore.context import ContextManager
    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys")
    cm.add_message("user", "hello")
    prompt = cm.build_prompt(query="hello")
    assert prompt[0]["role"] == "system"


def test_context_no_health_no_error():
    """ContextManager works fine without health monitoring."""
    from lore.context import ContextManager
    mock_server = MagicMock()
    mock_server.tokenize.return_value = 5
    cm = ContextManager({"working_context": 4096}, mock_server, system_prompt="sys")
    cm.add_message("user", "hello")
    prompt = cm.build_prompt()
    assert cm.last_health_report is None
