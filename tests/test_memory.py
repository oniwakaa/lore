# tests/test_memory.py
import pytest
import numpy as np
from unittest.mock import MagicMock

def test_memory_store_and_retrieve():
    """Memory stores embeddings and retrieves by cosine similarity."""
    from lore.memory import EpisodicMemory
    mock_server = MagicMock()
    # Return deterministic embeddings
    call_count = [0]
    def mock_embed(text):
        call_count[0] += 1
        # Simple embedding: based on text length
        return [float(len(text))] + [0.0] * 767
    mock_server.embed.side_effect = mock_embed

    cfg = {"top_k": 2, "max_entries": 200, "similarity_threshold": 0.0, "max_text_chars": 500}
    mem = EpisodicMemory(cfg, mock_server)

    mem.store("user asked about Python", "user")
    mem.store("assistant explained Python loops", "assistant")
    mem.store("user asked about databases", "user")

    results = mem.retrieve("Tell me about Python")
    assert len(results) <= 2  # top_k=2
    assert "Python" in results[0]  # most similar to "Python"

def test_memory_max_entries_circular():
    """Memory acts as circular buffer, dropping oldest beyond max_entries."""
    from lore.memory import EpisodicMemory
    mock_server = MagicMock()
    mock_server.embed.return_value = [1.0] + [0.0] * 767

    cfg = {"top_k": 3, "max_entries": 3, "similarity_threshold": 0.0, "max_text_chars": 500}
    mem = EpisodicMemory(cfg, mock_server)

    for i in range(5):
        mem.store(f"entry {i}", "user")

    assert len(mem._entries) == 3  # only max_entries kept

def test_memory_clear():
    """Clear wipes all stored entries."""
    from lore.memory import EpisodicMemory
    mock_server = MagicMock()
    mock_server.embed.return_value = [1.0] + [0.0] * 767

    cfg = {"top_k": 3, "max_entries": 200, "similarity_threshold": 0.0, "max_text_chars": 500}
    mem = EpisodicMemory(cfg, mock_server)
    mem.store("test", "user")
    mem.clear()
    assert len(mem._entries) == 0


# --- Hierarchical memory tests ---

def test_episodic_store_summary():
    """store_summary embeds and stores a summary."""
    from lore.memory import EpisodicMemory
    mock_server = MagicMock()
    mock_server.embed.return_value = [1.0] + [0.0] * 767

    mem = EpisodicMemory({"similarity_threshold": 0.0, "max_entries": 200}, mock_server)
    mem.store_summary("User worked on a FastAPI authentication module.")
    assert len(mem._entries) == 1
    assert mem._entries[0][0] == "User worked on a FastAPI authentication module."

def test_summarize_session_uses_specialist_model():
    """summarize_session calls the specialist model to compress messages."""
    from lore.memory import EpisodicMemory
    mock_server = MagicMock()
    mock_server.chat.return_value = {
        "choices": [{"message": {"content": "User built a REST API with authentication."}}]
    }
    mem = EpisodicMemory({}, mock_server)
    messages = [
        {"role": "user", "content": "How do I add JWT auth to FastAPI?"},
        {"role": "assistant", "content": "You can use python-jose..."},
        {"role": "user", "content": "How do I handle token refresh?"},
        {"role": "assistant", "content": "Use a refresh token endpoint..."},
    ]
    summary = mem.summarize_session(messages)
    assert "REST API" in summary or "authentication" in summary.lower()
    mock_server.chat.assert_called_once()
    # Should call specialist model
    assert mock_server.chat.call_args[0][0] == "specialist"

def test_summarize_session_falls_back_on_failure():
    """summarize_session uses extractive fallback when specialist fails."""
    from lore.memory import EpisodicMemory
    mock_server = MagicMock()
    mock_server.chat.side_effect = Exception("model unavailable")
    mem = EpisodicMemory({}, mock_server)
    messages = [{"role": "user", "content": "Test message for fallback summarization."}]
    summary = mem.summarize_session(messages)
    assert "Test message" in summary  # extractive fallback returns raw content

def test_semantic_memory_add_and_retrieve():
    """SemanticMemory stores and retrieves facts by similarity."""
    from lore.memory import SemanticMemory
    mock_server = MagicMock()
    # Different embeddings for different facts
    def mock_embed(text):
        if "Python" in text:
            return [1.0, 0.0, 0.0] + [0.0] * 765
        if "PostgreSQL" in text:
            return [0.0, 1.0, 0.0] + [0.0] * 765
        return [0.0, 0.0, 1.0] + [0.0] * 765
    mock_server.embed.side_effect = mock_embed

    cfg = {"semantic_top_k": 2, "semantic_similarity_threshold": 0.0, "max_facts": 100}
    sm = SemanticMemory(cfg, mock_server)
    sm.add_fact("User prefers Python 3.12", source="episode_1")
    sm.add_fact("Project uses PostgreSQL for database", source="episode_2")

    results = sm.retrieve("What language does the user like?")
    assert len(results) <= 2
    assert "Python" in results[0]

def test_semantic_memory_dedup():
    """add_fact skips identical facts."""
    from lore.memory import SemanticMemory
    mock_server = MagicMock()
    mock_server.embed.return_value = [1.0] + [0.0] * 767
    sm = SemanticMemory({"semantic_similarity_threshold": 0.0}, mock_server)
    sm.add_fact("User prefers dark mode")
    sm.add_fact("User prefers dark mode")  # duplicate
    assert sm.count == 1

def test_extract_facts_uses_specialist_model():
    """extract_facts calls the specialist model to pull durable facts."""
    from lore.memory import SemanticMemory
    mock_server = MagicMock()
    mock_server.chat.return_value = {
        "choices": [{"message": {"content": "User prefers TypeScript\nProject uses React\nDeployment on Vercel"}}]
    }
    sm = SemanticMemory({}, mock_server)
    facts = sm.extract_facts("User worked on a React app deployed on Vercel, written in TypeScript.")
    assert len(facts) == 3
    assert any("TypeScript" in f for f in facts)
    mock_server.chat.assert_called_once()

def test_extract_facts_falls_back_on_failure():
    """extract_facts uses heuristic fallback when specialist fails."""
    from lore.memory import SemanticMemory
    mock_server = MagicMock()
    mock_server.chat.side_effect = Exception("model unavailable")
    sm = SemanticMemory({}, mock_server)
    summary = "The user prefers Python. They are building a web service. It uses PostgreSQL."
    facts = sm.extract_facts(summary)
    assert len(facts) <= 3
    assert len(facts) > 0

def test_hierarchical_memory_maybe_summarize():
    """maybe_summarize triggers when enough messages have accumulated."""
    from lore.memory import HierarchicalMemory
    mock_server = MagicMock()
    mock_server.embed.return_value = [1.0] + [0.0] * 767
    mock_server.chat.return_value = {
        "choices": [{"message": {"content": "Summary of conversation."}}]
    }
    cfg = {"summarize_after_turns": 3, "extract_facts_every_n_episodes": 1,
           "similarity_threshold": 0.0, "semantic_similarity_threshold": 0.0}
    hm = HierarchicalMemory(cfg, mock_server)

    # 2 turns — not enough
    messages = [
        {"role": "user", "content": "msg 1"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "msg 2"},
        {"role": "assistant", "content": "reply 2"},
    ]
    assert hm.maybe_summarize(messages) is None

    # 3+ turns — should summarize
    messages.extend([
        {"role": "user", "content": "msg 3"},
        {"role": "assistant", "content": "reply 3"},
    ])
    result = hm.maybe_summarize(messages)
    assert result is not None
    assert hm.episodic.count == 1

def test_hierarchical_memory_retrieve_merges_tiers():
    """retrieve returns episodic summaries + semantic facts, deduplicated."""
    from lore.memory import HierarchicalMemory
    mock_server = MagicMock()
    mock_server.embed.return_value = [1.0] + [0.0] * 767
    cfg = {"top_k": 3, "semantic_top_k": 5,
           "similarity_threshold": 0.0, "semantic_similarity_threshold": 0.0}
    hm = HierarchicalMemory(cfg, mock_server)

    hm.episodic.store_summary("Built a REST API with FastAPI")
    hm.semantic.add_fact("User prefers Python")

    results = hm.retrieve("API development")
    assert len(results) == 2  # 1 episodic + 1 semantic

def test_hierarchical_memory_clear():
    """clear wipes both episodic and semantic tiers."""
    from lore.memory import HierarchicalMemory
    mock_server = MagicMock()
    mock_server.embed.return_value = [1.0] + [0.0] * 767
    hm = HierarchicalMemory({"similarity_threshold": 0.0, "semantic_similarity_threshold": 0.0}, mock_server)
    hm.episodic.store_summary("test episode")
    hm.semantic.add_fact("test fact")
    hm.clear()
    assert hm.episodic.count == 0
    assert hm.semantic.count == 0
