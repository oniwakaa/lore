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
