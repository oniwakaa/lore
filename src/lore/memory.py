# src/lore/memory.py
"""Episodic memory with embeddings. Stores raw text, retrieves by cosine similarity."""
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

class EpisodicMemory:
    """Embed conversation turns, retrieve relevant context by similarity."""

    def __init__(self, config: dict, model_server):
        self._config = config
        self._server = model_server
        self._entries: list[tuple[str, list[float], float]] = []  # (text, embedding, timestamp)
        self._max_entries = config.get("max_entries", 200)
        self._top_k = config.get("top_k", 3)
        self._threshold = config.get("similarity_threshold", 0.5)
        self._max_chars = config.get("max_text_chars", 500)

    def store(self, text: str, role: str) -> None:
        """Embed raw text (first N chars) and store. No summarization."""
        truncated = text[:self._max_chars]
        try:
            embedding = self._server.embed(truncated)
        except Exception as e:
            logger.warning(f"Embed failed, skipping store: {e}")
            return
        self._entries.append((text, embedding, time.time()))
        # Circular buffer: drop oldest if over max
        while len(self._entries) > self._max_entries:
            self._entries.pop(0)

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        """Embed query, compute cosine similarity, return top-k texts."""
        if not self._entries:
            return []

        k = top_k or self._top_k
        try:
            query_emb = np.array(self._server.embed(query))
        except Exception as e:
            logger.warning(f"Embed query failed, skipping retrieval: {e}")
            return []

        texts, embeddings, _ = zip(*self._entries)
        emb_matrix = np.array(embeddings)

        # Cosine similarity
        norms = np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(query_emb)
        norms[norms == 0] = 1  # avoid div by zero
        sims = emb_matrix @ query_emb / norms

        # Filter by threshold, sort, take top-k
        scored = [(float(sims[i]), texts[i]) for i in range(len(texts))]
        scored = [(s, t) for s, t in scored if s >= self._threshold]
        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:k]]

    def clear(self) -> None:
        """Wipe all stored entries."""
        self._entries.clear()
