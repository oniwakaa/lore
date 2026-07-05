"""Hierarchical memory: working → episodic → semantic.

Working memory: current session, last 5-10 turns (lives in ContextManager._history).
Episodic memory: compressed session summaries of old conversations, 50-200 entries.
Semantic memory: durable facts extracted from episode summaries, 20-100 entries.

Retrieval: top-3 episodic summaries + top-5 semantic facts by embedding similarity.
"""
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _cosine_sim_matrix(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity of query against each row of matrix. Safe for zero norms."""
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    norms[norms == 0] = 1.0
    return matrix @ query / norms


class EpisodicMemory:
    """Embed conversation turns, retrieve relevant context by similarity.

    Upgraded to store compressed summaries of old conversations rather than
    raw message pairs. Summarization is delegated to the specialist model
    (Falcon-H1) via the model_server.chat() interface.
    """

    def __init__(self, config: dict, model_server):
        self._config = config
        self._server = model_server
        self._entries: list[tuple[str, list[float], float]] = []  # (summary, embedding, timestamp)
        self._max_entries = config.get("max_entries", 200)
        self._top_k = config.get("top_k", 3)
        self._threshold = config.get("similarity_threshold", 0.5)
        self._max_chars = config.get("max_text_chars", 500)

    def store(self, text: str, role: str) -> None:
        """Embed raw text (first N chars) and store. Used for low-level storage.

        For hierarchical memory, prefer store_summary() which stores a
        compressed summary rather than raw text.
        """
        truncated = text[:self._max_chars]
        try:
            embedding = self._server.embed(truncated)
        except Exception as e:
            logger.warning(f"Embed failed, skipping store: {e}")
            return
        self._entries.append((text, embedding, time.time()))
        while len(self._entries) > self._max_entries:
            self._entries.pop(0)

    def store_summary(self, summary: str) -> None:
        """Embed and store an episode summary."""
        truncated = summary[:self._max_chars]
        try:
            embedding = self._server.embed(truncated)
        except Exception as e:
            logger.warning(f"Embed failed, skipping store_summary: {e}")
            return
        self._entries.append((summary, embedding, time.time()))
        while len(self._entries) > self._max_entries:
            self._entries.pop(0)
        logger.info(f"Stored episodic summary ({len(self._entries)} total)")

    def summarize_session(self, messages: list[dict]) -> str:
        """Compress a batch of old messages into a 2-3 sentence summary.

        Uses the specialist model (Falcon-H1) for summarization to avoid
        consuming primary model context. Falls back to a simple extractive
        summary (first N chars of concatenated messages) on any failure.
        """
        if not messages:
            return ""
        # Build a compact representation for the summarizer
        conversation = "\n".join(
            f"{m['role']}: {m['content'][:200]}" for m in messages[:20]
        )
        prompt = (
            f"Summarize the following conversation in 2-3 sentences, "
            f"focusing on key topics, decisions, and outcomes:\n\n{conversation}\n\nSummary:"
        )
        try:
            result = self._server.chat(
                "specialist",
                [{"role": "system", "content": "You are a concise summarizer."},
                 {"role": "user", "content": prompt}],
                max_tokens=128, temperature=0.3,
            )
            summary = result["choices"][0]["message"]["content"].strip()
            if summary:
                return summary
        except Exception as e:
            logger.warning(f"Specialist summarization failed ({e}), using extractive fallback")
        # Extractive fallback: first 300 chars of concatenated content
        joined = " ".join(m["content"] for m in messages)
        return joined[:300] + ("..." if len(joined) > 300 else "")

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        """Embed query, compute cosine similarity, return top-k summaries."""
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
        sims = _cosine_sim_matrix(emb_matrix, query_emb)
        scored = [(float(sims[i]), texts[i]) for i in range(len(texts))]
        scored = [(s, t) for s, t in scored if s >= self._threshold]
        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:k]]

    def clear(self) -> None:
        """Wipe all stored entries."""
        self._entries.clear()

    @property
    def count(self) -> int:
        return len(self._entries)


class SemanticMemory:
    """Durable facts extracted from episode summaries.

    Stores key-value facts (user preferences, project names, coding conventions,
    recurring topics) with source episode references for provenance tracking.
    """

    def __init__(self, config: dict, model_server):
        self._config = config
        self._server = model_server
        self._facts: list[tuple[str, list[float], float, str]] = []  # (fact, embedding, timestamp, source)
        self._max_facts = config.get("max_facts", 100)
        self._top_k = config.get("semantic_top_k", 5)
        self._threshold = config.get("semantic_similarity_threshold", 0.4)

    def add_fact(self, fact: str, source: str = "") -> None:
        """Embed and store a durable fact with optional source episode reference."""
        try:
            embedding = self._server.embed(fact)
        except Exception as e:
            logger.warning(f"Embed failed for fact, skipping: {e}")
            return
        # Deduplicate: skip if an identical fact already exists
        for existing, _, _, _ in self._facts:
            if existing == fact:
                return
        self._facts.append((fact, embedding, time.time(), source))
        while len(self._facts) > self._max_facts:
            self._facts.pop(0)
        logger.info(f"Stored semantic fact ({len(self._facts)} total)")

    def extract_facts(self, summary: str) -> list[str]:
        """Pull durable facts from an episode summary.

        Uses the specialist model to extract user preferences, project names,
        coding conventions, and recurring topics. Falls back to a simple
        sentence-split heuristic on any failure.
        """
        if not summary or not summary.strip():
            return []
        prompt = (
            "Extract durable facts from the following summary. "
            "Output one fact per line, each as a simple statement. "
            "Only extract facts that would be useful in future sessions "
            "(user preferences, project details, conventions, recurring topics).\n\n"
            f"Summary:\n{summary}\n\nFacts:"
        )
        try:
            result = self._server.chat(
                "specialist",
                [{"role": "system", "content": "You extract structured facts from text."},
                 {"role": "user", "content": prompt}],
                max_tokens=128, temperature=0.2,
            )
            text = result["choices"][0]["message"]["content"].strip()
            facts = [line.strip("- ").strip() for line in text.split("\n") if line.strip()]
            # Filter out empty/too-short lines
            facts = [f for f in facts if len(f) > 5]
            if facts:
                return facts[:10]  # cap at 10 facts per extraction
        except Exception as e:
            logger.warning(f"Specialist fact extraction failed ({e}), using heuristic fallback")
        # Heuristic fallback: split on sentences, take first 3
        sentences = summary.replace("!", ".").replace("?", ".").split(".")
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        return sentences[:3]

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        """Retrieve top-k facts by embedding similarity to query."""
        if not self._facts:
            return []
        k = top_k or self._top_k
        try:
            query_emb = np.array(self._server.embed(query))
        except Exception as e:
            logger.warning(f"Embed query failed for semantic retrieval: {e}")
            return []
        texts, embeddings, _, _ = zip(*self._facts)
        emb_matrix = np.array(embeddings)
        sims = _cosine_sim_matrix(emb_matrix, query_emb)
        scored = [(float(sims[i]), texts[i]) for i in range(len(texts))]
        scored = [(s, t) for s, t in scored if s >= self._threshold]
        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:k]]

    def clear(self) -> None:
        self._facts.clear()

    @property
    def count(self) -> int:
        return len(self._facts)


class HierarchicalMemory:
    """3-tier memory orchestrator: working → episodic → semantic.

    Working memory lives in ContextManager._history (last 5-10 turns).
    This class manages the episodic and semantic tiers and provides
    a unified retrieve() that merges results from both.
    """

    def __init__(self, config: dict, model_server):
        self._config = config
        self._server = model_server
        self.episodic = EpisodicMemory(config, model_server)
        self.semantic = SemanticMemory(config, model_server)
        self._summarize_after_turns = config.get("summarize_after_turns", 10)
        self._extract_facts_every_n = config.get("extract_facts_every_n_episodes", 5)
        self._episode_count = 0

    def store(self, text: str, role: str) -> None:
        """Delegate raw message storage to episodic tier (backward compat)."""
        self.episodic.store(text, role)

    def maybe_summarize(self, messages: list[dict]) -> str | None:
        """If enough messages have accumulated, summarize and store an episode.

        Call this after adding messages to working memory. Returns the
        summary if one was created, None otherwise.
        """
        if len(messages) < self._summarize_after_turns * 2:
            return None
        summary = self.episodic.summarize_session(messages)
        if summary:
            self.episodic.store_summary(summary)
            self._episode_count += 1
            # Periodically extract facts from the new episode
            if self._episode_count % self._extract_facts_every_n == 0:
                facts = self.semantic.extract_facts(summary)
                for fact in facts:
                    self.semantic.add_fact(fact, source=f"episode_{self._episode_count}")
            return summary
        return None

    def retrieve(self, query: str) -> list[str]:
        """Retrieve relevant memories from both tiers.

        Returns top-3 episodic summaries + top-5 semantic facts, merged
        and deduplicated, ordered by tier (episodic first, then semantic).
        """
        episodes = self.episodic.retrieve(query, top_k=3)
        facts = self.semantic.retrieve(query, top_k=5)
        # Deduplicate while preserving order
        seen = set()
        result = []
        for item in episodes + facts:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def clear(self) -> None:
        self.episodic.clear()
        self.semantic.clear()
        self._episode_count = 0
