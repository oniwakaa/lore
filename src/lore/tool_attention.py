"""Tool Attention: lazy schema loading (NTILC pattern).

Instead of injecting the full tool registry into every prompt, embed each
schema once and select only the top-k most relevant to the current query.

Below `min_tools_for_attention` (default 15), the embed() round-trip overhead
exceeds the token savings, so the full registry is returned without embedding.
"""
import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class ToolAttention:
    """Stores tool schemas + their embeddings, selects top-k by cosine similarity."""

    def __init__(self, model_server, tool_schemas: list[dict],
                 min_tools_for_attention: int = 15, default_k: int = 3):
        self._server = model_server
        self._schemas = tool_schemas
        self._min_tools = min_tools_for_attention
        self._default_k = default_k
        # Only embed schemas when the registry is large enough to justify
        # embedding-based selection. Below the gate, select_tools() returns
        # all tools without any embed() call, so skip the one-time embedding
        # cost entirely.
        if len(tool_schemas) > min_tools_for_attention:
            self._embeddings = self._embed_schemas(tool_schemas)
        else:
            self._embeddings = [None] * len(tool_schemas)

    @property
    def registry_size(self) -> int:
        return len(self._schemas)

    def _embed_schemas(self, schemas: list[dict]) -> list[np.ndarray | None]:
        embeddings = []
        for schema in schemas:
            text = f"{schema.get('name', '')}: {schema.get('description', '')}"
            try:
                embeddings.append(np.array(self._server.embed(text)))
            except Exception as e:
                logger.warning(f"Failed to embed tool schema {schema.get('name')}: {e}")
                embeddings.append(None)
        return embeddings

    def select_tools(self, query: str, k: int | None = None) -> list[dict]:
        """Return the top-k tool schemas most similar to query.

        If the registry is at or below `min_tools_for_attention`, return ALL
        schemas without any embed() call — the per-query embedding overhead
        exceeds the token savings at small registry sizes.
        """
        effective_k = k or self._default_k

        # Size gate: below threshold, inject all tools (skip embed round-trip)
        if len(self._schemas) <= self._min_tools:
            return list(self._schemas)

        valid = [(s, e) for s, e in zip(self._schemas, self._embeddings) if e is not None]
        if not valid:
            return []

        try:
            query_emb = np.array(self._server.embed(query))
        except Exception as e:
            logger.warning(f"Failed to embed query for tool selection: {e}")
            return [s for s, _ in valid[:effective_k]]  # fallback: first k, no ranking

        scored = []
        for schema, emb in valid:
            denom = np.linalg.norm(emb) * np.linalg.norm(query_emb)
            sim = float(emb @ query_emb / denom) if denom else 0.0
            scored.append((sim, schema))

        scored.sort(key=lambda x: -x[0])
        return [schema for _, schema in scored[:effective_k]]

    @classmethod
    def from_config(cls, model_server, config_path: str = "configs/tools.yaml") -> "ToolAttention":
        """Load tool schemas + gating config from YAML. Missing file -> empty registry."""
        p = Path(config_path)
        if p.exists():
            data = yaml.safe_load(p.read_text()) or {}
        else:
            data = {}
        schemas = data.get("tools", [])
        min_tools = data.get("min_tools_for_attention", 15)
        default_k = data.get("default_k", 3)
        return cls(model_server, schemas, min_tools_for_attention=min_tools, default_k=default_k)
