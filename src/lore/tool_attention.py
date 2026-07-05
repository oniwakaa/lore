"""Tool Attention: lazy schema loading (NTILC pattern).

Instead of injecting the full tool registry into every prompt, embed each
schema once and select only the top-k most relevant to the current query.
"""
import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class ToolAttention:
    """Stores tool schemas + their embeddings, selects top-k by cosine similarity."""

    def __init__(self, model_server, tool_schemas: list[dict]):
        self._server = model_server
        self._schemas = tool_schemas
        self._embeddings = self._embed_schemas(tool_schemas)

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

    def select_tools(self, query: str, k: int = 3) -> list[dict]:
        """Return the top-k tool schemas most similar to query."""
        valid = [(s, e) for s, e in zip(self._schemas, self._embeddings) if e is not None]
        if not valid:
            return []

        try:
            query_emb = np.array(self._server.embed(query))
        except Exception as e:
            logger.warning(f"Failed to embed query for tool selection: {e}")
            return [s for s, _ in valid[:k]]  # fallback: first k, no ranking

        scored = []
        for schema, emb in valid:
            denom = np.linalg.norm(emb) * np.linalg.norm(query_emb)
            sim = float(emb @ query_emb / denom) if denom else 0.0
            scored.append((sim, schema))

        scored.sort(key=lambda x: -x[0])
        return [schema for _, schema in scored[:k]]

    @classmethod
    def from_config(cls, model_server, config_path: str = "configs/tools.yaml") -> "ToolAttention":
        """Load tool schemas from YAML. Missing file -> empty registry."""
        p = Path(config_path)
        schemas = yaml.safe_load(p.read_text()).get("tools", []) if p.exists() else []
        return cls(model_server, schemas)
