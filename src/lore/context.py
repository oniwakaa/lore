# src/lore/context.py
"""Token-aware context manager. Budgets tokens across system prompt, memory, history."""
import json
import logging

from lore.compression import compress_context

try:
    from tokenizers import Tokenizer
except ImportError:  # dependency missing, local counting disabled
    Tokenizer = None

logger = logging.getLogger(__name__)

class ContextManager:
    """Manages conversation context within configurable token budget."""

    def __init__(self, config: dict, model_server, system_prompt: str = "",
                 tokenizer_source: str = "local", tokenizer_repo: str | None = None,
                 tool_attention=None, compression: dict | None = None):
        self._config = config
        self._server = model_server
        self._system_prompt = system_prompt
        self._history: list[dict] = []
        self._truncated = False
        self._tokenizer = self._load_tokenizer(tokenizer_source, tokenizer_repo)
        self._tool_attention = tool_attention
        self._compression = compression or {"enabled": False}

    def _load_tokenizer(self, source: str, repo: str | None):
        """Load and cache a local HF tokenizer once. None means fall back to HTTP."""
        if source != "local" or not repo or Tokenizer is None:
            return None
        try:
            return Tokenizer.from_pretrained(repo)
        except Exception as e:
            logger.warning(f"Local tokenizer load failed ({e}), falling back to HTTP /tokenize")
            return None

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self._history.append({"role": role, "content": content})

    def build_prompt(self, memories: list[str] | None = None, query: str | None = None,
                      tool_k: int = 3) -> list[dict]:
        """Build the full message list for a model request.

        If a ToolAttention instance and query are provided, inject only the
        top-k relevant tool schemas instead of the full registry.
        """
        messages = [{"role": "system", "content": self._system_prompt}]

        # Inject memories if provided
        if memories:
            memory_text = "\n".join(f"- {m}" for m in memories)
            messages.append({"role": "system", "content": f"Relevant context:\n{memory_text}"})

        # Inject only the top-k relevant tool schemas (Tool Attention / NTILC pattern)
        if self._tool_attention is not None and query:
            tools = self._tool_attention.select_tools(query, k=tool_k)
            if tools:
                messages.append({"role": "system", "content": f"Available tools:\n{json.dumps(tools)}"})

        # Add truncated history
        history = self._truncate_to_budget(self._history)
        messages.extend(history)

        return messages

    def token_count(self, text: str) -> int:
        """Count tokens. Local tokenizer first, HTTP /tokenize fallback."""
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text).ids)
            except Exception as e:
                logger.warning(f"Local tokenizer encode failed ({e}), falling back to HTTP")
        try:
            return self._server.tokenize("primary", text)
        except Exception:
            # Fallback: rough estimate
            return len(text) // 4

    def _truncate_to_budget(self, messages: list[dict]) -> list[dict]:
        """Compress old messages, then drop oldest, when working_context budget exceeded."""
        budget = self._config.get("working_context", 4096)
        keep_last = 6  # always keep last 3 turns (6 messages)

        if len(messages) <= keep_last:
            return messages

        # Estimate total tokens (sum of all message contents)
        total = sum(self.token_count(m["content"]) for m in messages)

        # Soft degradation: compress everything except the latest 2 turns (4 messages)
        # before resorting to hard-dropping messages entirely.
        if self._compression.get("enabled", False) and total > budget * 0.8:
            keep_uncompressed = 4
            old, recent = messages[:-keep_uncompressed], messages[-keep_uncompressed:]
            if old:
                old = compress_context(
                    old,
                    ratio=self._compression.get("ratio", 0.5),
                    model_path=self._compression.get("model_path", "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"),
                    device_map=self._compression.get("device_map", "cpu"),
                )
                messages = old + recent
                total = sum(self.token_count(m["content"]) for m in messages)
                logger.info(f"Compressed {len(old)} old messages, {total} tokens remain")

        if total <= budget:
            return messages

        # Drop oldest until under budget
        self._truncated = True
        while len(messages) > keep_last:
            total -= self.token_count(messages[0]["content"])
            messages = messages[1:]
            if total <= budget:
                break

        logger.info(f"Context truncated to {len(messages)} messages ({total} tokens)")
        return messages

    def clear(self) -> None:
        """Clear conversation history."""
        self._history.clear()
        self._truncated = False

    @property
    def was_truncated(self) -> bool:
        return self._truncated
