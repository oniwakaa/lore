# src/lore/context.py
"""Token-aware context manager. Budgets tokens across system prompt, memory, history."""
import logging

logger = logging.getLogger(__name__)

class ContextManager:
    """Manages conversation context within configurable token budget."""

    def __init__(self, config: dict, model_server, system_prompt: str = ""):
        self._config = config
        self._server = model_server
        self._system_prompt = system_prompt
        self._history: list[dict] = []
        self._truncated = False

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self._history.append({"role": role, "content": content})

    def build_prompt(self, memories: list[str] | None = None) -> list[dict]:
        """Build the full message list for a model request."""
        messages = [{"role": "system", "content": self._system_prompt}]

        # Inject memories if provided
        if memories:
            memory_text = "\n".join(f"- {m}" for m in memories)
            messages.append({"role": "system", "content": f"Relevant context:\n{memory_text}"})

        # Add truncated history
        history = self._truncate_to_budget(self._history)
        messages.extend(history)

        return messages

    def token_count(self, text: str) -> int:
        """Count tokens via model server /tokenize endpoint."""
        # ponytail: first optimization target for Phase 1.5 — local tokenizer cache
        # 4-6 HTTP calls per request at 5-20ms each = 40-120ms overhead
        try:
            return self._server.tokenize("primary", text)
        except Exception:
            # Fallback: rough estimate
            return len(text) // 4

    def _truncate_to_budget(self, messages: list[dict]) -> list[dict]:
        """Drop oldest messages when working_context budget exceeded."""
        budget = self._config.get("working_context", 4096)
        keep_last = 6  # always keep last 3 turns (6 messages)

        if len(messages) <= keep_last:
            return messages

        # Estimate total tokens (sum of all message contents)
        total = sum(self.token_count(m["content"]) for m in messages)

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
