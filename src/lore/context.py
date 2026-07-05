"""Token-aware context manager. Budgets tokens across system prompt, memory, history.

Integrates hierarchical memory retrieval, context health monitoring, and
conditional compression. Health checks run every N turns and can trigger
compression or summarization actions.
"""
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
                 tool_attention=None, compression: dict | None = None,
                 memory=None, health=None):
        self._config = config
        self._server = model_server
        self._system_prompt = system_prompt
        self._history: list[dict] = []
        self._truncated = False
        self._tokenizer = self._load_tokenizer(tokenizer_source, tokenizer_repo)
        self._tool_attention = tool_attention
        self._compression = compression or {"enabled": False}
        self._memory = memory  # HierarchicalMemory or None
        self._health = health  # ContextHealth or None
        self._last_health_report = None

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

        If a HierarchicalMemory instance is provided, retrieve relevant
        episodic summaries + semantic facts by embedding similarity to the
        query and inject them as compressed context.

        If a ContextHealth instance is provided, check context health before
        building the prompt. If action is "summarize", trigger episodic
        summarization. If "compress", trigger LLMLingua-2.

        If a ToolAttention instance and query are provided, inject only the
        top-k relevant tool schemas instead of the full registry.
        """
        # 1. Health check: run every N turns, trigger actions if needed
        if self._health is not None and self._health.should_check():
            self._run_health_check()

        # 2. Retrieve relevant memories from hierarchical memory
        if self._memory is not None and query:
            retrieved = self._memory.retrieve(query)
            if retrieved:
                memories = (memories or []) + retrieved

        # 3. Build system message: base prompt + memories + tool schemas
        # Ornith's chat template requires a single system message at the start
        # (multiple system messages raise a template error) — fold memories and
        # tool schemas into it instead of appending separate system turns.
        system_parts = [self._system_prompt]

        if memories:
            memory_text = "\n".join(f"- {m}" for m in memories)
            system_parts.append(f"Relevant context:\n{memory_text}")

        # Inject only the top-k relevant tool schemas (Tool Attention / NTILC pattern)
        if self._tool_attention is not None and query:
            tools = self._tool_attention.select_tools(query, k=tool_k)
            if tools:
                system_parts.append(f"Available tools:\n{json.dumps(tools)}")

        messages = [{"role": "system", "content": "\n\n".join(system_parts)}]

        # 4. Add truncated history (with conditional compression)
        history = self._truncate_to_budget(self._history)
        messages.extend(history)

        return messages

    def _run_health_check(self) -> None:
        """Check context health and trigger actions (summarize, compress)."""
        budget = self._config.get("working_context", 4096)
        total = sum(self.token_count(m["content"]) for m in self._history)
        report = self._health.check(self._history, total, budget)
        self._last_health_report = report

        if report.action == "summarize" and self._memory is not None:
            # Trigger episodic summarization of oldest messages
            stale_threshold = self._health.stale_message_count()
            old_messages = self._history[:stale_threshold]
            if old_messages:
                summary = self._memory.maybe_summarize(old_messages)
                if summary:
                    logger.info(f"Health-triggered summarization: {summary[:80]}...")
                    # Remove summarized messages from working memory
                    self._history = self._history[stale_threshold:]

        elif report.action == "compress":
            # Force compression on next _truncate_to_budget call by ensuring gate passes
            # The gate checks min_turns and usage_ratio, which are already true
            # if health triggered "compress". Just log it.
            logger.info(f"Health recommends compression (utilization {report.context_utilization:.0%})")

        for warning in report.warnings:
            logger.warning(f"Context health: {warning}")

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
        """Compress old messages, then drop oldest, when working_context budget exceeded.

        Compression is gated on session maturity so the per-call overhead
        (~233ms LLMLingua-2 inference) doesn't dominate short sessions where
        savings are negligible. Activates only when ALL of these are true:
          - compression.enabled is True in config
          - session has >= min_turns turns (default 10)
          - context usage > 70% of budget
          - there are old messages to compress (beyond preserve_recent_turns)
        """
        budget = self._config.get("working_context", 4096)
        keep_last = 6  # always keep last 3 turns (6 messages) as floor

        if len(messages) <= keep_last:
            return messages

        # Estimate total tokens (sum of all message contents)
        total = sum(self.token_count(m["content"]) for m in messages)

        # Conditional compression gate: only when session is mature and budget pressured
        min_turns = self._compression.get("min_turns", 10)
        preserve = self._compression.get("preserve_recent_turns", 3)
        keep_uncompressed = preserve * 2  # turns -> messages
        usage_ratio = total / budget if budget > 0 else 0.0
        session_turns = len(messages) // 2
        has_old = len(messages) > keep_uncompressed

        if (self._compression.get("enabled", False)
                and session_turns >= min_turns
                and usage_ratio > 0.70
                and has_old):
            old, recent = messages[:-keep_uncompressed], messages[-keep_uncompressed:]
            if old:
                before_tokens = total
                old = compress_context(
                    old,
                    ratio=self._compression.get("ratio", 0.5),
                    model_path=self._compression.get("model_path", "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"),
                    device_map=self._compression.get("device_map", "cpu"),
                )
                messages = old + recent
                total = sum(self.token_count(m["content"]) for m in messages)
                # Record compression effectiveness for health monitoring
                if self._health is not None:
                    self._health.record_compression(before_tokens, total)
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

    def restore(self, system_prompt: str, history: list[dict]) -> None:
        """Restore context from saved session (used by SessionManager)."""
        self._system_prompt = system_prompt
        self._history = list(history)
        self._truncated = False

    @property
    def was_truncated(self) -> bool:
        return self._truncated

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    @property
    def last_health_report(self):
        """Last HealthReport from the health check, or None."""
        return self._last_health_report
