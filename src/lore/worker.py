"""Worker: executes a single subtask with its own scoped context.

Not a process or thread — just a scoped execution context that reuses the
existing ModelServer HTTP connection. Each worker gets its own ContextManager
with the subtask's budget and system prompt.
"""
import logging
import time

from lore.context import ContextManager
from lore.memory import HierarchicalMemory

logger = logging.getLogger(__name__)

# Dynamic temperature by output format — deterministic for code/json, creative for free text
TEMPERATURE_MAP = {
    "code_python": 0.1,
    "code_bash": 0.1,
    "json": 0.1,
    "free": 0.7,
}

# Default max_tokens if subtask doesn't specify
_DEFAULT_MAX_TOKENS = 2048


def _estimate_max_tokens(description: str, output_format: str) -> int:
    """Estimate output token budget from task description and format."""
    words = len(description.split())

    if output_format in ("code_python", "code_bash"):
        if words < 30:
            return 1024
        elif words < 80:
            return 2048
        else:
            return 4096
    elif output_format == "json":
        return 1024
    else:
        desc_lower = description.lower()
        if any(kw in desc_lower for kw in ("summarize", "brief", "short", "one sentence")):
            return 256
        elif any(kw in desc_lower for kw in ("explain", "describe", "list")):
            return 1024
        else:
            return 2048


class WorkerResult:
    """Result of a single worker execution."""

    __slots__ = ("subtask_id", "content", "success", "latency_ms",
                 "tokens_used", "model", "error")

    def __init__(self, subtask_id: str, content: str, success: bool,
                 latency_ms: float, tokens_used: int, model: str,
                 error: str | None = None):
        self.subtask_id = subtask_id
        self.content = content
        self.success = success
        self.latency_ms = latency_ms
        self.tokens_used = tokens_used
        self.model = model
        self.error = error

    def to_dict(self) -> dict:
        return {
            "subtask_id": self.subtask_id,
            "content": self.content,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "model": self.model,
            "error": self.error,
        }


class Worker:
    """Executes a single subtask with its own context.

    Reuses the existing ModelServer (HTTP to llama-server), just with a
    separate ContextManager scoped to the subtask's budget and system prompt.
    """

    def __init__(self, subtask, server, memory: HierarchicalMemory | None = None):
        self._subtask = subtask
        self._server = server
        self._memory = memory
        self._ctx = ContextManager(
            config={"working_context": subtask.context_budget},
            model_server=server,
            system_prompt=subtask.system_prompt,
            memory=None,    # workers don't do memory retrieval
            health=None,    # workers don't need health monitoring
        )
        # Override default max_tokens with a smarter estimate
        if subtask.max_tokens == _DEFAULT_MAX_TOKENS:
            subtask.max_tokens = _estimate_max_tokens(subtask.description, subtask.output_format)

    def run(self, previous_outputs: dict[str, str] | None = None) -> WorkerResult:
        """Execute this subtask.

        Args:
            previous_outputs: dict of {subtask_id: output} for dependent subtasks.
                              Injected into the user message as context.

        Returns:
            WorkerResult with content, success status, and metrics.
        """
        t0 = time.time()
        model = self._subtask.model
        prev_outputs = previous_outputs or {}
        temperature = TEMPERATURE_MAP.get(self._subtask.output_format, 0.7)

        # Build user message: inject previous outputs if this subtask depends on them
        if self._subtask.depends_on_outputs and prev_outputs:
            deps_text = "\n\n".join(
                f"[{sid}]:\n{out[:2000]}"
                for sid, out in prev_outputs.items()
            )
            user_msg = (
                f"Previous step results:\n{deps_text}\n\n"
                f"Now: {self._subtask.description}"
            )
        else:
            user_msg = self._subtask.description

        self._ctx.add_message("user", user_msg)
        messages = self._ctx.build_prompt(query=user_msg)

        try:
            result = self._server.chat(
                model,
                messages,
                max_tokens=self._subtask.max_tokens,
                temperature=temperature,
            )
            content = result["choices"][0]["message"]["content"]
            success = True
            error = None
        except Exception as e:
            # Specialist fallback → primary
            if model == "specialist":
                logger.warning(f"Specialist failed for {self._subtask.id}: {e}, retrying on primary")
                try:
                    result = self._server.chat(
                        "primary",
                        messages,
                        max_tokens=self._subtask.max_tokens,
                        temperature=temperature,
                    )
                    content = result["choices"][0]["message"]["content"]
                    success = True
                    error = None
                    model = "primary"
                except Exception as e2:
                    content = f"Error: {e2}"
                    success = False
                    error = str(e2)
            else:
                content = f"Error: {e}"
                success = False
                error = str(e)

        latency = (time.time() - t0) * 1000
        tokens_out = len(content.split())  # rough estimate

        # Store subtask summary to shared memory if configured
        if self._memory is not None and success:
            try:
                self._memory.episodic.store_summary(
                    f"Subtask {self._subtask.id}: {self._subtask.description[:100]}. "
                    f"Result: {content[:200]}"
                )
            except Exception as e:
                logger.debug(f"Memory store failed for subtask {self._subtask.id}: {e}")

        logger.info(
            f"Worker {self._subtask.id} ({model}): "
            f"{'ok' if success else 'FAIL'} {latency:.0f}ms {tokens_out} tokens"
        )

        return WorkerResult(
            subtask_id=self._subtask.id,
            content=content,
            success=success,
            latency_ms=latency,
            tokens_used=tokens_out,
            model=model,
            error=error,
        )

    def run_with_retry(self, max_retries: int = 1,
                       previous_outputs: dict[str, str] | None = None) -> WorkerResult:
        """Run subtask. Retry on generation errors, NOT on timeouts.

        On timeout: use partial output if available (>200 chars), or mark failed.
        On generation error: retry once with escalation (more tokens, error context).
        """
        result = self.run(previous_outputs=previous_outputs)
        if result.success:
            return result

        is_timeout = result.error and (
            "timeout" in str(result.error).lower()
            or "timed out" in str(result.error).lower()
        )

        if is_timeout:
            if result.content and len(result.content) >= 200:
                logger.warning(
                    f"Subtask {self._subtask.id} timed out with partial output "
                    f"({len(result.content)} chars), using it"
                )
                return WorkerResult(
                    subtask_id=result.subtask_id,
                    content=result.content,
                    success=True,
                    latency_ms=result.latency_ms,
                    tokens_used=result.tokens_used,
                    model=result.model,
                    error="timeout_with_partial_output",
                )
            logger.warning(f"Subtask {self._subtask.id} timed out with no useful output")
            return result

        # Generation error — retry once with escalation
        if max_retries > 0:
            logger.warning(
                f"Subtask {self._subtask.id} failed ({result.error}), retrying"
            )
            self._subtask.max_tokens = min(self._subtask.max_tokens * 2, 4096)
            self._subtask.system_prompt += (
                f"\n\nPrevious attempt failed: {result.error}. Be careful."
            )
            if self._subtask.model == "specialist":
                self._subtask.model = "primary"
            self._ctx = ContextManager(
                config={"working_context": self._subtask.context_budget},
                model_server=self._server,
                system_prompt=self._subtask.system_prompt,
                memory=None,
                health=None,
            )
            return self.run(previous_outputs=previous_outputs)

        return result
