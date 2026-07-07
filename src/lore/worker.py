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
                temperature=0.7,
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
                        temperature=0.7,
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
