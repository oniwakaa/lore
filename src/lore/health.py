"""Context health monitor. Detects when the model is drowning in stale context.

Tracks utilization, message age, repetition, and staleness. Returns a HealthReport
with metrics and a recommended action (ok, compress, summarize, prune, warn).
"""
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

_HEALTH_LOG = Path("logs/context_health.jsonl")


@dataclass
class HealthReport:
    """Context health metrics + recommended action."""
    context_utilization: float = 0.0  # token_usage / budget
    message_age_ratio: float = 0.0  # avg age / session length
    repetition_score: float = 0.0  # 0-1, near-duplicate fraction
    stale_context_ratio: float = 0.0  # % of context older than stale_after_turns
    compression_effectiveness: float = 0.0  # how much compression reduced tokens (0-1)
    action: str = "ok"  # ok | compress | summarize | prune | warn_degradation
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ContextHealth:
    """Monitors context quality metrics. Returns warnings and triggers actions."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._warn_threshold = cfg.get("warn_threshold", 0.80)
        self._critical_threshold = cfg.get("critical_threshold", 0.90)
        self._stale_after_turns = cfg.get("stale_after_turns", 10)
        self._check_every_n = cfg.get("check_every_n_turns", 5)
        self._turn_count = 0
        self._last_compression_ratio = 0.0

    def record_compression(self, before_tokens: int, after_tokens: int) -> None:
        """Track how much the last compression actually reduced tokens."""
        if before_tokens > 0:
            self._last_compression_ratio = 1.0 - (after_tokens / before_tokens)

    def should_check(self) -> bool:
        """Returns True every N turns to avoid per-turn overhead."""
        self._turn_count += 1
        return (self._turn_count % self._check_every_n) == 0

    def check(self, context: list[dict], token_usage: int, budget: int) -> HealthReport:
        """Check context health. Returns metrics + recommended action."""
        report = HealthReport()
        if not context or budget <= 0:
            return report

        # Utilization
        report.context_utilization = min(token_usage / budget, 1.0)

        n_messages = len(context)
        n_turns = n_messages // 2

        # Message age ratio: average position from end / total length
        # High ratio = most messages are old (stale)
        if n_messages > 0:
            ages = [n_messages - i for i in range(n_messages)]
            report.message_age_ratio = sum(ages) / (n_messages * n_messages) if n_messages > 0 else 0.0

        # Repetition score: fraction of messages that are near-duplicates
        report.repetition_score = self._compute_repetition(context)

        # Stale context ratio: % of messages older than stale_after_turns
        stale_threshold_msgs = self._stale_after_turns * 2
        if n_messages > stale_threshold_msgs:
            stale_count = n_messages - stale_threshold_msgs
            report.stale_context_ratio = stale_count / n_messages
        else:
            report.stale_context_ratio = 0.0

        # Compression effectiveness
        report.compression_effectiveness = self._last_compression_ratio

        # Determine action
        report.action = self._recommend_action(report)
        report.warnings = self._generate_warnings(report)

        # Log to JSONL for post-hoc analysis
        self._log_health(report)

        return report

    def _compute_repetition(self, context: list[dict]) -> float:
        """Compute fraction of messages that have a near-duplicate in the context."""
        contents = [m.get("content", "") for m in context]
        if len(contents) < 4:
            return 0.0
        duplicates = 0
        for i, text in enumerate(contents):
            # Quick check: does any other message share >60% of words?
            words_i = set(text.lower().split())
            if not words_i:
                continue
            for j in range(i + 1, len(contents)):
                words_j = set(contents[j].lower().split())
                if not words_j:
                    continue
                overlap = len(words_i & words_j) / min(len(words_i), len(words_j))
                if overlap > 0.6:
                    duplicates += 1
                    break  # count each message once
        return duplicates / len(contents) if contents else 0.0

    def _recommend_action(self, report: HealthReport) -> str:
        """Determine the recommended action based on metrics."""
        # Critical: context is almost full
        if report.context_utilization >= self._critical_threshold:
            if report.stale_context_ratio > 0.5:
                return "summarize"  # lots of old context → summarize into episodic
            return "prune"  # otherwise hard-drop oldest

        # Warning: context is getting full
        if report.context_utilization >= self._warn_threshold:
            if report.stale_context_ratio > 0.3:
                return "compress"  # compress old messages
            return "warn_degradation"

        # Quality degradation: lots of repetition even if not full
        if report.repetition_score > 0.5:
            return "warn_degradation"

        return "ok"

    def _generate_warnings(self, report: HealthReport) -> list[str]:
        """Generate human-readable warnings."""
        warnings = []
        if report.context_utilization >= self._critical_threshold:
            warnings.append(f"Context at {report.context_utilization:.0%} utilization — critical")
        elif report.context_utilization >= self._warn_threshold:
            warnings.append(f"Context at {report.context_utilization:.0%} utilization — warning")
        if report.stale_context_ratio > 0.5:
            warnings.append(f"{report.stale_context_ratio:.0%} of context is stale")
        if report.repetition_score > 0.5:
            warnings.append(f"High repetition score: {report.repetition_score:.0%}")
        return warnings

    def _log_health(self, report: HealthReport) -> None:
        """Append health metrics to logs/context_health.jsonl."""
        try:
            _HEALTH_LOG.parent.mkdir(exist_ok=True)
            entry = report.to_dict()
            entry["timestamp"] = time.time()
            with open(_HEALTH_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log context health: {e}")
