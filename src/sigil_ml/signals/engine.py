"""SignalEngine: orchestrates all signal models on each poll cycle.

Called by the EventPoller on every poll cycle. Runs the BehaviorProfile
update, PatternDetector, NextActionPredictor, and FileRecommender.
Writes signals to ml_signals via the DataStore.
"""

from __future__ import annotations

import logging
import time

from sigil_ml.datastore import DataStore
from sigil_ml.features import extract_action_token
from sigil_ml.signals import Signal
from sigil_ml.signals.file_recommender import FileRecommender
from sigil_ml.signals.next_action import NextActionPredictor
from sigil_ml.signals.pattern_detector import PatternDetector
from sigil_ml.signals.profile import BehaviorProfile

logger = logging.getLogger(__name__)

# Rate limiting constants
SIGNAL_TYPE_COOLDOWN_SEC = 300  # 5 minutes per signal type
SIGNAL_TOTAL_WINDOW_SEC = 300  # 5-minute window for total count
SIGNAL_TOTAL_MAX = 10  # Max 10 signals per window
DISMISSED_COOLDOWN_SEC = 1800  # 30-minute cooldown after dismissal


class SignalEngine:
    """Orchestrates signal detection across all models.

    Called on every poll cycle (0.5s). Runs profile updates and
    all signal models. Writes emitted signals to the DataStore.
    """

    def __init__(
        self,
        store: DataStore,
        profile: BehaviorProfile,
        pattern_detector: PatternDetector,
        next_action: NextActionPredictor,
        file_recommender: FileRecommender,
    ) -> None:
        self.store = store
        self.profile = profile
        self.pattern_detector = pattern_detector
        self.next_action = next_action
        self.file_recommender = file_recommender
        # Rate limiting state
        self._recent_signals: list[tuple[str, float]] = []  # (signal_type, timestamp)
        self._dismissed_types: dict[str, float] = {}  # signal_type -> dismissed_at
        # Double-counting prevention: track last-seen event ID
        self._last_profile_event_id: int = 0
        self._last_ngram_event_id: int = 0

    def process_events(
        self,
        buffer: list[dict],
        task_context: dict | None = None,
    ) -> int:
        """Run all signal models on the current event buffer.

        Called by the poller on every poll cycle. Writes signals
        immediately to the DataStore.

        Args:
            buffer: Classified event buffer (last 200 events).
            task_context: Active task info (task_id, repo_root, etc.).

        Returns:
            Number of signals written.
        """
        if not buffer:
            return 0

        try:
            return self._process_events_inner(buffer, task_context)
        except Exception:
            logger.debug("signal_engine: error in signal processing", exc_info=True)
            return 0

    def _process_events_inner(self, buffer: list[dict], task_context: dict | None) -> int:
        """Inner processing loop -- separated for error isolation."""
        # 1. Update behavior profile (only unseen events)
        new_profile_events = [e for e in buffer if e.get("id", 0) > self._last_profile_event_id]
        if new_profile_events:
            self.profile.update(new_profile_events)
            self._last_profile_event_id = new_profile_events[-1].get("id", 0)

        # 2. Update n-gram model incrementally (only unseen events)
        new_ngram_events = [e for e in buffer if e.get("id", 0) > self._last_ngram_event_id]
        if new_ngram_events:
            tokens = [extract_action_token(e) for e in new_ngram_events]
            self.next_action.train_incremental(tokens)
            self._last_ngram_event_id = new_ngram_events[-1].get("id", 0)

        # 3. Collect signals from all models
        signals: list[Signal] = []
        signals.extend(self.pattern_detector.detect(buffer, self.profile))
        signals.extend(self.next_action.check_divergence(buffer, self.profile))
        signals.extend(self.file_recommender.check(buffer, task_context, self.profile))

        # 4. Apply rate limiting and cooldown
        filtered = self._apply_rate_limits(signals)

        # 5. Write signals to DataStore
        written = 0
        for signal in filtered:
            try:
                signal_id = self.store.insert_signal(
                    signal_type=signal.signal_type,
                    confidence=signal.confidence,
                    evidence=signal.evidence,
                    suggested_action=signal.suggested_action,
                    ttl_sec=signal.ttl_sec,
                )
                self._record_signal(signal.signal_type)
                written += 1
                logger.info(
                    "signal: type=%s confidence=%.2f id=%d",
                    signal.signal_type,
                    signal.confidence,
                    signal_id,
                )
            except Exception:
                logger.debug("signal_engine: failed to write signal", exc_info=True)

        return written

    # --- Rate limiting ---

    def _apply_rate_limits(self, signals: list[Signal]) -> list[Signal]:
        """Filter signals through rate limiting and cooldown rules."""
        now = time.time()
        self._prune_old_records(now)

        filtered: list[Signal] = []
        for signal in signals:
            # Check dismissed cooldown
            if self._is_type_dismissed(signal.signal_type, now):
                logger.debug("signal: suppressed (dismissed) type=%s", signal.signal_type)
                continue

            # Check per-type rate limit
            if self._is_type_rate_limited(signal.signal_type, now):
                logger.debug("signal: rate-limited type=%s", signal.signal_type)
                continue

            # Check total rate limit
            if self._is_total_rate_limited(now):
                logger.debug("signal: total rate limit reached")
                break  # Stop processing remaining signals

            filtered.append(signal)

        return filtered

    def _is_type_rate_limited(self, signal_type: str, now: float) -> bool:
        """Check if this signal type has been emitted within the cooldown window."""
        for st, ts in self._recent_signals:
            if st == signal_type and (now - ts) < SIGNAL_TYPE_COOLDOWN_SEC:
                return True
        return False

    def _is_total_rate_limited(self, now: float) -> bool:
        """Check if total signal count exceeds the window limit."""
        recent_count = sum(1 for _, ts in self._recent_signals if (now - ts) < SIGNAL_TOTAL_WINDOW_SEC)
        return recent_count >= SIGNAL_TOTAL_MAX

    def _record_signal(self, signal_type: str) -> None:
        """Record that a signal was emitted (for rate limiting)."""
        self._recent_signals.append((signal_type, time.time()))

    def _prune_old_records(self, now: float) -> None:
        """Remove expired rate limiting records."""
        cutoff = now - max(SIGNAL_TYPE_COOLDOWN_SEC, SIGNAL_TOTAL_WINDOW_SEC)
        self._recent_signals = [(st, ts) for st, ts in self._recent_signals if ts > cutoff]

    # --- Dismissed signal cooldown ---

    def _is_type_dismissed(self, signal_type: str, now: float) -> bool:
        """Check if this signal type is in dismissed cooldown."""
        dismissed_at = self._dismissed_types.get(signal_type)
        if dismissed_at is None:
            return False
        if (now - dismissed_at) > DISMISSED_COOLDOWN_SEC:
            del self._dismissed_types[signal_type]
            return False
        return True

    def refresh_dismissed_types(self) -> None:
        """Refresh dismissed signal types from feedback data.

        Reads recent feedback from the DataStore and updates the
        cooldown state. Called periodically (e.g., every prediction cycle).
        """
        now = time.time()
        now_ms = int(now * 1000)
        since_ms = now_ms - (DISMISSED_COOLDOWN_SEC * 1000)

        try:
            feedback = self.store.get_signal_feedback(since_ms)
            for fb in feedback:
                if fb["status"] == "dismissed":
                    signal_type = fb["signal_type"]
                    dismissed_at = fb["created_at"] / 1000.0
                    self._dismissed_types[signal_type] = dismissed_at
        except Exception:
            logger.debug("signal_engine: failed to refresh dismissed types", exc_info=True)
