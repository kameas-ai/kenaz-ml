"""Next-action prediction using n-gram models on composite action tokens.

Learns typical action sequences from event history. Emits divergence
signals when the user's actual behavior has low predicted probability.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

from kenaz_ml.modelstore import ModelStore
from kenaz_ml.signals import Signal
from kenaz_ml.signals.profile import BehaviorProfile

logger = logging.getLogger(__name__)

# Minimum events before the predictor starts emitting signals.
MIN_EVENTS_FOR_PREDICTION = 1000
# Divergence threshold: signal when actual probability < this.
DIVERGENCE_THRESHOLD = 0.05
# Minimum total n-gram count for a context to be predictive.
MIN_CONTEXT_COUNT = 10


class NextActionPredictor:
    """Predicts likely next actions from recent event sequences.

    Uses n-gram frequency tables (order 3 with backoff to 2 and 1)
    on composite action tokens ({category}:{tool}).
    """

    def __init__(self, n: int = 3) -> None:
        self._n = n
        self._ngrams: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self._total_tokens: int = 0

    def check_divergence(self, buffer: list[dict], profile: BehaviorProfile) -> list[Signal]:
        """Check for divergence between predicted and actual behavior.

        Args:
            buffer: Recent classified events from the poller.
            profile: Current user behavior profile.

        Returns:
            List of Signal objects for detected divergences.
        """
        if self._total_tokens < MIN_EVENTS_FOR_PREDICTION:
            return []

        tokens = self._extract_tokens(buffer)
        if len(tokens) < self._n:
            return []

        return self._check_latest_divergence(tokens, profile)

    # --- N-gram prediction ---

    def predict(self, recent_tokens: list[str]) -> dict[str, float] | None:
        """Predict probability distribution over next token.

        Uses n-gram with backoff: tries order n, then n-1, then 1.

        Args:
            recent_tokens: Recent composite action tokens.

        Returns:
            Dict mapping token -> probability for top predictions,
            or None if no prediction is possible.
        """
        # Try decreasing context lengths
        for order in range(self._n, 0, -1):
            if len(recent_tokens) < order - 1:
                continue
            context = tuple(recent_tokens[-(order - 1) :]) if order > 1 else ()
            counts = self._ngrams.get(context)
            if counts is None:
                continue
            total = sum(counts.values())
            if total < MIN_CONTEXT_COUNT:
                continue
            return {token: count / total for token, count in counts.most_common(10)}
        return None

    # --- Incremental training ---

    def train_incremental(self, tokens: list[str]) -> None:
        """Update n-gram tables from a token sequence.

        Called on every poll cycle with the full buffer's tokens.
        For full rebuild (training), call with all historical tokens.

        Args:
            tokens: List of composite action tokens.
        """
        if len(tokens) < self._n:
            return

        # Build n-grams for all orders (1 through n)
        for order in range(1, self._n + 1):
            for i in range(len(tokens) - order + 1):
                if order > 1:
                    context = tuple(tokens[i : i + order - 1])
                    next_token = tokens[i + order - 1]
                else:
                    context = ()
                    next_token = tokens[i]
                self._ngrams[context][next_token] += 1

        self._total_tokens += len(tokens)

    def reset(self) -> None:
        """Clear all n-gram tables for full retraining."""
        self._ngrams = defaultdict(Counter)
        self._total_tokens = 0

    # --- Divergence detection ---

    def _check_latest_divergence(self, tokens: list[str], profile: BehaviorProfile) -> list[Signal]:
        """Check if the most recent action diverges from prediction."""
        signals: list[Signal] = []

        if len(tokens) < self._n:
            return signals

        # Get prediction for what the next token should be
        context_tokens = tokens[:-1]
        actual_token = tokens[-1]
        prediction = self.predict(context_tokens)

        if prediction is None:
            return signals  # No confident prediction possible

        # Check if actual action has low predicted probability
        actual_prob = prediction.get(actual_token, 0.0)
        top_predicted = max(prediction, key=prediction.get)  # type: ignore[arg-type]
        top_prob = prediction[top_predicted]

        if actual_prob < DIVERGENCE_THRESHOLD and top_prob > 0.3:
            # Divergence: actual action was unexpected AND we had a confident prediction
            confidence = min((top_prob - actual_prob) / top_prob, 0.95)
            signals.append(
                Signal(
                    signal_type="action_divergence",
                    confidence=round(confidence, 4),
                    evidence={
                        "source_model": "next_action",
                        "predicted_action": top_predicted,
                        "predicted_probability": round(top_prob, 4),
                        "actual_action": actual_token,
                        "actual_probability": round(actual_prob, 4),
                        "sequence_length": len(tokens),
                        "context": {
                            "recent_tokens": tokens[-5:],
                        },
                    },
                    suggested_action=self._action_hint(top_predicted),
                )
            )

        return signals

    def _action_hint(self, predicted_token: str) -> str | None:
        """Infer a suggested action from the predicted token."""
        category = predicted_token.split(":")[0] if ":" in predicted_token else predicted_token
        hint_map = {
            "verifying": "test",
            "integrating": "commit",
            "navigating": None,
            "researching": None,
            "editing": None,
            "idle": "take_break",
        }
        return hint_map.get(category)

    # --- Token extraction ---

    def _extract_tokens(self, buffer: list[dict]) -> list[str]:
        """Convert an event buffer into a list of composite action tokens."""
        from kenaz_ml.features import extract_action_token

        return [extract_action_token(e) for e in buffer]

    # --- Model persistence ---

    def save(self, model_store: ModelStore) -> None:
        """Persist n-gram tables via ModelStore."""
        import io

        import joblib

        data = {
            "ngrams": dict(self._ngrams),  # Convert defaultdict to regular dict
            "total_tokens": self._total_tokens,
            "n": self._n,
        }
        buf = io.BytesIO()
        joblib.dump(data, buf)
        model_store.save("next_action", buf.getvalue())
        logger.info(
            "NextActionPredictor: saved %d n-gram contexts",
            len(self._ngrams),
        )

    def load(self, model_store: ModelStore) -> bool:
        """Load n-gram tables from ModelStore. Returns True if loaded."""
        raw = model_store.load("next_action")
        if raw is None:
            return False
        import io

        import joblib

        try:
            data = joblib.load(io.BytesIO(raw))
            loaded_ngrams = data.get("ngrams", {})
            self._ngrams = defaultdict(Counter)
            for context, counts in loaded_ngrams.items():
                self._ngrams[context] = Counter(counts)
            self._total_tokens = data.get("total_tokens", 0)
            self._n = data.get("n", self._n)
            logger.info(
                "NextActionPredictor: loaded %d contexts, %d total tokens",
                len(self._ngrams),
                self._total_tokens,
            )
            return True
        except Exception:
            logger.warning("NextActionPredictor: failed to load model", exc_info=True)
            return False

    # --- Serialization helpers (dict-based, no ModelStore) ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize n-gram state to a JSON-compatible dict."""
        # Convert tuple keys to string representation for JSON
        serialized_ngrams: dict[str, dict[str, int]] = {}
        for context, counts in self._ngrams.items():
            key = "|".join(context) if context else ""
            serialized_ngrams[key] = dict(counts)
        return {
            "ngrams": serialized_ngrams,
            "total_tokens": self._total_tokens,
            "n": self._n,
        }

    @classmethod
    def from_dict(cls, data: dict) -> NextActionPredictor:
        """Restore from a serialized dict."""
        n = data.get("n", 3)
        pred = cls(n=n)
        pred._total_tokens = data.get("total_tokens", 0)
        serialized_ngrams = data.get("ngrams", {})
        for key, counts in serialized_ngrams.items():
            context = tuple(key.split("|")) if key else ()
            pred._ngrams[context] = Counter(counts)
        return pred
