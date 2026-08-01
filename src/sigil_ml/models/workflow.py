"""Workflow state predictor — assesses user flow state from classified events."""

from __future__ import annotations

import io
import logging
import math

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from sigil_ml.features import extract_workflow_features
from sigil_ml.modelstore import LocalModelStore, ModelStore

logger = logging.getLogger(__name__)

FLOW_STATES = [
    "deep_work",
    "shallow_work",
    "exploring",
    "blocked",
    "winding_down",
]


class WorkflowStatePredictor:
    """Predicts workflow state from a window of classified events.

    Outputs a probability distribution over flow states plus momentum,
    focus score, and activity distribution. Starts rule-based, upgrades
    to GradientBoostingClassifier after sufficient training data.

    Flow states:
        deep_work     — sustained focused activity on few files
        shallow_work  — steady but scattered activity
        exploring     — high navigation/research, low creation
        blocked       — repeated verify-fail-edit cycles
        winding_down  — decreasing velocity, more integration
    """

    def __init__(self, model_store: ModelStore | None = None) -> None:
        self._store = model_store or LocalModelStore()
        self._ml_model: GradientBoostingClassifier | None = None
        self._trained = False

        data = self._store.load("workflow")
        if data is not None:
            try:
                self._ml_model = joblib.load(io.BytesIO(data))
                self._trained = True
                logger.info("Loaded workflow model from %s", type(self._store).__name__)
            except Exception:
                logger.warning("Failed to load workflow model, using rules")
                self._ml_model = None

    @classmethod
    def from_trained_model(
        cls, model: GradientBoostingClassifier, store: ModelStore | None = None
    ) -> WorkflowStatePredictor:
        """Create an instance from an already-trained sklearn model.

        Use this instead of ``__new__`` to avoid bypassing ``__init__``.
        """
        instance = object.__new__(cls)
        instance._store = store or LocalModelStore()
        instance._ml_model = model
        instance._trained = True
        return instance

    @property
    def is_trained(self) -> bool:
        return self._trained

    def predict(self, classified_events: list[dict], session_info: dict) -> dict:
        """Predict workflow state from classified events.

        Args:
            classified_events: List of event dicts, each with at least
                '_category' key (from ActivityClassifier).
            session_info: Dict with 'session_elapsed_min', 'task_phase',
                'test_failures'.

        Returns:
            Full state assessment dict with flow_state probabilities,
            dominant_state, momentum, focus_score, activity_distribution, etc.
        """
        if self._trained and self._ml_model is not None:
            return self._predict_ml(classified_events, session_info)
        return self._predict_rules(classified_events, session_info)

    def _predict_rules(self, classified_events: list[dict], session_info: dict) -> dict:
        """Rule-based state prediction from activity distributions."""
        activity_dist = self._activity_distribution(classified_events)
        momentum = self._compute_momentum(classified_events)
        focus = self._compute_focus(activity_dist)
        dominant_activity = max(activity_dist, key=lambda k: activity_dist[k]) if activity_dist else "idle"
        session_min = session_info.get("session_elapsed_min", 0.0)
        test_failures = session_info.get("test_failures", 0)

        # Rule-based flow state probabilities.
        probs = {s: 0.0 for s in FLOW_STATES}

        editing = activity_dist.get("editing", 0) + activity_dist.get("creating", 0) + activity_dist.get("refining", 0)
        verifying = activity_dist.get("verifying", 0)
        navigating = activity_dist.get("navigating", 0)
        researching = activity_dist.get("researching", 0)
        integrating = activity_dist.get("integrating", 0)

        # deep_work: high editing, low navigating, good focus.
        if editing > 0.4 and navigating < 0.2 and focus > 0.5:
            probs["deep_work"] = 0.7
            probs["shallow_work"] = 0.2
        # blocked: high verifying with test failures.
        elif verifying > 0.3 and test_failures > 2:
            probs["blocked"] = 0.7
            probs["shallow_work"] = 0.2
        # exploring: high navigating + researching.
        elif (navigating + researching) > 0.4:
            probs["exploring"] = 0.7
            probs["shallow_work"] = 0.2
        # winding_down: high integrating, negative momentum.
        elif integrating > 0.3 and momentum < -0.2:
            probs["winding_down"] = 0.6
            probs["shallow_work"] = 0.3
        # default: shallow_work.
        else:
            probs["shallow_work"] = 0.6
            probs["deep_work"] = 0.2

        # Fill remaining probability to sum to 1.0.
        total = sum(probs.values())
        remainder = 1.0 - total
        if remainder > 0:
            # Spread remainder across zero-prob states.
            zero_states = [s for s in FLOW_STATES if probs[s] == 0.0]
            if zero_states:
                each = remainder / len(zero_states)
                for s in zero_states:
                    probs[s] = round(each, 4)

        # Normalize to ensure sum = 1.0.
        total = sum(probs.values())
        if total > 0:
            probs = {s: round(v / total, 4) for s, v in probs.items()}

        dominant_state = max(probs, key=lambda k: probs[k])

        return {
            "flow_state": probs,
            "dominant_state": dominant_state,
            "momentum": round(momentum, 4),
            "focus_score": round(focus, 4),
            "dominant_activity": dominant_activity,
            "activity_distribution": {k: round(v, 4) for k, v in activity_dist.items()},
            "session_elapsed_min": round(session_min, 1),
            "method": "rules",
            "confidence": 0.5,
        }

    def _predict_ml(self, classified_events: list[dict], session_info: dict) -> dict:
        """ML-based state prediction using trained GradientBoostingClassifier."""
        features = extract_workflow_features(classified_events, session_info)
        feature_names = sorted(features.keys())
        x = np.array([[features[f] for f in feature_names]])

        try:
            predicted = self._ml_model.predict(x)[0]
            proba = self._ml_model.predict_proba(x)[0]
            classes = list(self._ml_model.classes_)

            flow_state = {s: 0.0 for s in FLOW_STATES}
            for i, cls in enumerate(classes):
                if cls in flow_state:
                    flow_state[cls] = round(float(proba[i]), 4)

            confidence = float(max(proba))
        except Exception:
            logger.debug("ML prediction failed, falling back to rules", exc_info=True)
            return self._predict_rules(classified_events, session_info)

        activity_dist = self._activity_distribution(classified_events)
        momentum = self._compute_momentum(classified_events)
        focus = self._compute_focus(activity_dist)
        dominant_activity = max(activity_dist, key=lambda k: activity_dist[k]) if activity_dist else "idle"

        return {
            "flow_state": flow_state,
            "dominant_state": predicted,
            "momentum": round(momentum, 4),
            "focus_score": round(focus, 4),
            "dominant_activity": dominant_activity,
            "activity_distribution": {k: round(v, 4) for k, v in activity_dist.items()},
            "session_elapsed_min": round(session_info.get("session_elapsed_min", 0.0), 1),
            "method": "ml",
            "confidence": round(confidence, 4),
        }

    @staticmethod
    def _activity_distribution(classified_events: list[dict]) -> dict[str, float]:
        """Compute normalized activity category distribution."""
        counts: dict[str, int] = {}
        for e in classified_events:
            cat = e.get("_category", "idle")
            counts[cat] = counts.get(cat, 0) + 1

        total = sum(counts.values())
        if total == 0:
            return {"idle": 1.0}

        return {k: v / total for k, v in counts.items()}

    @staticmethod
    def _compute_momentum(classified_events: list[dict]) -> float:
        """Compare event rate in recent half vs older half.

        Returns value in [-1, 1]. Positive = accelerating.
        """
        n = len(classified_events)
        if n < 2:
            return 0.0

        mid = n // 2
        older = mid
        recent = n - mid

        if older == 0:
            return 0.0

        return max(-1.0, min(1.0, (recent - older) / max(n, 1)))

    @staticmethod
    def _compute_focus(activity_dist: dict[str, float]) -> float:
        """Compute focus score as inverse of Shannon entropy, normalized to [0, 1].

        High focus = activity concentrated in few categories.
        Low focus = activity spread across many categories.
        """
        values = [v for v in activity_dist.values() if v > 0]
        if len(values) <= 1:
            return 1.0

        entropy = -sum(p * math.log2(p) for p in values)
        max_entropy = math.log2(len(values))

        if max_entropy == 0:
            return 1.0

        return max(0.0, min(1.0, 1.0 - entropy / max_entropy))

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the workflow state predictor.

        Args:
            X: Feature matrix from extract_workflow_features.
            y: Flow state labels (strings from FLOW_STATES).
        """
        self._ml_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
        self._ml_model.fit(X, y)
        self._trained = True

        buf = io.BytesIO()
        joblib.dump(self._ml_model, buf)
        self._store.save("workflow", buf.getvalue())
        logger.info("Saved workflow model via %s", type(self._store).__name__)
