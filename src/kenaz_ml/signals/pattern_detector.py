"""Pattern detection via z-score deviation from personal behavioral baselines.

Cold start: per-metric rolling z-score with configurable threshold.
ML upgrade: IsolationForest after 500+ labeled feedback events.
"""

from __future__ import annotations

import logging
from typing import Any

from kenaz_ml.modelstore import ModelStore
from kenaz_ml.signals import Signal
from kenaz_ml.signals.profile import BehaviorProfile, RollingStat

logger = logging.getLogger(__name__)

# Minimum observations per metric before signals are emitted.
MIN_OBSERVATIONS = 50
DEFAULT_Z_THRESHOLD = 2.0


class PatternDetector:
    """Detects behavioral deviations from personal baselines.

    Uses per-metric z-score analysis in cold-start mode.
    Upgrades to IsolationForest after sufficient training data.
    """

    def __init__(
        self,
        z_threshold: float = DEFAULT_Z_THRESHOLD,
        min_observations: int = MIN_OBSERVATIONS,
    ) -> None:
        self._z_threshold = z_threshold
        self._min_observations = min_observations
        self._isolation_forest: Any = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """True if IsolationForest model is loaded and ready."""
        return self._is_trained

    def has_sufficient_data(self, profile: BehaviorProfile) -> bool:
        """Check if the profile has enough data for meaningful detection.

        Returns True if at least one metric has sufficient observations.
        """
        return any(stat.count >= self._min_observations for stat in profile.metrics.values())

    def detect(self, buffer: list[dict], profile: BehaviorProfile) -> list[Signal]:
        """Run pattern detection on the current event buffer.

        Args:
            buffer: Recent classified events from the poller.
            profile: Current user behavior profile.

        Returns:
            List of Signal objects for detected deviations.
        """
        if not self.has_sufficient_data(profile):
            logger.debug("pattern_detector: insufficient baseline data, skipping")
            return []

        if self._is_trained and self._isolation_forest is not None:
            return self._detect_ml(buffer, profile)
        return self._detect_zscore(buffer, profile)

    # --- Z-score detection (cold start) ---

    def _detect_zscore(self, buffer: list[dict], profile: BehaviorProfile) -> list[Signal]:
        """Detect deviations using per-metric z-scores."""
        signals: list[Signal] = []
        current_metrics = self._compute_current_metrics(buffer)

        for name, value in current_metrics.items():
            stat = profile.metrics.get(name)
            if stat is None or stat.count < self._min_observations:
                continue  # Not enough baseline data for this metric

            z = stat.z_score(value)
            if z is None:
                continue

            if abs(z) > self._z_threshold:
                confidence = min(abs(z) / 4.0, 0.95)
                signals.append(
                    Signal(
                        signal_type=f"{name}_deviation",
                        confidence=round(confidence, 4),
                        evidence=self._build_evidence(name, value, stat, z),
                        suggested_action=self._infer_action(name, z),
                    )
                )

        return signals

    def _compute_current_metrics(self, buffer: list[dict]) -> dict[str, float]:
        """Compute behavioral metrics from the current event buffer."""
        if not buffer:
            return {}

        metrics: dict[str, float] = {}
        now_ms = buffer[-1].get("ts", 0)
        first_ts = buffer[0].get("ts", now_ms)
        window_min = max((now_ms - first_ts) / 60000.0, 1 / 60.0)

        # Edit velocity: file events per minute
        edit_count = sum(1 for e in buffer if e.get("kind") == "file")
        if edit_count > 0:
            metrics["edit_velocity"] = edit_count / window_min

        # Context switch rate: category transitions
        transitions = 0
        for i in range(1, len(buffer)):
            if buffer[i].get("_category") != buffer[i - 1].get("_category"):
                transitions += 1
        if len(buffer) > 1:
            metrics["context_switch_rate"] = transitions / (len(buffer) - 1)

        # File focus: 1 - (unique_files / total_edits)
        edit_events = [e for e in buffer if e.get("kind") == "file"]
        if edit_events:
            files = set()
            for e in edit_events:
                p = e.get("payload") or {}
                if isinstance(p, dict) and "path" in p:
                    files.add(p["path"])
            metrics["file_focus"] = 1.0 - (len(files) / max(len(edit_events), 1))

        return metrics

    # --- Evidence generation ---

    def _build_evidence(self, metric: str, observed: float, stat: RollingStat, z: float) -> dict[str, Any]:
        """Build structured evidence dict for LLM rendering."""
        return {
            "source_model": "pattern_detector",
            "metric": metric,
            "observed": round(observed, 4),
            "baseline_mean": round(stat.mean, 4),
            "baseline_std": round(stat.std, 4),
            "z_score": round(z, 4),
            "observation_count": stat.count,
        }

    def _infer_action(self, metric: str, z: float) -> str | None:
        """Infer a generic suggested action from the metric and direction."""
        action_map = {
            "edit_velocity": "take_break" if z > 0 else None,
            "test_cadence": "test" if z > 0 else None,
            "commit_cadence": "commit" if z > 0 else None,
            "context_switch_rate": "investigate" if z > 0 else None,
            "file_focus": "investigate",
        }
        return action_map.get(metric)

    # --- IsolationForest ML upgrade path (stub) ---

    def train(self, feature_matrix: Any) -> None:
        """Train IsolationForest from feedback-labeled behavioral vectors.

        Called by Trainer (WP07) after 500+ labeled feedback events.

        Args:
            feature_matrix: numpy array of behavioral feature vectors.
        """
        from sklearn.ensemble import IsolationForest

        self._isolation_forest = IsolationForest(
            n_estimators=100,
            contamination="auto",
            random_state=42,
        )
        self._isolation_forest.fit(feature_matrix)
        self._is_trained = True
        logger.info(
            "PatternDetector: IsolationForest trained with %d samples",
            len(feature_matrix),
        )

    def save(self, model_store: ModelStore) -> None:
        """Persist trained IsolationForest via ModelStore."""
        if self._isolation_forest is None:
            return
        import io

        import joblib

        buf = io.BytesIO()
        joblib.dump(self._isolation_forest, buf)
        model_store.save("pattern_detector", buf.getvalue())

    def load(self, model_store: ModelStore) -> bool:
        """Load trained IsolationForest from ModelStore. Returns True if loaded."""
        data = model_store.load("pattern_detector")
        if data is None:
            return False
        import io

        import joblib

        self._isolation_forest = joblib.load(io.BytesIO(data))
        self._is_trained = True
        return True

    def _detect_ml(self, buffer: list[dict], profile: BehaviorProfile) -> list[Signal]:
        """ML-based detection using IsolationForest (stub).

        Active only after train() is called with sufficient data.
        Falls back to z-score if ML detection fails.
        """
        try:
            # TODO: Extract feature vector from buffer, run isolation forest
            # For now, fall back to z-score
            return self._detect_zscore(buffer, profile)
        except Exception:
            logger.debug("PatternDetector: ML detection failed, falling back to z-score")
            return self._detect_zscore(buffer, profile)
