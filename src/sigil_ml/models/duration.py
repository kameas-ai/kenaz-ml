"""Duration estimator using GradientBoostingRegressor."""

from __future__ import annotations

import io
import logging
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from sigil_ml.modelstore import LocalModelStore, ModelStore

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "file_count",
    "total_edits",
    "time_of_day_hour",
    "branch_name_length",
]


class DurationEstimator:
    """Estimates task duration in minutes."""

    def __init__(self, model_store: ModelStore | None = None) -> None:
        self._store = model_store or LocalModelStore()
        self.model: GradientBoostingRegressor | None = None
        self._trained = False
        data = self._store.load("duration")
        if data is not None:
            try:
                self.model = joblib.load(io.BytesIO(data))
                self._trained = True
                logger.info("Loaded duration model from %s", type(self._store).__name__)
            except Exception:
                logger.warning("Failed to load duration model, starting fresh")
                self.model = None

    @classmethod
    def from_trained_model(cls, model: GradientBoostingRegressor, store: ModelStore | None = None) -> DurationEstimator:
        """Create an instance from an already-trained sklearn model.

        Use this instead of ``__new__`` to avoid bypassing ``__init__``.
        """
        instance = object.__new__(cls)
        instance._store = store or LocalModelStore()
        instance.model = model
        instance._trained = True
        return instance

    @property
    def is_trained(self) -> bool:
        return self._trained

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        """Predict task duration from feature dict.

        Returns:
            {"estimated_minutes": float, "confidence_interval": [low, high]}
        """
        if self.model is None or not self._trained:
            return {
                "estimated_minutes": 60.0,
                "confidence_interval": [30.0, 90.0],
            }

        x = np.array([[features.get(f, 0.0) for f in FEATURE_NAMES]])

        # Get predictions from each tree for confidence interval
        individual_predictions = []
        for tree in self.model.estimators_.flatten():
            individual_predictions.append(float(tree.predict(x)[0]))

        # The ensemble prediction
        mean_pred = float(self.model.predict(x)[0])

        # Confidence interval: +/- 1 std dev of tree predictions
        if len(individual_predictions) > 1:
            std = float(np.std(individual_predictions))
        else:
            std = mean_pred * 0.3  # fallback: 30% of estimate

        low = max(0.0, mean_pred - std)
        high = mean_pred + std

        return {
            "estimated_minutes": round(mean_pred, 1),
            "confidence_interval": [round(low, 1), round(high, 1)],
        }

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model and save weights.

        Args:
            X: Feature matrix of shape (n_samples, 4).
            y: Duration in minutes.
        """
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
        self.model.fit(X, y)
        self._trained = True

        buf = io.BytesIO()
        joblib.dump(self.model, buf)
        self._store.save("duration", buf.getvalue())
        logger.info("Saved duration model via %s", type(self._store).__name__)
