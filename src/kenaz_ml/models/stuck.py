"""Stuck predictor using GradientBoostingClassifier."""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from kenaz_ml.modelstore import LocalModelStore, ModelStore
from kenaz_ml.modelstore.loader import resolve_for_serving  # not in the pinned package __all__

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kenaz_ml.modelstore.registry import Resolution

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "test_failure_count",
    "time_in_phase_sec",
    "edit_velocity",
    "file_switch_rate",
    "session_length_sec",
    "time_since_last_commit_sec",
]


class StuckPredictor:
    """Predicts whether a developer is stuck on a task."""

    def __init__(self, model_store: ModelStore | None = None, *, registry: bool = False) -> None:
        self._store = model_store or LocalModelStore()
        self.model: GradientBoostingClassifier | None = None
        self._trained = False
        self.resolution: Resolution | None = None
        self._load_stuck(registry)

    def _load_stuck(self, registry: bool) -> None:
        """Load the persisted stuck model — through the registry when the store is a filesystem.

        two-client-engine-01MSK2EN WP01 (FR-001, FR-022): a filesystem-backed
        store resolves local slot -> base slot -> cold start with integrity,
        ordered-contract and runtime checks before deserialization, and a
        pre-registry artifact (no manifest) is migrated in place. The outcome is
        kept on ``self.resolution`` for ``/introspect`` and ``/health``. Any
        other store (S3, a test double) keeps the legacy byte-load path.
        Neither path raises: no usable artifact means untrained, as before.

        Opt-in (``registry=True``), set by the serving path
        (``AppState.load_models``). The trainers also construct predictors --
        inside the window where the stale manifest has been cleared and the new
        artifact not yet written -- and a constructor-time migration there would
        write a synthesized manifest for the *old* bytes, the mismatched pair the
        trainer's write ordering exists to prevent. Training therefore keeps the
        legacy load (whose result ``train()`` discards anyway).
        """
        resolution = resolve_for_serving(self._store, "stuck") if registry else None
        if resolution is not None:
            self.resolution = resolution
            if resolution.served:
                self.model = resolution.model
                self._trained = True
                logger.info("Loaded stuck model from %s (%s slot)", type(self._store).__name__, resolution.slot)
            return

        data = self._store.load("stuck")
        if data is not None:
            try:
                self.model = joblib.load(io.BytesIO(data))
                self._trained = True
                logger.info("Loaded stuck model from %s", type(self._store).__name__)
            except Exception:
                logger.warning("Failed to load stuck model weights, starting fresh")
                self.model = None

    @classmethod
    def from_trained_model(cls, model: GradientBoostingClassifier, store: ModelStore | None = None) -> StuckPredictor:
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
        """Predict stuck probability from feature dict.

        Returns:
            {"probability": float, "confidence": "weak"|"moderate"|"strong"}
        """
        if self.model is None or not self._trained:
            return {"probability": 0.5, "confidence": "weak"}

        x = np.array([[features.get(f, 0.0) for f in FEATURE_NAMES]])
        prob = float(self.model.predict_proba(x)[0, 1])

        if prob < 0.4:
            confidence = "weak"
        elif prob < 0.7:
            confidence = "moderate"
        else:
            confidence = "strong"

        return {"probability": round(prob, 4), "confidence": confidence}

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model and save weights.

        Args:
            X: Feature matrix of shape (n_samples, 6).
            y: Binary labels (0 = not stuck, 1 = stuck).
        """
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
        self.model.fit(X, y)
        self._trained = True

        buf = io.BytesIO()
        joblib.dump(self.model, buf)
        self._store.save("stuck", buf.getvalue())
        logger.info("Saved stuck model via %s", type(self._store).__name__)
