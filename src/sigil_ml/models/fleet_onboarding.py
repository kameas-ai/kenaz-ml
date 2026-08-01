"""Fleet onboarding model — predict ramp-up trajectory for new team members."""

from __future__ import annotations

import io
import logging
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

from sigil_ml.modelstore import LocalModelStore, ModelStore

logger = logging.getLogger(__name__)


class FleetOnboardingModel:
    """Predicts ramp-up trajectory for new team members.

    Estimates how many days until a new member reaches 80% of team
    baseline productivity. Model artifacts stored per-team:
    ``fleet_onboarding_{team_id}``.
    """

    def __init__(self, team_id: int, model_store: ModelStore | None = None) -> None:
        self._store = model_store or LocalModelStore()
        self._team_id = team_id
        self.model: LinearRegression | None = None
        self._trained = False
        self._samples = 0

        data = self._store.load(self._model_name)
        if data is not None:
            try:
                loaded = joblib.load(io.BytesIO(data))
                self.model = loaded["model"]
                self._samples = loaded.get("samples", 0)
                self._trained = True
                logger.info("Loaded fleet onboarding model for team %d", team_id)
            except Exception:
                logger.warning("Failed to load fleet onboarding model for team %d", team_id)
                self.model = None

    @property
    def _model_name(self) -> str:
        return f"fleet_onboarding_{self._team_id}"

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Train on new-member ramp-up data.

        Each row should have: day_number, performance_pct (relative to
        team baseline).

        Returns:
            Training result with r2 metric.
        """
        if len(data) < 5:
            raise ValueError("Need at least 5 data points for training")

        features = []
        targets = []
        for row in data:
            features.append([row.get("day_number", 1)])
            targets.append(row.get("performance_pct", 50))

        X = np.array(features)
        y = np.array(targets)

        self.model = LinearRegression()
        self.model.fit(X, y)
        self._trained = True
        self._samples = len(data)

        r2 = float(self.model.score(X, y))

        buf = io.BytesIO()
        joblib.dump({"model": self.model, "samples": self._samples}, buf)
        self._store.save(self._model_name, buf.getvalue())
        logger.info("Saved fleet onboarding model for team %d (%d samples)", self._team_id, self._samples)

        return {
            "model": "fleet_onboarding",
            "team_id": self._team_id,
            "samples": self._samples,
            "metrics": {"r2": round(r2, 3)},
        }

    def predict(self) -> dict[str, Any]:
        """Predict ramp-up trajectory over 90 days.

        Returns:
            Dict with day-by-day trajectory and predicted ramp-up day.
        """
        if self.model is None or not self._trained:
            return {"trajectory": [], "predicted_ramp_up_days": None}

        trajectory = []
        ramp_up_day = None
        for day in range(1, 91):
            pct = float(self.model.predict(np.array([[day]]))[0])
            pct = max(0.0, min(100.0, pct))
            trajectory.append({"day": day, "predicted_pct": round(pct, 1)})
            if ramp_up_day is None and pct >= 80:
                ramp_up_day = day

        return {
            "trajectory": trajectory,
            "predicted_ramp_up_days": ramp_up_day,
            "samples": self._samples,
        }
