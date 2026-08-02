"""Fleet team focus model — predict focus score by hour/day for a team."""

from __future__ import annotations

import io
import logging
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from kenaz_ml.modelstore import LocalModelStore, ModelStore

logger = logging.getLogger(__name__)

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


class FleetFocusModel:
    """Predicts focus score by hour and day-of-week for a team.

    Used by the fleet aggregation layer to identify optimal work windows.
    Model artifacts are stored per-team: ``fleet_focus_{team_id}``.
    """

    def __init__(self, team_id: int, model_store: ModelStore | None = None) -> None:
        self._store = model_store or LocalModelStore()
        self._team_id = team_id
        self.model: GradientBoostingRegressor | None = None
        self._trained = False
        self._samples = 0

        data = self._store.load(self._model_name)
        if data is not None:
            try:
                loaded = joblib.load(io.BytesIO(data))
                self.model = loaded["model"]
                self._samples = loaded.get("samples", 0)
                self._trained = True
                logger.info("Loaded fleet focus model for team %d", team_id)
            except Exception:
                logger.warning("Failed to load fleet focus model for team %d", team_id)
                self.model = None

    @property
    def _model_name(self) -> str:
        return f"fleet_focus_{self._team_id}"

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Train on hourly team aggregate data.

        Each row should have: hour, day_of_week, meeting_minutes,
        context_switches, focus_score.

        Returns:
            Training result with metrics (rmse, r2).
        """
        if len(data) < 5:
            raise ValueError("Need at least 5 data points for training")

        features = []
        targets = []
        for row in data:
            features.append(
                [
                    row.get("hour", 12),
                    row.get("day_of_week", 0),
                    row.get("meeting_minutes", 0),
                    row.get("context_switches", 0),
                ]
            )
            targets.append(row.get("focus_score", 50))

        X = np.array(features)
        y = np.array(targets)

        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=42,
        )
        self.model.fit(X, y)
        self._trained = True
        self._samples = len(data)

        predictions = self.model.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, predictions)))
        r2 = float(r2_score(y, predictions))

        buf = io.BytesIO()
        joblib.dump({"model": self.model, "samples": self._samples}, buf)
        self._store.save(self._model_name, buf.getvalue())
        logger.info("Saved fleet focus model for team %d (%d samples)", self._team_id, self._samples)

        return {
            "model": "fleet_focus",
            "team_id": self._team_id,
            "samples": self._samples,
            "metrics": {"rmse": round(rmse, 3), "r2": round(r2, 3)},
        }

    def predict(self) -> dict[str, Any]:
        """Predict focus score for each hour/day combination.

        Returns:
            Dict with per-day predictions (7x24 grid) and optimal windows.
        """
        if self.model is None or not self._trained:
            return {"predictions": {}, "optimal_windows": []}

        predictions: dict[str, list[float]] = {}
        optimal_windows: list[dict[str, Any]] = []

        for day_idx, day_name in enumerate(DAYS):
            day_preds = []
            for hour in range(24):
                X = np.array([[hour, day_idx, 30, 10]])
                score = float(self.model.predict(X)[0])
                day_preds.append(round(max(0, min(100, score)), 1))
            predictions[day_name] = day_preds

            # Find optimal windows (contiguous blocks above 75th percentile)
            threshold = float(np.percentile(day_preds, 75))
            in_window = False
            start = 0
            for h, s in enumerate(day_preds):
                if s >= threshold and not in_window:
                    in_window = True
                    start = h
                elif (s < threshold or h == 23) and in_window:
                    end = h if s < threshold else h + 1
                    if end - start >= 2:
                        avg_score = float(np.mean(day_preds[start:end]))
                        optimal_windows.append(
                            {
                                "day": day_name,
                                "start": start,
                                "end": end,
                                "predicted_score": round(avg_score, 1),
                            }
                        )
                    in_window = False

        optimal_windows.sort(key=lambda x: x["predicted_score"], reverse=True)

        return {
            "predictions": predictions,
            "optimal_windows": optimal_windows[:5],
            "samples": self._samples,
        }
