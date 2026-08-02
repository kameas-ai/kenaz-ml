"""Fleet meeting impact model — quantify meeting disruption for a team."""

from __future__ import annotations

import io
import logging
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from kenaz_ml.modelstore import LocalModelStore, ModelStore

logger = logging.getLogger(__name__)

DISRUPTION_LABELS = {0: "low", 1: "medium", 2: "high"}


class FleetMeetingModel:
    """Classifies meeting disruption level and estimates recovery time.

    Used by the fleet aggregation layer to quantify how meetings affect
    team productivity. Model artifacts stored per-team: ``fleet_meeting_{team_id}``.
    """

    def __init__(self, team_id: int, model_store: ModelStore | None = None) -> None:
        self._store = model_store or LocalModelStore()
        self._team_id = team_id
        self.model: RandomForestClassifier | None = None
        self._trained = False
        self._samples = 0

        data = self._store.load(self._model_name)
        if data is not None:
            try:
                loaded = joblib.load(io.BytesIO(data))
                self.model = loaded["model"]
                self._samples = loaded.get("samples", 0)
                self._trained = True
                logger.info("Loaded fleet meeting model for team %d", team_id)
            except Exception:
                logger.warning("Failed to load fleet meeting model for team %d", team_id)
                self.model = None

    @property
    def _model_name(self) -> str:
        return f"fleet_meeting_{self._team_id}"

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Train on meeting impact data.

        Each row should have: meeting_duration, time_of_day, focus_before,
        focus_after.

        Returns:
            Training result with accuracy metric.
        """
        if len(data) < 5:
            raise ValueError("Need at least 5 data points for training")

        features = []
        targets = []
        for row in data:
            duration = row.get("meeting_duration", 30)
            time_of_day = row.get("time_of_day", 10)
            focus_before = row.get("focus_before", 70)
            focus_after = row.get("focus_after", 50)
            features.append([duration, time_of_day, focus_before])
            delta = focus_before - focus_after
            if delta > 20:
                targets.append(2)  # high disruption
            elif delta > 10:
                targets.append(1)  # medium
            else:
                targets.append(0)  # low

        X = np.array(features)
        y = np.array(targets)

        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(X, y)
        self._trained = True
        self._samples = len(data)

        accuracy = float(self.model.score(X, y))

        buf = io.BytesIO()
        joblib.dump({"model": self.model, "samples": self._samples}, buf)
        self._store.save(self._model_name, buf.getvalue())
        logger.info("Saved fleet meeting model for team %d (%d samples)", self._team_id, self._samples)

        return {
            "model": "fleet_meeting",
            "team_id": self._team_id,
            "samples": self._samples,
            "metrics": {"accuracy": round(accuracy, 3)},
        }

    def predict(self) -> dict[str, Any]:
        """Predict disruption level for common meeting scenarios.

        Returns:
            Dict with disruption scenarios (duration x time_of_day grid).
        """
        if self.model is None or not self._trained:
            return {"scenarios": []}

        scenarios = []
        for duration in [15, 30, 45, 60, 90]:
            for tod in [9, 11, 14, 16]:
                X = np.array([[duration, tod, 70]])
                pred = int(self.model.predict(X)[0])
                scenarios.append(
                    {
                        "duration": duration,
                        "time_of_day": tod,
                        "disruption": DISRUPTION_LABELS.get(pred, "unknown"),
                        "recovery_estimate_min": pred * 30,
                    }
                )

        return {"scenarios": scenarios, "samples": self._samples}
