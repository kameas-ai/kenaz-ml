"""Fleet model API routes — training and prediction endpoints for team-level models."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from kenaz_ml.models.fleet_focus import FleetFocusModel
from kenaz_ml.models.fleet_meeting import FleetMeetingModel
from kenaz_ml.models.fleet_onboarding import FleetOnboardingModel

if TYPE_CHECKING:
    from kenaz_ml.app import AppState

logger = logging.getLogger(__name__)


# ---------- Request / Response schemas ----------


class FleetTrainRequest(BaseModel):
    team_id: int
    data: list[dict[str, Any]] = Field(..., min_length=5)


class FleetTrainResponse(BaseModel):
    model: str
    team_id: int
    samples: int
    trained_at: str
    metrics: dict[str, float]


class FleetPredictResponse(BaseModel):
    model: str
    team_id: int
    predictions: dict[str, Any]
    trained_at: str | None = None
    samples: int = 0


# ---------- Route registration ----------

MODEL_NAMES = {"focus", "meeting", "onboarding"}


def register_fleet_routes(fastapi_app: FastAPI, state: AppState) -> None:
    """Register fleet model training and prediction routes."""

    # Cache model instances to avoid reloading from disk on every request.
    _cache: dict[str, FleetFocusModel | FleetMeetingModel | FleetOnboardingModel] = {}

    def _get_model(model_name: str, team_id: int, *, force_reload: bool = False):
        key = f"{model_name}_{team_id}"
        if not force_reload and key in _cache:
            return _cache[key]

        store = state.model_store
        if model_name == "focus":
            m = FleetFocusModel(team_id, model_store=store)
        elif model_name == "meeting":
            m = FleetMeetingModel(team_id, model_store=store)
        elif model_name == "onboarding":
            m = FleetOnboardingModel(team_id, model_store=store)
        else:
            raise HTTPException(status_code=404, detail=f"Unknown fleet model: {model_name}")
        _cache[key] = m
        return m

    @fastapi_app.post("/fleet/train/{model_name}", response_model=FleetTrainResponse)
    async def fleet_train(model_name: str, req: FleetTrainRequest) -> FleetTrainResponse:
        """Train a fleet model for a specific team."""
        if model_name not in MODEL_NAMES:
            raise HTTPException(status_code=404, detail=f"Unknown fleet model: {model_name}")

        model = _get_model(model_name, req.team_id, force_reload=True)
        try:
            result = model.train(req.data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return FleetTrainResponse(
            model=result["model"],
            team_id=result["team_id"],
            samples=result["samples"],
            trained_at=datetime.now(timezone.utc).isoformat(),
            metrics=result["metrics"],
        )

    @fastapi_app.get("/fleet/predict/{model_name}", response_model=FleetPredictResponse)
    async def fleet_predict(model_name: str, team_id: int) -> FleetPredictResponse:
        """Get predictions from a trained fleet model."""
        if model_name not in MODEL_NAMES:
            raise HTTPException(status_code=404, detail=f"Unknown fleet model: {model_name}")

        model = _get_model(model_name, team_id)
        if not model.is_trained:
            return FleetPredictResponse(
                model=f"fleet_{model_name}",
                team_id=team_id,
                predictions={},
                trained_at=None,
                samples=0,
            )

        preds = model.predict()
        samples = preds.get("samples", 0)
        # Build predictions dict without the samples key.
        predictions = {k: v for k, v in preds.items() if k != "samples"}
        return FleetPredictResponse(
            model=f"fleet_{model_name}",
            team_id=team_id,
            predictions=predictions,
            samples=samples,
        )

    @fastapi_app.get("/fleet/health")
    async def fleet_health() -> dict[str, Any]:
        """Fleet model subsystem health check."""
        return {
            "status": "ok",
            "models": list(MODEL_NAMES),
            "store": type(state.model_store).__name__ if state.model_store else "none",
        }
