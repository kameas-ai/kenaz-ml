"""API endpoint handlers and request/response schemas."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from kenaz_ml.config import ServingMode
from kenaz_ml.feature_store.resolve import resolve_duration_features, resolve_stuck_features
from kenaz_ml.models.duration import DurationEstimator
from kenaz_ml.models.quality import QualityEstimator
from kenaz_ml.models.stuck import StuckPredictor
from kenaz_ml.models.workflow import WorkflowStatePredictor
from kenaz_ml.plugins import fetch_capabilities
from kenaz_ml.tenant import TenantContext, make_tenant_dependency
from kenaz_ml.training.trainer import Trainer

if TYPE_CHECKING:
    from kenaz_ml.app import AppState

logger = logging.getLogger("kenaz_ml")


# ---------- Request / Response schemas ----------


class StuckRequest(BaseModel):
    task_id: str | None = None
    features: dict[str, float] | None = None


class StuckResponse(BaseModel):
    probability: float
    confidence: str


class WorkflowStateRequest(BaseModel):
    task_id: str | None = None
    classified_events: list[dict] | None = None


class WorkflowStateResponse(BaseModel):
    flow_state: dict[str, float]
    dominant_state: str
    momentum: float
    focus_score: float
    dominant_activity: str
    activity_distribution: dict[str, float]
    session_elapsed_min: float
    method: str
    confidence: float


class DurationRequest(BaseModel):
    task_id: str | None = None
    features: dict[str, float] | None = None


class DurationResponse(BaseModel):
    estimated_minutes: float
    confidence_interval: list[float]


class QualityRequest(BaseModel):
    features: dict[str, float]


class QualityResponse(BaseModel):
    score: int
    components: dict[str, float]
    status: str


class TrainRequest(BaseModel):
    db: str | None = Field(None, description="Deprecated: ignored, kept for backward compat")


class TrainResponse(BaseModel):
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    mode: str = "local"  # Default for backward compatibility
    models: dict[str, str]
    uptime_sec: float


class ModelIntrospection(BaseModel):
    """Per-model metadata for the /introspect endpoint.

    Honesty contract (kenaz spec 060): fields the sidecar does not track
    today are returned as null/zero — never invented.
    """

    name: str
    display_name: str
    # ml_predictions.model value this model's predictions are written under
    # (per the WAL contract: "stuck"|"suggest"|"duration"|"quality"|"profile"),
    # or null for models that never write predictions directly.
    prediction_model: str | None = None
    # Underlying sklearn estimator class name, "rules" for rule-based models,
    # or null when the model object is not loaded.
    algorithm: str | None = None
    # Mirrors /health: "ready" | "untrained" | "not_loaded".
    status: str
    trained: bool
    # ISO-8601 UTC timestamp of the last persisted weights write (file mtime
    # via LocalModelStore), or null when untracked/never trained.
    last_trained: str | None = None
    # Training sample count is not tracked per-model today — always null.
    sample_count: int | None = None
    # Count of this model's non-expired ml_predictions rows from the last 24h.
    recent_predictions: int = 0
    # No per-predictor toggle exists today (capabilities.toggle=false);
    # reflects whether the model is loaded and will serve predictions.
    enabled: bool


class IntrospectResponse(BaseModel):
    service: str
    version: str
    mode: str
    uptime_sec: float
    models: list[ModelIntrospection]
    # Capability discovery for kenaz (spec 060 FR-008): controls are hidden
    # in the UI when the capability is absent, rather than erroring.
    capabilities: dict[str, bool]


# (internal model attr, display name, ml_predictions.model key or None)
_INTROSPECT_MODEL_SPECS: list[tuple[str, str, str | None]] = [
    ("stuck", "Stuck Predictor", "stuck"),
    ("activity", "Activity Classifier", None),
    ("workflow", "Suggestion Policy", "suggest"),
    ("duration", "Duration Estimator", "duration"),
    ("quality", "Quality Estimator", "quality"),
]

_RECENT_PREDICTIONS_WINDOW_SEC = 24 * 3600


_start_time = time.time()


# ---------- Centralized fallback predictions for cloud mode ----------
# These match existing fallbacks in poller.py and routes.py.

FALLBACK_STUCK = StuckResponse(probability=0.5, confidence="weak")

FALLBACK_SUGGEST = WorkflowStateResponse(
    flow_state={
        "shallow_work": 1.0,
        "deep_work": 0.0,
        "exploring": 0.0,
        "blocked": 0.0,
        "winding_down": 0.0,
    },
    dominant_state="shallow_work",
    momentum=0.0,
    focus_score=0.5,
    dominant_activity="idle",
    activity_distribution={},
    session_elapsed_min=0.0,
    method="rules",
    confidence=0.5,
)

FALLBACK_DURATION = DurationResponse(estimated_minutes=60.0, confidence_interval=[30.0, 90.0])

FALLBACK_QUALITY = QualityResponse(score=50, components={}, status="normal")


# ---------- Route registration ----------


def register_routes(fastapi_app: FastAPI, state: AppState) -> None:
    """Register all API routes on the given FastAPI app."""

    get_tenant = make_tenant_dependency(state)

    @fastapi_app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        if state.mode == ServingMode.CLOUD:
            models_status: dict[str, str] = {}
            if state.model_cache:
                tenants = state.model_cache.loaded_tenants()
                all_cached_models: set[str] = set()
                for model_list in tenants.values():
                    all_cached_models.update(model_list)
                for name in ["stuck", "activity", "workflow", "duration", "quality"]:
                    models_status[name] = "cached" if name in all_cached_models else "on_demand"
            else:
                for name in ["stuck", "activity", "workflow", "duration", "quality"]:
                    models_status[name] = "not_initialized"

            return HealthResponse(
                status="ok",
                mode="cloud",
                models=models_status,
                uptime_sec=round(time.time() - _start_time, 1),
            )

        # Local mode: existing behavior with mode field added
        models_status = {}
        for name, model in [
            ("stuck", state.stuck),
            ("activity", state.activity),
            ("workflow", state.workflow),
            ("duration", state.duration),
        ]:
            if model is not None:
                models_status[name] = "ready" if model.is_trained else "untrained"
            else:
                models_status[name] = "not_loaded"

        models_status["quality"] = "ready" if state.quality is not None else "not_loaded"

        return HealthResponse(
            status="ok",
            mode="local",
            models=models_status,
            uptime_sec=round(time.time() - _start_time, 1),
        )

    @fastapi_app.get("/status")
    async def status() -> dict:
        if state.mode == ServingMode.CLOUD:
            cache_stats = state.model_cache.stats() if state.model_cache else {}
            loaded = state.model_cache.loaded_tenants() if state.model_cache else {}
            return {
                "mode": "cloud",
                "cache": cache_stats,
                "loaded_tenants": loaded,
                "request_counts": dict(state.request_counters),
                "poller_running": False,
            }

        # Local mode: existing SQLite-based status (unchanged)
        try:
            status_data = state.store.get_status_data()
            return {
                "mode": "local",
                "cursor": status_data["cursor"],
                "latest_predictions": status_data["latest_predictions"],
                "poller_running": state.poller is not None and state.poller._running,
            }
        except Exception:
            return {"mode": "local", "cursor": None, "latest_predictions": [], "poller_running": False}

    @fastapi_app.get("/introspect", response_model=IntrospectResponse)
    async def introspect() -> IntrospectResponse:
        """Sidecar self-description for the kenaz ML Config view (spec 060).

        Returns the sidecar version plus per-model identity, training
        freshness, and recent prediction activity. Values not tracked today
        are honest nulls/zeros — see ModelIntrospection.
        """
        from kenaz_ml import __version__

        if state.mode == ServingMode.CLOUD:
            # Cloud mode hosts per-tenant models loaded on demand; a global
            # per-model listing is not cheaply available, so the model list
            # is honestly empty and retrain is not offered.
            return IntrospectResponse(
                service="kenaz-ml",
                version=__version__,
                mode="cloud",
                uptime_sec=round(time.time() - _start_time, 1),
                models=[],
                capabilities={"retrain": False, "toggle": False},
            )

        # Recent prediction counts per ml_predictions.model key (last 24h,
        # non-expired rows only — get_status_data already filters expiry).
        recent_counts: dict[str, int] = {}
        if state.store is not None:
            try:
                cutoff_ms = int((time.time() - _RECENT_PREDICTIONS_WINDOW_SEC) * 1000)
                for row in state.store.get_status_data()["latest_predictions"]:
                    created_at = row.get("created_at")
                    if isinstance(created_at, (int, float)) and created_at >= cutoff_ms:
                        model_key = row.get("model")
                        if isinstance(model_key, str):
                            recent_counts[model_key] = recent_counts.get(model_key, 0) + 1
            except Exception:
                # WAL locked / sigild not started yet: degrade to zeros
                # rather than failing the whole introspection.
                logger.warning("introspect: prediction counts unavailable", exc_info=True)

        # last-trained timestamps come from the LocalModelStore weight-file
        # mtime; other backends don't track this (getattr probe → nulls).
        last_modified = getattr(state.model_store, "last_modified", None)

        models: list[ModelIntrospection] = []
        for attr, display_name, prediction_model in _INTROSPECT_MODEL_SPECS:
            obj = getattr(state, attr, None)
            if obj is None:
                status_str = "not_loaded"
                trained = False
            else:
                trained = bool(getattr(obj, "is_trained", True))
                status_str = "ready" if trained else "untrained"

            algorithm: str | None = None
            if obj is not None:
                inner = getattr(obj, "model", None)
                if inner is not None:
                    algorithm = type(inner).__name__
                elif attr == "quality":
                    algorithm = "rules"

            last_trained: str | None = None
            if callable(last_modified):
                mtime = last_modified(attr)
                if mtime is not None:
                    from datetime import datetime, timezone

                    last_trained = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

            models.append(
                ModelIntrospection(
                    name=attr,
                    display_name=display_name,
                    prediction_model=prediction_model,
                    algorithm=algorithm,
                    status=status_str,
                    trained=trained,
                    last_trained=last_trained,
                    sample_count=None,  # not tracked per-model today
                    recent_predictions=recent_counts.get(prediction_model, 0) if prediction_model else 0,
                    enabled=obj is not None,
                )
            )

        return IntrospectResponse(
            service="kenaz-ml",
            version=__version__,
            mode="local",
            uptime_sec=round(time.time() - _start_time, 1),
            models=models,
            capabilities={"retrain": True, "toggle": False},
        )

    @fastapi_app.post("/predict/stuck", response_model=StuckResponse)
    async def predict_stuck(
        req: StuckRequest,
        tenant: TenantContext = Depends(get_tenant),
    ) -> StuckResponse:
        if state.mode == ServingMode.CLOUD:
            state.count_request(tenant.tenant_id)
            if req.features is None:
                if req.task_id is not None:
                    raise HTTPException(
                        status_code=400,
                        detail="Cloud mode requires 'features' in request body. "
                        "'task_id' lookup is not available without SQLite.",
                    )
                return FALLBACK_STUCK

            model = state.resolve_model(tenant.tenant_id, "stuck")
            if model is None:
                return FALLBACK_STUCK

            predictor = StuckPredictor.from_trained_model(model)
            result = predictor.predict(req.features)
            return StuckResponse(**result)

        # Local mode: unchanged
        if state.stuck is None:
            return StuckResponse(probability=0.5, confidence="weak")

        if req.features is not None:
            features = req.features
        elif req.task_id is not None:
            features = resolve_stuck_features(state.store, req.task_id)
        else:
            return StuckResponse(probability=0.5, confidence="weak")

        result = state.stuck.predict(features)
        return StuckResponse(**result)

    @fastapi_app.post("/predict/suggest", response_model=WorkflowStateResponse)
    async def predict_suggest(
        req: WorkflowStateRequest,
        tenant: TenantContext = Depends(get_tenant),
    ) -> WorkflowStateResponse:
        if state.mode == ServingMode.CLOUD:
            state.count_request(tenant.tenant_id)
            classified_events = req.classified_events or []

            if not classified_events:
                raise HTTPException(
                    status_code=400,
                    detail="Cloud mode requires 'classified_events' in request body. Poller buffer is not available.",
                )

            model = state.resolve_model(tenant.tenant_id, "workflow")
            if model is None:
                # Use rules-based fallback via a fresh predictor (no model store load)
                predictor = WorkflowStatePredictor()
            else:
                predictor = WorkflowStatePredictor.from_trained_model(model)

            session_info = {
                "session_elapsed_min": 0.0,
                "task_phase": None,
                "test_failures": 0,
            }
            result = predictor.predict(classified_events, session_info)
            return WorkflowStateResponse(**result)

        # Local mode: unchanged
        if state.workflow is None:
            return WorkflowStateResponse(
                flow_state={
                    "shallow_work": 1.0,
                    "deep_work": 0.0,
                    "exploring": 0.0,
                    "blocked": 0.0,
                    "winding_down": 0.0,
                },
                dominant_state="shallow_work",
                momentum=0.0,
                focus_score=0.5,
                dominant_activity="idle",
                activity_distribution={},
                session_elapsed_min=0.0,
                method="rules",
                confidence=0.5,
            )

        classified_events = req.classified_events or []
        session_info = {"session_elapsed_min": 0.0, "task_phase": None, "test_failures": 0}

        if not classified_events and state.poller:
            classified_events = state.poller._buffer

        result = state.workflow.predict(classified_events, session_info)
        return WorkflowStateResponse(**result)

    @fastapi_app.post("/predict/duration", response_model=DurationResponse)
    async def predict_duration(
        req: DurationRequest,
        tenant: TenantContext = Depends(get_tenant),
    ) -> DurationResponse:
        if state.mode == ServingMode.CLOUD:
            state.count_request(tenant.tenant_id)
            if req.features is None:
                if req.task_id is not None:
                    raise HTTPException(
                        status_code=400,
                        detail="Cloud mode requires 'features' in request body. "
                        "'task_id' lookup is not available without SQLite.",
                    )
                return FALLBACK_DURATION

            model = state.resolve_model(tenant.tenant_id, "duration")
            if model is None:
                return FALLBACK_DURATION

            predictor = DurationEstimator.from_trained_model(model)
            result = predictor.predict(req.features)
            return DurationResponse(**result)

        # Local mode: unchanged
        if state.duration is None:
            return DurationResponse(estimated_minutes=60.0, confidence_interval=[30.0, 90.0])

        if req.features is not None:
            features = req.features
        elif req.task_id is not None:
            features = resolve_duration_features(state.store, req.task_id)
        else:
            return DurationResponse(estimated_minutes=60.0, confidence_interval=[30.0, 90.0])

        result = state.duration.predict(features)
        return DurationResponse(**result)

    # Cloud-safe: QualityRequest.features is required (no task_id lookup path).
    # QualityEstimator.predict() is purely functional, no DB access.
    @fastapi_app.post("/predict/quality", response_model=QualityResponse)
    async def predict_quality(
        req: QualityRequest,
        tenant: TenantContext = Depends(get_tenant),
    ) -> QualityResponse:
        if state.mode == ServingMode.CLOUD:
            state.count_request(tenant.tenant_id)
            # Quality is rule-based, no per-tenant model needed
            estimator = QualityEstimator.from_trained_model()
            result = estimator.predict(req.features)
            return QualityResponse(
                score=result["score"],
                components=result["components"],
                status=result["status"],
            )

        # Local mode: unchanged
        if state.quality is None:
            return QualityResponse(score=50, components={}, status="normal")

        result = state.quality.predict(req.features)
        return QualityResponse(
            score=result["score"],
            components=result["components"],
            status=result["status"],
        )

    @fastapi_app.get("/plugins")
    async def plugins() -> dict:
        return fetch_capabilities()

    @fastapi_app.post("/train", response_model=TrainResponse)
    async def train(req: TrainRequest, background_tasks: BackgroundTasks) -> TrainResponse:
        if state.mode == ServingMode.CLOUD:
            raise HTTPException(
                status_code=405,
                detail="Training is not supported in cloud mode. "
                "Train models via the training pipeline and deploy weights to storage.",
            )

        if state.training_in_progress:
            return TrainResponse(status="busy", message="Training already in progress")

        background_tasks.add_task(_run_training, state)
        return TrainResponse(status="started", message="Training started")

    @fastapi_app.get("/")
    async def root() -> dict:
        return {
            "service": "kenaz-ml",
            "mode": state.mode.value,
            "version": "0.1.0",
        }


def _run_training(state: AppState) -> None:
    """Run training in a background thread."""
    try:
        state.training_in_progress = True
        trainer = Trainer(state.store, model_store=state.model_store)
        result = trainer.train_all()
        logger.info("Training complete: %s", result)
        state.load_models()
    except Exception:
        logger.exception("Training failed")
    finally:
        state.training_in_progress = False
