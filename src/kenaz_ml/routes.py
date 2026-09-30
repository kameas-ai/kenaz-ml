"""API endpoint handlers and request/response schemas."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from kenaz_ml.advice.dispatch import (
    REFUSAL_STATUS_CODE,
    ContractsResponse,
    RecommendRefusal,
    RecommendRequest,
    RecommendResponse,
    Refused,
    contracts_payload,
    timed_dispatch,
)
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
    # --- Registry provenance (two-client-engine-01MSK2EN WP01, additive) ---
    # Which registry slot served this model: "local" | "base" | "cold_start",
    # or null when the model is not resolved through the registry (quality,
    # whose JSON weights the registry cannot govern; or a non-filesystem store).
    serving_slot: str | None = None
    # The serving manifest's provenance.training_source ("local" | "base" | ...),
    # or null when no manifest served (cold start / not registry-resolved).
    training_source: str | None = None
    # The serving manifest's version (Manifest.version), or null.
    manifest_version: str | None = None
    # Registry refusal text collected while resolving ("<model> [<slot>]
    # <check>/<reason>: <detail>"; empty-slot misses omitted), or null when
    # nothing was refused.
    refusal: str | None = None


class IntrospectResponse(BaseModel):
    service: str
    version: str
    mode: str
    uptime_sec: float
    models: list[ModelIntrospection]
    # Capability discovery for kenaz (spec 060 FR-008): controls are hidden
    # in the UI when the capability is absent, rather than erroring.
    capabilities: dict[str, bool]


# ---------- /v1/labels/{kind} — frozen ingest contract (Amendment A3.3) ----------


class LabelCursor(BaseModel):
    """The ack cursor: a position over ``(ts, revision)``."""

    ts: int
    revision: int


class LabelRow(BaseModel):
    """One ``advice_labels`` row as the harness pushes it (frozen contract, A3.3).

    Idempotency key ``(client, kind, features_hash, ts)``; a higher ``revision``
    replaces the stored row, an equal or lower one is a duplicate. Propensity
    fields are kept verbatim in the kind's label log; only ``accepted`` /
    ``auto_acted`` (y=1) and ``dismissed`` (y=0) rows with
    ``features_complete=true`` reach the retained training set.
    """

    client: str
    kind: str
    features_hash: str
    ts: int = Field(..., description="Decision time (ms since epoch); part of the idempotency key.")
    revision: int = Field(..., ge=0, description="Monotonic per key; a higher revision replaces the stored row.")
    feature_contract_version: str
    features: dict[str, float] = Field(default_factory=dict)
    features_complete: bool
    shown: bool
    confidence: int | None = Field(None, description="Confidence at decision time (0-100), as the harness saw it.")
    model_id: str | None = None
    rung: str | None = None
    prompt_version: str | None = None
    user_action: str = Field(..., description="accepted | dismissed | ignored | auto_acted")
    recommendation: dict | None = Field(None, description="The recommendation as served (decision/score/...).")
    latency_ms: float | None = None
    session_id: str | None = None
    as_of_ms: int | None = Field(None, description="Feature snapshot time; ts is used when absent (never recomputed).")


class LabelBatchRequest(BaseModel):
    client: str
    cursor: LabelCursor | None = Field(None, description="The client's current ack cursor (informational).")
    rows: list[LabelRow]


class LabelRowRefusal(BaseModel):
    index: int
    features_hash: str
    ts: int
    revision: int
    reason: str


class LabelBatchResponse(BaseModel):
    kind: str
    ack: LabelCursor | None = Field(
        None, description="Advanced through the contiguous (ts, revision)-ordered prefix of non-refused rows."
    )
    accepted: int
    superseded: int
    duplicate: int
    refused: int
    refusals: list[LabelRowRefusal]
    retained_appended: int
    retained_rebuilt: bool
    retained_generation: str | None = None


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

            provenance = _resolution_provenance(getattr(state, "resolutions", {}).get(attr))

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
                    **provenance,
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

    # ---------- /v1 — harness advice surface (two-client-engine-01MSK2EN) ----------
    # Additive only (C-008). In cloud mode the dispatch table is empty: the
    # layer reads local registry slots, so no kinds are published and every
    # kind refuses unknown_kind.

    @fastapi_app.post(
        "/v1/recommend/{kind}",
        response_model=RecommendResponse,
        responses={REFUSAL_STATUS_CODE: {"model": RecommendRefusal, "description": "Typed refusal"}},
    )
    async def recommend(kind: str, req: RecommendRequest) -> RecommendResponse | JSONResponse:
        """Dispatch one recommendation to the kind's single registered backend.

        Every refusal (kind_not_served, laya_backend_not_installed,
        contract_mismatch, unknown_kind, ...) is HTTP 422 with a
        ``{"refusal": {kind_id, reason, detail}}`` body. Never substitutes a
        different backend; never clamps confidence.
        """
        try:
            return timed_dispatch(state.dispatch_table, kind, req)
        except Refused as refusal:
            return JSONResponse(status_code=REFUSAL_STATUS_CODE, content=refusal.body())

    @fastapi_app.get("/v1/contracts", response_model=ContractsResponse)
    async def contracts() -> ContractsResponse:
        """Every registered kind's ordered feature contract and serving availability."""
        return contracts_payload(state.dispatch_table.snapshot())

    @fastapi_app.post(
        "/v1/labels/{kind}",
        response_model=LabelBatchResponse,
        responses={
            404: {"model": RecommendRefusal, "description": "Unknown kind"},
            409: {
                "model": RecommendRefusal,
                "description": "Contract / names / retained-header mismatch; nothing written",
            },
            503: {"model": RecommendRefusal, "description": "Label log I/O failure; nothing acked"},
        },
    )
    def labels(kind: str, req: LabelBatchRequest) -> LabelBatchResponse | JSONResponse:
        """Cursor-acked, revision-upserted label ingest (FR-007..009). Loopback only; no outbound call."""
        from kenaz_ml.advice.label_log import (
            STATUS_ACCEPTED,
            STATUS_DUPLICATE,
            STATUS_REFUSED,
            STATUS_SUPERSEDED,
            ingest,
        )

        entry = state.dispatch_table.snapshot().get(kind)
        if entry is None or entry.contract is None:
            return JSONResponse(
                status_code=404,
                content=Refused(kind, "unknown_kind", f"unknown kind {kind!r}: no registered contract").body(),
            )
        result = ingest(kind, req.client, [row.model_dump() for row in req.rows], entry.contract)
        if result.refusal is not None:
            status = 503 if result.refusal == "io_error" else 409
            return JSONResponse(
                status_code=status, content=Refused(kind, result.refusal, result.refusal_detail or "").body()
            )
        return LabelBatchResponse(
            kind=kind,
            ack=LabelCursor(ts=result.ack[0], revision=result.ack[1]) if result.ack else None,
            accepted=result.count(STATUS_ACCEPTED),
            superseded=result.count(STATUS_SUPERSEDED),
            duplicate=result.count(STATUS_DUPLICATE),
            refused=result.count(STATUS_REFUSED),
            refusals=[
                LabelRowRefusal(
                    index=o.index, features_hash=o.features_hash, ts=o.ts, revision=o.revision, reason=o.reason or ""
                )
                for o in result.outcomes
                if o.status == STATUS_REFUSED
            ],
            retained_appended=result.retained_appended,
            retained_rebuilt=result.retained_rebuilt,
            retained_generation=result.retained_generation,
        )

    @fastapi_app.get("/")
    async def root() -> dict:
        return {
            "service": "kenaz-ml",
            "mode": state.mode.value,
            "version": "0.1.0",
        }


def _resolution_provenance(resolution: object | None) -> dict[str, str | None]:
    """Render a registry ``Resolution`` as the additive /introspect fields.

    Honest nulls throughout when the model was not resolved through the
    registry. Refusal text omits the ordinary empty-slot miss, which is the
    state of every install before a base model ships and is not a refusal an
    operator can act on.
    """
    out: dict[str, str | None] = {
        "serving_slot": None,
        "training_source": None,
        "manifest_version": None,
        "refusal": None,
    }
    if resolution is None:
        return out
    out["serving_slot"] = getattr(resolution, "slot", None)
    manifest = getattr(resolution, "manifest", None)
    if manifest is not None and getattr(resolution, "served", False):
        out["training_source"] = manifest.provenance.training_source
        out["manifest_version"] = manifest.version
    refused = [str(r) for r in getattr(resolution, "refusals", ()) if getattr(r, "reason", None) != "slot_empty"]
    if refused:
        out["refusal"] = "; ".join(refused)
    return out


def _run_training(state: AppState) -> None:
    """Run training in a background thread."""
    try:
        state.training_in_progress = True
        trainer = Trainer(state.store, model_store=state.model_store)
        result = trainer.train_all()
        logger.info("Training complete: %s", result)
        # FR-002: a retrain can change a base-version relationship, so the
        # registry refresh runs before the reload, exactly as the scheduler's
        # reload callback does.
        state.refresh_registry()
        state.load_models()
    except Exception:
        logger.exception("Training failed")
    finally:
        state.training_in_progress = False
