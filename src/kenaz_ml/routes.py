"""API endpoint handlers and request/response schemas."""

from __future__ import annotations

import logging
import time
from datetime import UTC
from typing import TYPE_CHECKING, Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

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
from kenaz_ml.laya import eligibility as laya_eligibility
from kenaz_ml.models.duration import DurationEstimator
from kenaz_ml.models.quality import QualityEstimator
from kenaz_ml.models.stuck import StuckPredictor
from kenaz_ml.models.workflow import WorkflowStatePredictor
from kenaz_ml.plugins import fetch_capabilities
from kenaz_ml.tenant import TenantContext, TenantResolver, make_tenant_dependency
from kenaz_ml.training.trainer import Trainer
from kenaz_ml.typewriter.models import (
    DurationRequest,
    DurationResponse,
    FeatureEventRefusal,
    FeaturesPushRequest,
    FeaturesPushResponse,
    HealthResponse,
    IntrospectResponse,
    LabelBatchRequest,
    LabelBatchResponse,
    LabelCursor,
    LabelRow,
    LabelRowRefusal,
    LaneRefusal,
    LayaEligibility,
    LeaseRequest,
    LeaseResponse,
    ModelHealth,
    ModelIntrospection,
    QualityRequest,
    QualityResponse,
    ShutdownResponse,
    StuckRequest,
    StuckResponse,
    TrainRequest,
    TrainResponse,
    WorkflowStateRequest,
    WorkflowStateResponse,
)

if TYPE_CHECKING:
    from kenaz_ml.app import AppState

logger = logging.getLogger("kenaz_ml")


# ---------- Request / Response schemas ----------


# Every schema below is generated from ``typewriter/spec.yaml``; the spec is the
# only place these shapes are written.

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


def register_routes(fastapi_app: FastAPI, state: AppState, tenant_resolver: TenantResolver | None = None) -> None:
    """Register all API routes on the given FastAPI app.

    ``tenant_resolver`` is the cloud-mode tenant seam (see
    :mod:`kenaz_ml.tenant`); ``None`` keeps the header resolver.
    """

    get_tenant = make_tenant_dependency(state, tenant_resolver)
    # Computed once per app, at startup (registration), never per request.
    state.engine_sha256 = engine_sha256()

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
                **_identity(state),
                model_details={
                    name: ModelHealth(status=value, slot=None, refusal=None) for name, value in models_status.items()
                },
                lifecycle_protocol=0,
                laya_eligibility=LayaEligibility(**laya_eligibility.describe()),
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

        # Every poll is an implicit 90 s lease (FR-013, repair R4).
        leases = getattr(state, "leases", None)
        if leases is not None:
            leases.note_health_poll()

        resolutions = getattr(state, "resolutions", {})
        details: dict[str, ModelHealth] = {}
        for name, value in models_status.items():
            prov = _resolution_provenance(resolutions.get(name))
            details[name] = ModelHealth(status=value, slot=prov["serving_slot"], refusal=prov["refusal"])

        from kenaz_ml.lifecycle.leases import LIFECYCLE_PROTOCOL

        return HealthResponse(
            status="ok",
            mode="local",
            models=models_status,
            uptime_sec=round(time.time() - _start_time, 1),
            **_identity(state),
            model_details=details,
            lifecycle_protocol=LIFECYCLE_PROTOCOL,
            laya_eligibility=LayaEligibility(**laya_eligibility.describe()),
        )

    @fastapi_app.post("/v1/clients/lease", response_model=LeaseResponse, responses={404: {"description": "Cloud mode"}})
    async def client_lease(req: LeaseRequest) -> LeaseResponse | JSONResponse:
        """Register or renew a client lease (FR-011). Local mode only."""
        if state.mode == ServingMode.CLOUD:
            return JSONResponse(status_code=404, content={"detail": "leases are a local-mode feature"})
        from kenaz_ml import __version__
        from kenaz_ml.lifecycle.leases import LIFECYCLE_PROTOCOL, is_managed

        leases = state.leases
        leases.renew(req.client, req.pid, req.client_version, req.min_contracts)
        versions = _contract_versions(state)
        incompatible: dict[str, str] = {}
        for kind_id, wanted in req.min_contracts.items():
            supported = versions.get(kind_id)
            if supported is None:
                incompatible[kind_id] = "unknown kind"
            elif wanted not in supported:
                incompatible[kind_id] = f"contract {wanted} not supported (engine supports {supported})"
        return LeaseResponse(
            lifecycle_protocol=LIFECYCLE_PROTOCOL,
            sidecar_version=__version__,
            contract_versions=versions,
            incompatible_kinds=incompatible,
            client=req.client,
            pid=req.pid,
            live_leases=leases.live_count(),
            implicit_lease_sec=leases.implicit_lease_sec,
            idle_exit_sec=leases.idle_exit_sec,
            managed=is_managed(),
        )

    @fastapi_app.post(
        "/v1/admin/shutdown",
        response_model=ShutdownResponse,
        status_code=202,
        responses={
            401: {"description": "No bearer token"},
            403: {"description": "Token refused"},
            404: {"description": "Cloud mode"},
        },
    )
    async def admin_shutdown(
        background_tasks: BackgroundTasks, authorization: str | None = Header(None)
    ) -> ShutdownResponse | JSONResponse:
        """Graceful shutdown, authorized by ``Authorization: Bearer <token>`` (FR-015).

        The token is read on every request from the client-owned
        ``<install_root>/lease/shutdown.token``. Fails closed: a missing lease
        directory or token file, an empty or unreadable file, no token or a
        wrong one all refuse and the engine keeps running. On success the
        response is sent first; then in-flight training drains (bounded), the
        poller stops, and the process exits through the graceful path.
        """
        from kenaz_ml import config
        from kenaz_ml.lifecycle.shutdown import (
            REASON_NO_TOKEN,
            check_token,
            drain_and_exit,
            presented_token,
        )

        if state.mode == ServingMode.CLOUD:
            return JSONResponse(status_code=404, content={"detail": "admin shutdown is a local-mode feature"})
        presented = presented_token(authorization)
        result = check_token(presented, config.shutdown_token_path())
        if not result.ok:
            logger.warning("admin shutdown refused: %s", result.reason)
            code = 401 if result.reason == REASON_NO_TOKEN or presented is None else 403
            return JSONResponse(
                status_code=code, content={"error": result.reason, "detail": f"shutdown refused: {result.reason}"}
            )
        if not getattr(state, "exiting", False):
            state.exiting = True
            background_tasks.add_task(drain_and_exit, state, reason="admin_shutdown")
        return ShutdownResponse(status="shutting_down")

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
            if state.store is None:
                raise RuntimeError("the data store is not initialised")
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
                    from datetime import datetime

                    last_trained = datetime.fromtimestamp(mtime, tz=UTC).isoformat()

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
        elif req.task_id is not None and state.store is not None:
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
        elif req.task_id is not None and state.store is not None:
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
        ``{"error": <code>, "refusal": {kind_id, reason, detail}}`` body. Never substitutes a
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
            413: {"model": RecommendRefusal, "description": "More rows than the batch limit; nothing acked"},
            409: {
                "model": RecommendRefusal,
                "description": "Contract / names / retained-header mismatch; nothing written",
            },
            503: {"model": RecommendRefusal, "description": "Label log I/O failure; nothing acked"},
        },
    )
    def labels(kind: str, req: LabelBatchRequest) -> LabelBatchResponse | JSONResponse:
        """Cursor-acked, revision-upserted label ingest (FR-007..009). Loopback only; no outbound call."""
        from kenaz_ml.advice.label_log import MAX_BATCH as MAX_LABEL_BATCH
        from kenaz_ml.advice.label_log import (
            STATUS_ACCEPTED,
            STATUS_DUPLICATE,
            STATUS_REFUSED,
            STATUS_SUPERSEDED,
            ingest,
        )

        if len(req.rows) > MAX_LABEL_BATCH:
            return JSONResponse(
                status_code=413,
                content=Refused(
                    kind, "batch_too_large", f"{len(req.rows)} rows exceeds the batch limit of {MAX_LABEL_BATCH}"
                ).body(),
            )
        entry = state.dispatch_table.snapshot().get(kind)
        if entry is None or entry.contract is None:
            return JSONResponse(
                status_code=404,
                content=Refused(kind, "unknown_kind", f"unknown kind {kind!r}: no registered contract").body(),
            )
        result = ingest(kind, req.client, [_label_row(row, req.client) for row in req.rows], entry.contract)
        if result.refusal is not None:
            status = 503 if result.refusal == "io_error" else 409
            return JSONResponse(
                status_code=status, content=Refused(kind, result.refusal, result.refusal_detail or "").body()
            )
        return LabelBatchResponse(
            acked=LabelCursor(ts=result.ack[0], revision=result.ack[1]) if result.ack else None,
            applied=result.count(STATUS_ACCEPTED),
            replaced=result.count(STATUS_SUPERSEDED),
            stale=result.count(STATUS_DUPLICATE),
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

    # ---------- /v1/features (WP09) ----------
    from kenaz_ml import features_push as _features_push

    push_store = _features_push.install(_features_push.FeaturePushStore())
    state.features_push = push_store

    @fastapi_app.post(
        "/v1/features",
        response_model=FeaturesPushResponse,
        responses={422: {"model": LaneRefusal, "description": "Batch too large, or the lane is unavailable"}},
    )
    async def push_features(req: FeaturesPushRequest) -> FeaturesPushResponse | JSONResponse:
        """Stateless feature-event push (FR-023): validated against a closed class
        registry, held in a bounded in-process store. No table, no file, no outbound call.
        Local mode only: in cloud mode the route answers ``lane_unavailable``."""
        if state.mode == ServingMode.CLOUD:
            return JSONResponse(
                status_code=422,
                content={
                    "error": _features_push.REASON_LANE_UNAVAILABLE,
                    "refusal": {
                        "reason": _features_push.REASON_LANE_UNAVAILABLE,
                        "detail": "the /v1/features lane is a local-mode feature",
                    },
                },
            )
        if len(req.events) > _features_push.MAX_BATCH:
            return JSONResponse(
                status_code=422,
                content={
                    "error": _features_push.REASON_BATCH_TOO_LARGE,
                    "refusal": {
                        "reason": _features_push.REASON_BATCH_TOO_LARGE,
                        "detail": f"{len(req.events)} events exceeds the batch limit of {_features_push.MAX_BATCH}",
                    },
                },
            )
        outcome = push_store.push(req.client, [e.model_dump() for e in req.events])
        return FeaturesPushResponse(
            accepted=outcome.accepted,
            refused=len(outcome.refusals),
            refusals=[FeatureEventRefusal(**r) for r in outcome.refusals],
        )

    @fastapi_app.get("/")
    async def root() -> dict:
        from kenaz_ml import __version__

        return {
            "service": "kenaz-ml",
            "mode": state.mode.value,
            "version": __version__,
        }


def _label_row(row: LabelRow, client: str) -> dict[str, Any]:
    """Normalize one wire row for the label log (client defaulted, features parsed, model_id alias)."""
    import json

    record = row.model_dump(exclude_none=True)
    record.setdefault("client", client)
    features = record.get("features")
    if isinstance(features, str):
        try:
            parsed = json.loads(features)
        except ValueError:
            parsed = None
        record["features"] = parsed if isinstance(parsed, dict) else {}
    if "model" in record:
        # The sibling mission reads ``model_id`` from the label log.
        record.setdefault("model_id", record["model"])
    return record


def _contract_versions(state: AppState) -> dict[str, list[str]]:
    """Per recommend kind, the contract versions the engine accepts (from the dispatch table)."""
    table = getattr(state, "dispatch_table", None)
    if table is None:
        return {}
    return {kind: list(entry.supported_versions) for kind, entry in table.snapshot().items()}


def _exe_path() -> str:
    """This process's own executable, reported truthfully — never verified here (C-001)."""
    import os
    import sys

    if getattr(sys, "frozen", False):
        return os.path.realpath(sys.executable)
    argv0 = sys.argv[0] if sys.argv and sys.argv[0] else ""
    if argv0 and os.path.exists(argv0):
        return os.path.realpath(argv0)
    return os.path.realpath(sys.executable)


def engine_sha256() -> str | None:
    """sha256 of this engine's own executable file, resolved through symlinks.

    ``sys.executable``: under the frozen onedir that is the frozen
    launcher; from source it is the interpreter. Reported for the client's
    adoption cross-check (design F2) — kenaz-ml never verifies it itself
    (C-001). ``None`` when the file cannot be read.
    """
    import hashlib
    import os
    import sys

    try:
        path = os.path.realpath(sys.executable)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError, TypeError, ValueError:
        logger.warning("health: could not hash the engine executable", exc_info=True)
        return None


def _identity(state: AppState) -> dict:
    from kenaz_ml import __version__

    return {
        "product": "kenaz-ml",
        "sidecar_version": __version__,
        "contract_versions": _contract_versions(state),
        "exe_path": _exe_path(),
        "engine_sha256": getattr(state, "engine_sha256", None),
        "device": "cpu",
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
        if state.store is None:
            raise RuntimeError("the data store is not initialised")
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
