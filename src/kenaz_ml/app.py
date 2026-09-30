"""FastAPI application factory and model lifecycle."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kenaz_ml.signals.engine import SignalEngine

from fastapi import FastAPI

from kenaz_ml.advice.dispatch import DispatchTable, build_table
from kenaz_ml.config import ServingMode, resolve_mode
from kenaz_ml.datastore import DataStore, create_store
from kenaz_ml.lifecycle.leases import LeaseTable, is_managed
from kenaz_ml.models.activity import ActivityClassifier
from kenaz_ml.models.duration import DurationEstimator
from kenaz_ml.models.fleet_routes import register_fleet_routes
from kenaz_ml.models.quality import QualityEstimator
from kenaz_ml.models.stuck import StuckPredictor
from kenaz_ml.models.workflow import WorkflowStatePredictor
from kenaz_ml.modelstore import ModelStore, model_store_factory
from kenaz_ml.poller import EventPoller
from kenaz_ml.routes import register_routes
from kenaz_ml.training.scheduler import TrainingScheduler

logger = logging.getLogger("kenaz_ml")

#: The local models whose artifacts are registry-format (joblib + manifest) and
#: so are governed by ``refresh_all()``. ``quality`` persists JSON weights and
#: stays on its legacy path pending an owner ruling (tasks/README item 7).
REGISTRY_ROSTER: tuple[str, ...] = ("stuck", "activity", "workflow", "duration")


def refresh_registry_roster(model_store: ModelStore | None, sink: dict[str, dict[str, Any]] | None = None) -> None:
    """Call ``refresh_all()`` for :data:`REGISTRY_ROSTER` and log every outcome.

    Only for a filesystem-backed store (manifests are a filesystem concept,
    D-004); anything else is logged and skipped. Never raises.
    """
    import time

    from kenaz_ml.modelstore.loader import filesystem_slot_dir

    local_dir = filesystem_slot_dir(model_store) if model_store is not None else None
    if local_dir is None:
        logger.info("refresh_all: skipped -- model store %s is not filesystem-backed", type(model_store).__name__)
        return
    started = time.monotonic()
    try:
        from kenaz_ml.modelstore.registry import describe, refresh_all

        results = refresh_all(REGISTRY_ROSTER, local_dir=local_dir)
    except Exception:
        logger.warning("refresh_all: failed; serving continues on the current slots", exc_info=True)
        return
    elapsed_ms = (time.monotonic() - started) * 1000
    for result in results:
        try:
            info = describe(result)
        except Exception:  # pragma: no cover - describe is plain data
            info = {"name": getattr(result, "name", "?"), "action": getattr(result, "action", "?")}
        if sink is not None:
            sink[str(info.get("name"))] = info
        logger.info(
            "refresh_all: model=%s action=%s ok=%s reason=%s",
            info.get("name"),
            info.get("action"),
            info.get("ok"),
            info.get("reason"),
        )
    logger.info("refresh_all: %d model(s) refreshed in %.1f ms", len(results), elapsed_ms)


class AppState:
    """Holds model instances and runtime state, passed to routes."""

    def __init__(self, mode: ServingMode = ServingMode.LOCAL) -> None:
        self.mode = mode
        self.store: DataStore | None = None
        self.model_store: ModelStore | None = None
        self.stuck: StuckPredictor | None = None
        self.activity: ActivityClassifier | None = None
        self.workflow: WorkflowStatePredictor | None = None
        self.duration: DurationEstimator | None = None
        self.quality: QualityEstimator | None = None
        self.poller: EventPoller | None = None
        self.signal_engine: SignalEngine | None = None
        self.training_in_progress: bool = False
        # Cloud-mode fields (initialized by cloud startup path)
        self.model_cache: Any = None
        self.model_loader: Any = None
        # Per-tenant request counters (cloud mode, reset on restart)
        self.request_counters: dict[str, int] = {}
        # two-client-engine-01MSK2EN WP01 (T004, ruling R5): the registry's
        # resolution outcome per local model -- slot, manifest, refusals. The
        # shared seam /introspect (provenance) and /health (refusal text) read.
        # A model absent from this map was not resolved through the registry
        # (legacy store, or quality -- see models/quality.py).
        self.resolutions: dict[str, Any] = {}
        # The last refresh_all() outcome per model, as registry.describe() data.
        self.refresh_results: dict[str, dict[str, Any]] = {}
        # /v1/recommend dispatch table (WP03). Empty until local-mode startup
        # populates it; stays empty in cloud mode.
        self.dispatch_table: DispatchTable = DispatchTable()
        # Lifecycle (WP05): client leases + the self-termination timer. The
        # clock starts now (process start). ``exit_fn`` is how the engine
        # leaves -- SIGTERM to itself by default; tests inject their own.
        self.leases: LeaseTable = LeaseTable()
        self.exit_fn: Any = None
        self.exiting: bool = False
        # /v1/features push store (WP09); created per app by register_routes.
        self.features_push: Any = None
        # sha256 of the engine executable, computed once by register_routes.
        self.engine_sha256: str | None = None

    def load_models(self, model_store: ModelStore | None = None) -> None:
        """Load or reload all model instances."""
        ms = model_store or self.model_store
        self.stuck = StuckPredictor(model_store=ms, registry=True)
        self.activity = ActivityClassifier(model_store=ms, registry=True)
        self.workflow = WorkflowStatePredictor(model_store=ms, registry=True)
        self.duration = DurationEstimator(model_store=ms, registry=True)
        self.quality = QualityEstimator(model_store=ms)

        resolutions: dict[str, Any] = {}
        for name, predictor in (
            ("stuck", self.stuck),
            ("activity", self.activity),
            ("workflow", self.workflow),
            ("duration", self.duration),
        ):
            resolution = getattr(predictor, "resolution", None)
            if resolution is not None:
                resolutions[name] = resolution
        self.resolutions = resolutions

    def refresh_registry(self, model_store: ModelStore | None = None) -> None:
        """Run the registry's base-refresh policy over the local roster (FR-002).

        The first production caller of ``refresh_all()`` (WP01 T003). Runs
        before models are (re)loaded so a rebuilt or adopted artifact is what
        gets served. Logs one INFO line per model -- including the ordinary
        ``no_base`` case -- which is SC-002's evidence that it ran. Never
        raises: a refresh failure is logged and serving continues on whatever
        the slots already hold.
        """
        ms = model_store or self.model_store
        refresh_registry_roster(ms, self.refresh_results)

    def reload_models_into_poller(self) -> None:
        """Reload model instances after retraining."""
        self.refresh_registry()
        self.load_models()
        if self.poller:
            self.poller.stuck = self.stuck
            self.poller.activity = self.activity
            self.poller.workflow = self.workflow
            self.poller.duration = self.duration
            self.poller.quality = self.quality
        if self.signal_engine and self.model_store:
            self.signal_engine.pattern_detector.load(self.model_store)
            self.signal_engine.next_action.load(self.model_store)
            self.signal_engine.file_recommender.load(self.model_store)
        logger.info("models reloaded into poller")

    def resolve_model(self, tenant_id: str, model_name: str) -> Any | None:
        """Resolve a model for the given tenant, using cache then loader.

        Returns the model object or None if no model is available.
        Only used in cloud mode.
        """
        if self.model_cache is None or self.model_loader is None:
            return None

        # Check cache first
        model = self.model_cache.get(tenant_id, model_name)
        if model is not None:
            logger.debug(
                "model-resolve: cache_hit tenant=%s model=%s",
                tenant_id,
                model_name,
            )
            return model

        # Cache miss: load from backend
        model = self.model_loader.load(tenant_id, model_name)
        if model is not None:
            self.model_cache.put(tenant_id, model_name, model)
            logger.info(
                "model-resolve: cache_miss+loaded tenant=%s model=%s",
                tenant_id,
                model_name,
            )
            return model

        logger.info(
            "model-resolve: cache_miss+fallback tenant=%s model=%s",
            tenant_id,
            model_name,
        )
        return None

    def count_request(self, tenant_id: str) -> None:
        """Increment the request counter for a tenant."""
        self.request_counters[tenant_id] = self.request_counters.get(tenant_id, 0) + 1


async def lifecycle_loop(state: AppState) -> None:
    """Sweep leases and run the self-termination timer, off the request path (NFR-003).

    Self-termination applies only to a managed install -- one whose ``lease/``
    directory exists (FR-014, ruled 2026-09-30). Without it (a developer's
    ``kenaz-ml serve``) the engine never exits on its own (D-A2).
    """
    from kenaz_ml.lifecycle.shutdown import drain_and_exit

    leases = state.leases
    # Startup (refresh_all, model load) ran before any client could connect;
    # the window counts from the moment the engine is reachable.
    leases.restart_countdown()
    while not state.exiting:
        await asyncio.sleep(leases.sweep_interval_sec)
        try:
            leases.sweep()
            if leases.should_exit(is_managed()):
                state.exiting = True
                logger.info(
                    "lifecycle: no live lease for %.0fs (window %.0fs); self-terminating",
                    leases.idle_for(),
                    leases.idle_exit_sec,
                )
                await drain_and_exit(state, reason="idle")
                return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("lifecycle: sweep failed; will retry", exc_info=True)


def create_app(mode: ServingMode | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if mode is None:
        mode = resolve_mode()  # reads KENAZ_ML_MODE env var, defaults to LOCAL

    state = AppState(mode=mode)
    state_tasks: list[asyncio.Task] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Manage startup and shutdown lifecycle for the application."""
        # --- Startup ---
        if state.mode == ServingMode.LOCAL:
            store = create_store()
            state.store = store

            ms = model_store_factory()
            state.model_store = ms

            logger.info("kenaz-ml: using %s data backend, %s model backend", type(store).__name__, type(ms).__name__)

            try:
                store.ensure_tables()
            except Exception:
                logger.warning("schema bootstrap failed (sigild may not have started yet)", exc_info=True)

            # FR-002: the registry's refresh policy runs before the first load,
            # off the event loop. It blocks startup (not the loop) for as long
            # as a due rebuild takes -- serving begins on the refreshed slots.
            await asyncio.get_running_loop().run_in_executor(None, state.refresh_registry, ms)

            state.load_models(ms)

            # /v1/recommend dispatch table (WP03, D-A3): seed kinds + registry.
            populated = await asyncio.get_running_loop().run_in_executor(None, build_table)
            state.dispatch_table.replace_all(populated.snapshot())

            # Initialize signal pipeline (additive, does not modify existing models)
            from kenaz_ml.signals.engine import SignalEngine
            from kenaz_ml.signals.file_recommender import FileRecommender
            from kenaz_ml.signals.next_action import NextActionPredictor
            from kenaz_ml.signals.pattern_detector import PatternDetector
            from kenaz_ml.signals.profile import BehaviorProfile

            profile = BehaviorProfile()
            pattern_detector = PatternDetector()
            next_action_predictor = NextActionPredictor()
            file_recommender = FileRecommender()

            # Load persisted signal models
            next_action_predictor.load(ms)
            file_recommender.load(ms)
            pattern_detector.load(ms)

            signal_engine = SignalEngine(
                store=store,
                profile=profile,
                pattern_detector=pattern_detector,
                next_action=next_action_predictor,
                file_recommender=file_recommender,
            )
            state.signal_engine = signal_engine

            state.poller = EventPoller(
                store=store,
                models={
                    "stuck": state.stuck,
                    "activity": state.activity,
                    "workflow": state.workflow,
                    "duration": state.duration,
                    "quality": state.quality,
                },
                signal_engine=signal_engine,
            )
            asyncio.create_task(state.poller.run())

            scheduler = TrainingScheduler(store, model_store=ms, reload_callback=state.reload_models_into_poller)

            async def _schedule_loop():
                while True:
                    await asyncio.get_event_loop().run_in_executor(None, scheduler.check_and_retrain)
                    await asyncio.sleep(600)

            asyncio.create_task(_schedule_loop())

            lifecycle_task = asyncio.create_task(lifecycle_loop(state))
            state_tasks.append(lifecycle_task)

            logger.info("kenaz-ml: local mode -- models loaded, poller started, scheduler active")
        else:
            # Cloud mode: no SQLite, no poller, no scheduler.
            # Models loaded lazily per-tenant via cache + loader.
            from kenaz_ml.modelstore import FilesystemModelLoader, create_model_cache

            state.model_cache = create_model_cache()
            state.model_loader = FilesystemModelLoader()
            logger.info("kenaz-ml: cloud mode -- stateless serving, cache and loader initialized")

        yield

        # --- Shutdown ---
        for task in state_tasks:
            task.cancel()
        if state.poller:
            state.poller.stop()
            logger.info("poller stopped")
        if state.store:
            state.store.close()
            logger.info("store connection closed")

    application = FastAPI(
        title="kenaz-ml",
        version="0.1.0",
        description=f"kenaz-ml — the ML sidecar for Sigil ({mode.value} mode)",
        lifespan=lifespan,
    )

    register_routes(application, state)
    register_fleet_routes(application, state)

    return application


# Module-level app instance for uvicorn import (kenaz_ml.app:app).
app = create_app()
