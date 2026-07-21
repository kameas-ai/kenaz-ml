"""FastAPI application factory and model lifecycle."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sigil_ml.signals.engine import SignalEngine

from fastapi import FastAPI

from sigil_ml.config import ServingMode, resolve_mode
from sigil_ml.models.activity import ActivityClassifier
from sigil_ml.models.duration import DurationEstimator
from sigil_ml.models.fleet_routes import register_fleet_routes
from sigil_ml.models.quality import QualityEstimator
from sigil_ml.models.stuck import StuckPredictor
from sigil_ml.models.workflow import WorkflowStatePredictor
from sigil_ml.poller import EventPoller
from sigil_ml.routes import register_routes
from sigil_ml.storage.model_store import ModelStore, model_store_factory
from sigil_ml.store import DataStore, create_store
from sigil_ml.training.scheduler import TrainingScheduler

logger = logging.getLogger("sigil_ml")


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

    def load_models(self, model_store: ModelStore | None = None) -> None:
        """Load or reload all model instances."""
        ms = model_store or self.model_store
        self.stuck = StuckPredictor(model_store=ms)
        self.activity = ActivityClassifier(model_store=ms)
        self.workflow = WorkflowStatePredictor(model_store=ms)
        self.duration = DurationEstimator(model_store=ms)
        self.quality = QualityEstimator(model_store=ms)

    def reload_models_into_poller(self) -> None:
        """Reload model instances after retraining."""
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


def create_app(mode: ServingMode | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if mode is None:
        mode = resolve_mode()  # reads SIGIL_ML_MODE env var, defaults to LOCAL

    state = AppState(mode=mode)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Manage startup and shutdown lifecycle for the application."""
        # --- Startup ---
        if state.mode == ServingMode.LOCAL:
            store = create_store()
            state.store = store

            ms = model_store_factory()
            state.model_store = ms

            logger.info("kameas-ml: using %s data backend, %s model backend", type(store).__name__, type(ms).__name__)

            try:
                store.ensure_tables()
            except Exception:
                logger.warning("schema bootstrap failed (sigild may not have started yet)", exc_info=True)

            state.load_models(ms)

            # Initialize signal pipeline (additive, does not modify existing models)
            from sigil_ml.signals.engine import SignalEngine
            from sigil_ml.signals.file_recommender import FileRecommender
            from sigil_ml.signals.next_action import NextActionPredictor
            from sigil_ml.signals.pattern_detector import PatternDetector
            from sigil_ml.signals.profile import BehaviorProfile

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

            logger.info("kameas-ml: local mode -- models loaded, poller started, scheduler active")
        else:
            # Cloud mode: no SQLite, no poller, no scheduler.
            # Models loaded lazily per-tenant via cache + loader.
            from sigil_ml.cache import create_model_cache
            from sigil_ml.loader import FilesystemModelLoader

            state.model_cache = create_model_cache()
            state.model_loader = FilesystemModelLoader()
            logger.info("kameas-ml: cloud mode -- stateless serving, cache and loader initialized")

        yield

        # --- Shutdown ---
        if state.poller:
            state.poller.stop()
            logger.info("poller stopped")
        if state.store:
            state.store.close()
            logger.info("store connection closed")

    application = FastAPI(
        title="kameas-ml",
        version="0.1.0",
        description=f"Sigil ML sidecar ({mode.value} mode)",
        lifespan=lifespan,
    )

    register_routes(application, state)
    register_fleet_routes(application, state)

    return application


# Module-level app instance for uvicorn import (sigil_ml.app:app).
app = create_app()
