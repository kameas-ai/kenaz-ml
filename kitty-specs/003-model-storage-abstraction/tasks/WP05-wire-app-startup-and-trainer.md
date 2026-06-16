---
work_package_id: WP05
title: Wire ModelStore into App Startup & Trainer
lane: planned
dependencies:
- WP01
subtasks:
- T022
- T023
- T024
- T025
- T026
- T027
phase: Phase 3 - Integration
assignee: ''
agent: ''
shell_pid: ''
review_status: ''
reviewed_by: ''
history:
- timestamp: '2026-03-29T16:30:00Z'
  lane: planned
  agent: system
  shell_pid: ''
  action: Prompt generated via /spec-kitty.tasks
requirement_refs:
- FR-010
- FR-007
- FR-011
---

# Work Package Prompt: WP05 -- Wire ModelStore into App Startup & Trainer

## Review Feedback

> **Populated by `/spec-kitty.review`** -- Reviewers add detailed feedback here when work needs changes.

*[This section is empty initially.]*

---

## Objectives & Success Criteria

- Update the application startup sequence to select and instantiate the correct `ModelStore` backend based on operating mode.
- Inject the `ModelStore` into all model classes via `AppState.load_models()`.
- Update `Trainer`, `TrainingScheduler`, and the `/train` endpoint to use the injected `ModelStore`.
- Add startup validation for cloud mode (verify S3 connectivity).
- After this WP, the entire pipeline -- startup, serving, and training -- uses the `ModelStore` abstraction end-to-end.

**Success Criteria**:
- In local mode: behavior is identical to before the refactor. Models load from `~/.local/share/sigild/ml-models/`.
- In cloud mode: models load from S3, cached in memory with TTL.
- `Trainer` saves model weights through the `ModelStore`, not directly to disk.
- Startup in cloud mode validates S3 connectivity and fails with a clear error if misconfigured.
- No model class, trainer, or route handler directly accesses the filesystem for model weights.

## Context & Constraints

- **Spec**: `kitty-specs/003-model-storage-abstraction/spec.md` -- FR-010, FR-007, FR-011.
- **Depends on**: WP01 (protocol + local store), WP02 (S3 store), WP03 (cache), WP04 (refactored models).
- **Key files to modify**:
  - `src/sigil_ml/app.py` -- `AppState`, `create_app()`, startup event
  - `src/sigil_ml/training/trainer.py` -- `Trainer` class
  - `src/sigil_ml/training/scheduler.py` -- `TrainingScheduler` class
  - `src/sigil_ml/routes.py` -- `_run_training()` helper, `/train` endpoint
- **Constraint**: No heavyweight dependencies. Only use existing imports + `sigil_ml.storage`.
- **Constraint**: Cloud mode wraps S3 store with `ModelCache`. Local mode uses `LocalModelStore` directly (no cache needed).
- **Constraint**: The `EventPoller` does NOT change in this WP. It receives model instances (which now internally use the store). The poller's `models` dict API stays the same.

## Subtasks & Detailed Guidance

### Subtask T022 -- Update AppState.load_models() to accept and pass ModelStore

- **Purpose**: `AppState.load_models()` is the central point where all model instances are created. It must now accept a `ModelStore` and pass it to each model constructor.

- **Steps**:
  1. Modify `src/sigil_ml/app.py`:
     ```python
     from __future__ import annotations
     from typing import TYPE_CHECKING

     if TYPE_CHECKING:
         from sigil_ml.storage import ModelStore

     class AppState:
         def __init__(self) -> None:
             self.stuck: StuckPredictor | None = None
             self.activity: ActivityClassifier | None = None
             self.workflow: WorkflowStatePredictor | None = None
             self.duration: DurationEstimator | None = None
             self.quality: QualityEstimator | None = None
             self.poller: EventPoller | None = None
             self.training_in_progress: bool = False
             self.model_store: "ModelStore | None" = None

         def load_models(self, store: "ModelStore") -> None:
             """Load or reload all model instances using the given store."""
             self.model_store = store
             self.stuck = StuckPredictor(store)
             self.activity = ActivityClassifier(store)
             self.workflow = WorkflowStatePredictor(store)
             self.duration = DurationEstimator(store)
             self.quality = QualityEstimator(store)

         def reload_models_into_poller(self) -> None:
             """Reload model instances after retraining."""
             if self.model_store is not None:
                 self.load_models(self.model_store)
             if self.poller:
                 self.poller.stuck = self.stuck
                 self.poller.activity = self.activity
                 self.poller.workflow = self.workflow
                 self.poller.duration = self.duration
                 self.poller.quality = self.quality
             logger.info("models reloaded into poller")
     ```
  2. The `model_store` attribute is stored on `AppState` so that `reload_models_into_poller()` and the training endpoint can access it.

- **Files**:
  - Update: `src/sigil_ml/app.py`

- **Parallel?**: No -- establishes the wiring pattern for T023-T026.

- **Notes**:
  - The `load_models()` method signature changes from `() -> None` to `(store) -> None`. All callers must be updated (T023).
  - `reload_models_into_poller()` uses `self.model_store` to reload -- this ensures the same store (and cache) is used across reloads.

### Subtask T023 -- Update create_app() startup to instantiate the correct ModelStore

- **Purpose**: The startup event in `create_app()` is where the operating mode is determined and the `ModelStore` is created. This is the main wiring point.

- **Steps**:
  1. Modify the `startup_event()` in `src/sigil_ml/app.py`:
     ```python
     from sigil_ml.storage.factory import model_store_factory
     from sigil_ml.storage.cache import ModelCache

     @application.on_event("startup")
     async def startup_event() -> None:
         db = config.db_path()

         try:
             ensure_ml_tables(db)
         except Exception:
             logger.warning("schema bootstrap failed", exc_info=True)

         # Create the appropriate model store
         backend = config.model_store_backend()
         store = model_store_factory(backend=backend)

         # In cloud mode, wrap with cache
         if backend == "s3":
             store = ModelCache(store, ttl_sec=config.model_cache_ttl())
             logger.info("Cloud mode: S3 store with %ds cache TTL", config.model_cache_ttl())
         else:
             logger.info("Local mode: filesystem store")

         state.load_models(store)

         # Only start poller in local mode
         mode = config.serving_mode()
         if mode == "local":
             state.poller = EventPoller(
                 db_path=db,
                 models={
                     "stuck": state.stuck,
                     "activity": state.activity,
                     "workflow": state.workflow,
                     "duration": state.duration,
                     "quality": state.quality,
                 },
             )
             asyncio.create_task(state.poller.run())

             scheduler = TrainingScheduler(
                 db, store, reload_callback=state.reload_models_into_poller
             )

             async def _schedule_loop():
                 while True:
                     await asyncio.get_event_loop().run_in_executor(
                         None, scheduler.check_and_retrain
                     )
                     await asyncio.sleep(600)

             asyncio.create_task(_schedule_loop())

         logger.info(
             "kenaz-ml: mode=%s, backend=%s, models loaded",
             mode, backend,
         )
     ```
  2. **Important**: The poller and training scheduler are only started in local mode. In cloud mode, the service is stateless (no SQLite, no poller). This aligns with Feature 001 (Cloud Serving Mode) requirements.
  3. The `TrainingScheduler` constructor now receives the `store` (see T025).

- **Files**:
  - Update: `src/sigil_ml/app.py`

- **Parallel?**: No -- depends on T022.

- **Notes**:
  - The startup event conditionally wraps with `ModelCache` only in cloud mode. Local mode reads from disk on every load (which is fast and avoids cache staleness issues).
  - The `ensure_ml_tables(db)` call stays even though cloud mode doesn't need SQLite -- it's wrapped in a try/except and won't fail in cloud mode (the db path just won't exist).
  - This subtask introduces a forward dependency on Feature 001 (Cloud Serving Mode) for the `mode == "local"` conditional. For now, the default mode is `local`, so this is backward-compatible.

### Subtask T024 -- Update Trainer to accept ModelStore

- **Purpose**: The `Trainer` class creates model instances during training. These instances must now receive a `ModelStore` so that trained weights are saved through the abstraction.

- **Steps**:
  1. Modify `src/sigil_ml/training/trainer.py`:
     ```python
     from __future__ import annotations
     from typing import TYPE_CHECKING

     if TYPE_CHECKING:
         from sigil_ml.storage import ModelStore

     class Trainer:
         def __init__(self, db_path: str | Path, store: "ModelStore") -> None:
             self.db_path = Path(db_path)
             self._store = store

         def _train_stuck(self) -> int:
             # ... (data loading unchanged) ...
             predictor = StuckPredictor(self._store)
             predictor.train(X, y)
             return len(X)

         def _train_duration(self) -> int:
             # ... (data loading unchanged) ...
             estimator = DurationEstimator(self._store)
             estimator.train(X, y)
             return len(X)
     ```
  2. Update both `StuckPredictor()` and `DurationEstimator()` instantiations to pass `self._store`.
  3. The `train_all()` method is unchanged in structure -- it still calls `_train_stuck()` and `_train_duration()`.

- **Files**:
  - Update: `src/sigil_ml/training/trainer.py`

- **Parallel?**: Yes -- can proceed alongside T025 once T022 establishes the pattern.

- **Notes**:
  - The `Trainer` creates new model instances for training. These are temporary -- the trained model is saved to the store, and the running models are reloaded via `reload_models_into_poller()`.
  - The `Trainer` currently only trains `stuck` and `duration` models. Other models (`activity`, `workflow`, `quality`) have their own training paths. The Trainer can be extended later.

### Subtask T025 -- Update TrainingScheduler to pass ModelStore

- **Purpose**: `TrainingScheduler` creates `Trainer` instances. It must now pass the `ModelStore` through.

- **Steps**:
  1. Modify `src/sigil_ml/training/scheduler.py`:
     ```python
     from __future__ import annotations
     from typing import TYPE_CHECKING

     if TYPE_CHECKING:
         from sigil_ml.storage import ModelStore

     class TrainingScheduler:
         def __init__(
             self, db_path: Path, store: "ModelStore", reload_callback
         ) -> None:
             self.db_path = db_path
             self._store = store
             self._reload = reload_callback
             self._last_retrain: float = 0.0
             self._baseline_tasks: int = self._count_completed()

         def check_and_retrain(self) -> None:
             # ... (unchanged logic) ...
             try:
                 result = Trainer(self.db_path, self._store).train_all()
                 # ... rest unchanged ...
     ```
  2. The only change is adding `store` to the constructor and passing it to `Trainer`.

- **Files**:
  - Update: `src/sigil_ml/training/scheduler.py`

- **Parallel?**: Yes -- can proceed alongside T024.

- **Notes**:
  - The scheduler's own methods (`_count_completed`, `_log_retrain`) still access SQLite directly. These are data operations, not model storage. They are unaffected by the `ModelStore` abstraction.

### Subtask T026 -- Update /train endpoint and _run_training()

- **Purpose**: The `/train` endpoint triggers background training. The `_run_training()` helper must pass the `ModelStore` to the `Trainer`.

- **Steps**:
  1. Modify `src/sigil_ml/routes.py`:
     ```python
     def _run_training(state: AppState, db_path: str) -> None:
         """Run training in a background thread."""
         try:
             state.training_in_progress = True
             if state.model_store is None:
                 logger.error("Cannot train: no model store configured")
                 return
             trainer = Trainer(db_path, state.model_store)
             result = trainer.train_all()
             logger.info("Training complete: %s", result)
             state.reload_models_into_poller()
         except Exception:
             logger.exception("Training failed")
         finally:
             state.training_in_progress = False
     ```
  2. Replace `state.load_models()` with `state.reload_models_into_poller()` -- this reloads models using the existing store.
  3. The `Trainer` now receives `state.model_store`.

- **Files**:
  - Update: `src/sigil_ml/routes.py`

- **Parallel?**: No -- depends on T022 (for `state.model_store` attribute).

- **Notes**:
  - The current code calls `state.load_models()` after training. With the refactor, `reload_models_into_poller()` is the correct method -- it uses the stored `model_store` and updates the poller references.
  - Guard against `model_store is None` defensively, though in practice it should always be set by startup.

### Subtask T027 -- Add startup validation for cloud mode

- **Purpose**: In cloud mode, verify S3 connectivity at startup. Fail with a clear error if the bucket is inaccessible, rather than silently serving fallback predictions for all tenants.

- **Steps**:
  1. Add validation logic after creating the store in `startup_event()`:
     ```python
     if backend == "s3":
         store = ModelCache(store, ttl_sec=config.model_cache_ttl())

         # Validate S3 connectivity
         # The S3ModelStore constructor already validates the bucket.
         # If we get here, the bucket is accessible.
         logger.info(
             "Cloud mode: S3 store validated (bucket=%s, prefix=%s)",
             config.s3_bucket(), config.tenant_id(),
         )
     ```
  2. The `S3ModelStore.__init__` already calls `_validate_bucket()` (from WP02/T006). If the bucket is invalid or credentials are wrong, the constructor raises an exception.
  3. The `startup_event()` should **not** catch this exception -- it should propagate, causing uvicorn to fail loudly at startup. This is the correct fail-fast behavior.
  4. Add a try/except with a clear log message only if a graceful degradation mode is desired:
     ```python
     try:
         store = model_store_factory(backend=backend)
     except (ValueError, PermissionError, ImportError) as e:
         logger.error(
             "FATAL: Failed to initialize model store: %s. "
             "Check SIGIL_MODEL_BUCKET, AWS credentials, and "
             "ensure kenaz-ml[cloud] is installed.",
             e,
         )
         raise SystemExit(1) from e
     ```
  5. Decision: **Fail-fast is preferred.** Cloud deployments should not silently degrade to rule-based-only predictions. The operator must fix the configuration.

- **Files**:
  - Update: `src/sigil_ml/app.py`

- **Parallel?**: No -- integrates into the startup event from T023.

- **Notes**:
  - This is a lightweight subtask -- most validation is already in `S3ModelStore.__init__`. This subtask ensures the startup event handles the validation result correctly.
  - The error message must be actionable: mention specific env vars to check, the `pip install kenaz-ml[cloud]` command, and IAM permission requirements.

## Risks & Mitigations

- **Risk**: Circular imports between `app.py`, `storage/`, and `models/`. **Mitigation**: All storage imports use `TYPE_CHECKING` guards. Runtime imports are in function bodies (e.g., `from sigil_ml.storage.factory import model_store_factory` inside the startup event).
- **Risk**: Breaking the existing local-mode startup path. **Mitigation**: The default backend is `local`, which uses `LocalModelStore` -- a direct equivalent of the current filesystem behavior. The `model_store_factory()` defaults to `local` when `SIGIL_MODE` is not set.
- **Risk**: `reload_models_into_poller()` fails if `model_store` is None. **Mitigation**: Guard with `if self.model_store is not None:` check (already in T022's implementation).
- **Risk**: Training in cloud mode writes to S3, which is slower than local disk. **Mitigation**: Training is already a background task with no latency SLA. S3 write latency (50-200ms) is acceptable.

## Review Guidance

- Verify `create_app()` startup:
  1. Creates the correct store based on mode.
  2. Wraps with `ModelCache` in cloud mode.
  3. Passes store to `state.load_models()`.
  4. Passes store to `TrainingScheduler`.
  5. Only starts poller/scheduler in local mode.
- Verify `Trainer` receives and uses the store.
- Verify `TrainingScheduler` passes the store to `Trainer`.
- Verify `_run_training()` uses `state.model_store`.
- Verify startup fails fast in cloud mode when S3 is misconfigured.
- **Integration test**: Start in local mode, verify everything works as before (no behavioral change). Set `SIGIL_MODE=cloud` with valid S3 config, verify S3 store is selected.

## Implementation Command

```bash
spec-kitty implement WP05 --base WP04
```

## Activity Log

- 2026-03-29T16:30:00Z -- system -- lane=planned -- Prompt created.
