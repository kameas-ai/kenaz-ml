---
work_package_id: WP05
title: Wire ModelStore into App Startup, Trainer & CLI
lane: planned
dependencies:
- WP01
subtasks:
- T023
- T024
- T025
- T026
- T027
- T028
- T029
phase: Phase 3 - Integration
assignee: ''
agent: ''
shell_pid: ''
review_status: ''
reviewed_by: ''
history:
- timestamp: '2026-03-30T01:45:11Z'
  lane: planned
  agent: system
  shell_pid: ''
  action: Prompt generated via /spec-kitty.tasks
requirement_refs:
- FR-010
- FR-007
- FR-011
---

# Work Package Prompt: WP05 -- Wire ModelStore into App Startup, Trainer & CLI

## Review Feedback

> **Populated by `/spec-kitty.review`** -- Reviewers add detailed feedback here when work needs changes.

*[This section is empty initially.]*

---

## Markdown Formatting
Wrap HTML/XML tags in backticks: `` `<div>` ``, `` `<script>` ``
Use language identifiers in code blocks: ````python`, ````bash`

---

## Objectives & Success Criteria

- Update the application startup sequence to select and instantiate the correct `ModelStore` backend based on operating mode (`local` or `cloud`).
- Inject the `ModelStore` into all model classes via `AppState.load_models()`.
- Update `Trainer`, `TrainingScheduler`, and the `/train` endpoint to use the injected `ModelStore`.
- Update the CLI `train` subcommand to create appropriate `ModelStore` via factory.
- Add startup validation for cloud mode (verify S3 connectivity at boot).
- After this WP, the entire pipeline -- startup, serving, training, and CLI -- uses the `ModelStore` abstraction end-to-end.

**Success Criteria**:
- In local mode (default): behavior is identical to before. Models load from `~/.local/share/sigild/ml-models/`.
- In cloud mode (`SIGIL_MODE=cloud`): models load from S3 via `CachedModelStore(S3ModelStore(...))`.
- `Trainer` saves weights through `ModelStore`, not directly to disk.
- `/train` endpoint uses the app's configured `ModelStore`.
- CLI `kameas-ml train` creates the correct `ModelStore` and passes to `Trainer`.
- Cloud mode startup validates S3 connectivity; fails with clear error if misconfigured.
- No model class, trainer, route, or CLI handler directly accesses the filesystem for model weights.

## Context & Constraints

- **Spec**: `kitty-specs/003-model-storage-abstraction/spec.md` -- FR-010, FR-007, FR-011.
- **Depends on**: WP01 (protocol + local store + factory), WP02 (S3 store), WP03 (cache), WP04 (refactored models).
- **Key files to modify**:
  - `src/sigil_ml/app.py` -- `AppState`, `create_app()`, startup/shutdown events
  - `src/sigil_ml/training/trainer.py` -- `Trainer` class
  - `src/sigil_ml/training/scheduler.py` -- `TrainingScheduler` class
  - `src/sigil_ml/routes.py` -- `_run_training()` helper, `/train` endpoint
  - `src/sigil_ml/cli.py` -- `train` subcommand
- **Current `app.py`**: `create_app()` creates `AppState`, calls `state.load_models()` (no args), creates `EventPoller`, creates `TrainingScheduler(db, callback)`. All model loading is filesystem-based.
- **Current `trainer.py`**: `Trainer(db_path)` creates `StuckPredictor()` and `DurationEstimator()` (no store arg). After WP04, these accept optional `model_store` param.
- **Current `scheduler.py`**: `TrainingScheduler(db_path, reload_callback)` creates `Trainer(self.db_path)`.
- **Current `routes.py`**: `_run_training(state, db_path)` creates `Trainer(db_path)` and calls `state.load_models()` after training.
- **Current `cli.py`**: `train` subcommand creates `Trainer(db)` and calls `train_all()`.
- **Constraint**: `storage/` is a leaf module -- no circular imports allowed.
- **Constraint**: Cloud mode wraps S3 with cache. Local mode uses `LocalModelStore` directly.
- **Constraint**: `EventPoller` does NOT change. It receives model instances that internally use the store.

## Subtasks & Detailed Guidance

### Subtask T023 -- Update AppState in `src/sigil_ml/app.py`

- **Purpose**: `AppState.load_models()` is the central point where all model instances are created. It must accept a `ModelStore` and pass it to each model constructor.

- **Steps**:
  1. Add `model_store` attribute to `AppState.__init__`:
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
             self.model_store: ModelStore | None = None
     ```
  2. Update `load_models()` to accept and use the store:
     ```python
     def load_models(self, store: ModelStore) -> None:
         """Load or reload all model instances using the given store."""
         self.model_store = store
         self.stuck = StuckPredictor(model_store=store)
         self.activity = ActivityClassifier(model_store=store)
         self.workflow = WorkflowStatePredictor(model_store=store)
         self.duration = DurationEstimator(model_store=store)
         self.quality = QualityEstimator(model_store=store)
     ```
  3. Update `reload_models_into_poller()`:
     ```python
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

- **Files**:
  - Update: `src/sigil_ml/app.py`

- **Parallel?**: No -- establishes the wiring pattern for T024-T027.

- **Notes**:
  - The `model_store` attribute is stored on `AppState` so `reload_models_into_poller()` and the training endpoint can access it without needing the store passed again.
  - `reload_models_into_poller()` now uses the stored `model_store` to reconstruct all model instances and update poller references.
  - `load_models()` signature changes from `() -> None` to `(store: ModelStore) -> None`. All callers must be updated (T024).

### Subtask T024 -- Update create_app() startup

- **Purpose**: The startup event determines the operating mode, creates the appropriate `ModelStore`, and wires it into `AppState` and the training scheduler.

- **Steps**:
  1. Update the `startup_event()` in `src/sigil_ml/app.py`:
     ```python
     from sigil_ml.storage.model_store import model_store_factory, CachedModelStore

     @application.on_event("startup")
     async def startup_event() -> None:
         db = config.db_path()

         try:
             ensure_ml_tables(db)
         except Exception:
             logger.warning("schema bootstrap failed (sigild may not have started yet)", exc_info=True)

         # Create the model store based on operating mode
         backend = config.model_store_backend()
         try:
             store = model_store_factory(backend=backend)
         except (ValueError, PermissionError, ImportError) as e:
             logger.error(
                 "FATAL: Failed to initialize model store (%s backend): %s. "
                 "Check SIGIL_S3_BUCKET, AWS credentials, and ensure "
                 "kameas-ml[cloud] is installed for cloud mode.",
                 backend, e,
             )
             raise SystemExit(1) from e

         # In cloud mode, wrap with cache
         if backend == "s3":
             from sigil_ml.storage.model_store import CachedModelStore
             store = CachedModelStore(store, ttl_seconds=config.model_cache_ttl())
             logger.info("Cloud mode: S3 store with %ds cache TTL", config.model_cache_ttl())
         else:
             logger.info("Local mode: filesystem store at %s", config.models_dir())

         state.load_models(store)

         # Poller and scheduler only in local mode (cloud is stateless, no SQLite)
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
             db, model_store=store, reload_callback=state.reload_models_into_poller
         )

         async def _schedule_loop():
             while True:
                 await asyncio.get_event_loop().run_in_executor(None, scheduler.check_and_retrain)
                 await asyncio.sleep(600)

         asyncio.create_task(_schedule_loop())

         logger.info(
             "kameas-ml: mode=%s, backend=%s, models loaded, poller started",
             config.serving_mode(), backend,
         )
     ```
  2. **Note**: The poller and scheduler remain for both modes currently. Cloud-only optimization (removing poller in cloud mode) belongs to Feature 001, not this feature. This feature only changes how models are stored, not the overall app lifecycle.

- **Files**:
  - Update: `src/sigil_ml/app.py`

- **Parallel?**: No -- depends on T023.

- **Notes**:
  - Store creation is wrapped in try/except for `ValueError`, `PermissionError`, `ImportError`. These are the errors from `S3ModelStore._validate_bucket()` and the `boto3` import guard.
  - `SystemExit(1)` ensures the process exits loudly when cloud mode is misconfigured. This is the correct behavior for K8s deployments (pod crash loop -> operator investigates).
  - The `CachedModelStore` import is deferred to the `if` block to avoid requiring WP03 imports when running in local mode. Alternatively, it can be at the module top since the import itself doesn't require `boto3`.
  - `TrainingScheduler` now receives `model_store=store` (T026).

### Subtask T025 -- Update Trainer to accept ModelStore

- **Purpose**: `Trainer` creates model instances during training. These instances must receive a `ModelStore` so trained weights are saved through the abstraction.

- **Steps**:
  1. Modify `src/sigil_ml/training/trainer.py`:
     ```python
     from __future__ import annotations

     import logging
     import sqlite3
     import time
     from pathlib import Path
     from typing import TYPE_CHECKING

     import numpy as np

     from sigil_ml.features import extract_duration_features, extract_stuck_features
     from sigil_ml.models.duration import FEATURE_NAMES as DURATION_FEATURES
     from sigil_ml.models.duration import DurationEstimator
     from sigil_ml.models.stuck import FEATURE_NAMES as STUCK_FEATURES
     from sigil_ml.models.stuck import StuckPredictor
     from sigil_ml.training.synthetic import generate_duration_data, generate_stuck_data

     if TYPE_CHECKING:
         from sigil_ml.storage import ModelStore

     logger = logging.getLogger(__name__)


     class Trainer:
         """Orchestrates training of all kameas-ml models from local data."""

         def __init__(self, db_path: str | Path, model_store: "ModelStore | None" = None) -> None:
             self.db_path = Path(db_path)
             self._store = model_store

         # train_all() unchanged in structure

         def _train_stuck(self) -> int:
             # ... data loading logic unchanged ...
             predictor = StuckPredictor(model_store=self._store)
             predictor.train(X, y)
             return len(X)  # or 500

         def _train_duration(self) -> int:
             # ... data loading logic unchanged ...
             estimator = DurationEstimator(model_store=self._store)
             estimator.train(X, y)
             return len(X)  # or 500
     ```
  2. Both `StuckPredictor()` and `DurationEstimator()` instantiations pass `model_store=self._store`.
  3. When `self._store` is `None`, the model classes create `LocalModelStore()` internally (WP04 backward compatibility).

- **Files**:
  - Update: `src/sigil_ml/training/trainer.py`

- **Parallel?**: Yes -- can proceed alongside T026 once T023 establishes the pattern.

- **Notes**:
  - The `Trainer` currently only trains `stuck` and `duration` models. Other models have separate training paths. This is unchanged.
  - Data loading logic (`sqlite3` queries, `extract_stuck_features`, `generate_stuck_data`) is completely untouched.
  - The `Trainer` creates temporary model instances for training. After training, `reload_models_into_poller()` creates fresh instances that load the newly saved weights.

### Subtask T026 -- Update TrainingScheduler to pass ModelStore

- **Purpose**: `TrainingScheduler` creates `Trainer` instances. It must forward the `ModelStore` to keep the pipeline consistent.

- **Steps**:
  1. Modify `src/sigil_ml/training/scheduler.py`:
     ```python
     from __future__ import annotations

     import logging
     import sqlite3
     import time
     from pathlib import Path
     from typing import TYPE_CHECKING

     from sigil_ml.training.trainer import Trainer

     if TYPE_CHECKING:
         from sigil_ml.storage import ModelStore

     logger = logging.getLogger(__name__)

     MIN_NEW_TASKS = 10
     MIN_INTERVAL_SEC = 3600


     class TrainingScheduler:
         def __init__(
             self,
             db_path: Path,
             model_store: "ModelStore | None" = None,
             reload_callback=None,
         ) -> None:
             self.db_path = db_path
             self._store = model_store
             self._reload = reload_callback
             self._last_retrain: float = 0.0
             self._baseline_tasks: int = self._count_completed()

         def check_and_retrain(self) -> None:
             elapsed = time.time() - self._last_retrain
             if elapsed < MIN_INTERVAL_SEC and self._last_retrain > 0:
                 return

             current = self._count_completed()
             if (current - self._baseline_tasks) < MIN_NEW_TASKS:
                 return

             logger.info(
                 "scheduler: triggering retrain (%d new tasks)",
                 current - self._baseline_tasks,
             )
             try:
                 result = Trainer(self.db_path, model_store=self._store).train_all()
                 self._last_retrain = time.time()
                 self._baseline_tasks = current
                 self._log_retrain(result)
                 if self._reload:
                     self._reload()
                 logger.info("scheduler: retrain complete -- %s", result)
             except Exception:
                 logger.exception("scheduler: retrain failed")

         # _count_completed() and _log_retrain() UNCHANGED
     ```
  2. The only structural change is adding `model_store` to constructor and passing it to `Trainer`.

- **Files**:
  - Update: `src/sigil_ml/training/scheduler.py`

- **Parallel?**: Yes -- can proceed alongside T025.

- **Notes**:
  - The scheduler's SQLite operations (`_count_completed`, `_log_retrain`) are data operations, not model storage. They are unaffected by `ModelStore`.
  - `reload_callback` is now keyword argument `reload_callback=None` for clarity (was positional before).

### Subtask T027 -- Update /train endpoint and _run_training()

- **Purpose**: The `/train` endpoint triggers background training. `_run_training()` must use the app's `ModelStore` instead of creating a bare `Trainer`.

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
             trainer = Trainer(db_path, model_store=state.model_store)
             result = trainer.train_all()
             logger.info("Training complete: %s", result)
             state.reload_models_into_poller()
         except Exception:
             logger.exception("Training failed")
         finally:
             state.training_in_progress = False
     ```
  2. **Key change**: `Trainer(db_path)` becomes `Trainer(db_path, model_store=state.model_store)`.
  3. **Key change**: `state.load_models()` becomes `state.reload_models_into_poller()` -- this reloads using the stored `model_store` and updates poller references.
  4. The `/train` endpoint handler itself is unchanged -- it still calls `_run_training(state, db)`.

- **Files**:
  - Update: `src/sigil_ml/routes.py`

- **Parallel?**: No -- depends on T023 (for `state.model_store` attribute).

- **Notes**:
  - The defensive `if state.model_store is None` check guards against an edge case where training is triggered before startup completes. In practice, `model_store` is always set by `startup_event()`.

### Subtask T028 -- Update CLI train subcommand

- **Purpose**: The `kameas-ml train` CLI command must create the appropriate `ModelStore` via factory and pass it to `Trainer`, matching the server's behavior.

- **Steps**:
  1. Modify `src/sigil_ml/cli.py`:
     ```python
     elif args.command == "train":
         from sigil_ml.storage.model_store import model_store_factory

         db = args.db or str(config.db_path())
         try:
             store = model_store_factory()
         except (ValueError, PermissionError, ImportError) as e:
             print(f"Failed to initialize model store: {e}", file=sys.stderr)
             sys.exit(1)

         print(f"Training models from {db} (store: {store.__class__.__name__})...")
         trainer = Trainer(db, model_store=store)
         result = trainer.train_all()
         print(f"Done: {result}")
     ```
  2. The factory is called without arguments, so it reads from env vars (`SIGIL_MODE`, `SIGIL_MODEL_STORE`, etc.).
  3. Error handling catches store initialization failures and prints actionable messages.

- **Files**:
  - Update: `src/sigil_ml/cli.py`

- **Parallel?**: Yes -- independent of T025-T027.

- **Notes**:
  - The import is deferred into the `elif` block to avoid loading storage modules when running `serve` or `health-check` commands.
  - The `Trainer` import already exists at the top of `cli.py`.
  - The printed message now includes the store class name for debugging (`LocalModelStore` or `S3ModelStore`).

### Subtask T029 -- Add cloud startup validation

- **Purpose**: In cloud mode, verify S3 connectivity at startup and fail with a clear, actionable error message if the bucket is inaccessible. Operators must be able to diagnose misconfiguration quickly.

- **Steps**:
  1. The validation is already handled by the try/except in T024's startup event:
     ```python
     try:
         store = model_store_factory(backend=backend)
     except (ValueError, PermissionError, ImportError) as e:
         logger.error(
             "FATAL: Failed to initialize model store (%s backend): %s. "
             "Check SIGIL_S3_BUCKET, AWS credentials, and ensure "
             "kameas-ml[cloud] is installed for cloud mode.",
             backend, e,
         )
         raise SystemExit(1) from e
     ```
  2. The `S3ModelStore.__init__` (from WP02) calls `_validate_bucket()` which raises:
     - `ValueError` for missing/nonexistent bucket
     - `PermissionError` for access denied or no credentials
     - `ImportError` for missing boto3
  3. Verify the error messages mention specific env vars: `SIGIL_S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.
  4. Verify `SystemExit(1)` is raised (not caught and swallowed). The process must exit so K8s can detect the failure.
  5. In local mode, no validation is needed -- `LocalModelStore` always succeeds (it creates the directory).

- **Files**:
  - Verify: `src/sigil_ml/app.py`

- **Parallel?**: No -- integrated into T024's startup event.

- **Notes**:
  - This is a verification/completion subtask. The implementation is in T024's error handling.
  - The fail-fast design ensures cloud deployments surface configuration issues at startup, not on the first prediction request minutes later.
  - In K8s, `SystemExit(1)` causes the pod to crash. The CrashLoopBackOff pattern alerts operators to investigate.

## Risks & Mitigations

- **Risk**: Circular imports between `app.py`, `storage/`, and `models/`. **Mitigation**: `storage/` is a leaf module. `app.py` imports from `storage.model_store` directly. Model imports use `TYPE_CHECKING` guards.
- **Risk**: Breaking the existing local-mode startup path. **Mitigation**: Default backend is `local` (from `config.model_store_backend()`). `model_store_factory()` returns `LocalModelStore()` when `SIGIL_MODE` is not set. Backward compatible.
- **Risk**: `reload_models_into_poller()` fails if `model_store` is `None`. **Mitigation**: Guard with `if self.model_store is not None:` (in T023's implementation).
- **Risk**: Training in cloud mode writes to S3, which has higher latency than local disk. **Mitigation**: Training is already a background task (`BackgroundTasks` in FastAPI). S3 write latency (50-200ms per model) is acceptable for batch operations.
- **Risk**: CLI `train` and server `train` use different store configurations. **Mitigation**: Both use `model_store_factory()` which reads from the same env vars.

## Review Guidance

- **AppState wiring**:
  - `load_models(store)` passes store to all 5 model constructors.
  - `model_store` attribute is set and available for training and reload.
  - `reload_models_into_poller()` uses stored `model_store`.
- **create_app() startup**:
  - Creates correct store via factory.
  - Wraps with `CachedModelStore` in cloud mode only.
  - Error handling catches all store init failures with clear messages.
  - `SystemExit(1)` on cloud misconfiguration.
- **Trainer**: receives and uses `model_store` for model instantiation.
- **TrainingScheduler**: receives `model_store` and forwards to `Trainer`.
- **Routes**: `_run_training()` uses `state.model_store` and calls `reload_models_into_poller()`.
- **CLI**: creates store via factory, passes to `Trainer`.
- **Integration test**: Start in local mode, verify everything works (no behavioral change). Set `SIGIL_MODE=cloud` with valid S3 config, verify S3 store selected in logs.

## Implementation Command

```bash
spec-kitty implement WP05 --base WP04
```

## Activity Log

- 2026-03-30T01:45:11Z -- system -- lane=planned -- Prompt created.
