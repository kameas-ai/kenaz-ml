---
work_package_id: "WP01"
title: "Cloud Training Entrypoint & CLI"
lane: "planned"
dependencies: []
subtasks:
  - "T001"
  - "T002"
  - "T003"
  - "T004"
  - "T005"
  - "T006"
phase: "Phase 1 - Foundation"
assignee: ""
agent: ""
shell_pid: ""
review_status: ""
reviewed_by: ""
requirement_refs:
  - "FR-001"
  - "FR-004"
  - "FR-011"
  - "FR-012"
history:
  - timestamp: "2026-03-30T01:45:09Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt regenerated via /spec-kitty.tasks"
---

# Work Package Prompt: WP01 -- Cloud Training Entrypoint & CLI

## IMPORTANT: Review Feedback Status

**Read this first if you are implementing this task!**

- **Has review feedback?**: Check the `review_status` field above. If it says `has_feedback`, scroll to the **Review Feedback** section immediately.
- **You must address all feedback** before your work is complete.
- **Mark as acknowledged**: When you understand the feedback and begin addressing it, update `review_status: acknowledged` in the frontmatter.

---

## Review Feedback

> **Populated by `/spec-kitty.review`** -- Reviewers add detailed feedback here when work needs changes.

*[This section is empty initially.]*

---

## Markdown Formatting
Wrap HTML/XML tags in backticks: `` `<div>` ``, `` `<script>` ``
Use language identifiers in code blocks: ````python`, ````bash`

---

## Implementation Command

```bash
spec-kitty implement WP01
```

No dependencies -- this is the starting work package.

---

## Objectives & Success Criteria

1. The `kameas-ml train` CLI gains `--mode` (local|cloud), `--tenant <id>`, `--all-tenants`, `--aggregate`, `--min-interval`, `--min-tasks`, and `--max-tasks-per-tenant` flags.
2. A `CloudTrainer` class exists in `src/sigil_ml/training/cloud_trainer.py` that accepts `DataStore` and `ModelStore` protocol objects as constructor dependencies.
3. `CloudTrainer.train_tenant(tenant_id)` returns a `TrainingRun` result (skeleton implementation -- detailed training logic is WP02).
4. Structured dataclasses `TrainingRun`, `TrainingBatch`, `TrainingSummary`, and `CloudTrainingConfig` are defined in `src/sigil_ml/training/models.py`.
5. Running `kameas-ml train --mode cloud --tenant <id>` invokes CloudTrainer and prints structured JSON output.
6. Running `kameas-ml train` (no flags) or `kameas-ml train --mode local` behaves **identically** to the current implementation -- zero changes to the local path (FR-011).
7. Cloud mode validates that `SIGIL_ML_DB_URL` and `SIGIL_ML_S3_BUCKET` environment variables are present before proceeding.

## Context & Constraints

- **Spec**: `kitty-specs/004-cloud-training-pipeline/spec.md` -- User Stories 1 and 4, FR-001, FR-004, FR-011, FR-012
- **Plan**: `kitty-specs/004-cloud-training-pipeline/plan.md` -- Sections D1 (Trainer Refactoring), D2 (CloudTrainer Orchestration), D4 (CLI Extension), D9 (Configuration)
- **Current CLI**: `src/sigil_ml/cli.py` uses `argparse` with subparsers (`serve`, `train`, `health-check`). The `train` subcommand currently only takes `--db`. Extend, do not replace.
- **Current Trainer**: `src/sigil_ml/training/trainer.py` -- `Trainer(db_path)` reads SQLite directly and saves via `config.weights_path()`. This file MUST NOT be modified.
- **Dependencies**: Features 002 (DataStore protocol in `src/sigil_ml/storage/`) and 003 (ModelStore protocol in `src/sigil_ml/model_storage/`) provide the abstract interfaces. If these protocols are not yet available at development time, use `typing.Protocol` stubs in `cloud_trainer.py` that match the expected API.
- **Config pattern**: `src/sigil_ml/config.py` reads from environment variables and XDG paths. Add cloud training config functions following the same style.

### Key Architecture Decision

The `CloudTrainer` wraps the same model training algorithms as the local `Trainer` but sources data from `DataStore` and persists weights via `ModelStore`. It does NOT subclass `Trainer` -- it is a parallel implementation that reuses model classes' `.train()` methods directly. This avoids entangling local and cloud code paths.

---

## Subtasks & Detailed Guidance

### Subtask T001 -- Extend CLI with Cloud Training Flags

- **Purpose**: Add `--mode`, `--tenant`, `--all-tenants`, `--aggregate`, and training-parameter flags to the existing `train` subcommand so operators can invoke cloud training from the command line.
- **Steps**:
  1. Open `src/sigil_ml/cli.py`. Locate the `train_parser` (line ~21).
  2. Add arguments to the existing `train_parser`:
     ```python
     train_parser.add_argument(
         "--mode", choices=["local", "cloud"], default="local",
         help="Training mode: local (SQLite) or cloud (Postgres/S3)"
     )
     train_parser.add_argument(
         "--tenant", type=str, default=None,
         help="Train models for a specific tenant ID (cloud mode only)"
     )
     train_parser.add_argument(
         "--all-tenants", action="store_true", default=False,
         help="Discover and train all eligible tenants (cloud mode only)"
     )
     train_parser.add_argument(
         "--aggregate", action="store_true", default=False,
         help="Train aggregate model from pooled opted-in data (cloud mode only)"
     )
     train_parser.add_argument(
         "--min-interval", type=int, default=None,
         help="Minimum seconds between retraining a tenant (default: 3600)"
     )
     train_parser.add_argument(
         "--min-tasks", type=int, default=None,
         help="Minimum completed tasks for ML training (default: 10)"
     )
     train_parser.add_argument(
         "--max-tasks-per-tenant", type=int, default=None,
         help="Cap per-tenant tasks for aggregate training (default: 1000)"
     )
     ```
  3. Add validation after `args = parser.parse_args()` in the train handler:
     - If `--mode cloud` is set, at least one of `--tenant`, `--all-tenants`, or `--aggregate` must be provided.
     - `--tenant`, `--all-tenants`, and `--aggregate` are mutually exclusive.
     - If `--mode local` and any cloud-only flags are provided, print a clear error.
  4. The local path (`--mode local` or no `--mode` flag) must continue to use the existing `Trainer` class exactly as today.
- **Files**: `src/sigil_ml/cli.py` (modify, ~40 lines added)
- **Parallel?**: No -- this is the CLI skeleton other subtasks wire into.
- **Validation**:
  - [ ] `kameas-ml train --help` shows all new flags
  - [ ] `kameas-ml train --mode cloud` without --tenant/--all-tenants/--aggregate prints a clear error
  - [ ] `kameas-ml train --mode cloud --tenant X --all-tenants` prints a mutual exclusivity error
  - [ ] `kameas-ml train` (no flags) still works identically to current behavior

### Subtask T002 -- Create CloudTrainer Class Skeleton

- **Purpose**: Establish the `CloudTrainer` class that will orchestrate all cloud training operations. It accepts abstract `DataStore` and `ModelStore` interfaces, making it testable with mocks.
- **Steps**:
  1. Create `src/sigil_ml/training/cloud_trainer.py`.
  2. Import the DataStore and ModelStore protocols. If features 002/003 are not yet available, define Protocol stubs:
     ```python
     """Cloud training orchestrator using DataStore and ModelStore abstractions."""

     from __future__ import annotations

     import logging
     import time
     from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

     from sigil_ml.training.models import CloudTrainingConfig, TrainingBatch, TrainingRun, TrainingSummary

     logger = logging.getLogger(__name__)


     # Protocol stubs -- replace with imports from features 002/003 when available
     try:
         from sigil_ml.storage.datastore import DataStore
         from sigil_ml.model_storage.model_store import ModelStore
     except ImportError:
         @runtime_checkable
         class DataStore(Protocol):
             """Data access interface (feature 002 stub)."""
             def query_completed_tasks(self, tenant_id: str) -> list[dict]: ...
             def query_events_for_task(self, tenant_id: str, task_id: str) -> list[dict]: ...
             def get_last_training_ts(self, tenant_id: str) -> int | None: ...
             def record_training_event(self, tenant_id: str, event: dict) -> None: ...
             def list_tenants(self) -> list[str]: ...
             def list_opted_in_tenants(self) -> list[str]: ...
             def for_tenant(self, tenant_id: str) -> "DataStore": ...

         @runtime_checkable
         class ModelStore(Protocol):
             """Model weight storage interface (feature 003 stub)."""
             def load(self, model_name: str) -> bytes | None: ...
             def save(self, model_name: str, data: bytes) -> None: ...
             def for_tenant(self, tenant_id: str) -> "ModelStore": ...
     ```
  3. Define the `CloudTrainer` class:
     ```python
     class CloudTrainer:
         """Orchestrates model training for cloud deployments.

         Uses DataStore for reading training data and ModelStore for
         persisting trained model weights. Supports per-tenant, batch,
         and aggregate training modes.
         """

         def __init__(
             self,
             data_store: DataStore,
             model_store: ModelStore,
             config: CloudTrainingConfig | None = None,
         ) -> None:
             self.data_store = data_store
             self.model_store = model_store
             self.config = config or CloudTrainingConfig()

         def train_tenant(self, tenant_id: str) -> TrainingRun:
             """Train all models for a single tenant."""
             raise NotImplementedError("Skeleton -- implemented in T003")

         def train_all_tenants(self) -> TrainingBatch:
             """Discover and train all eligible tenants."""
             raise NotImplementedError("Implemented in WP03")

         def train_aggregate(self) -> TrainingRun:
             """Train aggregate models from pooled opted-in data."""
             raise NotImplementedError("Implemented in WP05")
     ```
  4. Add proper module docstring, `__all__` export, and type hints.
- **Files**: `src/sigil_ml/training/cloud_trainer.py` (new, ~80 lines)
- **Parallel?**: No -- T003 and T005 depend on this.
- **Notes**: The Protocol stubs serve as a contract. When features 002/003 land, replace the try/except block with direct imports. The stubs document exactly what CloudTrainer expects from the data layer.

### Subtask T003 -- Implement train_tenant() Skeleton

- **Purpose**: Provide the initial `train_tenant()` implementation that queries data, records timing, and returns a `TrainingRun`. Full training logic (threshold checks, interval enforcement, feature extraction, model training) comes in WP02 -- this subtask establishes the structural scaffolding and end-to-end wiring.
- **Steps**:
  1. In `CloudTrainer.train_tenant()`:
     ```python
     def train_tenant(self, tenant_id: str) -> TrainingRun:
         """Train all models for a single tenant.

         Skeleton implementation that validates end-to-end wiring.
         Full training logic is added in WP02.
         """
         start = time.time()
         logger.info("Starting training for tenant %s", tenant_id)

         try:
             # Query completed tasks to validate DataStore connectivity
             tasks = self.data_store.query_completed_tasks(tenant_id)
             sample_count = len(tasks)

             # TODO (WP02): threshold check, interval check
             # TODO (WP02): feature extraction from DataStore
             # TODO (WP02): train all 5 model types
             # TODO (WP02): save weights via ModelStore
             # TODO (WP02): record audit event

             elapsed = time.time() - start
             return TrainingRun(
                 tenant_id=tenant_id,
                 status="trained",
                 sample_count=sample_count,
                 models_trained=[],  # populated in WP02
                 duration_ms=int(elapsed * 1000),
             )
         except Exception as e:
             elapsed = time.time() - start
             logger.error("Training failed for tenant %s: %s", tenant_id, e, exc_info=True)
             return TrainingRun(
                 tenant_id=tenant_id,
                 status="failed",
                 error=str(e),
                 duration_ms=int(elapsed * 1000),
             )
     ```
  2. This skeleton validates the end-to-end wiring: CLI -> CloudTrainer -> DataStore -> result. It returns a valid `TrainingRun` so T005 (CLI wiring) can be tested immediately.
- **Files**: `src/sigil_ml/training/cloud_trainer.py` (modify, ~30 lines added)
- **Parallel?**: No -- depends on T002.
- **Validation**:
  - [ ] `train_tenant("test-1")` returns a `TrainingRun` with correct fields
  - [ ] Errors during DataStore queries are caught and returned as a failed TrainingRun
  - [ ] Timing is captured in `duration_ms`

### Subtask T004 -- Create Training Data Models

- **Purpose**: Define the dataclasses that structure training output and configuration. These are the shared data contracts used across all WPs.
- **Steps**:
  1. Create `src/sigil_ml/training/models.py`:
     ```python
     """Data models for the cloud training pipeline."""

     from __future__ import annotations

     import json
     from dataclasses import dataclass, field
     from datetime import datetime, timezone
     from typing import Any


     @dataclass
     class CloudTrainingConfig:
         """Configuration for cloud training runs."""
         min_interval_sec: int = 3600
         min_tasks: int = 10
         max_tasks_per_tenant: int = 1000
         aggregate_min_tenants: int = 3


     @dataclass
     class TrainingRun:
         """Result of training models for a single tenant or aggregate pool."""
         tenant_id: str
         status: str  # "trained", "failed", "skipped", "skipped_locked"
         models_trained: list[str] = field(default_factory=list)
         sample_count: int = 0
         duration_ms: int = 0
         error: str | None = None
         started_at: datetime | None = None
         completed_at: datetime | None = None

         def to_dict(self) -> dict[str, Any]:
             """Serialize to a JSON-compatible dictionary."""
             d: dict[str, Any] = {
                 "tenant_id": self.tenant_id,
                 "status": self.status,
                 "models_trained": self.models_trained,
                 "sample_count": self.sample_count,
                 "duration_ms": self.duration_ms,
             }
             if self.error is not None:
                 d["error"] = self.error
             if self.started_at is not None:
                 d["started_at"] = self.started_at.isoformat()
             if self.completed_at is not None:
                 d["completed_at"] = self.completed_at.isoformat()
             return d

         def to_json(self, indent: int | None = 2) -> str:
             return json.dumps(self.to_dict(), indent=indent)


     @dataclass
     class TrainingBatch:
         """Aggregated result of batch training across multiple tenants."""
         runs: list[TrainingRun] = field(default_factory=list)
         total_duration_ms: int = 0
         started_at: datetime | None = None
         completed_at: datetime | None = None

         @property
         def trained(self) -> int:
             return sum(1 for r in self.runs if r.status == "trained")

         @property
         def skipped(self) -> int:
             return sum(1 for r in self.runs if r.status.startswith("skipped"))

         @property
         def failed(self) -> int:
             return sum(1 for r in self.runs if r.status == "failed")

         @property
         def total(self) -> int:
             return len(self.runs)

         def to_dict(self) -> dict[str, Any]:
             return {
                 "total": self.total,
                 "trained": self.trained,
                 "skipped": self.skipped,
                 "failed": self.failed,
                 "total_duration_ms": self.total_duration_ms,
                 "runs": [r.to_dict() for r in self.runs],
             }

         def to_json(self, indent: int | None = 2) -> str:
             return json.dumps(self.to_dict(), indent=indent)


     @dataclass
     class TrainingSummary:
         """Human-readable summary for CLI output."""
         mode: str  # "single", "batch", "aggregate"
         total_tenants: int = 0
         trained: int = 0
         skipped: int = 0
         failed: int = 0
         total_samples: int = 0
         total_duration_ms: int = 0
         per_tenant: list[dict] = field(default_factory=list)

         def to_dict(self) -> dict[str, Any]:
             return {
                 "mode": self.mode,
                 "total_tenants": self.total_tenants,
                 "trained": self.trained,
                 "skipped": self.skipped,
                 "failed": self.failed,
                 "total_samples": self.total_samples,
                 "total_duration_ms": self.total_duration_ms,
                 "per_tenant": self.per_tenant,
             }

         def to_json(self, indent: int | None = 2) -> str:
             return json.dumps(self.to_dict(), indent=indent)
     ```
  2. All dataclasses use `from __future__ import annotations` for forward references.
  3. Every dataclass has a `to_dict()` method producing JSON-serializable output and a `to_json()` convenience method.
  4. `TrainingBatch` has computed properties (`trained`, `skipped`, `failed`) that count by status.
- **Files**: `src/sigil_ml/training/models.py` (new, ~110 lines)
- **Parallel?**: Yes -- defines data structures only, no dependencies on T002/T003.
- **Validation**:
  - [ ] All dataclasses can be instantiated with defaults: `TrainingRun(tenant_id="x", status="trained")`
  - [ ] `TrainingBatch.trained`, `.skipped`, `.failed` compute correctly from `.runs`
  - [ ] `to_dict()` produces dicts that pass `json.dumps()` without errors
  - [ ] `to_json()` produces valid JSON strings

### Subtask T005 -- Wire CLI Cloud Tenant Path

- **Purpose**: Connect the `--mode cloud --tenant <id>` CLI path to `CloudTrainer.train_tenant()` and print the result as structured JSON.
- **Steps**:
  1. In `cli.py`, in the `train` command handler, add the cloud dispatch:
     ```python
     elif args.command == "train":
         if args.mode == "cloud":
             # Validate cloud flags
             cloud_actions = [args.tenant, args.all_tenants, args.aggregate]
             if not any(cloud_actions):
                 print("Error: Cloud mode requires --tenant, --all-tenants, or --aggregate", file=sys.stderr)
                 sys.exit(1)
             if sum(bool(a) for a in cloud_actions) > 1:
                 print("Error: --tenant, --all-tenants, --aggregate are mutually exclusive", file=sys.stderr)
                 sys.exit(1)

             # Validate required environment variables
             from sigil_ml.config import cloud_db_url, cloud_s3_bucket
             db_url = cloud_db_url()
             s3_bucket = cloud_s3_bucket()
             if not db_url or not s3_bucket:
                 print("Error: SIGIL_ML_DB_URL and SIGIL_ML_S3_BUCKET required for cloud mode", file=sys.stderr)
                 sys.exit(1)

             # Construct stores from config
             from sigil_ml.training.cloud_trainer import CloudTrainer
             from sigil_ml.config import cloud_training_config
             # TODO: Replace with real store constructors when features 002/003 land
             data_store = _create_data_store(db_url)
             model_store = _create_model_store(s3_bucket)
             cfg = cloud_training_config(
                 min_interval=args.min_interval,
                 min_tasks=args.min_tasks,
                 max_tasks_per_tenant=args.max_tasks_per_tenant,
             )
             trainer = CloudTrainer(data_store, model_store, cfg)

             if args.tenant:
                 result = trainer.train_tenant(args.tenant)
                 print(result.to_json())
                 sys.exit(0 if result.status != "failed" else 1)
             elif args.all_tenants:
                 print("Error: Batch training not yet implemented (see WP03)", file=sys.stderr)
                 sys.exit(1)
             elif args.aggregate:
                 print("Error: Aggregate training not yet implemented (see WP05)", file=sys.stderr)
                 sys.exit(1)
         else:
             # Existing local training path -- COMPLETELY UNCHANGED
             db = args.db or str(config.db_path())
             print(f"Training models from {db} ...")
             trainer = Trainer(db)
             result = trainer.train_all()
             print(f"Done: {result}")
     ```
  2. Add cloud config functions to `src/sigil_ml/config.py`:
     ```python
     def cloud_db_url() -> str | None:
         """Return the Postgres connection URL for cloud mode."""
         return os.environ.get("SIGIL_ML_DB_URL")

     def cloud_s3_bucket() -> str | None:
         """Return the S3 bucket name for model weight storage."""
         return os.environ.get("SIGIL_ML_S3_BUCKET")

     def cloud_s3_region() -> str:
         """Return the AWS region for S3."""
         return os.environ.get("SIGIL_ML_S3_REGION", "us-east-1")

     def cloud_s3_endpoint() -> str | None:
         """Return the S3-compatible endpoint (e.g., MinIO). None uses AWS."""
         return os.environ.get("SIGIL_ML_S3_ENDPOINT")

     def cloud_training_config(
         min_interval: int | None = None,
         min_tasks: int | None = None,
         max_tasks_per_tenant: int | None = None,
     ) -> "CloudTrainingConfig":
         """Build a CloudTrainingConfig from env vars with CLI overrides."""
         from sigil_ml.training.models import CloudTrainingConfig
         return CloudTrainingConfig(
             min_interval_sec=min_interval or int(os.environ.get("SIGIL_ML_TRAIN_MIN_INTERVAL", "3600")),
             min_tasks=min_tasks or int(os.environ.get("SIGIL_ML_TRAIN_MIN_TASKS", "10")),
             max_tasks_per_tenant=max_tasks_per_tenant or int(os.environ.get("SIGIL_ML_TRAIN_MAX_TASKS_PER_TENANT", "1000")),
         )
     ```
  3. Create stub factory functions for data_store and model_store in `cli.py`:
     ```python
     def _create_data_store(db_url: str):
         """Create a DataStore from configuration. Stub until features 002/003 land."""
         try:
             from sigil_ml.storage.postgres_store import PostgresStore
             return PostgresStore(db_url)
         except ImportError:
             raise SystemExit("Error: DataStore (feature 002) not yet available")

     def _create_model_store(s3_bucket: str):
         """Create a ModelStore from configuration. Stub until features 002/003 land."""
         try:
             from sigil_ml.model_storage.s3_store import S3ModelStore
             return S3ModelStore(s3_bucket)
         except ImportError:
             raise SystemExit("Error: ModelStore (feature 003) not yet available")
     ```
- **Files**: `src/sigil_ml/cli.py` (modify, ~50 lines), `src/sigil_ml/config.py` (modify, ~35 lines)
- **Parallel?**: No -- depends on T001 (CLI flags), T002 (CloudTrainer), T004 (TrainingRun).
- **Validation**:
  - [ ] `SIGIL_ML_DB_URL=x SIGIL_ML_S3_BUCKET=y kameas-ml train --mode cloud --tenant test-1` runs without crashing (or errors on missing store implementations, which is expected)
  - [ ] Missing env vars produce: "Error: SIGIL_ML_DB_URL and SIGIL_ML_S3_BUCKET required for cloud mode"
  - [ ] CLI args override env var defaults: `--min-interval 1800` overrides the default 3600

### Subtask T006 -- Ensure Local Training Unchanged

- **Purpose**: Verify that the existing local training path is completely unmodified and continues to work identically. This satisfies FR-011.
- **Steps**:
  1. Verify the local path in `cli.py`: when `args.mode == "local"` (or mode is unset), the code executes:
     ```python
     db = args.db or str(config.db_path())
     print(f"Training models from {db} ...")
     trainer = Trainer(db)
     result = trainer.train_all()
     print(f"Done: {result}")
     ```
     This must be character-for-character identical to the pre-WP01 code.
  2. Verify `src/sigil_ml/training/trainer.py` has NO modifications (zero diff from pre-WP01 state). The `Trainer.__init__` signature must remain `def __init__(self, db_path: str | Path) -> None`.
  3. Verify `src/sigil_ml/training/scheduler.py` has NO modifications. `TrainingScheduler` must remain unchanged.
  4. Verify `src/sigil_ml/app.py` has NO modifications. The startup sequence (model loading, poller, scheduler) must be unchanged.
  5. Use lazy imports for all cloud-specific modules (inside the `if args.mode == "cloud"` branch). The local path must never trigger imports of `cloud_trainer`, `postgres_store`, `s3_store`, etc.
  6. Run existing tests: `pytest tests/` -- all must pass.
- **Files**:
  - `src/sigil_ml/cli.py` (verify local path unchanged)
  - `src/sigil_ml/training/trainer.py` (verify NO modifications)
  - `src/sigil_ml/training/scheduler.py` (verify NO modifications)
  - `src/sigil_ml/app.py` (verify NO modifications)
- **Parallel?**: No -- validation step after T001-T005.
- **Validation**:
  - [ ] `git diff src/sigil_ml/training/trainer.py` shows no changes
  - [ ] `git diff src/sigil_ml/training/scheduler.py` shows no changes
  - [ ] `git diff src/sigil_ml/app.py` shows no changes
  - [ ] `pytest tests/` passes with no regressions
  - [ ] `kameas-ml train --help` shows new flags without breaking old behavior
  - [ ] `kameas-ml train` (no flags) executes the local path

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Features 002/003 protocols not yet available | Medium | Medium | Use Protocol stubs with try/except imports; replace when features land |
| CLI argument parsing conflicts | Low | Low | Validate mutual exclusivity early; use clear error messages |
| Cloud imports breaking local mode | Medium | High | Use lazy imports inside the cloud branch only. The local path must never touch cloud imports. |
| DataStore/ModelStore API mismatch | Medium | Medium | Protocol stubs document expectations; update when real interfaces are known |

## Review Guidance

- **Critical check -- local path unchanged**: Diff `trainer.py`, `scheduler.py`, and `app.py` against their pre-WP01 state. There must be zero changes.
- **Protocol usage**: Confirm `CloudTrainer` only imports protocol types (or stubs), never concrete store implementations.
- **CLI validation**: Verify all mutual exclusivity rules are enforced with clear error messages.
- **Config pattern**: Cloud config functions in `config.py` should follow the same style as existing functions (env var with fallback).
- **Dataclass completeness**: Verify all four dataclasses (`TrainingRun`, `TrainingBatch`, `TrainingSummary`, `CloudTrainingConfig`) have all fields documented in `plan.md` section "Data Model".
- **Lazy imports**: No cloud-specific `import` statements at module level in `cli.py`.

---

## Activity Log

- 2026-03-30T01:45:09Z -- system -- lane=planned -- Prompt regenerated.
