# Feature Specification: Cloud Training Pipeline

**Feature Branch**: `004-cloud-training-pipeline`
**Created**: 2026-03-25
**Status**: Draft
**Input**: Add a cloud-oriented training entrypoint to kameas-ml that runs as a K8s CronJob. It reads synced event data from Postgres, trains per-user models, optionally trains aggregate models from pooled opted-in data, and saves weights to S3. This replaces the local background training scheduler in cloud deployments.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Per-Tenant Scheduled Retraining (Priority: P1)

A Sigil Cloud operator schedules a recurring training job in Kubernetes. The job iterates over all tenants that have synced enough new data since their last training run, trains per-user models using the same algorithms as local mode, and writes updated weights to object storage. The prediction API picks up new weights on its next cache refresh.

**Why this priority**: Per-tenant training is the core value proposition of cloud ML. Without it, cloud predictions are stuck on stale or synthetic models.

**Independent Test**: Populate a tenant's Postgres schema with events and completed tasks. Run the training entrypoint for that tenant. Verify new model weights appear in the expected storage location and contain valid trained models.

**Acceptance Scenarios**:

1. **Given** a tenant has 15 completed tasks synced to Postgres since their last training run, **When** the training job executes, **Then** all 5 models are retrained using that tenant's data and weights are saved to their storage prefix.
2. **Given** a tenant has fewer than 10 completed tasks total, **When** the training job executes, **Then** models are trained using synthetic data (matching local cold-start behavior) and weights are saved.
3. **Given** a tenant was already trained within the last hour, **When** the training job executes, **Then** that tenant is skipped (respecting the minimum interval).

---

### User Story 2 - All-Tenant Batch Training (Priority: P1)

The operator runs a single command that discovers all eligible tenants and trains each one sequentially. This is the standard CronJob entrypoint — it handles the full batch without manual intervention.

**Why this priority**: The operator should not need to run one command per tenant. Batch execution is the expected CronJob interface.

**Independent Test**: Create 3 tenants with varying data volumes. Run the batch training command. Verify all eligible tenants are trained and skipped tenants are logged.

**Acceptance Scenarios**:

1. **Given** 10 tenants exist with synced data, **When** `kameas-ml train --mode cloud --all-tenants` is run, **Then** each eligible tenant is trained and a summary report is produced listing trained/skipped/failed tenants.
2. **Given** one tenant's training fails (e.g., corrupted data), **When** the batch runs, **Then** the failure is logged, that tenant is skipped, and remaining tenants continue training.
3. **Given** no tenants have enough new data for retraining, **When** the batch runs, **Then** the job completes successfully with a "nothing to train" summary.

---

### User Story 3 - Aggregate Model Training (Priority: P2)

For Team-tier customers, an aggregate model is trained by pooling events from all opted-in tenants. The aggregate model captures cross-user patterns that no single user's data contains. Team-tier predictions can blend per-user and aggregate model outputs for better accuracy.

**Why this priority**: Aggregate models are the key differentiator for Team tier. However, they require a meaningful number of opted-in users to be valuable, so this is a later milestone.

**Independent Test**: Opt in 5 tenants with diverse event data. Run aggregate training. Verify aggregate model weights are saved to a shared storage location and contain patterns from all contributing tenants.

**Acceptance Scenarios**:

1. **Given** 5 tenants have opted in to data pooling, **When** `kameas-ml train --mode cloud --aggregate` is run, **Then** events from all 5 tenants are combined for training and aggregate weights are saved to a shared storage prefix.
2. **Given** only 2 tenants have opted in, **When** aggregate training is attempted, **Then** the job completes but logs a warning that the dataset may be insufficient for robust aggregate patterns.
3. **Given** aggregate models exist, **When** a Team-tier prediction request is served, **Then** the prediction API can access both the per-user model and the aggregate model for blending.

---

### User Story 4 - Local Training Unchanged (Priority: P1)

A free-tier developer runs `kameas-ml train` locally. The existing training behavior — reading from SQLite, training all models, saving `.joblib` files locally — works identically. The background training scheduler continues to function.

**Why this priority**: Local training must not regress. Cloud training is additive.

**Independent Test**: Run `kameas-ml train` without cloud flags. Verify models train from SQLite and save to the local models directory.

**Acceptance Scenarios**:

1. **Given** kameas-ml is in local mode, **When** `kameas-ml train` is run, **Then** behavior is identical to current implementation.
2. **Given** kameas-ml is in local mode, **When** the background training scheduler triggers, **Then** it retrains from SQLite and reloads models into the running poller.

---

### User Story 5 - Training Observability (Priority: P3)

The operator can observe training job progress and outcomes. Each training run produces structured output indicating which tenants were trained, how many samples were used, training duration, and any failures. Training events are logged for audit.

**Why this priority**: Operational visibility is important for production but not for initial development.

**Independent Test**: Run a batch training job. Verify structured output includes per-tenant status, sample counts, and duration. Verify training events are recorded in the audit log.

**Acceptance Scenarios**:

1. **Given** a batch training job completes, **When** the operator reviews the output, **Then** they see a per-tenant summary: tenant ID, status (trained/skipped/failed), sample count, duration, models trained.
2. **Given** a training run fails for a tenant, **When** the operator reviews the output, **Then** the error message identifies the root cause (insufficient data, storage write failure, etc.).

---

### Edge Cases

- What happens when the Postgres connection drops mid-training? The system should fail that tenant's training gracefully and continue with remaining tenants.
- What happens when S3 is unreachable when saving weights? The training run should be marked as failed for that tenant with a clear error.
- How does the system handle a tenant whose data was partially synced (events exist but no completed tasks)? Training should use synthetic data as in cold-start, not fail.
- What happens when two training jobs overlap (e.g., CronJob takes longer than its interval)? The system should use locking or skip-if-running to prevent concurrent training for the same tenant.
- How does aggregate training handle tenants with vastly different data volumes? Sampling or weighting strategies to prevent large tenants from dominating the aggregate model.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support a `kameas-ml train --mode cloud --tenant <id>` command that trains all models for a single tenant from Postgres data.
- **FR-002**: System MUST support a `kameas-ml train --mode cloud --all-tenants` command that discovers and trains all eligible tenants in batch.
- **FR-003**: System MUST support a `kameas-ml train --mode cloud --aggregate` command that trains aggregate models from pooled opted-in tenant data.
- **FR-004**: The training pipeline MUST use the same model algorithms and training logic as local training (reuse existing trainer module through the DataStore and ModelStore abstractions).
- **FR-005**: Per-tenant training MUST respect a minimum data threshold (10 completed tasks for ML training, falling back to synthetic data below that threshold).
- **FR-006**: Per-tenant training MUST respect a minimum interval between retraining runs (configurable, default: 1 hour).
- **FR-007**: Batch training MUST continue processing remaining tenants when one tenant's training fails, and MUST report all failures in the summary output.
- **FR-008**: Training output MUST be structured (parseable by monitoring systems) and include: tenant ID, status, sample count, models trained, duration.
- **FR-009**: The training pipeline MUST record training events to the audit log for each tenant processed.
- **FR-010**: Aggregate training MUST only include data from tenants who have explicitly opted in to data pooling.
- **FR-011**: The existing local training command (`kameas-ml train`) and background scheduler MUST continue to work without modification.
- **FR-012**: The training pipeline MUST use the DataStore interface (feature 002) for reading data and the ModelStore interface (feature 003) for saving weights.
- **FR-013**: Concurrent training for the same tenant MUST be prevented (via locking or skip-if-running logic).

### Key Entities

- **TrainingRun**: A single execution of model training for one tenant or the aggregate pool. Tracks status, sample count, duration, and which models were trained.
- **TrainingBatch**: A collection of TrainingRuns for all eligible tenants. Produces a summary report.
- **TrainingSummary**: Structured output of a batch run listing per-tenant results: trained, skipped (insufficient data or too recent), or failed (with error).
- **AggregatePool**: The combined dataset from all opted-in tenants used for aggregate model training.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Per-tenant training produces models that generate valid predictions for all 5 model types when loaded by the prediction API.
- **SC-002**: Batch training of 100 tenants completes within 30 minutes (assuming average data volumes).
- **SC-003**: Training failures for individual tenants do not interrupt the batch — 100% of remaining tenants are still processed.
- **SC-004**: Local training (`kameas-ml train`) passes all existing tests with no regressions after cloud training changes.
- **SC-005**: Aggregate models trained on pooled data produce predictions with equal or better accuracy than per-user models for users with limited data (fewer than 50 completed tasks).
- **SC-006**: Training job output is structured and parseable, enabling integration with monitoring and alerting systems.

## Dependencies

- **Feature 002 (Storage Abstraction)**: Training pipeline reads data through the DataStore interface. Requires PostgresStore implementation.
- **Feature 003 (Model Storage Abstraction)**: Training pipeline saves weights through the ModelStore interface. Requires S3ModelStore implementation.
- **External: Postgres with synced data**: Training requires events and tasks synced from user laptops via the sync agent (sigild feature, not kameas-ml).
- **External: S3 bucket**: Model weights are persisted to S3. Must be accessible from the K8s cluster.
- **External: K8s CronJob**: The training entrypoint is invoked as a CronJob. K8s scheduling is configured by the operator, not by kameas-ml.
