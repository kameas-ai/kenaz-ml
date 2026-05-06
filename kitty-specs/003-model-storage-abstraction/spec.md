# Feature Specification: Model Storage Abstraction

**Feature Branch**: `003-model-storage-abstraction`
**Created**: 2026-03-25
**Status**: Draft
**Input**: Introduce a ModelStore protocol that abstracts how kenaz-ml loads and saves trained model weights. Provide a local filesystem implementation (preserving current behavior) and an S3 implementation (for cloud/K8s deployment with per-tenant model isolation).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Local Model Loading Unchanged (Priority: P1)

A free-tier developer runs kenaz-ml locally. Model weights are loaded from and saved to `~/.local/share/sigild/ml-models/*.joblib` exactly as they are today. The abstraction is invisible to the local experience.

**Why this priority**: Current behavior must be preserved. The abstraction must not regress model loading for local users.

**Independent Test**: Run the existing test suite. All model load/save operations work. Start the server, trigger training, verify `.joblib` files appear in the expected directory.

**Acceptance Scenarios**:

1. **Given** kenaz-ml is in local mode, **When** a model is loaded, **Then** it reads from `~/.local/share/sigild/ml-models/{model_name}.joblib`.
2. **Given** kenaz-ml is in local mode, **When** a model is trained and saved, **Then** the `.joblib` file is written to the local models directory.
3. **Given** a model file does not exist on disk, **When** loading is attempted, **Then** the system returns None and the model falls back to rule-based predictions (matching current behavior).

---

### User Story 2 - Cloud Mode Loads Models from Object Storage (Priority: P1)

A cloud deployment of kenaz-ml loads model weights from an S3-compatible object store. Each tenant's models are stored under a tenant-specific prefix. When a prediction request arrives, the service loads the correct tenant's model weights from the configured bucket.

**Why this priority**: Without remote model storage, cloud kenaz-ml cannot serve tenant-specific predictions.

**Independent Test**: Configure kenaz-ml with an S3 bucket, upload a model file to a tenant prefix, start the service in cloud mode, and verify the model loads correctly for prediction requests.

**Acceptance Scenarios**:

1. **Given** kenaz-ml is in cloud mode with an S3 bucket configured, **When** a prediction request arrives for tenant A, **Then** the model weights are loaded from `s3://{bucket}/{tenant_a}/{model_name}.joblib`.
2. **Given** no model exists in S3 for a tenant, **When** loading is attempted, **Then** the system returns None and falls back to rule-based predictions.
3. **Given** model weights exist in S3, **When** the training pipeline saves updated weights, **Then** the new weights are written to the correct tenant prefix in S3.

---

### User Story 3 - Model Weight Caching (Priority: P2)

In cloud mode, model weights are cached in memory after first load. Subsequent prediction requests for the same tenant reuse the cached model without re-downloading from S3. The cache has a configurable TTL so that updated models are eventually picked up.

**Why this priority**: Without caching, every prediction request would require an S3 download, adding unacceptable latency. This is critical for production but can be added after basic S3 loading works.

**Independent Test**: Load a model, make two prediction requests, verify S3 is only accessed once. Wait for the TTL to expire, make another request, verify S3 is accessed again.

**Acceptance Scenarios**:

1. **Given** a model is loaded from S3 for tenant A, **When** a second prediction request arrives for tenant A within the cache TTL, **Then** the cached model is used without accessing S3.
2. **Given** the cache TTL has expired for tenant A's model, **When** the next prediction request arrives, **Then** the model is re-fetched from S3.
3. **Given** the training pipeline writes updated weights to S3, **When** the cache TTL expires and the model is reloaded, **Then** the new weights are used for subsequent predictions.

---

### User Story 4 - Training Pipeline Saves to Correct Backend (Priority: P2)

The training pipeline (both local and cloud) saves trained model weights through the ModelStore interface. In local mode it writes to disk. In cloud mode it writes to S3. The trainer code does not know which backend is in use.

**Why this priority**: Training must be backend-agnostic for the same reason as serving. However, training is less latency-sensitive than serving.

**Independent Test**: Run training in local mode, verify `.joblib` files on disk. Run training in cloud mode, verify objects in S3 at the expected prefix.

**Acceptance Scenarios**:

1. **Given** training completes in local mode, **When** the trainer saves model weights, **Then** the ModelStore writes to the local filesystem.
2. **Given** training completes in cloud mode for tenant A, **When** the trainer saves model weights, **Then** the ModelStore writes to `s3://{bucket}/{tenant_a}/{model_name}.joblib`.

---

### Edge Cases

- What happens when S3 is temporarily unreachable during model load? The system should serve rule-based fallback predictions and retry on subsequent requests.
- What happens when a cached model's `.joblib` file is corrupted in S3? The system should catch deserialization errors, evict the cache entry, and fall back to rule-based predictions.
- What happens when the S3 bucket doesn't exist or credentials are invalid? The system should fail clearly at startup with an actionable error message.
- How does the system handle concurrent model saves from multiple training pipeline instances? S3's last-writer-wins semantics are acceptable; no locking is required.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST define a ModelStore protocol (interface) with methods for loading and saving model weights by name.
- **FR-002**: The ModelStore protocol MUST support a `load(model_name) -> bytes | None` operation that returns serialized model weights or None if not found.
- **FR-003**: The ModelStore protocol MUST support a `save(model_name, data) -> None` operation that persists serialized model weights.
- **FR-004**: System MUST provide a local filesystem implementation that reads/writes `.joblib` files from the configured models directory (default: `~/.local/share/sigild/ml-models/`).
- **FR-005**: System MUST provide an S3 implementation that reads/writes model files from an S3-compatible object store with a per-tenant key prefix.
- **FR-006**: The S3 implementation MUST accept configuration via environment variables or config file: bucket name, region, and optional endpoint URL (for S3-compatible stores like MinIO).
- **FR-007**: All model classes (StuckPredictor, ActivityClassifier, WorkflowStatePredictor, DurationEstimator, QualityEstimator) MUST load and save weights through the ModelStore interface instead of directly accessing the filesystem.
- **FR-008**: In cloud mode, loaded models MUST be cached in memory with a configurable TTL (default: 5 minutes).
- **FR-009**: The cache MUST be keyed by tenant ID and model name so that multiple tenants' models can coexist in memory.
- **FR-010**: The app startup sequence MUST select the appropriate ModelStore backend based on the operating mode (local → filesystem, cloud → S3).
- **FR-011**: When a model cannot be loaded (missing, corrupted, backend unreachable), the system MUST fall back to rule-based predictions rather than returning an error.

### Key Entities

- **ModelStore**: The protocol (interface) for loading and saving serialized model weights.
- **LocalModelStore**: Implementation backed by the local filesystem. Reads/writes `.joblib` files.
- **S3ModelStore**: Implementation backed by S3-compatible object storage. Supports per-tenant key prefixes.
- **ModelCache**: In-memory cache wrapping any ModelStore, adding TTL-based expiration. Used in cloud mode to avoid repeated remote reads.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All existing model tests pass without modification after the refactor.
- **SC-002**: No model class directly imports filesystem I/O operations (e.g., `open()`, `pathlib`, `os.path`) for weight persistence.
- **SC-003**: Model load latency from local filesystem is within 5% of current performance (no regression).
- **SC-004**: Cached model load in cloud mode adds less than 1ms of overhead compared to direct in-memory access.
- **SC-005**: Model weights round-trip correctly through both backends: save via one instance, load via another, predictions match.

## Dependencies

- **Feature 001 (Cloud Serving Mode)**: Determines when S3 backend is selected vs local filesystem.
- **Feature 002 (Storage Abstraction)**: Parallel effort but independent — DataStore handles event/prediction data, ModelStore handles weight files. No dependency between them.
- **External: S3 or S3-compatible store**: Cloud mode requires an accessible S3 bucket. MinIO can be used for local development and testing.
