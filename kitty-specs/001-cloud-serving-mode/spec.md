# Feature Specification: Cloud Serving Mode

**Feature Branch**: `001-cloud-serving-mode`
**Created**: 2026-03-25
**Status**: Draft
**Input**: Add a `--mode cloud` option to kameas-ml so it can run as a stateless prediction API in Kubernetes, serving requests from remote Go daemons without a local SQLite poller.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Stateless Prediction Serving (Priority: P1)

A Sigil Cloud operator deploys kameas-ml in Kubernetes with `--mode cloud`. The service starts without a poller, without SQLite, and without local model file dependencies. It accepts prediction requests over HTTP from authenticated Go daemons and returns results. The operator can horizontally scale the deployment based on request rate.

**Why this priority**: This is the core capability that enables the entire cloud ML offering. Without stateless serving, kameas-ml cannot run in K8s.

**Independent Test**: Start kameas-ml with `--mode cloud`, send a prediction request to any `/predict/*` endpoint with features in the request body, and receive a valid prediction response. No SQLite database exists on the machine.

**Acceptance Scenarios**:

1. **Given** kameas-ml is started with `--mode cloud`, **When** a `POST /predict/stuck` request arrives with a feature payload, **Then** the service returns a prediction response with probability and confidence fields.
2. **Given** kameas-ml is started with `--mode cloud`, **When** the service starts, **Then** no SQLite connection is opened and no poller background task is created.
3. **Given** kameas-ml is started with `--mode cloud`, **When** `GET /health` is called, **Then** the response indicates cloud mode and reports model availability.

---

### User Story 2 - Local Mode Unchanged (Priority: P1)

A free-tier developer runs `kameas-ml serve` (no mode flag) and the service behaves identically to today: the poller starts, SQLite is read, predictions are written back to the database. No regressions from the cloud mode changes.

**Why this priority**: Protecting the existing local-first experience is equally critical. Cloud mode must not break local users.

**Independent Test**: Run `kameas-ml serve` without `--mode` flag, verify the poller starts and predictions appear in SQLite.

**Acceptance Scenarios**:

1. **Given** kameas-ml is started with no mode flag, **When** events exist in SQLite, **Then** the poller reads them and writes predictions to `ml_predictions`.
2. **Given** kameas-ml is started with `--mode local`, **When** the service starts, **Then** behavior is identical to the current default.

---

### User Story 3 - Multi-Tenant Request Routing (Priority: P2)

The cloud prediction API receives requests from multiple Go daemons belonging to different tenants. Each request includes a tenant identifier (set by the API gateway). The service loads the correct model weights for that tenant and returns tenant-specific predictions.

**Why this priority**: Multi-tenancy is required before the cloud service can serve more than one user, but a single-tenant deployment is viable for initial testing.

**Independent Test**: Send two prediction requests with different tenant identifiers and verify each receives predictions from their respective models.

**Acceptance Scenarios**:

1. **Given** two tenants with different trained models, **When** tenant A sends a stuck prediction request, **Then** the response reflects tenant A's model weights.
2. **Given** a request arrives with an unknown tenant identifier, **When** no model weights exist for that tenant, **Then** the service returns a graceful fallback (rule-based prediction) rather than an error.

---

### User Story 4 - Health and Observability (Priority: P3)

An operator monitoring the cloud deployment can check service health, see which models are loaded, and observe per-tenant request metrics. The `/health` and `/status` endpoints reflect cloud-mode state accurately.

**Why this priority**: Operational visibility is needed for production but not for initial development or testing.

**Independent Test**: Start in cloud mode, call `/health` and `/status`, verify responses reflect cloud-specific information (mode, loaded tenants, no poller status).

**Acceptance Scenarios**:

1. **Given** kameas-ml is running in cloud mode, **When** `GET /health` is called, **Then** the response includes `mode: "cloud"` and does not reference SQLite or poller state.
2. **Given** kameas-ml is running in cloud mode with models loaded for 3 tenants, **When** `GET /status` is called, **Then** the response lists the loaded tenants and their model versions.

---

### Edge Cases

- What happens when the service starts in cloud mode but no model storage backend is configured?
- How does the service behave when a prediction request arrives for a model type that hasn't been trained for that tenant?
- What happens when model weights in the storage backend are corrupted or unparseable?
- How does the service handle concurrent requests for the same tenant while model weights are being reloaded?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept a `--mode` CLI flag with values `local` (default) and `cloud`.
- **FR-002**: In cloud mode, the system MUST NOT start the EventPoller or open any SQLite connections.
- **FR-003**: In cloud mode, the system MUST serve all existing `/predict/*` endpoints, accepting features in the request body and returning predictions.
- **FR-004**: In cloud mode, the system MUST read a tenant identifier from incoming requests (header or token claim) and use it to load tenant-specific model weights.
- **FR-005**: In cloud mode, the system MUST return rule-based fallback predictions when no trained model exists for a tenant or model type.
- **FR-006**: In local mode, the system MUST behave identically to the current implementation with no regressions.
- **FR-007**: The `/health` endpoint MUST reflect the current operating mode and model availability.
- **FR-008**: In cloud mode, the system MUST NOT write to any SQLite tables (`ml_predictions`, `ml_events`, `ml_cursor`).
- **FR-009**: The system MUST support loading model weights from a pluggable storage backend (see feature 003: Model Storage Abstraction).
- **FR-010**: In cloud mode, the system MUST cache loaded model weights in memory with a configurable TTL to avoid repeated storage reads.

### Key Entities

- **ServingMode**: Enum representing `local` or `cloud` operating mode, determines which components are initialized at startup.
- **TenantContext**: Per-request context containing tenant identifier and tier, extracted from the authenticated request by middleware.
- **ModelCache**: In-memory cache of loaded model weights keyed by tenant ID and model name, with TTL-based expiration.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: kameas-ml starts in cloud mode in under 2 seconds with no local dependencies (no SQLite, no local model files required at startup).
- **SC-002**: Prediction latency in cloud mode is within 20% of local mode for the same model and features (excluding model load time).
- **SC-003**: All 5 prediction endpoints (`stuck`, `suggest`, `duration`, `activity`, `quality`) return valid responses in cloud mode.
- **SC-004**: Existing test suite passes without modification when running in local mode after cloud mode changes are merged.
- **SC-005**: The service handles 50 concurrent prediction requests per second per replica without errors.

## Dependencies

- **Feature 002 (Storage Abstraction)**: Cloud mode needs the DataStore interface to avoid direct SQLite imports. However, cloud serving mode can initially accept features directly in request payloads (stateless), deferring full DataStore integration.
- **Feature 003 (Model Storage Abstraction)**: Cloud mode needs to load model weights from S3. Can be stubbed initially with local filesystem for testing.
- **External: API Gateway**: Tenant identification depends on an upstream API gateway setting headers. For initial development, a simple header-based tenant ID is sufficient.
