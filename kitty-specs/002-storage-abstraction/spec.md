# Feature Specification: Storage Abstraction

**Feature Branch**: `002-storage-abstraction`
**Created**: 2026-03-25
**Status**: Draft
**Input**: Introduce a DataStore protocol that abstracts kenaz-ml's data access layer, replacing direct SQLite calls in the poller, schema, and app modules with a pluggable interface. Provide a SQLite implementation (preserving current behavior) and a Postgres implementation (for cloud/K8s deployment).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Local Mode Uses SQLite Backend Transparently (Priority: P1)

A free-tier developer runs kenaz-ml locally. The system uses the SQLite backend by default. All existing behavior — polling events, reading tasks, writing predictions, managing the cursor — works identically to today. The developer notices no change.

**Why this priority**: The abstraction must not break the existing local experience. This validates that the refactor is safe.

**Independent Test**: Run the existing test suite after the refactor. All tests pass without modification. Start the server, verify predictions appear in SQLite.

**Acceptance Scenarios**:

1. **Given** kenaz-ml is started in local mode, **When** the DataStore is initialized, **Then** a SQLite backend is created pointing to the configured database path with WAL mode and busy_timeout=5000.
2. **Given** the SQLite backend is active, **When** events are queried since a cursor position, **Then** the results are identical to the current direct SQLite queries.
3. **Given** the SQLite backend is active, **When** a prediction is inserted, **Then** it appears in the `ml_predictions` table with the correct model, result, confidence, and expiry.

---

### User Story 2 - Cloud Mode Uses Postgres Backend (Priority: P1)

A cloud operator deploys kenaz-ml in Kubernetes. The system initializes a Postgres backend using a connection URL from configuration. The Postgres backend supports the same data operations as SQLite — reading events/tasks and writing predictions — but against per-tenant schemas in a shared Postgres cluster.

**Why this priority**: This is the primary motivator for the abstraction. Without a Postgres backend, cloud deployment is not possible.

**Independent Test**: Configure kenaz-ml with a Postgres connection URL, run prediction operations, verify data is read from and written to Postgres.

**Acceptance Scenarios**:

1. **Given** kenaz-ml is started in cloud mode with a Postgres connection URL, **When** the DataStore is initialized, **Then** a Postgres backend is created using the provided connection URL.
2. **Given** the Postgres backend is active, **When** events are queried for a tenant, **Then** only that tenant's events are returned.
3. **Given** the Postgres backend is active, **When** a prediction is inserted, **Then** it is written to the correct tenant's schema with the same fields as the SQLite implementation.

---

### User Story 3 - Poller and Routes Use DataStore Interface (Priority: P1)

The EventPoller, training modules, and API routes interact only with the DataStore interface — never with SQLite or Postgres directly. Swapping backends requires no changes to business logic.

**Why this priority**: This is the architectural guarantee that makes the abstraction valuable. If any component bypasses the interface, the abstraction is incomplete.

**Independent Test**: Provide a mock DataStore implementation in tests. Verify the poller, trainer, and routes operate correctly against the mock without any real database.

**Acceptance Scenarios**:

1. **Given** a mock DataStore implementation, **When** the EventPoller runs a poll cycle, **Then** it calls the DataStore interface methods and processes the returned events correctly.
2. **Given** a mock DataStore implementation, **When** a `/predict/stuck` request is handled, **Then** the route handler uses only DataStore methods for any data access.
3. **Given** the DataStore interface, **When** a new backend is needed in the future, **Then** it can be implemented by satisfying the interface contract without modifying any existing business logic.

---

### Edge Cases

- What happens when the Postgres connection drops mid-operation? The system should handle transient connection failures gracefully with retries.
- What happens when a tenant's schema doesn't exist yet in Postgres? The system should create it on first access or return a clear error.
- How does the SQLite backend handle the case where the database file doesn't exist yet? It should create it (matching current behavior via `schema.ensure_ml_tables()`).
- What happens when a query returns no results (e.g., no events since cursor)? Both backends should return empty collections, not errors.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST define a DataStore protocol (interface) with methods for all data operations currently performed directly against SQLite.
- **FR-002**: The DataStore protocol MUST include operations for: querying events since a cursor, querying the active task, querying completed tasks, inserting predictions, inserting ML audit events, and managing the poll cursor.
- **FR-003**: System MUST provide a SQLite implementation of the DataStore protocol that preserves all current SQLite behavior including WAL mode and busy_timeout settings.
- **FR-004**: System MUST provide a Postgres implementation of the DataStore protocol that supports per-tenant data isolation.
- **FR-005**: The EventPoller MUST be refactored to use the DataStore protocol instead of direct SQLite access.
- **FR-006**: The API route handlers MUST be refactored to use the DataStore protocol for any data access.
- **FR-007**: The training modules MUST be refactored to use the DataStore protocol for reading training data and writing audit events.
- **FR-008**: The app startup sequence MUST select the appropriate DataStore backend based on the operating mode (local → SQLite, cloud → Postgres).
- **FR-009**: The Postgres backend MUST accept a connection URL via environment variable or configuration.
- **FR-010**: The SQLite backend MUST enforce the existing invariants: `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000` on every connection.
- **FR-011**: Both backends MUST return data in the same format so that consumers (poller, routes, trainer) are backend-agnostic.
- **FR-012**: The Postgres implementation MUST respect table ownership rules: Python only writes to `ml_predictions`, `ml_events`, and `ml_cursor`; it only reads from `events`, `tasks`, `patterns`, and `suggestions`.

### Key Entities

- **DataStore**: The protocol (interface) defining all data access operations. This is the central abstraction that all components depend on.
- **SqliteStore**: Implementation of DataStore backed by a local SQLite file. Preserves all current behavior.
- **PostgresStore**: Implementation of DataStore backed by a Postgres database. Supports per-tenant schemas and cloud deployment.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All existing tests pass without modification after the refactor (zero regressions).
- **SC-002**: No module outside of the DataStore implementations imports `sqlite3` directly.
- **SC-003**: The poller, routes, and training modules can operate against a mock DataStore in tests without any real database.
- **SC-004**: Switching from SQLite to Postgres requires only a configuration change, not a code change.
- **SC-005**: Both backends produce identical results for the same input data (verified by integration tests against both).

## Dependencies

- **Feature 001 (Cloud Serving Mode)**: Cloud mode determines when the Postgres backend is selected. However, the abstraction can be built and tested independently using local mode + SQLite backend.
- **External: Postgres availability**: Integration testing of the Postgres backend requires a Postgres instance. A containerized test instance is sufficient.
