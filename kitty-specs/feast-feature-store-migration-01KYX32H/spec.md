# Feature Specification: Feast Feature Store Migration

**Feature Branch**: `feast-feature-store-migration-01KYX32H`
**Created**: 2026-07-31
**Status**: Draft
**Input**: Migrate feature management to Feast in both deployments. Feature definitions are authored once as Feast definitions, bundled into the open-source package, and shipped. The local install runs a self-contained Feast — file registry plus SQLite online store — with no network path to the cloud. Cloud runs the same definitions over PostgreSQL for offline retrieval, materialization, and online serving.

## Context

Feature definitions today are Python functions in `src/sigil_ml/features.py`, computed on demand in both deployments. There is no registry, no lineage, no versioning, and no offline store — training sets are rebuilt by replaying extractors over history rather than retrieved point-in-time-correct from stored values. The `feature-extraction-correctness` mission made that replay correct, but it remains a replay.

Feast makes the definitions a registered, versioned artifact; gives cloud training point-in-time-correct retrieval; and gives both deployments one authority for what a feature *is*.

**A previous attempt stalled.** Branch `feat/feast-feature-store` (four commits, tip 2026-04-10, never merged) wired only the push path against a SQLite online store, with placeholder `FileSource` batch sources pointing at parquet files that never existed. Its own docstring recorded the mismatch. With no offline half, Feast reduced to a TTL'd key-value cache in front of functions that already existed, and the final commit demoted it from a required dependency to an extra. This mission exists to do the offline half properly and to make the local half self-contained rather than vestigial.

**Measured cost, accepted.** `feast[sqlite]` adds **+309 MB and +338 native libraries** over the current local dependency set (137 MB / 199 libs → 446 MB / 537 libs). Heaviest contributors: `pyarrow` 123 MB, `pandas` 41 MB, and `mypy`/`mypyc` at 55 MB as runtime dependencies. Every one of those native libraries must be signed and stapled for macOS notarization. This cost was measured and accepted as a condition of the migration; it is not an open question, but it is what sizes the packaging work.

**Feast 0.65.0 carries no telemetry module.** The anonymous usage collection present in older 0.x releases is absent, and core contains no phone-home endpoints. The no-egress requirement is therefore satisfiable without patching the library — but it must be verified structurally rather than assumed.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - One definition, shipped (Priority: P1)

A feature is defined once, as a Feast definition, in the open-source package. That definition is what the local install uses and what cloud training uses. Changing what a feature means is a change to one file, reviewed once, shipped to both.

**Why this priority**: This is the migration's entire purpose. Two implementations reconciled by hand is the state being replaced; a registry that only one deployment consults would recreate it.

**Independent Test**: Alter one feature definition. Assert the change is visible in both the local registry and the cloud registry with no second edit, and that both produce the changed value for identical input.

**Acceptance Scenarios**:

1. **Given** a feature defined in the shipped definitions, **When** the local install resolves it, **Then** it uses the shipped definition rather than a separate local copy.
2. **Given** the same definition, **When** cloud training resolves it, **Then** it produces the same value for identical input.
3. **Given** a definition is changed, **When** both deployments are rebuilt, **Then** neither carries a stale copy of the old definition.
4. **Given** the registry is inspected, **When** a feature is looked up, **Then** its entity, source, type, and owning feature view are discoverable.

---

### User Story 2 - The local install never contacts the cloud (Priority: P1)

A developer runs the open-source product on their laptop. Feature resolution, materialization, and serving all happen on the machine, against a local registry and a local SQLite online store. Nothing is sent anywhere, and there is no configuration that would cause the local install to reach a cloud feature store.

**Why this priority**: This is a stated hard requirement and the product's central promise. Feast is a client-server-capable system, and its default configuration surface includes remote registries and remote online stores — so the absence of a cloud path must be enforced and proven, not merely left unconfigured.

**Independent Test**: Run the full local flow — apply, materialize, serve — with all outbound network blocked. Assert every operation succeeds and no socket is opened.

**Acceptance Scenarios**:

1. **Given** the local deployment, **When** any feature operation runs, **Then** no outbound network connection is attempted.
2. **Given** the local configuration, **When** it is inspected, **Then** it references no remote registry, no remote online store, and no remote provider.
3. **Given** outbound network is unavailable, **When** the local install starts and serves predictions, **Then** behaviour is unchanged.
4. **Given** cloud connection settings exist for the cloud deployment, **When** the local package is built, **Then** those settings are unreachable from the local configuration path.

---

### User Story 3 - Cloud training retrieves point-in-time-correct data (Priority: P1)

Cloud training builds a training set by asking for features as of each example's own moment, retrieved from stored values rather than recomputed by replaying extractors over history.

**Why this priority**: This is what Feast provides that the current code cannot. It is also the half the previous attempt skipped, which is why that attempt delivered no value.

**Independent Test**: Materialize features for a known set of historical tasks, then retrieve a training set with per-example timestamps. Assert each row carries the values that were current at its own timestamp, and that values recorded after a row's timestamp do not appear in it.

**Acceptance Scenarios**:

1. **Given** materialized historical features, **When** a training set is retrieved with per-example event timestamps, **Then** each row reflects the state as of its own timestamp.
2. **Given** a feature value recorded after an example's timestamp, **When** that example is retrieved, **Then** the later value does not appear.
3. **Given** a training retrieval, **When** it completes, **Then** the feature set consumed is identifiable and versioned.
4. **Given** the offline store is unavailable, **When** training runs, **Then** it fails with a clear diagnostic rather than silently training on partial data.

---

### User Story 4 - The frozen binary still ships (Priority: P1)

The open-source product continues to ship as a frozen, notarized, single-directory binary. It builds in CI, passes notarization, starts, and serves.

**Why this priority**: Feast adds 338 native libraries to an artifact that must be signed and stapled. If this fails, the migration cannot ship to open-source users at all — making it the highest-risk work in the mission despite being packaging rather than logic.

**Independent Test**: Build the frozen binary with Feast included, notarize it, install from the artifact on a clean machine, and run the full local flow.

**Acceptance Scenarios**:

1. **Given** the packaging configuration, **When** the frozen binary is built, **Then** the build succeeds with all Feast dependencies resolved, including dynamically-imported providers and stores.
2. **Given** a built binary, **When** it is notarized, **Then** notarization succeeds with every bundled native library signed and stapled.
3. **Given** a notarized binary on a clean machine, **When** it starts, **Then** it serves predictions using the bundled definitions.
4. **Given** the binary's application directory is read-only, **When** feature operations run, **Then** no write to that directory is attempted.

---

### User Story 5 - Serving stays responsive on active work (Priority: P2)

Predictions for the task a developer is working on right now reflect what is happening right now — not a value materialized minutes ago.

**Why this priority**: The models predict on *active* tasks. An online store returns the last materialized value, which for a "is this developer stuck right now" prediction is a downgrade rather than an optimization. This tension must be resolved deliberately rather than inherited from Feast's defaults.

**Independent Test**: Generate live activity on an active task, request a prediction, and assert the features consumed reflect events from the current moment rather than a stale materialized snapshot.

**Acceptance Scenarios**:

1. **Given** an active task with recent events, **When** a prediction is requested, **Then** the features consumed reflect those recent events.
2. **Given** a prediction request, **When** feature resolution runs, **Then** added latency stays within the stated budget.
3. **Given** the online store holds a stale value for an active task, **When** a prediction is requested, **Then** the stale value does not silently supersede current state.

---

### User Story 6 - Both deployments agree (Priority: P2)

The same task data produces the same feature values in the local install and in cloud.

**Why this priority**: Base models are trained centrally and extended locally, so a divergence between deployments is a correctness bug in the model pipeline, not merely an inconsistency.

**Independent Test**: Feed identical task and event data through both deployments and compare resulting vectors exactly.

**Acceptance Scenarios**:

1. **Given** identical input, **When** features are resolved in each deployment, **Then** the vectors are identical.
2. **Given** a feature definition change, **When** both deployments are rebuilt, **Then** they remain identical.

---

### Edge Cases

- **Read-only application directory.** The registry ships inside a notarized bundle; nothing may attempt to write it at serving time. Any operation requiring a writable registry must target the user data directory.
- **First run with an empty online store.** Must serve without error before anything has been materialized.
- **Registry produced by a different Feast version than the runtime.** Must fail with a clear diagnostic rather than a deserialization traceback.
- **Import cost in short-lived commands.** Feast's import pulls `pyarrow` and `grpcio`; CLI paths that never touch features should not pay it.
- **Offline store unreachable in cloud.** Fail loudly; never train on partial retrieval.
- **Event timestamps.** The reference-time semantics established by `feature-extraction-correctness` must map onto Feast's event-timestamp model without changing feature values.
- **Entity with no materialized row.** Serving must degrade predictably.
- **Concurrent materialization and serving** against the same SQLite online store, given WAL-mode constraints already governing the shared database.

## Requirements *(mandatory)*

### Functional Requirements

| ID | Requirement | Status |
|---|---|---|
| FR-001 | Feature definitions MUST be authored as Feast definitions and bundled into the open-source package as shipped artifacts. | Draft |
| FR-002 | Both deployments MUST resolve features from those shipped definitions; neither may carry a separate copy. | Draft |
| FR-003 | The local deployment MUST run a self-contained Feast using a file registry and a SQLite online store. | Draft |
| FR-004 | The local deployment MUST NOT open any network connection for any feature operation, under any configuration reachable from the local package. | Draft |
| FR-005 | The local configuration MUST NOT reference a remote registry, remote online store, or remote provider. | Draft |
| FR-006 | The cloud deployment MUST use PostgreSQL for both offline retrieval and online serving. | Draft |
| FR-007 | Cloud training MUST retrieve training sets with per-example event timestamps, returning values as of each example's own moment. | Draft |
| FR-008 | Values recorded after an example's event timestamp MUST NOT appear in that example's retrieved features. | Draft |
| FR-009 | Feature values MUST be materialized into the offline store with the event time they describe, never the time they were written. | Draft |
| FR-010 | The feature set consumed by a training run MUST be identifiable and versioned. | Draft |
| FR-011 | The frozen binary MUST build with all Feast dependencies resolved, including dynamically-imported providers and stores. | Draft |
| FR-012 | The frozen binary MUST pass notarization with every bundled native library signed and stapled. | Draft |
| FR-013 | No feature operation MUST require writing to the application directory. | Draft |
| FR-014 | Serving MUST reflect current state for active tasks rather than a stale materialized value. | Draft |
| FR-015 | Both deployments MUST produce identical feature values for identical input. | Draft |
| FR-016 | A registry incompatible with the running Feast version MUST be refused with a clear diagnostic. | Draft |
| FR-017 | Feature resolution MUST fail loudly when its store is unavailable, never returning partial results. | Draft |
| FR-018 | Existing model names, prediction endpoints, and database table ownership MUST be unchanged by this migration. | Draft |

### Non-Functional Requirements

| ID | Requirement | Threshold | Status |
|---|---|---|---|
| NFR-001 | Bundle growth stays within the measured and accepted budget. | No more than 350 MB over the pre-migration local dependency set | Draft |
| NFR-002 | Serving-path feature resolution latency does not regress materially. | Within 20% of pre-migration median over 1000 resolutions | Draft |
| NFR-003 | Process start-to-serving time stays acceptable for a background daemon. | Under 10 seconds cold start on the frozen binary | Draft |
| NFR-004 | The no-egress guarantee is verified structurally, not by inspection. | Automated test asserting no socket is opened across the full local flow | Draft |
| NFR-005 | Notarization succeeds in CI. | Automated, non-flaky, on every release build | Draft |

### Constraints

| ID | Constraint | Status |
|---|---|---|
| C-001 | The local deployment has no network path to the cloud feature store. This is a hard product requirement, not a default. | Draft |
| C-002 | The open-source product continues to ship as a frozen, notarized, single-directory binary. | Draft |
| C-003 | The local dependency ceiling in `CLAUDE.md` is superseded for this mission by the measured and accepted +309 MB / +338 native library cost; it remains in force for any dependency outside Feast's tree. | Draft |
| C-004 | Database table ownership rules are unchanged; the migration introduces no writes to Go-owned tables. | Draft |
| C-005 | Feature names, ordering, and semantics MUST NOT change as part of this migration. Behaviour-preserving migration only. | Draft |
| C-006 | The reference-time semantics from `feature-extraction-correctness` are preserved; Feast's event-timestamp model must accommodate them rather than replace them. | Draft |
| C-007 | Local online store is SQLite; cloud online store is PostgreSQL. | Draft |

## Key Entities

- **Feature definition**: The authored, shipped declaration of a feature — its entity, source, type, and owning feature view. One artifact, bundled in the package.
- **Registry**: The materialized form of the definitions that a deployment consults at runtime. File-based and read-only locally; the cloud equivalent serves the same role.
- **Online store**: Latest-value-per-entity storage for serving. SQLite locally, PostgreSQL in cloud.
- **Offline store**: Historical values with event timestamps, used for point-in-time-correct training retrieval. Cloud only.
- **Materialization**: The operation that moves computed feature values into a store, stamped with the event time they describe.
- **Feature service**: The named, versioned bundle of features a given model consumes.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A feature definition change requires exactly one edit and is reflected in both deployments.
- **SC-002**: With all outbound network blocked, the complete local flow succeeds and no socket is opened.
- **SC-003**: A training set retrieved with per-example timestamps carries values as of each example's own moment, with no leakage from later values.
- **SC-004**: The frozen binary builds, notarizes, and serves predictions from the bundled definitions on a clean machine.
- **SC-005**: Local and cloud produce identical feature vectors for identical input.
- **SC-006**: Feature names, ordering, and values are unchanged from pre-migration for the same input.
- **SC-007**: Bundle growth is within the accepted budget and recorded in the build output.

## Assumptions

- The measured +309 MB / +338 native library cost was accepted as a condition of this migration. It is treated as a design input, and NFR-001 exists to prevent it drifting further rather than to relitigate it.
- Feast 0.65.0 contains no telemetry or phone-home behaviour, verified against the installed package. NFR-004 verifies this structurally rather than relying on it.
- Feature computation remains in Python. Feast provides the registry, materialization, storage, and retrieval; it is not adopted as a transformation engine. Window aggregations over variable-length event sequences do not map cleanly onto request-time schema models.
- The registry ships pre-applied inside the bundle so no write is needed at serving time; the online store lives in the user data directory, which is writable.
- Cloud materialization runs as a scheduled or triggered job, not synchronously within a training run.

## Out of Scope

- Changing what any feature computes. This migration is behaviour-preserving; the event-vocabulary problems already identified are a separate mission.
- MLflow, the model registry, and base-model shipping — covered by `model-registry-and-base-refresh`.
- Retiring `sigil_ml.features`. Computation stays in Python; only registration, storage, and retrieval move.
- Feature selection.
- Adding cloud connectivity of any kind to the local deployment.
- Migrating the Go daemon or changing the shared database schema.
