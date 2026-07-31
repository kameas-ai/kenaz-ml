# Feature Specification: Feature Extraction Correctness

**Feature Branch**: `feature-extraction-correctness-01KYTR7N`
**Created**: 2026-07-31
**Status**: Draft
**Input**: Make feature computation point-in-time correct and collapse the duplicate extractor definitions in `src/sigil_ml/features.py`. Elapsed-time features are currently computed against wall clock, which is correct when serving an active task but wrong when replaying over completed tasks during training. Separately, two parallel extractor families promise identical output but are reconciled only by hand.

## Context

Two defects in the same module, fixed together because the second one determines where the first one gets fixed.

**Point-in-time.** The stuck extractor computes elapsed features against `time.time()` (`features.py:93`, `:105`, `:125`). At serving time the subject is an *active* task (`poller.py:141`, `routes.py:381`) and "now" is genuinely now. At training time the same function is replayed over *completed* tasks (`trainer.py:100`, `cloud_trainer.py:262`), so for a task that finished months ago `time_in_phase_sec` and `time_since_last_commit_sec` measure task age rather than anything about the task. The downstream label at `trainer.py:104` inherits it: `time_in_phase_sec > 600` is true for every task older than ten minutes, collapsing the heuristic to `test_failure_count > 3`.

**Duplication.** `extract_stuck_features(store, task_id)` and `extract_stuck_features_from_data(task, events)` are parallel implementations whose docstrings promise identical output. Nothing enforces it. The same holds for the duration pair.

This blocks base-model work. Base models are trained centrally and shipped to open-source installs, which extend them locally on user data. A base model trained on the current features bakes task-age into an artifact distributed to every user, which is then extended on top of. See `docs/ML_ARCHITECTURE.md` §3.1–3.2.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Training examples describe the moment they occurred (Priority: P1)

A model is trained from a user's completed task history. For each historical task, the features describe the state of that task *at the time it was active* — how long the developer had been in that phase then, how long since their last commit then. A task that completed in March and a task that completed yesterday both contribute honest examples; neither is distorted by how long ago it happened.

**Why this priority**: This is the defect. Without it, two of six stuck features encode recency instead of behavior, and every model trained from history — local, cloud, and base — learns from corrupted vectors.

**Independent Test**: Take a completed task with a known timeline. Compute its features. Compute them again with a simulated clock set weeks later. The two vectors must be identical.

**Acceptance Scenarios**:

1. **Given** a task that completed at time T, **When** features are extracted for training, **Then** elapsed-time features are measured relative to T, not to the current time.
2. **Given** the same completed task, **When** features are extracted on two different days, **Then** both extractions produce identical vectors.
3. **Given** a task whose events include entries after the reference time, **When** features are extracted as of that reference time, **Then** the later events do not contribute to any feature value.
4. **Given** a reference time earlier than a task's phase start, **When** features are extracted, **Then** elapsed durations are zero rather than negative.

---

### User Story 2 - Live predictions are unaffected (Priority: P1)

A developer is working right now. The poller and the HTTP endpoints predict against their active task, and "how long have you been in this phase" means how long as of this moment. That behavior does not change.

**Why this priority**: The serving path is currently correct. A fix aimed at training must not regress the path that already works, and the two paths share one function.

**Independent Test**: Run the serving path against an active task before and after the change with the clock held fixed. Outputs must be identical.

**Acceptance Scenarios**:

1. **Given** an active task, **When** the poller extracts features, **Then** elapsed features are measured against the current time.
2. **Given** an active task, **When** a prediction endpoint extracts features, **Then** the resulting vector matches the pre-change behavior exactly.
3. **Given** no reference time is supplied, **When** any extractor runs, **Then** it behaves as it does today.

---

### User Story 3 - One definition per feature (Priority: P1)

There is exactly one implementation of each feature's arithmetic. The variant that takes a data store fetches rows and delegates; it does not reimplement. A change to a feature's definition cannot land in one path and miss the other.

**Why this priority**: Two implementations of the same feature is training/serving skew waiting to happen, and it becomes a correctness bug rather than a maintenance annoyance once base models are trained against one path and served through the other.

**Independent Test**: For the same task, call both the store-backed and data-backed extractor with the same reference time. Assert exact equality across every key.

**Acceptance Scenarios**:

1. **Given** a task in the store, **When** both extractor forms are called with the same reference time, **Then** they return identical dictionaries.
2. **Given** a task that does not exist, **When** the store-backed extractor is called, **Then** it returns the documented empty-feature dictionary without raising.
3. **Given** a change to a feature's arithmetic, **When** the change is made in one place, **Then** both call paths reflect it with no second edit.

---

### User Story 4 - Local and cloud agree (Priority: P2)

The same task data produces the same feature vector whether it is read from SQLite on a laptop or Postgres in a container. Base models trained centrally are therefore valid for local extension.

**Why this priority**: Restates the existing principle that identical logic runs in both deployments. It is a consequence of Story 3 rather than separate work, but it needs its own verification because it is the assumption base-model shipping rests on.

**Independent Test**: Feed identical task and event payloads through both store implementations and compare the resulting vectors.

**Acceptance Scenarios**:

1. **Given** identical task and event data, **When** features are extracted via each store implementation, **Then** the vectors are identical.
2. **Given** the cloud trainer processes a completed task, **When** it extracts features, **Then** it supplies the same reference time the local trainer would.

---

### Edge Cases

- Reference time earlier than the task's `started_at` or phase start → durations clamp to zero, never negative.
- Task with `completed_at` unset → trainer must resolve a reference time or skip the example rather than silently fall back to wall clock.
- Task whose event list is empty → existing empty-feature behavior preserved.
- Task with no `phase_change` event → phase start falls back to `started_at`, as today.
- Events with timestamps after the reference time → excluded from all aggregations.
- Reference time exactly equal to an event timestamp → event is included (inclusive boundary).

## Requirements *(mandatory)*

### Functional Requirements

| ID | Requirement | Status |
|---|---|---|
| FR-001 | Every extractor in `features.py` MUST accept an explicit reference-time parameter; omitting it MUST mean "current wall clock". | Draft |
| FR-002 | All elapsed-time features MUST be computed relative to the supplied reference time rather than to `time.time()`. | Draft |
| FR-003 | Extractors MUST NOT incorporate events whose timestamp is later than the reference time. | Draft |
| FR-004 | Elapsed durations MUST be clamped at zero when the reference time precedes the point being measured from. | Draft |
| FR-005 | The data-backed extractor functions MUST be the single definition of each feature's arithmetic. | Draft |
| FR-006 | The store-backed extractor functions MUST fetch rows and delegate to the data-backed functions without reimplementing any feature. | Draft |
| FR-007 | The local trainer MUST supply each completed task's own reference time when extracting training features. | Draft |
| FR-008 | The cloud trainer MUST supply each completed task's own reference time when extracting training features. | Draft |
| FR-009 | Serving call sites (poller and prediction endpoints) MUST continue to extract against current time, with output identical to pre-change behavior. | Draft |
| FR-010 | The event-buffer extractor MUST accept the same reference-time parameter as the task-based extractors. | Draft |
| FR-011 | Feature names, ordering, and count MUST remain unchanged by this feature. | Draft |
| FR-012 | A task that cannot yield a valid reference time MUST be excluded from training rather than extracted against wall clock. | Draft |

### Non-Functional Requirements

| ID | Requirement | Threshold | Status |
|---|---|---|---|
| NFR-001 | No new runtime dependencies are introduced. | Zero additions to the local dependency set | Draft |
| NFR-002 | Serving-path feature extraction latency does not regress. | Within 5% of pre-change median, measured over 1000 extractions | Draft |
| NFR-003 | Point-in-time behavior is covered by automated tests. | At least one test per acceptance scenario in Stories 1–3 | Draft |
| NFR-004 | Path-equivalence is covered by automated tests. | Both extractor forms compared across all extractor families | Draft |

### Constraints

| ID | Constraint | Status |
|---|---|---|
| C-001 | The local runtime dependency set stays limited to `scikit-learn`, `numpy`, `fastapi`, `uvicorn`, `joblib` per `CLAUDE.md`. | Draft |
| C-002 | Database table ownership rules are unchanged; this feature reads only. | Draft |
| C-003 | Feature contract versioning and model invalidation are out of scope, deferred to the model-registry feature. | Draft |
| C-004 | Behavior must be identical across `SqliteStore` and `PostgresStore`. | Draft |
| C-005 | Existing feature names and vector ordering are frozen for this feature; changing them requires the contract work that does not yet exist. | Draft |

## Key Entities

- **Reference time**: The instant a feature vector describes. Current time when serving an active task; the task's completion or window-end time when replaying history.
- **Data-backed extractor**: The authoritative implementation, operating on an already-fetched task record and event list.
- **Store-backed extractor**: A convenience wrapper that fetches through the `DataStore` protocol and delegates.
- **Elapsed-time feature**: Any feature expressed as a duration between the reference time and an earlier event — currently `time_in_phase_sec` and `time_since_last_commit_sec`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Extracting features for a fixed historical task yields byte-identical vectors regardless of when the extraction is performed.
- **SC-002**: For a completed task, elapsed-time features match the values that would have been observed at that task's completion, within one second.
- **SC-003**: Predictions produced for active tasks are identical before and after the change when the clock is held fixed — zero regression on the serving path.
- **SC-004**: Both extractor forms return identical results for every extractor family, verified by automated test.
- **SC-005**: No feature name, ordering, or count changes.
- **SC-006**: The dependency set is unchanged.

## Assumptions

- A completed task's `completed_at` is the appropriate reference time for training examples; where it is absent, `last_active` is the fallback and a task with neither is skipped.
- Event timestamps are trustworthy and comparable to task timestamps (both epoch milliseconds), consistent with existing extractor behavior.
- Excluding post-reference events changes values for some historical tasks; this is the intended correction, not a regression.
- Models currently persisted on disk were fitted on the old vectors and become invalid when this lands. Invalidation is handled by the model-registry feature; until then, existing artifacts are expected to be retrained by the normal training cadence.
- The stuck label threshold at `trainer.py:104` becomes meaningful once the feature is correct, but remains uncalibrated. Validating it is explicitly deferred.

## Out of Scope

- Feature contract versioning, manifest schema, and load-time validation.
- Invalidating or force-retraining existing model artifacts.
- Re-deriving or retuning the stuck label heuristic.
- Materializing features to storage, and any feature-store tooling.
- Changes to feature definitions themselves — only when they are evaluated, not what they compute.
