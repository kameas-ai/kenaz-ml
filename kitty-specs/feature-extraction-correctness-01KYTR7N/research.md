# Phase 0 Research: Feature Extraction Correctness

**Date**: 2026-07-31 | **Plan**: [plan.md](./plan.md)

All open decisions resolved. No `[NEEDS CLARIFICATION]` markers remain.

---

## D-001: Placement of the no-lookahead filter

**Decision**: Filter `ts > as_of_ms` in Python, inside the data-backed extractors, immediately after the event list is received.

**Rationale**: The alternative pushes an `until` bound into `get_events_for_task()` on the `DataStore` protocol, which forces parallel changes in `store_sqlite.py` and `store_postgres.py` plus their tests. That widens the change surface for a correctness fix and converts C-004 (identical behavior across both stores) from a structural property into something requiring verification across two SQL dialects. Filtering once in Python means one implementation of the guarantee. The data volume is a single task's event window, so the discarded-I/O cost is immaterial.

**Alternatives considered**:
- *Push down into the query* — more efficient in principle, rejected for blast radius and the parity risk described above. Worth revisiting when cloud feature materialization needs bulk extraction over many tasks, where the I/O ratio changes.
- *Filter at the call site* — rejected: it would place the correctness guarantee in four places instead of one, and any future caller would have to remember it.

---

## D-002: Shape of the reference-time parameter

**Decision**: `as_of_ms: int | None = None`, keyword-only, on every extractor. `None` means current wall clock.

**Rationale**: Every existing call site remains valid unchanged, which makes FR-009 (no serving regression) true by construction rather than by inspection. Keyword-only prevents positional misuse against the existing `(store, task_id)` and `(task, events)` signatures. Epoch milliseconds matches the units already used throughout the extractors and the `events`/`tasks` schemas, avoiding a conversion boundary.

**Alternatives considered**:
- *Required parameter* — would force every call site to be audited, which has some appeal for correctness, but breaks the public extractor API for no behavioral gain and makes the serving path noisier.
- *A `Clock` object injected into the extractor* — more testable in the abstract, rejected as over-engineering for two elapsed features; a plain timestamp is directly what the arithmetic needs.
- *`datetime` instead of epoch ms* — rejected; would require conversion at every boundary and diverge from stored timestamp units.

---

## D-003: Reference-time resolution for training examples

**Decision**: `completed_at` → `last_active` → skip the example.

**Rationale**: The reference time must be the moment the example describes. For a completed task that is its completion. `last_active` is the closest honest fallback when `completed_at` is unset. A task with neither cannot produce an honest vector, and silently substituting wall clock would reintroduce exactly the defect this feature exists to remove — for the malformed rows most likely to be pathological. Dropping the example loses one training row; keeping it corrupts the model.

**Alternatives considered**:
- *Fall back to wall clock* — rejected outright; it is the bug.
- *Fall back to the last event timestamp* — plausible and close to correct, but it silently redefines what the row means depending on data completeness. Skipping is more honest and the row count impact is expected to be negligible.
- *Fail the training run* — rejected as too brittle; one malformed historical row should not block training.

---

## D-004: Negative elapsed durations

**Decision**: Clamp at zero via a shared `_elapsed_sec()` helper.

**Rationale**: A reference time earlier than the measured-from point is reachable in real data — clock skew across event sources, out-of-order ingestion, or a `phase_change` event recorded after `completed_at`. A negative duration has no meaning to any of the consuming models, and raising would make training fragile against a condition the extractor can absorb sensibly. Routing every elapsed computation through one helper makes FR-004 hold in a single place rather than at each subtraction site.

**Alternatives considered**:
- *Raise* — rejected; converts a data-quality wrinkle into a training outage.
- *Leave negative* — rejected; silently feeds nonsense into the model, which is the class of failure this feature exists to eliminate.
- *Treat as missing / NaN* — rejected; the downstream vector builder assumes floats, and introducing NaN would require imputation logic that does not exist and is out of scope.

---

## D-005: No API contracts generated

**Decision**: Skip the `contracts/` directory.

**Rationale**: The feature changes no external interface. `as_of_ms` is internal; HTTP request and response shapes served from `routes.py` are untouched, and the `:7774` endpoint stability requirement in `CLAUDE.md` is satisfied trivially. Generating empty or restated contracts would add noise without adding a check.

---

## Prior art consulted

- `docs/ML_ARCHITECTURE.md` §3.1–3.2 — established the defect analysis and the `as_of_ms` approach that this feature implements.
- `src/sigil_ml/models/activity.py` — the `partial_fit` incremental path, reviewed to confirm no extractor coupling that would be disturbed by signature changes.
- `kitty-specs/002-storage-abstraction/` — the `DataStore` protocol boundary, reviewed to confirm D-001 keeps the protocol frozen and inherits its existing parity guarantees.
