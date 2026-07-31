# Tasks: Feature Extraction Correctness

**Mission**: `feature-extraction-correctness-01KYTR7N`
**Branch**: `main` (planning base and merge target)
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Research**: [research.md](./research.md)
**Generated**: 2026-07-31

## Overview

24 subtasks across 4 work packages. WP01 is the foundation — it holds all feature arithmetic. WP02, WP03, and WP04 each depend on WP01 and are mutually independent.

Note on execution: `finalize-tasks` collapsed all four into a single lane (`lane-a`), because the lane algorithm merges on dependency edges and all three downstream WPs depend on WP01. They therefore execute sequentially in one worktree rather than in parallel worktrees. The mutual independence of WP02–WP04 still means they can be implemented and reviewed in any order once WP01 lands.

File ownership is disjoint by construction: WP01 owns `features.py`, WP02 owns the extractor test module, WP03 owns the two training modules and their test, WP04 owns the two serving modules and their test.

## Subtask Index

> **On the `[P]` column.** It marks *logical* independence — the subtask does not depend on its siblings' output. It does **not** mean two agents may run concurrently, which is bounded by file ownership. Most subtasks within a WP edit the same file and must be done sequentially by one agent. The only genuinely concurrent-safe pairs here are **T021 ∥ T022** (`poller.py` vs `routes.py`) and **T019** against T017/T018 (`cloud_trainer.py` vs `trainer.py`). Concurrency between *work packages* is the real lever — see Dependencies below.

| ID | Description | WP | Parallel |
|---|---|---|---|
| T001 | Add `_elapsed_sec()` clamp helper | WP01 | | [D] |
| T002 | Add `_resolve_now_ms()` and `_events_at_or_before()` helpers | WP01 | | [D] |
| T003 | Thread `as_of_ms` through `extract_stuck_features_from_data` | WP01 | | [D] |
| T004 | Thread `as_of_ms` through `extract_duration_features_from_data` | WP01 | [P] |
| T005 | Thread `as_of_ms` through `extract_features_from_buffer` | WP01 | [P] |
| T006 | Extract `_empty_stuck_features()` / `_empty_duration_features()` | WP01 | |
| T007 | Reduce `extract_stuck_features` to fetch-and-delegate | WP01 | |
| T008 | Reduce `extract_duration_features` to fetch-and-delegate | WP01 | [P] |
| T009 | Frozen-clock and fake-store test fixtures | WP02 | |
| T010 | Determinism tests | WP02 | |
| T011 | No-lookahead and inclusive-boundary tests | WP02 | [P] |
| T012 | Negative-clamp tests | WP02 | [P] |
| T013 | Path-equivalence tests | WP02 | [P] |
| T014 | Vector-layout-frozen test | WP02 | [P] |
| T015 | Edge-case tests | WP02 | [P] |
| T016 | Add `_reference_time_for()` resolver | WP03 | |
| T017 | `trainer.py` stuck path passes reference time | WP03 | |
| T018 | `trainer.py` duration path passes reference time | WP03 | [P] |
| T019 | `cloud_trainer.py` stuck and duration paths | WP03 | [P] |
| T020 | Training reference-time and skip tests | WP03 | |
| T021 | Audit `poller.py` call sites | WP04 | [P] |
| T022 | Audit `routes.py` call sites | WP04 | [P] |
| T023 | Capture pre-change baseline vectors | WP04 | |
| T024 | Serving non-regression tests | WP04 | |

---

## WP01 — Point-in-Time Extractor Core

**Prompt**: [tasks/WP01-point-in-time-extractor-core.md](./tasks/WP01-point-in-time-extractor-core.md)
**Priority**: P1 · **Dependencies**: none · **Estimated prompt size**: ~330 lines

**Goal**: Make `src/sigil_ml/features.py` the single, point-in-time-correct definition of every feature. Add the reference-time parameter, the no-lookahead filter, and the zero clamp; collapse the store-backed extractors into thin delegates.

**Independent test**: Extract features for a fixed task twice under two different simulated wall clocks with the same `as_of_ms`; assert identical vectors. Assert both extractor forms agree.

**Included subtasks**:

- [x] T001 Add `_elapsed_sec()` clamp helper (WP01)
- [x] T002 Add `_resolve_now_ms()` and `_events_at_or_before()` helpers (WP01)
- [x] T003 Thread `as_of_ms` through `extract_stuck_features_from_data` (WP01)
- [ ] T004 Thread `as_of_ms` through `extract_duration_features_from_data` (WP01)
- [ ] T005 Thread `as_of_ms` through `extract_features_from_buffer` (WP01)
- [ ] T006 Extract `_empty_stuck_features()` / `_empty_duration_features()` (WP01)
- [ ] T007 Reduce `extract_stuck_features` to fetch-and-delegate (WP01)
- [ ] T008 Reduce `extract_duration_features` to fetch-and-delegate (WP01)

**Implementation sketch**: helpers first (T001–T002), then the data-backed extractors that use them (T003–T005), then the empty-vector helpers (T006) that the delegates need, then the delegates themselves (T007–T008).

**Parallel opportunities**: T004 and T005 are independent of T003 once the helpers exist. T008 is independent of T007.

**Risks**: The empty-vector contract must be preserved byte-for-byte — callers build vectors positionally and a changed default silently corrupts training rows. `time_of_day_hour` is the one non-zero default in the duration family.

**Requirements**: FR-001, FR-002, FR-003, FR-004, FR-005, FR-006, FR-010, FR-011

---

## WP02 — Extractor Correctness Test Suite

**Prompt**: [tasks/WP02-extractor-correctness-tests.md](./tasks/WP02-extractor-correctness-tests.md)
**Priority**: P1 · **Dependencies**: WP01 · **Estimated prompt size**: ~300 lines

**Goal**: Prove the properties WP01 claims — determinism across wall clocks, no lookahead, zero clamping, path equivalence, and frozen vector layout.

**Independent test**: The suite itself. Every test must fail against the pre-WP01 implementation and pass after it.

**Included subtasks**:

- [ ] T009 Frozen-clock and fake-store test fixtures (WP02)
- [ ] T010 Determinism tests (WP02)
- [ ] T011 No-lookahead and inclusive-boundary tests (WP02)
- [ ] T012 Negative-clamp tests (WP02)
- [ ] T013 Path-equivalence tests (WP02)
- [ ] T014 Vector-layout-frozen test (WP02)
- [ ] T015 Edge-case tests (WP02)

**Implementation sketch**: fixtures first (T009), then one test group per property. T010–T015 are independent once fixtures exist.

**Parallel opportunities**: T011–T015 all parallelize after T009 and T010.

**Risks**: A determinism test that passes trivially (both calls made within the same millisecond) proves nothing. The frozen-clock fixture must actually advance wall clock between the two extractions.

**Requirements**: FR-003, FR-004, FR-006, FR-011

---

## WP03 — Training Reference-Time Wiring

**Prompt**: [tasks/WP03-training-reference-time.md](./tasks/WP03-training-reference-time.md)
**Priority**: P1 · **Dependencies**: WP01 · **Estimated prompt size**: ~290 lines

**Goal**: Both trainers resolve each completed task's own reference time and pass it to the extractors, skipping rows that cannot supply one.

**Independent test**: Build a training matrix from a fixture set containing one task with no `completed_at` and no `last_active`; assert that row is absent from the matrix and that the remaining rows carry historically-correct elapsed features.

**Included subtasks**:

- [ ] T016 Add `_reference_time_for()` resolver (WP03)
- [ ] T017 `trainer.py` stuck path passes reference time (WP03)
- [ ] T018 `trainer.py` duration path passes reference time (WP03)
- [ ] T019 `cloud_trainer.py` stuck and duration paths (WP03)
- [ ] T020 Training reference-time and skip tests (WP03)

**Implementation sketch**: shared resolver first (T016), then the four call sites, then tests.

**Parallel opportunities**: T018 and T019 are independent of T017 once T016 exists.

**Risks**: The stuck label at `trainer.py:104` reads `features["time_in_phase_sec"]`, whose meaning changes under this WP. The label expression itself must not be touched — retuning it is explicitly out of scope — but its behavior will shift, and that is expected. Skipping rows changes training-set size; if a fixture asserts an exact sample count, it needs updating.

**Requirements**: FR-007, FR-008, FR-012

---

## WP04 — Serving Non-Regression

**Prompt**: [tasks/WP04-serving-non-regression.md](./tasks/WP04-serving-non-regression.md)
**Priority**: P1 · **Dependencies**: WP01 · **Estimated prompt size**: ~250 lines

**Goal**: Prove the serving path is byte-identical before and after. This is predominantly verification; the expected code delta is zero or near-zero.

**Independent test**: With the clock pinned, predictions for an active task produce the same feature vectors and the same model outputs as the pre-change implementation.

**Included subtasks**:

- [ ] T021 Audit `poller.py` call sites (WP04)
- [ ] T022 Audit `routes.py` call sites (WP04)
- [ ] T023 Capture pre-change baseline vectors (WP04)
- [ ] T024 Serving non-regression tests (WP04)

**Implementation sketch**: capture baselines from the pre-WP01 implementation first (T023 must be done against the original code or from recorded fixtures), then audit, then assert.

**Parallel opportunities**: T021 and T022 are independent.

**Risks**: T023 is order-sensitive — baselines captured after WP01 lands prove nothing. Record them as literal fixture values committed to the test module, derived from the pre-change behavior described in `quickstart.md`.

**Requirements**: FR-009

---

## Dependencies

```
WP01 (features.py core)
 ├── WP02 (extractor tests)
 ├── WP03 (training call sites)
 └── WP04 (serving verification)
```

WP02, WP03, and WP04 have no dependencies among themselves, so they may be implemented in any order after WP01. All four share `lane-a` — see the execution note in the Overview.

## MVP Scope

**WP01 alone** delivers the correctness fix. WP02–WP04 prove it and wire the consumers. WP01 + WP03 is the minimum that changes observable training behavior; WP02 and WP04 are the evidence.

## Out of Scope

No work packages exist for these, deliberately:

- Model invalidation or forced retraining of existing `.joblib` artifacts
- Retuning the stuck label threshold at `trainer.py:104`
- Feature contract versioning and manifest schema
- Any feature-store tooling or materialization
- Changes to `DataStore`, `store_sqlite.py`, or `store_postgres.py`
