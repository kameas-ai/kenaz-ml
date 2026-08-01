# Tasks: Feast Feature Store Migration

**Mission**: `feast-feature-store-migration-01KYX32H`
**Branch**: `main` (planning base and merge target)
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Research**: [research.md](./research.md) | **Data model**: [data-model.md](./data-model.md)
**Generated**: 2026-07-31

## Overview

25 subtasks across 6 work packages. WP01 defines the features; WP02 establishes configuration and the no-egress guarantee. WP03 (local serving), WP04 (cloud), and WP05 (packaging) then proceed independently of one another. WP06 proves nothing changed.

**WP05 carries the mission's delivery risk.** Feast adds 338 native libraries that must each be signed and stapled for macOS notarization, and Feast resolves providers by dynamic import, which PyInstaller cannot follow. A binary that builds cleanly and fails on first feature call is the expected failure shape.

> **On the `[P]` column.** It marks *logical* independence, not safe concurrency — that is bounded by file ownership. Most subtasks within a WP edit one module and must be done sequentially. Concurrency between *work packages* is the real lever.

## Subtask Index

| ID | Description | WP | Parallel |
|---|---|---|---|
| T001 | Entities: task, node, tenant | WP01 | | [D] |
| T002 | Feature views mirroring `FEATURE_NAMES` exactly | WP01 | | [D] |
| T003 | Feature services, one per model | WP01 | | [D] |
| T004 | Definition-vs-constant conformance tests | WP01 | | [D] |
| T005 | Local `feature_store.yaml` — no network surface | WP02 | | [D] |
| T006 | Cloud `feature_store.yaml` — Postgres both halves | WP02 | [D] |
| T007 | Config resolution by operating mode | WP02 | | [D] |
| T008 | Structural no-egress test and config lint | WP02 | | [D] |
| T009 | Live-compute-then-push resolver | WP03 | | [D] |
| T010 | Route `poller.py` and `routes.py` through the resolver | WP03 | | [D] |
| T011 | Best-effort push semantics | WP03 | [D] |
| T012 | Live-not-stale serving tests | WP03 | | [D] |
| T013 | `ml_features` table and schema | WP04 | | [D] |
| T014 | Materialization writing `event_timestamp` | WP04 | |
| T015 | `PostgreSQLSource` wiring | WP04 | [P] |
| T016 | Materialization tests | WP04 | |
| T017 | Cloud training via `get_historical_features` | WP04 | |
| T018 | Point-in-time and leakage tests | WP04 | |
| T019 | Build-time `feast apply`, ship registry read-only | WP05 | |
| T020 | PyInstaller hidden imports and binaries | WP05 | |
| T021 | Read-only app-directory verification | WP05 | [P] |
| T022 | Notarization and bundle-size gates in CI | WP05 | |
| T023 | Pre-migration baseline capture | WP06 | |
| T024 | Value-equality regression tests | WP06 | |
| T025 | Cross-deployment agreement tests | WP06 | [P] |

---

## WP01 — Feature Definitions

**Prompt**: [tasks/WP01-feature-definitions.md](./tasks/WP01-feature-definitions.md)
**Priority**: P1 · **Dependencies**: none · **Estimated prompt size**: ~260 lines

**Goal**: Author the entities, feature views, and feature services that ship with the package and serve as the single definition for both deployments.

**Independent test**: Every feature view's ordered field list is identical to the corresponding `FEATURE_NAMES` constant the trainers index against.

**Included subtasks**:

- [x] T001 Entities: task, node, tenant (WP01)
- [x] T002 Feature views mirroring `FEATURE_NAMES` exactly (WP01)
- [x] T003 Feature services, one per model (WP01)
- [x] T004 Definition-vs-constant conformance tests (WP01)

**Risks**: Field ordering is the vector layout — both trainers build inputs positionally, so a reordering here silently permutes every model input. The conformance test must compare ordered lists, never sets.

**Requirements**: FR-001, FR-002

---

## WP02 — Configuration and No-Egress Guarantee

**Prompt**: [tasks/WP02-config-and-no-egress.md](./tasks/WP02-config-and-no-egress.md)
**Priority**: P1 · **Dependencies**: WP01 · **Estimated prompt size**: ~250 lines

**Goal**: Two configurations — local self-contained, cloud Postgres — and a structural proof that the local deployment opens no socket.

**Independent test**: Run the full local flow with socket creation patched to record calls; assert the record is empty.

**Included subtasks**:

- [x] T005 Local `feature_store.yaml` — no network surface (WP02)
- [x] T006 Cloud `feature_store.yaml` — Postgres both halves (WP02)
- [x] T007 Config resolution by operating mode (WP02)
- [x] T008 Structural no-egress test and config lint (WP02)

**Risks**: This encodes a **hard product requirement** (C-001). Feast supports remote registries, remote online stores, and remote providers by configuration, so a guarantee resting on "we didn't configure that" is one edit from false. The test must assert no socket at all, not the absence of known upload calls.

**Requirements**: FR-003, FR-004, FR-005, FR-006, NFR-004

---

## WP03 — Local Serving Resolver

**Prompt**: [tasks/WP03-local-serving-resolver.md](./tasks/WP03-local-serving-resolver.md)
**Priority**: P1 · **Dependencies**: WP01, WP02 · **Estimated prompt size**: ~260 lines

**Goal**: Active-task predictions compute live and push the result; the online store is never the source for an active-task prediction.

**Independent test**: Seed the online store with a deliberately wrong value, generate live activity, request a prediction, and assert the wrong value does not appear.

**Included subtasks**:

- [x] T009 Live-compute-then-push resolver (WP03)
- [x] T010 Route `poller.py` and `routes.py` through the resolver (WP03)
- [x] T011 Best-effort push semantics (WP03)
- [x] T012 Live-not-stale serving tests (WP03)

**Risks**: Reading the online store for an active task is the conventional Feast pattern and the wrong choice here (D-003) — it would serve a value stale by up to a poll interval on the exact path where freshness is the product. This is what the April attempt got backwards.

**Requirements**: FR-014, FR-017, NFR-002

---

## WP04 — Cloud Offline Store and Training Retrieval

**Prompt**: [tasks/WP04-cloud-offline-and-retrieval.md](./tasks/WP04-cloud-offline-and-retrieval.md)
**Priority**: P1 · **Dependencies**: WP01, WP02 · **Estimated prompt size**: ~300 lines

**Goal**: The half the previous attempt skipped — a real offline store with event-time-stamped values, and point-in-time-correct training retrieval over it.

**Independent test**: Write two values for one entity at different event timestamps; retrieve as of the earlier one; assert the later value does not appear.

**Included subtasks**:

- [x] T013 `ml_features` table and schema (WP04)
- [ ] T014 Materialization writing `event_timestamp` (WP04)
- [ ] T015 `PostgreSQLSource` wiring (WP04)
- [ ] T016 Materialization tests (WP04)
- [ ] T017 Cloud training via `get_historical_features` (WP04)
- [ ] T018 Point-in-time and leakage tests (WP04)

**Risks**: `event_timestamp` must carry the moment the values *describe* — the `as_of_ms` reference time — never the write time. Store the write time and every point-in-time guarantee silently becomes false while the API still returns rows.

**Requirements**: FR-007, FR-008, FR-009, FR-010

---

## WP05 — Packaging and Notarization

**Prompt**: [tasks/WP05-packaging-and-notarization.md](./tasks/WP05-packaging-and-notarization.md)
**Priority**: P1 · **Dependencies**: WP01, WP02 · **Estimated prompt size**: ~280 lines

**Goal**: The frozen binary builds with Feast included, notarizes with all 338 additional native libraries signed, and serves from the shipped read-only registry.

**Independent test**: Build, notarize, install from the artifact on a clean machine, and exercise an actual feature call — not merely startup.

**Included subtasks**:

- [ ] T019 Build-time `feast apply`, ship registry read-only (WP05)
- [ ] T020 PyInstaller hidden imports and binaries (WP05)
- [ ] T021 Read-only app-directory verification (WP05)
- [ ] T022 Notarization and bundle-size gates in CI (WP05)

**Risks**: **Highest-risk package in the mission.** Feast resolves providers and stores by dynamic import, which static analysis cannot follow — the expected failure is a binary that builds cleanly and fails on first feature call. Test the call path, not startup. Note also that CI's existing `build` job is currently flaky for unrelated reasons (fixed `sleep 3` racing server startup); fix or account for that before trusting a green result here.

**Requirements**: FR-011, FR-012, FR-013, FR-016, NFR-001, NFR-003, NFR-005

---

## WP06 — Behaviour-Preservation Regression

**Prompt**: [tasks/WP06-behaviour-preservation.md](./tasks/WP06-behaviour-preservation.md)
**Priority**: P1 · **Dependencies**: WP03, WP04 · **Estimated prompt size**: ~200 lines

**Goal**: Prove the migration changed no feature value in either deployment.

**Independent test**: Vectors captured from `main` before the migration, committed as literals, compared after.

**Included subtasks**:

- [ ] T023 Pre-migration baseline capture (WP06)
- [ ] T024 Value-equality regression tests (WP06)
- [ ] T025 Cross-deployment agreement tests (WP06)

**Risks**: T023 is order-sensitive — baselines captured after the migration prove nothing. Derive them from the pre-migration commit and commit them as literals with documented provenance.

**Requirements**: FR-015, FR-018

---

## Dependencies

```
WP01 (definitions)
 └── WP02 (config + no-egress)
      ├── WP03 (local serving) ─┐
      ├── WP04 (cloud offline)  ├── WP06 (regression)
      └── WP05 (packaging)      ┘
```

WP03, WP04, and WP05 are mutually independent with disjoint file ownership.

## MVP Scope

**WP01 + WP02 + WP05** is the minimum that proves the migration is *shippable* — definitions exist, the no-egress guarantee holds, and the notarized binary still works. If WP05 fails, the migration cannot reach open-source users, so it is worth attempting early rather than last.

## Out of Scope

- Changing any feature's value or meaning — behaviour-preserving migration only
- Moving computation into Feast; `sigil_ml.features` stays the arithmetic
- MLflow, model artifacts, base-model shipping — `model-registry-and-base-refresh`
- Feature selection
- Any cloud connectivity from the local deployment
- Go daemon or shared-schema changes
