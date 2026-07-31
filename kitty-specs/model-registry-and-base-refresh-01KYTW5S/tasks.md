# Tasks: Model Registry and Base Refresh

**Mission**: `model-registry-and-base-refresh-01KYTW5S`
**Branch**: `main` (planning base and merge target)
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Research**: [research.md](./research.md) | **Data model**: [data-model.md](./data-model.md)
**Generated**: 2026-07-31

## Overview

24 subtasks across 6 work packages. WP01 defines the manifest and the validation it enables; everything else builds on it. WP02 and WP03 are independent of each other. WP04 needs all three. WP05 wires the registry into existing call sites, and WP06 verifies the guarantees that span packages.

Each WP owns its own test module, so file ownership is disjoint throughout.

**Mission dependency**: this mission must land after `feature-extraction-correctness-01KYTR7N`. That mission owns `trainer.py` and `cloud_trainer.py`, which WP05 also edits — the two must not run in concurrent lanes.

## Subtask Index

> **On the `[P]` column.** It marks *logical* independence — the subtask does not depend on its siblings' output. It does **not** mean two agents may run concurrently, which is bounded by file ownership. Most subtasks within a WP edit a single module and must be done sequentially by one agent. The only genuinely concurrent-safe pair here is **T019 ∥ T020** (`trainer.py` vs `cloud_trainer.py`). Concurrency between *work packages* is the real lever — WP02 ∥ WP03, then WP04 ∥ WP05.

| ID | Description | WP | Parallel |
|---|---|---|---|
| T001 | Manifest schema and JSON read/write | WP01 | |
| T002 | Integrity verification before deserialization | WP01 | [P] |
| T003 | Ordered feature-contract validation | WP01 | [P] |
| T004 | Runtime version compatibility check | WP01 | [P] |
| T005 | Manifest and validation tests | WP01 | |
| T006 | `base_models_dir()` and `retained_data_dir()` resolution | WP02 | |
| T007 | Two-slot resolution order | WP02 | |
| T008 | Unusable-artifact fallthrough | WP02 | [P] |
| T009 | Slot resolution tests | WP02 | |
| T010 | Retained-set JSONL writer with header | WP03 | |
| T011 | Tolerant reader | WP03 | |
| T012 | Bound enforcement and eviction | WP03 | [P] |
| T013 | Deletion and generation increment, plus tests | WP03 | |
| T014 | Base-version change detection | WP04 | |
| T015 | Same-contract rebuild | WP04 | |
| T016 | Changed-contract reset | WP04 | [P] |
| T017 | Atomicity and refresh tests | WP04 | |
| T018 | `loader.py` uses registry validation | WP05 | |
| T019 | `trainer.py` retains examples, strict lookup | WP05 | |
| T020 | `cloud_trainer.py` strict lookup | WP05 | [P] |
| T021 | Manifest written on every local training run | WP05 | |
| T022 | No-base default-state verification | WP06 | [P] |
| T023 | No-egress verification | WP06 | [P] |
| T024 | End-to-end provenance verification | WP06 | |

---

## WP01 — Manifest and Validation Core

**Prompt**: [tasks/WP01-manifest-and-validation.md](./tasks/WP01-manifest-and-validation.md)
**Priority**: P1 · **Dependencies**: none · **Estimated prompt size**: ~300 lines

**Goal**: The manifest schema, its serialization, and the three validation checks — integrity, contract, runtime — returning structured results rather than raising.

**Independent test**: A tampered artifact is refused before deserialization; a manifest whose feature list disagrees with the extractors is refused with a diagnostic naming the disagreement.

**Included subtasks**:

- [ ] T001 Manifest schema and JSON read/write (WP01)
- [ ] T002 Integrity verification before deserialization (WP01)
- [ ] T003 Ordered feature-contract validation (WP01)
- [ ] T004 Runtime version compatibility check (WP01)
- [ ] T005 Manifest and validation tests (WP01)

**Risks**: Contract comparison must be ordered — a set comparison passes on a reordering that silently permutes every model input. Integrity must run before `joblib.load`, not after; verifying a checksum post-deserialization provides no protection.

**Requirements**: FR-001, FR-002, FR-005, FR-006, FR-008

---

## WP02 — Slot Resolution

**Prompt**: [tasks/WP02-slot-resolution.md](./tasks/WP02-slot-resolution.md)
**Priority**: P1 · **Dependencies**: WP01 · **Estimated prompt size**: ~250 lines

**Goal**: Resolve the read-only base slot inside the distribution and the writable local slot under `models_dir()`, in that preference order, falling through to cold start.

**Independent test**: With no base models present — the actual current state — resolution falls through to existing cold-start behavior with no error and no change from today.

**Included subtasks**:

- [ ] T006 `base_models_dir()` and `retained_data_dir()` resolution (WP02)
- [ ] T007 Two-slot resolution order (WP02)
- [ ] T008 Unusable-artifact fallthrough (WP02)
- [ ] T009 Slot resolution tests (WP02)

**Risks**: The base slot has a different root from the local slot (D-001) and resolves differently under a frozen bundle versus a source install. Getting the frozen-bundle path wrong fails only in the packaged build, which ordinary tests will not catch — assert the resolution logic directly.

**Requirements**: FR-003, FR-004, FR-017

---

## WP03 — Retained Training Set

**Prompt**: [tasks/WP03-retained-training-set.md](./tasks/WP03-retained-training-set.md)
**Priority**: P1 · **Dependencies**: WP01 · **Estimated prompt size**: ~260 lines

**Goal**: A local, inspectable, bounded, deletable store of the training examples behind local extensions, stamped with the contract version they were computed under.

**Independent test**: Round-trip a set, exceed the bound and confirm eviction keeps the newest examples with a valid header, delete it and confirm the install still serves.

**Included subtasks**:

- [ ] T010 Retained-set JSONL writer with header (WP03)
- [ ] T011 Tolerant reader (WP03)
- [ ] T012 Bound enforcement and eviction (WP03)
- [ ] T013 Deletion and generation increment, plus tests (WP03)

**Risks**: This is new retention of user-derived data in a product whose central promise is that nothing leaves the machine. The no-egress guarantee is verified in WP06, but nothing in this package may open a socket or add a code path that could.

**Requirements**: FR-009, FR-010, FR-011, FR-018

---

## WP04 — Refresh Policy

**Prompt**: [tasks/WP04-refresh-policy.md](./tasks/WP04-refresh-policy.md)
**Priority**: P1 · **Dependencies**: WP01, WP02, WP03 · **Estimated prompt size**: ~280 lines

**Goal**: Detect a base version change and act on it — rebuild from retained data when the contract matches, reset honestly when it does not, and leave the previous model servable if anything fails.

**Independent test**: Ship a new base with an unchanged contract; assert the rebuilt model's predictions are measurably closer to the previous local model than a bare new base is. Then ship one with a changed contract and assert the retained set is discarded and the reset recorded.

**Included subtasks**:

- [ ] T014 Base-version change detection (WP04)
- [ ] T015 Same-contract rebuild (WP04)
- [ ] T016 Changed-contract reset (WP04)
- [ ] T017 Atomicity and refresh tests (WP04)

**Risks**: Rebuild is full retraining per D-003 — do not reach for warm-start, which is a later mission. Atomicity is load-bearing: a half-written artifact leaves the install with no servable model, which is worse than not refreshing.

**Requirements**: FR-012, FR-013, FR-014, FR-019

---

## WP05 — Integration

**Prompt**: [tasks/WP05-integration.md](./tasks/WP05-integration.md)
**Priority**: P1 · **Dependencies**: WP01, WP02, WP03 · **Estimated prompt size**: ~270 lines

**Goal**: Wire the registry into the existing loader and trainers — validated loading, retained-example capture, strict vector construction, and manifest writes on every local training run.

**Independent test**: Run local training; assert a manifest was written with incremented provenance, examples were retained, and vector construction used strict lookup.

**Included subtasks**:

- [ ] T018 `loader.py` uses registry validation (WP05)
- [ ] T019 `trainer.py` retains examples, strict lookup (WP05)
- [ ] T020 `cloud_trainer.py` strict lookup (WP05)
- [ ] T021 Manifest written on every local training run (WP05)

**Risks**: This package edits `trainer.py` and `cloud_trainer.py`, which the `feature-extraction-correctness` mission also owns — that mission must be merged first. Replacing `.get(f, 0.0)` with strict lookup is only safe after contract validation has run; the order matters and a `KeyError` here means validation was skipped.

**Requirements**: FR-007, FR-015, FR-016

---

## WP06 — Guarantee Verification

**Prompt**: [tasks/WP06-guarantee-verification.md](./tasks/WP06-guarantee-verification.md)
**Priority**: P1 · **Dependencies**: WP05 · **Estimated prompt size**: ~200 lines

**Goal**: Verify the three claims that span packages and cannot be tested inside any one of them.

**Independent test**: The suite itself.

**Included subtasks**:

- [ ] T022 No-base default-state verification (WP06)
- [ ] T023 No-egress verification (WP06)
- [ ] T024 End-to-end provenance verification (WP06)

**Risks**: T022 covers the state every install is actually in today, since no base models exist yet — it is the most important test in the mission and the easiest to treat as an afterthought. T023 must be structural (no socket opened) rather than asserting that a known upload function went uncalled.

**Requirements**: FR-011, FR-004, FR-017

---

## Dependencies

```
WP01 (manifest + validation)
 ├── WP02 (slots) ─┐
 ├── WP03 (retained) ─┤
 │                    ├── WP04 (refresh)
 │                    └── WP05 (integration) ── WP06 (guarantees)
```

## MVP Scope

**WP01 + WP02 + WP05** is the minimum that makes shipping a base model safe: manifests exist, validation refuses bad artifacts, the base slot is never overwritten. WP03 and WP04 deliver the chosen re-extend refresh policy, which only matters once a *second* base version ships.

## Out of Scope

No work packages exist for these:

- The `/introspect` discovery surface — next mission
- Feature selection of any kind
- Warm-start extension mechanics and forgetting mitigation
- Cloud registry, MLflow, and the manifest export job
- Building base models
- Signing artifacts; checksums only
- Any database schema or table-ownership change
