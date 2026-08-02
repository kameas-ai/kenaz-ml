# Tasks: Storage Layer Reorganization

**Mission**: `storage-layer-reorganization-01KYXE54`
**Branch**: `main` (planning base and merge target) · **Change mode**: `bulk_edit`
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Research**: [research.md](./research.md)
**Generated**: 2026-08-01

## Overview

17 subtasks across 3 work packages, **strictly sequential**. The ordering is not incidental — two of the constraints are about sequence:

- **The occurrence map comes before any move** (D-006, C-006). It is the bulk-edit guardrail and the review artifact.
- **Stack B tests come before the move** (D-005). Written afterwards they would only prove the code works in its new home; written before, they prove it behaves identically in both. That is what turns "behaviour-preserving" from an assertion into a check.

WP02 is a single large package by necessity, not preference. It was originally two — write the tests, then move the code — and ownership validation rejected that split: the Stack B tests must be written against the old import paths and carried to the new ones, so both packages claimed the same two files. The tests and the move are one unit of work.

The ordering survives as an **enforced commit boundary**: Part 1 (T004–T007, the tests) must be committed before Part 2 (T008–T013, the move) begins. That commit order is now the only evidence the tests were written against pre-move code.

## Subtask Index

| ID | Description | WP |
|---|---|---|
| T001 | Enumerate occurrences across all 8 bulk-edit categories | WP01 | [D] |
| T002 | Confirm no moved module is string-referenced | WP01 | [D] |
| T003 | Reconcile the map against the measured 41 imports | WP01 | [D] |
| T004 | `ModelCache` tests — TTL expiry, LRU eviction, statistics | WP02 | [D] |
| T005 | `FilesystemModelLoader` tests — tenant scoping, shared fallback, `None` on missing | WP02 | [D] |
| T006 | Prove the new tests fail when the behaviour is broken | WP02 | [D] |
| T007 | Capture a pre-move behavioural baseline | WP02 |
| T008 | Create `datastore/` and move the three data modules | WP02 |
| T009 | Create `modelstore/` and move the three model modules | WP02 |
| T010 | Package `__init__.py` re-exports for both | WP02 |
| T011 | Rewrite all import sites named in the occurrence map | WP02 |
| T012 | Delete the old module paths — no shims | WP02 |
| T013 | Carry the Stack B tests across to the new paths | WP02 |
| T014 | Verify old paths are gone and the public surface resolves | WP03 |
| T015 | Verify the frozen binary builds and serves | WP03 |
| T016 | Verify pre-change artifacts still load; check import time | WP03 |
| T017 | Update `CLAUDE.md` module references; cross-check the occurrence map | WP03 |

---

## WP01 — Occurrence Map

**Prompt**: [tasks/WP01-occurrence-map.md](./tasks/WP01-occurrence-map.md)
**Priority**: P1 · **Dependencies**: none · **Estimated prompt size**: ~180 lines

**Goal**: An exhaustive, reviewed inventory of every site a moved symbol appears, covering all eight standard bulk-edit categories.

**Independent test**: Every one of the 41 measured import statements appears in the map, and the map names an explicit action for each of the eight categories — including the ones that turn out empty.

**Included subtasks**:

- [x] T001 Enumerate occurrences across all 8 bulk-edit categories (WP01)
- [x] T002 Confirm no moved module is string-referenced (WP01)
- [x] T003 Reconcile the map against the measured 41 imports (WP01)

**Risks**: The categories most likely to be missed are the ones that are not code — documentation, `CLAUDE.md`'s direct module references, and the freeze spec. A map covering only `src/` and `tests/` is incomplete.

**Requirements**: NFR-004, C-006

---

## WP02 — Stack B Coverage, Then The Move

**Prompt**: [tasks/WP02-tests-then-move.md](./tasks/WP02-tests-then-move.md)
**Priority**: P1 · **Dependencies**: WP01 · **Estimated prompt size**: ~330 lines

**Goal**: Give the untested model-loading path coverage, then move six modules into `datastore/` and `modelstore/` and rewrite every import site.

**Independent test**: Part 1's tests each fail when their behaviour is broken in a scratch copy. Part 2's suite passes with no count regression, and every old module path raises `ModuleNotFoundError`.

**Included subtasks** — Part 1 (coverage) must be committed before Part 2 (the move) begins:

- [x] T004 `ModelCache` tests — TTL expiry, LRU eviction, statistics (WP02)
- [x] T005 `FilesystemModelLoader` tests — tenant scoping, shared fallback, `None` on missing (WP02)
- [x] T006 Prove the new tests fail when the behaviour is broken (WP02)
- [ ] T007 Capture a pre-move behavioural baseline (WP02)
- [ ] T008 Create `datastore/` and move the three data modules (WP02)
- [ ] T009 Create `modelstore/` and move the three model modules (WP02)
- [ ] T010 Package `__init__.py` re-exports for both (WP02)
- [ ] T011 Rewrite all import sites named in the occurrence map (WP02)
- [ ] T012 Delete the old module paths — no shims (WP02)
- [ ] T013 Carry the Stack B tests across to the new paths (WP02)

**Risks**: The commit boundary between Part 1 and Part 2 is the only remaining evidence the tests were written against pre-move code — the package boundary that used to enforce it is gone. Moves must be 1:1 with no logic edits (C-001, D-002); tidying while moving is what makes a safe refactor unreviewable. No compatibility shims (D-004). The two caches must not be reconciled (C-002). If an assertion needs changing to pass, the move was not behaviour-preserving — that is a finding, not a fix.

**Requirements**: FR-001, FR-002, FR-003, FR-004, FR-005, FR-008, FR-009, FR-010, NFR-002

---

## WP03 — Verification and Documentation

**Prompt**: [tasks/WP03-verification.md](./tasks/WP03-verification.md)
**Priority**: P1 · **Dependencies**: WP02 · **Estimated prompt size**: ~200 lines

**Goal**: Prove the move changed nothing, and update the documentation that names the old paths.

**Independent test**: A pre-change model artifact loads unchanged; the frozen binary builds and serves a real prediction; every old import path is gone.

**Included subtasks**:

- [ ] T014 Verify old paths are gone and the public surface resolves (WP03)
- [ ] T015 Verify the frozen binary builds and serves (WP03)
- [ ] T016 Verify pre-change artifacts still load; check import time (WP03)
- [ ] T017 Update `CLAUDE.md` module references; cross-check the occurrence map (WP03)

**Risks**: A clean frozen build proves little — the Feast mission found two packaging bugs that appeared only on first *use*. Exercise a real prediction in the built binary. `CLAUDE.md` names `src/sigil_ml/store.py` directly and is wrong the moment WP02 lands.

**Requirements**: FR-006, FR-007, NFR-001, NFR-003

---

## Dependencies

```
WP01 (occurrence map) → WP02 (tests, then the move) → WP03 (verification)
```

Strictly sequential. No parallelism is available and none should be manufactured — each package's output is the next one's precondition.

## MVP Scope

There is no meaningful MVP. A partial move is worse than no move: half-updated imports leave the codebase in a state neither structure explains. The mission lands whole or reverts whole.

## Out of Scope

- Reconciling `ModelCache` with `CachedModelStore` (C-002 — behaviour change)
- Deleting either model stack (C-003)
- Decomposing `stores.py` further (D-002 — deferred until the package boundary exists)
- Reorganizing `feature_store/`, `models/`, `signals/`, `training/` (C-005)
- Any change to the `DataStore` or `ModelStore` protocols
