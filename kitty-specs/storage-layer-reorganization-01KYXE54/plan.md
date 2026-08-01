# Implementation Plan: Storage Layer Reorganization

**Branch**: `main` (planning base and merge target) | **Date**: 2026-08-01 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `kitty-specs/storage-layer-reorganization-01KYXE54/spec.md`
**Change mode**: `bulk_edit` — an occurrence map is required before implementation (C-006)

## Summary

Split the overloaded word "store" into two clearly-named packages — `datastore/` for data access and `modelstore/` for model artifacts — and gather the scattered model-object loader and cache into the latter. Write tests for the model-loading path first, so the move itself is verifiable rather than hoped about.

41 import statements, 6 modules, no behaviour change.

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**: unchanged — no additions (NFR-002)
**Storage**: unchanged — SQLite local, PostgreSQL cloud; this moves the code that reaches them, not the schemas
**Testing**: `pytest`; current baseline **487 passed, 9 skipped**
**Target Platform**: frozen PyInstaller `onedir` (local), container (cloud)
**Project Type**: single
**Performance Goals**: import time within 10% of pre-change (NFR-003)
**Constraints**: behaviour-preserving (C-001); no cache reconciliation (C-002); nothing deleted (C-003)
**Scale/Scope**: 41 import statements across 6 modules; 2 new packages; ~268 lines of previously-untested code gaining coverage

## Planning Decisions

| ID | Decision | Rationale |
|---|---|---|
| D-001 | Packages are named **`datastore/`** and **`modelstore/`**. | Chosen by the product owner. Symmetric with the `DataStore` and `ModelStore` protocol names they contain, so the mapping from protocol to package is obvious. Rejected `data/` + `modelstore/` (asymmetric) and nesting under `models/` (that package holds estimator classes; mixing storage machinery in would change what it means, and the registry mission would then nest three levels deep). |
| D-002 | Moves are **1:1 wherever possible**; only one file is renamed for sense. `storage/model_store.py` becomes `modelstore/stores.py` because `modelstore/model_store.py` stutters. | The mission's value depends on being obviously safe. Splitting the 271-line `model_store.py` into `protocol/local/s3/cached` would be nicer organization but multiplies review surface on a change whose whole claim is "nothing happened". Decomposition can follow once the package boundary exists. |
| D-003 | Each package gets an `__init__.py` re-exporting its public names; callers import from the package, not the submodule. | Gives the follow-on registry mission a stable surface to import, and means a later internal decomposition (D-002) will not ripple through call sites a second time. FR-010 requires these paths be stated explicitly. |
| D-004 | **No compatibility shims at the old paths.** Old modules are deleted, not left re-exporting. | FR-004 requires no module remain importable at its old path. A shim would let stale imports survive silently, which is exactly the ambiguity being removed — and with 487 tests, a missed import fails loudly and immediately, which is the outcome we want. |
| D-005 | **Tests for Stack B are written before the move**, against the current paths, then carried across. | They exist to verify the move. Written afterwards they would only prove the code works in its new home, not that it behaves as it did in the old one. This ordering is what turns FR-005 from an assertion into a check. |
| D-006 | The occurrence map is produced and reviewed **before any file moves**. | C-006, and the bulk-edit guardrail. 41 sites is small enough to enumerate exhaustively and large enough that a missed category is plausible. |

## Charter Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Skipped — no charter exists.** `.kittify/charter/charter.md` is absent.

`CLAUDE.md` constraints are respected and untouched by this mission: no dependency change (NFR-002), no table-ownership change, model names unchanged, `:7774` endpoints unchanged (C-004). Invariant 5 — "never import `sqlite3` or `psycopg2` directly, go through the `DataStore` protocol" — is *strengthened*, since the protocol gets a package that names it.

`CLAUDE.md` will need its module paths updated as part of this mission; it names `src/sigil_ml/store.py` directly.

*Post-Phase 1 re-check*: unchanged. No new dependencies, no new runtime surface, no persistence change.

## Project Structure

### Documentation (this feature)

```
kitty-specs/storage-layer-reorganization-01KYXE54/
├── spec.md · plan.md · research.md · quickstart.md
├── occurrence_map.yaml      # bulk-edit inventory, required before implementation
├── meta.json · checklists/requirements.md
└── tasks.md                 # Phase 2 — created by /spec-kitty.tasks
```

No `data-model.md` — this mission introduces no entities, schemas, or persisted structures; it relocates existing modules. No `contracts/` — no external interface changes.

### Source Code — before and after

```
BEFORE                              AFTER
src/sigil_ml/                       src/sigil_ml/
├── store.py               (189) →  ├── datastore/
├── store_sqlite.py        (323) →  │   ├── __init__.py     re-exports
├── store_postgres.py      (395) →  │   ├── protocol.py     DataStore, create_store
│                                   │   ├── sqlite.py       SqliteStore
│                                   │   └── postgres.py     PostgresStore
├── storage/                        │
│   └── model_store.py     (271) →  ├── modelstore/
├── loader.py              (100) →  │   ├── __init__.py     re-exports
├── cache.py               (168) →  │   ├── stores.py       ModelStore, Local, S3, Cached
│                                   │   ├── loader.py       ModelLoader, FilesystemModelLoader
│                                   │   └── cache.py        ModelCache
└── (unchanged)                     └── (unchanged)
    app.py cli.py config.py             feature_store/ models/ signals/ training/
    features.py poller.py routes.py
```

**Structure Decision**: Two sibling packages at `src/sigil_ml/`, matching the existing convention (`feature_store/`, `models/`, `signals/`, `training/` are all top-level packages). `storage/` disappears entirely — its single module moves into `modelstore/`, which is where the registry mission's `manifest.py`, `slots.py`, `retained.py`, and `refresh.py` will subsequently land.

## Public import surface (FR-010)

What callers and the follow-on mission should import:

```python
from sigil_ml.datastore import DataStore, create_store
from sigil_ml.modelstore import (
    ModelStore, LocalModelStore, S3ModelStore, CachedModelStore, model_store_factory,
    ModelLoader, FilesystemModelLoader,
    ModelCache, create_model_cache,
)
```

Submodule paths (`sigil_ml.modelstore.stores`) remain importable but are not the supported surface; D-003 exists so a future decomposition does not break callers twice.

## Phase 0 — Research

Complete. See [research.md](./research.md). Six decisions resolved; no `[NEEDS CLARIFICATION]` markers remain.

## Phase 1 — Design

The design *is* the structure above plus the occurrence map. See [quickstart.md](./quickstart.md) for verification. `occurrence_map.yaml` is produced as the first work package, before any file moves (D-006).

## Complexity Tracking

*No Charter Check violations — no charter exists, and no `CLAUDE.md` constraint is violated.*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
