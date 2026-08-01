# Feature Specification: Storage Layer Reorganization

**Feature Branch**: `storage-layer-reorganization-01KYXE54`
**Created**: 2026-07-31
**Status**: Draft
**Change mode**: `bulk_edit`
**Input**: Reorganize the storage layer so that data access and model-artifact storage are distinguishable, and all model-artifact concerns live in one package. Behaviour-preserving: no runtime behaviour changes. Also add test coverage for the model-loading path that currently has none, so the move is verifiable.

## Context

The word "store" means two unrelated things in this codebase, and model concerns are scattered across three locations with no principle explaining the split.

**"store" is overloaded.** `store.py` is *data* access — the `DataStore` protocol over `events` and `tasks`. `storage/model_store.py` is *model artifact* storage — the `ModelStore` protocol over `.joblib` bytes. Nothing in the naming distinguishes them, and both are heavily imported (14 and 21 importers respectively).

**Two parallel model stacks exist, both live.**

| | Stack A | Stack B |
|---|---|---|
| Module | `storage/model_store.py` | `loader.py` + `cache.py` |
| Abstraction | `ModelStore` — bytes | `ModelLoader` — objects |
| Caching | `CachedModelStore` | `ModelCache` |
| Importers | 21 | 1 (`app.py`) |
| Tests | `tests/test_model_store.py` | **none** |

Stack B is not dead — `app.py:87` calls `self.model_cache.get(tenant_id, model_name)` on the serving path. But it has zero test coverage, which means any refactor touching it is unverifiable today.

**This is urgent rather than cosmetic** because the `model-registry-and-base-refresh` mission adds four more model-artifact modules (`manifest.py`, `slots.py`, `retained.py`, `refresh.py`), and its integration work edits `loader.py` directly. Landing those into the current ambiguity compounds it, and would mean integrating with the untested stack.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The two kinds of storage are distinguishable (Priority: P1)

Someone reading the tree can tell at a glance which package handles observed data and which handles model artifacts, without opening either.

**Why this priority**: This is the reported problem. It also determines where the registry mission's four new modules land.

**Independent Test**: A newcomer asked "where do model artifacts live" and "where does event data come from" answers both correctly from the directory listing alone.

**Acceptance Scenarios**:

1. **Given** the source tree, **When** the package names are read, **Then** data access and model-artifact storage are separately named and neither is called `store`.
2. **Given** a model-artifact concern, **When** its module is located, **Then** it sits in the same package as the other model-artifact concerns.
3. **Given** the reorganization, **When** imports are resolved, **Then** no module has two plausible homes.

---

### User Story 2 - Nothing changes at runtime (Priority: P1)

The service behaves identically before and after. Same predictions, same endpoints, same artifacts on disk, same frozen binary.

**Why this priority**: This is a move, not a redesign. Any behaviour change is a defect, and the value of the reorganization depends entirely on it being safe to do.

**Independent Test**: The full suite passes unchanged, and the frozen binary builds and serves.

**Acceptance Scenarios**:

1. **Given** the reorganization, **When** the suite runs, **Then** the same tests pass with no count regression.
2. **Given** a prediction request, **When** it is served, **Then** the response is identical to pre-reorganization.
3. **Given** the frozen binary, **When** it is built, **Then** it builds and serves as before.
4. **Given** on-disk model artifacts written before the change, **When** they are loaded after, **Then** they load unchanged.

---

### User Story 3 - The model-loading path becomes verifiable (Priority: P1)

The tenant-scoped model loading and caching path has tests, so a future change to it can be checked rather than hoped about.

**Why this priority**: `ModelCache` and `FilesystemModelLoader` are 268 lines on the serving path with zero coverage. Moving untested code is a leap of faith; testing it first is what makes the move verifiable at all — and the registry mission is about to build on it.

**Independent Test**: Tests exist that fail if cache expiry, LRU eviction, tenant scoping, or shared-fallback resolution regress.

**Acceptance Scenarios**:

1. **Given** the cache, **When** an entry exceeds its TTL, **Then** it is treated as a miss.
2. **Given** the cache at capacity, **When** a new entry is added, **Then** the least-recently-used entry is evicted.
3. **Given** two tenants, **When** each loads the same model name, **Then** they receive their own artifact and never each other's.
4. **Given** no tenant-specific artifact, **When** a model is requested, **Then** the shared fallback is used.
5. **Given** a missing artifact, **When** load is attempted, **Then** `None` is returned rather than an exception, per the documented protocol.

---

### Edge Cases

- Modules imported by string rather than statement — verified absent for the modules being moved; only `sigil_ml.app:app` is string-referenced, and it is not moving.
- The frozen binary's `collect_submodules("sigil_ml")` must still reach the relocated packages.
- Circular imports introduced by the new package boundaries.
- Tests importing internal paths directly rather than through public names.
- Any module left importable at its old path, which would let stale imports survive silently.

## Requirements *(mandatory)*

### Functional Requirements

| ID | Requirement | Status |
|---|---|---|
| FR-001 | Data access modules MUST live in a package whose name denotes data, not the generic word "store". | Draft |
| FR-002 | Model-artifact modules MUST live in a single package, distinct from the data package. | Draft |
| FR-003 | The model-object loading and caching concerns currently at top level MUST move into the model-artifact package. | Draft |
| FR-004 | Every import of a moved module MUST be updated; no module may remain importable at its old path. | Draft |
| FR-005 | Runtime behaviour MUST be unchanged — same predictions, endpoints, artifact formats, and on-disk layout. | Draft |
| FR-006 | The frozen binary MUST build and serve after the move. | Draft |
| FR-007 | Model artifacts written before the change MUST load unchanged after it. | Draft |
| FR-008 | The model-object caching path MUST have tests covering TTL expiry, LRU eviction, and statistics. | Draft |
| FR-009 | The model-loading path MUST have tests covering tenant-scoped resolution, shared fallback, and the documented `None`-on-missing contract. | Draft |
| FR-010 | Public import paths used by tests and callers MUST be stated explicitly, so a follow-on mission knows what to import. | Draft |

### Non-Functional Requirements

| ID | Requirement | Threshold | Status |
|---|---|---|---|
| NFR-001 | No test count regression. | Pass count no lower than pre-change (487 passed, 9 skipped), plus the new coverage | Draft |
| NFR-002 | No new dependencies. | Zero additions to `pyproject.toml` | Draft |
| NFR-003 | Import time does not regress materially. | Within 10% of pre-change `import sigil_ml.app` | Draft |
| NFR-004 | Every moved symbol is accounted for. | An occurrence map covering all 8 bulk-edit categories, reviewed before implementation | Draft |

### Constraints

| ID | Constraint | Status |
|---|---|---|
| C-001 | Behaviour-preserving. No logic change, no signature change, no default change. | Draft |
| C-002 | The two caching implementations are **not** reconciled here. That is a behaviour change and a separate decision. | Draft |
| C-003 | Neither model stack is deleted. Establishing coverage is in scope; removing code is not. | Draft |
| C-004 | Table ownership, model names, and the `:7774` endpoints are unchanged. | Draft |
| C-005 | The `feature_store/` package is not reorganized — it is newly landed and coherent. | Draft |
| C-006 | This is a `bulk_edit`; an occurrence map must be produced and approved before implementation begins. | Draft |

## Key Entities

- **Data package**: The `DataStore` protocol and its SQLite and PostgreSQL implementations.
- **Model-artifact package**: `ModelStore` and implementations, the model-object loader, and the model-object cache — everything concerning a model artifact's bytes or its loaded form.
- **Occurrence map**: The reviewed inventory of every site a moved symbol appears, across all eight bulk-edit categories.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Data access and model-artifact storage are separately named packages; neither is called `store`.
- **SC-002**: All model-artifact concerns are in one package.
- **SC-003**: The full suite passes with no count regression.
- **SC-004**: The frozen binary builds and serves.
- **SC-005**: A pre-existing model artifact loads unchanged.
- **SC-006**: The previously-untested model-loading path has tests that fail when its behaviour regresses.
- **SC-007**: No module remains importable at its old path.

## Assumptions

- 41 import statements across 6 modules require rewriting, counted before planning. No moved module is referenced by string, so there is no dynamic-lookup hazard; only `sigil_ml.app:app` is string-referenced and it is not moving.
- `collect_submodules("sigil_ml")` in the freeze spec will reach relocated subpackages without change, but this is verified rather than assumed.
- The 487-test suite is sufficient to catch behavioural regressions in Stack A. Stack B has none, which is why FR-008 and FR-009 exist — the tests are written *before* the move so they can verify it.

## Out of Scope

- Reconciling the two caching implementations, or choosing between the two model stacks.
- Deleting any module.
- Reorganizing `feature_store/`, `models/`, `signals/`, or `training/`.
- The `model-registry-and-base-refresh` mission's modules, which land after this.
- Any change to the `DataStore` or `ModelStore` protocols themselves.
