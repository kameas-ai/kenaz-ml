# Phase 0 Research: Storage Layer Reorganization

**Date**: 2026-08-01 | **Plan**: [plan.md](./plan.md)

All decisions resolved. No `[NEEDS CLARIFICATION]` markers remain.

---

## Measured before planning

| | Value |
|---|---|
| Import statements to rewrite | **41** — `store` 11, `storage` 22, `store_sqlite` 4, `store_postgres` 2, `loader` 1, `cache` 1 |
| Modules moving | 6 |
| Suite baseline | 487 passed, 9 skipped |
| Untested lines gaining coverage | ~268 (`cache.py` 168 + `loader.py` 100) |
| String-referenced modules among those moving | **none** |

That last row is what makes this safe. The only string-referenced module in the codebase is `sigil_ml.app:app` (uvicorn, at `cli.py:98` and in the freeze spec), and it is not moving. There is no dynamic-lookup path that a rename could silently break — every reference is a real import statement that will fail loudly if missed.

---

## D-001: Package names

**Decision**: `datastore/` and `modelstore/`.

**Rationale**: Chosen by the product owner from three options. Symmetric with the `DataStore` and `ModelStore` protocol names each contains, so the protocol-to-package mapping needs no explanation.

**Alternatives considered**:
- *`data/` + `modelstore/`* — asymmetric; "data" reads as broader than the `DataStore` protocol it actually holds.
- *`data/` + `models/artifacts/`* — keeps model concerns under one top-level name, but `models/` currently holds estimator classes (`stuck`, `duration`, `fleet_*`). Mixing storage machinery in changes what that package means, and the registry mission's four modules would land three levels deep.

---

## D-002: Move granularity

**Decision**: 1:1 moves, with one rename — `storage/model_store.py` → `modelstore/stores.py`.

**Rationale**: The mission's entire value rests on the move being obviously safe. Splitting the 271-line `model_store.py` into `protocol.py` / `local.py` / `s3.py` / `cached.py` would be better organization, but it multiplies review surface on a change whose central claim is that nothing happened. `modelstore/model_store.py` stutters, hence the one rename.

**Alternatives considered**:
- *Full decomposition now* — rejected as above. It becomes cheap and low-risk once the package boundary exists, and D-003's re-exports mean it will not ripple through call sites.
- *Keep the filename `model_store.py`* — rejected only for the stutter; harmless otherwise.

---

## D-003: Package-level re-exports

**Decision**: Each package's `__init__.py` re-exports its public names; callers import from the package.

**Rationale**: Gives the follow-on registry mission a stable surface, and means a later internal decomposition (D-002) does not force a second round of call-site edits. FR-010 requires the surface be stated, which this makes possible.

**Alternatives considered**:
- *Import submodules directly everywhere* — simpler today, but couples every caller to the current file layout, which D-002 explicitly anticipates changing.

---

## D-004: No compatibility shims

**Decision**: Old module paths are deleted, not left re-exporting.

**Rationale**: FR-004 requires that no module remain importable at its old path, and a shim defeats it — stale imports would keep working and the ambiguity would persist in a new form. With 487 tests, a missed import fails immediately and points at itself, which is a better outcome than a silent shim. The blast radius is 41 known sites, all enumerable.

**Alternatives considered**:
- *Deprecation shims with warnings* — appropriate for a published library with external consumers. This is an internal package with every caller in the same repository.

---

## D-005: Tests before the move

**Decision**: Write coverage for `ModelCache` and `FilesystemModelLoader` against their **current** paths first, then carry the tests across with the code.

**Rationale**: These 268 lines are on the live serving path — `app.py:87` calls `model_cache.get(tenant_id, model_name)` — with zero tests. Moving untested code cannot be verified as behaviour-preserving, which is the mission's central claim (FR-005, C-001). Written *after* the move, the same tests would only prove the code works in its new home; written before, they prove it behaves identically in both. This ordering is what makes the claim checkable rather than asserted.

**Alternatives considered**:
- *Move first, test after* — faster, and forfeits the guarantee.
- *Move without testing* — the current situation projected forward; leaves the registry mission integrating against an unverified path.

---

## D-006: Occurrence map before any move

**Decision**: Produce and review `occurrence_map.yaml` covering all eight standard categories before any file is moved.

**Rationale**: C-006 and the bulk-edit guardrail. 41 sites is small enough to enumerate exhaustively and large enough that missing a category — documentation, or the `CLAUDE.md` module references, or test fixtures — is plausible. The map is cheap insurance and doubles as the review artifact.

---

## Prior art consulted

- `src/sigil_ml/feature_store/` — the most recently added package, and the convention this follows: a top-level package with an `__init__.py` exposing a public surface.
- `src/sigil_ml/app.py:83-87` — the only consumer of Stack B, and the reason it is live rather than dead.
- `kitty-specs/model-registry-and-base-refresh-01KYTW5S/` — the follow-on mission whose `manifest.py`, `slots.py`, `retained.py` and `refresh.py` land in `modelstore/`, and whose integration work edits `loader.py`. Its existence is why this mission runs first.
