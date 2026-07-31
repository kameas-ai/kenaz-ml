# Implementation Plan: Feast Feature Store Migration

**Branch**: `main` (planning base and merge target) | **Date**: 2026-07-31 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `kitty-specs/feast-feature-store-migration-01KYX32H/spec.md`

## Summary

Author feature definitions once as Feast definitions, bundle them into the open-source package, and run them in both deployments: a self-contained local Feast (file registry + SQLite online store, no network path to cloud) and a cloud Feast over PostgreSQL with a real offline store, materialization, and point-in-time-correct training retrieval.

Feature computation stays in Python. Feast provides the registry, storage, and retrieval — not the arithmetic. Feature values do not change.

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**: existing `scikit-learn`, `numpy`, `fastapi`, `uvicorn`, `joblib`, plus `feast[sqlite]` locally and `feast[postgres]` in cloud. Measured cost: **+309 MB, +338 native libraries** (137 MB/199 → 446 MB/537). Pinned exactly — the registry format is coupled to the Feast version.
**Storage**: local — file registry (shipped, read-only) + SQLite online store under `~/.local/share/sigild/`. Cloud — PostgreSQL for offline (`ml_features`) and online.
**Testing**: `pytest`; plus a structural no-egress harness and a notarization check in CI
**Target Platform**: frozen PyInstaller `onedir`, macOS-notarized (local); Linux container (cloud)
**Project Type**: single
**Performance Goals**: serving resolution within 20% of pre-migration median (NFR-002); cold start under 10s on the frozen binary (NFR-003)
**Constraints**: no network path from local to cloud (C-001); frozen binary retained (C-002); behaviour-preserving (C-005); reference-time semantics from `feature-extraction-correctness` preserved (C-006)
**Scale/Scope**: 8 model families, ~10 feature definitions, 2 deployments, 1 new package plus packaging and CI changes

## Planning Decisions

| ID | Decision | Rationale |
|---|---|---|
| D-001 | The registry is **applied at build time and shipped read-only inside the bundle**. The online store lives in `~/.local/share/sigild/`. | The notarized application directory is read-only and its contents are signed, so nothing may write there at runtime (FR-013). Shipping a pre-applied registry also makes the definitions part of the signed artifact, so they inherit notarization's integrity guarantee. `feast apply` becomes a build step, not a first-run step — which additionally removes a startup failure mode on a machine with no writable app dir. |
| D-002 | **Feast does not compute features.** `sigil_ml.features` remains the arithmetic; Feast registers, stores, materializes, and retrieves. | The features are window aggregations over variable-length event sequences (`edit_velocity`, `file_switch_rate`, `category_entropy`). Feast's request-time schema model expects flat typed columns, so expressing these as transformations would mean passing an event list as an opaque JSON string — at which point Feast's typing validates nothing. Keeping computation in Python also preserves the `as_of_ms` work just merged (C-006) and keeps both deployments bit-identical by construction. |
| D-003 | **Active-task predictions compute live and then push**; the online store is never the source for an active-task prediction. | The models predict on tasks a developer is working on *right now*. An online store returns the last materialized value, so reading it for an active task would trade correctness for microseconds (US5). Computing live and pushing the result keeps serving current, still populates the online store for other consumers, and makes the online store a byproduct rather than a dependency. This is the specific failure the April branch had inverted. |
| D-004 | No-egress is enforced by **configuration plus a structural test**, not by convention. | Feast supports remote registries, remote online stores, and remote providers by configuration. C-001 is a hard product requirement, so the local `feature_store.yaml` pins `provider: local`, `registry_type: file`, `online_store: sqlite`, and NFR-004 asserts no socket is opened across the entire local flow. Feast 0.65.0 has no telemetry module — verified against the installed package — but that is a fact to confirm in CI, not to rely on. |
| D-005 | PyInstaller integration **extends the existing `collect_submodules` pattern** in `freeze/kameas-ml.spec` rather than introducing a new mechanism. | The spec already collects `sklearn`, `scipy`, `numpy`, and `joblib` this way. Feast additionally loads providers and online/offline stores through dynamic imports that static analysis cannot follow, so those need explicit `hiddenimports` entries; `pyarrow` and `grpcio` need their binaries collected. |
| D-006 | The cloud offline store is a Python-owned **`ml_features` table in PostgreSQL**, read through a `PostgreSQLSource`. | Matches `docs/ML_ARCHITECTURE.md` §2. Keeps table ownership rules intact (C-004) and gives point-in-time joins an event-time column to sort on. Materialization writes values with the event time they describe (FR-009), which is what makes FR-007 and FR-008 achievable at all. |
| D-007 | The Feast version is **pinned exactly**, and the registry records the version that produced it. | A registry is a serialized protobuf coupled to the library version. A runtime/registry mismatch must fail with a diagnostic (FR-016) rather than a deserialization traceback, and a floating version would make the shipped registry a moving target across OSS installs. |

## Charter Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Skipped — no charter exists.** `.kittify/charter/charter.md` is absent.

Project constraints from `CLAUDE.md` are respected with one deliberate, recorded exception: the dependency ceiling. C-003 records that the measured +309 MB / +338 native library cost was accepted by the requester as a condition of this migration, superseding the ceiling **for Feast's dependency tree only**; it remains in force for anything else, and NFR-001 caps further drift. Table ownership (C-004), model names, and the `:7774` endpoints are unchanged (FR-018). The local no-egress guarantee is strengthened, not weakened, by D-004.

*Post-Phase 1 re-check*: unchanged. One new package, no schema changes to Go-owned tables, no new local network surface.

## Project Structure

### Documentation (this feature)

```
kitty-specs/feast-feature-store-migration-01KYX32H/
├── spec.md · plan.md · research.md · data-model.md · quickstart.md
├── meta.json · checklists/requirements.md
└── tasks.md            # Phase 2 — created by /spec-kitty.tasks
```

No `contracts/` — this mission changes no external interface, consistent with missions 002, 003, and 005.

### Source Code (repository root)

```
src/sigil_ml/
├── feature_store/              # NEW — Feast integration
│   ├── __init__.py
│   ├── definitions.py          # Entities, FeatureViews, FeatureServices (shipped)
│   ├── config.py               # feature_store.yaml resolution, local vs cloud
│   ├── resolve.py              # Live-compute-then-push serving path (D-003)
│   └── materialize.py          # Cloud materialization into ml_features
├── features.py                 # UNCHANGED — still the arithmetic (D-002)
├── poller.py · routes.py       # Serving call sites, routed through resolve.py
└── training/
    ├── trainer.py              # Local: unchanged retrieval path
    └── cloud_trainer.py        # Cloud: point-in-time retrieval via Feast

freeze/
├── kameas-ml.spec              # + Feast/pyarrow/grpcio hiddenimports & binaries
└── entrypoint.py

.github/workflows/ci.yml        # + notarization check, + bundle-size gate
tests/
├── test_feature_store_defs.py · test_feature_store_local.py
├── test_feature_store_cloud.py · test_no_egress.py
```

**Structure Decision**: A new `feature_store/` package rather than folding into `features.py`. The two have different jobs — `features.py` computes, `feature_store/` registers, stores, and retrieves — and D-002 depends on that separation staying legible. Four cohesive modules, each independently testable.

## Phase 0 — Research

Complete. See [research.md](./research.md). Seven decisions resolved; no `[NEEDS CLARIFICATION]` markers remain.

## Phase 1 — Design

Complete. See [data-model.md](./data-model.md) for definitions, registry layout, `ml_features` schema, and the serving/materialization flows; [quickstart.md](./quickstart.md) for verification.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Local dependency ceiling exceeded by +309 MB / +338 native libs | Requester decided to migrate to Feast in both deployments, with the frozen binary retained. Cost was measured and accepted as a condition. | Cloud-only Feast would avoid it entirely and was explicitly considered and rejected by the requester. Bounded going forward by NFR-001. |
