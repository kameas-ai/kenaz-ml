# Implementation Plan: Model Registry and Base Refresh

**Branch**: `main` (planning base and merge target) | **Date**: 2026-07-31 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `kitty-specs/model-registry-and-base-refresh-01KYTW5S/spec.md`

## Summary

Add a registry layer above the existing `ModelStore`: a sidecar JSON manifest per artifact recording provenance, feature contract, runtime compatibility, and integrity; a two-slot resolution scheme separating shipped base models from locally-extended ones; validation that refuses an artifact rather than degrading silently; and a retained training set that lets personalization be replayed onto a newly shipped base.

`ModelStore` keeps its bytes-in/bytes-out protocol untouched. The registry wraps it.

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**: `scikit-learn`, `numpy`, `fastapi`, `uvicorn`, `joblib` — no additions (C-001). Manifest handling, hashing, and retained-data serialization use the standard library (`json`, `hashlib`).
**Storage**: Filesystem only. No database schema, no table-ownership change (C-007). Local slot under `~/.local/share/sigild/ml-models`; base slot inside the distribution (see D-001).
**Testing**: `pytest`, with fixtures for tampered artifacts, mismatched contracts, and simulated base upgrades
**Target Platform**: frozen PyInstaller `onedir` binary, notarized (local); source install via pip (local); container (cloud, unaffected by this feature)
**Project Type**: single
**Performance Goals**: manifest read+validate under 50ms/model (NFR-002); integrity check under 200ms for a 50MB artifact (NFR-003)
**Constraints**: no new dependencies; retained data local-only with no egress path (C-003); base slot not writable at runtime (C-004); depends on `feature-extraction-correctness` (C-005)
**Scale/Scope**: 8 model families; retained set capped at 50MB/model default (NFR-004); one new module plus a loader change and two trainer call sites

## Planning Decisions

| ID | Decision | Rationale |
|---|---|---|
| D-001 | The base slot lives **inside the distribution** and is read in place, not copied into the user data directory. Base and local slots therefore have different roots and separate resolvers. | `models_dir()` resolves to `~/.local/share/sigild/ml-models`, which is user-writable — an unsuitable home for an artifact that must be read-only and tamper-evident. Shipping base models inside the notarized `onedir` bundle makes C-004 structural rather than conventional: notarization covers their integrity, upgrade replaces them for free, and base-version-change detection falls out of comparing the bundle manifest against the local manifest's recorded `base_version`. The alternative — copying base into the user directory on first run — reintroduces writability, adds a partial-copy failure mode, and creates a staleness problem after upgrade. |
| D-002 | Retained training data is stored as **JSONL**, one example per line, with a header record carrying the contract version. | FR-018 and SC-006 require the user be able to inspect what is retained; an opaque binary format undercuts the privacy claim that motivates retention being local-only. At roughly 100 bytes per 6-float row, the 50MB default bound (NFR-004) holds ~500k examples — far beyond what a single install produces. Compactness is not the binding constraint; inspectability is. |
| D-003 | Rebuild-on-refresh is performed by **full retraining** on the retained set, not by incremental extension. | C-006 defers warm-start mechanics. Full retraining satisfies every User Story 3 acceptance scenario, works for every estimator in the roster regardless of `partial_fit` support, and at these data volumes costs seconds. The later warm-start feature optimizes this path without changing its contract. |
| D-004 | The manifest is a **sidecar JSON file** per artifact, not a database row. | Provenance travels with a copied artifact, survives a `data.db` reset, does not require the Go daemon to be running, and is directly readable. Consistent with the local-first, inspectable posture. |
| D-005 | The registry is a **layer above `ModelStore`**, which is unchanged. | `ModelStore` is a bytes protocol with two implementations and existing call sites. Wrapping it keeps the blast radius to the new module plus the loader, and leaves the cloud `S3ModelStore` path untouched by this feature. |
| D-006 | Contract validation compares the **ordered** feature-name list and dtypes, and fails closed. | Order is the vector layout — both trainers index positionally. A set comparison would pass on a reordering that silently permutes every input. |
| D-007 | Refresh is evaluated at **startup and on demand**, triggered by version comparison only. | Timestamps are unreliable across upgrade and file-copy operations; `base_version` in the manifest is authoritative (FR-012). |

## Charter Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Skipped — no charter exists.** `.kittify/charter/charter.md` is absent; `charter context --action plan` reports `mode: missing`.

Project-level constraints from `CLAUDE.md` are carried as spec constraints and respected by this design: no new dependencies (C-001); no change to table ownership or schema (C-007); no change to model names or the `:7774` endpoints; the security-first, local-only posture is strengthened rather than weakened, since this feature adds integrity verification before deserialization and an explicit no-egress guarantee for retained data.

*Post-Phase 1 re-check*: unchanged. One new module, no new dependencies, no persistence beyond files under paths the product already owns.

## Project Structure

### Documentation (this feature)

```
kitty-specs/model-registry-and-base-refresh-01KYTW5S/
├── spec.md
├── plan.md                    # This file
├── research.md                # Phase 0 output
├── data-model.md              # Phase 1 output — manifest and retained-record schemas
├── quickstart.md              # Phase 1 output
├── meta.json
├── checklists/requirements.md
└── tasks.md                   # Phase 2 — created by /spec-kitty.tasks
```

No `contracts/` directory: this feature changes no HTTP interface. The manifest schema is a data contract and is specified in `data-model.md`.

### Source Code (repository root)

```
src/sigil_ml/
├── registry/                  # NEW — the registry layer
│   ├── __init__.py
│   ├── manifest.py            # Schema, read/write, validation
│   ├── slots.py               # Base/local resolution, D-001 two-root paths
│   ├── retained.py            # Retained training set (JSONL), bounds, deletion
│   └── refresh.py             # Base-version detection and rebuild policy
├── config.py                  # + base_models_dir(), retained_data_dir()
├── modelstore/loader.py       # FilesystemModelLoader gains validation
├── modelstore/stores.py       # UNCHANGED (D-005)
└── training/
    ├── trainer.py             # Retain examples; strict vector lookup
    └── cloud_trainer.py       # Strict vector lookup

tests/
├── test_registry_manifest.py
├── test_registry_slots.py
├── test_registry_retained.py
└── test_registry_refresh.py
```

**Structure Decision**: Single-project layout, already established. The registry is a new package under `src/sigil_ml/` because it is four cohesive concerns (manifest, slots, retained data, refresh) that other modules consume through one entry point; putting it in a single flat module would produce one oversized file, and scattering it across existing modules would spread the validation guarantee across call sites.

## Phase 0 — Research

Complete. See [research.md](./research.md). Seven decisions resolved; no `[NEEDS CLARIFICATION]` markers remain.

## Phase 1 — Design

Complete. See [data-model.md](./data-model.md) for the manifest schema, retained-record schema, slot layout, and the resolution and refresh state machines; [quickstart.md](./quickstart.md) for verification.

## Complexity Tracking

*No Charter Check violations — no charter exists, and no `CLAUDE.md` constraint is violated.*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
