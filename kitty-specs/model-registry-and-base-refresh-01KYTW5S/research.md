# Phase 0 Research: Model Registry and Base Refresh

**Date**: 2026-07-31 | **Plan**: [plan.md](./plan.md)

All open decisions resolved. No `[NEEDS CLARIFICATION]` markers remain.

---

## D-001: Where the base slot lives

**Decision**: Base artifacts ship inside the distribution and are read in place. Base and local slots have different roots, resolved by separate functions.

**Rationale**: `config.models_dir()` returns `~/.local/share/sigild/ml-models` — user-writable by design, since local training writes there. A shipped base model must be read-only and tamper-evident, so that directory is the wrong home for it. Inside the frozen `onedir` bundle, base artifacts are covered by the same notarization and signing as the binary, which makes C-004 a structural property rather than a convention the code has to maintain. Upgrade replaces them as a side effect of replacing the app. And base-version-change detection (FR-012) reduces to comparing the bundle's manifest version against the `base_version` recorded in the local manifest.

Resolution differs by distribution form: `sys._MEIPASS` under a frozen bundle, package resources for a source install. Both are read-only in practice.

**Alternatives considered**:
- *Copy base into the user data directory on first run, as a sibling of local* — simplest resolution and matches the spec's "two-slot" phrasing most literally. Rejected: it makes the base copy writable, introduces a partial-copy failure mode, and creates a staleness problem where an upgraded bundle and a stale copy disagree, requiring a version check that D-001 gets for free.
- *Read base from the bundle but mirror its manifest into the user directory* — no artifact duplication, but two sources of truth for one manifest. Rejected as strictly more complex than reading both from where they live.

---

## D-002: Retained training data format

**Decision**: JSONL — one example per line, preceded by a header record carrying the contract version and schema.

**Rationale**: FR-018 requires the user be able to inspect retained data, and SC-006 requires the no-egress guarantee be verifiable. Both are undermined by an opaque format: "we keep some of your data locally, in a binary blob" is a materially weaker claim than a file the user can open. At roughly 100 bytes for a 6-float row plus a label, the 50MB default bound holds on the order of 500,000 examples — orders of magnitude beyond what one install accumulates. Compactness is therefore not the binding constraint.

**Alternatives considered**:
- *`.npz` via numpy* — roughly 10× smaller and natively typed. Rejected: opaque to inspection, which is the point of retaining locally in the first place. Revisit only if the size bound turns out to bind in practice.
- *A SQLite table* — queryable and transactional. Rejected: the existing `data.db` is the Go daemon's, and creating a second database for a bounded append-only log is heavier than the problem. Also loses the "open it and look" property.
- *Retain raw events instead of computed vectors* — would survive a contract change and remove the need for FR-014. Rejected: unbounded growth, and it duplicates data the Go daemon already owns and may prune. The contract-change reset is the accepted cost.

---

## D-003: How rebuild-on-refresh works

**Decision**: Full retraining on the retained set, seeded from the new base where the estimator supports it, without incremental extension mechanics.

> **Clarified during WP04.** "Seeded from the new base" was ambiguous and read as warm-start, which C-006 forbids. Settled reading: `sklearn.base.clone` takes the new base's **hyperparameters** and discards its fitted state, then fits on the retained set.
>
> The consequence is worth stating plainly rather than leaving implicit: the rebuilt model's **learned parameters come entirely from local data**. What the user inherits from base v2 is its *tuning*, not its *fit*. That is a real limitation of deferring warm-start, and it narrows User Story 3's promise — "the improved starting point" means improved hyperparameters, not an improved starting fit. Genuine base-fit inheritance is exactly what the deferred warm-start mission would add.

**Rationale**: C-006 defers warm-start to a later feature. Full retraining satisfies every acceptance scenario in User Story 3, works uniformly across the estimator roster — which includes several with no `partial_fit` support — and at these data volumes takes seconds. The later warm-start feature can optimize this path without altering its inputs, outputs, or the manifest it writes.

**Alternatives considered**:
- *Block refresh until warm-start exists* — rejected; it makes the chosen refresh policy undeliverable in this mission for no benefit.
- *Serve the new base and rebuild lazily in the background* — appealing for startup latency, but introduces a window where provenance and served model disagree. Deferred; refresh is infrequent and bounded by the retained-set size.

---

## D-004: Manifest as a sidecar file

**Decision**: One JSON file per artifact, alongside it.

**Rationale**: Provenance travels with the artifact if it is copied, survives a `data.db` reset, requires nothing of the Go daemon, and is readable without tooling. The same reasoning as D-002, applied to metadata rather than data. It is also the natural export target for the cloud job (C-002) — MLflow can emit a JSON file without knowing anything about local paths.

**Alternatives considered**:
- *A registry index file listing all models* — one file to read instead of N. Rejected: at 8 models the read cost is irrelevant, and a central index breaks the travels-with-the-artifact property while introducing a write-contention point.
- *A database table* — rejected as in D-002.

---

## D-005: Registry as a layer over `ModelStore`

**Decision**: `ModelStore` is unchanged. The registry wraps it.

**Rationale**: `ModelStore` is a stable bytes-in/bytes-out protocol with two implementations and existing call sites, including the cloud `S3ModelStore`. Changing it would pull cloud storage into a local-registry feature. Wrapping keeps the change surface to the new package plus `loader.py` and the two trainer call sites.

**Alternatives considered**:
- *Extend the `ModelStore` protocol with manifest methods* — more cohesive in the abstract, rejected because it forces `S3ModelStore` to implement local-registry semantics that the cloud registry (MLflow) will own instead.

---

## D-006: Contract comparison semantics

**Decision**: Compare the ordered feature-name list and dtypes; fail closed on any difference.

**Rationale**: Order *is* the vector layout — both trainers build inputs as `[features.get(f, 0.0) for f in FEATURE_NAMES]`, indexing positionally. A set-based comparison would accept a reordering that silently permutes every feature into the wrong model input, which is precisely the silent-corruption class this feature exists to eliminate. Failing closed is correct because a mismatch means predictions would be meaningless, and a refused model falls through to the next resolution step (FR-017) rather than taking the service down.

**Alternatives considered**:
- *Set comparison with a reordering shim* — tolerant of benign reorderings, and able to remap. Rejected: it converts a loud failure into a silent remap, and the remap itself becomes an unversioned transformation nothing validates.
- *Warn and proceed* — rejected outright; it is the current behavior in effect, and the reason this feature exists.

---

## D-007: Refresh trigger

**Decision**: Evaluate at startup and on demand; trigger on `base_version` comparison only.

**Rationale**: FR-012 requires detection from provenance without reference to timestamps. File mtimes are unreliable across upgrades, package installs, and copies — `last_modified()` already exists in the codebase with a docstring acknowledging it is a proxy. Version comparison is exact and intentional.

**Alternatives considered**:
- *Watch the filesystem for base changes* — unnecessary; base changes only at upgrade, which implies a restart.
- *Refresh on a schedule* — rejected; nothing changes between upgrades, so a schedule is pure overhead.

---

## Prior art consulted

- `docs/ML_ARCHITECTURE.md` §5 — established the manifest shape, two-slot layout, and the MLflow export seam this schema must satisfy.
- `src/sigil_ml/modelstore/stores.py` — `S3ModelStore` already implements versioned keys with a `latest` pointer, reviewed to confirm the manifest schema can describe both local and cloud artifacts (C-002).
- `src/sigil_ml/modelstore/loader.py` — the existing tenant-then-shared fallback, whose resolution shape the local-then-base slot order deliberately mirrors.
- `src/sigil_ml/config.py` — XDG path resolution, confirming `models_dir()` is user-writable and therefore unsuitable for the base slot.
