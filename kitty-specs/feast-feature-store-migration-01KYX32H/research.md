# Phase 0 Research: Feast Feature Store Migration

**Date**: 2026-07-31 | **Plan**: [plan.md](./plan.md)

All open decisions resolved. No `[NEEDS CLARIFICATION]` markers remain.

---

## Measured inputs

Two facts were established empirically before planning, against `feast[sqlite]` **0.65.0**:

**Dependency cost.** Installing `feast[sqlite]` alongside the existing local dependency set takes site-packages from **137 MB / 199 native libraries to 446 MB / 537** — a delta of **+309 MB and +338 native libraries**. Heaviest contributors: `pyarrow` 123 MB, `pandas` 41 MB, and `mypy` + `mypyc` at 55 MB as *runtime* dependencies. Every native library must be signed and stapled for macOS notarization, which is what makes packaging (WP on the freeze work) the highest-risk part of this mission.

**No telemetry.** Feast 0.65.0 contains no `usage.py` or equivalent telemetry module — the anonymous usage collection present in older 0.x releases has been removed — and core contains no phone-home endpoints (the URLs present are documentation links in docstrings). The no-egress requirement is therefore satisfiable without patching the library. D-004 verifies rather than assumes this.

---

## D-001: Registry placement

**Decision**: Apply the registry at build time and ship it read-only inside the bundle. The online store lives in `~/.local/share/sigild/`.

**Rationale**: The notarized application directory is read-only and every file in it is signed. `feast apply` writes the registry, so it cannot run at first launch against the bundle. Making it a build step also folds the definitions into the signed artifact, so they inherit notarization's integrity guarantee — a shipped registry cannot be tampered with without breaking the signature. It additionally removes a class of first-run failure on machines where the app directory is not writable.

**Alternatives considered**:
- *Apply on first run into the user data directory* — avoids a build step and allows per-machine registry regeneration. Rejected: it puts the definitions outside the signed artifact, adds a startup failure mode, and means two installs of the same version could hold different registries.
- *Registry in the user data directory, refreshed on version change* — a hybrid, and the fallback if a shipped registry proves impractical inside the bundle. Carries the same integrity gap as above.

---

## D-002: Feast does not compute

**Decision**: `sigil_ml.features` remains the arithmetic. Feast registers, stores, materializes, and retrieves.

**Rationale**: The features are window aggregations over variable-length event sequences — `edit_velocity` over a time window, `file_switch_rate` as a ratio of distinct files to edits, `category_entropy` across a classified window. Feast's request-time transformation model expects flat typed columns; expressing these through it would mean passing the event list as an opaque JSON string, at which point Feast's typing and validation contribute nothing. Keeping computation in Python preserves the reference-time semantics merged in `feature-extraction-correctness` (C-006) and makes both deployments bit-identical by construction rather than by test.

This is also the distinction the April branch got right and then undermined: it kept computation in `sigil_ml.features` but wired only the push path, leaving Feast with nothing to do but cache.

**Alternatives considered**:
- *Move definitions into Feast transformations* — a "truer" migration with one registry owning both meaning and computation. Rejected on the schema-shape mismatch above, and because it would discard the just-merged point-in-time work.
- *Hybrid — simple features in Feast, aggregations in Python* — worst of both: two places to look, and the split would follow no principle a reader could predict.

---

## D-003: Serving computes live, then pushes

**Decision**: Active-task predictions compute features live and push the result to the online store. The online store is never the source for an active-task prediction.

**Rationale**: The models answer questions about *right now* — is this developer stuck, how long will this task take. An online store returns the last materialized value, so reading it at serving time would trade correctness for microseconds on the exact path where freshness is the product. Computing live and pushing keeps predictions current, still populates the online store for other consumers and for cloud parity, and makes the store a byproduct rather than a dependency.

This resolves the tension flagged in US5 explicitly rather than inheriting Feast's default read path. It is also the inverse of the April branch, which adopted the online half and stubbed the offline half — the arrangement that left Feast earning nothing.

**Alternatives considered**:
- *Read from the online store at serving* — the conventional Feast pattern and lowest latency. Rejected: staleness is bounded by the poll interval, which is unacceptable for a stuck-detection signal.
- *Push on every poller cycle and read from the store* — narrows staleness but does not eliminate it, and adds a failure mode where a missed push silently serves an old vector.
- *Skip the local online store entirely* — was considered during design. Rejected because the requester specified SQLite locally and PostgreSQL in cloud, and because a populated store is what lets cloud and local share one retrieval path.

---

## D-004: No-egress enforcement

**Decision**: Pin `provider: local`, `registry_type: file`, `online_store: sqlite` in the local configuration, and assert structurally in CI that no socket is opened across the full local flow.

**Rationale**: C-001 is a hard product requirement, and Feast is a system that *supports* remote registries, remote online stores, and remote providers through configuration. A guarantee that rests on "we did not configure that" is one edit away from being false. The structural test — patch socket creation, run apply/materialize/serve, assert never called — catches a path nobody anticipated, which an allow-list of known network calls would not.

Feast 0.65.0's lack of a telemetry module makes this achievable, but that is a property of the pinned version and belongs in CI where a version bump would surface a regression.

**Alternatives considered**:
- *Assert no known upload function is called* — weaker; proves only that the paths we thought of are quiet.
- *Network namespace isolation in tests* — stronger still, and worth doing if the socket-level assertion proves leaky, but heavier to run in CI across platforms.

---

## D-005: PyInstaller integration

**Decision**: Extend the existing `collect_submodules` pattern in `freeze/kameas-ml.spec`; add explicit `hiddenimports` for Feast's dynamically-loaded providers and stores; collect `pyarrow` and `grpcio` binaries.

**Rationale**: The spec already collects `sklearn`, `scipy`, `numpy`, and `joblib` this way, so the mechanism is established and reviewers know it. Feast resolves providers and online/offline store implementations by name at runtime, which static analysis cannot follow — those need naming explicitly or the frozen binary will build cleanly and fail at first use, which is the worst failure shape available.

**Alternatives considered**:
- *A custom PyInstaller hook module* — cleaner separation if the entries grow large, and a reasonable refactor later. Rejected for now as a second mechanism for the same job.
- *Vendoring a trimmed Feast* — would cut the 309 MB substantially. Rejected: it forks a dependency, and the cost was measured and accepted.

---

## D-006: Cloud offline store

**Decision**: A Python-owned `ml_features` table in PostgreSQL, read through a `PostgreSQLSource`.

**Rationale**: Matches `docs/ML_ARCHITECTURE.md` §2, keeps table ownership intact (C-004), and gives point-in-time joins an event-time column. The critical property is FR-009 — values are stored with the event time they *describe*, not the time they were written. Without that, FR-007 and FR-008 are unachievable no matter what retrieval API sits on top.

**Alternatives considered**:
- *Parquet files in object storage* — Feast's most common offline store and cheaper at scale. Rejected: Postgres already exists in cloud, and a second storage system earns nothing at current volumes.
- *Recompute training features on demand instead of materializing* — what happens today. Rejected: it is precisely the replay this migration exists to remove.

---

## D-007: Version pinning

**Decision**: Pin Feast exactly; record the producing version in the registry.

**Rationale**: The registry is a serialized protobuf coupled to the library version. A floating version would make the shipped registry a moving target across open-source installs, and a runtime/registry mismatch would surface as a deserialization traceback rather than a diagnosis. FR-016 requires the mismatch be refused clearly.

---

## Prior art consulted

- `feat/feast-feature-store` (April 2026, four commits, unmerged) — the previous attempt. Its placeholder `FileSource` entries pointing at non-existent parquet paths, and its own docstring acknowledging them, are the clearest statement of what happens when the offline half is skipped. Superseded by this plan.
- `docs/ML_ARCHITECTURE.md` §2, §3.5, §10 — the `ml_features` schema, the cloud/local split, and the recorded reasons the earlier attempt was rejected.
- `freeze/kameas-ml.spec` — the established `collect_submodules` pattern that D-005 extends.
- `kitty-specs/feature-extraction-correctness-01KYTR7N/` — the reference-time semantics C-006 preserves.
