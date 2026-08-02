# Feature Specification: Model Registry and Base Refresh

**Feature Branch**: `model-registry-and-base-refresh-01KYTW5S`
**Created**: 2026-07-31
**Status**: Draft
**Input**: Give the local install a model registry: a manifest per artifact, a two-slot layout separating shipped base models from locally-extended ones, contract validation at load, and integrity verification. Retain the local training data so that when a new base model ships, personalization can be replayed onto it rather than discarded.

## Context

Base models will be built centrally and shipped with the product; each install extends them on the user's own data. Nothing in the current code can support that safely.

`LocalModelStore` writes `{models_dir}/{name}.joblib` and overwrites in place — there is no versioning, so the first local extension destroys the pristine base and there is nothing to roll back to. `last_modified()` carries a docstring acknowledging that `/introspect` uses file mtime as a proxy for "last trained" because no metadata exists. And both trainers build vectors as `[features.get(f, 0.0) for f in FEATURE_NAMES]`, which is safe while one process trains and serves, and becomes a silent-corruption vector the moment an artifact trained elsewhere is loaded against a different extractor version.

The registry closes all of that with one artifact: a manifest carrying identity, provenance, the feature contract, runtime compatibility, and integrity. It is also the interface to the cloud side — the MLflow export job writes the same schema the local registry reads.

Base refresh policy is **re-extend**: retained local training data is replayed onto a newly shipped base so users do not lose personalization on upgrade. This widens the registry to govern retained training data, not only artifacts. See `docs/ML_ARCHITECTURE.md` §5.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A shipped base model loads safely or not at all (Priority: P1)

The product ships with base models. On load, the install verifies the artifact matches its recorded checksum, that the recorded feature contract matches what the local extractors actually produce, and that the runtime can deserialize it. Any mismatch refuses the artifact with a diagnostic naming what disagreed. Nothing is loaded on hope.

**Why this priority**: Deserializing a `.joblib` is arbitrary code execution, and a contract mismatch silently degrades every prediction. Both failure modes are invisible without this check, and both arrive with the first shipped base model.

**Independent Test**: Ship a manifest whose feature list omits a feature the extractors produce. Attempt to load. Assert refusal with a diagnostic naming the missing feature, and assert no model object was constructed.

**Acceptance Scenarios**:

1. **Given** an artifact whose bytes do not match the checksum in its manifest, **When** load is attempted, **Then** the artifact is refused and never deserialized.
2. **Given** a manifest whose feature contract disagrees with the extractors, **When** load is attempted, **Then** the artifact is refused with a diagnostic naming the specific disagreement.
3. **Given** a manifest recording a runtime version incompatible with the running one, **When** load is attempted, **Then** the artifact is refused with a diagnostic rather than a deserialization traceback.
4. **Given** a valid artifact and manifest, **When** load is attempted, **Then** the model loads and reports its provenance.

---

### User Story 2 - Local extension is always recoverable (Priority: P1)

Extending a model locally never touches the shipped base. The pristine artifact remains on disk exactly as delivered, so a local extension that degrades quality can be discarded and the base re-served. At any moment the install can answer: which base is this descended from, how many times has it been extended, and against which contract.

**Why this priority**: Local extension can degrade a model, and without a preserved base there is nothing to fall back to. This is also what makes support tractable — "is this user running our base or their own" is currently unanswerable.

**Independent Test**: Extend a model locally three times, then reset. Assert the served model is byte-identical to the shipped base and that provenance reported three extensions before the reset.

**Acceptance Scenarios**:

1. **Given** a local extension runs, **When** it completes, **Then** the base artifact and its manifest are unmodified.
2. **Given** a locally-extended model exists, **When** provenance is queried, **Then** it reports the base version it descended from and the number of extensions applied.
3. **Given** a local extension has degraded quality, **When** the model is reset, **Then** the shipped base is served and local provenance is cleared.
4. **Given** no local model exists, **When** a model is requested, **Then** the base is served.
5. **Given** neither a local nor a base model exists, **When** a model is requested, **Then** the existing cold-start behavior applies unchanged.

---

### User Story 3 - A base refresh preserves personalization (Priority: P1)

A user has been running for months; their local model has adapted to how they work. They upgrade, and the release carries a newer base model. Their personalization is not thrown away — the training data behind it was retained, and it is replayed onto the new base so they get both the improved starting point and their own adaptation.

**Why this priority**: This is the chosen refresh policy and the reason base models are worth shipping at all. Without it, either users lose their adaptation on every upgrade or base improvements never reach long-running installs.

**Independent Test**: Build a local model by extending base v1 with a known dataset. Ship base v2 with an unchanged contract. Trigger refresh. Assert the resulting model reflects both the v2 base and the retained data — measurably closer to the pre-refresh local model than a bare v2 base is.

**Acceptance Scenarios**:

1. **Given** a local model descended from base v1 and a newly shipped base v2 with the same contract version, **When** refresh runs, **Then** a new local model is built from v2 plus the retained training data.
2. **Given** refresh has completed, **When** provenance is queried, **Then** it reports descent from v2 and the retained-data generation used.
3. **Given** no retained data exists, **When** a base refresh occurs, **Then** the new base is served directly without error.
4. **Given** refresh fails partway, **When** the failure is detected, **Then** the previous local model continues to be served rather than leaving the install with no model.

---

### User Story 4 - A contract change is handled honestly (Priority: P2)

Sometimes a new base ships with a changed feature contract — features added, removed, or reordered. Retained training data was recorded against the old contract and cannot be replayed onto the new one. The system does not pretend otherwise: it discards the stale data, serves the new base, and says so.

**Why this priority**: This is the failure mode that would otherwise silently corrupt a refresh, feeding vectors with one meaning into a model expecting another. It is lower priority only because it cannot occur until a second contract version exists.

**Independent Test**: Retain data under contract v1. Ship base v2 with contract v2. Trigger refresh. Assert the retained data is discarded, the new base is served unextended, and the reset is recorded where an operator can see it.

**Acceptance Scenarios**:

1. **Given** retained data recorded under one contract version and a base shipping a different one, **When** refresh runs, **Then** the retained data is discarded rather than replayed.
2. **Given** such a reset occurs, **When** provenance is queried, **Then** it reports that personalization was reset and why.
3. **Given** such a reset occurs, **When** local training next runs, **Then** retained-data accumulation begins again under the new contract version.

---

### User Story 5 - Retained data stays on the machine (Priority: P1)

The retained training data is derived from the user's own activity. It is written locally, never transmitted, inspectable in place, and deletable on demand. Deleting it costs personalization and nothing else — the install continues serving the base model.

**Why this priority**: Retaining user-derived data is new behavior for the open-source install, and the product's central promise is that nothing leaves the machine. The guarantee has to be explicit and verifiable, not implied.

**Independent Test**: Run local training to accumulate retained data. Assert no network egress occurs. Delete the retained data. Assert the install continues serving and begins re-accumulating.

**Acceptance Scenarios**:

1. **Given** local training runs, **When** retained data is written, **Then** no network transmission occurs.
2. **Given** retained data exists, **When** the user inspects the storage location, **Then** its contents and size are discoverable without special tooling.
3. **Given** the user deletes retained data, **When** a model is next requested, **Then** the current model continues to serve and accumulation restarts.
4. **Given** retained data reaches its configured bound, **When** more is produced, **Then** the bound is enforced by a documented, predictable policy.

---

### Edge Cases

- No base models shipped at all — the current state. Resolution must fall through to existing cold-start behavior without error.
- Local model present, base absent (base removed by a later release) — local continues to serve; provenance reports a base that is no longer on disk.
- Manifest present, artifact missing, or the reverse — treated as unusable; resolution falls through.
- Manifest is corrupt or unparseable JSON — treated as unusable, not as a crash.
- Checksum mismatch on the base slot specifically — the install cannot self-repair; must fail loudly enough to be reportable.
- Retained data exists but is empty or truncated — treated as no retained data.
- Base refresh interrupted mid-write — previous local model must remain servable.
- Two model families disagree on contract version — validation is per-model, not global.
- Clock-independent: refresh triggers on base *version* change, never on timestamps.

## Requirements *(mandatory)*

### Functional Requirements

| ID | Requirement | Status |
|---|---|---|
| FR-001 | Every model artifact MUST have an accompanying manifest describing it. | Draft |
| FR-002 | The manifest MUST record identity, provenance, runtime compatibility, feature contract, training summary, and artifact integrity. | Draft |
| FR-003 | Local storage MUST separate a read-only shipped-base slot from a writable local slot, per model. | Draft |
| FR-004 | Model resolution MUST prefer the local slot, then the base slot, then existing cold-start behavior. | Draft |
| FR-005 | Artifact integrity MUST be verified against the manifest before deserialization, never after. | Draft |
| FR-006 | The recorded feature contract MUST be validated against what the local extractors produce before a model is used; mismatch MUST refuse the model with a diagnostic identifying the disagreement. | Draft |
| FR-007 | Vector construction MUST use strict lookup after contract validation, replacing the silent zero-default currently used by both trainers. | Draft |
| FR-008 | The manifest MUST record the runtime version used to serialize the artifact, and load MUST refuse incompatible versions with a diagnostic. | Draft |
| FR-009 | Local training MUST retain the training examples it used, stamped with the contract version they were computed under. | Draft |
| FR-010 | Retained training data MUST be bounded by a documented, predictable retention policy. | Draft |
| FR-011 | Retained training data MUST NOT be transmitted off the machine under any configuration of the local deployment. | Draft |
| FR-012 | A change of shipped base version MUST be detectable from manifest provenance alone, without reference to timestamps. | Draft |
| FR-013 | On base refresh where the contract version is unchanged, the local model MUST be rebuilt from the new base plus retained data. | Draft |
| FR-014 | On base refresh where the contract version differs, retained data MUST be discarded, the new base served unextended, and the reset recorded. | Draft |
| FR-015 | Local training MUST NOT write to the base slot. | Draft |
| FR-016 | Every local training run MUST write an updated manifest recording the new provenance. | Draft |
| FR-017 | An unusable artifact — missing, corrupt, or mismatched manifest — MUST fall through the resolution order rather than raise to the caller. | Draft |
| FR-018 | The user MUST be able to inspect and delete retained training data, and deletion MUST leave the install servable. | Draft |
| FR-019 | A failed refresh MUST leave the previously-served model intact and servable. | Draft |

### Non-Functional Requirements

| ID | Requirement | Threshold | Status |
|---|---|---|---|
| NFR-001 | No new runtime dependencies are introduced. | Zero additions to the local dependency set | Draft |
| NFR-002 | Manifest read, parse, and validation is not a perceptible cost at load. | Under 50ms per model | Draft |
| NFR-003 | Integrity verification is not a perceptible cost at load. | Under 200ms for an artifact up to 50MB | Draft |
| NFR-004 | Retained training data is bounded on disk. | Configurable cap, default not exceeding 50MB per model | Draft |
| NFR-005 | Manifest schema is expressible by the cloud export job. | Schema documented and validated by an automated check both sides can run | Draft |

### Constraints

| ID | Constraint | Status |
|---|---|---|
| C-001 | Manifest handling uses the standard library only — no new dependencies. Note the project ceiling now also permits `feast` (recorded exception in `CLAUDE.md`); that does not license anything further. | Draft |
| C-002 | The manifest schema is a shared interface with the cloud registry; it MUST be implementable by an MLflow export job without local-only assumptions. | Draft |
| C-003 | Retained training data is local-only. No configuration may enable its transmission from the open-source deployment. | Draft |
| C-004 | The base slot is read-only at runtime; it is written only by installation or upgrade. | Draft |
| C-005 | This feature depends on `feature-extraction-correctness` having landed — a stable, correct feature vector is a precondition for recording it in a contract. | Draft |
| C-006 | Warm-start extension mechanics are out of scope. Rebuild-from-retained-data is performed by full retraining; optimizing it is a later feature. | Draft |
| C-007 | Existing database table ownership is unchanged. The registry is filesystem-based and introduces no schema. | Draft |

## Key Entities

- **Manifest**: The record describing one model artifact — identity, provenance, runtime compatibility, feature contract, training summary, metrics, integrity. The interface between the cloud export job and the local registry.
- **Base slot**: Read-only storage for shipped base artifacts and their manifests. Written only by installation or upgrade.
- **Local slot**: Writable storage for locally-extended artifacts. The served model when present.
- **Feature contract**: Ordered feature names with types and a version. Recorded in the manifest, validated against extractor output before use.
- **Retained training set**: Locally-stored training examples with labels, stamped with the contract version under which they were computed. Input to base refresh.
- **Provenance**: Which base a model descended from, how many extensions have been applied, what data trained it, and whether personalization has been reset.
- **Refresh**: The operation triggered by a base version change, which either replays retained data onto the new base or resets to it.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A model whose recorded contract disagrees with the extractors is refused, and the diagnostic names the specific disagreement.
- **SC-002**: A modified artifact is refused before any deserialization occurs.
- **SC-003**: After a same-contract base refresh, the rebuilt model's predictions on the retained set are measurably closer to the pre-refresh local model than a bare new base is — personalization survives.
- **SC-004**: After a changed-contract base refresh, the new base is served and the reset is discoverable by an operator.
- **SC-005**: The shipped base artifact is byte-identical to its delivered form after any number of local extensions.
- **SC-006**: Zero bytes of retained training data leave the machine, verified by an automated check.
- **SC-007**: For any served model, provenance answers which base it descended from, how many extensions it carries, and under which contract version.
- **SC-008**: With no base models present, behavior is unchanged from today.

## Assumptions

- Base models do not exist yet, so the base slot will be empty on every install until the first is built. All resolution paths must behave correctly in that state, and it is the default case for the foreseeable term.
- Retained training data is stored as computed feature vectors with labels rather than raw events, because vectors are bounded, already derived, and survive event pruning by the Go daemon. The consequence is that they are only replayable against the contract version they were computed under, which is what FR-014 exists to handle.
- The manifest is a sidecar file per artifact rather than a database table, so that provenance travels with a copied artifact, survives a database reset, and is readable by the user — which matters for an open-source privacy claim.
- Runtime compatibility checking targets serialization-format compatibility. The frozen binary pins its runtime; source installs do not, which is where the check earns its keep.
- Refresh is triggered by comparing recorded base version against the shipped base version at startup or on demand, not on a schedule.

## Known Limitation — the local serving path does not consult the registry

**Found by WP06, verified, and not fixed in this mission.**

`app.py` wires `FilesystemModelLoader` only in the **cloud** branch. In local mode each predictor loads its own artifact directly — `StuckPredictor.__init__` calls `LocalModelStore.load()` then `joblib.load()` — bypassing `resolve_model`, integrity verification, ordered contract validation, the runtime check, and **the base slot entirely**.

Consequences, stated plainly:

- **FR-004's base-slot step is unreachable in the open-source local deployment.** A shipped base model would not be served locally today.
- FR-005, FR-006 and FR-008 gate *training* and the *cloud* loader, not local serving.
- This is also *why* SC-008 holds so cleanly: local serving is literally untouched.

So this mission delivers the registry, the slot model, retained data, the refresh policy, and validated loading for training and cloud — but **not** the end-to-end "ship a base model and serve it locally" path that motivated it. Wiring that requires changing eight predictor classes and would alter local serving behaviour, which is precisely what SC-008 protects; it belongs in its own mission with its own verification.

No work package owned `app.py` or `models/*.py`, so this is a gap in the task decomposition rather than a work package failing its brief.

---

## Out of Scope

- The discovery surface exposing this registry — deferred to the feature-discovery mission.
- Feature selection of any kind.
- Warm-start extension mechanics and forgetting mitigation; rebuild is full retraining here.
- The cloud-side registry, MLflow integration, and the export job that produces manifests.
- Building the base models themselves, which is manual central work.
- Signing base artifacts. Checksums are in scope; a signing chain is a follow-on.
- Any change to the database schema or table ownership.
