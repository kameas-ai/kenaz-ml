# Phase 1 Data Model: Model Registry and Base Refresh

**Date**: 2026-07-31 | **Plan**: [plan.md](./plan.md)

No database schema. All structures are files under paths the product already owns.

---

## Slot Layout

Two roots, per D-001.

```
<distribution>/ml-base/                      # read-only, ships with the product
  stuck.joblib
  stuck.json                                 # manifest
  duration.joblib
  duration.json
  ...

~/.local/share/sigild/ml-models/             # config.models_dir(), user-writable
  stuck.joblib                               # locally-extended artifact
  stuck.json                                 # manifest
  ...
  retained/
    stuck.jsonl                              # retained training set
    duration.jsonl
```

The distribution root resolves to `sys._MEIPASS/ml-base` under a frozen bundle and to package resources for a source install. Both are read-only in practice; neither is written at runtime.

The local root keeps its existing flat `{name}.joblib` layout, so pre-registry artifacts are found unchanged — they simply have no manifest and are handled per the resolution rules below.

---

## Manifest

One JSON file per artifact, named `{model_name}.json`. This schema is the shared interface with the cloud export job (C-002).

```json
{
  "schema_version": "1",
  "name": "stuck",
  "version": "4",
  "created_at": 1753900000000,

  "provenance": {
    "base_version": "1",
    "base_sha256": "9f2c…",
    "n_local_extensions": 3,
    "training_source": "local",
    "reset_reason": null
  },

  "runtime": {
    "estimator": "GradientBoostingClassifier",
    "sklearn_version": "1.5.2",
    "python_version": "3.12"
  },

  "feature_contract": {
    "service": "stuck",
    "service_version": "9f2c4a71b3e05d18",
    "names": ["test_failure_count", "time_in_phase_sec", "edit_velocity",
              "file_switch_rate", "session_length_sec", "time_since_last_commit_sec"],
    "dtypes": ["float64", "float64", "float64", "float64", "float64", "float64"]
  },

  "training": {
    "n_samples": 847,
    "retained_generation": "2",
    "as_of_ms": 1753900000000
  },

  "metrics": { "accuracy": 0.81, "n_holdout": 120 },

  "artifact_sha256": "4a71…"
}
```

### Field semantics

| Field | Required | Meaning |
|---|---|---|
| `schema_version` | yes | Manifest format version. Distinct from `feature_contract.version`. |
| `name` | yes | Model family. Must match one of the names Go queries. |
| `version` | yes | Monotonic per slot. Base versions are assigned centrally; local versions increment per extension. |
| `provenance.base_version` | yes for local | Which base this descended from. `null` for a base manifest itself. |
| `provenance.base_sha256` | yes for local | Identifies the exact base artifact, so a same-numbered but different base is detectable. |
| `provenance.n_local_extensions` | yes | `0` for a pristine base being served directly. |
| `provenance.training_source` | yes | `base` \| `synthetic` \| `local`. |
| `provenance.reset_reason` | no | Set when personalization was discarded, e.g. `contract_version_changed`. Answers FR-014's "recorded" requirement. |
| `runtime.sklearn_version` | yes | Checked at load (FR-008). |
| `feature_contract.names` | yes | **Ordered.** This list is the vector layout. |
| `feature_contract.service` | yes | Feast feature service name. The contract is *sourced* from Feast, not authored here. |
| `feature_contract.service_version` | yes | From `feature_store.materialize.feature_service_version()` — changes exactly when the contract does. Primary check. |
| `training.retained_generation` | no | Which generation of retained data trained this; increments on reset. |
| `artifact_sha256` | yes | Verified before deserialization (FR-005). |

A base manifest omits `provenance.base_version`/`base_sha256` and carries `training_source: "base"`.

---

## Retained Training Set

One JSONL file per model under `{models_dir}/retained/{name}.jsonl`. Header record first, then one example per line.

```jsonl
{"record":"header","contract_version":"2","generation":"2","names":["test_failure_count","…"],"created_at":1753900000000}
{"record":"example","x":[5.0,1800.0,2.3,0.4,3600.0,900.0],"y":1.0,"as_of_ms":1753812345678}
{"record":"example","x":[1.0,300.0,0.8,0.9,1200.0,1200.0],"y":0.0,"as_of_ms":1753814445678}
```

| Field | Meaning |
|---|---|
| `contract_version` | The contract the vectors were computed under. Compared on refresh (FR-014). |
| `generation` | Increments whenever the set is reset. Recorded into the manifest that trains from it. |
| `names` | Ordered names, duplicated from the contract so the file is self-describing when read alone. |
| `x` | The feature vector, in `names` order. |
| `y` | The label. |
| `as_of_ms` | The reference time the vector was computed at — from `feature-extraction-correctness`. |

**Invariants**: append-only during normal operation; rewritten only on eviction or reset. A file whose header is missing or unparseable is treated as no retained data. A truncated final line is discarded rather than failing the read.

**Bound** (NFR-004): enforced by size check on append. When exceeded, the oldest examples are evicted — the file is rewritten retaining the newest examples that fit. The policy is documented and predictable, per FR-010.

---

## Resolution

Applied per model, on load.

```
1. Local slot has artifact + manifest?
     → validate(manifest, artifact)
         → pass: serve it. done.
         → fail: log the specific failure, continue to 2.
2. Base slot has artifact + manifest?
     → validate(manifest, artifact)
         → pass: serve it. done.
         → fail: log loudly (the install cannot self-repair), continue to 3.
3. Existing cold-start behavior, unchanged.
```

`validate()` is three checks, in this order, short-circuiting:

1. **Integrity** — recomputed SHA-256 of the artifact bytes equals `artifact_sha256`. Runs *before* any deserialization (FR-005).
2. **Contract** — `feature_contract.names` equals, in order, the names the local extractors produce for this model; dtypes match. Fails closed (D-006).
3. **Runtime** — `runtime.sklearn_version` is compatible with the running version.

Any failure returns a structured result naming which check failed and what disagreed (FR-006), never a bare exception to the caller (FR-017).

---

## Refresh

Evaluated at startup and on demand (D-007). Compares the base slot's manifest `version` against the local manifest's `provenance.base_version`.

```
shipped_base.version == local.provenance.base_version?
   → no change. done.

Otherwise a base refresh is due:

   retained.contract_version == shipped_base.feature_contract.version?
     → YES  rebuild: retrain from shipped base + retained set (D-003),
             write local artifact + manifest with:
               provenance.base_version   = shipped_base.version
               provenance.n_local_extensions = 1
               training.retained_generation  = retained.generation
     → NO   reset: discard retained set, increment generation,
             serve shipped base unextended, record
               provenance.reset_reason = "contract_version_changed"
```

**Atomicity** (FR-019): the rebuilt artifact and manifest are written to temporary paths and moved into place only on success. A failure at any point leaves the previous local artifact and manifest untouched and servable.

---

## Vector Construction

Replaces the current silent-default construction at `trainer.py:126/175` and `cloud_trainer.py:377/431`.

| Today | After |
|---|---|
| `[features.get(f, 0.0) for f in FEATURE_NAMES]` | `[features[f] for f in contract.names]`, after contract validation has established that every name is present |

The strict lookup is safe precisely because validation ran first — a `KeyError` here would mean validation was skipped, which is a defect rather than a data condition (FR-007).
