# kenaz-ml Platform Architecture

**Status:** Draft · **Date:** 2026-07-30 · **Scope:** feature layer, model lifecycle, registry, training, local/cloud deployment split

## Overview

`kenaz-ml` ships in two deployments from one codebase:

- **Local** — bundled with the open-source `sigil` product. A frozen, notarized binary on a developer laptop, reading SQLite. No data leaves the machine, ever.
- **Cloud** — an enterprise product. A container in our ecosystem, reading Postgres, serving many tenants.

The models themselves cross that boundary: we build **base models** centrally, ship them with the product, and each local install **extends them on the user's own data**. That single fact drives most of the design below — it means the feature definitions, the vector layout, and the model artifact format are a versioned contract between our build pipeline and every install in the wild.

**Guiding principles:**

- Local-first and private by default. Local training never transmits data or telemetry.
- One definition of every feature, shared by both deployments and by base-model training. (This restates the existing principle in `SIGIL_CLOUD_ARCHITECTURE.md`: "the same ML models and feature extraction logic run locally and in the cloud.")
- Heavyweight tooling is a cloud-side and build-time concern. The local runtime stays on `scikit-learn`, `numpy`, `fastapi`, `uvicorn`, `joblib`.
- Contracts fail loud. A model whose expected features don't match what the runtime produces must error, not silently degrade.
- The database schema and table ownership rules in `CLAUDE.md` are invariant.

---

## 1. Deployment split

| | **Local (OSS)** | **Cloud (Enterprise)** |
|---|---|---|
| Packaging | PyInstaller `onedir`, notarized (spec 069) | Container image |
| Data store | SQLite (`~/.local/share/sigild/data.db`, WAL) | Postgres |
| `DataStore` impl | `SqliteStore` | `PostgresStore` |
| Tenancy | Single user | Multi-tenant |
| Model artifacts | Filesystem, two slots per model | S3 / MinIO, versioned keys |
| Model registry | Sidecar JSON manifests (stdlib) | MLflow (Postgres backend + S3 artifacts) |
| Feature tooling | Python extractors only | Python extractors + Feast (offline half) |
| Training | Local extension of shipped base models | Per-tenant and pooled training |
| Extra deps | none | `kenaz-ml[cloud]` |
| Privacy posture | No egress | Contractual, tenant-isolated |

The split is enforced by optional dependencies: a `cloud` extra carrying `psycopg2-binary`, `boto3`, and `feast[postgres]`. **Feast itself is now an unconditional local dependency** (`feast==0.65.0`) — see the superseding note in §3.5. MLflow remains cloud-only and must not enter the local import path.

---

## 2. Data layer

Unchanged from today. All access goes through the `DataStore` protocol (`src/kenaz_ml/datastore/protocol.py`, imported as `kenaz_ml.datastore`), implemented by `SqliteStore` (`datastore/sqlite.py`) and `PostgresStore` (`datastore/postgres.py`). Table ownership per `CLAUDE.md`: Python reads `events`, `tasks`, `patterns`, `suggestions`; Python writes `ml_predictions`, `ml_events`, `ml_signals`; Python owns `ml_cursor`.

**One addition, cloud only:** an `ml_features` table, Python-owned, holding materialized feature rows for historical training.

```sql
CREATE TABLE ml_features (
    entity_type    TEXT    NOT NULL,   -- 'task' | 'node'
    entity_id      TEXT    NOT NULL,
    feature_view   TEXT    NOT NULL,   -- 'stuck' | 'duration' | ...
    event_timestamp BIGINT NOT NULL,   -- epoch ms; the as-of clock
    contract_version TEXT  NOT NULL,
    features       JSONB   NOT NULL,
    created_at     BIGINT  NOT NULL,
    PRIMARY KEY (entity_type, entity_id, feature_view, event_timestamp)
);
```

`event_timestamp` is the load-bearing column: it is what makes point-in-time-correct training joins possible (§3.2). It must be the time the features *describe*, never the time the row was written.

Local does not get this table. Local training reads events directly and computes features on demand.

---

## 3. Feature layer

### 3.1 Computation authority

`src/kenaz_ml/features.py` is the single authority for feature computation, in both deployments and for base-model training. Nothing else computes features.

Today the module has two parallel families — `extract_stuck_features(store, task_id)` and `extract_stuck_features_from_data(task, events)` — with a docstring promising they produce identical output. That promise is enforced only by hand, and it is the exact skew a feature store is meant to prevent.

**Target:** the `_from_data` form is the definition; the store-backed form becomes a query wrapper.

```python
def extract_stuck_features(store, task_id, *, as_of_ms=None):
    task = store.get_task_by_id(task_id)
    if task is None:
        return _empty_stuck_features()
    return extract_stuck_features_from_data(
        task, store.get_events_for_task(task_id), as_of_ms=as_of_ms
    )
```

### 3.2 Point-in-time correctness

**This is a live bug and it blocks base-model creation.**

The stuck extractor computes elapsed-time features against wall clock (`features.py:93`, `:105`, `:125`):

```python
now_ms = int(time.time() * 1000)
time_in_phase_sec = (now_ms - phase_start) / 1000.0
time_since_last_commit_sec = (now_ms - last_commit_ts) / 1000.0
```

That is correct at **serving** time — `poller.py:141` and `routes.py:381` predict on an *active* task, where "now" is genuinely now.

It is wrong at **training** time. `trainer.py:100` replays the same function over *completed* tasks:

```python
for task_id in task_ids:                      # completed tasks
    features = extract_stuck_features(self.store, task_id)
    x = [features.get(f, 0.0) for f in STUCK_FEATURES]
    stuck = features["test_failure_count"] > 3 and features["time_in_phase_sec"] > 600
```

For a task that finished three months ago, `time_in_phase_sec` is three months. Two of six stuck features measure task age rather than stuckness, and the label degenerates: `time_in_phase_sec > 600` is true for every task older than ten minutes, so the heuristic silently collapses to `test_failure_count > 3`. `cloud_trainer.py:262` has the same defect via `extract_stuck_features_from_data`.

Duration training is unaffected — `time_of_day_hour` derives from `started_at`.

**Fix:** thread an explicit `as_of_ms` through every extractor. Serving passes `None` (meaning now); training passes the example's reference time (`completed_at`, or the window end).

```python
def extract_stuck_features_from_data(task, events, *, as_of_ms=None):
    now_ms = as_of_ms if as_of_ms is not None else int(time.time() * 1000)
```

**Why it blocks base models:** a base model trained today bakes "task age" into a shipped artifact, every OSS install then `warm_start`s on top of it, and the contract version is now committed. Fix before the first base model exists, not after.

### 3.3 The feature contract

Both trainers build vectors positionally:

```python
x = [features.get(f, 0.0) for f in STUCK_FEATURES]   # trainer.py:101, cloud_trainer.py:263
```

`.get(f, 0.0)` is safe today because one process trains and serves. It becomes a silent-corruption vector the moment base models ship: if an artifact was trained with a feature the local extractor no longer emits — renamed, dropped, reordered — the install substitutes `0.0` and keeps predicting. Across a user base on mixed versions that failure is invisible and unreportable.

A **feature contract** is therefore part of every model artifact:

```json
{
  "version": "2",
  "names": ["test_failure_count", "time_in_phase_sec", "edit_velocity",
            "file_switch_rate", "session_length_sec", "time_since_last_commit_sec"],
  "dtypes": ["float64", "float64", "float64", "float64", "float64", "float64"]
}
```

The ordered `names` list *is* the vector layout. On load, the runtime compares it against what the extractors produce and **raises on mismatch**. `.get(f, 0.0)` is replaced by strict lookup after validation.

Contract version bumps on any change to feature names, order, semantics, or scaling. A changed contract invalidates every base model trained against the old one.

### 3.4 Feature discovery

Extend `GET /introspect` (spec 060) to expose the registry: available features, dtypes, producing extractor, consuming models, contract version per loaded model, and any contract mismatch. This is the local discovery surface and requires no dependencies — it reads manifests (§5) and the extractor registry.

### 3.5 Feast: cloud, build-time, offline half only

> **SUPERSEDED 2026-07-31.** The product owner elected to migrate to Feast in **both** deployments, with the frozen binary retained. Local now runs a self-contained Feast — file registry plus SQLite online store — with no network path to cloud. Cloud runs the same shipped definitions over PostgreSQL. See the `feast-feature-store-migration` mission. The rationale below is retained because the reasoning about *where Feast earns its keep* still holds; only the local/cloud boundary changed.

**Original decision: Feast is used in the cloud training pipeline. It is never imported by the local runtime.**

Rationale:

- **Not local.** `feast[sqlite]` pulls pandas, pyarrow, protobuf, and grpcio. In a notarized `onedir` bundle every native library must be signed and stapled, for a feature set a single-user daemon does not use. This is what stalled the `feat/feast-feature-store` branch in April (§10).
- **Offline half, not online half.** Serving predicts on an active task from live events. An online store returns the last *materialized* value; for a "is this developer stuck right now" model, that staleness is a downgrade, not an optimization. The value is in point-in-time-correct historical retrieval, not in key-value serving.
- **Feast orchestrates; it does not compute.** Feature values are produced by `kenaz_ml.features` (§3.1) and written to `ml_features` with correct `event_timestamp`s. Feast registers a `PostgreSQLSource` over that table and provides as-of joins, `FeatureService` versioning per model, and lineage. Computation stays in one place, so both deployments remain bit-identical.

Point-in-time correctness comes from *storing values with their event time*, not from Feast computing them — which is why §3.2 and the `ml_features` schema matter more than the tool choice, and why the cloud pipeline works with or without Feast.

**Reassess if:** the model roster shrinks materially (a hand-written `DISTINCT ON` as-of join is ~15 lines and would suffice for three or four models), or if cloud adds a genuine low-latency batch-serving path that wants an online store.

---

## 4. Model layer

### 4.1 Base models and local extension

```
        ┌─────────────────────────────────────────┐
        │ Central (ours)                          │
        │  curated/synthetic data                 │
        │      → train base model                 │
        │      → MLflow registry → promote        │
        │      → export artifact + manifest       │
        └───────────────────┬─────────────────────┘
                            │ bundled in release
                            ▼
        ┌─────────────────────────────────────────┐
        │ Local install                           │
        │  base/  (pristine, read-only)           │
        │      ↓ extend on user's own data        │
        │  local/ (working, extended)  ← served   │
        └─────────────────────────────────────────┘
```

The pristine base artifact is never overwritten. Extension always writes to the `local/` slot, so a bad extension round is recoverable and the base-vs-local distinction stays answerable.

### 4.2 Extension mechanism per model

Extension is estimator-dependent. Current roster:

| Model | Estimator | Mechanism |
|---|---|---|
| `ActivityClassifier` | `SGDClassifier(loss="log_loss")` | True `partial_fit` — already implemented at `activity.py:228` |
| `StuckPredictor` | `GradientBoostingClassifier` | `warm_start` + appended boosting stages |
| `DurationEstimator` | `GradientBoostingRegressor` | `warm_start` + appended stages |
| `WorkflowStatePredictor` | `GradientBoostingClassifier` | `warm_start` + appended stages |
| `fleet_focus` | `GradientBoostingRegressor` | `warm_start` + appended stages |
| `fleet_meeting` | `RandomForestClassifier` | `warm_start` adds trees — crude; trees see different distributions |
| `fleet_onboarding` | `LinearRegression` | **No incremental path.** Migrate to `SGDRegressor` |
| `PatternDetector` | `IsolationForest` | No incremental path; unsupervised and cheap — retrain locally from scratch |

Gradient boosting `warm_start` is a legitimate mechanism rather than a workaround: bumping `n_estimators` and refitting appends stages fitted to the residuals of the *current* model on the new data, which is precisely "correct the base model using local data."

**Open risk — catastrophic forgetting.** Stages fitted purely on local data optimize only for local data and can degrade base behavior. Standard mitigation is mixing a slice of the shipped base/synthetic set into each local fit with `sample_weight` favoring local data. **This requires a spike on `StuckPredictor` before the architecture is committed** (§8).

### 4.3 Cold start

Today a fresh install with fewer than ten completed tasks trains on synthetic data (`trainer.py:90`, `generate_stuck_data(500)`). Once base models ship, cold start becomes "serve the base model" and synthetic generation is demoted to a fallback for models with no base artifact yet.

---

## 5. Model registry

### 5.1 What exists

Everything below lives in `src/kenaz_ml/modelstore/` and is imported from the package (`from kenaz_ml.modelstore import ...`), not from its submodules.

- `LocalModelStore` (`modelstore/stores.py`) — `{models_dir}/{name}.joblib`. **No versioning**; `save()` overwrites in place.
- `S3ModelStore` (`modelstore/stores.py`) — versioned keys `{tenant}/models/{name}/{version}/model.joblib` plus a `latest` pointer. **No `list_versions`, no `set_latest`** — `save()` only advances, so there is no rollback.
- `FilesystemModelLoader` (`modelstore/loader.py`) — resolves tenant-specific, then shared fallback.
- `last_modified()` (`modelstore/stores.py`) — file mtime, used by `/introspect` as an explicitly acknowledged proxy for "last trained" because no metadata exists.
- `ModelCache` (`modelstore/cache.py`) — in-memory TTL/LRU of loaded model objects.

`ModelStore` remains a bytes-in/bytes-out protocol. The registry is a thin layer above it; existing call sites are undisturbed.

### 5.2 The manifest — the interface between deployments

One schema, written by the cloud export job and by local training, read by both.

```json
{
  "name": "stuck",
  "version": "4",
  "created_at": 1753900000000,
  "provenance": {
    "base_version": "1",
    "base_sha256": "9f2c…",
    "n_local_extensions": 3,
    "training_source": "local"
  },
  "runtime": {
    "estimator": "GradientBoostingClassifier",
    "sklearn_version": "1.5.2",
    "python_version": "3.12"
  },
  "feature_contract": {
    "version": "2",
    "names": ["test_failure_count", "…"],
    "dtypes": ["float64", "…"]
  },
  "training": { "n_samples": 847, "as_of_ms": 1753900000000 },
  "metrics": { "accuracy": 0.81, "n_holdout": 120 },
  "artifact_sha256": "4a71…"
}
```

`training_source` is one of `base` | `synthetic` | `local`. `provenance` answers "is this user running our base or their own, and how far has it drifted" — currently unanswerable.

### 5.3 Local registry

Sidecar JSON next to each artifact, stdlib only:

```
{models_dir}/
  base/stuck.joblib    base/stuck.json      # shipped; never written locally
  local/stuck.joblib   local/stuck.json     # working copy; served
```

Sidecars rather than a SQLite table, deliberately: the artifact is self-describing and provenance travels with it if copied; it survives a `data.db` reset and does not depend on `sigild` running; and it is human-readable, which is worth real weight in an open-source privacy story — the user can open the file and see exactly what is on their machine. At current model counts, globbing the directory is free.

Resolution order at load: `local/{name}` → `base/{name}` → synthetic cold start.

### 5.4 Cloud registry — MLflow

MLflow lands on infrastructure that already exists: Postgres as backend store, S3/MinIO as artifact store. It provides versions, aliases and stages for promotion **and rollback** (the capability `S3ModelStore.save()` lacks), and experiment tracking for the manually built base models. Its dependency weight is irrelevant in a container.

### 5.5 The release seam

```
central training → MLflow run → register version → promote alias "base"
                 → export job: artifact + manifest JSON
                 → bundled into the OSS release
                 → local install reads manifest; MLflow never imported locally
```

The manifest schema is the entire contract between the two halves. Design it once; both sides implement against it.

---

## 6. Training pipelines

**Local.** Triggered on a cadence or on completed-task count. Loads `local/` (else `base/`), validates the feature contract, extracts features with `as_of_ms` set per example, extends via the §4.2 mechanism, evaluates against a held-out local slice, and writes `local/` plus manifest **only if the evaluation does not regress** against the previous local version. That guard is the practical defense against forgetting.

**Cloud.** Per-tenant and pooled (`get_opted_in_tenant_ids()`). Materializes features into `ml_features` with correct event timestamps, retrieves training sets via point-in-time joins, trains, logs to MLflow, promotes.

**Base.** Manual, ours, central. Curated and synthetic data, evaluated deliberately, promoted by hand, exported to the release. Base models do not exist yet — §8 sequences what must land first.

---

## 7. Security and privacy

- **No local egress.** Local training and inference transmit nothing. This is a product guarantee, not a default.
- **Pickle deserialization.** `joblib.load` on a shipped artifact is arbitrary code execution. Verify `artifact_sha256` against the manifest **before** loading; sign base artifacts in the release pipeline. This is cheap once manifests exist and is consistent with the security-first posture in `CLAUDE.md`.
- **sklearn pickle portability.** Base models are pickles that installs then `warm_start`. The frozen binary pins sklearn, so that path is controlled; a `pip install kenaz-ml` user resolves whatever satisfies the version range, and sklearn pickles are not guaranteed portable across versions. Record `sklearn_version` in the manifest, check at load, and pin sklearn tightly for distributions that consume base models.
- **Tenant isolation (cloud).** Per-tenant key prefixes and per-tenant model resolution already exist; pooled training is restricted to explicitly opted-in tenants.

---

## 8. Sequencing

Ordered by what blocks what.

1. **Point-in-time fix** — `as_of_ms` through all extractors; training passes reference time. *Blocks base-model creation (§3.2).*
2. **Collapse the extractor twins** — `_from_data` becomes the definition. *Blocks contract stability.*
3. **Manifest schema** — the interface everything else keys off (§5.2).
4. **Local registry + contract validation** — two-slot layout, strict feature lookup, sha256 verification. *Blocks safely shipping base models.*
5. **`warm_start` forgetting spike** — validate on `StuckPredictor`; decide mitigation. *Blocks the extension architecture.*
6. **Migrate `fleet_onboarding`** to `SGDRegressor`.
7. **Build base models** — manual, central, once 1–5 hold.
8. **Cloud: `ml_features` + MLflow**, then Feast when convenient.
9. **`/introspect` feature and model discovery.**

Items 1–2 are worth doing regardless of whether the rest of this document is adopted; they fix a live defect in both deployments.

---

## 9. Open decisions

- **Forgetting mitigation** (§4.2) — pending the spike. If `warm_start` on local-only data proves too destructive, the fallback is shipping a base *dataset* alongside the base model and retraining locally on base + local with weighting. At current data volumes a full GBM refit is cheap.
- **Base-model refresh policy** — when release N+1 ships base v2, does a locally-extended model reset to v2, or persist? Manifest provenance makes either policy implementable; the product call is unmade.
- **Contract version granularity** — one global contract version, or one per feature view. Per-view is finer but multiplies bookkeeping.
- **MLflow deployment shape** — tracking server versus direct backend-store access from the training job.

---

## 10. Rejected alternatives

**Feast in the local runtime.** Attempted on `feat/feast-feature-store` (four commits, tip 2026-04-10, never merged). The branch registered `PushSource` + `FeatureView` against a SQLite online store, with computation left in `kenaz_ml.features`. Its own docstring records the mismatch: placeholder `FileSource` entries pointing at non-existent parquet paths, "because sigil-ml never runs offline batch materialization." With only the push path wired, Feast reduced to a TTL'd key-value cache in front of functions that already existed. The final commit demoted Feast from a required dependency to an extra, and the branch stopped. **Superseded by §3.5** — the offline half in cloud is the half with value.

**Feast with no online store, resolving from posted JSON.** Achievable via `OnDemandFeatureView` over a `RequestSource`, and closer to the right shape than the branch. Rejected because the dependency cost is unchanged (importing Feast pulls the full tree regardless of storage config), the registry remains mandatory, and `RequestSource` schemas are flat typed columns while our inputs are variable-length event sequences — the event list would pass as an opaque JSON string, so Feast's typing would validate nothing. Net: Feast with both of its hard features disabled, at full cost, wrapping extractors that already have exactly this signature.

**MLflow locally.** Correct tool, wrong deployment. Its dependency weight is a non-issue in a container and disqualifying in a notarized bundle.

**Model registry as a SQLite table.** Viable, and consistent with the `ml_cursor` ownership pattern. Rejected for local in favor of sidecars (§5.3) because artifacts should be self-describing and independent of the daemon's database.
