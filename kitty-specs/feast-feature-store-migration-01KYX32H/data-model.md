# Phase 1 Data Model: Feast Feature Store Migration

**Date**: 2026-07-31 | **Plan**: [plan.md](./plan.md)

Feature *values* are unchanged (C-005). This describes where definitions live, how they are stored, and how the two deployments differ.

---

## Deployment configuration

Two `feature_store.yaml` files, selected by `config.operating_mode()`.

**Local** — self-contained, no network path (C-001, FR-004, FR-005):

```yaml
project: sigil_ml
provider: local
registry:
  registry_type: file
  path: <bundle>/feature_store/registry.db     # shipped, read-only (D-001)
online_store:
  type: sqlite
  path: ~/.local/share/sigild/feast_online.db  # writable
# No offline_store. No remote registry. No remote provider.
```

**Cloud** — PostgreSQL for both halves:

```yaml
project: sigil_ml
provider: local
registry:
  registry_type: sql
online_store:
  type: postgres
offline_store:
  type: postgres
```

**Invariant**: the local file must contain no host, URL, port, or credential of any kind. This is asserted, not reviewed.

---

## Entities

| Entity | Join key | Notes |
|---|---|---|
| `task` | `task_id` | Primary entity for stuck and duration features |
| `node` | `node_id` | Fleet-level features |
| `tenant` | `tenant_id` | Cloud only; absent from the local registry |

---

## Feature views

One per model family that declares a `FEATURE_NAMES` constant, mirroring it exactly — ordering included, since both trainers index positionally.

| Feature view | Entity | Features | Source |
|---|---|---|---|
| `stuck_features` | `task` | 6 — matches `models/stuck.py:FEATURE_NAMES` | Push (local) / `ml_features` (cloud) |
| `duration_features` | `task` | 4 — matches `models/duration.py:FEATURE_NAMES` | Push (local) / `ml_features` (cloud) |

**Only these two are registrable today** (corrected during WP01). `stuck.py` and `duration.py` are the only model modules declaring a `FEATURE_NAMES` constant. `workflow.py` builds its vector from `sorted(features.keys())` at predict time (`workflow.py:164`), `activity.py` likewise, `quality.py` is a rules scorer reading keys ad hoc, and the `fleet_*` models take fixed positional rows. Registering any of them would mean retyping feature names as literals — the second source of truth FR-001 exists to prevent.

Those models therefore need a `FEATURE_NAMES` constant *before* they can be registered. That is deliberately separate work: `sorted(features.keys())` means their current vector layout is alphabetical and data-dependent, so freezing it into a constant is a behaviour-affecting change needing its own verification. A conformance test fails the moment any model gains a constant without a matching view, so none can ship unregistered.

Each carries a TTL bounding how far back a point-in-time lookup will reach.

**Feature services** — one per model, naming exactly the features that model consumes. This is the versioned contract a training run records (FR-010).

---

## Registry

| | Local | Cloud |
|---|---|---|
| Form | Protobuf file, `registry.db` | SQL registry |
| Location | Inside the signed bundle | Postgres |
| Writable at runtime | **No** (FR-013) | Yes |
| Produced by | `feast apply` at **build time** (D-001) | Deploy-time apply |

The registry records the Feast version that produced it. A runtime/registry version mismatch is refused with a diagnostic (FR-016, D-007), never surfaced as a deserialization traceback.

---

## Offline store — `ml_features` (cloud only)

Python-owned; no change to Go-owned tables (C-004).

```sql
CREATE TABLE ml_features (
    entity_type      TEXT    NOT NULL,   -- 'task' | 'node'
    entity_id        TEXT    NOT NULL,
    feature_view     TEXT    NOT NULL,
    event_timestamp  TIMESTAMPTZ NOT NULL,  -- the moment the values DESCRIBE
    created_at       TIMESTAMPTZ NOT NULL,  -- when the row was written
    feature_values   JSONB   NOT NULL,
    PRIMARY KEY (entity_type, entity_id, feature_view, event_timestamp)
);
CREATE INDEX ml_features_pit ON ml_features (entity_type, entity_id, feature_view, event_timestamp DESC);
```

**`event_timestamp` is the load-bearing column.** It carries the reference time from `feature-extraction-correctness` — `completed_at` for a historical task, current time for a live one — never the write time. FR-009 depends on this, and FR-007/FR-008 are unachievable without it. `created_at` exists only for auditing and must never be used for retrieval ordering.

---

## Serving flow (D-003)

Active-task predictions compute live; the online store is a byproduct.

```
prediction request for an ACTIVE task
   → sigil_ml.features computes the vector (as_of_ms = None → now)
   → prediction returned from that vector          ← never read from the online store
   → vector pushed to the online store asynchronously
```

The push is best-effort: a failed push logs and does not fail the prediction. Reading the online store for an active task is prohibited — it would serve a value stale by up to one poll interval on the exact path where freshness is the product.

Historical and non-active entities may read the online store normally.

---

## Materialization flow (cloud)

```
completed tasks in Postgres
   → sigil_ml.features computes vectors with per-example as_of_ms
   → rows written to ml_features with event_timestamp = that as_of_ms
   → Feast materializes ml_features → online store
```

Runs as a scheduled or triggered job, never synchronously inside a training run.

---

## Training retrieval (cloud)

```
entity_df: task_id + event_timestamp + label, one row per example
   → get_historical_features(entity_df, feature_service)
   → each row receives values as of ITS OWN event_timestamp, within TTL
```

This is what replaces replaying extractors over history. FR-008 — no value recorded after a row's timestamp may appear in it — is a property of the as-of join, and must be asserted rather than assumed.

---

## What does not change

- `sigil_ml.features` — still the arithmetic (D-002)
- Feature names, ordering, count, and computed values (C-005)
- `FEATURE_NAMES` constants in the model modules
- `DataStore`, `store_sqlite.py`, `store_postgres.py`
- Model names, `:7774` endpoints, Go-owned tables (FR-018)
- Local training's retrieval path — it computes directly, as today
