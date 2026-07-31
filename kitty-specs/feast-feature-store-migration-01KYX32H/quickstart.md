# Quickstart: Feast Feature Store Migration

**Date**: 2026-07-31 | **Plan**: [plan.md](./plan.md)

Verification in the order a reviewer should check it. The first two are the ones that decide whether this migration is acceptable at all.

## Setup

```bash
pip install -e ".[dev]"            # local: brings feast[sqlite]
pip install -e ".[dev,cloud]"      # cloud paths: brings feast[postgres]
pytest tests/ -v
```

---

## 1. No egress from the local install (SC-002, FR-004) — **the hard requirement**

The structural form. An allow-list of known upload calls is not sufficient; the assertion must be that *no socket is opened at all*.

```python
def test_local_flow_opens_no_socket(monkeypatch):
    opened = []
    real = socket.socket
    monkeypatch.setattr(socket, "socket", lambda *a, **k: opened.append(a) or real(*a, **k))

    store = local_feature_store()
    store.push("stuck_features", vector_df)
    resolve_features("stuck", task_id)        # full serving path
    assert opened == []
```

Then the configuration check:

```bash
grep -iE "http|://|host|port|password|token|endpoint" src/sigil_ml/feature_store/feature_store.local.yaml
# must return nothing
```

And end to end, with the network actually gone:

```bash
# run the suite with outbound blocked; every local test must still pass
```

## 2. The frozen binary still ships (SC-004, FR-011, FR-012)

The highest-risk item — 338 additional native libraries need signing and stapling.

```bash
make freeze                       # or the project's freeze target
codesign --verify --deep --strict dist/kameas-ml/
xcrun stapler validate dist/kameas-ml/
```

Then on a **clean machine**, from the notarized artifact:

```bash
./kameas-ml serve &
curl -sf http://127.0.0.1:7774/health
curl -sf -X POST http://127.0.0.1:7774/predict/stuck -d '{"task_id":"..."}'
```

Two failure modes to probe specifically:

- **Dynamic imports.** Feast resolves providers and stores by name. A binary that builds cleanly and fails on first feature call means `hiddenimports` is incomplete — test the actual call path, not just startup.
- **Read-only app dir.** Confirm nothing attempts to write inside the bundle (FR-013): run with the app directory mounted read-only and assert clean operation.

Record the bundle size delta against NFR-001's 350 MB ceiling.

## 3. Values did not change (SC-006, C-005) — the regression gate

This migration is behaviour-preserving. Capture vectors from `main` before the change, commit them as literals, and assert equality after.

```python
def test_migration_preserves_feature_values():
    with frozen_clock(NOW):
        assert resolve_features("stuck", TASK_A) == PRE_MIGRATION_STUCK_TASK_A
```

Also assert ordering, since both trainers index positionally:

```python
assert list(v.keys()) == stuck_model.FEATURE_NAMES
```

## 4. Point-in-time correctness in cloud (SC-003, FR-007, FR-008)

```python
entity_df = pd.DataFrame({"task_id": [...], "event_timestamp": [...], "label": [...]})
df = store.get_historical_features(entity_df=entity_df, features=stuck_service).to_df()
```

Assert each row reflects its own timestamp, and — the leakage check — that a value written with a *later* `event_timestamp` does not appear:

```python
materialize(task_id="t1", event_timestamp=T0, values={"edit_velocity": 1.0})
materialize(task_id="t1", event_timestamp=T0 + 3600_000, values={"edit_velocity": 9.0})
row = retrieve(task_id="t1", event_timestamp=T0 + 60_000)
assert row["edit_velocity"] == 1.0        # not 9.0
```

## 5. Serving is live, not stale (SC / FR-014, D-003)

```python
push_stale_value("stuck_features", task_id, {"edit_velocity": 99.0})
generate_live_events(task_id)
v = resolve_features("stuck", task_id)
assert v["edit_velocity"] != 99.0        # live compute won, store did not
```

Assert the push happened, and that a **failing** push does not fail the prediction.

## 6. Both deployments agree (SC-005, FR-015)

Feed identical task and event data through the local and cloud paths; assert vectors are identical, key order included.

## 7. Registry (FR-013, FR-016)

- The shipped registry is read-only and no runtime path writes it
- A registry produced by a different Feast version is refused with a diagnostic, not a traceback
- Definitions in the registry match `FEATURE_NAMES` in the model modules exactly

## 8. Budgets

- **NFR-002** serving resolution within 20% of pre-migration median, 1000 resolutions
- **NFR-003** cold start under 10s on the frozen binary — `import feast` pulls `pyarrow` and `grpcio`, so measure the real binary, not a source checkout
- **NFR-001** bundle growth within 350 MB

---

## What this migration deliberately does not do

- **Change any feature's value or meaning.** The event-vocabulary problems already identified are a separate mission.
- **Move computation into Feast.** `sigil_ml.features` stays the arithmetic (D-002).
- **Touch MLflow or model artifacts** — that is `model-registry-and-base-refresh`.
- **Add any cloud connectivity to the local deployment.**

## Rollback

Revert the commit and rebuild. The local online store and cloud `ml_features` become orphaned data — inert, since nothing else reads them. Feature computation is unchanged throughout, so no model retraining is required either way.
