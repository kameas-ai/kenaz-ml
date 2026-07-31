# Quickstart: Feature Extraction Correctness

**Date**: 2026-07-31 | **Plan**: [plan.md](./plan.md)

How to verify this feature behaves correctly, in the order a reviewer should check it.

## Setup

```bash
pip install -e ".[dev]"
pytest tests/test_features.py -v
```

No new dependencies, no migrations, no service restart required — the change is confined to pure functions and their call sites.

---

## 1. Determinism (SC-001, the core property)

The defining test: a fixed historical task must extract identically no matter when extraction runs.

```python
task = {"started_at": T0, "completed_at": T0 + 3_600_000, "last_active": T0 + 3_600_000, "test_fails": 5}
events = [...]                      # fixed event list

a = extract_stuck_features_from_data(task, events, as_of_ms=task["completed_at"])
b = extract_stuck_features_from_data(task, events, as_of_ms=task["completed_at"])
assert a == b                       # trivially, but also:

# ...and identical when the wall clock has moved months on.
with frozen_clock(T0 + 90 * 86_400_000):
    c = extract_stuck_features_from_data(task, events, as_of_ms=task["completed_at"])
assert a == c
```

Before the change, `c` diverges from `a` on `time_in_phase_sec` and `time_since_last_commit_sec` by roughly the elapsed wall-clock gap. That divergence is the bug; its absence is the fix.

## 2. Serving non-regression (SC-003)

With the clock held fixed, omitting `as_of_ms` must reproduce pre-change behavior exactly.

```python
with frozen_clock(NOW):
    assert extract_stuck_features_from_data(task, events) == BASELINE_VECTOR
```

Capture `BASELINE_VECTOR` from the current implementation before making changes.

## 3. Path equivalence (SC-004)

Both extractor forms must agree for every family.

```python
store = FakeStore(task=task, events=events)
assert (extract_stuck_features(store, "task-1", as_of_ms=REF)
        == extract_stuck_features_from_data(task, events, as_of_ms=REF))
```

Repeat for the duration family. A failure here means the delegation refactor left arithmetic behind in the store-backed path.

## 4. No lookahead (FR-003)

Events after the reference time must not contribute.

```python
past  = [e for e in events if e["ts"] <= REF]
assert (extract_stuck_features_from_data(task, events, as_of_ms=REF)
        == extract_stuck_features_from_data(task, past,   as_of_ms=REF))
```

Boundary check: an event at exactly `ts == REF` is **included**.

## 5. Negative clamp (FR-004)

```python
v = extract_stuck_features_from_data(task, events, as_of_ms=task["started_at"] - 10_000)
assert v["time_in_phase_sec"] >= 0.0
assert v["time_since_last_commit_sec"] >= 0.0
```

## 6. Training reference time (FR-007, FR-008, FR-012)

Confirm the trainers supply per-example reference times and skip unusable rows:

```bash
pytest tests/ -k "trainer and as_of" -v
```

A task with neither `completed_at` nor `last_active` must be **absent** from the training matrix — not present with wall-clock features.

## 7. Vector layout unchanged (SC-005)

```python
assert list(extract_stuck_features_from_data(task, events).keys()) == EXPECTED_KEYS
assert len(STUCK_FEATURES) == 6 and len(DURATION_FEATURES) == 4
```

---

## What this feature deliberately does not do

Confirm these remain untouched, since each belongs to a later mission:

- Model artifacts on disk are **not** invalidated or retrained. Existing `.joblib` files were fitted on the old vectors and are stale after this lands; handling that is the model-registry mission's job.
- The stuck label threshold at `trainer.py:104` is **not** retuned. It becomes meaningful once the feature is correct, but remains uncalibrated.
- No feature is added, removed, or renamed.
- `DataStore` and both store implementations are unmodified.

## Rollback

Revert the commit. No state, schema, or artifact changes exist to unwind — though any model retrained after the change will have been fitted on the corrected vectors and should be retrained again if rolling back.
