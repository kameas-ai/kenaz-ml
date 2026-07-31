# Phase 1 Data Model: Feature Extraction Correctness

**Date**: 2026-07-31 | **Plan**: [plan.md](./plan.md)

No persistence changes. No schema migrations. This document describes the in-memory values the extractors consume and produce, and the semantics the feature attaches to them.

---

## Reference Time

The instant a feature vector describes. Introduced by this feature as an explicit value where it was previously implicit in `time.time()`.

| Property | Value |
|---|---|
| Representation | `int` — epoch milliseconds, matching `events.ts` and `tasks.started_at` |
| Parameter | `as_of_ms`, keyword-only, default `None` |
| `None` semantics | Current wall clock — `int(time.time() * 1000)` |
| Serving value | `None` (the subject is an active task; now is correct) |
| Training value | Task `completed_at`, falling back to `last_active`; example skipped if neither exists |

**Invariant**: for a fixed task and a fixed `as_of_ms`, extraction is deterministic. Repeated calls at any later time return identical vectors. This is the property SC-001 tests.

---

## Task Record (consumed, read-only)

Fields the extractors read from a `tasks` row. Ownership is unchanged — Go writes, Python reads.

| Field | Type | Used for |
|---|---|---|
| `started_at` | epoch ms | Session length; phase-start fallback; time-of-day derivation |
| `last_active` | epoch ms | Session length; reference-time fallback |
| `completed_at` | epoch ms \| null | Primary reference time for training examples |
| `test_fails` | int | `test_failure_count` |
| `files` | JSON object or string | `file_count` |
| `branch` | string | `branch_name_length` |

---

## Event Record (consumed, read-only)

Fields read from an `events` row within a task's window.

| Field | Type | Used for |
|---|---|---|
| `ts` | epoch ms | **Lookahead filter** (D-001); phase-start detection; commit recency |
| `kind` | string | Edit / commit / terminal / phase-change classification |
| `payload` | JSON object or string | File path extraction; exit codes |

**Filter rule (FR-003)**: events with `ts > as_of_ms` are excluded before any aggregation. The boundary is inclusive — an event with `ts == as_of_ms` is retained.

---

## Feature Vectors (produced)

Names, ordering, and count are **frozen** by C-005. This feature changes values only.

### Stuck feature vector

Ordering is defined by `FEATURE_NAMES` in `src/sigil_ml/models/stuck.py` and consumed positionally by both trainers.

| Feature | Type | Reference-time dependent |
|---|---|---|
| `test_failure_count` | float | No |
| `time_in_phase_sec` | float | **Yes** — the primary defect |
| `edit_velocity` | float | No (window-relative) |
| `file_switch_rate` | float | No (ratio) |
| `session_length_sec` | float | No (`last_active − started_at`, both historical) |
| `time_since_last_commit_sec` | float | **Yes** — the secondary defect |

### Duration feature vector

Ordering defined by `FEATURE_NAMES` in `src/sigil_ml/models/duration.py`.

| Feature | Type | Reference-time dependent |
|---|---|---|
| `file_count` | float | No |
| `total_edits` | float | No |
| `time_of_day_hour` | float | No (derived from `started_at`) |
| `branch_name_length` | float | No |

Duration features are unaffected in value by this change, but the extractor still accepts `as_of_ms` for signature uniformity and for the lookahead filter, which does affect `total_edits` when events exist past the reference time.

### Other extractor families

| Family | Reference-time dependent | Notes |
|---|---|---|
| Activity (`extract_activity_features`) | No | Per-event, no clock reference. Unchanged. |
| Workflow (`extract_workflow_features`) | No | Window-relative; timestamps compared to each other, not to now. |
| Buffer (`extract_features_from_buffer`) | **Yes** | Uses wall clock; gains `as_of_ms` per FR-010. Serving-only today. |

---

## Empty-Vector Contract

Preserved exactly as it exists today. When a task cannot be resolved, the store-backed extractor returns the documented all-zero dictionary rather than raising — with `time_of_day_hour` remaining the one non-zero default in the duration family. The delegation refactor must not alter these values, since callers build vectors positionally and a shape change would silently corrupt training rows.

---

## Derived Value Semantics

| Computation | Rule |
|---|---|
| Any elapsed duration | `max(0.0, (as_of − since) / 1000.0)` via the shared `_elapsed_sec()` helper (FR-004) |
| Phase start | Latest `phase_change` event at or before the reference time; falls back to `started_at` |
| Last commit | Latest commit-kind event at or before the reference time; falls back to session length when absent |
| Edit velocity | Edit count over elapsed window minutes, floor 1.0 minute — unchanged |
