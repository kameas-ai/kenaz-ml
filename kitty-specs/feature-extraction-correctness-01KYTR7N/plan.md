# Implementation Plan: Feature Extraction Correctness

**Branch**: `main` (planning base and merge target) | **Date**: 2026-07-31 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `kitty-specs/feature-extraction-correctness-01KYTR7N/spec.md`

## Summary

Thread an explicit reference time (`as_of_ms`) through every feature extractor so that elapsed-time features describe the moment being modelled rather than the moment of computation, and collapse the two parallel extractor families into a single definition with the store-backed forms delegating to the data-backed forms.

The serving path keeps current-time semantics by passing nothing; the training paths pass each completed task's own reference time. Feature names, ordering, and count are frozen — this feature changes *when* features are evaluated, never *what* they compute.

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**: `scikit-learn`, `numpy`, `fastapi`, `uvicorn`, `joblib` — no additions (C-001)
**Storage**: SQLite in WAL mode via `SqliteStore` (local); PostgreSQL via `PostgresStore` (cloud). Read-only for this feature; no schema changes and no table-ownership changes (C-002)
**Testing**: `pytest`, primarily `tests/test_features.py`, with a fixed-clock harness for determinism assertions
**Target Platform**: macOS/Linux laptop as a frozen PyInstaller `onedir` binary (local); Linux container (cloud)
**Project Type**: single
**Performance Goals**: serving-path extraction latency within 5% of the pre-change median over 1000 extractions (NFR-002)
**Constraints**: no new runtime dependencies; feature vector layout frozen (C-005); behavior identical across both `DataStore` implementations (C-004)
**Scale/Scope**: one primary module (`src/sigil_ml/features.py`, ~477 lines, 8 extractor entry points), four call-site modules, one test module

## Planning Decisions

Captured before Phase 0, per the planner gate. Full rationale in [research.md](./research.md).

| ID | Decision | Rationale |
|---|---|---|
| D-001 | The no-lookahead filter (`ts > as_of_ms`) is applied in Python inside the data-backed extractors, not pushed into the store query. | Keeps the `DataStore` protocol frozen, so `store_sqlite.py` and `store_postgres.py` are out of the blast radius and C-004 parity is structural rather than something to verify across two SQL dialects. Volume is one task's events; the I/O saving is negligible at this scale. |
| D-002 | `as_of_ms` is keyword-only with default `None`, meaning "current wall clock". | Preserves every existing call signature, so the serving path is unchanged by construction and FR-009 is satisfied without touching behavior. |
| D-003 | Training reference time resolves `completed_at` → `last_active` → skip the example. | A task with neither timestamp cannot yield an honest reference time; silently falling back to wall clock would reintroduce the defect for exactly the rows most likely to be malformed (FR-012). |
| D-004 | Elapsed durations clamp at zero rather than raising when the reference time precedes the measured-from point. | Clock skew and out-of-order events are realistic in observed data; a negative duration is meaningless to the model, and raising would make training brittle against a condition the extractor can reasonably absorb (FR-004). |
| D-005 | No `contracts/` artifacts are generated. | The feature changes no external interface. HTTP request and response shapes in `routes.py` are untouched; `as_of_ms` is an internal parameter never exposed over the wire. |

## Charter Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Skipped — no charter exists.** `spec-kitty charter context --action plan` reports `mode: missing`; `.kittify/charter/charter.md` is absent. No governance gates to evaluate. Project-level constraints from `CLAUDE.md` (dependency ceiling, table ownership, model-name stability, endpoint stability on `:7774`) are carried as C-001 and C-002 in the spec and are respected by this design: no dependencies added, no writes introduced, no model names touched, no endpoint signatures changed.

*Post-Phase 1 re-check*: unchanged. Design introduces no new modules, no new dependencies, and no persistence.

## Project Structure

### Documentation (this feature)

```
kitty-specs/feature-extraction-correctness-01KYTR7N/
├── spec.md                    # Feature specification
├── plan.md                    # This file
├── research.md                # Phase 0 output
├── data-model.md              # Phase 1 output
├── quickstart.md              # Phase 1 output
├── meta.json                  # Mission identity
├── checklists/
│   └── requirements.md        # Spec quality validation
└── tasks.md                   # Phase 2 — created by /spec-kitty.tasks, not here
```

No `contracts/` directory — see D-005.

### Source Code (repository root)

```
src/sigil_ml/
├── features.py                # PRIMARY — all extractor definitions
├── poller.py                  # Serving call site (lines ~141, ~163)
├── routes.py                  # Serving call site (lines ~381, ~478)
├── store.py                   # DataStore protocol — UNCHANGED (D-001)
├── store_sqlite.py            # UNCHANGED (D-001)
├── store_postgres.py          # UNCHANGED (D-001)
└── training/
    ├── trainer.py             # Local training call site (lines ~100, ~133)
    └── cloud_trainer.py       # Cloud training call site (lines ~262, ~289)

tests/
└── test_features.py           # Determinism, path equivalence, edge cases
```

**Structure Decision**: Single-project layout, already established. This feature is a behavior correction confined to `src/sigil_ml/features.py` plus its four call sites; no new modules, packages, or directories are introduced. The `DataStore` layer is deliberately excluded from the change surface by D-001.

## Implementation Shape

The target signature pattern, applied uniformly across the stuck, duration, and buffer extractor families:

```python
def extract_stuck_features_from_data(
    task: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    as_of_ms: int | None = None,
) -> dict[str, float]:
    """Authoritative definition. All feature arithmetic lives here."""
    now_ms = as_of_ms if as_of_ms is not None else int(time.time() * 1000)
    events = [e for e in events if e.get("ts", 0) <= now_ms]   # D-001, FR-003
    ...


def extract_stuck_features(
    store: DataStore,
    task_id: str,
    *,
    as_of_ms: int | None = None,
) -> dict[str, float]:
    """Fetch-and-delegate. Contains no feature arithmetic."""
    task = store.get_task_by_id(task_id)
    if task is None:
        return _empty_stuck_features()
    return extract_stuck_features_from_data(
        task, store.get_events_for_task(task_id), as_of_ms=as_of_ms
    )
```

Elapsed computations route through a shared clamp so FR-004 holds in one place:

```python
def _elapsed_sec(now_ms: int, since_ms: int) -> float:
    return max(0.0, (now_ms - since_ms) / 1000.0)
```

Training call sites resolve the reference time per example (D-003) and skip rows that cannot supply one (FR-012).

## Phase 0 — Research

Complete. See [research.md](./research.md). Four decisions resolved (filter placement, parameter shape, reference-time resolution, negative-duration handling); no `[NEEDS CLARIFICATION]` markers remain.

## Phase 1 — Design

Complete. See [data-model.md](./data-model.md) for the entities and value semantics, and [quickstart.md](./quickstart.md) for the verification path. No contracts generated (D-005).

## Complexity Tracking

*No Charter Check violations — no charter exists, and no project-level constraint from `CLAUDE.md` is violated.*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
