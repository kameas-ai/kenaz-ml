"""Behaviour-preservation regression: the migration changed no feature value (WP06).

C-005 is the mission's central claim -- feature names, ordering and computed
values are identical before and after; only registration, storage and retrieval
moved. That claim is easy to state and easy to violate quietly. Both models index
their vectors positionally against ``FEATURE_NAMES``
(``[features.get(f, 0.0) for f in FEATURE_NAMES]``), so a rewiring that reordered
a vector would raise nothing at all -- it would just permute every model input.

What this module does
---------------------
It compares the **post-migration** paths against feature values **recorded from
the pre-migration commit**. The baselines below are literals, captured by running
the pre-migration extractors under a pinned clock and a pinned timezone and
writing down what came back. They are not computed at test time, and nothing in
the post-migration tree contributed to them. Provenance is on every block, and
:func:`test_baselines_were_recorded_from_the_pre_migration_commit` re-derives them
from the git blob when history is available, so a reader does not have to take the
comments on faith.

Both deployments are covered:

* **Local** -- ``sigil_ml.feature_store.resolve``, the live-compute-then-push
  serving resolver WP03 routed ``poller.py`` and ``routes.py`` through. Fully
  exercised: real extractors, real resolver, real push into a recording store.
* **Cloud** -- ``sigil_ml.feature_store.materialize`` plus
  ``CloudTrainer._retrieve_offline_features``. Real materializer, real row
  objects, real production ``cloud_feature_views`` binding, real Feast
  ``get_historical_features`` as-of join, real retrieval verification. The one
  approximation is physical storage: no PostgreSQL server exists in this
  environment, so the offline rows are projected through the production
  ``rows_to_source_frame`` (which shares its column list with the production SQL)
  into parquet behind Feast's Dask offline store. This is exactly the harness
  ``tests/test_feature_store_cloud.py`` established for WP04, deliberately reused
  rather than reinvented. The JSONB serialization the PostgreSQL sink performs is
  covered separately and exactly by
  :func:`test_the_postgres_sink_serializes_every_baseline_value_without_loss`.

Why the clock and the timezone are pinned
-----------------------------------------
Three stuck features are elapsed durations measured against the reference time,
and ``duration``'s ``time_of_day_hour`` comes from ``time.localtime()`` -- so it
is a function of the process timezone, and in the empty-vector case a function of
the wall clock as well. Without pinning both, the baselines would be
irreproducible and this module would assert nothing. The sibling mission found
this the hard way.

Why exact equality
------------------
Every pre-migration value here was exact, so every assertion here is exact. A
value that moved is a finding to report, not a tolerance to widen -- if one
legitimately has to change, it belongs in the spec as an amendment, not absorbed
into ``pytest.approx`` here.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from sigil_ml.feature_store import materialize
from sigil_ml.feature_store import resolve as resolve_module
from sigil_ml.feature_store.materialize import (
    DURATION_OFFLINE_VIEW,
    ML_FEATURES_TABLE,
    OFFLINE_FEATURE_VIEWS,
    STUCK_OFFLINE_VIEW,
    build_offline_rows,
    cloud_feature_views,
    rows_to_source_frame,
)
from sigil_ml.models.duration import FEATURE_NAMES as DURATION_FEATURE_NAMES
from sigil_ml.models.stuck import FEATURE_NAMES as STUCK_FEATURE_NAMES
from sigil_ml.training.cloud_trainer import CloudTrainer
from sigil_ml.training.models import CloudTrainingConfig

pd = pytest.importorskip("pandas")

# ===========================================================================
# The pinned conditions the baselines were captured under
# ===========================================================================

#: The commit the baselines were recorded from. It is the merge-base of this
#: lane: the last commit before any feature-store code existed. ``git show
#: PRE_MIGRATION_SHA:src/sigil_ml/features.py`` is the module that produced every
#: literal in the next section.
PRE_MIGRATION_SHA = "ef67e0539feaa914dbd0c39b92500474fdd92b78"

#: sha256 of that blob, so a reader can confirm which source text was measured
#: without trusting the sha alone.
PRE_MIGRATION_FEATURES_SHA256 = "2b7fd419583068944ea195ae0b2bfeb6b6be84829dec738ffcc5d322c556bec5"

PINNED_TZ = "UTC"
PINNED_NOW = datetime(2026, 3, 17, 14, 30, 0, tzinfo=timezone.utc)
NOW_MS = int(PINNED_NOW.timestamp() * 1000)  # 1773757800000
NOW_S = NOW_MS / 1000.0

MINUTE_MS = 60_000

#: The materialization write clock, six months after the tasks. Deliberately far
#: away: if a value ever started depending on write time rather than on the moment
#: it describes, the failure is half a year wide, not a few milliseconds.
WRITE_TIME = datetime(2026, 9, 21, 8, 0, 0, tzinfo=timezone.utc)
WRITE_TIME_MS = int(WRITE_TIME.timestamp() * 1000)


class PinnedClock:
    """``time`` frozen at :data:`NOW_S`, with ``localtime`` following it.

    Patched over ``sigil_ml.features.time`` -- the single authority for feature
    computation -- so both deployments read the same instant. ``as_of_ms`` is
    still ``None`` on the serving path, still meaning "now"; this only fixes
    which "now" that is.
    """

    def __init__(self, real: Any) -> None:
        self._real = real

    def time(self) -> float:
        return NOW_S

    def localtime(self, secs: float | None = None) -> Any:
        return self._real.localtime(NOW_S if secs is None else secs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


@pytest.fixture(autouse=True)
def pinned_clock_and_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduce the exact conditions the baselines were captured under.

    Both halves matter. The clock fixes the elapsed features and the empty
    duration vector's ``time_of_day_hour``; the timezone fixes
    ``time.localtime(started_at)``, which is otherwise a function of whichever
    machine the suite runs on.
    """
    import sigil_ml.features as features_module

    previous_tz = os.environ.get("TZ")
    os.environ["TZ"] = PINNED_TZ
    time.tzset()

    monkeypatch.setattr(features_module, "time", PinnedClock(time))
    yield

    if previous_tz is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = previous_tz
    time.tzset()


@pytest.fixture(autouse=True)
def isolated_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep model weights out of the developer's real data directory."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))


@pytest.fixture(autouse=True)
def clean_resolver() -> Any:
    """Leave no push worker state, cached store or counter behind."""
    resolve_module.use_feature_store(None)
    resolve_module.reset_push_state()
    yield
    resolve_module.flush_pushes()
    resolve_module.use_feature_store(None)
    resolve_module.reset_push_state()


# ===========================================================================
# Fixtures -- byte-for-byte what the pre-migration capture ran against
# ===========================================================================
#
# These records are part of the baseline's provenance: a literal is only
# meaningful alongside the input that produced it. Editing anything below
# invalidates every literal in the next section, and
# `test_baselines_were_recorded_from_the_pre_migration_commit` will say so.

#: A rich, fully populated task. ``completed_at`` equals the pinned wall clock on
#: purpose -- that makes the cloud path's reference time (``completed_at``) and
#: the local path's reference time (now) the *same instant*, so T025 can attribute
#: any divergence to the deployment rather than to timing.
TASK_A: dict[str, Any] = {
    "id": "task-alpha",
    "started_at": NOW_MS - 90 * MINUTE_MS,
    "last_active": NOW_MS - 90_000,
    "completed_at": NOW_MS,
    "test_fails": 4,
    "phase": "implementing",
    "branch": "feat/behaviour-preservation",
    "files": '{"alpha.py": 7, "beta.py": 3, "gamma.py": 2, "delta.py": 1, "epsilon.py": 5}',
}

#: Events 11 and 12 sit *after* the reference time. The no-lookahead filter must
#: drop them, so their presence is itself an assertion: if the migration ever
#: stopped filtering, `total_edits` and `time_since_last_commit_sec` would move.
EVENTS_A: list[dict[str, Any]] = [
    {"id": 1, "ts": NOW_MS - 88 * MINUTE_MS, "kind": "edit", "payload": {"file": "alpha.py"}},
    {"id": 2, "ts": NOW_MS - 70 * MINUTE_MS, "kind": "edit", "payload": {"file": "beta.py"}},
    {"id": 3, "ts": NOW_MS - 45 * MINUTE_MS, "kind": "commit", "payload": {"sha": "abc123"}},
    {"id": 4, "ts": NOW_MS - 40 * MINUTE_MS, "kind": "file_edit", "payload": {"file": "gamma.py"}},
    {"id": 5, "ts": NOW_MS - 22 * MINUTE_MS, "kind": "phase_change", "payload": {"phase": "debugging"}},
    {"id": 6, "ts": NOW_MS - 18 * MINUTE_MS, "kind": "save", "payload": {"file": "alpha.py"}},
    {"id": 7, "ts": NOW_MS - 12 * MINUTE_MS, "kind": "edit", "payload": {"file": "beta.py"}},
    {"id": 8, "ts": NOW_MS - 8 * MINUTE_MS, "kind": "edit", "payload": {"file": "alpha.py"}},
    {"id": 9, "ts": NOW_MS - 3 * MINUTE_MS, "kind": "terminal", "payload": {"cmd": "pytest", "exit_code": 1}},
    {"id": 10, "ts": NOW_MS - 2 * MINUTE_MS, "kind": "edit", "payload": {"file": "delta.py"}},
    {"id": 11, "ts": NOW_MS + 5 * MINUTE_MS, "kind": "edit", "payload": {"file": "omega.py"}},
    {"id": 12, "ts": NOW_MS + 6 * MINUTE_MS, "kind": "commit", "payload": {"sha": "def456"}},
]

#: The empty-input case: a real task with no events at all, an empty files map and
#: an empty branch. Every aggregate falls back to its default, which is exactly
#: where a rewiring most easily changes a value without anyone noticing.
TASK_B: dict[str, Any] = {
    "id": "task-beta",
    "started_at": NOW_MS - 200 * MINUTE_MS,
    "last_active": NOW_MS - 17 * MINUTE_MS,
    "completed_at": NOW_MS,
    "test_fails": 0,
    "phase": "planning",
    "branch": "",
    "files": "{}",
}
EVENTS_B: list[dict[str, Any]] = []

#: The id the store has never heard of -- the missing-entity case.
MISSING_TASK_ID = "task-does-not-exist"

#: The raw daemon-stream buffer the poller falls back to when there is no active
#: task. A different event vocabulary from the task-window stream: `file`/`edit`
#: for edits, `git` (not `commit`) for commits, `terminal` exit codes for test
#: failures. Event 8 is after the reference time and must be dropped.
BUFFER: list[dict[str, Any]] = [
    {"id": 1, "ts": NOW_MS - 27 * MINUTE_MS, "kind": "file", "payload": {"path": "/src/one.py"}},
    {"id": 2, "ts": NOW_MS - 21 * MINUTE_MS, "kind": "terminal", "payload": {"cmd": "pytest", "exit_code": 1}},
    {"id": 3, "ts": NOW_MS - 19 * MINUTE_MS, "kind": "file", "payload": {"path": "/src/two.py"}},
    {"id": 4, "ts": NOW_MS - 14 * MINUTE_MS, "kind": "git", "payload": {"branch": "main"}},
    {"id": 5, "ts": NOW_MS - 11 * MINUTE_MS, "kind": "edit", "payload": {"path": "/src/one.py"}},
    {"id": 6, "ts": NOW_MS - 6 * MINUTE_MS, "kind": "terminal", "payload": {"cmd": "pytest", "exit_code": 0}},
    {"id": 7, "ts": NOW_MS - 4 * MINUTE_MS, "kind": "terminal", "payload": {"cmd": "go test", "exit_code": 2}},
    {"id": 8, "ts": NOW_MS + 9 * MINUTE_MS, "kind": "file", "payload": {"path": "/src/three.py"}},
]


# ===========================================================================
# T023 -- pre-migration baselines, recorded not typed
# ===========================================================================
#
# Every dict below was produced by running the extractors in
# `ef67e0539feaa914dbd0c39b92500474fdd92b78:src/sigil_ml/features.py` -- the
# merge-base, before a single line of feature-store code existed -- against the
# fixtures above, with TZ=UTC and the wall clock pinned to
# 1773757800000 (2026-03-17T14:30:00+00:00). The floats are `repr()` output, so
# they carry full double precision rather than a rounded transcription.
#
# These are recorded observations. Nothing post-migration produced them, and
# nothing computes them at test time.

# Captured from ef67e05, fixture TASK_A/EVENTS_A, clock pinned to NOW_MS, TZ=UTC.
# Pre-migration call: extract_stuck_features(store, "task-alpha", as_of_ms=NOW_MS).
PRE_MIGRATION_STUCK_TASK_A = {
    "test_failure_count": 4.0,
    "time_in_phase_sec": 1320.0,
    "edit_velocity": 0.07909604519774012,
    "file_switch_rate": 0.5714285714285714,
    "session_length_sec": 5310.0,
    "time_since_last_commit_sec": 2700.0,
}

# Captured from ef67e05, fixture TASK_A/EVENTS_A, clock pinned to NOW_MS, TZ=UTC.
# Pre-migration call: extract_duration_features(store, "task-alpha", as_of_ms=NOW_MS).
# `time_of_day_hour` is 13.0 -- the UTC hour of `started_at`, not of NOW_MS (14).
# The two differ on purpose: a value of 14.0 here would mean something started
# measuring the reference time instead of the task's start.
PRE_MIGRATION_DURATION_TASK_A = {
    "file_count": 5.0,
    "total_edits": 7.0,
    "time_of_day_hour": 13.0,
    "branch_name_length": 27.0,
}

# Captured from ef67e05, fixture TASK_B (no events), clock pinned to NOW_MS, TZ=UTC.
# Pre-migration call: extract_stuck_features(store, "task-beta", as_of_ms=NOW_MS).
PRE_MIGRATION_STUCK_TASK_B = {
    "test_failure_count": 0.0,
    "time_in_phase_sec": 12000.0,
    "edit_velocity": 0.0,
    "file_switch_rate": 0.0,
    "session_length_sec": 10980.0,
    "time_since_last_commit_sec": 10980.0,
}

# Captured from ef67e05, fixture TASK_B (no events), clock pinned to NOW_MS, TZ=UTC.
# Pre-migration call: extract_duration_features(store, "task-beta", as_of_ms=NOW_MS).
PRE_MIGRATION_DURATION_TASK_B = {
    "file_count": 0.0,
    "total_edits": 0.0,
    "time_of_day_hour": 11.0,
    "branch_name_length": 0.0,
}

# Captured from ef67e05, missing entity, clock pinned to NOW_MS, TZ=UTC.
# Pre-migration call: extract_stuck_features(store, "task-does-not-exist", as_of_ms=NOW_MS).
PRE_MIGRATION_STUCK_MISSING_ENTITY = {
    "test_failure_count": 0.0,
    "time_in_phase_sec": 0.0,
    "edit_velocity": 0.0,
    "file_switch_rate": 0.0,
    "session_length_sec": 0.0,
    "time_since_last_commit_sec": 0.0,
}

# Captured from ef67e05, missing entity, clock pinned to NOW_MS, TZ=UTC.
# Pre-migration call: extract_duration_features(store, "task-does-not-exist", as_of_ms=NOW_MS).
# `time_of_day_hour` is 14.0, NOT 0.0. The empty duration vector deliberately
# defaults that feature to the current local hour -- pre-existing behaviour that
# `_empty_duration_features` documents, and the reason the wall clock has to be
# pinned for this baseline to mean anything.
PRE_MIGRATION_DURATION_MISSING_ENTITY = {
    "file_count": 0.0,
    "total_edits": 0.0,
    "time_of_day_hour": 14.0,
    "branch_name_length": 0.0,
}

# Captured from ef67e05, fixture BUFFER, clock pinned to NOW_MS, TZ=UTC.
# Pre-migration call: extract_features_from_buffer(BUFFER, as_of_ms=NOW_MS).
PRE_MIGRATION_BUFFER_POPULATED = {
    "test_failure_count": 2.0,
    "time_in_phase_sec": 1380.0,
    "edit_velocity": 0.13043478260869565,
    "file_switch_rate": 0.6666666666666666,
    "session_length_sec": 1380.0,
    "time_since_last_commit_sec": 840.0,
}

# Captured from ef67e05, empty buffer, clock pinned to NOW_MS, TZ=UTC.
# Pre-migration call: extract_features_from_buffer([], as_of_ms=NOW_MS).
PRE_MIGRATION_BUFFER_EMPTY = {
    "test_failure_count": 0.0,
    "time_in_phase_sec": 0.0,
    "edit_velocity": 0.0,
    "file_switch_rate": 0.0,
    "session_length_sec": 0.0,
    "time_since_last_commit_sec": 0.0,
}

# Captured from ef67e05 with *no* `as_of_ms` -- the argument the serving path
# actually passes (C-006: omitted means now). Under the pinned clock these are
# identical to the `as_of_ms=NOW_MS` captures above, which is the property that
# lets the local resolver -- which has no `as_of_ms` parameter at all -- be
# compared against the same numbers as the cloud path.
PRE_MIGRATION_STUCK_TASK_A_WALL_CLOCK = dict(PRE_MIGRATION_STUCK_TASK_A)
PRE_MIGRATION_DURATION_TASK_A_WALL_CLOCK = dict(PRE_MIGRATION_DURATION_TASK_A)

#: Ordered key lists, recorded from the same pre-migration run. Asserted
#: separately from the values because `dict.__eq__` ignores insertion order: a
#: reordered vector compares equal as a dict and is silently wrong once a model
#: indexes it positionally.
PRE_MIGRATION_STUCK_KEY_ORDER = [
    "test_failure_count",
    "time_in_phase_sec",
    "edit_velocity",
    "file_switch_rate",
    "session_length_sec",
    "time_since_last_commit_sec",
]
PRE_MIGRATION_DURATION_KEY_ORDER = [
    "file_count",
    "total_edits",
    "time_of_day_hour",
    "branch_name_length",
]


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


class FakeDataStore:
    """The slice of ``DataStore`` the serving path reads."""

    def __init__(self, tasks: dict[str, dict], events: dict[str, list[dict]]) -> None:
        self._tasks = tasks
        self._events = events

    def get_task_by_id(self, task_id: str) -> dict | None:
        task = self._tasks.get(task_id)
        return dict(task) if task is not None else None

    def get_events_for_task(self, task_id: str, since: int | None = None) -> list[dict]:
        return [dict(event) for event in self._events.get(task_id, [])]

    # -- the slice the cloud materializer reads -------------------------------

    def get_completed_tasks_for_tenant(self, tenant_id: str) -> list[dict]:
        return [dict(task) for task in self._tasks.values()]

    def get_events_for_task_id(self, task_id: str) -> list[dict]:
        return [dict(event) for event in self._events.get(task_id, [])]

    def get_last_training_ts(self, tenant_id: str) -> float | None:
        return None

    def insert_ml_event(self, kind: str, endpoint: str, routing: str, latency_ms: int) -> None:
        return None

    def commit(self) -> None:
        return None


class FakeModelStore:
    def __init__(self) -> None:
        self.saved: dict[str, bytes] = {}

    def load(self, model_name: str) -> bytes | None:
        return self.saved.get(model_name)

    def save(self, model_name: str, data: bytes) -> None:
        self.saved[model_name] = data

    def exists(self, model_name: str) -> bool:
        return model_name in self.saved


@pytest.fixture
def data_store() -> FakeDataStore:
    return FakeDataStore(
        {TASK_A["id"]: TASK_A, TASK_B["id"]: TASK_B},
        {TASK_A["id"]: EVENTS_A, TASK_B["id"]: EVENTS_B},
    )


class RecordingFeatureStore:
    """A Feast ``FeatureStore`` stand-in that records what the resolver pushes.

    The push is a byproduct (FR-014) -- the resolver returns its live computation
    regardless -- but what lands in the online store has to carry the same values,
    so it is captured and asserted rather than discarded.
    """

    def __init__(self) -> None:
        self.pushes: list[tuple[str, Any]] = []

    def push(self, source_name: str, frame: Any, to: Any = None) -> None:
        self.pushes.append((source_name, frame))

    def vector(self, source_name: str, names: list[str]) -> dict[str, float]:
        for pushed_name, frame in self.pushes:
            if pushed_name == source_name:
                row = frame.iloc[0]
                return {name: float(row[name]) for name in names}
        raise AssertionError(f"nothing was pushed to {source_name!r}")


@pytest.fixture
def recording_store() -> RecordingFeatureStore:
    store = RecordingFeatureStore()
    resolve_module.use_feature_store(store)
    return store


# ===========================================================================
# T024 -- value equality through the LOCAL resolver path (WP03)
# ===========================================================================


class TestLocalResolverPreservesValues:
    """``feature_store/resolve.py`` -- the path ``poller.py`` and ``routes.py`` now take."""

    def test_stuck_vector_equals_the_pre_migration_baseline(self, data_store: FakeDataStore) -> None:
        vector = resolve_module.resolve_stuck_features(data_store, TASK_A["id"])
        assert vector == PRE_MIGRATION_STUCK_TASK_A

    def test_stuck_key_order_equals_the_pre_migration_order(self, data_store: FakeDataStore) -> None:
        """Separate from value equality: `dict.__eq__` would pass on a reordering."""
        vector = resolve_module.resolve_stuck_features(data_store, TASK_A["id"])
        assert list(vector.keys()) == PRE_MIGRATION_STUCK_KEY_ORDER
        assert list(vector.keys()) == STUCK_FEATURE_NAMES

    def test_stuck_positional_vector_is_unchanged(self, data_store: FakeDataStore) -> None:
        """What ``StuckPredictor.predict`` actually builds, element for element."""
        vector = resolve_module.resolve_stuck_features(data_store, TASK_A["id"])
        positional = [vector.get(name, 0.0) for name in STUCK_FEATURE_NAMES]
        assert positional == [PRE_MIGRATION_STUCK_TASK_A[name] for name in STUCK_FEATURE_NAMES]

    def test_duration_vector_equals_the_pre_migration_baseline(self, data_store: FakeDataStore) -> None:
        vector = resolve_module.resolve_duration_features(data_store, TASK_A["id"])
        assert vector == PRE_MIGRATION_DURATION_TASK_A

    def test_duration_key_order_equals_the_pre_migration_order(self, data_store: FakeDataStore) -> None:
        vector = resolve_module.resolve_duration_features(data_store, TASK_A["id"])
        assert list(vector.keys()) == PRE_MIGRATION_DURATION_KEY_ORDER
        assert list(vector.keys()) == DURATION_FEATURE_NAMES

    def test_duration_positional_vector_is_unchanged(self, data_store: FakeDataStore) -> None:
        vector = resolve_module.resolve_duration_features(data_store, TASK_A["id"])
        positional = [vector.get(name, 0.0) for name in DURATION_FEATURE_NAMES]
        assert positional == [PRE_MIGRATION_DURATION_TASK_A[name] for name in DURATION_FEATURE_NAMES]

    def test_empty_event_stream_still_yields_the_pre_migration_defaults(self, data_store: FakeDataStore) -> None:
        """TASK_B has no events at all: every aggregate falls back to its default."""
        stuck = resolve_module.resolve_stuck_features(data_store, TASK_B["id"])
        duration = resolve_module.resolve_duration_features(data_store, TASK_B["id"])
        assert stuck == PRE_MIGRATION_STUCK_TASK_B
        assert duration == PRE_MIGRATION_DURATION_TASK_B
        assert list(stuck.keys()) == PRE_MIGRATION_STUCK_KEY_ORDER
        assert list(duration.keys()) == PRE_MIGRATION_DURATION_KEY_ORDER

    def test_missing_entity_still_yields_the_pre_migration_empty_vectors(self, data_store: FakeDataStore) -> None:
        """The degenerate case a rewiring most easily changes."""
        stuck = resolve_module.resolve_stuck_features(data_store, MISSING_TASK_ID)
        duration = resolve_module.resolve_duration_features(data_store, MISSING_TASK_ID)
        assert stuck == PRE_MIGRATION_STUCK_MISSING_ENTITY
        assert duration == PRE_MIGRATION_DURATION_MISSING_ENTITY
        assert list(stuck.keys()) == PRE_MIGRATION_STUCK_KEY_ORDER
        assert list(duration.keys()) == PRE_MIGRATION_DURATION_KEY_ORDER

    def test_the_empty_duration_vector_still_defaults_to_the_local_hour(self, data_store: FakeDataStore) -> None:
        """Guards the one non-zero default. 0.0 here would be a silent shift."""
        duration = resolve_module.resolve_duration_features(data_store, MISSING_TASK_ID)
        assert duration["time_of_day_hour"] == 14.0
        assert duration["time_of_day_hour"] != 0.0

    def test_buffer_fallback_vector_equals_the_pre_migration_baseline(self) -> None:
        """The path the poller takes when there is no active task."""
        vector = resolve_module.resolve_stuck_features_from_buffer(BUFFER)
        assert vector == PRE_MIGRATION_BUFFER_POPULATED
        assert list(vector.keys()) == PRE_MIGRATION_STUCK_KEY_ORDER

    def test_empty_buffer_still_yields_the_pre_migration_empty_vector(self) -> None:
        vector = resolve_module.resolve_stuck_features_from_buffer([])
        assert vector == PRE_MIGRATION_BUFFER_EMPTY
        assert list(vector.keys()) == PRE_MIGRATION_STUCK_KEY_ORDER

    def test_events_after_the_reference_time_are_still_excluded(self, data_store: FakeDataStore) -> None:
        """EVENTS_A carries two events past NOW_MS. Counting them would move two features."""
        vector = resolve_module.resolve_duration_features(data_store, TASK_A["id"])
        assert vector["total_edits"] == 7.0, "an event after the reference time was counted"
        stuck = resolve_module.resolve_stuck_features(data_store, TASK_A["id"])
        assert stuck["time_since_last_commit_sec"] == 2700.0, "a later commit leaked backwards"

    def test_the_pushed_vector_carries_the_baseline_values(
        self, data_store: FakeDataStore, recording_store: RecordingFeatureStore
    ) -> None:
        """The byproduct must not diverge from what was served."""
        returned = resolve_module.resolve_stuck_features(data_store, TASK_A["id"])
        assert resolve_module.flush_pushes(timeout=30.0), "the push queue did not drain"

        pushed = recording_store.vector(resolve_module.STUCK.push_source_name, STUCK_FEATURE_NAMES)
        assert pushed == PRE_MIGRATION_STUCK_TASK_A
        assert pushed == returned

    def test_the_pushed_duration_vector_carries_the_baseline_values(
        self, data_store: FakeDataStore, recording_store: RecordingFeatureStore
    ) -> None:
        returned = resolve_module.resolve_duration_features(data_store, TASK_A["id"])
        assert resolve_module.flush_pushes(timeout=30.0), "the push queue did not drain"

        pushed = recording_store.vector(resolve_module.DURATION.push_source_name, DURATION_FEATURE_NAMES)
        assert pushed == PRE_MIGRATION_DURATION_TASK_A
        assert pushed == returned


# ===========================================================================
# The cloud harness -- WP04's, reused rather than reinvented
# ===========================================================================


def _parquet_source_factory(directory: Path) -> Any:
    """Bind each production offline view to a parquet file instead of PostgreSQL.

    The projection into that file is the production
    :func:`rows_to_source_frame`, which shares its column list with the
    production SQL via ``source_query_columns``. Everything downstream -- the
    feature views, TTLs, feature services, entity frame and as-of join -- is the
    production object on the production code path. Only physical storage differs.
    """
    from feast import FileSource

    def factory(view: materialize.OfflineFeatureView) -> Any:
        return FileSource(
            name=f"{view.name}__{ML_FEATURES_TABLE}",
            path=str(directory / f"{view.name}.parquet"),
            timestamp_field="event_timestamp",
            created_timestamp_column="created_at",
        )

    return factory


def _offline_feature_store(tmp_path: Path, rows: list[materialize.OfflineFeatureRow]) -> Any:
    from feast import FeatureStore
    from feast.repo_config import RepoConfig

    repo = tmp_path / "feast_repo"
    data = repo / "data"
    data.mkdir(parents=True, exist_ok=True)

    for view in OFFLINE_FEATURE_VIEWS.values():
        rows_to_source_frame(rows, view).to_parquet(data / f"{view.name}.parquet", index=False)

    store = FeatureStore(
        config=RepoConfig(
            project="sigil_ml",
            provider="local",
            repo_path=str(repo),
            registry={"registry_type": "file", "path": str(repo / "registry.db")},
            online_store={"type": "sqlite", "path": str(repo / "online.db")},
            offline_store={"type": "dask"},
            entity_key_serialization_version=3,
        )
    )
    materialize.apply_cloud_definitions(store, cloud_feature_views(source_factory=_parquet_source_factory(data)))
    return store


@pytest.fixture
def cloud_rows(data_store: FakeDataStore) -> list[materialize.OfflineFeatureRow]:
    """Both tasks materialized by the production materializer at the write clock."""
    result = build_offline_rows(
        [TASK_A, TASK_B],
        data_store.get_events_for_task_id,
        now_ms=WRITE_TIME_MS,
    )
    assert result.skipped == 0
    return result.rows


@pytest.fixture
def cloud_trainer(tmp_path: Path, data_store: FakeDataStore, cloud_rows: list) -> CloudTrainer:
    return CloudTrainer(
        data_store=data_store,
        model_store=FakeModelStore(),
        config=CloudTrainingConfig(min_interval_sec=0),
        feature_store=_offline_feature_store(tmp_path, cloud_rows),
    )


def _retrieved(trainer: CloudTrainer, tasks: list[dict]) -> dict[str, Any]:
    return trainer._retrieve_offline_features(tasks, "acme")


# ===========================================================================
# T024 -- value equality through the CLOUD retrieval path (WP04)
# ===========================================================================


class TestCloudRetrievalPreservesValues:
    """``materialize.py`` write + Feast point-in-time read + ``cloud_trainer`` verify."""

    def test_materialized_stuck_row_equals_the_pre_migration_baseline(self, cloud_rows: list) -> None:
        """What the writer puts in ``ml_features``, before any retrieval."""
        row = next(r for r in cloud_rows if r.entity_id == TASK_A["id"] and r.feature_view == STUCK_OFFLINE_VIEW.name)
        assert row.feature_values == PRE_MIGRATION_STUCK_TASK_A
        assert list(row.feature_values.keys()) == PRE_MIGRATION_STUCK_KEY_ORDER
        assert row.event_timestamp_ms == NOW_MS, "the row must describe the pinned reference time"

    def test_materialized_duration_row_equals_the_pre_migration_baseline(self, cloud_rows: list) -> None:
        row = next(
            r for r in cloud_rows if r.entity_id == TASK_A["id"] and r.feature_view == DURATION_OFFLINE_VIEW.name
        )
        assert row.feature_values == PRE_MIGRATION_DURATION_TASK_A
        assert list(row.feature_values.keys()) == PRE_MIGRATION_DURATION_KEY_ORDER

    def test_retrieved_stuck_vector_equals_the_pre_migration_baseline(self, cloud_trainer: CloudTrainer) -> None:
        retrieved = _retrieved(cloud_trainer, [TASK_A])
        assert retrieved["stuck"].features[0] == PRE_MIGRATION_STUCK_TASK_A

    def test_retrieved_stuck_key_order_equals_the_pre_migration_order(self, cloud_trainer: CloudTrainer) -> None:
        retrieved = _retrieved(cloud_trainer, [TASK_A])
        assert list(retrieved["stuck"].features[0].keys()) == PRE_MIGRATION_STUCK_KEY_ORDER
        assert list(retrieved["stuck"].record.feature_names) == STUCK_FEATURE_NAMES

    def test_retrieved_stuck_positional_vector_is_unchanged(self, cloud_trainer: CloudTrainer) -> None:
        """The list that actually becomes a training row."""
        retrieved = _retrieved(cloud_trainer, [TASK_A])
        assert retrieved["stuck"].vectors[0] == [PRE_MIGRATION_STUCK_TASK_A[n] for n in STUCK_FEATURE_NAMES]

    def test_retrieved_duration_vector_equals_the_pre_migration_baseline(self, cloud_trainer: CloudTrainer) -> None:
        retrieved = _retrieved(cloud_trainer, [TASK_A])
        assert retrieved["duration"].features[0] == PRE_MIGRATION_DURATION_TASK_A
        assert list(retrieved["duration"].features[0].keys()) == PRE_MIGRATION_DURATION_KEY_ORDER

    def test_retrieved_duration_positional_vector_is_unchanged(self, cloud_trainer: CloudTrainer) -> None:
        retrieved = _retrieved(cloud_trainer, [TASK_A])
        assert retrieved["duration"].vectors[0] == [PRE_MIGRATION_DURATION_TASK_A[n] for n in DURATION_FEATURE_NAMES]

    def test_the_empty_input_task_survives_the_round_trip_unchanged(self, cloud_trainer: CloudTrainer) -> None:
        """TASK_B's all-default vector must come back as it went in."""
        retrieved = _retrieved(cloud_trainer, [TASK_B])
        assert retrieved["stuck"].features[0] == PRE_MIGRATION_STUCK_TASK_B
        assert retrieved["duration"].features[0] == PRE_MIGRATION_DURATION_TASK_B
        assert list(retrieved["stuck"].features[0].keys()) == PRE_MIGRATION_STUCK_KEY_ORDER
        assert list(retrieved["duration"].features[0].keys()) == PRE_MIGRATION_DURATION_KEY_ORDER

    def test_both_tasks_retrieve_together_without_crossing_values(self, cloud_trainer: CloudTrainer) -> None:
        """Two entities in one entity frame: each must get its own vector back."""
        retrieved = _retrieved(cloud_trainer, [TASK_A, TASK_B])
        assert retrieved["stuck"].task_ids == [TASK_A["id"], TASK_B["id"]]
        assert retrieved["stuck"].features[0] == PRE_MIGRATION_STUCK_TASK_A
        assert retrieved["stuck"].features[1] == PRE_MIGRATION_STUCK_TASK_B
        assert retrieved["duration"].features[0] == PRE_MIGRATION_DURATION_TASK_A
        assert retrieved["duration"].features[1] == PRE_MIGRATION_DURATION_TASK_B

    def test_a_missing_entity_is_refused_rather_than_defaulted(self, cloud_trainer: CloudTrainer) -> None:
        """The cloud degenerate case.

        The local path has a documented empty vector for an unknown task. The
        cloud path must NOT acquire one: an entity with nothing in ``ml_features``
        has to fail the retrieval loudly (FR-017), because a zero vector here
        would quietly teach the model that every feature was zero.
        """
        unknown = dict(TASK_A, id=MISSING_TASK_ID)
        with pytest.raises(materialize.OfflineRetrievalError):
            _retrieved(cloud_trainer, [unknown])

    def test_an_unmaterializable_task_is_skipped_not_zero_filled(self, data_store: FakeDataStore) -> None:
        """No reference time means no example -- never a default-valued row."""
        homeless = {k: v for k, v in TASK_A.items() if k not in ("completed_at", "last_active")}
        result = build_offline_rows([homeless], lambda _: EVENTS_A, now_ms=WRITE_TIME_MS)
        assert result.rows == []
        assert result.skipped == 1

    def test_the_postgres_sink_serializes_every_baseline_value_without_loss(self, cloud_rows: list) -> None:
        """Closes the one gap the parquet harness leaves on the write side.

        Production persists ``feature_values`` as JSONB through
        ``as_insert_params``. No PostgreSQL server exists here, but the
        serialization itself is production code and can be round-tripped exactly:
        every baseline float must survive ``json.dumps``/``json.loads`` bit for
        bit, including key order.
        """
        expected = {
            STUCK_OFFLINE_VIEW.name: {
                TASK_A["id"]: PRE_MIGRATION_STUCK_TASK_A,
                TASK_B["id"]: PRE_MIGRATION_STUCK_TASK_B,
            },
            DURATION_OFFLINE_VIEW.name: {
                TASK_A["id"]: PRE_MIGRATION_DURATION_TASK_A,
                TASK_B["id"]: PRE_MIGRATION_DURATION_TASK_B,
            },
        }
        for row in cloud_rows:
            serialized = row.as_insert_params()[5]
            assert json.loads(serialized) == expected[row.feature_view][row.entity_id]


# ===========================================================================
# T025 -- cross-deployment agreement (FR-015)
# ===========================================================================


class TestDeploymentsAgree:
    """The same task and events, through both deployments, at the same instant.

    ``TASK_A["completed_at"]`` is the pinned wall clock, so the local path's
    reference time (now, because the serving resolver takes no ``as_of_ms``) and
    the cloud path's reference time (``_reference_time_for`` -> ``completed_at``)
    are the *same* millisecond. Any divergence is therefore attributable to the
    deployment, not to timing.
    """

    def test_stuck_vectors_are_identical_across_deployments(
        self, data_store: FakeDataStore, cloud_trainer: CloudTrainer
    ) -> None:
        local = resolve_module.resolve_stuck_features(data_store, TASK_A["id"])
        cloud = _retrieved(cloud_trainer, [TASK_A])["stuck"].features[0]
        assert local == cloud

    def test_stuck_key_order_is_identical_across_deployments(
        self, data_store: FakeDataStore, cloud_trainer: CloudTrainer
    ) -> None:
        local = resolve_module.resolve_stuck_features(data_store, TASK_A["id"])
        cloud = _retrieved(cloud_trainer, [TASK_A])["stuck"].features[0]
        assert list(local.keys()) == list(cloud.keys())
        assert list(local.keys()) == PRE_MIGRATION_STUCK_KEY_ORDER

    def test_both_deployments_match_the_pre_migration_stuck_baseline(
        self, data_store: FakeDataStore, cloud_trainer: CloudTrainer
    ) -> None:
        """Agreement with each other is necessary but not sufficient -- two paths
        can agree on the same wrong answer. Both are pinned to the recorded
        pre-migration values, which is what makes their agreement meaningful."""
        local = resolve_module.resolve_stuck_features(data_store, TASK_A["id"])
        cloud = _retrieved(cloud_trainer, [TASK_A])["stuck"].features[0]
        assert local == PRE_MIGRATION_STUCK_TASK_A
        assert cloud == PRE_MIGRATION_STUCK_TASK_A

    def test_duration_vectors_are_identical_across_deployments(
        self, data_store: FakeDataStore, cloud_trainer: CloudTrainer
    ) -> None:
        local = resolve_module.resolve_duration_features(data_store, TASK_A["id"])
        cloud = _retrieved(cloud_trainer, [TASK_A])["duration"].features[0]
        assert local == cloud
        assert list(local.keys()) == list(cloud.keys())

    def test_both_deployments_match_the_pre_migration_duration_baseline(
        self, data_store: FakeDataStore, cloud_trainer: CloudTrainer
    ) -> None:
        local = resolve_module.resolve_duration_features(data_store, TASK_A["id"])
        cloud = _retrieved(cloud_trainer, [TASK_A])["duration"].features[0]
        assert local == PRE_MIGRATION_DURATION_TASK_A
        assert cloud == PRE_MIGRATION_DURATION_TASK_A

    def test_positional_vectors_are_identical_across_deployments(
        self, data_store: FakeDataStore, cloud_trainer: CloudTrainer
    ) -> None:
        """The form both trainers actually feed to scikit-learn."""
        retrieved = _retrieved(cloud_trainer, [TASK_A])
        for names, resolver, model in (
            (STUCK_FEATURE_NAMES, resolve_module.resolve_stuck_features, "stuck"),
            (DURATION_FEATURE_NAMES, resolve_module.resolve_duration_features, "duration"),
        ):
            local_vector = [resolver(data_store, TASK_A["id"]).get(name, 0.0) for name in names]
            assert local_vector == retrieved[model].vectors[0], model

    def test_the_empty_input_task_agrees_across_deployments(
        self, data_store: FakeDataStore, cloud_trainer: CloudTrainer
    ) -> None:
        """Degenerate inputs are where two paths most easily drift apart."""
        retrieved = _retrieved(cloud_trainer, [TASK_B])
        assert resolve_module.resolve_stuck_features(data_store, TASK_B["id"]) == retrieved["stuck"].features[0]
        assert resolve_module.resolve_duration_features(data_store, TASK_B["id"]) == retrieved["duration"].features[0]

    def test_the_registered_schema_order_matches_both_deployments(self) -> None:
        """The registry is the third place ordering could drift.

        Asserted against ``FeatureView.features``, never ``.schema`` --
        ``FeatureView.schema`` is built as ``list(set(...))`` and returns fields
        in arbitrary order, which would make this pass by luck.
        """
        from sigil_ml.feature_store.definitions import duration_feature_view, stuck_feature_view

        assert [f.name for f in stuck_feature_view().features] == PRE_MIGRATION_STUCK_KEY_ORDER
        assert [f.name for f in duration_feature_view().features] == PRE_MIGRATION_DURATION_KEY_ORDER
        assert list(STUCK_OFFLINE_VIEW.feature_names) == PRE_MIGRATION_STUCK_KEY_ORDER
        assert list(DURATION_OFFLINE_VIEW.feature_names) == PRE_MIGRATION_DURATION_KEY_ORDER


# ===========================================================================
# Baseline provenance (T023)
# ===========================================================================


def test_baselines_were_recorded_from_the_pre_migration_commit(tmp_path: Path) -> None:
    """Re-derive every literal above from the merge-base blob, straight from git.

    This is the reviewer's check that the numbers are recorded observations and
    not expectations someone typed: it loads
    ``ef67e05:src/sigil_ml/features.py`` -- code that predates every line of
    feature-store work -- runs it under the documented pinned clock and timezone,
    and compares. It is NOT the source of the literals; they stand on their own
    above, and every value-equality test in this module would still assert
    something if this one were deleted.

    Skips rather than fails when the blob is unreachable, which is what a shallow
    CI clone looks like.
    """
    import hashlib
    import importlib.util

    repo_root = Path(__file__).resolve().parent.parent
    try:
        completed = subprocess.run(
            ["git", "show", f"{PRE_MIGRATION_SHA}:src/sigil_ml/features.py"],
            cwd=repo_root,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"pre-migration blob {PRE_MIGRATION_SHA} is unreachable from this checkout: {exc}")

    blob = completed.stdout
    digest = hashlib.sha256(blob).hexdigest()
    assert digest == PRE_MIGRATION_FEATURES_SHA256, (
        f"{PRE_MIGRATION_SHA}:src/sigil_ml/features.py hashes to {digest}, not the recorded "
        f"{PRE_MIGRATION_FEATURES_SHA256} -- the baselines were measured against different source text."
    )

    module_path = tmp_path / "pre_migration_features.py"
    module_path.write_bytes(blob)
    spec = importlib.util.spec_from_file_location("pre_migration_features", module_path)
    assert spec is not None and spec.loader is not None
    pre = importlib.util.module_from_spec(spec)
    # The historical blob does `from sigil_ml.store import DataStore` (line 13).
    # The storage-layer reorganization moved that module to
    # `sigil_ml.datastore.protocol`, so alias it for the duration of the exec.
    #
    # This is safe precisely because the SHA256 assertion above already proved
    # the blob is the real historical text: the alias only lets the recorded
    # source import, it does not change a byte of what runs. A shim module at
    # the old path would also work but is forbidden by D-004 of the
    # storage-layer-reorganization mission.
    import sys

    from sigil_ml.datastore import protocol as _pre_migration_datastore

    _saved = sys.modules.get("sigil_ml.store")
    sys.modules["sigil_ml.store"] = _pre_migration_datastore
    try:
        spec.loader.exec_module(pre)
    finally:
        if _saved is None:
            sys.modules.pop("sigil_ml.store", None)
        else:
            sys.modules["sigil_ml.store"] = _saved
    # Same pinning as the post-migration paths get from `pinned_clock_and_timezone`.
    pre.time = PinnedClock(time)

    store = FakeDataStore(
        {TASK_A["id"]: TASK_A, TASK_B["id"]: TASK_B},
        {TASK_A["id"]: EVENTS_A, TASK_B["id"]: EVENTS_B},
    )

    recorded = [
        (pre.extract_stuck_features(store, TASK_A["id"], as_of_ms=NOW_MS), PRE_MIGRATION_STUCK_TASK_A),
        (pre.extract_duration_features(store, TASK_A["id"], as_of_ms=NOW_MS), PRE_MIGRATION_DURATION_TASK_A),
        (pre.extract_stuck_features(store, TASK_B["id"], as_of_ms=NOW_MS), PRE_MIGRATION_STUCK_TASK_B),
        (pre.extract_duration_features(store, TASK_B["id"], as_of_ms=NOW_MS), PRE_MIGRATION_DURATION_TASK_B),
        (pre.extract_stuck_features(store, MISSING_TASK_ID, as_of_ms=NOW_MS), PRE_MIGRATION_STUCK_MISSING_ENTITY),
        (
            pre.extract_duration_features(store, MISSING_TASK_ID, as_of_ms=NOW_MS),
            PRE_MIGRATION_DURATION_MISSING_ENTITY,
        ),
        (pre.extract_features_from_buffer(BUFFER, as_of_ms=NOW_MS), PRE_MIGRATION_BUFFER_POPULATED),
        (pre.extract_features_from_buffer([], as_of_ms=NOW_MS), PRE_MIGRATION_BUFFER_EMPTY),
        # No `as_of_ms` -- exactly what the serving path passes (C-006).
        (pre.extract_stuck_features(store, TASK_A["id"]), PRE_MIGRATION_STUCK_TASK_A_WALL_CLOCK),
        (pre.extract_duration_features(store, TASK_A["id"]), PRE_MIGRATION_DURATION_TASK_A_WALL_CLOCK),
    ]
    for observed, literal in recorded:
        assert observed == literal
        assert list(observed.keys()) == list(literal.keys())
