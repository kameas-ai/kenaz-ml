"""Tests for feature extraction using the DataStore abstraction.

The point-in-time correctness suite lives at the bottom of this file, from
``TestFrozenClockFixture`` onward. It deliberately avoids any real store: the
properties under test are properties of pure functions, and a SQLite round-trip
would only add a way for the test to fail for an unrelated reason. The
SQLite-backed tests above it predate the point-in-time work and are retained
because they are the only coverage of the real ``DataStore`` wiring (spec
User Story 4, "local and cloud agree").
"""

import contextlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from sigil_ml.datastore.sqlite import SqliteStore
from sigil_ml.features import (
    extract_duration_features,
    extract_duration_features_from_data,
    extract_features_from_buffer,
    extract_stuck_features,
    extract_stuck_features_from_data,
)
from sigil_ml.models import duration as duration_model
from sigil_ml.models import stuck as stuck_model


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite DB with the sigild schema and sample data."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))

    conn.executescript("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            repo_root TEXT,
            branch TEXT,
            phase TEXT,
            files TEXT,
            started_at INTEGER,
            last_active INTEGER,
            completed_at INTEGER,
            commit_count INTEGER,
            test_runs INTEGER,
            test_fails INTEGER
        );

        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT,
            source TEXT,
            payload TEXT,
            ts INTEGER
        );
    """)

    now_ms = int(time.time() * 1000)

    # Insert a task
    conn.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "task-1",
            "/tmp/repo",
            "feature/add-widget",
            "coding",
            json.dumps({"main.py": 5, "utils.py": 3}),
            now_ms - 3600_000,  # started 1 hour ago
            now_ms - 60_000,  # last active 1 min ago
            None,  # not completed
            2,
            5,
            3,
        ),
    )

    # Insert a completed task
    conn.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "task-2",
            "/tmp/repo",
            "fix/bug",
            "testing",
            json.dumps({"fix.py": 2}),
            now_ms - 7200_000,
            now_ms - 300_000,
            now_ms - 300_000,
            3,
            10,
            1,
        ),
    )

    # Insert events for task-1
    events = [
        ("edit", "editor", json.dumps({"file": "main.py"}), now_ms - 3000_000),
        ("edit", "editor", json.dumps({"file": "main.py"}), now_ms - 2800_000),
        ("edit", "editor", json.dumps({"file": "utils.py"}), now_ms - 2600_000),
        ("save", "editor", json.dumps({"file": "main.py"}), now_ms - 2500_000),
        ("commit", "git", json.dumps({"hash": "abc123"}), now_ms - 1800_000),
        ("phase_change", "sigild", json.dumps({"phase": "coding"}), now_ms - 1200_000),
        ("edit", "editor", json.dumps({"file": "main.py"}), now_ms - 600_000),
    ]
    conn.executemany(
        "INSERT INTO events (kind, source, payload, ts) VALUES (?, ?, ?, ?)",
        events,
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def store(test_db: Path) -> SqliteStore:
    """Create a SqliteStore backed by the test database."""
    s = SqliteStore(test_db)
    return s


class TestQueryHelpers:
    def test_query_task_exists(self, store: SqliteStore) -> None:
        task = store.get_task_by_id("task-1")
        assert task is not None
        assert task["branch"] == "feature/add-widget"
        assert task["phase"] == "coding"

    def test_query_task_not_found(self, store: SqliteStore) -> None:
        assert store.get_task_by_id("nonexistent") is None

    def test_query_events(self, store: SqliteStore) -> None:
        events = store.get_events_for_task("task-1")
        assert len(events) > 0
        kinds = [e["kind"] for e in events]
        assert "edit" in kinds
        assert "commit" in kinds


class TestStuckFeatures:
    def test_basic_extraction(self, store: SqliteStore) -> None:
        features = extract_stuck_features(store, "task-1")
        assert "test_failure_count" in features
        assert "time_in_phase_sec" in features
        assert "edit_velocity" in features
        assert "file_switch_rate" in features
        assert "session_length_sec" in features
        assert "time_since_last_commit_sec" in features

        assert features["test_failure_count"] == 3.0
        assert features["session_length_sec"] > 0
        assert features["edit_velocity"] >= 0

    def test_missing_task(self, store: SqliteStore) -> None:
        features = extract_stuck_features(store, "nonexistent")
        assert features["test_failure_count"] == 0.0
        assert features["session_length_sec"] == 0.0

    def test_file_switch_rate(self, store: SqliteStore) -> None:
        features = extract_stuck_features(store, "task-1")
        # We have edits to main.py and utils.py
        assert 0.0 < features["file_switch_rate"] <= 1.0


class TestDurationFeatures:
    def test_basic_extraction(self, store: SqliteStore) -> None:
        features = extract_duration_features(store, "task-1")
        assert features["file_count"] == 2.0
        assert features["total_edits"] >= 0
        assert 0 <= features["time_of_day_hour"] < 24
        assert features["branch_name_length"] == len("feature/add-widget")

    def test_missing_task(self, store: SqliteStore) -> None:
        features = extract_duration_features(store, "nonexistent")
        assert features["file_count"] == 0.0
        assert features["branch_name_length"] == 0.0


# ===========================================================================
# Point-in-time correctness suite (WP02)
#
# Every test below the T009 fixtures is written so that it FAILS against the
# pre-WP01 implementation. Two mechanisms are at work, and it is worth being
# explicit about which is which:
#
#   1. Signature. Pre-WP01 the extractors took no `as_of_ms` at all, so every
#      call here raises TypeError. That alone is a failure, but it is a weak
#      one -- it proves the parameter exists, not that it means anything.
#   2. Semantics. Each determinism/lookahead test was also checked against a
#      shim of the pre-WP01 body that accepts `as_of_ms` and ignores it (i.e.
#      still computes `now_ms = int(time.time() * 1000)` and never filters
#      events). Under that shim these tests still fail: the frozen clocks in a
#      paired extraction are 90 days apart, so `time_in_phase_sec` and
#      `time_since_last_commit_sec` diverge by ~7.8e6 seconds, and the
#      unfiltered event list pulls post-reference events into every aggregate.
#
# That second property is the one that matters. A determinism test whose two
# extractions happen in the same millisecond passes trivially against the buggy
# code and proves nothing.
# ===========================================================================


# --- T009: frozen-clock and fake-store fixtures ---------------------------


@contextlib.contextmanager
def frozen_clock(epoch_ms: int):
    """Pin wall clock as observed by sigil_ml.features.

    `features.py` does `import time`, so `sigil_ml.features.time` is the time
    module itself; patching its `time` attribute is what the extractors see
    when they resolve `as_of_ms=None`.
    """
    with mock.patch("sigil_ml.features.time.time", return_value=epoch_ms / 1000.0):
        yield


@contextlib.contextmanager
def frozen_local_hour(hour: int):
    """Pin the local hour as observed by sigil_ml.features.

    `time.localtime()` is implemented in C and does not route through the
    Python-level `time.time`, so `frozen_clock` cannot reach it. The empty
    duration vector's `time_of_day_hour` default is the only thing that needs
    this.
    """
    real_localtime = time.localtime

    def fake_localtime(secs: float | None = None) -> time.struct_time:
        base = real_localtime(secs) if secs is not None else real_localtime()
        return time.struct_time((*base[:3], hour, *base[4:]))

    with mock.patch("sigil_ml.features.time.localtime", side_effect=fake_localtime):
        yield


class FakeStore:
    """Minimal stand-in for the DataStore methods the extractors call.

    Deliberately not a subclass of, and not validated against, `DataStore`:
    this suite must not depend on SQLite or Postgres. The extractors call
    exactly two methods.
    """

    def __init__(self, task: dict[str, Any] | None, events: list[dict[str, Any]] | None = None) -> None:
        self._task = task
        self._events = events if events is not None else []
        self.calls: list[tuple[str, str]] = []

    def get_task_by_id(self, task_id: str) -> dict[str, Any] | None:
        self.calls.append(("get_task_by_id", task_id))
        return self._task

    def get_events_for_task(self, task_id: str, since: int | None = None) -> list[dict[str, Any]]:
        self.calls.append(("get_events_for_task", task_id))
        return list(self._events)


# A fixed epoch so the timeline is readable and the expected values below are
# arithmetic anyone can check by hand.
T0 = 1_700_000_000_000

# Task-window event timeline. Offsets from T0, in ms.
E_EDIT_1 = T0 + 60_000  # edit main.py
E_EDIT_2 = T0 + 120_000  # edit utils.py
E_COMMIT_1 = T0 + 300_000  # commit
E_PHASE = T0 + 600_000  # phase_change -> "coding"
E_EDIT_3 = T0 + 900_000  # edit main.py
E_TERMINAL = T0 + 1_200_000  # failing test run
E_SAVE = T0 + 1_800_000  # save utils.py
E_COMMIT_2 = T0 + 3_000_000  # commit

TASK_END = T0 + 3_600_000  # completed_at / last_active

# Reference times used repeatedly below.
REF_END = TASK_END  # end of the task's life
REF_MID = E_PHASE  # exactly on an event timestamp -- the inclusive boundary

# The wall clock every reference-time test runs under. It is deliberately far
# from any reference time used below: whenever `as_of_ms` is supplied, the wall
# clock must contribute nothing, and pinning it 90 days out is what turns that
# claim into something a test can fail on.
FAR_FUTURE = REF_END + 90 * 86_400_000


def make_task(**overrides: Any) -> dict[str, Any]:
    """A completed task with a known, fully specified timeline."""
    task: dict[str, Any] = {
        "id": "task-1",
        "repo_root": "/tmp/repo",
        "branch": "feature/add-widget",
        "phase": "coding",
        "files": {"main.py": 5, "utils.py": 3},
        "started_at": T0,
        "last_active": TASK_END,
        "completed_at": TASK_END,
        "commit_count": 2,
        "test_runs": 5,
        "test_fails": 3,
    }
    task.update(overrides)
    return task


def make_events() -> list[dict[str, Any]]:
    """Events in the *task-window* vocabulary that the task extractors read.

    Note the commit events use `kind == "commit"` (with `source == "git"`).
    That is what `extract_stuck_features_from_data` looks for, and it differs
    from the raw daemon-stream vocabulary that `extract_features_from_buffer`
    reads, where commits arrive as `kind == "git"`. Both vocabularies are
    genuine; neither is a typo.

    Caveat for future readers: production data currently contains *no* events
    of kind "commit" or "git" at all -- the local sigild database only ever
    emits file/process/hyprland/browser/terminal/power. So in production the
    commit-recency branch never fires and `time_since_last_commit_sec` always
    falls back to `session_length_sec`. Commit events appear here to exercise
    the non-fallback branch, not because the fixture mirrors real traffic.
    """
    return [
        {"id": 1, "kind": "edit", "source": "editor", "payload": {"file": "main.py"}, "ts": E_EDIT_1},
        {"id": 2, "kind": "edit", "source": "editor", "payload": {"file": "utils.py"}, "ts": E_EDIT_2},
        {"id": 3, "kind": "commit", "source": "git", "payload": {"hash": "aaa111"}, "ts": E_COMMIT_1},
        {"id": 4, "kind": "phase_change", "source": "sigild", "payload": {"phase": "coding"}, "ts": E_PHASE},
        {"id": 5, "kind": "edit", "source": "editor", "payload": {"file": "main.py"}, "ts": E_EDIT_3},
        {
            "id": 6,
            "kind": "terminal",
            "source": "shell",
            "payload": {"cmd": "pytest", "exit_code": 1},
            "ts": E_TERMINAL,
        },
        {"id": 7, "kind": "save", "source": "editor", "payload": {"file": "utils.py"}, "ts": E_SAVE},
        {"id": 8, "kind": "commit", "source": "git", "payload": {"hash": "bbb222"}, "ts": E_COMMIT_2},
    ]


# Buffer-extractor timeline, in the raw daemon-stream vocabulary.
B_FILE_1 = T0 + 10_000
B_FILE_2 = T0 + 20_000
B_GIT = T0 + 30_000
B_TERMINAL = T0 + 40_000
B_FILE_3 = T0 + 70_000
B_NOW = T0 + 100_000


def make_buffer_events() -> list[dict[str, Any]]:
    """Events as the poller buffers them: kind "file"/"git"/"terminal"."""
    return [
        {"id": 1, "kind": "file", "source": "fswatch", "payload": {"path": "/repo/a.py"}, "ts": B_FILE_1},
        {"id": 2, "kind": "file", "source": "fswatch", "payload": {"path": "/repo/b.py"}, "ts": B_FILE_2},
        {"id": 3, "kind": "git", "source": "git", "payload": {"branch": "main"}, "ts": B_GIT},
        {
            "id": 4,
            "kind": "terminal",
            "source": "shell",
            "payload": {"cmd": "pytest", "exit_code": 1},
            "ts": B_TERMINAL,
        },
        {"id": 5, "kind": "file", "source": "fswatch", "payload": {"path": "/repo/a.py"}, "ts": B_FILE_3},
    ]


# Hand-computed expectations at REF_END. Every one is checkable from the
# timeline above without running the code.
EXPECTED_STUCK_AT_END = {
    "test_failure_count": 3.0,  # task["test_fails"]
    "time_in_phase_sec": 3000.0,  # REF_END - E_PHASE
    "edit_velocity": 4.0 / 60.0,  # 4 edit-ish events over a 60-minute session
    "file_switch_rate": 0.5,  # {main.py, utils.py} / 4 edits
    "session_length_sec": 3600.0,  # last_active - started_at
    "time_since_last_commit_sec": 600.0,  # REF_END - E_COMMIT_2
}

EXPECTED_DURATION_AT_END = {
    "file_count": 2.0,
    "total_edits": 4.0,
    "branch_name_length": 18.0,  # len("feature/add-widget")
}


class TestFrozenClockFixture:
    """T009 -- the fixtures themselves must be load-bearing."""

    def test_frozen_clock_changes_what_the_extractor_observes(self) -> None:
        """Without `as_of_ms`, the extractor tracks the (frozen) wall clock.

        This is the control for every determinism test that follows: if the
        clock patch did not reach `features.py`, these two vectors would be
        equal and the determinism tests would be vacuous.
        """
        task, events = make_task(), make_events()

        with frozen_clock(REF_END):
            near = extract_stuck_features_from_data(task, events)
        with frozen_clock(REF_END + 90 * 86_400_000):
            far = extract_stuck_features_from_data(task, events)

        assert near["time_in_phase_sec"] == pytest.approx(3000.0)
        assert far["time_in_phase_sec"] == pytest.approx(3000.0 + 90 * 86_400.0)
        assert near != far

    def test_frozen_local_hour_reaches_the_empty_duration_default(self) -> None:
        with frozen_local_hour(13):
            empty = extract_duration_features(FakeStore(task=None), "missing")
        assert empty["time_of_day_hour"] == 13.0

    def test_fake_store_satisfies_every_method_the_extractors_call(self) -> None:
        task, events = make_task(), make_events()
        store = FakeStore(task=task, events=events)

        extract_stuck_features(store, "task-1", as_of_ms=REF_END)
        extract_duration_features(store, "task-1", as_of_ms=REF_END)

        called = {name for name, _ in store.calls}
        assert called == {"get_task_by_id", "get_events_for_task"}
        assert all(task_id == "task-1" for _, task_id in store.calls)

    def test_fixture_timeline_spans_the_asserted_window(self) -> None:
        task, events = make_task(), make_events()
        timestamps = [e["ts"] for e in events]

        assert min(timestamps) == T0 + 60_000
        assert max(timestamps) == T0 + 3_000_000
        assert task["started_at"] < min(timestamps)
        assert max(timestamps) < task["completed_at"] == TASK_END
        assert task["completed_at"] - task["started_at"] == 3_600_000


class TestDeterminism:
    """T010 -- SC-001 and User Story 1, acceptance scenarios 1 and 2."""

    def test_stuck_extraction_is_independent_of_wall_clock(self) -> None:
        task, events = make_task(), make_events()
        ref = task["completed_at"]

        with frozen_clock(ref + 60_000):
            a = extract_stuck_features_from_data(task, events, as_of_ms=ref)
        with frozen_clock(ref + 90 * 86_400_000):
            b = extract_stuck_features_from_data(task, events, as_of_ms=ref)

        # Stability...
        assert a == b
        # ...and correctness. A function returning constants would satisfy the
        # equality above; only these fix the values to the reference moment.
        assert a["time_in_phase_sec"] == pytest.approx(3000.0, abs=1.0)
        assert a["time_since_last_commit_sec"] == pytest.approx(600.0, abs=1.0)
        assert a["session_length_sec"] == pytest.approx(3600.0, abs=1.0)

    def test_stuck_vector_matches_the_hand_computed_reference_vector(self) -> None:
        task, events = make_task(), make_events()

        with frozen_clock(REF_END + 400 * 86_400_000):
            v = extract_stuck_features_from_data(task, events, as_of_ms=REF_END)

        assert v == pytest.approx(EXPECTED_STUCK_AT_END)

    def test_duration_extraction_is_independent_of_wall_clock(self) -> None:
        """The duration family, deliberately at a *truncating* reference time.

        None of the four duration features is computed from the clock, so a
        reference time at the end of the task would make this test pass against
        the buggy implementation too. REF_MID cuts the event list in half, so
        `total_edits` is 2 only if the reference time is honoured.
        """
        task, events = make_task(), make_events()
        ref = REF_MID

        with frozen_clock(ref + 60_000):
            a = extract_duration_features_from_data(task, events, as_of_ms=ref)
        with frozen_clock(ref + 90 * 86_400_000):
            b = extract_duration_features_from_data(task, events, as_of_ms=ref)

        assert a == b
        assert a["file_count"] == 2.0
        assert a["total_edits"] == 2.0  # 4 edits in the task, 2 by REF_MID
        assert a["branch_name_length"] == 18.0
        # time_of_day_hour derives from started_at, never from the clock.
        assert a["time_of_day_hour"] == float(time.localtime(T0 / 1000.0).tm_hour)

    def test_buffer_extraction_is_independent_of_wall_clock(self) -> None:
        events = make_buffer_events()

        with frozen_clock(B_NOW + 60_000):
            a = extract_features_from_buffer(events, as_of_ms=B_NOW)
        with frozen_clock(B_NOW + 90 * 86_400_000):
            b = extract_features_from_buffer(events, as_of_ms=B_NOW)

        assert a == b
        assert a["session_length_sec"] == pytest.approx(60.0)
        assert a["time_since_last_commit_sec"] == pytest.approx(70.0)  # B_NOW - B_GIT
        assert a["edit_velocity"] == pytest.approx(3.0)
        assert a["file_switch_rate"] == pytest.approx(2.0 / 3.0)
        assert a["test_failure_count"] == 1.0

    def test_same_reference_time_extracts_identically_across_two_days(self) -> None:
        """User Story 1 scenario 2, stated as literally as it is written."""
        task, events = make_task(), make_events()
        ref = task["completed_at"]

        day_one = ref + 3_600_000
        day_two = day_one + 86_400_000

        with frozen_clock(day_one):
            first = extract_stuck_features_from_data(task, events, as_of_ms=ref)
        with frozen_clock(day_two):
            second = extract_stuck_features_from_data(task, events, as_of_ms=ref)

        assert first == second
        assert first["time_in_phase_sec"] == pytest.approx(3000.0)


class TestNoLookahead:
    """T011 -- FR-003 and User Story 1 scenario 3."""

    def test_stuck_events_after_reference_time_do_not_contribute(self) -> None:
        task, events = make_task(), make_events()
        ref = task["started_at"] + 600_000
        past = [e for e in events if e["ts"] <= ref]

        assert len(past) < len(events)  # the filter has something to do
        with frozen_clock(FAR_FUTURE):
            full = extract_stuck_features_from_data(task, events, as_of_ms=ref)
            prefiltered = extract_stuck_features_from_data(task, past, as_of_ms=ref)

        assert full == prefiltered
        # And the values are the ones the truncated window implies, not the
        # ones the full window would have produced.
        assert full["edit_velocity"] == pytest.approx(2.0 / 60.0)
        assert full["file_switch_rate"] == pytest.approx(1.0)
        assert full["time_since_last_commit_sec"] == pytest.approx(300.0)

    def test_duration_events_after_reference_time_do_not_contribute(self) -> None:
        task, events = make_task(), make_events()
        ref = task["started_at"] + 600_000
        past = [e for e in events if e["ts"] <= ref]

        with frozen_clock(FAR_FUTURE):
            full = extract_duration_features_from_data(task, events, as_of_ms=ref)
            prefiltered = extract_duration_features_from_data(task, past, as_of_ms=ref)

        assert full == prefiltered
        assert full["total_edits"] == 2.0  # E_EDIT_1, E_EDIT_2 only
        assert extract_duration_features_from_data(task, events, as_of_ms=REF_END)["total_edits"] == 4.0

    def test_buffer_events_after_reference_time_do_not_contribute(self) -> None:
        events = make_buffer_events()
        ref = B_GIT
        past = [e for e in events if e["ts"] <= ref]

        with frozen_clock(FAR_FUTURE):
            full = extract_features_from_buffer(events, as_of_ms=ref)
            prefiltered = extract_features_from_buffer(past, as_of_ms=ref)

        assert full == prefiltered
        assert full["session_length_sec"] == pytest.approx(20.0)  # B_GIT - B_FILE_1
        assert full["time_since_last_commit_sec"] == pytest.approx(0.0)

    def test_event_exactly_at_reference_time_is_included(self) -> None:
        """Inclusive boundary, asserted rather than implied.

        The two lists below differ by exactly one event -- the `phase_change`
        whose `ts` equals the reference time. Included, it moves the phase
        start onto the reference moment and `time_in_phase_sec` collapses to
        zero. Excluded, phase start falls back to `started_at`.
        """
        task = make_task()
        boundary_event = {
            "id": 4,
            "kind": "phase_change",
            "source": "sigild",
            "payload": {"phase": "coding"},
            "ts": REF_MID,
        }
        without = [e for e in make_events() if e["ts"] < REF_MID]
        with_boundary = [*without, boundary_event]

        with frozen_clock(FAR_FUTURE):
            included = extract_stuck_features_from_data(task, with_boundary, as_of_ms=REF_MID)
            excluded = extract_stuck_features_from_data(task, without, as_of_ms=REF_MID)

        assert included != excluded
        assert included["time_in_phase_sec"] == pytest.approx(0.0)
        assert excluded["time_in_phase_sec"] == pytest.approx(600.0)

    def test_commit_exactly_at_reference_time_is_included(self) -> None:
        task = make_task()
        base = [e for e in make_events() if e["ts"] < E_COMMIT_1]
        boundary_commit = {"id": 3, "kind": "commit", "source": "git", "payload": {}, "ts": E_COMMIT_1}

        with frozen_clock(FAR_FUTURE):
            included = extract_stuck_features_from_data(task, [*base, boundary_commit], as_of_ms=E_COMMIT_1)
            excluded = extract_stuck_features_from_data(task, base, as_of_ms=E_COMMIT_1)

        assert included["time_since_last_commit_sec"] == pytest.approx(0.0)
        # No commit in window -> documented fallback to session length.
        assert excluded["time_since_last_commit_sec"] == pytest.approx(3600.0)
        assert included != excluded

    def test_post_reference_event_changes_aggregates_only_once_reference_moves(self) -> None:
        """Proves the filter is load-bearing rather than incidentally inert."""
        task, events = make_task(), make_events()
        extra = {"id": 99, "kind": "edit", "source": "editor", "payload": {"file": "extra.py"}, "ts": REF_MID + 1}
        augmented = [*events, extra]

        with frozen_clock(FAR_FUTURE):
            at_ref = extract_stuck_features_from_data(task, augmented, as_of_ms=REF_MID)
            baseline = extract_stuck_features_from_data(task, events, as_of_ms=REF_MID)
            later = extract_stuck_features_from_data(task, augmented, as_of_ms=REF_MID + 1)

        # One millisecond too late to count.
        assert at_ref == baseline
        assert at_ref["edit_velocity"] == pytest.approx(2.0 / 60.0)
        assert at_ref["file_switch_rate"] == pytest.approx(1.0)

        # One millisecond later it counts, and three aggregates move.
        assert later["edit_velocity"] == pytest.approx(3.0 / 60.0)
        assert later["file_switch_rate"] == pytest.approx(1.0)  # 3 files / 3 edits
        assert later != at_ref

        with frozen_clock(FAR_FUTURE):
            dur_at_ref = extract_duration_features_from_data(task, augmented, as_of_ms=REF_MID)
            dur_later = extract_duration_features_from_data(task, augmented, as_of_ms=REF_MID + 1)
        assert dur_at_ref["total_edits"] == 2.0
        assert dur_later["total_edits"] == 3.0


class TestNegativeClamp:
    """T012 -- FR-004 and User Story 1 scenario 4."""

    def test_reference_time_before_started_at_yields_no_negative_durations(self) -> None:
        task, events = make_task(), make_events()
        early = task["started_at"] - 10_000

        with frozen_clock(FAR_FUTURE):
            v = extract_stuck_features_from_data(task, events, as_of_ms=early)

        assert v["time_in_phase_sec"] >= 0.0
        assert v["time_since_last_commit_sec"] >= 0.0
        assert v["time_in_phase_sec"] == 0.0  # clamped, not -10.0
        for name, value in v.items():
            assert value >= 0.0, f"{name} went negative"

    def test_phase_change_after_reference_time_does_not_move_phase_start(self) -> None:
        """The only phase_change in the window carries a ts past the reference."""
        task = make_task()
        events = [{"id": 1, "kind": "phase_change", "source": "sigild", "payload": {}, "ts": E_PHASE}]
        ref = T0 + 300_000

        with frozen_clock(FAR_FUTURE):
            v = extract_stuck_features_from_data(task, events, as_of_ms=ref)

        # Filtered out, so phase_start falls back to started_at.
        assert v["time_in_phase_sec"] == pytest.approx(300.0)
        assert v["time_in_phase_sec"] >= 0.0

    def test_phase_change_after_an_early_reference_time_still_clamps(self) -> None:
        task = make_task()
        events = [{"id": 1, "kind": "phase_change", "source": "sigild", "payload": {}, "ts": E_PHASE}]
        early = task["started_at"] - 60_000

        with frozen_clock(FAR_FUTURE):
            v = extract_stuck_features_from_data(task, events, as_of_ms=early)

        assert v["time_in_phase_sec"] == 0.0

    def test_early_reference_time_does_not_raise(self) -> None:
        """Clamping is the specified behavior, not an error path."""
        task, events = make_task(), make_events()
        far_before = task["started_at"] - 10 * 86_400_000

        with frozen_clock(FAR_FUTURE):
            stuck = extract_stuck_features_from_data(task, events, as_of_ms=far_before)
            dur = extract_duration_features_from_data(task, events, as_of_ms=far_before)
            buf = extract_features_from_buffer(make_buffer_events(), as_of_ms=far_before)

        assert stuck["time_in_phase_sec"] == 0.0
        assert dur["total_edits"] == 0.0
        assert buf == {
            "test_failure_count": 0.0,
            "time_in_phase_sec": 0.0,
            "edit_velocity": 0.0,
            "file_switch_rate": 0.0,
            "session_length_sec": 0.0,
            "time_since_last_commit_sec": 0.0,
        }


class TestPathEquivalence:
    """T013 -- FR-006, SC-004, User Story 3."""

    def test_store_backed_stuck_matches_data_backed(self) -> None:
        task, events = make_task(), make_events()
        store = FakeStore(task=task, events=events)
        ref = task["completed_at"]

        with frozen_clock(FAR_FUTURE):
            via_store = extract_stuck_features(store, "task-1", as_of_ms=ref)
            via_data = extract_stuck_features_from_data(task, events, as_of_ms=ref)

        assert via_store == via_data
        # Dict equality ignores ordering, so pin ordering separately.
        assert list(via_store.keys()) == list(via_data.keys())
        assert via_store == pytest.approx(EXPECTED_STUCK_AT_END)

    def test_store_backed_duration_matches_data_backed(self) -> None:
        task, events = make_task(), make_events()
        store = FakeStore(task=task, events=events)
        ref = task["completed_at"]

        with frozen_clock(FAR_FUTURE):
            via_store = extract_duration_features(store, "task-1", as_of_ms=ref)
            via_data = extract_duration_features_from_data(task, events, as_of_ms=ref)

        assert via_store == via_data
        assert list(via_store.keys()) == list(via_data.keys())
        for key, expected in EXPECTED_DURATION_AT_END.items():
            assert via_store[key] == pytest.approx(expected)

        # Again at a truncating reference time, so that the equality above
        # cannot be satisfied by two paths that both ignore `as_of_ms`.
        with frozen_clock(FAR_FUTURE):
            mid_store = extract_duration_features(store, "task-1", as_of_ms=REF_MID)
            mid_data = extract_duration_features_from_data(task, events, as_of_ms=REF_MID)
        assert mid_store == mid_data
        assert mid_store["total_edits"] == 2.0

    def test_store_backed_matches_data_backed_without_a_reference_time(self) -> None:
        """`as_of_ms=None` under a frozen clock -- the serving shape."""
        task, events = make_task(), make_events()
        store = FakeStore(task=task, events=events)

        with frozen_clock(REF_END + 3_600_000):
            stuck_store = extract_stuck_features(store, "task-1")
            stuck_data = extract_stuck_features_from_data(task, events)
            dur_store = extract_duration_features(store, "task-1")
            dur_data = extract_duration_features_from_data(task, events)

        assert stuck_store == stuck_data
        assert list(stuck_store.keys()) == list(stuck_data.keys())
        assert dur_store == dur_data
        assert list(dur_store.keys()) == list(dur_data.keys())
        # With no reference time the clock is the reference: one hour later
        # than REF_END, so the phase has been running an hour longer.
        assert stuck_store["time_in_phase_sec"] == pytest.approx(3000.0 + 3600.0)

    def test_missing_task_returns_the_empty_stuck_vector_key_by_key(self) -> None:
        store = FakeStore(task=None)

        v = extract_stuck_features(store, "nonexistent")

        assert v == {
            "test_failure_count": 0.0,
            "time_in_phase_sec": 0.0,
            "edit_velocity": 0.0,
            "file_switch_rate": 0.0,
            "session_length_sec": 0.0,
            "time_since_last_commit_sec": 0.0,
        }
        assert list(v.keys()) == stuck_model.FEATURE_NAMES
        # Delegation must not have gone looking for events.
        assert [name for name, _ in store.calls] == ["get_task_by_id"]

    def test_missing_task_returns_the_empty_duration_vector_key_by_key(self) -> None:
        store = FakeStore(task=None)

        with frozen_local_hour(9):
            v = extract_duration_features(store, "nonexistent")

        assert v == {
            "file_count": 0.0,
            "total_edits": 0.0,
            "time_of_day_hour": 9.0,
            "branch_name_length": 0.0,
        }
        assert list(v.keys()) == duration_model.FEATURE_NAMES
        assert [name for name, _ in store.calls] == ["get_task_by_id"]

    def test_empty_duration_vector_keeps_a_non_zero_time_of_day_default(self) -> None:
        """The one non-zero default in the empty contract.

        Easy to flatten to 0.0 during a delegation refactor, and nothing
        downstream would complain -- it would just quietly become a wrong
        feature value. Two different hours are checked so that a hardcoded
        constant cannot satisfy this either.
        """
        for hour in (9, 17):
            with frozen_local_hour(hour):
                v = extract_duration_features(FakeStore(task=None), "nonexistent")
            assert v["time_of_day_hour"] == float(hour)
            assert v["time_of_day_hour"] != 0.0


class TestVectorLayoutFrozen:
    """T014 -- FR-011, SC-005, C-005."""

    STUCK_KEYS = [
        "test_failure_count",
        "time_in_phase_sec",
        "edit_velocity",
        "file_switch_rate",
        "session_length_sec",
        "time_since_last_commit_sec",
    ]
    DURATION_KEYS = [
        "file_count",
        "total_edits",
        "time_of_day_hour",
        "branch_name_length",
    ]

    def test_ordered_key_lists_are_asserted_literally(self) -> None:
        task, events = make_task(), make_events()

        with frozen_clock(FAR_FUTURE):
            stuck = extract_stuck_features_from_data(task, events, as_of_ms=REF_END)
            dur = extract_duration_features_from_data(task, events, as_of_ms=REF_END)
            buf = extract_features_from_buffer(make_buffer_events(), as_of_ms=B_NOW)

        assert list(stuck.keys()) == self.STUCK_KEYS
        assert list(dur.keys()) == self.DURATION_KEYS
        # The buffer extractor is documented as stuck-shaped.
        assert list(buf.keys()) == self.STUCK_KEYS

    def test_key_order_is_actually_load_bearing(self) -> None:
        """Guard the guard.

        The trainers index these vectors positionally, so a permutation of the
        same names is a different vector. `set(...)` or `sorted(...)` equality
        would not notice; list equality must.
        """
        permuted = [self.STUCK_KEYS[1], self.STUCK_KEYS[0], *self.STUCK_KEYS[2:]]
        assert set(permuted) == set(self.STUCK_KEYS)
        assert permuted != self.STUCK_KEYS

        task, events = make_task(), make_events()
        with frozen_clock(FAR_FUTURE):
            stuck = extract_stuck_features_from_data(task, events, as_of_ms=REF_END)
        assert list(stuck.keys()) != permuted

    def test_layout_matches_the_constants_the_trainers_index(self) -> None:
        task, events = make_task(), make_events()

        with frozen_clock(FAR_FUTURE):
            stuck = extract_stuck_features_from_data(task, events, as_of_ms=REF_END)
            dur = extract_duration_features_from_data(task, events, as_of_ms=REF_END)

        assert list(stuck.keys()) == stuck_model.FEATURE_NAMES
        assert list(dur.keys()) == duration_model.FEATURE_NAMES
        assert len(stuck) == len(stuck_model.FEATURE_NAMES)
        assert len(dur) == len(duration_model.FEATURE_NAMES)
        # The literal lists above and the model constants must not drift apart.
        assert self.STUCK_KEYS == stuck_model.FEATURE_NAMES
        assert self.DURATION_KEYS == duration_model.FEATURE_NAMES

    def test_every_value_is_a_float(self) -> None:
        task, events = make_task(), make_events()

        with frozen_clock(FAR_FUTURE):
            vectors = [
                extract_stuck_features_from_data(task, events, as_of_ms=REF_END),
                extract_duration_features_from_data(task, events, as_of_ms=REF_END),
                extract_features_from_buffer(make_buffer_events(), as_of_ms=B_NOW),
                extract_stuck_features(FakeStore(task=None), "missing"),
                extract_duration_features(FakeStore(task=None), "missing"),
                extract_features_from_buffer([], as_of_ms=B_NOW),
            ]

        for vector in vectors:
            for name, value in vector.items():
                assert isinstance(value, float), f"{name} is {type(value).__name__}, not float"
                assert not isinstance(value, bool)


class TestEdgeCases:
    """T015 -- the spec's Edge Cases section. None of these may raise."""

    def test_empty_event_list_preserves_the_documented_behavior(self) -> None:
        task = make_task()

        with frozen_clock(FAR_FUTURE):
            stuck = extract_stuck_features_from_data(task, [], as_of_ms=REF_END)
            dur = extract_duration_features_from_data(task, [], as_of_ms=REF_END)

        # No events, so phase start falls back to started_at and the whole
        # task window counts as time in phase.
        assert stuck["time_in_phase_sec"] == pytest.approx(3600.0)
        assert stuck["edit_velocity"] == 0.0
        assert stuck["file_switch_rate"] == 0.0
        assert stuck["session_length_sec"] == pytest.approx(3600.0)
        assert stuck["time_since_last_commit_sec"] == pytest.approx(3600.0)
        assert dur["total_edits"] == 0.0
        assert dur["file_count"] == 2.0

    def test_no_phase_change_event_falls_back_to_started_at(self) -> None:
        task = make_task()
        events = [e for e in make_events() if e["kind"] != "phase_change"]

        with frozen_clock(FAR_FUTURE):
            v = extract_stuck_features_from_data(task, events, as_of_ms=REF_END)

        assert v["time_in_phase_sec"] == pytest.approx(3600.0)  # REF_END - started_at

    def test_no_commit_events_falls_back_to_session_length(self) -> None:
        task = make_task()
        events = [e for e in make_events() if e["kind"] != "commit"]

        with frozen_clock(FAR_FUTURE):
            v = extract_stuck_features_from_data(task, events, as_of_ms=REF_END)

        assert v["time_since_last_commit_sec"] == v["session_length_sec"]
        assert v["time_since_last_commit_sec"] == pytest.approx(3600.0)

    def test_events_missing_a_ts_key_are_retained(self) -> None:
        """`ts` defaults to 0, which is at-or-before any sane reference time."""
        task = make_task()
        events = [
            {"id": 1, "kind": "edit", "source": "editor", "payload": {"file": "main.py"}},
            {"id": 2, "kind": "edit", "source": "editor", "payload": {"file": "utils.py"}, "ts": E_EDIT_2},
        ]

        with frozen_clock(FAR_FUTURE):
            stuck = extract_stuck_features_from_data(task, events, as_of_ms=REF_END)
            dur = extract_duration_features_from_data(task, events, as_of_ms=REF_END)

        assert stuck["edit_velocity"] == pytest.approx(2.0 / 60.0)
        assert stuck["file_switch_rate"] == pytest.approx(1.0)
        assert dur["total_edits"] == 2.0

    def test_phase_change_missing_a_ts_key_does_not_move_phase_start(self) -> None:
        task = make_task()
        events = [{"id": 1, "kind": "phase_change", "source": "sigild", "payload": {}}]

        with frozen_clock(FAR_FUTURE):
            v = extract_stuck_features_from_data(task, events, as_of_ms=REF_END)

        assert v["time_in_phase_sec"] == pytest.approx(3600.0)

    def test_json_string_payloads_preserve_parse_or_ignore_behavior(self) -> None:
        """The task extractors ignore string payloads; they never parse them.

        This is the pre-existing contract and this feature does not change it:
        `file_switch_rate` counts only events whose payload is already a dict.
        """
        task = make_task()
        events = [
            {"id": 1, "kind": "edit", "source": "editor", "payload": json.dumps({"file": "main.py"}), "ts": E_EDIT_1},
            {"id": 2, "kind": "edit", "source": "editor", "payload": json.dumps({"file": "b.py"}), "ts": E_EDIT_2},
        ]

        with frozen_clock(FAR_FUTURE):
            stuck = extract_stuck_features_from_data(task, events, as_of_ms=REF_END)
            dur = extract_duration_features_from_data(task, events, as_of_ms=REF_END)

        assert stuck["edit_velocity"] == pytest.approx(2.0 / 60.0)  # still counted
        assert stuck["file_switch_rate"] == 0.0  # but no file names extracted
        assert dur["total_edits"] == 2.0

    def test_task_files_as_a_json_string_is_parsed(self) -> None:
        task = make_task(files=json.dumps({"main.py": 5, "utils.py": 3, "third.py": 1}))

        with frozen_clock(FAR_FUTURE):
            v = extract_duration_features_from_data(task, make_events(), as_of_ms=REF_END)

        assert v["file_count"] == 3.0

    def test_task_files_as_unparseable_string_is_ignored(self) -> None:
        task = make_task(files="not json at all")

        with frozen_clock(FAR_FUTURE):
            v = extract_duration_features_from_data(task, make_events(), as_of_ms=REF_END)

        assert v["file_count"] == 0.0

    def test_buffer_extractor_with_an_empty_buffer_returns_an_all_zero_vector(self) -> None:
        with frozen_clock(B_NOW):
            v = extract_features_from_buffer([])

        assert v == {
            "test_failure_count": 0.0,
            "time_in_phase_sec": 0.0,
            "edit_velocity": 0.0,
            "file_switch_rate": 0.0,
            "session_length_sec": 0.0,
            "time_since_last_commit_sec": 0.0,
        }

    def test_buffer_extractor_filters_before_the_empty_guard(self) -> None:
        """A reference time that excludes every buffered event.

        The lookahead filter runs before the empty check, so a full buffer can
        become an empty one. That must produce the documented all-zero vector
        rather than an IndexError off `events[0]`.
        """
        with frozen_clock(B_NOW):
            v = extract_features_from_buffer(make_buffer_events(), as_of_ms=0)

        assert v == {
            "test_failure_count": 0.0,
            "time_in_phase_sec": 0.0,
            "edit_velocity": 0.0,
            "file_switch_rate": 0.0,
            "session_length_sec": 0.0,
            "time_since_last_commit_sec": 0.0,
        }
