"""Training reference-time wiring (WP03: FR-007, FR-008, FR-012).

A training example describes an instant in the past. These tests pin down that
both trainers measure each example against *its own* completion moment rather
than against wall clock, and that a task which cannot supply an honest
reference time is absent from the training matrix entirely.

Every test runs under a frozen clock parked a full year past the newest
fixture, so any wall-clock regression shows up as a year-scale elapsed value
instead of the seconds-scale value asserted here.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from sigil_ml import features as features_mod
from sigil_ml.models.duration import FEATURE_NAMES as DURATION_FEATURES
from sigil_ml.models.duration import DurationEstimator
from sigil_ml.models.stuck import FEATURE_NAMES as STUCK_FEATURES
from sigil_ml.models.stuck import StuckPredictor
from sigil_ml.training.cloud_trainer import CloudTrainer
from sigil_ml.training.trainer import Trainer, _reference_time_for

# --- Fixture timeline -------------------------------------------------------

BASE_MS = 1_700_000_000_000  # 2023-11-14T22:13:20Z
HOUR_MS = 3_600_000
FROZEN_NOW_MS = BASE_MS + 365 * 24 * HOUR_MS  # one year after every fixture

USABLE_COUNT = 12  # tasks carrying a real completed_at
FALLBACK_LAST_ACTIVE_MS = BASE_MS + 50 * HOUR_MS

# Distinguishing feature values, used to prove a row is absent from a matrix
# rather than merely counting rows.
UNUSABLE_TEST_FAILS = 99
UNUSABLE_BRANCH = "u" * 77


class _FrozenTime:
    """Stand-in for the `time` module inside features.py.

    Patching the attribute on the module under test keeps the freeze scoped --
    the real `time` module is left alone for pytest and everything else.
    """

    def __init__(self, now_ms: int) -> None:
        self._now_sec = now_ms / 1000.0

    def time(self) -> float:
        return self._now_sec

    def localtime(self, secs: float | None = None):  # noqa: ANN201 - mirrors time.localtime
        return time.localtime(self._now_sec if secs is None else secs)


@pytest.fixture(autouse=True)
def _frozen_clock(monkeypatch):
    """Freeze wall clock inside the extractors, a year past every fixture."""
    monkeypatch.setattr(features_mod, "time", _FrozenTime(FROZEN_NOW_MS))


@pytest.fixture(autouse=True)
def _isolate_models(tmp_path, monkeypatch):
    """Keep any model weights that do get written out of the real config dir."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))


# --- Fixture construction ---------------------------------------------------


def _events_for(task_id: str, index: int, reference_ms: int) -> list[dict]:
    """Events whose aggregates depend on the reference time in a checkable way.

    - a `phase_change` at `reference_ms - (index + 1) * 60_000`, so
      `time_in_phase_sec` must come out as `(index + 1) * 60.0` for this task
      and no other;
    - a `phase_change` *after* the reference time, which the no-lookahead
      filter must discard (were it honored, `time_in_phase_sec` would clamp
      to 0.0);
    - `index + 1` edits before the reference time and 5 after, so
      `total_edits` must come out as `index + 1` rather than `index + 6`.

    No `commit` events: production data contains none of kind "commit" (the
    live sigild DB carries only file/process/hyprland/browser/terminal/power),
    so `time_since_last_commit_sec` always falls through to
    `session_length_sec` in practice. Fixtures here match that reality.
    """
    events: list[dict] = [
        {
            "kind": "phase_change",
            "source": "sigild",
            "payload": {"phase": "coding"},
            "ts": reference_ms - (index + 1) * 60_000,
        },
        {"kind": "phase_change", "source": "sigild", "payload": {"phase": "future"}, "ts": reference_ms + HOUR_MS},
    ]
    for n in range(index + 1):
        events.append(
            {
                "kind": "edit",
                "source": "editor",
                "payload": {"file": f"{task_id}-{n}.py"},
                "ts": reference_ms - (n + 1) * 1_000,
            }
        )
    for n in range(5):
        events.append(
            {
                "kind": "edit",
                "source": "editor",
                "payload": {"file": f"{task_id}-future-{n}.py"},
                "ts": reference_ms + (n + 1) * 1_000,
            }
        )
    return events


def _build_fixtures() -> tuple[list[dict], dict[str, list[dict]]]:
    """Twelve normal tasks, one `last_active`-only task, one unusable task."""
    tasks: list[dict] = []
    events: dict[str, list[dict]] = {}

    for i in range(USABLE_COUNT):
        completed_at = BASE_MS + i * HOUR_MS
        task_id = f"task-{i}"
        tasks.append(
            {
                "id": task_id,
                "repo_root": "/tmp/repo",
                "branch": "feature/reference-time",
                "phase": "done",
                "files": json.dumps({"main.py": 5}),
                "started_at": completed_at - 2 * HOUR_MS,
                "last_active": completed_at,
                "completed_at": completed_at,
                # Alternating labels keep both classes present in y.
                "test_fails": 5 if i % 2 == 0 else 0,
            }
        )
        events[task_id] = _events_for(task_id, i, completed_at)

    # completed_at absent -> must fall back to last_active, and still land in
    # the matrix measured against that fallback.
    tasks.append(
        {
            "id": "task-fallback",
            "repo_root": "/tmp/repo",
            "branch": "feature/fallback",
            "phase": "done",
            "files": json.dumps({"main.py": 5}),
            "started_at": FALLBACK_LAST_ACTIVE_MS - 2 * HOUR_MS,
            "last_active": FALLBACK_LAST_ACTIVE_MS,
            "completed_at": 0,
            "test_fails": 5,
        }
    )
    # index 0 -> time_in_phase_sec == 60.0 measured from last_active
    events["task-fallback"] = _events_for("task-fallback", 0, FALLBACK_LAST_ACTIVE_MS)

    # Neither timestamp: zero is not a real epoch, so this task cannot yield an
    # honest reference time and must be excluded from training (FR-012).
    tasks.append(
        {
            "id": "task-unusable",
            "repo_root": "/tmp/repo",
            "branch": UNUSABLE_BRANCH,
            "phase": "done",
            "files": json.dumps({"main.py": 5}),
            "started_at": 0,
            "last_active": 0,
            "completed_at": 0,
            "test_fails": UNUSABLE_TEST_FAILS,
        }
    )
    events["task-unusable"] = _events_for("task-unusable", 0, BASE_MS)

    return tasks, events


class FakeStore:
    """Minimal DataStore covering exactly what the local trainer calls."""

    def __init__(self, tasks: list[dict], events: dict[str, list[dict]]) -> None:
        self._tasks = {t["id"]: t for t in tasks}
        self._events = events

    def get_completed_task_ids(self) -> list[str]:
        return list(self._tasks)

    def get_task_by_id(self, task_id: str) -> dict | None:
        return self._tasks.get(task_id)

    def get_events_for_task(self, task_id: str, since: int | None = None) -> list[dict]:
        return list(self._events.get(task_id, []))

    def get_completed_tasks_with_timestamps(self) -> list[dict]:
        """Mirrors the real query: only id/started_at/completed_at, NOT NULL.

        A stored `0` passes an `IS NOT NULL` filter, so the unusable task is
        returned here and the trainer -- not the query -- must reject it.
        """
        return [
            {"id": t["id"], "started_at": t["started_at"], "completed_at": t["completed_at"]}
            for t in self._tasks.values()
            if t["started_at"] is not None and t["completed_at"] is not None
        ]


class _Capture:
    """Records the matrix handed to a model's train() without fitting it."""

    def __init__(self) -> None:
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def __call__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y


@pytest.fixture
def stuck_capture(monkeypatch) -> _Capture:
    capture = _Capture()
    monkeypatch.setattr(StuckPredictor, "train", lambda self, X, y: capture(X, y))
    return capture


@pytest.fixture
def duration_capture(monkeypatch) -> _Capture:
    capture = _Capture()
    monkeypatch.setattr(DurationEstimator, "train", lambda self, X, y: capture(X, y))
    return capture


# ===========================================================================
# T016 -- the resolver itself
# ===========================================================================


class TestReferenceTimeFor:
    def test_prefers_completed_at(self) -> None:
        assert _reference_time_for({"completed_at": 1234, "last_active": 9999}) == 1234

    def test_falls_back_to_last_active_when_completed_at_missing(self) -> None:
        assert _reference_time_for({"last_active": 9999}) == 9999

    def test_falls_back_to_last_active_when_completed_at_zero(self) -> None:
        assert _reference_time_for({"completed_at": 0, "last_active": 9999}) == 9999

    def test_falls_back_when_completed_at_none(self) -> None:
        assert _reference_time_for({"completed_at": None, "last_active": 9999}) == 9999

    def test_returns_none_when_both_absent(self) -> None:
        assert _reference_time_for({}) is None

    def test_returns_none_when_both_zero(self) -> None:
        assert _reference_time_for({"completed_at": 0, "last_active": 0}) is None

    def test_returns_none_when_both_none(self) -> None:
        assert _reference_time_for({"completed_at": None, "last_active": None}) is None

    def test_no_wall_clock_fallback(self) -> None:
        """The whole point of FR-012: never substitute "now" for a missing time."""
        now_ms = int(time.time() * 1000)
        resolved = _reference_time_for({"completed_at": 0, "last_active": 0})
        assert resolved is None
        assert resolved != now_ms

    def test_coerces_to_int(self) -> None:
        assert _reference_time_for({"completed_at": 1234.0}) == 1234
        assert isinstance(_reference_time_for({"completed_at": 1234.0}), int)


# ===========================================================================
# T017 -- local trainer, stuck path
# ===========================================================================


class TestLocalStuckTraining:
    def test_unusable_task_absent_from_matrix(self, stuck_capture) -> None:
        tasks, events = _build_fixtures()
        trainer = Trainer(FakeStore(tasks, events))

        samples = trainer._train_stuck()

        X = stuck_capture.X
        assert X is not None
        # 12 completed + 1 last_active fallback; the unusable task is dropped.
        assert X.shape[0] == USABLE_COUNT + 1 == samples
        # And it is dropped by identity, not just by count: no row carries its
        # distinguishing test-failure count.
        col = STUCK_FEATURES.index("test_failure_count")
        assert float(UNUSABLE_TEST_FAILS) not in set(X[:, col].tolist())

    def test_elapsed_features_use_each_task_own_reference_time(self, stuck_capture) -> None:
        tasks, events = _build_fixtures()
        trainer = Trainer(FakeStore(tasks, events))

        trainer._train_stuck()

        X = stuck_capture.X
        col = STUCK_FEATURES.index("time_in_phase_sec")
        observed = sorted(X[:, col].tolist())
        # task-i -> (i+1)*60s; task-fallback -> 60s (measured from last_active).
        expected = sorted([60.0] + [(i + 1) * 60.0 for i in range(USABLE_COUNT)])
        assert observed == expected

    def test_no_row_measured_against_wall_clock(self, stuck_capture) -> None:
        tasks, events = _build_fixtures()
        trainer = Trainer(FakeStore(tasks, events))

        trainer._train_stuck()

        X = stuck_capture.X
        col = STUCK_FEATURES.index("time_in_phase_sec")
        # Wall clock would put every row at roughly a year (>= 3e7 sec).
        assert X[:, col].max() < 100_000.0

    def test_lookahead_events_excluded(self, stuck_capture) -> None:
        """A post-reference phase_change would clamp time_in_phase to zero."""
        tasks, events = _build_fixtures()
        trainer = Trainer(FakeStore(tasks, events))

        trainer._train_stuck()

        col = STUCK_FEATURES.index("time_in_phase_sec")
        assert stuck_capture.X[:, col].min() > 0.0

    def test_skip_logged_once(self, stuck_capture, caplog) -> None:
        tasks, events = _build_fixtures()
        trainer = Trainer(FakeStore(tasks, events))

        with caplog.at_level("INFO", logger="sigil_ml.training.trainer"):
            trainer._train_stuck()

        matches = [r for r in caplog.records if "no resolvable reference time" in r.getMessage()]
        assert len(matches) == 1
        assert "skipped 1 task(s)" in matches[0].getMessage()

    def test_synthetic_fallback_uses_usable_count(self, stuck_capture, caplog) -> None:
        """Eleven raw tasks, only nine usable -> synthetic, not a 9-row fit."""
        tasks, events = _build_fixtures()
        tasks = tasks[:9] + [dict(t, id=f"dead-{i}", completed_at=0, last_active=0) for i, t in enumerate(tasks[:2])]
        trainer = Trainer(FakeStore(tasks, events))

        with caplog.at_level("INFO", logger="sigil_ml.training.trainer"):
            samples = trainer._train_stuck()

        assert samples == 500  # synthetic
        assert stuck_capture.X.shape[0] == 500
        assert any("9 usable of 11" in r.getMessage() for r in caplog.records)

    def test_label_expression_behavior_preserved(self, stuck_capture) -> None:
        """The label still reads the same two features; only their values moved."""
        tasks, events = _build_fixtures()
        trainer = Trainer(FakeStore(tasks, events))

        trainer._train_stuck()

        X, y = stuck_capture.X, stuck_capture.y
        fails = STUCK_FEATURES.index("test_failure_count")
        phase = STUCK_FEATURES.index("time_in_phase_sec")
        for row, label in zip(X, y):
            expected = 1.0 if (row[fails] > 3 and row[phase] > 600) else 0.0
            assert label == expected


# ===========================================================================
# T018 -- local trainer, duration path
# ===========================================================================


class TestLocalDurationTraining:
    def test_unusable_task_absent_from_matrix(self, duration_capture) -> None:
        tasks, events = _build_fixtures()
        trainer = Trainer(FakeStore(tasks, events))

        samples = trainer._train_duration()

        X = duration_capture.X
        # Both zero-completed_at rows are unusable here: the duration label
        # needs completed_at, which the last_active fallback cannot supply.
        assert X.shape[0] == USABLE_COUNT == samples
        col = DURATION_FEATURES.index("branch_name_length")
        assert float(len(UNUSABLE_BRANCH)) not in set(X[:, col].tolist())

    def test_total_edits_reflect_each_task_own_reference_time(self, duration_capture) -> None:
        tasks, events = _build_fixtures()
        trainer = Trainer(FakeStore(tasks, events))

        trainer._train_duration()

        X = duration_capture.X
        col = DURATION_FEATURES.index("total_edits")
        observed = sorted(X[:, col].tolist())
        # task-i has i+1 edits before its completion and 5 after; the 5 must
        # not be counted. Wall clock would yield i+6 for every row.
        assert observed == sorted([float(i + 1) for i in range(USABLE_COUNT)])

    def test_duration_label_arithmetic_unchanged(self, duration_capture) -> None:
        tasks, events = _build_fixtures()
        trainer = Trainer(FakeStore(tasks, events))

        trainer._train_duration()

        y = duration_capture.y
        # Every fixture task spans exactly two hours.
        assert set(y.tolist()) == {120.0}

    def test_skip_logged_once(self, duration_capture, caplog) -> None:
        tasks, events = _build_fixtures()
        trainer = Trainer(FakeStore(tasks, events))

        with caplog.at_level("INFO", logger="sigil_ml.training.trainer"):
            trainer._train_duration()

        matches = [r for r in caplog.records if "duration training: skipped" in r.getMessage()]
        assert len(matches) == 1
        assert "skipped 2 task(s)" in matches[0].getMessage()


# ===========================================================================
# T019 -- cloud trainer, both paths
# ===========================================================================


class FakeModelStore:
    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def load(self, model_name: str) -> bytes | None:
        return self._store.get(model_name)

    def save(self, model_name: str, data: bytes) -> None:
        self._store[model_name] = data

    def exists(self, model_name: str) -> bool:
        return model_name in self._store


class TestCloudTraining:
    def _run(self, tasks, events):
        trainer = CloudTrainer(data_store=None, model_store=FakeModelStore())
        return trainer._train_models_from_tasks(tasks, events, "tenant-a")

    def test_stuck_matrix_excludes_unusable_task(self, stuck_capture) -> None:
        tasks, events = _build_fixtures()

        self._run(tasks, events)

        X = stuck_capture.X
        assert X is not None
        assert X.shape[0] == USABLE_COUNT + 1
        col = STUCK_FEATURES.index("test_failure_count")
        assert float(UNUSABLE_TEST_FAILS) not in set(X[:, col].tolist())

    def test_stuck_elapsed_features_use_per_task_reference_time(self, stuck_capture) -> None:
        tasks, events = _build_fixtures()

        self._run(tasks, events)

        col = STUCK_FEATURES.index("time_in_phase_sec")
        observed = sorted(stuck_capture.X[:, col].tolist())
        expected = sorted([60.0] + [(i + 1) * 60.0 for i in range(USABLE_COUNT)])
        assert observed == expected

    def test_duration_matrix_excludes_unusable_task(self, duration_capture) -> None:
        tasks, events = _build_fixtures()

        self._run(tasks, events)

        X = duration_capture.X
        assert X is not None
        # The cloud duration label tolerates a missing completed_at (it falls
        # back to 60.0), so the last_active-only task stays; only the task with
        # no resolvable reference time at all is dropped.
        assert X.shape[0] == USABLE_COUNT + 1
        col = DURATION_FEATURES.index("branch_name_length")
        assert float(len(UNUSABLE_BRANCH)) not in set(X[:, col].tolist())

    def test_duration_total_edits_use_per_task_reference_time(self, duration_capture) -> None:
        tasks, events = _build_fixtures()

        self._run(tasks, events)

        col = DURATION_FEATURES.index("total_edits")
        observed = sorted(duration_capture.X[:, col].tolist())
        expected = sorted([1.0] + [float(i + 1) for i in range(USABLE_COUNT)])
        assert observed == expected

    def test_skips_logged_once_per_model(self, stuck_capture, duration_capture, caplog) -> None:
        tasks, events = _build_fixtures()

        with caplog.at_level("INFO", logger="sigil_ml.training.cloud_trainer"):
            self._run(tasks, events)

        messages = [r.getMessage() for r in caplog.records]
        assert len([m for m in messages if m.startswith("stuck training: skipped")]) == 1
        assert len([m for m in messages if m.startswith("duration training: skipped")]) == 1
        assert all("tenant-a" in m for m in messages if "training: skipped" in m)

    def test_resolver_is_shared_not_duplicated(self) -> None:
        """FR-008 depends on the cloud path applying the *same* rule."""
        from sigil_ml.training import cloud_trainer as cloud_mod
        from sigil_ml.training import trainer as local_mod

        assert cloud_mod._reference_time_for is local_mod._reference_time_for
