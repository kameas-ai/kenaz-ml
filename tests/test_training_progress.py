"""Training progress: first-run training, a persisted baseline, /introspect's
training block, 24 h prediction counts and manifest sample counts.

The scheduler used to record the completed-task count at startup and wait
for MIN_NEW_TASKS more, in memory only: every restart reset the wait and a
fresh install never trained. These tests pin the fix on both SQLite shapes.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kenaz_ml.datastore.sqlite import SqliteStore
from kenaz_ml.modelstore import LocalModelStore
from kenaz_ml.training import scheduler as scheduler_mod
from kenaz_ml.training.scheduler import MIN_NEW_TASKS, TrainingScheduler
from tests.test_activity_store import HARNESS_DDL, MIN

DAEMON_DDL = """
CREATE TABLE tasks (
    id TEXT PRIMARY KEY, repo_root TEXT, branch TEXT, phase TEXT, files TEXT,
    started_at INTEGER, last_active INTEGER, completed_at INTEGER,
    commit_count INTEGER, test_runs INTEGER, test_fails INTEGER
);
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, source TEXT, payload TEXT, ts INTEGER
);
CREATE TABLE ml_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT NOT NULL, result TEXT NOT NULL,
    confidence REAL NOT NULL, created_at INTEGER NOT NULL, expires_at INTEGER
);
CREATE TABLE ml_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, endpoint TEXT NOT NULL,
    routing TEXT NOT NULL, latency_ms INTEGER NOT NULL, ts INTEGER NOT NULL
);
"""

SHAPES = ("daemon", "harness")


def _now_ms() -> int:
    return int(time.time() * 1000)


class Db:
    def __init__(self, path: Path, shape: str) -> None:
        self.path = path
        self.shape = shape
        self._exec_script(HARNESS_DDL if shape == "harness" else DAEMON_DDL)

    def _exec_script(self, sql: str) -> None:
        conn = sqlite3.connect(str(self.path))
        try:
            conn.executescript(sql)
            conn.commit()
        finally:
            conn.close()

    def _exec(self, sql: str, params: tuple[Any, ...]) -> None:
        conn = sqlite3.connect(str(self.path))
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    def completed_tasks(self, n: int, *, prefix: str, completed_at: int) -> None:
        cols = "id, repo_root, branch, phase, files, started_at, last_active, completed_at"
        for i in range(n):
            values = (
                f"{prefix}{i}",
                "/tmp/repo",
                "main",
                "coding",
                "{}",
                completed_at - 30 * MIN,
                completed_at,
                completed_at,
            )
            if self.shape == "harness":
                self._exec(
                    f"INSERT INTO tasks (user_id, node_id, producer, {cols}) VALUES ('', '', 'harness', ?, ?, ?, ?, ?, ?, ?, ?)",
                    values,
                )
            else:
                self._exec(f"INSERT INTO tasks ({cols}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", values)

    def prediction(self, model: str, created_at: int, expires_at: int | None) -> None:
        self._exec(
            "INSERT INTO ml_predictions (model, result, confidence, created_at, expires_at) VALUES (?, '{}', 0.5, ?, ?)",
            (model, created_at, expires_at),
        )

    def retrain_rows(self) -> list[tuple[Any, ...]]:
        conn = sqlite3.connect(str(self.path))
        try:
            return conn.execute("SELECT kind, endpoint FROM ml_events WHERE kind = 'retrain'").fetchall()
        finally:
            conn.close()


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("KENAZ_ML_INSTALL_ROOT", str(tmp_path / "install"))


@pytest.fixture(params=SHAPES)
def db(request: pytest.FixtureRequest, tmp_path: Path) -> Db:
    return Db(tmp_path / f"{request.param}.db", request.param)


def _store(db: Db) -> SqliteStore:
    s = SqliteStore(db.path, active_task_window_ms=30 * MIN)
    s.ensure_tables()
    return s


# ---------------------------------------------------------------------------
# First run and the persisted baseline
# ---------------------------------------------------------------------------


def test_a_fresh_store_without_weights_trains_on_the_first_check(db: Db, tmp_path: Path) -> None:
    db.completed_tasks(2, prefix="t", completed_at=_now_ms() - 60 * MIN)
    ms = LocalModelStore(tmp_path / "models")
    store = _store(db)
    reloaded: list[bool] = []
    sched = TrainingScheduler(store, model_store=ms, reload_callback=lambda: reloaded.append(True))

    assert sched.progress()["last_result"] is None
    started = time.time()
    sched.check_and_retrain()

    # Two tasks are far below MIN_NEW_TASKS; it trained anyway (synthetic bootstrap).
    assert ms.exists("stuck") and ms.exists("duration")
    assert reloaded == [True]
    assert db.retrain_rows() == [("retrain", "scheduler")]
    progress = sched.progress()
    assert progress["last_result"] == "ok"
    assert progress["baseline"] == 2
    assert progress["completed_tasks"] == 2
    assert progress["threshold"] == MIN_NEW_TASKS
    # The next check is one interval after this one started.
    assert progress["next_check_at"] == pytest.approx(started + sched.check_interval_sec, abs=2)
    store.close()


def test_weights_present_waits_for_new_tasks(db: Db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ms = LocalModelStore(tmp_path / "models")
    ms.save("stuck", b"x")
    ms.save("duration", b"x")
    store = _store(db)
    sched = TrainingScheduler(store, model_store=ms)
    calls: list[int] = []
    monkeypatch.setattr(scheduler_mod.Trainer, "train_all", lambda self: calls.append(1) or {"duration_sec": 0})

    sched.check_and_retrain()

    assert calls == []
    assert sched.progress()["last_result"] is None
    store.close()


def test_a_restart_keeps_the_baseline(db: Db, tmp_path: Path) -> None:
    db.completed_tasks(3, prefix="old", completed_at=_now_ms() - 60 * MIN)
    ms = LocalModelStore(tmp_path / "models")
    first = _store(db)
    TrainingScheduler(first, model_store=ms).check_and_retrain()
    first.close()

    # Four more tasks complete after the retrain, then the engine restarts.
    db.completed_tasks(4, prefix="new", completed_at=_now_ms() + 60_000)
    second = _store(db)
    sched = TrainingScheduler(second, model_store=ms)
    progress = sched.progress()

    # The old in-memory baseline would have been 7 (the count at startup).
    assert progress["baseline"] == 3
    assert progress["completed_tasks"] == 7
    assert progress["last_result"] == "ok"
    assert progress["last_retrain_at"] is not None
    second.close()


def test_the_restored_baseline_still_triggers_a_retrain(
    db: Db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db.completed_tasks(1, prefix="old", completed_at=_now_ms() - 60 * MIN)
    ms = LocalModelStore(tmp_path / "models")
    first = _store(db)
    TrainingScheduler(first, model_store=ms).check_and_retrain()
    first.close()

    db.completed_tasks(MIN_NEW_TASKS, prefix="new", completed_at=_now_ms() + 60_000)
    second = _store(db)
    sched = TrainingScheduler(second, model_store=ms)
    sched._last_retrain = 0.0  # the hourly interval is not what this test is about
    calls: list[int] = []
    monkeypatch.setattr(scheduler_mod.Trainer, "train_all", lambda self: calls.append(1) or {"duration_sec": 0})

    sched.check_and_retrain()

    assert calls == [1]
    assert sched.progress()["baseline"] == 1 + MIN_NEW_TASKS
    second.close()


def test_a_failed_first_run_reports_failed_and_retries_next_check(
    db: Db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(db)
    sched = TrainingScheduler(store, model_store=LocalModelStore(tmp_path / "models"))
    calls: list[int] = []

    def boom(self: Any) -> dict:
        calls.append(1)
        raise RuntimeError("fit failed")

    monkeypatch.setattr(scheduler_mod.Trainer, "train_all", boom)
    sched.check_and_retrain()
    assert sched.progress()["last_result"] == "failed"
    sched.check_and_retrain()
    assert calls == [1, 1]
    store.close()


def test_a_store_without_the_new_readers_falls_back_to_the_current_count() -> None:
    class Legacy:
        def count_completed_tasks(self) -> int:
            return 5

    sched = TrainingScheduler(Legacy())  # type: ignore[arg-type]
    assert sched.progress()["baseline"] == 5
    assert sched.progress()["last_result"] is None


# ---------------------------------------------------------------------------
# Store readers, both shapes
# ---------------------------------------------------------------------------


def test_predictions_count_by_created_at_regardless_of_expiry(db: Db) -> None:
    now = _now_ms()
    db.prediction("stuck", now - 10 * MIN, now - 8 * MIN)  # expired, in window
    db.prediction("stuck", now - MIN, now + MIN)  # live
    db.prediction("suggest", now - 2 * MIN, None)
    db.prediction("stuck", now - 25 * 60 * MIN, None)  # older than 24 h
    store = _store(db)
    assert store.count_predictions_since(now - 24 * 60 * MIN) == {"stuck": 2, "suggest": 1}
    store.close()


def test_last_ml_event_ts_and_tasks_before(db: Db) -> None:
    store = _store(db)
    assert store.last_ml_event_ts("retrain", "scheduler") is None
    store.insert_ml_event("retrain", "scheduler", "local", 5)
    store.insert_ml_event("prediction", "poller", "local", 1)
    store.commit()
    ts = store.last_ml_event_ts("retrain", "scheduler")
    assert ts is not None and abs(ts - _now_ms()) < 5_000
    db.completed_tasks(2, prefix="a", completed_at=ts - 1)
    db.completed_tasks(1, prefix="b", completed_at=ts)
    db.completed_tasks(3, prefix="c", completed_at=ts + 1)
    assert store.count_completed_tasks_before(ts) == 3
    store.close()


def test_readers_tolerate_missing_tables(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "empty.db", active_task_window_ms=30 * MIN)
    assert store.last_ml_event_ts("retrain", "scheduler") is None
    assert store.count_predictions_since(0) == {}
    assert store.count_completed_tasks_before(_now_ms()) == 0
    store.close()


# ---------------------------------------------------------------------------
# /introspect
# ---------------------------------------------------------------------------


def test_introspect_reports_the_training_block_and_24h_counts(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from kenaz_ml import config
    from kenaz_ml.app import create_app

    path = config.local_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = Db(path, "daemon")
    now = _now_ms()
    db.prediction("stuck", now - 10 * MIN, now - 9 * MIN)  # expired: still counts
    db.prediction("stuck", now - 25 * 60 * MIN, None)  # outside 24 h

    with TestClient(create_app()) as client:
        data = client.get("/introspect").json()

    training = data["training"]
    assert training["threshold"] == MIN_NEW_TASKS
    assert training["completed_tasks"] == 0
    assert training["baseline"] == 0
    assert set(training) == {
        "completed_tasks",
        "baseline",
        "threshold",
        "last_result",
        "last_retrain_at",
        "next_check_at",
        "in_progress",
    }
    by_name = {m["name"]: m for m in data["models"]}
    assert by_name["stuck"]["recent_predictions"] == 1


def test_training_progress_renders_iso_times() -> None:
    from datetime import datetime

    from kenaz_ml.routes import _training_progress

    fake = SimpleNamespace(
        progress=lambda: {
            "completed_tasks": 12,
            "baseline": 4,
            "threshold": 10,
            "last_result": "ok",
            "last_retrain_at": 1_700_000_000.0,
            "next_check_at": None,
            "in_progress": False,
        }
    )
    out = _training_progress(fake)
    assert out is not None
    assert out.completed_tasks == 12 and out.baseline == 4
    assert datetime.fromisoformat(out.last_retrain_at or "").tzinfo is not None
    assert out.next_check_at is None
    assert _training_progress(None) is None


def test_sample_count_comes_from_the_serving_manifest() -> None:
    from kenaz_ml.routes import _manifest_sample_count

    def resolution(n: int | None, served: bool = True) -> Any:
        return SimpleNamespace(served=served, manifest=SimpleNamespace(training=SimpleNamespace(n_samples=n)))

    assert _manifest_sample_count(resolution(500)) == 500
    assert _manifest_sample_count(resolution(None)) is None
    assert _manifest_sample_count(resolution(500, served=False)) is None
    assert _manifest_sample_count(SimpleNamespace(served=True, manifest=None)) is None
    assert _manifest_sample_count(None) is None


# ---------------------------------------------------------------------------
# Poller failures are visible
# ---------------------------------------------------------------------------


def test_poller_failures_warn_once_per_streak(caplog: pytest.LogCaptureFixture) -> None:
    import asyncio

    from kenaz_ml.poller import EventPoller

    class Flaky:
        def __init__(self) -> None:
            self.n = 0

        def get_cursor(self) -> int:
            self.n += 1
            if self.n <= 3:
                raise sqlite3.OperationalError("no such table: ml_cursor")
            if self.n == 4:
                poller.stop()
            return 0

        def get_events_since(self, *a: Any, **k: Any) -> list:
            return []

    models = {k: None for k in ("stuck", "activity", "workflow", "duration", "quality")}
    poller = EventPoller(store=Flaky(), models=models, poll_interval_sec=0)  # type: ignore[arg-type]
    with caplog.at_level(logging.DEBUG, logger="kenaz_ml.poller"):
        asyncio.run(poller.run())

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "OperationalError" in warnings[0].getMessage()
    assert any("recovered after 3" in r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
