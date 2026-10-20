"""SqliteStore against both database shapes it reads.

- ``sigild``: the daemon's ``data.db`` (no tenancy columns).
- ``harness``: the harness's local activity store, built from the harness's
  own DDL (``tests/fixtures/harness_activity_db.sql``).

Every existing query must behave the same on both; the hosted-advice readers
must read real rows from either; writes must open ``BEGIN IMMEDIATE``.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from kenaz_ml.datastore.sqlite import SqliteStore

HARNESS_DDL = (Path(__file__).parent / "fixtures" / "harness_activity_db.sql").read_text()

SIGILD_DDL = """
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

SHAPES = ("sigild", "harness")
NOW = int(time.time() * 1000)
MIN = 60_000


class Db:
    """A database of one shape, with shape-aware inserts for the producer-owned tables."""

    def __init__(self, path: Path, shape: str) -> None:
        self.path = path
        self.shape = shape
        conn = sqlite3.connect(str(path))
        conn.executescript(HARNESS_DDL if shape == "harness" else SIGILD_DDL)
        conn.commit()
        conn.close()

    def _exec(self, sql: str, params: tuple[Any, ...]) -> int:
        conn = sqlite3.connect(str(self.path))
        try:
            cur = conn.execute(sql, params)
            conn.commit()
            return int(cur.lastrowid or 0)
        finally:
            conn.close()

    def task(
        self,
        task_id: str,
        *,
        phase: str = "coding",
        started_at: int = NOW - 60 * MIN,
        last_active: int = NOW - MIN,
        completed_at: int | None = None,
        files: dict[str, int] | None = None,
        commit_count: int = 1,
        test_runs: int = 4,
        test_fails: int = 2,
    ) -> None:
        values = (
            task_id,
            "/tmp/repo",
            "main",
            phase,
            json.dumps(files or {"a.py": 2, "b.py": 1}),
            started_at,
            last_active,
            completed_at,
            commit_count,
            test_runs,
            test_fails,
        )
        cols = (
            "id, repo_root, branch, phase, files, started_at, last_active, completed_at, "
            "commit_count, test_runs, test_fails"
        )
        if self.shape == "harness":
            self._exec(
                f"INSERT INTO tasks (user_id, node_id, producer, {cols}) VALUES ('', '', 'harness', {', '.join('?' * 11)})",
                values,
            )
        else:
            self._exec(f"INSERT INTO tasks ({cols}) VALUES ({', '.join('?' * 11)})", values)

    def event(self, kind: str, payload: dict[str, Any], ts: int, source: str = "harness.agent") -> int:
        body = json.dumps(payload)
        if self.shape == "harness":
            conn = sqlite3.connect(str(self.path))
            try:
                cur = conn.execute(
                    "INSERT INTO events (kind, source, payload, ts, user_id, node_id, device_event_id, producer) "
                    "VALUES (?, ?, ?, ?, '', '', 0, 'harness')",
                    (kind, source, body, ts),
                )
                row_id = int(cur.lastrowid or 0)
                # Local rows carry device_event_id = id (the dedup key needs it unique).
                conn.execute("UPDATE events SET device_event_id = id WHERE id = ?", (row_id,))
                conn.commit()
                return row_id
            finally:
                conn.close()
        return self._exec(
            "INSERT INTO events (kind, source, payload, ts) VALUES (?, ?, ?, ?)", (kind, source, body, ts)
        )

    def rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        conn = sqlite3.connect(str(self.path))
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()


@pytest.fixture(params=SHAPES)
def db(request: pytest.FixtureRequest, tmp_path: Path) -> Db:
    return Db(tmp_path / f"{request.param}.db", request.param)


@pytest.fixture
def store(db: Db):
    s = SqliteStore(db.path, active_task_window_ms=30 * MIN)
    s.ensure_tables()
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Every existing query, on both shapes
# ---------------------------------------------------------------------------


def test_harness_fixture_is_the_harness_shape(tmp_path: Path) -> None:
    db = Db(tmp_path / "h.db", "harness")
    cols = {r[1] for r in db.rows("PRAGMA table_info(tasks)")}
    assert {"user_id", "node_id", "producer"} <= cols
    assert {r[1] for r in db.rows("PRAGMA table_info(ml_predictions)")} >= {"user_id", "created_at"}


def test_ensure_tables_creates_cursor_and_signals_and_is_idempotent(db: Db, store: SqliteStore) -> None:
    store.ensure_tables()
    names = {r[0] for r in db.rows("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"ml_cursor", "ml_signals"} <= names
    assert db.rows("SELECT id, last_event_id FROM ml_cursor") == [(1, 0)]


def test_cursor_round_trip(store: SqliteStore) -> None:
    assert store.get_cursor() == 0
    store.update_cursor(17)
    store.commit()
    assert store.get_cursor() == 17


def test_get_events_since_and_kind_exclusion(db: Db, store: SqliteStore) -> None:
    a = db.event("file", {"path": "x.py"}, NOW - 5 * MIN)
    b = db.event("advice.features", {"task": "agent-0000000000000001"}, NOW - 4 * MIN)
    c = db.event("terminal", {"cmd": "pytest"}, NOW - 3 * MIN)
    rows = store.get_events_since(0)
    assert [r["id"] for r in rows] == [a, b, c]
    assert set(rows[0]) == {"id", "kind", "source", "payload", "ts"}
    assert isinstance(rows[0]["payload"], str)
    assert [r["id"] for r in store.get_events_since(a, exclude_kinds={"advice.features"})] == [c]
    assert [r["id"] for r in store.get_events_since(0, limit=1)] == [a]


def test_task_reads(db: Db, store: SqliteStore) -> None:
    db.task("t-open", last_active=NOW - MIN)
    db.task("t-done", started_at=NOW - 120 * MIN, last_active=NOW - 10 * MIN, completed_at=NOW - 10 * MIN)
    task = store.get_task_by_id("t-open")
    assert task is not None and task["id"] == "t-open" and json.loads(task["files"]) == {"a.py": 2, "b.py": 1}
    assert store.get_task_by_id("missing") is None
    assert store.get_session_info("t-open") == {"started_at": NOW - 60 * MIN, "phase": "coding", "test_fails": 2}
    assert store.get_session_info("missing") is None
    assert store.get_quality_task_stats() == {"test_runs": 4, "test_fails": 2, "commit_count": 1}
    assert store.get_completed_task_ids() == ["t-done"]
    assert store.get_completed_tasks_with_timestamps() == [
        {"id": "t-done", "started_at": NOW - 120 * MIN, "completed_at": NOW - 10 * MIN}
    ]
    assert store.count_completed_tasks() == 1


def test_get_events_for_task_parses_payloads_in_the_window(db: Db, store: SqliteStore) -> None:
    db.task("t1", started_at=NOW - 30 * MIN, last_active=NOW - MIN)
    db.event("file", {"path": "before.py"}, NOW - 40 * MIN)
    inside = db.event("file", {"path": "in.py"}, NOW - 20 * MIN)
    rows = store.get_events_for_task("t1")
    assert [r["id"] for r in rows] == [inside]
    assert rows[0]["payload"] == {"path": "in.py"}
    assert store.get_events_for_task("missing") == []


def test_features_extract_on_both_shapes(db: Db, store: SqliteStore) -> None:
    from kenaz_ml.features import extract_duration_features

    db.task("t1", started_at=NOW - 30 * MIN, last_active=NOW - MIN)
    db.event("edit", {"file": "a.py"}, NOW - 20 * MIN)
    vector = extract_duration_features(store, "t1")
    assert vector is not None


def test_writes_and_status(db: Db, store: SqliteStore) -> None:
    store.insert_prediction("stuck", {"probability": 0.7}, 0.7, 60)
    store.insert_ml_event("prediction", "poller", "local", 3)
    signal_id = store.insert_signal("velocity_deviation", 0.8, {"k": 1})
    store.commit()
    assert signal_id >= 1
    assert db.rows("SELECT model, confidence FROM ml_predictions") == [("stuck", 0.7)]
    assert db.rows("SELECT kind, endpoint FROM ml_events") == [("prediction", "poller")]
    status = store.get_status_data()
    assert status["cursor"] == {"last_event_id": 0, "updated_at": 0}
    assert [p["model"] for p in status["latest_predictions"]] == ["stuck"]
    if db.shape == "harness":
        assert db.rows("SELECT user_id FROM ml_predictions") == [("",)]


def test_signal_feedback_is_empty_without_a_suggestions_table(store: SqliteStore) -> None:
    assert store.get_signal_feedback(0) == []


# ---------------------------------------------------------------------------
# Active-task gate
# ---------------------------------------------------------------------------


def test_active_task_is_the_most_recent_uncompleted_task_in_the_window(db: Db, store: SqliteStore) -> None:
    db.task("older", last_active=NOW - 20 * MIN)
    db.task("newer", last_active=NOW - 2 * MIN)
    db.task("done", last_active=NOW - MIN, completed_at=NOW - MIN)
    assert store.get_active_task(now_ms=NOW) == "newer"


def test_an_idle_phase_does_not_hide_a_recently_active_task(db: Db, store: SqliteStore) -> None:
    db.task("idle-but-live", phase="idle", last_active=NOW - MIN)
    assert store.get_active_task(now_ms=NOW) == "idle-but-live"


def test_a_stale_uncompleted_task_is_not_active(db: Db, store: SqliteStore) -> None:
    db.task("stale", last_active=NOW - 31 * MIN)
    assert store.get_active_task(now_ms=NOW) is None
    db.task("edge", last_active=NOW - 30 * MIN)
    assert store.get_active_task(now_ms=NOW) == "edge", "the window boundary is inclusive"


def test_the_window_is_configurable(db: Db, monkeypatch: pytest.MonkeyPatch) -> None:
    db.task("t", last_active=NOW - 45 * MIN)
    monkeypatch.setenv("KENAZ_ACTIVE_TASK_WINDOW_MIN", "60")
    assert SqliteStore(db.path).get_active_task(now_ms=NOW) == "t"
    monkeypatch.delenv("KENAZ_ACTIVE_TASK_WINDOW_MIN")
    assert SqliteStore(db.path).get_active_task(now_ms=NOW) is None


@pytest.mark.parametrize("raw", ["0", "-5", "abc", "1.5"])
def test_an_unusable_window_falls_back_to_30_minutes(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    from kenaz_ml import config

    monkeypatch.setenv("KENAZ_ACTIVE_TASK_WINDOW_MIN", raw)
    assert config.active_task_window_ms() == 30 * MIN


# ---------------------------------------------------------------------------
# Cursor after the producer deletes rows
# ---------------------------------------------------------------------------


def test_a_cursor_past_the_max_id_reads_as_nothing_new(db: Db, store: SqliteStore) -> None:
    ids = [db.event("file", {"path": f"{i}.py"}, NOW - i) for i in range(5)]
    store.update_cursor(max(ids))
    store.commit()
    conn = sqlite3.connect(str(db.path))
    conn.execute("DELETE FROM events")  # the producer prunes its rows
    conn.commit()
    conn.close()
    assert store.get_events_since(store.get_cursor()) == []
    # AUTOINCREMENT never reuses an id, so the next row is past the cursor.
    new_id = db.event("file", {"path": "new.py"}, NOW)
    assert new_id > max(ids)
    assert [r["id"] for r in store.get_events_since(store.get_cursor())] == [new_id]


def test_a_cursor_beyond_every_id_reads_as_nothing_new(db: Db, store: SqliteStore) -> None:
    db.event("file", {"path": "a.py"}, NOW)
    store.update_cursor(10_000)
    store.commit()
    assert store.get_events_since(store.get_cursor()) == []


# ---------------------------------------------------------------------------
# BEGIN IMMEDIATE
# ---------------------------------------------------------------------------


def _traced(store: SqliteStore) -> list[str]:
    statements: list[str] = []
    store._get_conn().set_trace_callback(statements.append)
    return statements


@pytest.mark.parametrize(
    "write",
    [
        lambda s: s.insert_prediction("stuck", {"p": 1}, 0.5, 60),
        lambda s: s.insert_ml_event("prediction", "poller", "local", 1),
        lambda s: s.insert_signal("t", 0.5, {}),
        lambda s: s.update_cursor(3),
    ],
    ids=["prediction", "ml_event", "signal", "cursor"],
)
def test_every_write_opens_begin_immediate(store: SqliteStore, write) -> None:
    statements = _traced(store)
    write(store)
    store.commit()
    begins = [s for s in statements if s.upper().startswith("BEGIN")]
    assert begins == ["BEGIN IMMEDIATE"], statements


def test_ensure_tables_runs_in_begin_immediate(db: Db) -> None:
    s = SqliteStore(db.path)
    statements = _traced(s)
    s.ensure_tables()
    assert any(st.strip().upper().startswith("BEGIN IMMEDIATE") for st in statements), statements
    assert not s._get_conn().in_transaction


def test_reads_do_not_hold_a_transaction(db: Db, store: SqliteStore) -> None:
    db.task("t")
    store.get_active_task()
    store.get_events_since(0)
    store.get_feature_snapshots()
    assert not store._get_conn().in_transaction


def test_connections_keep_wal_and_busy_timeout(store: SqliteStore) -> None:
    conn = store._get_conn()
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


# ---------------------------------------------------------------------------
# Hosted-advice readers
# ---------------------------------------------------------------------------

TASK_A = "agent-00000000000000aa"
TASK_B = "agent-00000000000000bb"


def _snapshot(task: str, kind: str, h: str, features: Any = None) -> dict[str, Any]:
    return {
        "task": task,
        "kind": kind,
        "feature_contract_version": "v1",
        "features": features if features is not None else [1.0, 0.0, 2.5],
        "features_hash": h,
    }


def _label(h: str, ts: int, revision: int, action: str = "accepted", **extra: Any) -> dict[str, Any]:
    return {
        "task": TASK_A,
        "kind": "branch_now",
        "features_hash": h,
        "ts": ts,
        "revision": revision,
        "features": {"turns": 3},
        "features_complete": True,
        "shown": True,
        "decision": True,
        "confidence": 70,
        "user_action": action,
        **extra,
    }


def test_readers_return_empty_when_no_advice_events_exist(db: Db, store: SqliteStore) -> None:
    db.event("file", {"path": "a.py"}, NOW)
    assert store.get_feature_snapshots() == []
    assert store.get_label_events() == []
    assert store.get_sizing_events() == []


def test_readers_return_empty_without_an_events_table(tmp_path: Path) -> None:
    s = SqliteStore(tmp_path / "empty.db")
    assert s.get_feature_snapshots() == [] and s.get_label_events() == [] and s.get_sizing_events() == []


def test_feature_snapshots(db: Db, store: SqliteStore) -> None:
    e1 = db.event("advice.features", _snapshot(TASK_A, "branch_now", "h1"), NOW - 3 * MIN)
    e2 = db.event("advice.features", _snapshot(TASK_A, "right_size_model", "h2"), NOW - 2 * MIN)
    e3 = db.event("advice.features", _snapshot(TASK_A, "branch_now", "h3", {"x": 1.0}), NOW - MIN)
    e4 = db.event("advice.features", _snapshot(TASK_B, "branch_now", "h4"), NOW)
    db.event("advice.features", {"task": TASK_A, "kind": "branch_now"}, NOW)  # malformed: skipped
    db.event("file", {"path": "a.py"}, NOW)

    snaps = store.get_feature_snapshots()
    assert [s["event_id"] for s in snaps] == [e1, e2, e3, e4]
    assert snaps[0] == {
        "event_id": e1,
        "ts": NOW - 3 * MIN,
        "task": TASK_A,
        "kind": "branch_now",
        "feature_contract_version": "v1",
        "features": [1.0, 0.0, 2.5],
        "features_hash": "h1",
    }
    assert [s["features_hash"] for s in store.get_feature_snapshots(kind="branch_now", task=TASK_A)] == ["h1", "h3"]
    assert [s["features_hash"] for s in store.get_feature_snapshots(since_ms=NOW - 2 * MIN)] == ["h2", "h3", "h4"]
    latest = store.get_feature_snapshots(latest_only=True)
    assert {(s["task"], s["kind"]): s["features_hash"] for s in latest} == {
        (TASK_A, "branch_now"): "h3",
        (TASK_A, "right_size_model"): "h2",
        (TASK_B, "branch_now"): "h4",
    }


def test_label_events_keep_the_highest_revision_per_key(db: Db, store: SqliteStore) -> None:
    db.event("advice.label", _label("h1", 1000, 5, "ignored"), NOW - 5 * MIN)
    db.event("advice.label", _label("h1", 1000, 9, "accepted"), NOW - 4 * MIN)
    db.event("advice.label", _label("h1", 1000, 7, "dismissed"), NOW - 3 * MIN)  # stale: lower revision
    db.event("advice.label", _label("h1", 2000, 1, "dismissed"), NOW - 2 * MIN)  # same hash, new ts: new key
    db.event("advice.label", _label("h2", 1500, 3, features=json.dumps({"turns": 1})), NOW - MIN)
    db.event("advice.label", {**_label("h3", 1600, 1), "kind": "right_size_model"}, NOW - MIN)
    bad = _label("h9", 1700, 1)
    del bad["user_action"]
    db.event("advice.label", bad, NOW)  # malformed: skipped
    db.event("advice.label", {**_label("h8", 1800, 1), "ts": "1800"}, NOW)  # non-int ts: skipped

    labels = store.get_label_events()
    assert [(r["features_hash"], r["ts"], r["revision"], r["user_action"]) for r in labels] == [
        ("h1", 1000, 9, "accepted"),
        ("h2", 1500, 3, "accepted"),
        ("h3", 1600, 1, "accepted"),
        ("h1", 2000, 1, "dismissed"),
    ]
    first = labels[0]
    assert "model" not in first
    assert first["task"] == TASK_A
    assert first["features"] == {"turns": 3}
    assert labels[1]["features"] == {"turns": 1}, "features shipped as JSON text are decoded"
    assert [r["features_hash"] for r in store.get_label_events(kind="right_size_model")] == ["h3"]


def test_an_equal_revision_does_not_replace_the_first(db: Db, store: SqliteStore) -> None:
    db.event("advice.label", _label("h1", 1000, 4, "accepted"), NOW - 2 * MIN)
    db.event("advice.label", _label("h1", 1000, 4, "dismissed"), NOW - MIN)
    assert [r["user_action"] for r in store.get_label_events()] == ["accepted"]


def test_sizing_events(db: Db, store: SqliteStore) -> None:
    profile = {
        "task": TASK_A,
        "turn_ts": NOW - 3 * MIN,
        "model_class": "large",
        "rung": "2",
        "context_fill_bucket": 1,
        "tokens_in_bucket": 2,
        "tokens_out_bucket": 1,
        "tool_calls": 3,
        "tool_failures": 0,
        "retries": 0,
        "followup": "continue",
    }
    switch = {
        "task": TASK_A,
        "from_class": "large",
        "to_class": "medium",
        "from_rung": "2",
        "to_rung": "1",
        "direction": "down",
        "turns_since_session_start": 4,
        "by": "user",
    }
    p = db.event("agent.turn_profile", profile, NOW - 2 * MIN)
    s = db.event("model.switch", switch, NOW - MIN)
    other = db.event("model.switch", {**switch, "task": TASK_B}, NOW)
    db.event("agent.turn", {"outcome": "completed"}, NOW)
    db.event("model.switch", {"direction": "up"}, NOW)  # no task: skipped

    events = store.get_sizing_events()
    assert [(e["event_id"], e["kind"]) for e in events] == [
        (p, "agent.turn_profile"),
        (s, "model.switch"),
        (other, "model.switch"),
    ]
    assert events[0]["payload"] == profile and events[0]["task"] == TASK_A
    assert [e["event_id"] for e in store.get_sizing_events(task=TASK_A)] == [p, s]
    assert [e["event_id"] for e in store.get_sizing_events(since_ms=NOW - MIN)] == [s, other]


def test_the_poller_never_sees_the_advice_kinds(db: Db, store: SqliteStore) -> None:
    from kenaz_ml.poller import POLLER_EXCLUDED_KINDS

    db.event("advice.features", _snapshot(TASK_A, "branch_now", "h1"), NOW)
    db.event("advice.label", _label("h1", 1, 1), NOW)
    keep = db.event("file", {"path": "a.py"}, NOW)
    assert [r["id"] for r in store.get_events_since(0, exclude_kinds=POLLER_EXCLUDED_KINDS)] == [keep]
