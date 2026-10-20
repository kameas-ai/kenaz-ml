"""The poller reads only the event kinds its models are meant to use.

The kinds gated behind ML notice revision 2 exist for the hosted advice
trainers; they must never reach the activity classifier or any poller feature,
and the filter is in the query, so they never leave the database.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from kenaz_ml.datastore.postgres import PostgresStore
from kenaz_ml.datastore.sqlite import SqliteStore
from kenaz_ml.poller import POLLER_EXCLUDED_KINDS
from tests.test_postgres_stream_scope import fake_psycopg2  # noqa: F401  (fixture)
from tests.test_worker import DAEMON_SCHEMA

REV2_KINDS = {"advice.label", "advice.features", "model.switch", "branch.created", "agent.turn_profile"}


def test_the_excluded_kinds_are_exactly_the_revision_2_kinds() -> None:
    assert POLLER_EXCLUDED_KINDS == REV2_KINDS


def _db(path: Path) -> SqliteStore:
    conn = sqlite3.connect(str(path))
    conn.executescript(DAEMON_SCHEMA)
    now = int(time.time() * 1000)
    for i, kind in enumerate(["file", *sorted(REV2_KINDS), "terminal", "agent.tool"]):
        conn.execute(
            "INSERT INTO events (kind, source, payload, ts) VALUES (?,?,?,?)", (kind, "harness", "{}", now + i)
        )
    conn.commit()
    conn.close()
    return SqliteStore(str(path))


def test_sqlite_filters_the_excluded_kinds_in_the_query(tmp_path: Path) -> None:
    store = _db(tmp_path / "e.db")
    kinds = [e["kind"] for e in store.get_events_since(0, exclude_kinds=POLLER_EXCLUDED_KINDS)]
    assert kinds == ["file", "terminal", "agent.tool"]
    assert len(store.get_events_since(0)) == 8, "no exclusion means every kind, as before"


def test_postgres_filters_the_excluded_kinds_in_the_query(fake_psycopg2: dict[str, Any]) -> None:  # noqa: F811
    store = PostgresStore("postgresql://x", tenant="tenant_a", stream="user-1")
    store.get_events_since(7, exclude_kinds=POLLER_EXCLUDED_KINDS)
    ((sql, params),) = [(s, p) for s, p in fake_psycopg2["log"] if "FROM events" in s]
    assert "AND user_id = %s AND NOT (kind = ANY(%s))" in sql
    assert params == (7, "user-1", sorted(REV2_KINDS), 100)


def test_the_poller_never_buffers_an_excluded_kind(tmp_path: Path) -> None:
    from kenaz_ml.poller import EventPoller

    seen: list[str] = []

    class Recorder:
        def classify(self, e: dict) -> dict:
            seen.append(e["kind"])
            return {"category": "idle", "confidence": 0.5}

    store = _db(tmp_path / "p.db")
    store.ensure_tables()
    models = {"stuck": None, "activity": Recorder(), "workflow": None, "duration": None, "quality": None}
    poller = EventPoller(store, models)
    poller._last_predict_time = time.time()  # classification only; no prediction cycle
    poller._poll_once()
    assert set(seen).isdisjoint(REV2_KINDS)
    assert seen == ["file", "terminal", "agent.tool"]
    assert store.get_cursor() == 8, "the cursor still advances past the excluded rows"


class _OneSignal:
    """A signal engine that writes one signal per cycle, as the real one does."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def process_events(self, buffer: list[dict], task_context: dict | None) -> int:
        self.store.insert_signal("velocity_deviation", 0.8, {"n": len(buffer)})
        return 1


class _Classifier:
    def classify(self, e: dict) -> dict:
        return {"category": "idle", "confidence": 0.5}


def test_signals_are_durable_after_a_cycle_with_no_further_events(tmp_path: Path) -> None:
    from kenaz_ml.poller import EventPoller

    path = tmp_path / "s.db"
    store = _db(path)
    store.ensure_tables()
    models = {"stuck": None, "activity": _Classifier(), "workflow": None, "duration": None, "quality": None}
    poller = EventPoller(store, models, signal_engine=_OneSignal(store))
    poller._last_predict_time = time.time()
    poller._poll_once()
    poller._poll_once()  # no new events: an empty cycle must not be needed to commit
    # Another connection sees the signal while the store is still open ...
    other = sqlite3.connect(str(path))
    try:
        assert other.execute("SELECT signal_type FROM ml_signals").fetchall() == [("velocity_deviation",)]
    finally:
        other.close()
    # ... and it survives the store closing.
    store.close()
    reopened = sqlite3.connect(str(path))
    try:
        assert reopened.execute("SELECT COUNT(*) FROM ml_signals").fetchone()[0] == 1
    finally:
        reopened.close()


def test_the_poller_commits_after_the_signal_writes_in_any_store() -> None:
    """Store-agnostic (Postgres included): the last call of a cycle is a commit after insert_signal."""
    from kenaz_ml.poller import EventPoller

    calls: list[str] = []

    class Recording:
        def get_cursor(self) -> int:
            return 0

        def get_events_since(self, since_id: int, limit: int = 100, *, exclude_kinds: Any = ()) -> list[dict]:
            return [{"id": 1, "kind": "file", "source": "h", "payload": "{}", "ts": 1}]

        def update_cursor(self, event_id: int) -> None:
            calls.append("update_cursor")

        def get_active_task(self) -> None:
            return None

        def insert_signal(self, *a: Any, **k: Any) -> int:
            calls.append("insert_signal")
            return 1

        def commit(self) -> None:
            calls.append("commit")

    store = Recording()
    models = {"stuck": None, "activity": _Classifier(), "workflow": None, "duration": None, "quality": None}
    poller = EventPoller(store, models, signal_engine=_OneSignal(store))  # type: ignore[arg-type]
    poller._last_predict_time = time.time()
    poller._poll_once()
    assert calls[-2:] == ["insert_signal", "commit"], calls
