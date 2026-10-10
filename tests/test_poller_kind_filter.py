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
