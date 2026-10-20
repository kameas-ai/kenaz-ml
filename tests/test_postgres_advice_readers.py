"""PostgresStore's hosted-advice readers and active-task gate, against a fake psycopg2.

The SQL and its parameters are the contract for stream scoping; the rows the
fake returns prove the readers shape them exactly as the SQLite store does
(the shaping is shared, see ``kenaz_ml.datastore.advice_events``).
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from kenaz_ml.datastore.postgres import PostgresStore

TASK = "agent-00000000000000aa"


class FakeCursor:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.description = None
        self._rows: list[tuple[Any, ...]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        text = " ".join(str(sql).split())
        self.state["log"].append((text, params))
        self._rows = []
        if text.startswith("SELECT id, kind, payload, ts FROM events"):
            kinds = params[0]
            self._rows = [r for r in self.state["events"] if r[1] in kinds]

    def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[Any]:
        return list(self._rows)


class FakeConn:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.closed = False
        self.autocommit = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.state)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _SQL(str):
    def format(self, *args: Any) -> str:  # type: ignore[override]
        return str.format(self, *args)


@pytest.fixture
def pg(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"log": [], "events": []}
    mod = types.ModuleType("psycopg2")
    mod.connect = lambda url: FakeConn(state)  # type: ignore[attr-defined]
    errors = types.ModuleType("psycopg2.errors")
    errors.InsufficientPrivilege = type("InsufficientPrivilege", (Exception,), {})  # type: ignore[attr-defined]
    sql = types.ModuleType("psycopg2.sql")
    sql.SQL = _SQL  # type: ignore[attr-defined]
    sql.Identifier = lambda name: f'"{name}"'  # type: ignore[attr-defined]
    mod.errors = errors  # type: ignore[attr-defined]
    mod.sql = sql  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg2", mod)
    monkeypatch.setitem(sys.modules, "psycopg2.errors", errors)
    monkeypatch.setitem(sys.modules, "psycopg2.sql", sql)
    return state


def _reads(state: dict[str, Any]) -> list[tuple[str, Any]]:
    return [(s, p) for s, p in state["log"] if s.startswith("SELECT id, kind, payload, ts FROM events")]


def test_every_reader_is_stream_scoped(pg: dict[str, Any]) -> None:
    store = PostgresStore("postgresql://x", tenant="tenant_a", stream="user-1")
    store.get_feature_snapshots()
    store.get_label_events(since_ms=5)
    store.get_sizing_events()
    reads = _reads(pg)
    assert len(reads) == 3
    for sql, params in reads:
        assert sql.endswith("AND user_id = %s ORDER BY id ASC"), sql
        assert params[-1] == "user-1"
    assert reads[0][1][0] == ["advice.features"]
    assert reads[1][1] == (["advice.label"], 5, "user-1")
    assert reads[2][1][0] == ["agent.turn_profile", "model.switch"]


def test_an_unscoped_store_reads_the_whole_schema(pg: dict[str, Any]) -> None:
    PostgresStore("postgresql://x", tenant="tenant_a").get_feature_snapshots()
    ((sql, params),) = _reads(pg)
    assert "user_id" not in sql and params == (["advice.features"],)


def test_jsonb_and_text_payloads_shape_identically(pg: dict[str, Any]) -> None:
    snap = {
        "task": TASK,
        "kind": "branch_now",
        "feature_contract_version": "v1",
        "features": [1.0, 2.0],
        "features_hash": "h1",
    }
    pg["events"] = [
        (1, "advice.features", snap, 100),  # JSONB arrives as a dict
        (2, "advice.features", json.dumps({**snap, "features_hash": "h2"}), 200),  # TEXT as a string
    ]
    snaps = PostgresStore("postgresql://x", tenant="tenant_a", stream="u").get_feature_snapshots()
    assert [(s["event_id"], s["features_hash"], s["features"]) for s in snaps] == [
        (1, "h1", [1.0, 2.0]),
        (2, "h2", [1.0, 2.0]),
    ]


def test_label_events_dedupe_by_revision(pg: dict[str, Any]) -> None:
    def label(revision: int, action: str) -> dict[str, Any]:
        return {
            "task": TASK,
            "kind": "branch_now",
            "features_hash": "h",
            "ts": 1000,
            "revision": revision,
            "features": {},
            "features_complete": True,
            "shown": True,
            "user_action": action,
        }

    pg["events"] = [(1, "advice.label", label(2, "ignored"), 10), (2, "advice.label", label(3, "accepted"), 20)]
    (row,) = PostgresStore("postgresql://x", tenant="tenant_a", stream="u").get_label_events()
    assert (row["revision"], row["user_action"], row["task"]) == (3, "accepted", TASK)


def test_sizing_events_from_postgres(pg: dict[str, Any]) -> None:
    pg["events"] = [
        (5, "model.switch", {"task": TASK, "direction": "up", "by": "advisor"}, 50),
        (4, "agent.turn_profile", {"task": TASK, "turn_ts": 30, "followup": "continue"}, 40),
    ]
    events = PostgresStore("postgresql://x", tenant="tenant_a", stream="u").get_sizing_events(task=TASK)
    assert [(e["event_id"], e["kind"]) for e in events] == [(4, "agent.turn_profile"), (5, "model.switch")]


def test_readers_return_empty_without_rows(pg: dict[str, Any]) -> None:
    store = PostgresStore("postgresql://x", tenant="tenant_a", stream="u")
    assert store.get_feature_snapshots() == [] and store.get_label_events() == [] and store.get_sizing_events() == []


def test_active_task_gate_sql(pg: dict[str, Any]) -> None:
    store = PostgresStore("postgresql://x", tenant="tenant_a", stream="user-1", active_task_window_ms=30 * 60_000)
    store.get_active_task(now_ms=10_000_000)
    ((sql, params),) = [(s, p) for s, p in pg["log"] if "FROM tasks" in s]
    assert sql == (
        "SELECT id FROM tasks WHERE completed_at IS NULL AND last_active >= %s "
        "AND user_id = %s ORDER BY last_active DESC LIMIT 1"
    )
    assert "phase" not in sql
    assert params == (10_000_000 - 30 * 60_000, "user-1")


def test_the_postgres_window_reads_config(pg: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KENAZ_ACTIVE_TASK_WINDOW_MIN", "5")
    assert PostgresStore("postgresql://x", tenant="tenant_a").active_task_window_ms == 5 * 60_000
