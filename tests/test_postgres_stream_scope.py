"""Stream scoping in the Postgres store, against a recording fake of psycopg2.

The cloud worker opens one store per ``(tenant, user)`` stream; every read
and write of the engine must then carry ``user_id = stream``. No database is
needed to prove that: the SQL and its parameters are the contract.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from kenaz_ml.datastore.postgres import PostgresStore, validate_stream_id

ENGINE_TABLES = ("ml_cursor", "ml_predictions", "ml_events", "ml_signals")


class InsufficientPrivilege(Exception):
    pass


class FakeCursor:
    def __init__(
        self,
        log: list[tuple[str, Any]],
        *,
        deny_ddl: bool,
        tables_present: tuple[str, ...],
        fail_on: tuple[str, ...] = (),
    ) -> None:
        self.log = log
        self.deny_ddl = deny_ddl
        self.fail_on = fail_on
        self.tables_present = tables_present
        self.description = None
        self._last = ""

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        text = " ".join(str(sql).split())
        self._last = text
        self.log.append((text, params))
        if self.deny_ddl and text.startswith(("CREATE", "ALTER")):
            raise InsufficientPrivilege("permission denied for schema")
        if any(needle in text for needle in self.fail_on):
            raise RuntimeError(f"relation does not exist: {text[:40]}")

    def fetchone(self) -> Any:
        return None

    def fetchall(self) -> list[Any]:
        if "information_schema.tables" in self._last:
            return [(t,) for t in self.tables_present]
        return []


class FakeConn:
    def __init__(self, log: list[tuple[str, Any]], **cursor_kwargs: Any) -> None:
        self.log = log
        self.cursor_kwargs = cursor_kwargs
        self.closed = False
        self.autocommit = False
        self.rollbacks = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.log, **self.cursor_kwargs)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


class _SQL(str):
    def format(self, *args: Any) -> str:  # type: ignore[override]
        return str.format(self, *args)


class _Identifier(str):
    def __new__(cls, name: str) -> _Identifier:
        return super().__new__(cls, f'"{name}"')


@pytest.fixture
def fake_psycopg2(monkeypatch: pytest.MonkeyPatch):
    """Install a psycopg2 whose connections record every statement."""
    state: dict[str, Any] = {
        "log": [],
        "conn": None,
        "deny_ddl": False,
        "tables_present": ENGINE_TABLES,
        "fail_on": (),
    }

    def connect(url: str) -> FakeConn:
        conn = FakeConn(
            state["log"], deny_ddl=state["deny_ddl"], tables_present=state["tables_present"], fail_on=state["fail_on"]
        )
        state["conn"] = conn
        return conn

    mod = types.ModuleType("psycopg2")
    mod.connect = connect  # type: ignore[attr-defined]
    errors = types.ModuleType("psycopg2.errors")
    errors.InsufficientPrivilege = InsufficientPrivilege  # type: ignore[attr-defined]
    sql = types.ModuleType("psycopg2.sql")
    sql.SQL = _SQL  # type: ignore[attr-defined]
    sql.Identifier = _Identifier  # type: ignore[attr-defined]
    mod.errors = errors  # type: ignore[attr-defined]
    mod.sql = sql  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg2", mod)
    monkeypatch.setitem(sys.modules, "psycopg2.errors", errors)
    monkeypatch.setitem(sys.modules, "psycopg2.sql", sql)
    return state


def _statements(state: dict[str, Any], needle: str) -> list[tuple[str, Any]]:
    return [(s, p) for s, p in state["log"] if needle in s]


def test_a_scoped_store_filters_every_event_and_task_read_by_user_id(fake_psycopg2) -> None:
    store = PostgresStore("postgresql://x", tenant="tenant_a", stream="user-1")
    store.get_events_since(0)
    store.get_active_task()
    store.get_session_info("t1")
    store.get_completed_task_ids()
    store.count_completed_tasks()
    store.get_quality_task_stats()
    reads = [
        (s, p) for s, p in fake_psycopg2["log"] if s.startswith("SELECT") and ("FROM events" in s or "FROM tasks" in s)
    ]
    assert len(reads) == 6
    for sql, params in reads:
        assert "AND user_id = %s" in sql, sql
        assert "user-1" in tuple(params), (sql, params)


def test_an_unscoped_store_adds_no_user_filter(fake_psycopg2) -> None:
    store = PostgresStore("postgresql://x", tenant="tenant_a")
    store.get_events_since(0)
    store.get_active_task()
    for sql, _ in _statements(fake_psycopg2, "FROM events") + _statements(fake_psycopg2, "FROM tasks"):
        assert "user_id" not in sql


def test_writes_record_the_stream_as_user_id(fake_psycopg2) -> None:
    store = PostgresStore("postgresql://x", tenant="tenant_a", stream="user-1")
    store.insert_prediction("stuck", {"probability": 0.7}, 0.7, 60)
    store.insert_ml_event("prediction", "poller", "local", 3)
    store.update_cursor(42)
    ((pred_sql, pred_params),) = _statements(fake_psycopg2, "INSERT INTO ml_predictions")
    assert pred_sql.startswith("INSERT INTO ml_predictions (user_id, model,")
    assert pred_params[0] == "user-1"
    ((ev_sql, ev_params),) = _statements(fake_psycopg2, "INSERT INTO ml_events")
    assert ev_params[0] == "user-1"
    ((cur_sql, cur_params),) = _statements(fake_psycopg2, "INSERT INTO ml_cursor")
    assert "ON CONFLICT (stream) DO UPDATE" in cur_sql
    assert cur_params[0] == "user-1" and cur_params[1] == 42


def test_the_cursor_is_keyed_by_stream(fake_psycopg2) -> None:
    PostgresStore("postgresql://x", tenant="tenant_a", stream="user-1").get_cursor()
    PostgresStore("postgresql://x", tenant="tenant_a").get_cursor()
    reads = _statements(fake_psycopg2, "SELECT last_event_id FROM ml_cursor")
    assert [p for _, p in reads] == [("user-1",), ("",)]


def test_ensure_tables_creates_the_four_engine_tables_when_allowed(fake_psycopg2) -> None:
    PostgresStore("postgresql://x", tenant="tenant_a", stream="user-1").ensure_tables()
    created = [s for s, _ in fake_psycopg2["log"] if s.startswith("CREATE TABLE IF NOT EXISTS")]
    assert [s.split()[5] for s in created] == list(ENGINE_TABLES)
    assert any("ml_predictions ( id BIGSERIAL PRIMARY KEY, user_id TEXT NOT NULL DEFAULT ''" in s for s in created)


def test_ensure_tables_tolerates_a_platform_owned_schema(fake_psycopg2) -> None:
    fake_psycopg2["deny_ddl"] = True
    PostgresStore("postgresql://x", tenant="tenant_a", stream="user-1").ensure_tables()
    log = [s for s, _ in fake_psycopg2["log"]]
    assert "ROLLBACK TO SAVEPOINT engine_ddl" in log, "a refused statement is rolled back to its savepoint"
    assert "RELEASE SAVEPOINT engine_ddl" not in log
    assert any(s.startswith("INSERT INTO ml_cursor") for s in log), "the cursor row is still written"


def test_ensure_tables_fails_loudly_when_a_table_is_missing(fake_psycopg2) -> None:
    fake_psycopg2["deny_ddl"] = True
    fake_psycopg2["tables_present"] = ("ml_cursor", "ml_signals")
    with pytest.raises(RuntimeError, match="ml_predictions, ml_events"):
        PostgresStore("postgresql://x", tenant="tenant_a").ensure_tables()
    # One rollback for the refused CREATE SCHEMA on connect, one for the failed check.
    assert fake_psycopg2["conn"].rollbacks == 2


def test_list_streams_returns_empty_and_rolls_back_only_its_savepoint_when_the_column_is_absent(fake_psycopg2) -> None:
    store = PostgresStore("postgresql://x", tenant="tenant_a")

    class NoColumnCursor(FakeCursor):
        def execute(self, sql: str, params: Any = None) -> None:
            if "DISTINCT user_id" in sql:
                raise RuntimeError('column "user_id" does not exist')
            super().execute(sql, params)

    conn = store._get_conn()
    conn.cursor = lambda: NoColumnCursor(fake_psycopg2["log"], deny_ddl=False, tables_present=ENGINE_TABLES)  # type: ignore[method-assign]
    assert store.list_streams() == []
    assert conn.rollbacks == 0, "only the read's savepoint is rolled back, never pending writes"
    assert "ROLLBACK TO SAVEPOINT engine_optional_read" in [s for s, _ in fake_psycopg2["log"]]


@pytest.mark.parametrize("bad", ["", "has space", "tab\there", "x" * 129, "line\nbreak"])
def test_stream_ids_with_whitespace_or_wrong_length_are_refused(bad: str) -> None:
    assert validate_stream_id(bad) is False


def test_a_bad_stream_id_is_refused_at_construction(fake_psycopg2) -> None:
    with pytest.raises(ValueError, match="stream id"):
        PostgresStore("postgresql://x", tenant="tenant_a", stream="no spaces allowed")


def test_a_failed_feedback_read_keeps_the_predictions_written_before_it(fake_psycopg2) -> None:
    """Regression: a cloud tenant has no ``suggestions`` table.

    The signal engine's feedback read used to ``rollback()`` the connection,
    discarding the cycle's ``ml_predictions`` rows while the audit row written
    after it committed. The read must fail inside a savepoint instead.
    """
    fake_psycopg2["fail_on"] = ("FROM suggestions",)
    store = PostgresStore("postgresql://x", tenant="tenant_a", stream="user-1")
    store.insert_prediction("stuck", {"probability": 0.4}, 0.4, 90)

    assert store.get_signal_feedback(since_ms=0) == []

    assert fake_psycopg2["conn"].rollbacks == 0, "the shared transaction must not be rolled back"
    log = [s for s, _ in fake_psycopg2["log"]]
    assert "ROLLBACK TO SAVEPOINT engine_optional_read" in log
    assert log.index("SAVEPOINT engine_optional_read") > next(
        i for i, s in enumerate(log) if s.startswith("INSERT INTO ml_predictions")
    )
