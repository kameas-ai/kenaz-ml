"""DataStore implementation backed by a local SQLite file.

Every connection enforces: PRAGMA journal_mode=WAL, PRAGMA busy_timeout=5000.

Two database shapes are read through the same queries:

- sigild's ``data.db`` (``events``/``tasks`` without tenancy columns), and
- the harness's local activity store (``KENAZ_ML_ACTIVITY_DB``), whose
  ``events``/``tasks``/``ml_predictions``/``ml_events`` carry Fleet's tenant
  columns (``user_id``, ``node_id``, ``producer``) with local rows set to ''.

Every query here names only the columns both shapes share, so neither needs
its own branch. The engine creates ``ml_cursor`` and ``ml_signals`` in both.

Writes run in ``BEGIN IMMEDIATE`` transactions: another process (the harness)
writes the same file, and a deferred transaction that reads and then writes
must upgrade its lock, which fails with SQLITE_BUSY without consulting
``busy_timeout``. Taking the write lock at ``BEGIN`` makes the wait a plain
busy wait that the timeout covers.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Collection
from pathlib import Path
from typing import Any

from kenaz_ml.datastore.advice_events import (
    FEATURES_KIND,
    LABEL_KIND,
    SIZING_KINDS,
    feature_snapshots,
    label_rows,
    sizing_events,
)

logger = logging.getLogger(__name__)


def _is_no_such_table(exc: sqlite3.OperationalError) -> bool:
    return str(exc).lower().startswith("no such table")


class SqliteStore:
    """DataStore implementation backed by a local SQLite file.

    Args:
        db_path: Path to the SQLite database file.
        active_task_window_ms: How recently an uncompleted task must have been
            active to count as the active task. Defaults to the configured
            window (``KENAZ_ACTIVE_TASK_WINDOW_MIN``, 30 minutes).
    """

    def __init__(self, db_path: Path, *, active_task_window_ms: int | None = None) -> None:
        from kenaz_ml import config

        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        # Per table: whether it has the activity-DB ``producer`` column.
        self._has_producer: dict[str, bool] = {}
        self.active_task_window_ms = (
            active_task_window_ms if active_task_window_ms is not None else config.active_task_window_ms()
        )

    # --- Connection lifecycle ---

    def _connect(self) -> sqlite3.Connection:
        """Create a new connection with WAL mode and busy_timeout.

        ``isolation_level="IMMEDIATE"`` makes the implicit transaction the
        driver opens before the first INSERT/UPDATE/DELETE a ``BEGIN
        IMMEDIATE``; reads outside a write transaction stay in autocommit.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=5.0, isolation_level="IMMEDIATE")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        """Return the existing connection or create a new one (lazy init)."""
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

    def _harness_scope(self, table: str) -> str:
        """``" AND producer = 'harness'"`` on the activity-DB shape, ``""`` on sigild's.

        The activity DB's ``producer`` column admits ``'sigild'`` as well (the
        shape is Fleet's), and its task key is ``(user_id, node_id, producer,
        id)``, so a task id alone is not unique there. The engine reads the
        harness's rows only. Decided once the table exists; a missing table
        is not cached, so a later creation is seen.
        """
        has = self._has_producer.get(table)
        if has is None:
            cols = {r[1] for r in self._get_conn().execute(f"PRAGMA table_info({table})").fetchall()}
            if not cols:
                return ""
            has = self._has_producer[table] = "producer" in cols
        return " AND producer = 'harness'" if has else ""

    def commit(self) -> None:
        """Commit the current transaction."""
        if self._conn is not None:
            self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # --- Schema bootstrap ---

    def ensure_tables(self) -> None:
        """Create Python-owned tables if they don't exist (ml_cursor, ml_signals)."""
        conn = self._get_conn()
        try:
            self._ensure_tables_script(conn)
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        logger.info("schema: ml_cursor and ml_signals tables ensured")

    @staticmethod
    def _ensure_tables_script(conn: sqlite3.Connection) -> None:
        conn.executescript("""
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS ml_cursor (
                id            INTEGER PRIMARY KEY CHECK (id = 1),
                last_event_id INTEGER NOT NULL DEFAULT 0,
                updated_at    INTEGER NOT NULL DEFAULT 0
            );
            INSERT OR IGNORE INTO ml_cursor (id, last_event_id, updated_at)
            VALUES (1, 0, 0);

            CREATE TABLE IF NOT EXISTS ml_signals (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_type      TEXT    NOT NULL,
                confidence       REAL    NOT NULL,
                evidence         TEXT    NOT NULL,
                suggested_action TEXT,
                created_at       INTEGER NOT NULL,
                expires_at       INTEGER,
                rendered         INTEGER NOT NULL DEFAULT 0,
                suggestion_id    INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_ml_signals_created_at ON ml_signals(created_at);
            CREATE INDEX IF NOT EXISTS idx_ml_signals_rendered ON ml_signals(rendered);
            COMMIT;
        """)

    # --- Cursor operations ---

    def get_cursor(self) -> int:
        """Return the last processed event ID from ml_cursor."""
        conn = self._get_conn()
        row = conn.execute("SELECT last_event_id FROM ml_cursor WHERE id = 1").fetchone()
        return row[0] if row else 0

    def update_cursor(self, event_id: int) -> None:
        """Update ml_cursor.last_event_id to the given event_id."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE ml_cursor SET last_event_id = ?, updated_at = ? WHERE id = 1",
            (event_id, int(time.time() * 1000)),
        )

    # --- Event queries ---

    def get_events_since(
        self, since_id: int, limit: int = 100, *, exclude_kinds: Collection[str] = ()
    ) -> list[dict[str, Any]]:
        """Return events with id > since_id, ordered by id ASC, up to limit.

        Each dict has keys: id, kind, source, payload (raw string), ts.
        Rows whose kind is in ``exclude_kinds`` are filtered in the query.
        """
        excluded = sorted(exclude_kinds)
        not_in = f" AND kind NOT IN ({', '.join('?' * len(excluded))})" if excluded else ""
        conn = self._get_conn()
        rows = conn.execute(
            f"SELECT id, kind, source, payload, ts FROM events WHERE id > ?{not_in} ORDER BY id ASC LIMIT ?",
            (since_id, *excluded, limit),
        ).fetchall()
        columns = ["id", "kind", "source", "payload", "ts"]
        return [dict(zip(columns, row)) for row in rows]

    def get_events_for_task(self, task_id: str, since: int | None = None) -> list[dict[str, Any]]:
        """Return events within a task's time window, with JSON payloads parsed."""
        task = self.get_task_by_id(task_id)
        if task is None:
            return []

        start = since if since is not None else task.get("started_at", 0)
        end = task.get("completed_at") or task.get("last_active") or int(time.time() * 1000)

        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "SELECT * FROM events WHERE ts >= ? AND ts <= ? ORDER BY ts",
                (start, end),
            )
            rows = [dict(r) for r in cur.fetchall()]
        finally:
            conn.row_factory = None

        # Parse JSON payload
        for row in rows:
            if isinstance(row.get("payload"), str):
                try:
                    row["payload"] = json.loads(row["payload"])
                except json.JSONDecodeError, TypeError:
                    pass
        return rows

    # --- Task queries ---

    def get_active_task(self, *, now_ms: int | None = None) -> str | None:
        """Return the most recently active uncompleted task inside the window, or None.

        Active means ``completed_at IS NULL`` and ``last_active`` within
        :attr:`active_task_window_ms` of ``now_ms`` (default: wall clock).
        ``phase`` is not consulted: it is a feature, not a liveness signal.
        """
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id FROM tasks WHERE completed_at IS NULL AND last_active >= ? ORDER BY last_active DESC LIMIT 1",
            (now - self.active_task_window_ms,),
        ).fetchone()
        return row[0] if row else None

    def get_task_by_id(self, task_id: str) -> dict[str, Any] | None:
        """Return a full task row as a dict, or None if not found."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(f"SELECT * FROM tasks WHERE id = ?{self._harness_scope('tasks')}", (task_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.row_factory = None

    def get_session_info(self, task_id: str) -> dict[str, Any] | None:
        """Return started_at, phase, test_fails for a task."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT started_at, phase, test_fails FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return {"started_at": row[0], "phase": row[1], "test_fails": row[2]}

    def get_quality_task_stats(self) -> dict[str, Any] | None:
        """Return test_runs, test_fails, commit_count from the most recently completed task."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT test_runs, test_fails, commit_count FROM tasks "
            "WHERE completed_at IS NOT NULL ORDER BY completed_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {"test_runs": row[0], "test_fails": row[1], "commit_count": row[2]}

    def get_completed_task_ids(self) -> list[str]:
        """Return IDs of all completed tasks."""
        conn = self._get_conn()
        rows = conn.execute("SELECT id FROM tasks WHERE completed_at IS NOT NULL").fetchall()
        return [row[0] for row in rows]

    def get_completed_tasks_with_timestamps(self) -> list[dict[str, Any]]:
        """Return id, started_at, completed_at for completed tasks with both timestamps set."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, started_at, completed_at FROM tasks WHERE completed_at IS NOT NULL AND started_at IS NOT NULL"
        ).fetchall()
        return [{"id": row[0], "started_at": row[1], "completed_at": row[2]} for row in rows]

    def count_completed_tasks(self) -> int:
        """Return the count of completed tasks."""
        try:
            conn = self._get_conn()
            row = conn.execute("SELECT COUNT(*) FROM tasks WHERE completed_at IS NOT NULL").fetchone()
            return row[0] if row else 0
        except sqlite3.OperationalError:
            return 0

    def count_completed_tasks_before(self, ts_ms: int) -> int:
        """Return the count of tasks completed at or before ``ts_ms``."""
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE completed_at IS NOT NULL AND completed_at <= ?", (ts_ms,)
            ).fetchone()
            return row[0] if row else 0
        except sqlite3.OperationalError:
            return 0

    def last_ml_event_ts(self, kind: str, endpoint: str) -> int | None:
        """Return the newest ``ml_events.ts`` with this kind and endpoint, or None."""
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT MAX(ts) FROM ml_events WHERE kind = ? AND endpoint = ?", (kind, endpoint)
            ).fetchone()
        except sqlite3.OperationalError as exc:
            if _is_no_such_table(exc):
                return None
            raise
        return int(row[0]) if row and row[0] is not None else None

    def count_predictions_since(self, since_ms: int) -> dict[str, int]:
        """Return ``ml_predictions`` counts per model created at or after ``since_ms``, expired or not."""
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT model, COUNT(*) FROM ml_predictions WHERE created_at >= ? GROUP BY model", (since_ms,)
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if _is_no_such_table(exc):
                return {}
            raise
        return {str(r[0]): int(r[1]) for r in rows}

    # --- Status data ---

    def get_status_data(self) -> dict[str, Any]:
        """Return cursor info and latest non-expired predictions for /status."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        try:
            cursor_row = conn.execute("SELECT last_event_id, updated_at FROM ml_cursor WHERE id = 1").fetchone()

            now_ms = int(time.time() * 1000)
            preds = conn.execute(
                "SELECT model, confidence, created_at FROM ml_predictions "
                "WHERE expires_at IS NULL OR expires_at > ? "
                "ORDER BY created_at DESC",
                (now_ms,),
            ).fetchall()

            return {
                "cursor": dict(cursor_row) if cursor_row else None,
                "latest_predictions": [dict(r) for r in preds],
            }
        finally:
            conn.row_factory = None

    # --- Write operations (ml_predictions, ml_events only) ---

    def insert_prediction(self, model: str, result: dict, confidence: float, ttl_sec: int | None) -> None:
        """Insert a row into ml_predictions."""
        conn = self._get_conn()
        now_ms = int(time.time() * 1000)
        expires_ms = (now_ms + ttl_sec * 1000) if ttl_sec else None
        conn.execute(
            "INSERT INTO ml_predictions (model, result, confidence, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (model, json.dumps(result), round(confidence, 4), now_ms, expires_ms),
        )

    def insert_ml_event(self, kind: str, endpoint: str, routing: str, latency_ms: int) -> None:
        """Insert a row into ml_events."""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO ml_events (kind, endpoint, routing, latency_ms, ts) VALUES (?, ?, ?, ?, ?)",
            (kind, endpoint, routing, latency_ms, int(time.time() * 1000)),
        )

    # --- Signal operations ---

    def insert_signal(
        self,
        signal_type: str,
        confidence: float,
        evidence: dict,
        suggested_action: str | None = None,
        ttl_sec: int | None = None,
    ) -> int:
        """Insert a signal into ml_signals. Returns the signal ID."""
        conn = self._get_conn()
        now_ms = int(time.time() * 1000)
        expires_ms = (now_ms + ttl_sec * 1000) if ttl_sec else None
        cur = conn.execute(
            "INSERT INTO ml_signals "
            "(signal_type, confidence, evidence, suggested_action, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (signal_type, round(confidence, 4), json.dumps(evidence), suggested_action, now_ms, expires_ms),
        )
        signal_id = cur.lastrowid
        if signal_id is None:
            raise RuntimeError("insert_signal: the INSERT produced no row id")
        return signal_id

    def get_signal_feedback(self, since_ms: int) -> list[dict]:
        """Read feedback linkages from suggestions table for training."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT s.signal_id, ms.signal_type, s.status, s.created_at "
                "FROM suggestions s "
                "JOIN ml_signals ms ON s.signal_id = ms.id "
                "WHERE s.signal_id IS NOT NULL AND s.created_at > ? "
                "ORDER BY s.created_at ASC",
                (since_ms,),
            ).fetchall()
            return [{"signal_id": r[0], "signal_type": r[1], "status": r[2], "created_at": r[3]} for r in rows]
        except Exception:
            # signal_id column may not exist yet (Go Feature 021)
            logger.debug("get_signal_feedback: suggestions.signal_id not available yet")
            return []

    # --- Hosted-advice event readers ---

    def _select_events_of_kinds(self, kinds: tuple[str, ...], since_ms: int | None) -> list[dict[str, Any]]:
        """``events`` rows of the given kinds (``id, kind, payload, ts``), ``[]`` when there is no table."""
        params: list[Any] = list(kinds)
        since = ""
        if since_ms is not None:
            since = " AND ts >= ?"
            params.append(since_ms)
        try:
            rows = (
                self._get_conn()
                .execute(
                    f"SELECT id, kind, payload, ts FROM events WHERE kind IN ({', '.join('?' * len(kinds))}){since}"
                    f"{self._harness_scope('events')} "
                    "ORDER BY id ASC",
                    params,
                )
                .fetchall()
            )
        except sqlite3.OperationalError as exc:
            # Only a missing table means "nothing yet"; a locked database or
            # any other failure must reach the caller, which retries.
            if not _is_no_such_table(exc):
                raise
            logger.debug("advice readers: no events table yet")
            return []
        return [{"id": r[0], "kind": r[1], "payload": r[2], "ts": r[3]} for r in rows]

    def get_feature_snapshots(
        self,
        *,
        kind: str | None = None,
        task: str | None = None,
        since_ms: int | None = None,
        latest_only: bool = False,
    ) -> list[dict[str, Any]]:
        """``advice.features`` snapshots; see :func:`kenaz_ml.datastore.advice_events.feature_snapshots`."""
        rows = self._select_events_of_kinds((FEATURES_KIND,), since_ms)
        return feature_snapshots(rows, kind=kind, task=task, latest_only=latest_only)

    def get_label_events(self, *, kind: str | None = None, since_ms: int | None = None) -> list[dict[str, Any]]:
        """``advice.label`` rows, one per (kind, features_hash, ts), highest revision winning."""
        return label_rows(self._select_events_of_kinds((LABEL_KIND,), since_ms), kind=kind)

    def get_sizing_events(self, *, task: str | None = None, since_ms: int | None = None) -> list[dict[str, Any]]:
        """``agent.turn_profile`` and ``model.switch`` events for the sizing labeler."""
        return sizing_events(self._select_events_of_kinds(SIZING_KINDS, since_ms), task=task)

    # --- Local retention ---

    #: The only tables retention prunes, with their time column. ``events`` and
    #: ``tasks`` belong to the producer and are never pruned here.
    PRUNED_TABLES: tuple[tuple[str, str], ...] = (("ml_predictions", "created_at"), ("ml_events", "ts"))

    def prune_ml_rows(self, retention_days: int, *, now_ms: int | None = None) -> dict[str, int]:
        """Delete ``ml_predictions`` / ``ml_events`` rows older than ``retention_days``.

        Runs on its own short-lived connection (so it is safe to call from a
        worker thread while the poller holds the shared one) in a single
        ``BEGIN IMMEDIATE`` transaction. A table that does not exist yet is
        skipped. Returns the number of rows deleted per table.
        """
        if not isinstance(retention_days, int) or isinstance(retention_days, bool) or retention_days < 1:
            raise ValueError(f"retention_days must be a positive int, got {retention_days!r}")
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        cutoff = now - retention_days * 86_400_000
        deleted: dict[str, int] = {}
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            present = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)",
                    tuple(t for t, _ in self.PRUNED_TABLES),
                ).fetchall()
            }
            for table, column in self.PRUNED_TABLES:
                if table not in present:
                    continue
                cur = conn.execute(f"DELETE FROM {table} WHERE {column} < ?", (cutoff,))
                deleted[table] = cur.rowcount
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return deleted

    # --- Cloud training methods (not supported in local SQLite mode) ---

    def get_last_training_ts(self, tenant_id: str) -> float | None:
        """Not supported in local mode."""
        raise NotImplementedError("get_last_training_ts is a cloud-only method")

    def get_completed_tasks_for_tenant(self, tenant_id: str) -> list[dict]:
        """Not supported in local mode."""
        raise NotImplementedError("get_completed_tasks_for_tenant is a cloud-only method")

    def get_events_for_task_id(self, task_id: str) -> list[dict]:
        """Not supported in local mode."""
        raise NotImplementedError("get_events_for_task_id is a cloud-only method")

    def get_all_tenant_ids(self) -> list[str]:
        """Not supported in local mode."""
        raise NotImplementedError("get_all_tenant_ids is a cloud-only method")

    def get_opted_in_tenant_ids(self) -> list[str]:
        """Not supported in local mode."""
        raise NotImplementedError("get_opted_in_tenant_ids is a cloud-only method")

    def record_training_run(self, tenant_id: str, status: str, duration_ms: int) -> None:
        """Not supported in local mode."""
        raise NotImplementedError("record_training_run is a cloud-only method")
