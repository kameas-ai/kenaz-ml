"""DataStore implementation backed by a local SQLite file.

Preserves all existing SQLite behavior including WAL mode and busy_timeout.
Every connection enforces: PRAGMA journal_mode=WAL, PRAGMA busy_timeout=5000.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SqliteStore:
    """DataStore implementation backed by a local SQLite file.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    # --- Connection lifecycle ---

    def _connect(self) -> sqlite3.Connection:
        """Create a new connection with WAL mode and busy_timeout."""
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        """Return the existing connection or create a new one (lazy init)."""
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

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
        conn.executescript("""
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
        """)
        conn.commit()
        logger.info("schema: ml_cursor and ml_signals tables ensured")

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

    def get_events_since(self, since_id: int, limit: int = 100) -> list[dict[str, Any]]:
        """Return events with id > since_id, ordered by id ASC, up to limit.

        Each dict has keys: id, kind, source, payload (raw string), ts.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, kind, source, payload, ts FROM events WHERE id > ? ORDER BY id ASC LIMIT ?",
            (since_id, limit),
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
                except (json.JSONDecodeError, TypeError):
                    pass
        return rows

    # --- Task queries ---

    def get_active_task(self) -> str | None:
        """Return the ID of the active (non-idle, not completed) task, or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id FROM tasks WHERE phase != 'idle' AND completed_at IS NULL ORDER BY last_active DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def get_task_by_id(self, task_id: str) -> dict[str, Any] | None:
        """Return a full task row as a dict, or None if not found."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
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
        return cur.lastrowid

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
