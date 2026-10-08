"""DataStore implementation backed by PostgreSQL.

Supports per-tenant schema isolation. Each tenant's tables live
in a dedicated Postgres schema (e.g., tenant_abc.events).

Requires: pip install kenaz-ml[cloud]
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

#: ``user_id`` values are opaque ids the ingest wrote (fleet uses UUIDs);
#: they are bound as parameters, never interpolated, so the check is only
#: against garbage, not injection.
_STREAM_ID_MAX_LEN = 128


def validate_stream_id(stream: str) -> bool:
    """A stream id is 1-128 printable characters with no whitespace."""
    return 0 < len(stream) <= _STREAM_ID_MAX_LEN and stream.isprintable() and not any(c.isspace() for c in stream)


class PostgresStore:
    """DataStore implementation backed by a PostgreSQL database.

    Args:
        connection_url: Postgres connection URL (e.g., postgresql://user:pass@host:5432/dbname)
        tenant: Tenant identifier for schema isolation. Defaults to "public".
        stream: The device stream inside the tenant this store is scoped to --
            the ``user_id`` the ingest wrote on every ``events`` and ``tasks``
            row. The engine's loop is one person's loop (one active task, one
            cursor), so a cloud worker opens one store per stream and every
            read and write here carries ``user_id = stream``. ``None`` is the
            unscoped store: the whole schema is one stream, which is only
            right for a single-user schema or for cross-stream training.
    """

    def __init__(self, connection_url: str, tenant: str = "public", stream: str | None = None) -> None:
        try:
            import psycopg2
            from psycopg2 import sql as pg_sql
        except ImportError:
            raise ImportError(
                "psycopg2-binary is required for PostgresStore. Install with: pip install kenaz-ml[cloud]"
            ) from None

        from kenaz_ml.config import validate_tenant_id

        if not validate_tenant_id(tenant):
            raise ValueError(
                f"Invalid tenant ID '{tenant}'. "
                "Must be 1-63 characters of lowercase alphanumeric, hyphens, or underscores."
            )

        if stream is not None and not validate_stream_id(stream):
            raise ValueError(f"Invalid stream id {stream!r}: 1-128 printable characters, no whitespace.")
        self._connection_url = connection_url
        self._tenant = tenant
        self._stream = stream
        # The scope every events/tasks query carries: a clause appended inside
        # the WHERE, and the parameters it binds. Empty when unscoped.
        self._sc = " AND user_id = %s" if stream is not None else ""
        self._sp: tuple[Any, ...] = (stream,) if stream is not None else ()
        # What the engine-owned tables record as the writer; '' when unscoped.
        self._uid = stream if stream is not None else ""
        self._conn = None
        self._psycopg2 = psycopg2
        self._sql = pg_sql

    # --- Connection lifecycle ---

    def _get_conn(self):
        """Return existing connection or create a new one with tenant schema."""
        if self._conn is None or self._conn.closed:
            self._conn = self._psycopg2.connect(self._connection_url)
            self._conn.autocommit = False
            with self._conn.cursor() as cur:
                # A platform-provisioned schema already exists and the worker
                # role may not CREATE; the failed statement aborts the
                # transaction, so roll it back before setting the search path.
                try:
                    cur.execute(
                        self._sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(self._sql.Identifier(self._tenant))
                    )
                except self._psycopg2.errors.InsufficientPrivilege:
                    self._conn.rollback()
                cur.execute(self._sql.SQL("SET search_path TO {}, public").format(self._sql.Identifier(self._tenant)))
            self._conn.commit()
        return self._conn

    def commit(self) -> None:
        """Commit the current transaction."""
        if self._conn and not self._conn.closed:
            self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    # --- Schema bootstrap ---

    #: The engine-owned tables ``ensure_tables`` creates, or expects to find
    #: when the platform owns the DDL (fleet's provisioning creates them in
    #: exactly these shapes and grants the worker no CREATE).
    ENGINE_TABLES = ("ml_cursor", "ml_predictions", "ml_events", "ml_signals")

    def _ddl(self, cur: Any, statement: str) -> bool:
        """Run one DDL statement inside a savepoint.

        Returns ``False`` without failing the transaction when the role lacks
        the privilege: a platform-managed schema already has the tables, and
        the check at the end of :meth:`ensure_tables` is what proves it.
        """
        cur.execute("SAVEPOINT engine_ddl")
        try:
            cur.execute(statement)
        except self._psycopg2.errors.InsufficientPrivilege:
            cur.execute("ROLLBACK TO SAVEPOINT engine_ddl")
            return False
        cur.execute("RELEASE SAVEPOINT engine_ddl")
        return True

    def ensure_tables(self) -> None:
        """Create the engine-owned tables in the tenant schema if they don't exist.

        In the cloud nobody else creates ``ml_predictions`` and ``ml_events``
        (locally they are the daemon's), so they are created here with the
        daemon's columns plus ``user_id``. Every engine-owned table carries
        ``user_id`` and the cursor is keyed by stream, so many streams share
        one schema without sharing state.

        A schema whose DDL the platform owns (the worker role has no CREATE)
        is fine: each statement refused for lack of privilege is skipped, and
        the method then verifies every engine table exists, raising if one is
        missing. Any change to these shapes is a coordinated platform
        migration, not a silent ALTER here.
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            self._ddl(
                cur,
                """
                CREATE TABLE IF NOT EXISTS ml_cursor (
                    stream        TEXT PRIMARY KEY,
                    last_event_id BIGINT NOT NULL DEFAULT 0,
                    updated_at    BIGINT NOT NULL DEFAULT 0
                )
                """,
            )
            self._ddl(
                cur,
                """
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id         BIGSERIAL PRIMARY KEY,
                    user_id    TEXT   NOT NULL DEFAULT '',
                    model      TEXT   NOT NULL,
                    result     TEXT   NOT NULL,
                    confidence REAL   NOT NULL,
                    created_at BIGINT NOT NULL,
                    expires_at BIGINT
                )
                """,
            )
            self._ddl(
                cur,
                """
                CREATE INDEX IF NOT EXISTS idx_ml_predictions_user_model_created
                ON ml_predictions(user_id, model, created_at DESC)
                """,
            )
            self._ddl(
                cur,
                """
                CREATE TABLE IF NOT EXISTS ml_events (
                    id         BIGSERIAL PRIMARY KEY,
                    user_id    TEXT    NOT NULL DEFAULT '',
                    kind       TEXT    NOT NULL,
                    endpoint   TEXT    NOT NULL,
                    routing    TEXT    NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    ts         BIGINT  NOT NULL
                )
                """,
            )
            self._ddl(cur, "CREATE INDEX IF NOT EXISTS idx_ml_events_ts ON ml_events(ts)")
            self._ddl(
                cur,
                """
                CREATE TABLE IF NOT EXISTS ml_signals (
                    id               SERIAL PRIMARY KEY,
                    user_id          TEXT    NOT NULL DEFAULT '',
                    signal_type      TEXT    NOT NULL,
                    confidence       REAL    NOT NULL,
                    evidence         TEXT    NOT NULL,
                    suggested_action TEXT,
                    created_at       BIGINT  NOT NULL,
                    expires_at       BIGINT,
                    rendered         INTEGER NOT NULL DEFAULT 0,
                    suggestion_id    INTEGER
                )
                """,
            )
            # Schemas created before streams existed: add the column in place.
            self._ddl(cur, "ALTER TABLE ml_signals ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT ''")
            self._ddl(cur, "CREATE INDEX IF NOT EXISTS idx_ml_signals_created_at ON ml_signals(created_at)")
            self._ddl(cur, "CREATE INDEX IF NOT EXISTS idx_ml_signals_rendered ON ml_signals(rendered)")
            # Whoever created them, every engine table must now exist.
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s AND table_name = ANY(%s)",
                (self._tenant, list(self.ENGINE_TABLES)),
            )
            present = {row[0] for row in cur.fetchall()}
            missing = [t for t in self.ENGINE_TABLES if t not in present]
            if missing:
                conn.rollback()
                raise RuntimeError(
                    f"postgres: engine tables missing in schema {self._tenant!r}: {', '.join(missing)} "
                    "(the role cannot create them; the platform must provision them)"
                )
            # The cursor row for this stream (DML, which the worker may always do).
            cur.execute(
                """
                INSERT INTO ml_cursor (stream, last_event_id, updated_at)
                VALUES (%s, 0, 0)
                ON CONFLICT (stream) DO NOTHING
                """,
                (self._uid,),
            )
        conn.commit()
        logger.info(
            "postgres: engine tables ensured in schema %s (stream %s)", self._tenant, self._stream or "<unscoped>"
        )

    @property
    def tenant(self) -> str:
        """The schema this store is bound to."""
        return self._tenant

    @property
    def stream(self) -> str | None:
        """The device stream this store is scoped to, or ``None`` when unscoped."""
        return self._stream

    def list_streams(self) -> list[str]:
        """The distinct ``user_id`` values present in this schema's ``events``.

        The cloud worker's discovery: one worker per stream. Empty when the
        schema has no ``user_id`` column (a single-stream schema) or no rows.
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT user_id FROM events WHERE user_id IS NOT NULL ORDER BY user_id")
                return [str(row[0]) for row in cur.fetchall() if row[0] not in (None, "")]
        except Exception:
            conn.rollback()
            logger.debug("list_streams: events.user_id not available in schema %s", self._tenant)
            return []

    # --- Cursor operations ---

    def get_cursor(self) -> int:
        """Return the last processed event ID from ml_cursor."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT last_event_id FROM ml_cursor WHERE stream = %s", (self._uid,))
            row = cur.fetchone()
            return row[0] if row else 0

    def update_cursor(self, event_id: int) -> None:
        """Update ml_cursor.last_event_id to the given event_id."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ml_cursor (stream, last_event_id, updated_at) VALUES (%s, %s, %s) "
                "ON CONFLICT (stream) DO UPDATE SET last_event_id = EXCLUDED.last_event_id, "
                "updated_at = EXCLUDED.updated_at",
                (self._uid, event_id, int(time.time() * 1000)),
            )

    # --- Event queries ---

    def get_events_since(self, since_id: int, limit: int = 100) -> list[dict[str, Any]]:
        """Return events with id > since_id, ordered by id ASC, up to limit."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, kind, source, payload, ts FROM events WHERE id > %s{self._sc} ORDER BY id ASC LIMIT %s",
                (since_id, *self._sp, limit),
            )
            columns = ["id", "kind", "source", "payload", "ts"]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def get_events_for_task(self, task_id: str, since: int | None = None) -> list[dict[str, Any]]:
        """Return events within a task's time window, with JSON payloads parsed."""
        task = self.get_task_by_id(task_id)
        if task is None:
            return []

        start = since if since is not None else task.get("started_at", 0)
        end = task.get("completed_at") or task.get("last_active") or int(time.time() * 1000)

        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM events WHERE ts >= %s AND ts <= %s{self._sc} ORDER BY ts",
                (start, end, *self._sp),
            )
            if cur.description is None:
                return []
            columns = [desc[0] for desc in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]

        # Parse JSON payload
        for row in rows:
            if isinstance(row.get("payload"), str):
                try:
                    row["payload"] = json.loads(row["payload"])
                except json.JSONDecodeError, TypeError:
                    pass
        return rows

    # --- Task queries ---

    def get_active_task(self) -> str | None:
        """Return the ID of the active (non-idle, not completed) task, or None."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM tasks WHERE phase != 'idle' AND completed_at IS NULL"
                f"{self._sc} ORDER BY last_active DESC LIMIT 1",
                self._sp,
            )
            row = cur.fetchone()
            return row[0] if row else None

    def get_task_by_id(self, task_id: str) -> dict[str, Any] | None:
        """Return a full task row as a dict, or None if not found."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM tasks WHERE id = %s{self._sc}", (task_id, *self._sp))
            if cur.description is None:
                return None
            columns = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            if row is None:
                return None
            return dict(zip(columns, row))

    def get_session_info(self, task_id: str) -> dict[str, Any] | None:
        """Return started_at, phase, test_fails for a task."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT started_at, phase, test_fails FROM tasks WHERE id = %s{self._sc}",
                (task_id, *self._sp),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {"started_at": row[0], "phase": row[1], "test_fails": row[2]}

    def get_quality_task_stats(self) -> dict[str, Any] | None:
        """Return test_runs, test_fails, commit_count from the most recently completed task."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT test_runs, test_fails, commit_count FROM tasks "
                f"WHERE completed_at IS NOT NULL{self._sc} ORDER BY completed_at DESC LIMIT 1",
                self._sp,
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {"test_runs": row[0], "test_fails": row[1], "commit_count": row[2]}

    def get_completed_task_ids(self) -> list[str]:
        """Return IDs of all completed tasks."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(f"SELECT id FROM tasks WHERE completed_at IS NOT NULL{self._sc}", self._sp)
            return [row[0] for row in cur.fetchall()]

    def get_completed_tasks_with_timestamps(self) -> list[dict[str, Any]]:
        """Return id, started_at, completed_at for completed tasks with both timestamps set."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, started_at, completed_at FROM tasks "
                f"WHERE completed_at IS NOT NULL AND started_at IS NOT NULL{self._sc}",
                self._sp,
            )
            return [{"id": row[0], "started_at": row[1], "completed_at": row[2]} for row in cur.fetchall()]

    def count_completed_tasks(self) -> int:
        """Return the count of completed tasks."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM tasks WHERE completed_at IS NOT NULL{self._sc}", self._sp)
            row = cur.fetchone()
            return row[0] if row else 0

    # --- Status data ---

    def get_status_data(self) -> dict[str, Any]:
        """Return cursor info and latest non-expired predictions for /status."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT last_event_id, updated_at FROM ml_cursor WHERE stream = %s", (self._uid,))
            cursor_row = cur.fetchone()
            cursor_data = None
            if cursor_row:
                cursor_data = {"last_event_id": cursor_row[0], "updated_at": cursor_row[1]}

            now_ms = int(time.time() * 1000)
            cur.execute(
                "SELECT model, confidence, created_at FROM ml_predictions "
                f"WHERE (expires_at IS NULL OR expires_at > %s){self._sc} "
                "ORDER BY created_at DESC",
                (now_ms, *self._sp),
            )
            preds = [{"model": row[0], "confidence": row[1], "created_at": row[2]} for row in cur.fetchall()]

        return {
            "cursor": cursor_data,
            "latest_predictions": preds,
        }

    # --- Write operations (ml_predictions, ml_events only) ---

    def insert_prediction(self, model: str, result: dict, confidence: float, ttl_sec: int | None) -> None:
        """Insert a row into ml_predictions."""
        conn = self._get_conn()
        now_ms = int(time.time() * 1000)
        expires_ms = (now_ms + ttl_sec * 1000) if ttl_sec else None
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ml_predictions (user_id, model, result, confidence, created_at, expires_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (self._uid, model, json.dumps(result), round(confidence, 4), now_ms, expires_ms),
            )

    def insert_ml_event(self, kind: str, endpoint: str, routing: str, latency_ms: int) -> None:
        """Insert a row into ml_events."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ml_events (user_id, kind, endpoint, routing, latency_ms, ts) VALUES (%s, %s, %s, %s, %s, %s)",
                (self._uid, kind, endpoint, routing, latency_ms, int(time.time() * 1000)),
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
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ml_signals "
                "(user_id, signal_type, confidence, evidence, suggested_action, created_at, expires_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    self._uid,
                    signal_type,
                    round(confidence, 4),
                    json.dumps(evidence),
                    suggested_action,
                    now_ms,
                    expires_ms,
                ),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("insert_signal: INSERT ... RETURNING id returned no row")
            return row[0]

    def get_signal_feedback(self, since_ms: int) -> list[dict]:
        """Read feedback linkages from suggestions table for training."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT s.signal_id, ms.signal_type, s.status, s.created_at "
                    "FROM suggestions s "
                    "JOIN ml_signals ms ON s.signal_id = ms.id "
                    "WHERE s.signal_id IS NOT NULL AND s.created_at > %s "
                    "ORDER BY s.created_at ASC",
                    (since_ms,),
                )
                return [
                    {"signal_id": r[0], "signal_type": r[1], "status": r[2], "created_at": r[3]} for r in cur.fetchall()
                ]
        except Exception:
            conn.rollback()
            logger.debug("get_signal_feedback: suggestions.signal_id not available yet")
            return []

    # --- Cloud training methods ---

    def get_last_training_ts(self, tenant_id: str) -> float | None:
        """Return the last training timestamp (epoch ms) for a tenant, or None."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT MAX(ts) FROM ml_events WHERE kind = 'training' AND routing = %s",
                (tenant_id,),
            )
            row = cur.fetchone()
            return float(row[0]) if row and row[0] is not None else None

    def get_completed_tasks_for_tenant(self, tenant_id: str) -> list[dict]:
        """Return completed tasks for a specific tenant."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM tasks WHERE completed_at IS NOT NULL{self._sc} ORDER BY completed_at DESC", self._sp
            )
            if cur.description is None:
                return []
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def get_events_for_task_id(self, task_id: str) -> list[dict]:
        """Return events associated with a specific task ID."""
        return self.get_events_for_task(task_id)

    def get_all_tenant_ids(self) -> list[str]:
        """Return all known tenant IDs from the information schema."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name NOT IN ('public', 'information_schema', 'pg_catalog', 'pg_toast')"
            )
            return [row[0] for row in cur.fetchall()]

    def get_opted_in_tenant_ids(self) -> list[str]:
        """Return tenant IDs opted in to aggregate data pooling.

        Placeholder: returns all tenant IDs until an opt-in mechanism is implemented.
        """
        return self.get_all_tenant_ids()

    def record_training_run(self, tenant_id: str, status: str, duration_ms: int) -> None:
        """Record a training run audit entry for a tenant."""
        self.insert_ml_event(
            kind="training",
            endpoint="cloud_trainer",
            routing=tenant_id,
            latency_ms=duration_ms,
        )
