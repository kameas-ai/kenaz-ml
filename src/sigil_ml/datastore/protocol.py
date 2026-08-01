"""DataStore protocol — the central abstraction for all data access in kameas-ml.

All modules (poller, routes, trainer, scheduler) depend on this protocol
instead of importing sqlite3 or psycopg2 directly.

Implementations:
    SqliteStore  (local mode)  — src/sigil_ml/store_sqlite.py
    PostgresStore (cloud mode) — src/sigil_ml/store_postgres.py
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DataStore(Protocol):
    """Protocol for all data access operations in kameas-ml.

    Implementations: SqliteStore (local), PostgresStore (cloud).
    Python only writes to ml_predictions, ml_events, ml_cursor, ml_signals.
    Python only reads from events, tasks, patterns, suggestions.
    """

    def ensure_tables(self) -> None:
        """Create Python-owned tables if they don't exist (ml_cursor)."""
        ...

    def get_cursor(self) -> int:
        """Return the last processed event ID from ml_cursor. Returns 0 if no cursor."""
        ...

    def update_cursor(self, event_id: int) -> None:
        """Update ml_cursor.last_event_id to the given event_id."""
        ...

    def get_events_since(self, since_id: int, limit: int = 100) -> list[dict[str, Any]]:
        """Return events with id > since_id, ordered by id ASC, up to limit.

        Each dict has keys: id, kind, source, payload (raw string), ts.
        """
        ...

    def get_active_task(self) -> str | None:
        """Return the ID of the active (non-idle, not completed) task, or None."""
        ...

    def get_task_by_id(self, task_id: str) -> dict[str, Any] | None:
        """Return a full task row as a dict, or None if not found."""
        ...

    def get_events_for_task(self, task_id: str, since: int | None = None) -> list[dict[str, Any]]:
        """Return events within a task's time window [started_at..completed_at/last_active].

        If since is provided, use it as the lower bound instead of started_at.
        JSON payloads are parsed into dicts.
        """
        ...

    def get_session_info(self, task_id: str) -> dict[str, Any] | None:
        """Return started_at, phase, test_fails for a task. None if not found."""
        ...

    def get_quality_task_stats(self) -> dict[str, Any] | None:
        """Return test_runs, test_fails, commit_count from the most recently completed task.

        Returns None if no completed tasks.
        """
        ...

    def get_completed_task_ids(self) -> list[str]:
        """Return IDs of all completed tasks (for training)."""
        ...

    def get_completed_tasks_with_timestamps(self) -> list[dict[str, Any]]:
        """Return id, started_at, completed_at for completed tasks with both timestamps set."""
        ...

    def count_completed_tasks(self) -> int:
        """Return the count of completed tasks."""
        ...

    def get_status_data(self) -> dict[str, Any]:
        """Return cursor info and latest non-expired predictions for the /status endpoint."""
        ...

    def insert_prediction(self, model: str, result: dict, confidence: float, ttl_sec: int | None) -> None:
        """Insert a row into ml_predictions."""
        ...

    def insert_ml_event(self, kind: str, endpoint: str, routing: str, latency_ms: int) -> None:
        """Insert a row into ml_events."""
        ...

    def insert_signal(
        self,
        signal_type: str,
        confidence: float,
        evidence: dict,
        suggested_action: str | None = None,
        ttl_sec: int | None = None,
    ) -> int:
        """Insert a signal into ml_signals. Returns the signal ID.

        Args:
            signal_type: Model-generated type (e.g., "velocity_deviation").
            confidence: Model's confidence score (0.0 to 1.0).
            evidence: Structured JSON evidence for LLM rendering.
            suggested_action: Optional generic action hint (e.g., "test", "commit").
            ttl_sec: Optional time-to-live in seconds. None = no expiry.

        Returns:
            The auto-generated integer ID of the inserted signal.
        """
        ...

    def get_signal_feedback(self, since_ms: int) -> list[dict]:
        """Read feedback linkages from suggestions table for training.

        Returns rows where a suggestion was linked to an ml_signal
        (via signal_id column) and has a status of accepted/dismissed/ignored.

        Args:
            since_ms: Only return feedback newer than this Unix ms timestamp.

        Returns:
            List of dicts with keys: signal_id, signal_type, status, created_at.
        """
        ...

    def commit(self) -> None:
        """Commit the current transaction (for backends that batch writes)."""
        ...

    def close(self) -> None:
        """Close the underlying database connection."""
        ...

    # --- Cloud training methods ---

    def get_last_training_ts(self, tenant_id: str) -> float | None:
        """Return the last training timestamp (epoch ms) for a tenant, or None if never trained."""
        ...

    def get_completed_tasks_for_tenant(self, tenant_id: str) -> list[dict]:
        """Return completed tasks for a specific tenant (cloud multi-tenant queries)."""
        ...

    def get_events_for_task_id(self, task_id: str) -> list[dict]:
        """Return events associated with a specific task ID."""
        ...

    def get_all_tenant_ids(self) -> list[str]:
        """Return all known tenant IDs."""
        ...

    def get_opted_in_tenant_ids(self) -> list[str]:
        """Return tenant IDs opted in to aggregate data pooling."""
        ...

    def record_training_run(self, tenant_id: str, status: str, duration_ms: int) -> None:
        """Record a training run audit entry for a tenant."""
        ...


def create_store(mode: str | None = None) -> DataStore:
    """Create the appropriate DataStore based on operating mode.

    Args:
        mode: "local" or "cloud". Defaults to reading from config/environment.

    Returns:
        A DataStore implementation (SqliteStore or PostgresStore).
    """
    from sigil_ml import config
    from sigil_ml.datastore.sqlite import SqliteStore

    resolved_mode = mode or config.operating_mode()

    if resolved_mode == "cloud":
        from sigil_ml.datastore.postgres import PostgresStore

        url = config.postgres_url()
        if not url:
            raise ValueError("SIGIL_POSTGRES_URL environment variable is required in cloud mode")
        tenant = config.tenant_id()
        return PostgresStore(connection_url=url, tenant=tenant)

    return SqliteStore(config.db_path())
