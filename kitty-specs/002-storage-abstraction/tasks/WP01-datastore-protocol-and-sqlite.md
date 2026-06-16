---
work_package_id: "WP01"
title: "DataStore Protocol & SQLite Implementation"
lane: "planned"
dependencies: []
subtasks:
  - "T001"
  - "T002"
  - "T003"
  - "T004"
  - "T005"
phase: "Phase 1 - Foundation"
assignee: ""
agent: ""
shell_pid: ""
review_status: ""
reviewed_by: ""
history:
  - timestamp: "2026-03-30T01:45:06Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt generated via /spec-kitty.tasks"
requirement_refs:
  - "FR-001"
  - "FR-002"
  - "FR-003"
  - "FR-008"
  - "FR-010"
  - "FR-011"
---

# Work Package Prompt: WP01 -- DataStore Protocol & SQLite Implementation

## Important: Review Feedback Status

**Read this first if you are implementing this task!**

- **Has review feedback?**: Check the `review_status` field above. If it says `has_feedback`, scroll to the **Review Feedback** section immediately.
- **You must address all feedback** before your work is complete.
- **Mark as acknowledged**: When you understand the feedback and begin addressing it, update `review_status: acknowledged` in the frontmatter.

---

## Review Feedback

> **Populated by `/spec-kitty.review`** -- Reviewers add detailed feedback here when work needs changes.

*[This section is empty initially.]*

---

## Markdown Formatting
Wrap HTML/XML tags in backticks: `` `<div>` ``, `` `<script>` ``
Use language identifiers in code blocks: ````python`, ````bash`

---

## Objectives & Success Criteria

- Define a `DataStore` protocol in `src/sigil_ml/store.py` that covers **every** SQL operation currently scattered across `poller.py`, `features.py`, `routes.py`, `trainer.py`, and `scheduler.py`.
- Implement `SqliteStore` in `src/sigil_ml/store_sqlite.py` that preserves identical behavior to today's inline `sqlite3` calls.
- Implement a `create_store()` factory function that selects the right backend based on configuration.
- After this WP, importing `SqliteStore` and calling every method must produce the same results as the current direct SQL calls.

**Success criteria (from spec)**:
- SC-001: All existing tests pass without modification after the refactor.
- SC-005: Both backends produce identical results for the same input data.
- FR-010: `SqliteStore` enforces `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000` on every connection.

**Implementation command**: `spec-kitty implement WP01`

## Context & Constraints

- **Spec**: `kitty-specs/002-storage-abstraction/spec.md`
- **Tasks overview**: `kitty-specs/002-storage-abstraction/tasks.md`
- **Constraint (CLAUDE.md)**: No heavyweight dependencies -- `scikit-learn`, `numpy`, `fastapi`, `uvicorn`, `joblib` only. The DataStore protocol uses only stdlib + typing.
- **Constraint (CLAUDE.md)**: Python never writes to `events`, `tasks`, `patterns`, or `suggestions`. Python only writes to `ml_predictions`, `ml_events`, `ml_cursor`.
- **Constraint (CLAUDE.md)**: Every SQLite connection must set `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000`.

### Complete SQL Audit

Before defining the protocol, here is every SQL operation in the current codebase:

**poller.py**:
1. `SELECT last_event_id FROM ml_cursor WHERE id = 1` (line 65) -- get cursor
2. `SELECT id, kind, source, payload, ts FROM events WHERE id > ? ORDER BY id ASC LIMIT 100` (line 68-71) -- get events since cursor
3. `UPDATE ml_cursor SET last_event_id = ?, updated_at = ? WHERE id = 1` (line 97-99) -- update cursor
4. `INSERT INTO ml_predictions (model, result, confidence, created_at, expires_at) VALUES (?, ?, ?, ?, ?)` (line 244-246) -- write prediction
5. `INSERT INTO ml_events (kind, endpoint, routing, latency_ms, ts) VALUES (...)` (line 179-183) -- write audit event
6. `SELECT id FROM tasks WHERE phase != 'idle' AND completed_at IS NULL ORDER BY last_active DESC LIMIT 1` (line 250-252) -- get active task
7. `SELECT started_at, phase, test_fails FROM tasks WHERE id = ?` (line 218-220) -- get session info
8. `SELECT test_runs, test_fails, commit_count FROM tasks WHERE completed_at IS NOT NULL ORDER BY completed_at DESC LIMIT 1` (line 272-274) -- get quality task stats

**features.py**:
9. `SELECT * FROM tasks WHERE id = ?` (line 75) -- get task by id
10. `SELECT * FROM events WHERE ts >= ? AND ts <= ? ORDER BY ts` (line 102-105) -- get events for task time window

**routes.py**:
11. `SELECT last_event_id, updated_at FROM ml_cursor WHERE id = 1` (line 128) -- status: cursor info
12. `SELECT model, confidence, created_at FROM ml_predictions WHERE expires_at IS NULL OR expires_at > ? ORDER BY created_at DESC` (lines 129-133) -- status: latest predictions

**trainer.py**:
13. `SELECT id FROM tasks WHERE completed_at IS NOT NULL` (line 66) -- get completed task IDs (for stuck training)
14. `SELECT id, started_at, completed_at FROM tasks WHERE completed_at IS NOT NULL AND started_at IS NOT NULL` (line 107-109) -- get completed tasks with timestamps (for duration training)

**scheduler.py**:
15. `SELECT COUNT(*) FROM tasks WHERE completed_at IS NOT NULL` (line 60) -- count completed tasks
16. `INSERT INTO ml_events (kind, endpoint, routing, latency_ms, ts) VALUES (...)` (line 73-79) -- log retrain event (same shape as #5)

**schema.py**:
17. `CREATE TABLE IF NOT EXISTS ml_cursor ...` + `INSERT OR IGNORE INTO ml_cursor ...` (lines 23-30) -- ensure tables

---

## Subtasks & Detailed Guidance

### Subtask T001 -- Define the `DataStore` protocol in `src/sigil_ml/store.py`

- **Purpose**: Create the central interface that all modules will depend on instead of `sqlite3` directly. This protocol defines the contract that `SqliteStore`, `PostgresStore`, and any future backends must satisfy.
- **Files**: Create new file `src/sigil_ml/store.py`
- **Parallel?**: No -- must be done first; T002-T004 depend on these signatures.

**Steps**:
1. Create `src/sigil_ml/store.py` with a `DataStore` class using `typing.Protocol` (runtime-checkable):

```python
from __future__ import annotations
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class DataStore(Protocol):
    """Protocol for all data access operations in kenaz-ml.

    Implementations: SqliteStore (local), PostgresStore (cloud).
    Python only writes to ml_predictions, ml_events, ml_cursor.
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

    def insert_prediction(
        self, model: str, result: dict, confidence: float, ttl_sec: int | None
    ) -> None:
        """Insert a row into ml_predictions."""
        ...

    def insert_ml_event(
        self, kind: str, endpoint: str, routing: str, latency_ms: int
    ) -> None:
        """Insert a row into ml_events."""
        ...

    def commit(self) -> None:
        """Commit the current transaction (for backends that batch writes)."""
        ...
```

2. Add a `create_store()` factory stub that returns `SqliteStore` for now (full implementation in T005).

**Notes**:
- Use `from __future__ import annotations` for forward references.
- `@runtime_checkable` allows `isinstance(store, DataStore)` checks.
- The `commit()` method enables the poller's transactional pattern (poll + predict + cursor update in one commit).
- Payload is returned as a raw string from `get_events_since` -- callers handle JSON parsing. This matches current behavior.
- `get_status_data()` combines cursor + prediction queries for the `/status` endpoint to avoid exposing raw SQL in routes.

---

### Subtask T002 -- Implement `SqliteStore` -- cursor and schema operations

- **Purpose**: Implement the connection lifecycle, cursor management, and schema bootstrap operations.
- **Files**: Create new file `src/sigil_ml/store_sqlite.py`
- **Parallel?**: Yes -- can proceed in parallel with T003, T004 after T001 is done.

**Steps**:
1. Create `src/sigil_ml/store_sqlite.py` with:

```python
import sqlite3
import time
from pathlib import Path

class SqliteStore:
    """DataStore implementation backed by a local SQLite file."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
```

2. Implement connection management:
   - `_connect()` creates a connection with `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000` (matching `poller.py` line 296-298 and `schema.py` line 20-21).
   - `_get_conn()` returns the existing connection or creates a new one (lazy initialization).
   - `commit()` calls `self._conn.commit()` if a connection exists.
   - `close()` closes the connection if open.

3. Implement `ensure_tables()`:
   - Extract the exact SQL from `schema.py` lines 23-30:
     ```sql
     CREATE TABLE IF NOT EXISTS ml_cursor (
         id            INTEGER PRIMARY KEY CHECK (id = 1),
         last_event_id INTEGER NOT NULL DEFAULT 0,
         updated_at    INTEGER NOT NULL DEFAULT 0
     );
     INSERT OR IGNORE INTO ml_cursor (id, last_event_id, updated_at) VALUES (1, 0, 0);
     ```
   - Use `conn.executescript()` followed by `conn.commit()`.

4. Implement `get_cursor()`:
   - From `poller.py` line 65: `SELECT last_event_id FROM ml_cursor WHERE id = 1`
   - Return 0 if no row exists.

5. Implement `update_cursor(event_id)`:
   - From `poller.py` lines 97-99: `UPDATE ml_cursor SET last_event_id = ?, updated_at = ? WHERE id = 1`
   - Use `int(time.time() * 1000)` for `updated_at` (millisecond timestamp).

**Edge cases**:
- Database file doesn't exist: `sqlite3.connect()` creates it automatically (current behavior preserved).
- Connection already open: Reuse it (important for transactional consistency in poller).

---

### Subtask T003 -- Implement `SqliteStore` -- event and task querying operations

- **Purpose**: Implement all read operations for events and tasks tables.
- **Files**: `src/sigil_ml/store_sqlite.py` (continue from T002)
- **Parallel?**: Yes -- can proceed in parallel with T002, T004.

**Steps**:
1. Implement `get_events_since(since_id, limit=100)`:
   - From `poller.py` lines 68-71:
     ```sql
     SELECT id, kind, source, payload, ts FROM events
     WHERE id > ? ORDER BY id ASC LIMIT ?
     ```
   - Return list of dicts with keys: `id`, `kind`, `source`, `payload`, `ts`.
   - Payload remains as raw string (caller parses JSON).

2. Implement `get_active_task()`:
   - From `poller.py` lines 250-252:
     ```sql
     SELECT id FROM tasks WHERE phase != 'idle'
     AND completed_at IS NULL ORDER BY last_active DESC LIMIT 1
     ```
   - Return `str | None`.

3. Implement `get_task_by_id(task_id)`:
   - From `features.py` lines 73-79:
     ```sql
     SELECT * FROM tasks WHERE id = ?
     ```
   - Use `sqlite3.Row` row factory to return a full dict.
   - Return `None` if not found.

4. Implement `get_events_for_task(task_id, since=None)`:
   - Combines logic from `features.py` lines 84-116:
     - First query the task to get `started_at` and `completed_at`/`last_active` bounds.
     - Then query events in that time window:
       ```sql
       SELECT * FROM events WHERE ts >= ? AND ts <= ? ORDER BY ts
       ```
   - Parse JSON payload in each event row (matching `features.py` lines 108-113).
   - Return empty list if task not found.

5. Implement `get_session_info(task_id)`:
   - From `poller.py` lines 218-220:
     ```sql
     SELECT started_at, phase, test_fails FROM tasks WHERE id = ?
     ```
   - Return `{"started_at": int, "phase": str, "test_fails": int}` or `None`.

6. Implement `get_quality_task_stats()`:
   - From `poller.py` lines 272-274:
     ```sql
     SELECT test_runs, test_fails, commit_count FROM tasks
     WHERE completed_at IS NOT NULL ORDER BY completed_at DESC LIMIT 1
     ```
   - Return `{"test_runs": int, "test_fails": int, "commit_count": int}` or `None`.

7. Implement `get_completed_task_ids()`:
   - From `trainer.py` line 66:
     ```sql
     SELECT id FROM tasks WHERE completed_at IS NOT NULL
     ```
   - Return `list[str]`.

8. Implement `get_completed_tasks_with_timestamps()`:
   - From `trainer.py` lines 107-109:
     ```sql
     SELECT id, started_at, completed_at FROM tasks
     WHERE completed_at IS NOT NULL AND started_at IS NOT NULL
     ```
   - Return list of dicts.

9. Implement `count_completed_tasks()`:
   - From `scheduler.py` line 60:
     ```sql
     SELECT COUNT(*) FROM tasks WHERE completed_at IS NOT NULL
     ```
   - Return `int`. Return 0 on `OperationalError` (matching scheduler.py line 64).

10. Implement `get_status_data()`:
    - Combines `routes.py` lines 128-133:
      ```sql
      SELECT last_event_id, updated_at FROM ml_cursor WHERE id = 1
      SELECT model, confidence, created_at FROM ml_predictions
      WHERE expires_at IS NULL OR expires_at > ? ORDER BY created_at DESC
      ```
    - Return `{"cursor": dict | None, "latest_predictions": list[dict]}`.

**Notes**:
- For `get_events_for_task`, the end bound logic is: `task.completed_at or task.last_active or now_ms` (from `features.py` line 97).
- Use `sqlite3.Row` as `row_factory` where full row access is needed.

---

### Subtask T004 -- Implement `SqliteStore` -- prediction, audit writing, and quality stats operations

- **Purpose**: Implement all write operations to `ml_predictions` and `ml_events` tables.
- **Files**: `src/sigil_ml/store_sqlite.py` (continue from T002/T003)
- **Parallel?**: Yes -- can proceed in parallel with T002, T003.

**Steps**:
1. Implement `insert_prediction(model, result, confidence, ttl_sec)`:
   - From `poller.py` lines 242-247:
     ```python
     now_ms = int(time.time() * 1000)
     expires_ms = (now_ms + ttl_sec * 1000) if ttl_sec else None
     conn.execute(
         "INSERT INTO ml_predictions (model, result, confidence, created_at, expires_at) "
         "VALUES (?, ?, ?, ?, ?)",
         (model, json.dumps(result), round(confidence, 4), now_ms, expires_ms),
     )
     ```
   - Note: `result` is serialized to JSON string.
   - Note: `confidence` is rounded to 4 decimal places.
   - Note: `expires_at` is `None` for predictions without TTL (e.g., duration).

2. Implement `insert_ml_event(kind, endpoint, routing, latency_ms)`:
   - From `poller.py` lines 179-183 and `scheduler.py` lines 73-79:
     ```sql
     INSERT INTO ml_events (kind, endpoint, routing, latency_ms, ts)
     VALUES (?, ?, ?, ?, ?)
     ```
   - `ts` is `int(time.time() * 1000)`.
   - Used for both prediction audit logs (`kind='prediction'`) and retrain logs (`kind='retrain'`).

**Validation**:
- [ ] `insert_prediction` with TTL produces correct `expires_at` (now_ms + ttl * 1000).
- [ ] `insert_prediction` with `ttl_sec=None` produces `expires_at=NULL`.
- [ ] `insert_ml_event` timestamps are in milliseconds.
- [ ] Model names match Go's expectations: `"stuck"`, `"suggest"`, `"duration"`, `"quality"`, `"activity"`.

---

### Subtask T005 -- Add `create_store()` factory function in `src/sigil_ml/store.py`

- **Purpose**: Provide a single entry point for creating the appropriate DataStore backend based on runtime configuration.
- **Files**: `src/sigil_ml/store.py` (append to the file created in T001)
- **Parallel?**: No -- should be done after T001 defines the protocol and T002-T004 implement SqliteStore.

**Steps**:
1. Add to `src/sigil_ml/store.py`:

```python
from sigil_ml import config
from sigil_ml.store_sqlite import SqliteStore

def create_store(mode: str | None = None) -> DataStore:
    """Create the appropriate DataStore based on operating mode.

    Args:
        mode: "local" or "cloud". Defaults to reading from config/environment.

    Returns:
        A DataStore implementation (SqliteStore or PostgresStore).
    """
    resolved_mode = mode or os.environ.get("SIGIL_MODE", "local")

    if resolved_mode == "cloud":
        # PostgresStore will be implemented in WP04
        raise NotImplementedError("Postgres backend not yet implemented (see WP04)")

    return SqliteStore(config.db_path())
```

2. The cloud/Postgres branch raises `NotImplementedError` until WP04 implements `PostgresStore`.

3. The `mode` parameter allows explicit override for testing without changing environment variables.

**Notes**:
- Import `SqliteStore` at module level (it's always available).
- Import `PostgresStore` conditionally (lazy import) since `psycopg2` is optional. This will be done in WP04.
- The factory function is the only place that knows about concrete implementations.

---

## Risks & Mitigations

- **Risk**: Protocol may miss an operation discovered later during WP02/WP03 refactoring. **Mitigation**: The SQL audit above is exhaustive (17 distinct operations). If a new operation is found, add it to the protocol and both implementations.
- **Risk**: `SqliteStore` connection lifecycle differs from inline code (inline code creates + closes per operation; SqliteStore may keep a connection open). **Mitigation**: SqliteStore uses lazy connection creation and callers can call `commit()` to flush. The poller's single-connection-per-cycle pattern is preserved.
- **Risk**: Thread safety -- the poller runs in a thread executor. **Mitigation**: SQLite in WAL mode with busy_timeout handles concurrent access. Each poll cycle uses its own connection scope.

## Review Guidance

- Verify the protocol covers ALL 17 SQL operations identified in the audit.
- Verify `SqliteStore` enforces WAL mode and busy_timeout on every connection (FR-010).
- Verify `insert_prediction` serializes `result` to JSON and rounds `confidence` to 4 decimals.
- Verify `get_events_for_task` handles the case where task has no `completed_at` (falls back to `last_active` or now).
- Verify the factory function defaults to SQLite when no mode is specified.

## Activity Log

- 2026-03-30T01:45:06Z -- system -- lane=planned -- Prompt generated via /spec-kitty.tasks
