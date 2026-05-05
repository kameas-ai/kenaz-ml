---
work_package_id: "WP04"
title: "Postgres Implementation"
lane: "planned"
dependencies: ["WP01"]
subtasks:
  - "T017"
  - "T018"
  - "T019"
  - "T020"
  - "T021"
  - "T022"
phase: "Phase 3 - Cloud Backend"
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
  - "FR-004"
  - "FR-008"
  - "FR-009"
  - "FR-011"
  - "FR-012"
---

# Work Package Prompt: WP04 -- Postgres Implementation

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

- Implement `PostgresStore` -- a `DataStore` backend that connects to a Postgres database via a connection URL.
- Support per-tenant schema isolation (e.g., `tenant_abc.ml_predictions`).
- Produce identical results to `SqliteStore` for the same input data (FR-011).
- Respect table ownership rules: Python only writes to `ml_predictions`, `ml_events`, `ml_cursor` (FR-012).
- Update `create_store()` factory to instantiate `PostgresStore` when operating in cloud mode.

**Success criteria**:
- FR-004: Postgres implementation of DataStore exists and supports per-tenant data isolation.
- FR-009: Postgres backend accepts connection URL via `SIGIL_POSTGRES_URL`.
- SC-004: Switching from SQLite to Postgres requires only a configuration change.
- SC-005: Both backends produce identical results for the same input data.

**Implementation command**: `spec-kitty implement WP04 --base WP01`

## Context & Constraints

- **Prerequisite**: WP01 must be complete (DataStore protocol defined). WP04 can proceed in parallel with WP02/WP03 since it only depends on the protocol, not the refactored callers.
- **Spec**: `kitty-specs/002-storage-abstraction/spec.md` -- User Story 2 (Cloud Mode Uses Postgres).
- **CLAUDE.md constraint**: No heavyweight dependencies beyond `scikit-learn`, `numpy`, `fastapi`, `uvicorn`, `joblib`. `psycopg2-binary` is acceptable as an optional dependency.
- **Table ownership**: Python only writes to `ml_predictions`, `ml_events`, `ml_cursor`. Python only reads from `events`, `tasks`, `patterns`, `suggestions`. Go owns and migrates the main schema.

### Postgres vs SQLite: Key Differences

| Feature | SQLite | Postgres |
|---------|--------|----------|
| Auto-increment PK | `INTEGER PRIMARY KEY` | `SERIAL` or `BIGSERIAL` |
| WAL mode | `PRAGMA journal_mode=WAL` | Not applicable (built-in MVCC) |
| Busy timeout | `PRAGMA busy_timeout=5000` | Connection pool wait timeout |
| Schema isolation | Not applicable (single db file) | `SET search_path TO tenant_schema` |
| JSON handling | Stored as TEXT, parsed by Python | `JSONB` type available (use TEXT for compatibility) |
| Parameterized queries | `?` placeholders | `%s` placeholders |
| Boolean | 0/1 integers | Native `BOOLEAN` (but integers work) |
| Timestamps | Stored as integers (ms since epoch) | Same convention for compatibility |

---

## Subtasks & Detailed Guidance

### Subtask T017 -- Add `psycopg2-binary` as optional dependency

- **Purpose**: Add the Postgres adapter without making it a required dependency for local-only users.
- **Files**: `pyproject.toml` (or `setup.cfg` / `setup.py` depending on project structure)
- **Parallel?**: No -- should be done first so other subtasks can import psycopg2.

**Steps**:
1. Check the project's dependency management format:
   ```bash
   ls pyproject.toml setup.py setup.cfg
   ```

2. Add `psycopg2-binary` as an optional dependency group:

   **For pyproject.toml (Poetry)**:
   ```toml
   [tool.poetry.extras]
   postgres = ["psycopg2-binary"]
   ```

   **For pyproject.toml (setuptools)**:
   ```toml
   [project.optional-dependencies]
   postgres = ["psycopg2-binary>=2.9"]
   ```

3. The dependency is only imported when `PostgresStore` is instantiated (lazy import), so local-only installations don't need it.

4. Document installation for cloud users:
   ```bash
   pip install -e ".[postgres]"
   ```

---

### Subtask T018 -- Implement `PostgresStore` -- connection management and tenant schema

- **Purpose**: Create the PostgresStore class with connection lifecycle management and per-tenant schema isolation.
- **Files**: Create new file `src/sigil_ml/store_postgres.py`
- **Parallel?**: No -- must be done first; T019-T021 depend on connection management.

**Steps**:
1. Create `src/sigil_ml/store_postgres.py`:

```python
"""DataStore implementation backed by PostgreSQL."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class PostgresStore:
    """DataStore implementation backed by a PostgreSQL database.

    Supports per-tenant schema isolation. Each tenant's tables live
    in a dedicated Postgres schema (e.g., tenant_abc.events).

    Args:
        connection_url: Postgres connection URL (e.g., postgresql://user:pass@host:5432/dbname)
        tenant: Tenant identifier for schema isolation. Defaults to "public".
    """

    def __init__(self, connection_url: str, tenant: str = "public") -> None:
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2-binary is required for PostgresStore. "
                "Install with: pip install kameas-ml[postgres]"
            ) from None

        self._connection_url = connection_url
        self._tenant = tenant
        self._conn = None
        self._psycopg2 = psycopg2
```

2. Implement connection management:
   ```python
   def _get_conn(self):
       """Return existing connection or create a new one with tenant schema."""
       if self._conn is None or self._conn.closed:
           self._conn = self._psycopg2.connect(self._connection_url)
           self._conn.autocommit = False
           with self._conn.cursor() as cur:
               # Ensure tenant schema exists
               cur.execute("CREATE SCHEMA IF NOT EXISTS %s", (self._tenant,))
               # Set search path to tenant schema
               cur.execute("SET search_path TO %s, public", (self._tenant,))
           self._conn.commit()
       return self._conn

   def commit(self) -> None:
       if self._conn and not self._conn.closed:
           self._conn.commit()

   def close(self) -> None:
       if self._conn and not self._conn.closed:
           self._conn.close()
           self._conn = None
   ```

   **Important**: Use `psycopg2.sql` module for safe schema name interpolation (schema names cannot be parameterized with `%s`):
   ```python
   from psycopg2 import sql

   cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
       sql.Identifier(self._tenant)
   ))
   cur.execute(sql.SQL("SET search_path TO {}, public").format(
       sql.Identifier(self._tenant)
   ))
   ```

**Edge cases**:
- Tenant schema doesn't exist: `CREATE SCHEMA IF NOT EXISTS` handles this.
- Database user lacks `CREATE SCHEMA` permission: The `ImportError`-style error message should guide the user.
- Connection drops: Detect `self._conn.closed` and reconnect.

---

### Subtask T019 -- Implement `PostgresStore` -- read operations

- **Purpose**: Implement all read methods that query `events`, `tasks`, `ml_cursor`, and `ml_predictions`.
- **Files**: `src/sigil_ml/store_postgres.py`
- **Parallel?**: Yes -- can proceed in parallel with T020 after T018 is done.

**Steps**:
1. Implement `get_cursor()`:
   ```python
   def get_cursor(self) -> int:
       conn = self._get_conn()
       with conn.cursor() as cur:
           cur.execute("SELECT last_event_id FROM ml_cursor WHERE id = 1")
           row = cur.fetchone()
           return row[0] if row else 0
   ```

2. Implement `get_events_since(since_id, limit=100)`:
   ```python
   def get_events_since(self, since_id: int, limit: int = 100) -> list[dict[str, Any]]:
       conn = self._get_conn()
       with conn.cursor() as cur:
           cur.execute(
               "SELECT id, kind, source, payload, ts FROM events "
               "WHERE id > %s ORDER BY id ASC LIMIT %s",
               (since_id, limit),
           )
           columns = ["id", "kind", "source", "payload", "ts"]
           return [dict(zip(columns, row)) for row in cur.fetchall()]
   ```

   Note: Use `%s` placeholders (Postgres), not `?` (SQLite).

3. Implement `get_active_task()`:
   ```python
   def get_active_task(self) -> str | None:
       conn = self._get_conn()
       with conn.cursor() as cur:
           cur.execute(
               "SELECT id FROM tasks WHERE phase != 'idle' "
               "AND completed_at IS NULL ORDER BY last_active DESC LIMIT 1"
           )
           row = cur.fetchone()
           return row[0] if row else None
   ```

4. Implement `get_task_by_id(task_id)`:
   ```python
   def get_task_by_id(self, task_id: str) -> dict[str, Any] | None:
       conn = self._get_conn()
       with conn.cursor() as cur:
           cur.execute("SELECT * FROM tasks WHERE id = %s", (task_id,))
           if cur.description is None:
               return None
           columns = [desc[0] for desc in cur.description]
           row = cur.fetchone()
           if row is None:
               return None
           return dict(zip(columns, row))
   ```

5. Implement `get_events_for_task(task_id, since=None)`:
   - Same logic as SqliteStore: look up task, determine time window, query events.
   - Parse JSON payload for each event.

6. Implement `get_session_info(task_id)`:
   ```python
   def get_session_info(self, task_id: str) -> dict[str, Any] | None:
       conn = self._get_conn()
       with conn.cursor() as cur:
           cur.execute("SELECT started_at, phase, test_fails FROM tasks WHERE id = %s", (task_id,))
           row = cur.fetchone()
           if row is None:
               return None
           return {"started_at": row[0], "phase": row[1], "test_fails": row[2]}
   ```

7. Implement `get_quality_task_stats()`, `get_completed_task_ids()`, `get_completed_tasks_with_timestamps()`, `count_completed_tasks()`, `get_status_data()` -- following the same patterns as SqliteStore but with `%s` placeholders.

**Key difference from SQLite**: Postgres `cursor.description` provides column names, so you can build dicts dynamically without `sqlite3.Row`.

---

### Subtask T020 -- Implement `PostgresStore` -- write operations

- **Purpose**: Implement all write methods for `ml_predictions`, `ml_events`, and `ml_cursor`.
- **Files**: `src/sigil_ml/store_postgres.py`
- **Parallel?**: Yes -- can proceed in parallel with T019 after T018 is done.

**Steps**:
1. Implement `update_cursor(event_id)`:
   ```python
   def update_cursor(self, event_id: int) -> None:
       conn = self._get_conn()
       with conn.cursor() as cur:
           cur.execute(
               "UPDATE ml_cursor SET last_event_id = %s, updated_at = %s WHERE id = 1",
               (event_id, int(time.time() * 1000)),
           )
   ```

2. Implement `insert_prediction(model, result, confidence, ttl_sec)`:
   ```python
   def insert_prediction(
       self, model: str, result: dict, confidence: float, ttl_sec: int | None
   ) -> None:
       conn = self._get_conn()
       now_ms = int(time.time() * 1000)
       expires_ms = (now_ms + ttl_sec * 1000) if ttl_sec else None
       with conn.cursor() as cur:
           cur.execute(
               "INSERT INTO ml_predictions (model, result, confidence, created_at, expires_at) "
               "VALUES (%s, %s, %s, %s, %s)",
               (model, json.dumps(result), round(confidence, 4), now_ms, expires_ms),
           )
   ```

3. Implement `insert_ml_event(kind, endpoint, routing, latency_ms)`:
   ```python
   def insert_ml_event(
       self, kind: str, endpoint: str, routing: str, latency_ms: int
   ) -> None:
       conn = self._get_conn()
       with conn.cursor() as cur:
           cur.execute(
               "INSERT INTO ml_events (kind, endpoint, routing, latency_ms, ts) "
               "VALUES (%s, %s, %s, %s, %s)",
               (kind, endpoint, routing, latency_ms, int(time.time() * 1000)),
           )
   ```

**Table ownership validation**:
- [ ] `insert_prediction` writes to `ml_predictions` only (Python-owned).
- [ ] `insert_ml_event` writes to `ml_events` only (Python-owned).
- [ ] `update_cursor` writes to `ml_cursor` only (Python-owned).
- [ ] No write operations target `events`, `tasks`, `patterns`, or `suggestions`.

**Validation**:
- [ ] Result is JSON-serialized identically to SqliteStore (`json.dumps(result)`).
- [ ] Confidence is rounded to 4 decimal places.
- [ ] Timestamps are in milliseconds (same convention as SQLite).

---

### Subtask T021 -- Implement `PostgresStore` -- schema bootstrap

- **Purpose**: Create Python-owned tables in the tenant schema if they don't exist.
- **Files**: `src/sigil_ml/store_postgres.py`
- **Parallel?**: No -- should be done after T018 establishes connection management.

**Steps**:
1. Implement `ensure_tables()`:
   ```python
   def ensure_tables(self) -> None:
       """Create Python-owned tables in tenant schema if they don't exist."""
       conn = self._get_conn()
       with conn.cursor() as cur:
           cur.execute("""
               CREATE TABLE IF NOT EXISTS ml_cursor (
                   id            INTEGER PRIMARY KEY CHECK (id = 1),
                   last_event_id BIGINT NOT NULL DEFAULT 0,
                   updated_at    BIGINT NOT NULL DEFAULT 0
               )
           """)
           cur.execute("""
               INSERT INTO ml_cursor (id, last_event_id, updated_at)
               VALUES (1, 0, 0)
               ON CONFLICT (id) DO NOTHING
           """)
       conn.commit()
       logger.info("postgres: ml_cursor table ensured in schema %s", self._tenant)
   ```

**Postgres-specific DDL notes**:
- Use `BIGINT` instead of `INTEGER` for timestamp fields (Postgres `INTEGER` is 32-bit).
- Use `ON CONFLICT (id) DO NOTHING` instead of `INSERT OR IGNORE` (SQLite syntax).
- `INTEGER PRIMARY KEY` in Postgres does NOT auto-increment (unlike SQLite). For ml_cursor, the PK is always 1, so this is fine.
- The `events`, `tasks`, `patterns`, `suggestions` tables and `ml_predictions`, `ml_events` tables are created by Go. Python only creates `ml_cursor`.

**Edge cases**:
- Tenant schema doesn't exist yet: Handled in T018's connection setup (`CREATE SCHEMA IF NOT EXISTS`).
- Tables already exist: `IF NOT EXISTS` is idempotent.
- `ml_predictions` and `ml_events` must already exist (created by Go). If they don't, the write operations will fail with a clear Postgres error.

---

### Subtask T022 -- Update `create_store()` factory and add Postgres config

- **Purpose**: Wire PostgresStore into the factory function and add the `SIGIL_POSTGRES_URL` environment variable to config.
- **Files**: `src/sigil_ml/store.py`, `src/sigil_ml/config.py`
- **Parallel?**: No -- should be done last in this WP, after T018-T021 implement PostgresStore.

**Steps**:
1. Update `config.py` to add Postgres URL support:
   ```python
   def postgres_url() -> str | None:
       """Return the Postgres connection URL, or None if not configured."""
       return os.environ.get("SIGIL_POSTGRES_URL")

   def tenant_id() -> str:
       """Return the tenant identifier for multi-tenant Postgres schemas."""
       return os.environ.get("SIGIL_TENANT", "public")
   ```

2. Update `create_store()` in `src/sigil_ml/store.py`:

   **Before** (from WP01):
   ```python
   if resolved_mode == "cloud":
       raise NotImplementedError("Postgres backend not yet implemented")
   ```

   **After**:
   ```python
   if resolved_mode == "cloud":
       from sigil_ml.store_postgres import PostgresStore

       url = config.postgres_url()
       if not url:
           raise ValueError(
               "SIGIL_POSTGRES_URL environment variable is required in cloud mode"
           )
       tenant = config.tenant_id()
       return PostgresStore(connection_url=url, tenant=tenant)
   ```

   Note: `PostgresStore` is imported lazily (inside the if block) to avoid requiring `psycopg2` for local-only installations.

3. Verify the factory works for both modes:
   ```python
   # Local mode (default):
   store = create_store()  # Returns SqliteStore

   # Cloud mode:
   os.environ["SIGIL_MODE"] = "cloud"
   os.environ["SIGIL_POSTGRES_URL"] = "postgresql://user:pass@localhost:5432/sigil"
   store = create_store()  # Returns PostgresStore
   ```

---

## Risks & Mitigations

- **Risk**: Postgres SQL dialect differences (parameterized query placeholders `%s` vs `?`, `INSERT OR IGNORE` vs `ON CONFLICT`). **Mitigation**: Each method is written with Postgres-specific syntax. The subtask guidance above details every difference.
- **Risk**: Tenant schema creation requires `CREATE SCHEMA` privilege. **Mitigation**: Provide a clear error message if the privilege is missing. Document required Postgres permissions.
- **Risk**: `psycopg2-binary` adds ~3MB to the installation. **Mitigation**: It's an optional dependency -- local-only users never install it.
- **Risk**: Connection pooling is not implemented in v1. High-concurrency cloud deployments may exhaust connections. **Mitigation**: Acceptable for initial implementation. Add connection pooling (e.g., `psycopg2.pool.ThreadedConnectionPool`) in a future iteration.
- **Risk**: Schema name injection via tenant identifier. **Mitigation**: Use `psycopg2.sql.Identifier()` for all schema name interpolation.

## Review Guidance

- Verify every method in `PostgresStore` mirrors the corresponding `SqliteStore` method behavior.
- Verify all SQL uses `%s` placeholders (Postgres), not `?` (SQLite).
- Verify `CREATE TABLE` DDL uses Postgres-compatible types (`BIGINT` for timestamps).
- Verify `INSERT OR IGNORE` is replaced with `ON CONFLICT DO NOTHING`.
- Verify tenant schema isolation uses `psycopg2.sql.Identifier` (not string formatting).
- Verify `psycopg2` is imported lazily (only when PostgresStore is instantiated).
- Verify table ownership: only `ml_cursor` is created by `ensure_tables()`.

## Activity Log

- 2026-03-30T01:45:06Z -- system -- lane=planned -- Prompt generated via /spec-kitty.tasks
