---
work_package_id: "WP05"
title: "App Wiring, Config & Cleanup"
lane: "planned"
dependencies: ["WP01", "WP02", "WP03", "WP04"]
subtasks:
  - "T023"
  - "T024"
  - "T025"
  - "T026"
phase: "Phase 4 - Integration & Polish"
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
  - "FR-008"
  - "FR-009"
  - "FR-010"
---

# Work Package Prompt: WP05 -- App Wiring, Config & Cleanup

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

- Complete end-to-end integration: app startup selects the correct backend based on operating mode.
- Configuration supports both SQLite (local) and Postgres (cloud) backends.
- Retire `schema.py` since `DataStore.ensure_tables()` replaces it.
- Achieve SC-002: No module outside `store_sqlite.py` imports `sqlite3`.
- Achieve SC-004: Switching from SQLite to Postgres requires only a configuration change, not a code change.

**Success criteria**:
- FR-008: The app startup sequence selects the appropriate DataStore backend based on operating mode.
- FR-009: Postgres backend accepts a connection URL via `SIGIL_POSTGRES_URL`.
- SC-002: `grep -r "import sqlite3" src/sigil_ml/` returns only `store_sqlite.py`.
- SC-004: Start with `SIGIL_MODE=cloud SIGIL_POSTGRES_URL=...` and Postgres is used.

**Implementation command**: `spec-kitty implement WP05 --base WP04`

## Context & Constraints

- **Prerequisites**: WP01-WP04 must all be complete.
- **Spec**: `kitty-specs/002-storage-abstraction/spec.md` -- all user stories converge here.
- **Feature 001 dependency**: Cloud serving mode (feature 001) may or may not be merged. Use `SIGIL_MODE` environment variable as the primary mechanism, with CLI flag integration as a follow-up.
- **Current state after WP01-WP04**: The DataStore protocol exists, SqliteStore and PostgresStore are implemented, all modules use DataStore. What remains is final wiring, config cleanup, and schema.py retirement.

---

## Subtasks & Detailed Guidance

### Subtask T023 -- Update `config.py` to expose `operating_mode()` and `postgres_url()`

- **Purpose**: Centralize operating mode detection and Postgres configuration in the config module.
- **Files**: `src/sigil_ml/config.py`
- **Parallel?**: Yes -- can proceed in parallel with T025.

**Steps**:
1. Add the following functions to `config.py`:

```python
def operating_mode() -> str:
    """Return the operating mode: 'local' or 'cloud'.

    Reads from SIGIL_MODE environment variable.
    Defaults to 'local' if not set.
    """
    mode = os.environ.get("SIGIL_MODE", "local").lower()
    if mode not in ("local", "cloud"):
        raise ValueError(f"Invalid SIGIL_MODE: {mode!r}. Must be 'local' or 'cloud'.")
    return mode


def postgres_url() -> str | None:
    """Return the Postgres connection URL, or None if not configured.

    Set via SIGIL_POSTGRES_URL environment variable.
    Required when SIGIL_MODE=cloud.
    """
    return os.environ.get("SIGIL_POSTGRES_URL")


def tenant_id() -> str:
    """Return the tenant identifier for multi-tenant Postgres schemas.

    Set via SIGIL_TENANT environment variable.
    Defaults to 'public' if not set.
    """
    return os.environ.get("SIGIL_TENANT", "public")
```

2. If `postgres_url()` and `tenant_id()` were already added in WP04 T022, verify they're consistent and avoid duplication. T022 may have added them directly to `config.py` or inline in `store.py`. Consolidate everything into `config.py`.

3. Ensure `create_store()` in `store.py` uses `config.operating_mode()` instead of reading `os.environ` directly:

   **Before** (from WP01/WP04):
   ```python
   resolved_mode = mode or os.environ.get("SIGIL_MODE", "local")
   ```

   **After**:
   ```python
   resolved_mode = mode or config.operating_mode()
   ```

**Notes**:
- The CLI `--mode` flag from feature 001 would call `os.environ["SIGIL_MODE"] = mode` before app startup, so `operating_mode()` picks it up naturally.
- Validation of mode values happens once in `operating_mode()`, not scattered across the codebase.

---

### Subtask T024 -- Finalize `app.py` wiring; update `AppState` to hold DataStore reference

- **Purpose**: Ensure `create_app()` creates a DataStore once at startup and passes it to all consumers. Make `AppState.store` the canonical data access point.
- **Files**: `src/sigil_ml/app.py`
- **Parallel?**: No -- integrates changes from WP02, WP03, WP04.

**Steps**:
1. Verify `AppState` has a `store` attribute (may already exist from WP03 T016):
   ```python
   class AppState:
       def __init__(self) -> None:
           self.store: DataStore | None = None
           self.stuck: StuckPredictor | None = None
           self.activity: ActivityClassifier | None = None
           ...
   ```

   Add `from sigil_ml.store import DataStore` import if not present.

2. Verify `create_app()` startup sequence:
   ```python
   @application.on_event("startup")
   async def startup_event() -> None:
       store = create_store()
       state.store = store

       try:
           store.ensure_tables()
       except Exception:
           logger.warning("schema bootstrap failed (sigild may not have started yet)", exc_info=True)

       state.load_models()

       state.poller = EventPoller(store=store, models={...})
       asyncio.create_task(state.poller.run())

       scheduler = TrainingScheduler(store, reload_callback=state.reload_models_into_poller)
       ...
   ```

3. Remove these imports from `app.py` if they're no longer used:
   - `from sigil_ml import config` (if `create_store()` is the only entry point)
   - `from sigil_ml.schema import ensure_ml_tables` (replaced by `store.ensure_tables()`)

4. Verify shutdown:
   ```python
   @application.on_event("shutdown")
   async def shutdown_event() -> None:
       if state.poller:
           state.poller.stop()
       if state.store:
           state.store.close()  # Clean up database connection
       logger.info("kameas-ml shutdown complete")
   ```

   Add `store.close()` to the shutdown handler to properly close database connections.

5. Verify logging shows which backend was selected:
   ```python
   logger.info("kameas-ml: using %s backend", type(store).__name__)
   ```

**Notes**:
- `AppState.store` is now the single source of truth for data access. Routes access it via `state.store`, the poller has its own reference, and the scheduler has its own reference.
- The store must be created before any other component that needs it.

---

### Subtask T025 -- Retire `schema.py`

- **Purpose**: Remove `src/sigil_ml/schema.py` since `DataStore.ensure_tables()` replaces its functionality.
- **Files**: Remove `src/sigil_ml/schema.py`; update any imports.
- **Parallel?**: Yes -- can proceed in parallel with T023.

**Steps**:
1. Find all imports of `schema`:
   ```bash
   grep -rn "from sigil_ml.schema\|from sigil_ml import schema\|import schema" src/sigil_ml/
   ```

   Expected: Only `app.py` imports `ensure_ml_tables` from `schema.py` (line 17). After WP03 T016, this import should already be removed.

2. If any imports remain, remove them and replace with `store.ensure_tables()`.

3. Delete the file:
   ```bash
   rm src/sigil_ml/schema.py
   ```

4. Verify no remaining references:
   ```bash
   grep -rn "schema" src/sigil_ml/ | grep -v "store_postgres"  # Postgres has 'schema' in different context
   ```

**Edge cases**:
- If tests import `schema.py`, update them to use `SqliteStore.ensure_tables()` instead.
- If `schema.py` is referenced in `__init__.py` or any `__all__` export, remove the reference.

---

### Subtask T026 -- Final audit: verify no module outside `store_sqlite.py` imports `sqlite3`

- **Purpose**: Validate SC-002 -- the entire codebase outside of DataStore implementations should be free of `sqlite3` imports.
- **Files**: All files in `src/sigil_ml/`
- **Parallel?**: No -- must be done last, after all other subtasks are complete.

**Steps**:
1. Run the audit:
   ```bash
   grep -rn "import sqlite3\|from sqlite3" src/sigil_ml/
   ```

   **Expected output**: Only `src/sigil_ml/store_sqlite.py` should appear.

2. Run a broader check for any lingering direct database references:
   ```bash
   grep -rn "sqlite3\." src/sigil_ml/ | grep -v store_sqlite
   ```

   Should return no results.

3. Check for `config.db_path()` usage outside of store-related code:
   ```bash
   grep -rn "config\.db_path\|db_path" src/sigil_ml/ | grep -v store | grep -v config.py
   ```

   Should return no results (all `db_path` references should be contained within `store_sqlite.py` and `config.py`).

4. Verify the feature extraction functions have the new signature:
   ```bash
   grep -n "def extract_stuck_features\|def extract_duration_features" src/sigil_ml/features.py
   ```

   Should show `(store: DataStore, task_id: str)`, not `(db_path: str | Path, task_id: str)`.

5. Verify schema.py is gone:
   ```bash
   ls src/sigil_ml/schema.py 2>/dev/null && echo "ERROR: schema.py still exists" || echo "OK: schema.py removed"
   ```

6. Run the existing test suite to confirm zero regressions:
   ```bash
   pytest tests/
   ```

**Success**: All checks pass. The storage abstraction is complete.

---

## Risks & Mitigations

- **Risk**: Feature 001 (cloud serving mode) may not be merged yet, so `--mode cloud` CLI flag may not exist. **Mitigation**: Use `SIGIL_MODE` environment variable as the primary mechanism. The `operating_mode()` function reads from env, which is compatible with both CLI-driven and direct env configuration.
- **Risk**: Removing `schema.py` could break imports in test files or other modules. **Mitigation**: Grep for all references before deleting. The steps above include a comprehensive search.
- **Risk**: `store.close()` in shutdown handler might not be called if the process is killed. **Mitigation**: SQLite connections are cleaned up by the OS on process termination. Postgres connections have a server-side idle timeout. Both are safe.

## Review Guidance

- `grep -r "import sqlite3" src/sigil_ml/` must return ONLY `store_sqlite.py`.
- `ls src/sigil_ml/schema.py` must fail (file removed).
- `AppState` must have a `store` attribute of type `DataStore`.
- Shutdown handler must call `store.close()`.
- `config.py` must have `operating_mode()`, `postgres_url()`, and `tenant_id()` functions.
- `create_store()` must use `config.operating_mode()` for mode detection.
- Starting with `SIGIL_MODE=local` (or default) must use SqliteStore.
- Starting with `SIGIL_MODE=cloud SIGIL_POSTGRES_URL=...` must use PostgresStore.
- `pytest tests/` must pass with zero regressions.

## Activity Log

- 2026-03-30T01:45:06Z -- system -- lane=planned -- Prompt generated via /spec-kitty.tasks
