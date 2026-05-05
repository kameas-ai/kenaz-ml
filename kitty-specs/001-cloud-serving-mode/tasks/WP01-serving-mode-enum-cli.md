---
work_package_id: WP01
title: ServingMode Enum, CLI Flag, and Conditional Startup
lane: planned
dependencies: []
subtasks:
- T001
- T002
- T003
- T004
- T005
- T006
phase: Phase 1 - Foundation
assignee: ''
agent: ''
shell_pid: ''
review_status: ''
reviewed_by: ''
history:
- timestamp: '2026-03-30T01:45:14Z'
  lane: planned
  agent: system
  shell_pid: ''
  action: Prompt regenerated via /spec-kitty.tasks
requirement_refs:
- FR-001
- FR-002
- FR-006
- FR-008
---

# Work Package Prompt: WP01 -- ServingMode Enum, CLI Flag, and Conditional Startup

## Review Feedback Status

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

- Introduce a `ServingMode` str enum with `local` and `cloud` values in `config.py`.
- Add a `--mode` flag to the `serve` CLI subcommand with priority: CLI flag > `SIGIL_ML_MODE` env var > default `local`.
- Refactor `create_app()` to accept a `mode` parameter and store it on `AppState`.
- In cloud mode, skip SQLite schema bootstrap (`ensure_ml_tables`), `EventPoller`, and `TrainingScheduler` at startup.
- In local mode, behavior is 100% identical to current implementation (zero regression).
- Module-level `app = create_app()` continues to work for `uvicorn sigil_ml.app:app`.
- Add empty `[cloud]` optional extras group to `pyproject.toml` for future dependencies.

**Measurable**:
- `kameas-ml serve --mode cloud` starts without opening any SQLite connection.
- `kameas-ml serve` (no flag) starts poller and scheduler as before.
- `SIGIL_ML_MODE=cloud kameas-ml serve` behaves the same as `--mode cloud`.
- Existing test suite (`pytest tests/test_server.py`) passes without modification.

## Context & Constraints

- **Spec**: `kitty-specs/001-cloud-serving-mode/spec.md` -- FR-001, FR-002, FR-006, FR-008
- **Plan**: `kitty-specs/001-cloud-serving-mode/plan.md` -- Design Decisions D1 (ServingMode enum), D2 (mode resolution order), D7 (create_app factory changes), D8 (pyproject.toml changes)
- **Data Model**: `kitty-specs/001-cloud-serving-mode/data-model.md` -- ServingMode entity
- **Research**: `kitty-specs/001-cloud-serving-mode/research.md` -- R1 (branching strategy), R5 (conditional startup)

**Current architecture** (files you will modify):
- `src/sigil_ml/config.py` (26 lines): Contains `db_path()`, `models_dir()`, `weights_path()`, `sigild_plugin_url()`. No enum exists yet.
- `src/sigil_ml/cli.py` (63 lines): `serve` subcommand has `--host` and `--port` args. No `--mode` arg.
- `src/sigil_ml/app.py` (106 lines): `create_app()` takes no parameters. `AppState` has no `mode` attribute. Startup unconditionally calls `ensure_ml_tables()`, creates `EventPoller`, creates `TrainingScheduler`.
- `pyproject.toml`: Has `[project.optional-dependencies]` with only `dev` group.

**Backward compatibility constraint**: `app = create_app()` at module level (line 105 of `app.py`) is used by `uvicorn sigil_ml.app:app`. It MUST default to local mode.

**Implementation command**: `spec-kitty implement WP01`

## Subtasks & Detailed Guidance

### Subtask T001 -- Create `ServingMode` str enum in `src/sigil_ml/config.py`

- **Purpose**: Provide a type-safe, JSON-serializable enum that all downstream code uses to branch behavior.
- **Steps**:
  1. Add `import enum` at the top of `src/sigil_ml/config.py` (after the existing `import os`).
  2. Define the enum class near the top of the module, before the path helper functions:
     ```python
     class ServingMode(str, enum.Enum):
         """Operating mode for the kameas-ml service.

         LOCAL: Default. Poller, SQLite, local models. Current behavior.
         CLOUD: Stateless. No poller, no SQLite, tenant-aware model loading.
         """
         LOCAL = "local"
         CLOUD = "cloud"
     ```
  3. The `str` mixin ensures `ServingMode.CLOUD == "cloud"` evaluates to `True` and `json.dumps(ServingMode.CLOUD)` produces `"cloud"`.
- **Files**: `src/sigil_ml/config.py` (modify -- add ~10 lines)
- **Parallel?**: No -- foundational for all other subtasks.
- **Notes**: Keep placement near the top, before `_data_home()`. The existing functions (`db_path`, `models_dir`, etc.) remain unchanged.

### Subtask T002 -- Add mode resolution helper in `src/sigil_ml/config.py`

- **Purpose**: Centralize the priority chain: CLI flag > `SIGIL_ML_MODE` env var > default `local`. This single function is the only place mode resolution logic lives.
- **Steps**:
  1. Add the following function after the `ServingMode` enum definition:
     ```python
     def resolve_mode(cli_mode: str | None = None) -> ServingMode:
         """Resolve the serving mode from CLI flag or environment.

         Priority:
           1. cli_mode argument (from --mode flag)
           2. SIGIL_ML_MODE environment variable
           3. Default: LOCAL

         Raises:
             SystemExit: If the provided mode value is invalid.
         """
         raw = cli_mode or os.environ.get("SIGIL_ML_MODE", "local")
         try:
             return ServingMode(raw.lower())
         except ValueError:
             raise SystemExit(
                 f"Invalid serving mode: {raw!r}. Must be 'local' or 'cloud'."
             )
     ```
  2. Handle the edge case where `cli_mode` is an empty string by treating it as `None`.
- **Files**: `src/sigil_ml/config.py` (modify -- add ~15 lines)
- **Parallel?**: No -- depends on T001, used by T003.
- **Notes**: The CLI will call `resolve_mode(args.mode)`. The `create_app()` function will call `resolve_mode()` with no argument to pick up env var.

### Subtask T003 -- Add `--mode` argument to `serve` subcommand in `src/sigil_ml/cli.py`

- **Purpose**: Allow operators to select cloud mode from the command line.
- **Steps**:
  1. Add the argument to `serve_parser` (after the existing `--port` argument, line 19):
     ```python
     serve_parser.add_argument(
         "--mode",
         choices=["local", "cloud"],
         default=None,
         help="Serving mode: 'local' (default, with poller) or 'cloud' (stateless, no SQLite)",
     )
     ```
  2. In the `if args.command == "serve":` block (line 28), resolve the mode and set an env var bridge:
     ```python
     import os
     from sigil_ml.config import resolve_mode

     mode = resolve_mode(args.mode)
     # Bridge mode to create_app() via env var (uvicorn string import cannot pass args)
     os.environ["SIGIL_ML_MODE"] = mode.value
     uvicorn.run(
         "sigil_ml.app:app",
         host=args.host,
         port=args.port,
         log_level="info",
     )
     ```
  3. Add `import os` to the top of `cli.py`.
- **Files**: `src/sigil_ml/cli.py` (modify -- add ~10 lines)
- **Parallel?**: No -- depends on T001 and T002.
- **Notes**: The env var bridge is needed because `uvicorn.run("sigil_ml.app:app")` imports the module and evaluates `app = create_app()`. The `create_app()` function reads `SIGIL_ML_MODE` from the environment via `resolve_mode()`. An alternative is `uvicorn.run("sigil_ml.app:create_app", factory=True)` but that changes the import path convention.

### Subtask T004 -- Refactor `create_app()` to accept `mode` parameter, store on `AppState`

- **Purpose**: Thread the serving mode into the application so routes and startup can branch on it.
- **Steps**:
  1. Add `mode` attribute to `AppState.__init__()`:
     ```python
     from sigil_ml.config import ServingMode, resolve_mode

     class AppState:
         """Holds model instances and runtime state, passed to routes."""

         def __init__(self, mode: ServingMode = ServingMode.LOCAL) -> None:
             self.mode = mode
             self.stuck: StuckPredictor | None = None
             self.activity: ActivityClassifier | None = None
             self.workflow: WorkflowStatePredictor | None = None
             self.duration: DurationEstimator | None = None
             self.quality: QualityEstimator | None = None
             self.poller: EventPoller | None = None
             self.training_in_progress: bool = False
             # Cloud-mode fields (initialized by WP05, None for now)
             self.model_cache = None
             self.model_loader = None
     ```
  2. Update `create_app()` to accept and resolve mode:
     ```python
     def create_app(mode: ServingMode | None = None) -> FastAPI:
         """Create and configure the FastAPI application."""
         if mode is None:
             mode = resolve_mode()  # reads SIGIL_ML_MODE env var, defaults to LOCAL
         application = FastAPI(title="kameas-ml", version="0.1.0")
         state = AppState(mode=mode)

         register_routes(application, state)
         # ... startup/shutdown events ...
     ```
  3. The module-level `app = create_app()` (line 105) now calls `resolve_mode()` which defaults to LOCAL unless `SIGIL_ML_MODE` is set. No change to this line.
- **Files**: `src/sigil_ml/app.py` (modify -- change ~10 lines)
- **Parallel?**: No -- T005 depends on this.
- **Notes**: Keep `load_models()` and `reload_models_into_poller()` methods on `AppState` unchanged. They are only called in local mode (guarded by T005). The `model_cache` and `model_loader` fields are `None` placeholders for WP05.

### Subtask T005 -- Conditional startup: skip SQLite, EventPoller, and TrainingScheduler in cloud mode

- **Purpose**: In cloud mode, the startup event must NOT touch SQLite, create a poller, or schedule training. This is the core safety gate (FR-002, FR-008).
- **Steps**:
  1. Modify the `startup_event()` closure inside `create_app()` to branch on `state.mode`:
     ```python
     @application.on_event("startup")
     async def startup_event() -> None:
         if state.mode == ServingMode.LOCAL:
             db = config.db_path()

             try:
                 ensure_ml_tables(db)
             except Exception:
                 logger.warning(
                     "schema bootstrap failed (sigild may not have started yet)",
                     exc_info=True,
                 )

             state.load_models()

             state.poller = EventPoller(
                 db_path=db,
                 models={
                     "stuck": state.stuck,
                     "activity": state.activity,
                     "workflow": state.workflow,
                     "duration": state.duration,
                     "quality": state.quality,
                 },
             )
             asyncio.create_task(state.poller.run())

             scheduler = TrainingScheduler(
                 db, reload_callback=state.reload_models_into_poller
             )

             async def _schedule_loop():
                 while True:
                     await asyncio.get_event_loop().run_in_executor(
                         None, scheduler.check_and_retrain
                     )
                     await asyncio.sleep(600)

             asyncio.create_task(_schedule_loop())

             logger.info(
                 "kameas-ml: local mode -- models loaded, poller started, scheduler active"
             )
         else:
             # Cloud mode: no SQLite, no poller, no scheduler.
             # Models loaded lazily per-tenant (WP05 will add cache/loader init here).
             logger.info("kameas-ml: cloud mode -- stateless serving, no poller")
     ```
  2. The `shutdown_event()` already guards with `if state.poller:` so it handles cloud mode (poller is `None`) correctly. No change needed.
  3. Verify that NO call to `config.db_path()`, `ensure_ml_tables()`, `EventPoller()`, `TrainingScheduler()`, or `state.load_models()` happens in the cloud startup path.
- **Files**: `src/sigil_ml/app.py` (modify -- restructure startup_event)
- **Parallel?**: Yes -- can proceed once T004 is merged.
- **Notes**: After this subtask, cloud mode starts but all predict endpoints return fallback responses (models are `None`). This is correct and expected -- WP02 adds endpoint guards, WP05 adds tenant-aware model loading. The critical invariant is: **no SQLite file is touched in cloud mode**.

### Subtask T006 -- Add `[cloud]` optional extras group to `pyproject.toml`

- **Purpose**: Reserve an installation extra for future cloud-specific dependencies (boto3 for Feature 003, etc.).
- **Steps**:
  1. Add to `pyproject.toml` under `[project.optional-dependencies]` (after the existing `dev` group):
     ```toml
     [project.optional-dependencies]
     dev = ["pytest>=8.0", "httpx>=0.27", "ruff>=0.4", "pyre-check>=0.9.18"]
     cloud = []  # Reserved for future: boto3, asyncpg, etc.
     ```
  2. No actual dependencies added. This enables `pip install kameas-ml[cloud]` in the future.
- **Files**: `pyproject.toml` (modify -- add 1 line)
- **Parallel?**: Yes -- completely independent of all other subtasks.

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking `uvicorn sigil_ml.app:app` import | Low | High | `create_app()` defaults to LOCAL when no arg/env var. Module-level `app = create_app()` unchanged. |
| `SIGIL_ML_MODE` env var set unexpectedly | Low | Medium | `resolve_mode()` validates the value and raises `SystemExit` for invalid inputs. |
| Cloud mode starts with no models | Expected | None | Endpoints return fallback responses. WP05 adds model loading. |
| Import-time side effects | Low | Medium | `config.db_path()` is only called inside `startup_event()`, not at import time. |

## Review Guidance

- Verify `create_app()` with no arguments defaults to LOCAL mode.
- Verify `app = create_app()` at module level still works (the uvicorn import path).
- Verify in cloud mode: NO `sqlite3` operations, NO `db_path()` calls, NO background tasks created.
- Run `pytest tests/test_server.py` -- all existing tests must pass unchanged.
- Check `ServingMode` serialization: `ServingMode.CLOUD.value == "cloud"` and `str(ServingMode.CLOUD)` produces a usable string.
- Check mode resolution priority: `resolve_mode("cloud")` > `SIGIL_ML_MODE=cloud` > default `local`.

## Activity Log

- 2026-03-30T01:45:14Z -- system -- lane=planned -- Prompt regenerated.
