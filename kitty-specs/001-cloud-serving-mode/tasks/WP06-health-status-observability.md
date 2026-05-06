---
work_package_id: WP06
title: Cloud-Aware Health, Status, and Observability
lane: planned
dependencies: [WP05]
subtasks:
- T030
- T031
- T032
- T033
- T034
phase: Phase 4 - Polish
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
- FR-007
---

# Work Package Prompt: WP06 -- Cloud-Aware Health, Status, and Observability

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

- `/health` endpoint includes an optional `mode` field (`"local"` or `"cloud"`) in its response.
- In cloud mode, `/health` reports model availability based on cache state (not SQLite/poller).
- In cloud mode, `/status` lists loaded tenants, model names, cache statistics, and per-tenant request counts.
- Per-tenant request counters are tracked in-memory and exposed via `/status`.
- Startup logs clearly indicate the operating mode.
- In local mode, both endpoints remain exactly as they are today -- zero regression.

**Measurable**:
- `GET /health` in cloud mode returns `{"status": "ok", "mode": "cloud", "models": {...}, "uptime_sec": ...}`.
- `GET /status` in cloud mode returns cache stats, loaded tenants, and request counts.
- `GET /health` in local mode returns the existing response (now with `mode: "local"` added).
- No `X-Tenant-ID` header required for either endpoint.

## Context & Constraints

- **Spec**: FR-007 (health endpoint reflects operating mode and model availability)
- **Plan**: Design Decisions D6 (endpoint guards), D7 (create_app changes)
- **Research**: R7 (health endpoint extension -- optional `mode` field)
- **Dependencies**: WP01 (mode awareness), WP04 (cache stats via `ModelCache.stats()` and `loaded_tenants()`), WP05 (model loading wired up)

**Current code** (`src/sigil_ml/routes.py`):
- `HealthResponse` (line 83): Has `status`, `models`, `uptime_sec`. No `mode` field.
- `health()` handler (line 98): Iterates over `state.stuck`, `state.activity`, etc. Reports `is_trained` status.
- `status()` handler (line 120): Opens SQLite connection, queries `ml_cursor` and `ml_predictions`.

**Backward compatibility**: Adding `mode` to `HealthResponse` must not break existing consumers. Use `mode: str = "local"` (default value).

**Implementation command**: `spec-kitty implement WP06 --base WP05`

## Subtasks & Detailed Guidance

### Subtask T030 -- Add optional `mode` field to `HealthResponse` schema

- **Purpose**: Make the operating mode visible in health check responses for monitoring systems and K8s probes.
- **Steps**:
  1. Update `HealthResponse` in `src/sigil_ml/routes.py`:
     ```python
     class HealthResponse(BaseModel):
         status: str
         mode: str = "local"  # Default for backward compatibility
         models: dict[str, str]
         uptime_sec: float
     ```
  2. The default `"local"` ensures existing consumers parsing the response without expecting `mode` continue to work.
  3. Both cloud and local health handlers will explicitly set this field.
- **Files**: `src/sigil_ml/routes.py` (modify -- add 1 field)
- **Parallel?**: No -- T031 depends on this schema change.
- **Notes**: Using a plain `str` field (not the `ServingMode` enum) avoids coupling the API schema to internal types.

### Subtask T031 -- Update `/health` handler to report cloud-specific state

- **Purpose**: In cloud mode, the health response should reflect cache-based model availability rather than local model instance status.
- **Steps**:
  1. Update the `health()` handler to branch on mode:
     ```python
     @fastapi_app.get("/health", response_model=HealthResponse)
     async def health() -> HealthResponse:
         if state.mode == ServingMode.CLOUD:
             models_status: dict[str, str] = {}
             if state.model_cache:
                 tenants = state.model_cache.loaded_tenants()
                 all_cached_models: set[str] = set()
                 for model_list in tenants.values():
                     all_cached_models.update(model_list)
                 for name in [
                     "stuck", "activity", "workflow", "duration", "quality"
                 ]:
                     models_status[name] = (
                         "cached" if name in all_cached_models else "on_demand"
                     )
             else:
                 for name in [
                     "stuck", "activity", "workflow", "duration", "quality"
                 ]:
                     models_status[name] = "not_initialized"

             return HealthResponse(
                 status="ok",
                 mode="cloud",
                 models=models_status,
                 uptime_sec=round(time.time() - _start_time, 1),
             )

         # Local mode: existing behavior with mode field added
         models_status = {}
         for name, model in [
             ("stuck", state.stuck),
             ("activity", state.activity),
             ("workflow", state.workflow),
             ("duration", state.duration),
         ]:
             if model is not None:
                 models_status[name] = (
                     "ready" if model.is_trained else "untrained"
                 )
             else:
                 models_status[name] = "not_loaded"

         models_status["quality"] = (
             "ready" if state.quality is not None else "not_loaded"
         )

         return HealthResponse(
             status="ok",
             mode="local",
             models=models_status,
             uptime_sec=round(time.time() - _start_time, 1),
         )
     ```
  2. Cloud mode model statuses:
     - `"cached"`: Model is in the cache for at least one tenant.
     - `"on_demand"`: Model will be loaded on first request (cold cache).
     - `"not_initialized"`: Cache not yet initialized (should not happen post-startup).
  3. Local mode now explicitly sets `mode="local"` -- a minor addition to the existing response.
- **Files**: `src/sigil_ml/routes.py` (modify)
- **Parallel?**: No -- modifies the health handler.

### Subtask T032 -- Rewrite `/status` handler for cloud mode

- **Purpose**: Replace the WP02 stub with real operational data from the cache and loader.
- **Steps**:
  1. Update the cloud-mode branch of `status()` (replacing WP02's minimal stub):
     ```python
     @fastapi_app.get("/status")
     async def status() -> dict:
         if state.mode == ServingMode.CLOUD:
             cache_stats = (
                 state.model_cache.stats() if state.model_cache else {}
             )
             loaded = (
                 state.model_cache.loaded_tenants()
                 if state.model_cache
                 else {}
             )
             return {
                 "mode": "cloud",
                 "cache": cache_stats,
                 "loaded_tenants": loaded,
                 "request_counts": dict(getattr(state, "request_counters", {})),
                 "poller_running": False,
             }

         # Local mode: existing SQLite-based status (unchanged)
         db = config.db_path()
         try:
             conn = sqlite3.connect(str(db), timeout=5.0)
             # ... rest of existing code unchanged ...
     ```
  2. The response now includes:
     - `cache`: Stats from `ModelCache.stats()` -- entries, hits, misses, evictions, hit_rate.
     - `loaded_tenants`: Dict of `{tenant_id: [model_names]}` from `ModelCache.loaded_tenants()`.
     - `request_counts`: Per-tenant request counts from T033.
  3. Local mode status handler remains completely unchanged.
- **Files**: `src/sigil_ml/routes.py` (modify)
- **Parallel?**: No -- builds on WP02 T012 stub.

### Subtask T033 -- Add per-tenant request counters

- **Purpose**: Enable operators to see which tenants are active and their request volumes.
- **Steps**:
  1. Add a request counter dict to `AppState` in `src/sigil_ml/app.py`:
     ```python
     class AppState:
         def __init__(self, mode: ServingMode = ServingMode.LOCAL) -> None:
             # ... existing fields ...
             # Per-tenant request counters (cloud mode, reset on restart)
             self.request_counters: dict[str, int] = {}
     ```
  2. Add a helper method:
     ```python
     def count_request(self, tenant_id: str) -> None:
         """Increment the request counter for a tenant."""
         self.request_counters[tenant_id] = (
             self.request_counters.get(tenant_id, 0) + 1
         )
     ```
  3. Call `state.count_request(tenant.tenant_id)` at the top of each `/predict/*` handler in cloud mode:
     ```python
     if state.mode == ServingMode.CLOUD:
         state.count_request(tenant.tenant_id)
     ```
  4. Expose in `/status` response (done in T032).
- **Files**: `src/sigil_ml/app.py` (modify -- add ~10 lines), `src/sigil_ml/routes.py` (add counter call to each handler)
- **Parallel?**: Yes -- independent of health/status handler changes.
- **Notes**: Using a plain `dict[str, int]` is not strictly thread-safe for concurrent reads+writes, but Python's GIL makes single-operation increments safe in practice. For production, consider `threading.Lock` or `collections.Counter`. For this feature, a plain dict is acceptable.

### Subtask T034 -- Add `mode` field to startup log message and structured output

- **Purpose**: Make the operating mode immediately visible in logs for debugging.
- **Steps**:
  1. The startup log lines were already updated in WP01 T005. Verify they clearly state the mode:
     ```python
     logger.info("kenaz-ml: local mode -- models loaded, poller started, scheduler active")
     # and
     logger.info("kenaz-ml: cloud mode -- cache and loader initialized")
     ```
  2. Optionally, update the FastAPI app description to include mode:
     ```python
     application = FastAPI(
         title="kenaz-ml",
         version="0.1.0",
         description=f"Sigil ML sidecar ({mode.value} mode)",
     )
     ```
  3. Optionally, add a root endpoint for quick identification:
     ```python
     @fastapi_app.get("/")
     async def root() -> dict:
         return {
             "service": "kenaz-ml",
             "mode": state.mode.value,
             "version": "0.1.0",
         }
     ```
- **Files**: `src/sigil_ml/app.py` (verify/modify), `src/sigil_ml/routes.py` (optional root endpoint)
- **Parallel?**: Yes -- independent polish.
- **Notes**: The root endpoint is optional but useful for operators who `curl` the service.

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| HealthResponse schema change breaks consumers | Low | Medium | `mode` field has default value `"local"`. Existing parsers ignoring unknown fields are unaffected. |
| Request counter memory growth | Very Low | Low | Tenants are bounded in practice. Counter resets on restart. |
| `/status` information leak | Medium | Low | Reveals tenant IDs and request counts. Ensure endpoint is not publicly accessible (K8s internal service). |

## Review Guidance

- Verify `GET /health` returns `mode: "cloud"` in cloud mode and `mode: "local"` in local mode.
- Verify `GET /health` does NOT reference SQLite or poller in cloud mode.
- Verify `GET /status` returns cache stats (`hits`, `misses`, `evictions`, `hit_rate`) and loaded tenants in cloud mode.
- Verify `GET /status` returns the existing SQLite-based response in local mode (no regression).
- Verify `/health` and `/status` work WITHOUT `X-Tenant-ID` header in both modes.
- Verify request counters increment per-tenant when predictions are served in cloud mode.
- Run all existing tests -- zero regression.

## Activity Log

- 2026-03-30T01:45:14Z -- system -- lane=planned -- Prompt regenerated.
