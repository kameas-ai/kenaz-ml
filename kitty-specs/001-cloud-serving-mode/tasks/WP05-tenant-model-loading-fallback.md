---
work_package_id: WP05
title: Tenant-Aware Model Loading and Fallback Predictions
lane: planned
dependencies:
- WP04
subtasks:
- T024
- T025
- T026
- T027
- T028
- T029
phase: Phase 3 - Integration
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
- FR-004
- FR-005
- FR-009
- FR-010
---

# Work Package Prompt: WP05 -- Tenant-Aware Model Loading and Fallback Predictions

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

- Define a `ModelLoader` Protocol for pluggable storage backends.
- Implement `FilesystemModelLoader` as the initial concrete loader, reading from `{models_dir}/{tenant_id}/{model_name}.joblib`.
- Integrate `ModelCache` (from WP04) + `ModelLoader` into `AppState` for cloud-mode initialization.
- Add a `resolve_model()` method on `AppState` that checks cache first, then loader, then returns `None` for fallback.
- Update each `/predict/*` handler to use `resolve_model()` for tenant-specific models in cloud mode.
- Implement rule-based fallback predictions for all five model types, matching existing fallback values in `routes.py` and `poller.py`.
- Add structured logging for model load events: cache hit, cache miss + loaded, cache miss + fallback.

**Measurable**:
- Tenant A request returns prediction from tenant A's model.
- Tenant B request returns prediction from tenant B's model.
- Unknown tenant request returns rule-based fallback (not an error).
- Cache stats reflect hits and misses accurately.
- Local mode behavior is completely unchanged.

## Context & Constraints

- **Spec**: FR-004 (tenant-specific model weights), FR-005 (rule-based fallback), FR-009 (pluggable storage backend), FR-010 (cache with TTL)
- **Plan**: Design Decisions D4 (cache config), D5 (fallback behavior), D6 (endpoint guards)
- **Data Model**: `data-model.md` -- ModelLoader Protocol, FilesystemModelLoader, entity relationships diagram
- **Research**: R4 (model loading protocol -- typing.Protocol, not ABC)

**Dependencies**: WP02 (stateless endpoints with mode guards), WP03 (TenantContext available in handlers), WP04 (ModelCache implemented).

**Current model loading** (local mode):
- `AppState.load_models()` creates wrapper objects (`StuckPredictor()`, etc.) that load weights from `config.weights_path(name)` in their `__init__`.
- Each wrapper class has a `predict(features)` method and an `is_trained` property.
- In cloud mode, we need to load per-tenant models. Two approaches:
  1. Load raw sklearn models and call `predict_proba` directly.
  2. Load into wrapper objects (recommended -- reuses existing prediction logic).

**Existing fallback values** (from `poller.py` lines 113-131 and `routes.py`):
- Stuck: `probability=0.5, confidence="weak"`
- Workflow/Suggest: `shallow_work=1.0, method="rules", confidence=0.5`
- Duration: `estimated_minutes=60.0, confidence_interval=[30.0, 90.0]`
- Quality: `score=50, components={}, status="normal"`

**Implementation command**: `spec-kitty implement WP05 --base WP04`

## Subtasks & Detailed Guidance

### Subtask T024 -- Create `ModelLoader` Protocol in `src/sigil_ml/loader.py`

- **Purpose**: Define a pluggable interface for loading model weights from any storage backend. Feature 003 (Model Storage Abstraction) will provide an S3 implementation against this protocol.
- **Steps**:
  1. Create `src/sigil_ml/loader.py`:
     ```python
     """Model loading interface for pluggable storage backends.

     Defines the ModelLoader protocol that storage backends must implement.
     Feature 003 (Model Storage Abstraction) will provide an S3 implementation.
     """

     from __future__ import annotations

     import logging
     from pathlib import Path
     from typing import Any, Protocol, runtime_checkable

     logger = logging.getLogger(__name__)


     @runtime_checkable
     class ModelLoader(Protocol):
         """Protocol for loading model objects from a storage backend.

         Implementations must:
         - Handle tenant-specific model resolution.
         - Return None when no model exists (not raise exceptions).
         - Be thread-safe (may be called concurrently).
         """

         def load(self, tenant_id: str, model_name: str) -> Any | None:
             """Load a model for the given tenant and model name.

             Args:
                 tenant_id: Tenant identifier.
                 model_name: One of "stuck", "suggest", "workflow",
                             "duration", "activity", "quality".

             Returns:
                 The loaded model object, or None if not found.
             """
             ...
     ```
  2. Use `@runtime_checkable` so `isinstance(loader, ModelLoader)` works for validation.
  3. The protocol is intentionally minimal: one method, structural typing.
- **Files**: `src/sigil_ml/loader.py` (new file, ~30 lines initially)
- **Parallel?**: No -- T025 implements this.

### Subtask T025 -- Implement `FilesystemModelLoader` in `src/sigil_ml/loader.py`

- **Purpose**: Provide the initial concrete loader for local development and testing. Loads joblib-serialized model weights from the filesystem.
- **Steps**:
  1. Add to `src/sigil_ml/loader.py`:
     ```python
     import joblib
     from sigil_ml import config


     class FilesystemModelLoader:
         """Loads model weights from the local filesystem.

         Directory layout:
             {models_dir}/{tenant_id}/{model_name}.joblib  (tenant-specific)
             {models_dir}/{model_name}.joblib               (shared fallback)
         """

         def __init__(self, base_dir: Path | None = None) -> None:
             """Initialize with the base directory for model weights.

             Args:
                 base_dir: Root directory. Defaults to config.models_dir().
             """
             self._base_dir = base_dir or config.models_dir()

         def load(self, tenant_id: str, model_name: str) -> Any | None:
             """Load a model from the filesystem.

             Tries tenant-specific path first, then shared path.
             Returns None if neither exists or if loading fails.
             """
             # Tenant-specific path
             tenant_path = self._base_dir / tenant_id / f"{model_name}.joblib"
             if tenant_path.exists():
                 return self._safe_load(tenant_path, tenant_id, model_name)

             # Shared model fallback (no tenant directory)
             shared_path = self._base_dir / f"{model_name}.joblib"
             if shared_path.exists():
                 logger.info(
                     "loader: using shared model for %s/%s",
                     tenant_id, model_name,
                 )
                 return self._safe_load(shared_path, tenant_id, model_name)

             logger.debug(
                 "loader: no model found for %s/%s", tenant_id, model_name
             )
             return None

         def _safe_load(
             self, path: Path, tenant_id: str, model_name: str
         ) -> Any | None:
             """Load a joblib file with error handling."""
             try:
                 model = joblib.load(path)
                 logger.info(
                     "loader: loaded %s/%s from %s", tenant_id, model_name, path
                 )
                 return model
             except Exception:
                 logger.warning(
                     "loader: failed to load %s/%s from %s",
                     tenant_id, model_name, path,
                     exc_info=True,
                 )
                 return None
     ```
  2. The dual-path lookup (tenant-specific then shared) enables gradual rollout.
  3. All exceptions are caught and logged -- corrupt or incompatible files return `None` triggering fallback.
- **Files**: `src/sigil_ml/loader.py` (modify -- add ~50 lines)
- **Parallel?**: No -- builds on T024.
- **Notes**: The loader returns raw model objects (sklearn estimators). T027 handles how these are used in prediction handlers.

### Subtask T026 -- Integrate `ModelCache` + `ModelLoader` into `AppState` for cloud mode

- **Purpose**: Wire cache and loader into the application so prediction handlers can resolve tenant-specific models.
- **Steps**:
  1. Update imports in `src/sigil_ml/app.py`:
     ```python
     from sigil_ml.cache import ModelCache, create_model_cache
     from sigil_ml.loader import FilesystemModelLoader, ModelLoader
     ```
  2. Add cloud-mode fields to `AppState.__init__()` (updating the placeholders from WP01):
     ```python
     # Cloud-mode fields
     self.model_cache: ModelCache | None = None
     self.model_loader: ModelLoader | None = None
     ```
  3. Initialize in cloud startup path (update the `else` branch from WP01 T005):
     ```python
     else:
         # Cloud mode
         state.model_cache = create_model_cache()
         state.model_loader = FilesystemModelLoader()
         logger.info("kenaz-ml: cloud mode -- cache and loader initialized")
     ```
  4. Add `resolve_model()` method to `AppState`:
     ```python
     def resolve_model(self, tenant_id: str, model_name: str) -> Any | None:
         """Resolve a model for the given tenant, using cache then loader.

         Returns the model object or None if no model is available.
         Only used in cloud mode.
         """
         if self.model_cache is None or self.model_loader is None:
             return None

         # Check cache first
         model = self.model_cache.get(tenant_id, model_name)
         if model is not None:
             logger.debug(
                 "model-resolve: cache_hit tenant=%s model=%s",
                 tenant_id, model_name,
             )
             return model

         # Cache miss: load from backend
         model = self.model_loader.load(tenant_id, model_name)
         if model is not None:
             self.model_cache.put(tenant_id, model_name, model)
             logger.info(
                 "model-resolve: cache_miss+loaded tenant=%s model=%s",
                 tenant_id, model_name,
             )
             return model

         logger.info(
             "model-resolve: cache_miss+fallback tenant=%s model=%s",
             tenant_id, model_name,
         )
         return None
     ```
- **Files**: `src/sigil_ml/app.py` (modify -- add ~30 lines)
- **Parallel?**: No -- T027 depends on this.
- **Notes**: `resolve_model()` encapsulates the cache-then-load-then-fallback pattern. In local mode, this method is never called (local handlers use `state.stuck`, `state.workflow`, etc. directly).

### Subtask T027 -- Update each `/predict/*` handler to resolve via cache in cloud mode

- **Purpose**: Complete the cloud model serving loop: request -> tenant -> model resolution -> prediction -> response.
- **Steps**:
  1. Update `predict_stuck()` as the reference implementation:
     ```python
     @fastapi_app.post("/predict/stuck", response_model=StuckResponse)
     async def predict_stuck(
         req: StuckRequest,
         tenant: TenantContext = Depends(get_tenant),
     ) -> StuckResponse:
         if state.mode == ServingMode.CLOUD:
             model = state.resolve_model(tenant.tenant_id, "stuck")
             if model is None:
                 return StuckResponse(probability=0.5, confidence="weak")
             if req.features is None:
                 raise HTTPException(
                     status_code=400,
                     detail="Cloud mode requires 'features' in request body.",
                 )
             # Use the wrapper class's predict method if it's a wrapper,
             # or create a temporary wrapper if it's a raw sklearn model.
             predictor = StuckPredictor()
             predictor._model = model
             result = predictor.predict(req.features)
             return StuckResponse(**result)

         # Local mode: unchanged from WP02
         if state.stuck is None:
             return StuckResponse(probability=0.5, confidence="weak")
         if req.features is not None:
             features = req.features
         elif req.task_id is not None:
             features = extract_stuck_features(config.db_path(), req.task_id)
         else:
             return StuckResponse(probability=0.5, confidence="weak")
         result = state.stuck.predict(features)
         return StuckResponse(**result)
     ```
  2. Apply the same pattern to all `/predict/*` handlers. For each:
     - **stuck**: resolve `"stuck"` model, use `StuckPredictor` wrapper.
     - **suggest/workflow**: resolve `"workflow"` model, use `WorkflowStatePredictor` wrapper.
     - **duration**: resolve `"duration"` model, use `DurationEstimator` wrapper.
     - **quality**: `QualityEstimator` is rule-based (no trained model needed). In cloud mode, create a fresh `QualityEstimator()` instance. Quality does not need per-tenant model resolution.
  3. **Important**: Review each model wrapper class to understand how to inject a loaded sklearn model. The wrappers store their model as `self._model` or similar. Inspect `src/sigil_ml/models/stuck.py`, `workflow.py`, `duration.py` to find the attribute name.
  4. **Alternative approach**: If the wrapper classes accept a model in their constructor or have a `set_model()` method, use that. If not, setting `predictor._model = model` directly works but is fragile. Consider adding a `from_model(model)` classmethod to each wrapper.
- **Files**: `src/sigil_ml/routes.py` (modify), possibly `src/sigil_ml/models/*.py` (minor additions)
- **Parallel?**: No -- modifies all prediction endpoints.
- **Notes**: The key design choice is how to bridge raw sklearn models (from `joblib.load`) to the prediction wrapper classes. Investigate the wrapper classes' internals before implementing. The cleanest approach may be adding a classmethod to each wrapper.

### Subtask T028 -- Implement rule-based fallback predictions for all five model types

- **Purpose**: FR-005 requires graceful fallback when no trained model exists for a tenant. Centralize fallback values.
- **Steps**:
  1. Define centralized fallback constants at the top of `routes.py` (or in a separate `fallbacks.py`):
     ```python
     # Centralized fallback predictions for cloud mode when no model is available.
     # These match existing fallbacks in poller.py and routes.py.

     FALLBACK_STUCK = StuckResponse(probability=0.5, confidence="weak")

     FALLBACK_SUGGEST = WorkflowStateResponse(
         flow_state={
             "shallow_work": 1.0, "deep_work": 0.0,
             "exploring": 0.0, "blocked": 0.0, "winding_down": 0.0,
         },
         dominant_state="shallow_work",
         momentum=0.0,
         focus_score=0.5,
         dominant_activity="idle",
         activity_distribution={},
         session_elapsed_min=0.0,
         method="rules",
         confidence=0.5,
     )

     FALLBACK_DURATION = DurationResponse(
         estimated_minutes=60.0, confidence_interval=[30.0, 90.0]
     )

     FALLBACK_QUALITY = QualityResponse(
         score=50, components={}, status="normal"
     )
     ```
  2. Use these constants in each endpoint's cloud-mode fallback path (when `resolve_model()` returns `None`).
  3. Verify each fallback matches the existing values in `poller.py` (lines 113-131) and `routes.py` (scattered across handlers).
- **Files**: `src/sigil_ml/routes.py` (modify -- add ~25 lines of constants)
- **Parallel?**: Yes -- constants can be defined alongside T027 work.
- **Notes**: Centralizing prevents drift. Each fallback includes `method="rules"` or `confidence="weak"` so consumers know it is not a trained-model prediction.

### Subtask T029 -- Structured logging for model load events

- **Purpose**: Operators need visibility into cache behavior for debugging latency and model freshness issues.
- **Steps**:
  1. Logging is already embedded in `resolve_model()` (T026):
     - `DEBUG`: cache_hit (high frequency, low signal)
     - `INFO`: cache_miss+loaded (model loaded from storage)
     - `INFO`: cache_miss+fallback (no model available)
  2. Add request-level logging in prediction handlers (cloud mode only):
     ```python
     if state.mode == ServingMode.CLOUD:
         logger.info(
             "predict/stuck: tenant=%s model_resolved=%s",
             tenant.tenant_id, model is not None,
         )
     ```
  3. Log format should always include `tenant_id` and `model_name` for filtering.
- **Files**: `src/sigil_ml/app.py` (verify T026 logging), `src/sigil_ml/routes.py` (add handler-level logging)
- **Parallel?**: Yes -- logging additions are independent.
- **Notes**: Use `DEBUG` for cache hits (very frequent), `INFO` for misses and loads (important for debugging). Do NOT log the model object itself.

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Thundering herd on cache miss | Medium | Low | Multiple requests for same `(tenant, model)` may all trigger loads. Consider a per-key loading lock in `resolve_model()`. Basic implementation works without it. |
| Model format mismatch | Medium | Medium | Different sklearn versions produce incompatible joblib files. `_safe_load()` catches exceptions and returns `None`. Log sklearn version on load for debugging. |
| Memory pressure from models | Low | Medium | Each model is 1-50 MB. `ModelCache.max_size=100` limits total. With 5 models per tenant, supports ~20 tenants. |
| Wrapper class internal API changes | Low | Medium | Setting `predictor._model = model` is fragile. Consider adding `from_model()` classmethod to wrappers. |

## Review Guidance

- Verify the full request flow: request -> tenant extraction -> model resolution (cache -> loader -> fallback) -> prediction -> response.
- Verify fallback responses match existing defaults in `routes.py` and `poller.py`.
- Verify local mode is completely unchanged (no `resolve_model()` calls, no cache usage).
- Verify model loading failures (corrupt files) result in fallback responses, not 500 errors.
- Verify structured logs show cache hit/miss/fallback events with tenant_id.
- Verify `FilesystemModelLoader` handles both tenant-specific and shared model paths.
- Run all existing tests -- zero regression.

## Activity Log

- 2026-03-30T01:45:14Z -- system -- lane=planned -- Prompt regenerated.
