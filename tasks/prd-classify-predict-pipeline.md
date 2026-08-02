# PRD: Replace Hardcoded Suggestions with Classify → Predict → LLM Pipeline

## Introduction

kenaz-ml currently uses a Thompson Sampling bandit (`SuggestionPolicy`) that picks from 11 hardcoded action strings like `suggest_commit` and `suggest_break`. The context parameter is accepted but ignored — it is a context-free bandit. The `QualityEstimator` also contains hardcoded English suggestion strings mapped to quality components.

This architecture is rigid, developer-specific, and does not learn from the user's actual workflow. It cannot adapt to different roles (PM, data scientist, designer) or different working styles within the same role.

This PRD replaces the hardcoded suggestion system with a flexible classification and prediction pipeline. kenaz-ml will **classify events into semantic activity categories** and **predict workflow state as a probability distribution**, then write structured predictions to the shared SQLite database. The Go daemon's LLM (already connected via MCP tools) reads these predictions and generates natural language suggestions appropriate to the user's actual context.

The system starts rule-based on day 1 and upgrades to ML as data accumulates. It learns from task outcomes, suggestion feedback, and explicit corrections — getting sharper over time without manual tuning.

**Scope:** This PRD covers kenaz-ml (Python) changes only. A companion document covers the Go daemon changes needed to consume the new prediction format and generate LLM-driven suggestions.

## Goals

- Remove all hardcoded suggestion actions and suggestion text from kenaz-ml
- Classify raw events into 8 role-agnostic semantic activity categories that work for any knowledge worker
- Predict workflow state as a multi-dimensional assessment (probability distribution over flow states, momentum, focus, activity breakdown)
- Write structured, self-describing predictions to `ml_predictions` for LLM consumption
- Start useful on day 1 with rule-based classification and prediction
- Automatically upgrade to ML models as sufficient data accumulates (~500 events for activity, ~20 completed tasks for workflow)
- Learn continuously from implicit feedback (task outcomes), explicit feedback (suggestion accept/dismiss), and corrections (user-provided classification fixes)
- Preserve backward compatibility with existing `ml_predictions` model names that the Go daemon queries

## User Stories

### US-001: Create ActivityClassifier with rule-based cold start
**Description:** As a kenaz-ml developer, I need an ActivityClassifier model that categorizes raw events into semantic activity categories so that downstream models and the LLM understand what the user is doing, not just what tool generated the event.

**Acceptance Criteria:**
- [ ] New file `src/kenaz_ml/models/activity.py` with `ActivityClassifier` class
- [ ] 8 categories defined: `creating`, `refining`, `verifying`, `navigating`, `researching`, `integrating`, `communicating`, `idle`
- [ ] At cold start, both "creating" and "refining" map to a single `editing` category (split later when ML model trains)
- [ ] `classify(event: dict) -> dict` returns `{"category": str, "confidence": float, "method": "rules"|"ml"}`
- [ ] `classify_batch(events: list[dict]) -> list[dict]` classifies a buffer of events
- [ ] Rule-based classification logic covers all 6 event kinds (`file`, `process`, `hyprland`, `git`, `terminal`, `ai`)
- [ ] `is_trained` property returns `False` for rule-based, `True` after ML training
- [ ] Weight persistence via joblib at `weights_path("activity")`
- [ ] `pytest tests/test_models.py` passes with new ActivityClassifier tests

### US-002: Add activity feature extraction
**Description:** As a kenaz-ml developer, I need feature extraction for the ActivityClassifier so that ML training has proper input features derived from event kind and payload.

**Acceptance Criteria:**
- [ ] New function `extract_activity_features(event: dict) -> dict` in `features.py`
- [ ] Features extracted from event `kind`, `source`, and `payload` fields
- [ ] Features include: event kind (one-hot), payload key presence flags, command type classification (for terminal events), file extension category (for file events)
- [ ] Returns a flat dict of floats suitable for sklearn input
- [ ] `pytest tests/test_features.py` passes with new tests

### US-003: Integrate ActivityClassifier into poller event processing
**Description:** As a kenaz-ml user, I want every event classified as it enters the poller buffer so that downstream predictions have semantic activity context.

**Acceptance Criteria:**
- [ ] Poller constructor accepts `models["activity"]` (ActivityClassifier instance)
- [ ] Each event is classified as it enters the buffer in `_poll_once()`
- [ ] Classified events stored in buffer as `{...event, "_category": str, "_category_confidence": float}`
- [ ] Activity summary prediction written to `ml_predictions` with `model="activity"` on each prediction cycle
- [ ] Activity prediction format: `{"window_summary": {category: count}, "recent": [{ts, kind, category}], "dominant": str, "method": str, "confidence": float}`
- [ ] `recent` array capped at last 10 events for token efficiency
- [ ] `pytest tests/` passes

### US-004: Add /predict/activity API endpoint
**Description:** As a kenaz-ml API consumer, I want a `/predict/activity` endpoint so I can query event classifications on demand.

**Acceptance Criteria:**
- [ ] `POST /predict/activity` endpoint in `server.py`
- [ ] Request accepts optional `events` list or uses current buffer
- [ ] Response returns activity summary in same format as poller writes
- [ ] ActivityClassifier added to `/health` endpoint model status
- [ ] `pytest tests/test_server.py` passes with new endpoint tests

### US-005: Create WorkflowStatePredictor with rule-based cold start
**Description:** As a kenaz-ml developer, I need a WorkflowStatePredictor that assesses the user's current workflow state from a window of classified events, replacing the hardcoded SuggestionPolicy.

**Acceptance Criteria:**
- [ ] New file `src/kenaz_ml/models/workflow.py` with `WorkflowStatePredictor` class
- [ ] 5 flow states defined: `deep_work`, `shallow_work`, `exploring`, `blocked`, `winding_down`
- [ ] `predict(classified_events: list[dict], session_info: dict) -> dict` returns full state assessment
- [ ] State assessment includes: `flow_state` (probability distribution over 5 states), `dominant_state`, `momentum` (-1 to 1), `focus_score` (0-1), `dominant_activity`, `activity_distribution`, `session_elapsed_min`, `method`, `confidence`
- [ ] Rule-based prediction from activity distributions (high creating+refining with low navigating → deep_work, high verifying with failures → blocked, etc.)
- [ ] Momentum computed by comparing recent half vs older half of event window
- [ ] Focus score computed as inverse of activity category entropy
- [ ] `is_trained` property returns `False` for rule-based, `True` after ML training
- [ ] Weight persistence via joblib at `weights_path("workflow")`
- [ ] `pytest tests/test_models.py` passes with new WorkflowStatePredictor tests

### US-006: Add workflow state feature extraction
**Description:** As a kenaz-ml developer, I need feature extraction for the WorkflowStatePredictor so that ML training has proper window-level features.

**Acceptance Criteria:**
- [ ] New function `extract_workflow_features(classified_events: list[dict], session_info: dict) -> dict` in `features.py`
- [ ] Features include: category counts (8 floats, normalized), category entropy, event rate (events/min), category transition count, time in dominant category (fraction), recent bias (weighted recency)
- [ ] `session_info` dict includes `session_elapsed_min`, `task_phase`, `test_failures`
- [ ] Returns flat dict of floats suitable for sklearn input
- [ ] `pytest tests/test_features.py` passes with new tests

### US-007: Replace suggest prediction in poller with workflow state
**Description:** As a kenaz-ml user, I want the poller to write workflow state assessments instead of hardcoded suggestion actions so the LLM can generate context-appropriate suggestions.

**Acceptance Criteria:**
- [ ] Poller constructor accepts `models["workflow"]` (WorkflowStatePredictor) instead of `models["suggest"]` (SuggestionPolicy)
- [ ] `_predict_and_write()` calls `self.workflow.predict()` instead of `self.suggest.predict()`
- [ ] Prediction written to `ml_predictions` with `model="suggest"` (preserved name for Go compatibility)
- [ ] New prediction format: `{"flow_state": {...}, "dominant_state": str, "momentum": float, "focus_score": float, ...}`
- [ ] `_FALLBACK_SUGGEST` updated to new format: `{"flow_state": {"shallow_work": 1.0, ...}, "dominant_state": "shallow_work", "momentum": 0.0, ...}`
- [ ] `pytest tests/` passes

### US-008: Replace /predict/suggest endpoint with workflow state response
**Description:** As a kenaz-ml API consumer, I want the `/predict/suggest` endpoint to return workflow state assessments in the new format.

**Acceptance Criteria:**
- [ ] `/predict/suggest` endpoint returns new schema: `flow_state`, `dominant_state`, `momentum`, `focus_score`, `dominant_activity`, `activity_distribution`, `session_elapsed_min`, `method`, `confidence`
- [ ] Old `SuggestRequest`/`SuggestResponse` schemas replaced with `WorkflowStateRequest`/`WorkflowStateResponse`
- [ ] Request accepts optional `task_id` or pre-computed `classified_events`
- [ ] Breaking change — old `{"action": str, "confidence": float}` format no longer returned
- [ ] `pytest tests/test_server.py` passes with updated tests

### US-009: Delete SuggestionPolicy and related code
**Description:** As a kenaz-ml developer, I want to remove all Thompson Sampling bandit code and hardcoded action strings so the codebase reflects the new architecture.

**Acceptance Criteria:**
- [ ] `src/kenaz_ml/models/suggest.py` deleted
- [ ] `ACTIONS` list no longer exists anywhere in codebase
- [ ] `extract_suggest_features()` removed from `features.py`
- [ ] All references to `SuggestionPolicy` removed from `server.py`, `poller.py`, `training/trainer.py`
- [ ] `grep -r "suggest_commit\|suggest_break\|suggest_step_back\|ACTIONS\|SuggestionPolicy" src/` returns no results
- [ ] `pytest tests/` passes

### US-010: Remove hardcoded suggestion strings from QualityEstimator
**Description:** As a kenaz-ml developer, I want to remove hardcoded suggestion text from the QualityEstimator so that suggestion generation is entirely the LLM's responsibility.

**Acceptance Criteria:**
- [ ] `_suggest_for_degraded()` function deleted from `quality.py`
- [ ] `suggestion` key removed from `predict()` return dict
- [ ] `QualityResponse` in `server.py` no longer includes `suggestion` field
- [ ] All quality scoring logic preserved unchanged
- [ ] `grep -r "_suggest_for_degraded" src/` returns no results
- [ ] `pytest tests/` passes

### US-011: Add ML training path for ActivityClassifier
**Description:** As a kenaz-ml developer, I need ML training for the ActivityClassifier so it upgrades from rules to learned classification after sufficient data.

**Acceptance Criteria:**
- [ ] `ActivityClassifier.train(events, labels)` fits an `SGDClassifier` with `partial_fit()`
- [ ] `partial_fit()` used for incremental learning (no full retrain needed)
- [ ] New method `_train_activity()` added to `training/trainer.py`
- [ ] Training collects events from `events` table, applies rule-based labels as initial training data
- [ ] Reads `ml_correction` events (kind = `"ml_correction"`) for high-signal label overrides
- [ ] Minimum 500 events required before first ML training
- [ ] After training, `classify()` uses ML model with calibrated confidence instead of rules
- [ ] `train_all()` in trainer.py calls `_train_activity()` alongside other training methods
- [ ] `pytest tests/` passes

### US-012: Add ML training path for WorkflowStatePredictor
**Description:** As a kenaz-ml developer, I need ML training for the WorkflowStatePredictor so it upgrades from rules to learned state prediction after sufficient task data.

**Acceptance Criteria:**
- [ ] `WorkflowStatePredictor.train(X, y)` fits a `GradientBoostingClassifier`
- [ ] New method `_train_workflow()` added to `training/trainer.py`
- [ ] Training extracts classified-event windows from completed tasks
- [ ] Labels derived from task outcomes: fast completion + few failures → `deep_work`, long + many failures → `blocked` phases
- [ ] Reads `ml_feedback` events (kind = `"ml_feedback"`) to reinforce state predictions
- [ ] Minimum 20 completed tasks required before first ML training
- [ ] Falls back to synthetic data if fewer than 20 tasks (using `generate_workflow_data()`)
- [ ] `train_all()` in trainer.py calls `_train_workflow()` alongside other training methods
- [ ] `pytest tests/` passes

### US-013: Add synthetic data generators for new models
**Description:** As a kenaz-ml developer, I need synthetic data generators for the new models to support cold-start ML training.

**Acceptance Criteria:**
- [ ] `generate_activity_data(n=500)` added to `training/synthetic.py`
- [ ] Generates events with realistic kind/payload distributions and corresponding category labels
- [ ] `generate_workflow_data(n=200)` added to `training/synthetic.py`
- [ ] Generates classified-event windows with realistic activity distributions and flow state labels
- [ ] Both add Gaussian noise and shuffle before returning
- [ ] `pytest tests/` passes

### US-014: Update training scheduler for new feedback signals
**Description:** As a kenaz-ml developer, I want the training scheduler to trigger retraining based on new feedback events, not just completed tasks.

**Acceptance Criteria:**
- [ ] Scheduler tracks `ml_feedback` event count since last retrain
- [ ] Scheduler tracks `ml_correction` event count since last retrain
- [ ] Activity classifier retraining triggers after 500 new events (separate from full retrain cycle)
- [ ] Workflow retraining uses same trigger as stuck/duration (10 completed tasks, 1hr min interval)
- [ ] Correction events trigger immediate activity retrain after 5 corrections (high-signal)
- [ ] `pytest tests/` passes

### US-015: Update server startup and model wiring
**Description:** As a kenaz-ml developer, I need the server startup to load and wire the new models correctly.

**Acceptance Criteria:**
- [ ] `ActivityClassifier` instantiated and loaded at startup alongside other models
- [ ] `WorkflowStatePredictor` instantiated and loaded at startup alongside other models
- [ ] Old `SuggestionPolicy` import and instantiation removed
- [ ] Models dict passed to poller includes `"activity"` and `"workflow"` keys
- [ ] `/health` endpoint reports status of all models including `activity` and `workflow`
- [ ] `kenaz-ml serve` starts without errors
- [ ] `pytest tests/` passes

## Functional Requirements

- FR-1: The ActivityClassifier must classify every event into exactly one of 8 categories: `creating`, `refining`, `verifying`, `navigating`, `researching`, `integrating`, `communicating`, `idle`
- FR-2: At cold start (no ML model trained), `creating` and `refining` both classify as `editing` — they split into distinct categories only after ML training
- FR-3: The ActivityClassifier must include `method` field (`"rules"` or `"ml"`) in every classification so consumers know the confidence calibration source
- FR-4: The WorkflowStatePredictor must output a probability distribution over 5 flow states that sums to 1.0
- FR-5: The WorkflowStatePredictor must compute momentum by comparing event rates in the recent half vs older half of the window
- FR-6: The WorkflowStatePredictor must compute focus score as the inverse of Shannon entropy over the activity category distribution, normalized to [0, 1]
- FR-7: Predictions must be written to `ml_predictions` with model names `"activity"` and `"suggest"` (the latter preserves backward compatibility with Go daemon queries)
- FR-8: Every SQLite connection must set `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000`
- FR-9: The ActivityClassifier must support incremental learning via `SGDClassifier.partial_fit()` — no full retrain required
- FR-10: The WorkflowStatePredictor must fall back to synthetic data when fewer than 20 completed tasks exist
- FR-11: The system must read `ml_correction` events (kind = `"ml_correction"`, payload = `{"event_id": int, "correct_category": str}`) as gold-standard training labels
- FR-12: The system must read `ml_feedback` events (kind = `"ml_feedback"`, payload = `{"model": str, "accepted": bool}`) to reinforce or weaken state predictions
- FR-13: No hardcoded suggestion text may exist in kenaz-ml after implementation — all English-language suggestions are generated by the LLM in the Go daemon
- FR-14: The `QualityEstimator.predict()` return dict must not contain a `suggestion` key
- FR-15: All prediction outputs must be JSON-serializable and self-describing (the LLM reads them without external documentation)

## Non-Goals

- No changes to the Go daemon in this PRD (covered by companion document)
- No changes to the StuckPredictor or DurationEstimator models
- No changes to the `ml_predictions` or `ml_events` database schema (new model names use existing columns)
- No natural language generation in kenaz-ml — that is the LLM's job
- No user-facing UI changes
- No new Python dependencies (scikit-learn, numpy, joblib, fastapi, uvicorn only)
- No real-time streaming classification — events are classified in the polling loop, not via websocket
- No cross-session learning (each session starts fresh from persisted weights, not from in-memory state)

## Technical Considerations

### Existing code to reuse
- `config.weights_path(model_name)` for model persistence (`src/kenaz_ml/config.py`)
- `_connect()` pattern from `poller.py:226-230` for SQLite connections
- `_write()` method from `poller.py:163-177` for writing predictions
- `event.IsTestOrBuildCmd` pattern from Go's `payload.go` — replicate the command matching logic in Python for terminal event classification
- `FEATURE_NAMES` constant pattern from `stuck.py:14-21` and `duration.py:14-19`
- Synthetic data generation pattern from `synthetic.py:6-91`
- Model train/predict/is_trained/persist pattern from `stuck.py:24-84`

### Key constraints
- **No new dependencies.** SGDClassifier and GradientBoostingClassifier are both in scikit-learn >=1.4
- **SQLite WAL mode.** All connections must set WAL pragma. The Go daemon writes events concurrently
- **Model name `"suggest"` preserved.** The Go daemon's `get_predictions` MCP tool queries this name. Changing it requires coordinated Go changes (out of scope)
- **Buffer size.** Poller buffer is 200 events (recently expanded from 50). Activity classification should work within this buffer
- **Prediction TTL.** Activity and workflow predictions use 90-second TTL (same as current suggest)
- **Thread safety.** Models are accessed from the async poller loop and from FastAPI endpoint handlers. Model state must be safe for concurrent reads

### Performance
- Rule-based classification: <1ms per event (dict lookup + string matching)
- ML classification: <5ms per event (SGDClassifier.predict is fast)
- Workflow state prediction: <10ms per window (GradientBoosting on ~15 features)
- Total prediction cycle budget: <50ms (current cycle is ~20ms for 4 models)

## Success Metrics

- Zero hardcoded suggestion strings in kenaz-ml codebase
- All 6 event kinds classified into semantic categories with >0.7 confidence
- Workflow state predictions written to `ml_predictions` every prediction cycle
- ActivityClassifier upgrades from rules to ML within 2 days of active use (~500 events)
- WorkflowStatePredictor upgrades from rules to ML within 2 weeks of active use (~20 completed tasks)
- All existing tests pass plus new tests for activity and workflow models
- Go daemon's `get_predictions` MCP tool reads new prediction format without code changes (JSON is self-describing)

## Open Questions

- Should the `editing` cold-start category be written as `"editing"` or should it randomly assign `"creating"`/`"refining"` with 50/50 to generate diverse training data from day 1?
- Should the activity summary include plugin-sourced events (GitHub, Jira) or only core event kinds?
- What is the minimum buffer size needed for reliable workflow state prediction? (Current: 200 events, ~30 minutes of active work)
- Should the WorkflowStatePredictor use the stuck prediction as an input feature, or remain independent?
