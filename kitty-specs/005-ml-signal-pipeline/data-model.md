# Data Model: ML Signal Pipeline

**Feature**: 005-ml-signal-pipeline
**Date**: 2026-03-30

## Entities

### MLSignal

A structured event emitted when a model detects something noteworthy. Written by kameas-ml, read by sigild.

| Field | Type | Description |
|-------|------|-------------|
| id | integer (auto) | Primary key |
| signal_type | text | Model-generated type — NOT a fixed enum. Examples: "velocity_deviation", "test_cadence_drop", "file_cluster_incomplete" |
| confidence | real (0.0–1.0) | Model's confidence that this signal is worth surfacing |
| evidence | text (JSON) | Structured evidence for LLM rendering |
| suggested_action | text (nullable) | Generic action hint for LLM: "investigate", "test", "commit", "take_break" |
| created_at | integer | Unix ms timestamp |
| expires_at | integer (nullable) | Unix ms, NULL = no expiry |
| rendered | integer (0/1) | Whether sigild has processed this signal |
| suggestion_id | integer (nullable) | FK to suggestions.id after rendering |

**Table**: `ml_signals` (Python-owned)
**Indexes**: `created_at`, `rendered`

### Evidence JSON Structure

Evidence varies by signal source but follows a common envelope:

```json
{
  "source_model": "pattern_detector|next_action|file_recommender",
  "metric": "edit_velocity|test_cadence|file_focus|...",
  "observed": 12.5,
  "baseline_mean": 5.2,
  "baseline_std": 1.8,
  "z_score": 4.06,
  "window_seconds": 300,
  "context": {
    "file": "src/sigil_ml/routes.py",
    "task_id": "t_1774563398270",
    "recent_events": 45
  }
}
```

For next-action divergence:
```json
{
  "source_model": "next_action",
  "predicted_action": "verifying:pytest",
  "predicted_probability": 0.72,
  "actual_action": "editing:python",
  "sequence_length": 15,
  "context": {
    "recent_tokens": ["editing:python", "editing:python", "editing:python", "..."]
  }
}
```

For file recommendation:
```json
{
  "source_model": "file_recommender",
  "current_file": "src/sigil_ml/app.py",
  "recommended_files": [
    {"path": "src/sigil_ml/routes.py", "co_occurrence": 0.85},
    {"path": "tests/test_server.py", "co_occurrence": 0.72}
  ],
  "context": {
    "repo": "/Users/user/project",
    "task_id": "t_123"
  }
}
```

### BehaviorProfile

Per-user summary stored as `ml_predictions` row with `model = 'profile'`.

```json
{
  "tool_frequency": {
    "git": 1503,
    "pytest": 120,
    "node": 1307,
    "claude": 719
  },
  "file_type_distribution": {
    ".py": 0.45,
    ".go": 0.25,
    ".md": 0.12,
    ".json": 0.08,
    ".yaml": 0.05,
    "other": 0.05
  },
  "workflow_rhythms": {
    "avg_commit_cadence_min": 35.0,
    "avg_test_cadence_min": 12.0,
    "avg_session_length_min": 90.0,
    "avg_edit_velocity_per_min": 5.2,
    "peak_activity_hour": 14,
    "avg_context_switches_per_hour": 3.2
  },
  "active_sources": ["files", "git", "terminal", "process", "focus"],
  "total_events_processed": 149208,
  "profile_version": 1,
  "updated_at": 1774892127278
}
```

### CompositeActionToken

Not persisted — computed in-memory from classified events.

Format: `{category}:{tool}` or `{category}` when tool is unknown.

Examples: `verifying:pytest`, `editing:python`, `integrating:git`, `researching:claude`, `navigating`, `idle`

**Derivation**: ActivityClassifier category + tool inferred from event payload (process comm, terminal cmd, or event kind).

### SignalFeedback

Not a new table — extends existing `suggestions` table with `signal_id` column (implemented in sigil Feature 021). kameas-ml reads the linkage for training labels:
- `suggestions.status = 'accepted'` + `signal_id` → positive label
- `suggestions.status = 'dismissed'` + `signal_id` → negative label
- `suggestions.status = 'ignored'` + `signal_id` → weak negative label

## Relationships

```
events (Go-owned, read by Python)
    │
    ├──> BehaviorProfile (computed from events, stored in ml_predictions)
    │
    ├──> PatternDetector (reads event buffer, writes ml_signals)
    │
    ├──> NextActionPredictor (reads event sequences, writes ml_signals)
    │
    └──> FileRecommender (reads file events, writes ml_signals)

ml_signals (Python-owned, read by Go)
    │
    └──> suggestions.signal_id (Go writes, Python reads for training labels)
```
