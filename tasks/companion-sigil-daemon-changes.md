# Companion: Go Daemon Changes for Classify → Predict → LLM Pipeline

## Overview

This document outlines the changes needed in the [sigil](https://github.com/kameas-ai/sigil) Go daemon to consume the new prediction format from kenaz-ml and generate LLM-driven suggestions. These changes are **out of scope** for the kenaz-ml PRD but are documented here for planning and coordination.

## Context

After the kenaz-ml PRD is implemented, the `ml_predictions` table will contain two new prediction types:

1. **model `"activity"`** — classified event summary with semantic categories
2. **model `"suggest"`** (same name, new format) — workflow state assessment with flow state probabilities, momentum, focus score, and activity distribution

The Go daemon currently reads `ml_predictions` via the `get_predictions` MCP tool (`internal/mcp/tools.go:77-125`) and exposes them to the LLM. Since the `result` column is JSON and the LLM reads arbitrary JSON, **no code changes are strictly required** — the LLM will interpret the new format. However, the following changes would improve the experience.

## Recommended Changes

### 1. Update MCP tool known model list

**File:** `internal/mcp/tools.go:107`

Add `"activity"` and `"workflow"` to the known models list queried by `get_predictions` when no specific model is requested.

```go
// Current:
models := []string{"quality", "task_estimate", "context_switch", "productivity"}
// New:
models := []string{"activity", "suggest", "quality", "stuck", "duration"}
```

### 2. Add `get_workflow_state` MCP tool

**File:** `internal/mcp/tools.go`

A dedicated tool that returns the latest workflow state assessment formatted for LLM reasoning:

```go
// Tool: get_workflow_state
// Description: "Returns the user's current workflow state including flow state
//   probabilities (deep_work, shallow_work, exploring, blocked, winding_down),
//   momentum, focus score, and activity breakdown. Use this to understand what
//   the user is doing and how it's going before generating suggestions."
// No parameters.
```

This queries `ml_predictions WHERE model = 'suggest'` (latest non-expired) and returns the full state assessment.

### 3. Add `get_activity_stream` MCP tool

**File:** `internal/mcp/tools.go`

A tool that returns the recent classified activity stream:

```go
// Tool: get_activity_stream
// Description: "Returns recent events classified by activity type (creating,
//   refining, verifying, navigating, researching, integrating, communicating,
//   idle). Shows what the user has been doing in the last few minutes."
// No parameters.
```

### 4. Update analyzer system prompt for LLM suggestion generation

**File:** `internal/analyzer/analyzer.go` (cloudPass / buildPrompt)

The LLM prompt should instruct the model to use workflow state predictions when generating suggestions:

```
You are a workflow assistant. The user's current state is provided via tool calls.
Before making suggestions, call get_workflow_state to understand their flow state,
momentum, and focus level. Generate suggestions that are:
- Appropriate to their current state (don't interrupt deep_work, help with blocked)
- Specific to what they're actually doing (use activity stream for context)
- Actionable (suggest a concrete next step, not vague advice)
- Concise (one sentence)
```

### 5. Write feedback events for learning

**File:** `internal/notifier/notifier.go` or `internal/store/store.go`

When the user accepts or dismisses an LLM-generated suggestion, write feedback events that kenaz-ml can learn from:

```go
// On suggestion accept:
store.InsertEvent(ctx, event.Event{
    Kind:    "ml_feedback",
    Source:  "notifier",
    Payload: map[string]any{
        "model":    "suggest",
        "accepted": true,
        "state":    currentWorkflowState, // from latest prediction
    },
})

// On suggestion dismiss:
// Same with "accepted": false
```

### 6. Write correction events for classification fixes

**File:** `cmd/sigilctl/main.go` or new socket handler

Add a `sigilctl correct <event_id> <category>` command that lets users fix misclassified events:

```go
// sigilctl correct 12345 researching
// → writes event with kind="ml_correction", payload={"event_id": 12345, "correct_category": "researching"}
```

This is the highest-signal feedback for the ActivityClassifier but requires explicit user action.

### 7. Adjust task tracker ML integration

**File:** `internal/task/tracker.go`

The task tracker currently only uses stuck predictions (`predictStuck()` at line 284). Consider also reading workflow state to:
- Skip suggestions when `dominant_state == "deep_work"` and `focus_score > 0.8` (don't interrupt flow)
- Increase suggestion urgency when `dominant_state == "blocked"` and `momentum < -0.5`
- Use `session_elapsed_min` to suggest breaks after long sessions

## Event Schema for New Event Kinds

### ml_feedback
```json
{
    "kind": "ml_feedback",
    "source": "notifier",
    "payload": {
        "model": "suggest",
        "accepted": true,
        "state": "blocked"
    },
    "ts": 1711000000000
}
```

### ml_correction
```json
{
    "kind": "ml_correction",
    "source": "sigilctl",
    "payload": {
        "event_id": 12345,
        "correct_category": "researching"
    },
    "ts": 1711000000000
}
```

## Priority Order

1. **MCP tool updates** (items 1-3) — enables LLM to read new predictions immediately
2. **System prompt update** (item 4) — improves suggestion quality
3. **Feedback events** (items 5-6) — enables continuous learning
4. **Task tracker integration** (item 7) — enhances autonomous behavior

Items 1-2 are sufficient for the pipeline to work end-to-end. Items 3-7 improve quality over time.

## Dependencies

- kenaz-ml PRD must be implemented first (new prediction format must exist in `ml_predictions`)
- No database schema changes needed (uses existing `events` and `ml_predictions` tables)
- No new Go dependencies
