# Quickstart: Cloud Serving Mode

## Prerequisites

- Python 3.10+
- kenaz-ml installed: `pip install -e ".[dev]"`

## Running in Cloud Mode

```bash
# Start kenaz-ml in cloud mode
kenaz-ml serve --mode cloud --host 0.0.0.0 --port 7774

# Or via environment variable
SIGIL_ML_MODE=cloud kenaz-ml serve --host 0.0.0.0 --port 7774
```

Cloud mode starts without:
- SQLite database connection
- EventPoller background task
- TrainingScheduler background task

## Making Predictions (Cloud Mode)

All prediction requests in cloud mode require:
1. The `X-Tenant-ID` header
2. Features provided directly in the request body (no `task_id` lookup)

### Stuck Prediction

```bash
curl -X POST http://localhost:7774/predict/stuck \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: tenant-abc" \
  -d '{
    "features": {
      "test_failure_count": 5,
      "time_in_phase_sec": 1200,
      "edit_velocity": 4.0,
      "file_switch_rate": 0.7,
      "session_length_sec": 3600,
      "time_since_last_commit_sec": 1800
    }
  }'
```

### Workflow State (Suggest)

```bash
curl -X POST http://localhost:7774/predict/suggest \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: tenant-abc" \
  -d '{
    "classified_events": [
      {"kind": "file", "_category": "editing", "ts": 1000},
      {"kind": "terminal", "_category": "verifying", "ts": 2000}
    ]
  }'
```

### Duration Estimation

```bash
curl -X POST http://localhost:7774/predict/duration \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: tenant-abc" \
  -d '{
    "features": {
      "file_count": 10,
      "total_edits": 80,
      "time_of_day_hour": 14,
      "branch_name_length": 25
    }
  }'
```

### Quality Score

```bash
curl -X POST http://localhost:7774/predict/quality \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: tenant-abc" \
  -d '{
    "features": {
      "test_pass_rate": 0.85,
      "test_total": 20,
      "edit_focus": 0.7,
      "velocity_ratio": 1.2,
      "commits_in_window": 3,
      "expected_commits": 2.0,
      "revert_count": 0,
      "edits_in_window": 45
    }
  }'
```

## Health Check

```bash
# Cloud mode health check
curl http://localhost:7774/health
# Response: {"status": "ok", "mode": "cloud", "models": {...}, "uptime_sec": 12.3}
```

## Model Weights for Testing

To test with pre-trained models, place joblib files in the tenant directory:

```bash
# Create tenant model directory
mkdir -p ~/.local/share/sigild/ml-models/tenant-abc/

# Copy or train models (from local mode training)
cp ~/.local/share/sigild/ml-models/stuck.joblib \
   ~/.local/share/sigild/ml-models/tenant-abc/stuck.joblib
```

If no model exists for a tenant, all endpoints return rule-based fallback predictions.

## Environment Variables

| Variable                    | Default       | Description                                    |
|-----------------------------|---------------|------------------------------------------------|
| `SIGIL_ML_MODE`             | `local`       | Serving mode when `--mode` flag is not provided |
| `MODEL_CACHE_TTL_SECONDS`   | `300`         | Model cache TTL in seconds (cloud mode only)   |
| `MODEL_CACHE_MAX_SIZE`      | `100`         | Maximum cached model entries (cloud mode only) |
| `SIGIL_TENANT_HEADER`       | `X-Tenant-ID` | HTTP header name for tenant identification     |

## Running in Local Mode (Unchanged)

```bash
# Default behavior - identical to current
kenaz-ml serve

# Explicit local mode
kenaz-ml serve --mode local
```

Local mode requires the sigild SQLite database and behaves identically to the current implementation.
