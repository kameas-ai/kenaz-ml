# CLI Contract: Cloud Training Pipeline

**Feature**: 004-cloud-training-pipeline
**Date**: 2026-03-30

## Command: `kenaz-ml train`

### Synopsis

```
kenaz-ml train [--db PATH] [--mode {local,cloud}] [--tenant ID] [--all-tenants] [--aggregate]
               [--min-interval SECONDS] [--min-tasks COUNT] [--max-tasks-per-tenant COUNT]
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--db` | `str` | `~/.local/share/sigild/data.db` | SQLite database path (local mode only) |
| `--mode` | `str` | `local` | Operating mode: `local` or `cloud` |
| `--tenant` | `str` | None | Train a single tenant (cloud mode only) |
| `--all-tenants` | flag | False | Train all eligible tenants (cloud mode only) |
| `--aggregate` | flag | False | Train aggregate model from opted-in tenants (cloud mode only) |
| `--min-interval` | `int` | `3600` | Minimum seconds between retraining a tenant |
| `--min-tasks` | `int` | `10` | Minimum completed tasks for ML training |
| `--max-tasks-per-tenant` | `int` | `1000` | Cap per-tenant tasks in aggregate training |

### Mode: Local (default, unchanged)

```bash
# Train from local SQLite (existing behavior, no changes)
kenaz-ml train
kenaz-ml train --db /path/to/data.db
```

**Output**: Plain text summary to stdout.

```
Training models from /home/user/.local/share/sigild/data.db ...
Done: {'trained': ['stuck', 'duration'], 'samples': 150, 'duration_sec': 2.34}
```

**Exit codes**:
- `0`: Success
- `1`: Error (database not found, etc.)

### Mode: Cloud -- Single Tenant

```bash
kenaz-ml train --mode cloud --tenant abc123
```

**Required environment variables**:
- `SIGIL_ML_DB_URL`: Postgres connection URL
- `SIGIL_ML_S3_BUCKET`: S3 bucket for model weights

**Output**: Structured JSON to stdout.

```json
{"event": "tenant_start", "tenant_id": "abc123", "ts": "2026-03-30T01:00:00Z"}
{"event": "tenant_complete", "tenant_id": "abc123", "status": "trained", "models": ["stuck", "duration", "activity", "workflow", "quality"], "samples": 150, "duration_ms": 2340}
```

**Exit codes**:
- `0`: Training completed (success or skipped)
- `1`: Training failed
- `2`: Configuration error (missing env vars)

### Mode: Cloud -- All Tenants

```bash
kenaz-ml train --mode cloud --all-tenants
kenaz-ml train --mode cloud --all-tenants --min-interval 7200 --min-tasks 20
```

**Output**: Structured JSON to stdout, one line per event.

```json
{"event": "batch_start", "ts": "2026-03-30T01:00:00Z"}
{"event": "tenant_start", "tenant_id": "abc123", "ts": "2026-03-30T01:00:00Z"}
{"event": "tenant_complete", "tenant_id": "abc123", "status": "trained", "models": ["stuck", "duration", "activity", "workflow", "quality"], "samples": 150, "duration_ms": 2340}
{"event": "tenant_skip", "tenant_id": "def456", "reason": "recently_trained", "ts": "2026-03-30T01:00:03Z"}
{"event": "tenant_fail", "tenant_id": "ghi789", "error": "Connection timeout reading events", "ts": "2026-03-30T01:00:05Z"}
{"event": "batch_complete", "total": 3, "trained": 1, "skipped": 1, "failed": 1, "duration_ms": 5200, "ts": "2026-03-30T01:00:05Z"}
```

**Exit codes**:
- `0`: Batch completed (even if some tenants failed -- partial success)
- `1`: Batch failed entirely (e.g., cannot connect to Postgres)
- `2`: Configuration error

### Mode: Cloud -- Aggregate

```bash
kenaz-ml train --mode cloud --aggregate
kenaz-ml train --mode cloud --aggregate --max-tasks-per-tenant 500
```

**Output**: Structured JSON to stdout.

```json
{"event": "aggregate_start", "opted_in_tenants": 5, "ts": "2026-03-30T01:00:00Z"}
{"event": "aggregate_complete", "status": "trained", "models": ["stuck", "duration", "activity", "workflow", "quality"], "total_samples": 2500, "tenants_pooled": 5, "duration_ms": 8500, "ts": "2026-03-30T01:00:08Z"}
```

**Exit codes**:
- `0`: Aggregate training completed
- `1`: Aggregate training failed
- `2`: Configuration error

### Validation Rules

1. `--mode cloud` requires at least one of `--tenant`, `--all-tenants`, or `--aggregate`
2. `--tenant` and `--all-tenants` are mutually exclusive
3. `--aggregate` can be combined with `--all-tenants` to run both in sequence
4. `--db` is ignored when `--mode cloud` is set
5. In cloud mode, `SIGIL_ML_DB_URL` and `SIGIL_ML_S3_BUCKET` must be set; exit 2 if missing

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SIGIL_ML_MODE` | No | `local` | Can be used instead of `--mode` CLI arg |
| `SIGIL_ML_DB_URL` | Cloud only | - | Postgres connection URL |
| `SIGIL_ML_S3_BUCKET` | Cloud only | - | S3 bucket name |
| `SIGIL_ML_S3_REGION` | No | `us-east-1` | AWS region |
| `SIGIL_ML_S3_ENDPOINT` | No | - | S3-compatible endpoint URL |
| `SIGIL_ML_TRAIN_MIN_INTERVAL` | No | `3600` | Minimum retraining interval (seconds) |
| `SIGIL_ML_TRAIN_MIN_TASKS` | No | `10` | Minimum tasks for ML training |
| `SIGIL_ML_TRAIN_MAX_TASKS_PER_TENANT` | No | `1000` | Aggregate sampling cap |
