# Sigil Cloud Architecture

## Overview

Sigil Cloud extends the local-only Sigil developer intelligence platform into a cloud-hosted offering. The core product remains a local-first experience — the Go daemon (`sigild`) and optional Python ML sidecar (`kenaz-ml`) running on the developer's laptop. Sigil Cloud adds cloud-hosted LLM inference, cloud-hosted ML predictions, event sync, and team-level features for paying users.

**Guiding principles:**

- Local-first by default. Cloud is opt-in, never required.
- Paid users who opt into cloud ML do not need Python on their machine.
- Data sync is explicit consent — no silent uploads.
- The same ML models and feature extraction logic run locally and in the cloud.

---

## Tiers

| | Free | Pro | Team |
|---|---|---|---|
| **LLM inference** | Local (llama-server / Ollama) | Cloud default (can opt to local) | Cloud default (can opt to local) |
| **ML predictions** | Local (kenaz-ml sidecar) | Local default, **can opt into cloud** | Cloud default (can opt to local) |
| **Data sync** | None | None default, **full sync if cloud ML opted in** | Full sync default (can opt out) |
| **kenaz-ml on laptop** | Required (Python sidecar) | **Not needed if cloud ML** | **Not needed if cloud ML** |
| **Team dashboards** | — | — | Yes |
| **Aggregate models** | — | — | Yes (trained on pooled opted-in data) |
| **Install** | `brew install sigil kenaz-ml` | `brew install sigil` (single Go binary) | `brew install sigil` (single Go binary) |

### The Data Flywheel

Pro users who opt into cloud ML feed the aggregate training pool. Their events, task outcomes, and suggestion acceptance/dismissal signals become training data. This makes Team-tier aggregate models better, which sells Team. The more Pro users opt in, the better the product gets for everyone.

---

## Local Architecture (Unchanged)

```
┌─────────────────────────────────────────┐
│  Developer Laptop                        │
│                                          │
│  ┌─────────┐        ┌────────────────┐  │
│  │ sigild   │──────▶ │ SQLite (WAL)   │  │
│  │ (Go)     │ ◀───── │ ~/.local/share │  │
│  └─────────┘        │ /sigild/data.db │  │
│       │              └───────┬────────┘  │
│       │                      │           │
│       │              ┌───────▼────────┐  │
│       │              │ kenaz-ml       │  │
│       └─────────────▶│ (Python sidecar│  │
│        HTTP :7774    │  polls SQLite) │  │
│                      └────────────────┘  │
└─────────────────────────────────────────┘
```

- `sigild` collects events, tracks tasks, runs the analyzer, surfaces suggestions
- `kenaz-ml` polls SQLite for new events, runs 5 ML models, writes predictions back
- All communication via shared SQLite in WAL mode
- No network calls (except plugin capabilities on localhost)

---

## Cloud Architecture

```
┌─────────────────────────────────────────────┐
│  Developer Laptop                            │
│                                              │
│  ┌─────────┐        ┌────────────────┐      │
│  │ sigild   │──────▶ │ SQLite (WAL)   │      │
│  │ (Go)     │ ◀───── │                │      │
│  └────┬─────┘        └────────────────┘      │
│       │                                      │
│       ├─── ml.Engine ──── mode: "remote" ────┼──┐
│       ├─── inference.Engine ─ mode: "remote" ┼──┤
│       │                                      │  │
│       │  ┌──────────────┐                    │  │
│       └──│ sync agent   │ (if opted in) ─────┼──┤
│          │ (goroutine)  │                    │  │
│          └──────────────┘                    │  │
└──────────────────────────────────────────────┘  │
                                                  │
                    HTTPS (mTLS)                   │
                                                  │
┌─────────────────────────────────────────────────▼───┐
│  Sigil Cloud (K8s)                                   │
│                                                      │
│  ┌────────────┐                                     │
│  │ API Gateway │ ── auth (API key / JWT)            │
│  │ + billing   │ ── tenant isolation                │
│  │ + rate limit│ ── tier enforcement                │
│  └──────┬─────┘                                     │
│         │                                           │
│         ├───────────────────────────────────┐       │
│         │                                   │       │
│         ▼                                   ▼       │
│  ┌──────────────┐                   ┌─────────────┐│
│  │ LLM Proxy    │                   │ Ingest      ││
│  │              │                   │ Service     ││
│  │ routes to:   │                   │             ││
│  │ - OpenAI     │                   │ receives    ││
│  │ - Anthropic  │                   │ event stream││
│  │              │                   │ writes to   ││
│  └──────────────┘                   │ Postgres    ││
│                                     └──────┬──────┘│
│                                            │       │
│                                            ▼       │
│  ┌──────────────┐                   ┌─────────────┐│
│  │ kenaz-ml     │◀──────────────────│  Postgres   ││
│  │ Prediction   │                   │  per-tenant ││
│  │ API          │                   │  schemas    ││
│  │              │                   └──────┬──────┘│
│  │ - stateless  │                          │       │
│  │ - loads model│                          │       │
│  │   from S3    │                   ┌──────▼──────┐│
│  │ - /predict/* │                   │  Training   ││
│  └──────────────┘                   │  Pipeline   ││
│                                     │             ││
│                                     │  - CronJob  ││
│                                     │  - per-user ││
│                                     │  - aggregate││
│                                     │  - writes   ││
│                                     │    to S3    ││
│                                     └─────────────┘│
└─────────────────────────────────────────────────────┘
```

### No Python on the Laptop (Cloud ML Users)

When a paid user opts into cloud ML, the Go daemon's `ml.Engine` routes all prediction requests to the cloud API. The local `kenaz-ml` sidecar is never started. No Python runtime, no scikit-learn, no numpy — just the single Go binary.

The Go daemon already supports this via `ml.mode = "remote"`. The cloud backend (`ml/cloud.go`) sends `POST /predict/{endpoint}` with features in the request body and receives predictions in the response.

---

## Key Components

### 1. Sync Agent (in sigild)

A goroutine in the Go daemon that streams local SQLite changes to the cloud ingest API.

**What syncs:**

| Table | Synced | Why |
|---|---|---|
| `events` | Yes | Raw workflow signal — training data for all models |
| `tasks` | Yes | Task outcomes, phase transitions — labels for training |
| `suggestions` | Yes | Acceptance/dismissal status — ground truth for model evaluation |
| `ml_predictions` | Yes | Local model outputs become baselines for cloud model comparison |
| `ml_events` | Yes | Audit trail |
| `patterns` | Yes | Detected patterns — input to aggregate analysis |
| `ml_cursor` | No | Local bookkeeping only |
| Model weights (`.joblib`) | No | Derived artifacts, not source data |

**Behavior:**

- Tracks a per-table sync cursor (last synced row ID)
- Polls SQLite on a 5-second interval
- Batches rows and ships via HTTPS to the ingest service
- Retries with exponential backoff if cloud is unreachable (does not lose events)
- Only starts if `cloud.sync.enabled = true` and a valid API key is configured
- Bandwidth: ~500–2000 events/hour × ~200 bytes = <400 KB/hour. Trivial.

### 2. Cloud Ingest Service

Receives event streams from sync agents and writes to per-tenant Postgres.

- Authenticates via API key in request header
- Validates tenant tier (must be Pro with cloud ML opt-in, or Team)
- Writes to tenant-isolated schema in Postgres
- Schema mirrors SQLite table structure for compatibility
- Idempotent writes (sync cursor + event ID deduplication)

### 3. Cloud Prediction API (kenaz-ml in K8s)

The same kenaz-ml codebase, running in cloud mode. Serves prediction requests from Go daemons.

**Key differences from local mode:**

| | Local Sidecar | Cloud API |
|---|---|---|
| Entrypoint | `kenaz-ml serve` | `kenaz-ml serve --mode cloud` |
| Poller | Yes (polls SQLite every 500ms) | **No** (stateless, on-demand) |
| Data source | SQLite | Request payload from Go daemon |
| Model storage | `~/.local/share/sigild/ml-models/*.joblib` | S3 bucket, per-tenant prefix |
| Multi-tenant | No | Yes (tenant ID from auth header) |
| Scaling | Single process | Horizontal pod autoscaler on RPS |
| Training | Background scheduler | Separate K8s CronJob |

**Why stateless / no poller:**

Locally, kenaz-ml proactively polls and predicts every 60 seconds. In cloud mode this is wasteful — predictions are only needed at decision points:

- Task phase transitions (stuck detection)
- Analyzer cycles (hourly workflow analysis)
- MCP tool calls (user asks "what should I do next?")
- `sigilctl` health/status queries

The Go daemon already triggers predictions on-demand via `ml.Engine.Predict()`. Cloud kenaz-ml just needs to load the right tenant's model weights and run inference. No background work.

### 4. Training Pipeline

A K8s CronJob that trains models on synced data.

**Per-user training:**

- Reads events/tasks from tenant's Postgres schema
- Trains the same 5 models (stuck, activity, workflow, duration, quality)
- Same training logic as local (`training/trainer.py`), different data source
- Writes model weights to S3 at `s3://sigil-models/{tenant_id}/{model_name}.joblib`
- Triggered on schedule (e.g., daily) or when enough new completed tasks accumulate

**Aggregate training (Team tier):**

- Pools events from all opted-in tenants
- Trains aggregate models with more data and richer patterns
- Cross-user signals: "projects with this structure tend to get stuck at X"
- Writes to `s3://sigil-models/aggregate/{model_name}.joblib`
- Team-tier users get aggregate model predictions blended with their per-user models

### 5. LLM Proxy

Routes LLM inference requests from the Go daemon to cloud providers.

- Thin proxy — the Go daemon's `inference.Engine` already formats prompts and handles tool calling
- Adds: auth, billing metering, rate limiting, provider failover
- Supported providers: OpenAI, Anthropic (already implemented in `inference/cloud.go`)
- No kenaz-ml involvement — this is purely the Go daemon ↔ LLM provider path

---

## Configuration

### Free Tier (Default)

```toml
[inference]
mode = "local"

[inference.local]
enabled = true
server_bin = "llama-server"
model_path = "~/.cache/sigil/models/qwen2.5-1.5b.gguf"

[ml]
mode = "local"

[ml.local]
enabled = true
server_bin = "kenaz-ml"
```

### Pro Tier (Cloud LLM, Local ML)

```toml
[cloud]
tier = "pro"
api_key = "sk-sigil-..."

[inference]
mode = "remotefirst"   # cloud LLM, fall back to local if offline

[inference.cloud]
enabled = true

[ml]
mode = "local"         # still running kenaz-ml locally

[ml.local]
enabled = true
server_bin = "kenaz-ml"

[cloud.sync]
enabled = false        # no data sync — ML is local
```

### Pro Tier (Cloud LLM + Cloud ML, No Python)

```toml
[cloud]
tier = "pro"
api_key = "sk-sigil-..."

[inference]
mode = "remotefirst"

[inference.cloud]
enabled = true

[ml]
mode = "remote"        # all ML in cloud — no kenaz-ml on laptop

[ml.cloud]
enabled = true

[cloud.sync]
enabled = true         # events sync to cloud for model training
```

### Team Tier (Full Cloud, Data Sync)

```toml
[cloud]
tier = "team"
api_key = "sk-sigil-..."
org_id = "org-..."

[inference]
mode = "remotefirst"

[inference.cloud]
enabled = true

[ml]
mode = "remotefirst"   # cloud ML, fall back to local if offline

[ml.cloud]
enabled = true

[cloud.sync]
enabled = true
```

---

## Data Flow by Tier

### Free: Fully Local

```
events → SQLite → kenaz-ml polls → predictions → SQLite → sigild reads
                                                         → LLM (local) generates suggestions
```

### Pro (Cloud LLM Only)

```
events → SQLite → kenaz-ml polls → predictions → SQLite → sigild reads
                                                         → LLM (cloud) generates suggestions
```

### Pro (Cloud LLM + Cloud ML)

```
events → SQLite → sync agent → cloud Postgres → training pipeline → S3
                                                                      │
sigild → ml.Engine.Predict() → cloud kenaz-ml API ← loads model from S3
                                    │
                                    ▼
                             prediction returned
                                    │
sigild ← ──────────────────────────┘
  │
  └→ inference.Engine → cloud LLM → suggestion
```

### Team (Full Cloud + Aggregate Models)

Same as Pro Cloud ML, plus:

```
cloud Postgres (all opted-in tenants) → aggregate training pipeline → S3
                                                                        │
cloud kenaz-ml API ← loads per-user model + aggregate model ───────────┘
                   → blended prediction
```

---

## kenaz-ml Codebase Changes

### Phase 1: Mode Split

Make the poller optional and support a stateless cloud serving mode.

- Add `--mode local|cloud` flag to `kenaz-ml serve`
- Local mode: unchanged (poller + API on `:7774`)
- Cloud mode: API only, no poller, no SQLite, no cursor tracking
- Cloud mode reads `SIGIL_TENANT_ID` from request headers (set by API gateway)

### Phase 2: Storage Abstraction

Replace direct SQLite calls with a `DataStore` interface.

```python
class DataStore(Protocol):
    def get_events_since(self, cursor: int, limit: int) -> list[dict]: ...
    def get_active_task(self) -> dict | None: ...
    def get_completed_tasks(self, limit: int) -> list[dict]: ...
    def insert_prediction(self, model: str, result: dict, confidence: float, expires_at: int | None) -> None: ...
    def insert_ml_event(self, kind: str, endpoint: str, routing: str, latency_ms: int) -> None: ...
```

- `SqliteStore` — current implementation, used in local mode
- `PostgresStore` — new, used in cloud mode (per-tenant schema)

### Phase 3: Model Storage Abstraction

Replace local filesystem model loading with pluggable storage.

```python
class ModelStore(Protocol):
    def load(self, model_name: str) -> bytes | None: ...
    def save(self, model_name: str, data: bytes) -> None: ...
```

- `LocalModelStore` — reads/writes `~/.local/share/sigild/ml-models/*.joblib`
- `S3ModelStore` — reads/writes `s3://sigil-models/{tenant_id}/*.joblib`

### Phase 4: Multi-Tenant Middleware

FastAPI middleware that extracts tenant context from authenticated requests.

- Reads tenant ID and tier from JWT / API key (set by API gateway)
- Loads correct model weights for tenant
- Model weight caching with TTL (avoid S3 reads on every request)

### Phase 5: Training Pipeline

Separate entrypoint for batch training in K8s.

- `kenaz-ml train --mode cloud --tenant <id>` — per-user training from Postgres
- `kenaz-ml train --mode cloud --aggregate` — aggregate training from pooled data
- Reads from Postgres, writes weights to S3
- Runs as K8s CronJob on schedule

---

## New Dependencies (Cloud Mode Only)

These are only required for the cloud deployment, not the local sidecar:

| Package | Purpose |
|---|---|
| `psycopg[binary]` or `asyncpg` | Postgres access |
| `boto3` or `s3fs` | S3 model storage |
| `pyjwt` | JWT validation for tenant auth |

Local mode dependencies remain unchanged: `fastapi`, `uvicorn`, `scikit-learn`, `joblib`, `numpy`.

---

## K8s Deployment Topology

```yaml
# Prediction API — stateless, autoscaled
Deployment: kenaz-ml-api
  replicas: 2–10 (HPA on CPU/RPS)
  containers:
    - kenaz-ml serve --mode cloud
  resources:
    requests: { cpu: 250m, memory: 512Mi }
    limits:   { cpu: 1, memory: 1Gi }

# Training Pipeline — scheduled batch job
CronJob: kenaz-ml-train
  schedule: "0 */6 * * *"  # every 6 hours
  containers:
    - kenaz-ml train --mode cloud --all-tenants
  resources:
    requests: { cpu: 1, memory: 2Gi }
    # GPU node pool for future transformer models

# Ingest Service — receives event streams
Deployment: sigil-ingest
  replicas: 2–5 (HPA on RPS)
  # Separate service, not part of kenaz-ml

# Postgres
StatefulSet: sigil-postgres
  # Or managed (RDS, Cloud SQL)
  storage: 100Gi per tenant estimated at ~50MB/year

# S3 (model weights)
  # ~5 models × ~50KB each × N tenants
  # Negligible storage cost
```

---

## Open Questions

1. **Aggregate model architecture** — simple pooled training, or federated learning where raw events never leave tenant boundaries?
2. **Model upgrade path** — when do we move from scikit-learn to heavier models (transformers, sequence models) for cloud users? What's the trigger?
3. **Offline fallback for cloud ML users** — if the laptop goes offline, should we bundle a lightweight fallback model in the Go binary itself (rule-based, no Python)?
4. **Billing metering** — per-prediction, per-event-synced, or flat monthly?
5. **Data retention** — how long do we keep synced events in cloud Postgres? Configurable per tier?
6. **Model versioning** — how do we roll out new model versions without disrupting predictions? Blue/green model deployments?
