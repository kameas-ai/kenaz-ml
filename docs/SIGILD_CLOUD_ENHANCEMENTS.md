# Sigild Cloud Enhancements

These 4 features belong in the [`sigil`](https://github.com/wambozi/sigil) Go repository. They are the sigild-side counterparts to the kenaz-ml cloud features and are required for a complete Sigil Cloud deployment.

## 1. LLM Proxy Service

**What**: A cloud-hosted service that proxies LLM inference requests from Go daemons on developer laptops to cloud LLM providers (OpenAI, Anthropic).

**Why**: This is the fastest path to revenue. Pro-tier users get dramatically better suggestions (GPT-4o / Claude vs local Qwen 1.5B) with zero changes to kenaz-ml. The Go daemon's `inference.Engine` already supports cloud backends — this feature wraps them in a managed service with auth and billing.

**Scope**:
- API gateway with authentication (API key / JWT)
- Tier enforcement (Pro and Team only)
- Request metering for billing
- Rate limiting per tenant
- Provider failover (OpenAI → Anthropic or vice versa)
- No kenaz-ml involvement — purely the Go daemon ↔ LLM provider path

**Existing Go code to build on**:
- `internal/inference/cloud.go` — already supports OpenAI and Anthropic APIs
- `internal/inference/engine.go` — already has routing modes (local, localfirst, remotefirst, remote)

**Dependencies**: None. Can ship independently as the first paid feature.

---

## 2. Sync Agent

**What**: A goroutine inside `sigild` that streams local SQLite changes (events, tasks, suggestions, predictions) to the Sigil Cloud ingest API via HTTPS.

**Why**: This is the data flywheel. Without event sync, cloud ML has nothing to train on. Pro users who opt into cloud ML consent to syncing their workflow events to the cloud.

**Scope**:
- CDC (change data capture) from SQLite — tracks per-table sync cursors (last synced row ID)
- Polls SQLite on a 5-second interval, batches rows, ships via HTTPS
- Tables synced: `events`, `tasks`, `suggestions`, `ml_predictions`, `ml_events`, `patterns`
- Tables NOT synced: `ml_cursor` (local bookkeeping)
- Exponential backoff retry on cloud unreachability — never lose events
- Only starts when `cloud.sync.enabled = true` and a valid API key is present
- mTLS for transport security
- Bandwidth estimate: ~500–2000 events/hour × ~200 bytes = <400 KB/hour

**Existing Go code to build on**:
- `internal/store/` — all table access is already in the store package
- `internal/network/` — TLS and credential management
- `internal/config/` — TOML config parsing

**Configuration**:
```toml
[cloud.sync]
enabled = true
api_url = "https://ingest.sigil.cloud"
batch_size = 100
poll_interval_sec = 5
```

**Dependencies**: Requires Cloud Ingest Service (feature 3) to receive the data.

---

## 3. Cloud Ingest Service

**What**: A cloud-hosted HTTP service that receives event streams from sync agents on developer laptops and writes them to per-tenant schemas in Postgres.

**Why**: This is the cloud-side receiver for synced data. It provides the data foundation that cloud ML training and team dashboards build on.

**Scope**:
- HTTP endpoint for receiving batched event/task/suggestion data
- Authentication via API key in request header
- Tenant identification and tier validation (must be Pro with cloud ML opt-in, or Team)
- Writes to per-tenant schema in Postgres
- Schema mirrors SQLite table structure for maximum compatibility with kenaz-ml
- Idempotent writes using sync cursor + event ID deduplication
- Deployed in K8s, horizontally scalable on request rate

**Data model**:
```
Postgres cluster
├── tenant_{id}_001/
│   ├── events          (mirrors SQLite events table)
│   ├── tasks           (mirrors SQLite tasks table)
│   ├── suggestions     (mirrors SQLite suggestions table)
│   ├── ml_predictions  (mirrors SQLite ml_predictions table)
│   ├── ml_events       (mirrors SQLite ml_events table)
│   ├── patterns        (mirrors SQLite patterns table)
│   └── sync_cursor     (tracks last received row per source table)
├── tenant_{id}_002/
│   └── ...
```

**Dependencies**: None for the service itself. The Sync Agent (feature 2) is the primary producer.

---

## 4. Tier & Configuration System

**What**: Extend sigild's configuration to support cloud tiers (Free, Pro, Team) with mode selection for inference and ML engines, cloud API key management, and sync opt-in controls.

**Why**: The Go daemon needs to know what tier the user is on and what cloud features they've enabled. This drives routing decisions (local vs cloud for inference and ML), sync agent activation, and feature gating.

**Scope**:
- Add `[cloud]` config section: `tier`, `api_key`, `org_id` (Team)
- Add `[cloud.sync]` config section: `enabled`, `api_url`, `batch_size`, `poll_interval_sec`
- `sigil auth login` CLI command — authenticates with Sigil Cloud, stores API key
- `sigil auth status` CLI command — shows current tier and enabled features
- Tier-aware feature gates in the daemon startup:
  - Free: inference.mode defaults to "local", ml.mode defaults to "local", sync disabled
  - Pro: inference.mode defaults to "remotefirst", ml.mode defaults to "local", sync disabled by default
  - Team: inference.mode defaults to "remotefirst", ml.mode defaults to "remotefirst", sync enabled by default
- Users can override any default — tiers set defaults, not hard limits
- Validate API key on startup (warn if expired or invalid, don't block daemon)

**Existing Go code to build on**:
- `internal/config/` — TOML config with inference and ML sections already present
- `internal/inference/engine.go` — routing modes already implemented
- `internal/ml/ml.go` — routing modes already implemented
- `cmd/sigilctl/` — CLI for daemon control

**Configuration example (Pro, cloud LLM + cloud ML)**:
```toml
[cloud]
tier = "pro"
api_key = "sk-sigil-..."

[cloud.sync]
enabled = true

[inference]
mode = "remotefirst"

[ml]
mode = "remote"
```

**Dependencies**: LLM Proxy Service (feature 1) for cloud inference. Sync Agent (feature 2) + Ingest Service (feature 3) for data sync.

---

## Suggested Implementation Order

```
1. LLM Proxy Service      ← ships Pro tier, generates revenue, no ML changes needed
2. Tier & Config System    ← supports all subsequent features
3. Sync Agent              ← enables the data flywheel
4. Cloud Ingest Service    ← receives synced data into Postgres
```

Features 3 and 4 are tightly coupled and should be developed together.

## Relationship to kenaz-ml Features

| sigild Feature | kenaz-ml Feature | Relationship |
|---|---|---|
| LLM Proxy Service | — | Independent, no kenaz-ml changes |
| Sync Agent | 002 Storage Abstraction | Sync agent feeds data that PostgresStore reads |
| Cloud Ingest Service | 002 Storage Abstraction | Ingest writes to Postgres that PostgresStore reads |
| Tier & Config System | 001 Cloud Serving Mode | Tier config determines `ml.mode` which drives kenaz-ml mode |
