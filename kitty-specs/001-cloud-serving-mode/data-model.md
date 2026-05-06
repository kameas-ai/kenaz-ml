# Data Model: Cloud Serving Mode

**Feature**: 001-cloud-serving-mode
**Date**: 2026-03-30

## Entities

### ServingMode (Enum)

New enum representing the operating mode of the kenaz-ml service.

| Value   | Description                                      |
|---------|--------------------------------------------------|
| `LOCAL` | Default. Poller, SQLite, local models. Current behavior. |
| `CLOUD` | Stateless. No poller, no SQLite, tenant-aware model loading. |

**Location**: `src/sigil_ml/config.py`
**Type**: `enum.Enum` (str mixin for JSON serialization)

### TenantContext (Dataclass)

Per-request context extracted from the HTTP request in cloud mode.

| Field       | Type  | Default     | Description                        |
|-------------|-------|-------------|------------------------------------|
| `tenant_id` | `str` | (required)  | Tenant identifier from `X-Tenant-ID` header |
| `tier`      | `str` | `"default"` | Tenant tier (reserved for future rate limiting) |

**Location**: `src/sigil_ml/tenant.py`
**Constraints**:
- In cloud mode, `tenant_id` must be non-empty (enforced by FastAPI dependency).
- In local mode, a sentinel `TenantContext(tenant_id="local", tier="local")` is used; tenant-aware code paths are skipped.

### ModelCacheEntry (Internal)

Internal cache entry stored by `ModelCache`.

| Field        | Type    | Description                              |
|--------------|---------|------------------------------------------|
| `model`      | `Any`   | The loaded model object (e.g., `StuckPredictor`) |
| `loaded_at`  | `float` | `time.monotonic()` timestamp of when the entry was created |

**Location**: `src/sigil_ml/cache.py` (private to `ModelCache`)

### ModelCache

In-memory LRU cache of loaded model weights.

| Field          | Type                                      | Description                        |
|----------------|-------------------------------------------|------------------------------------|
| `_entries`     | `dict[tuple[str, str], ModelCacheEntry]`  | Cache storage keyed by `(tenant_id, model_name)` |
| `_ttl_seconds` | `float`                                   | TTL in seconds (from `MODEL_CACHE_TTL_SECONDS` env var, default 300) |
| `_max_size`    | `int`                                     | Maximum entries before LRU eviction (default 100) |
| `_lock`        | `threading.Lock`                          | Thread-safety lock |
| `_stats`       | `CacheStats`                              | Hit/miss/eviction counters |

**Location**: `src/sigil_ml/cache.py`

**Operations**:
- `get(tenant_id, model_name) -> Any | None` -- Returns model if cached and not expired; `None` otherwise.
- `put(tenant_id, model_name, model) -> None` -- Stores model; evicts LRU if at capacity.
- `evict(tenant_id) -> None` -- Removes all entries for a tenant.
- `evict_all() -> None` -- Clears entire cache.
- `stats() -> dict` -- Returns hit/miss/eviction counts.

### CacheStats (Dataclass)

| Field       | Type  | Default | Description                   |
|-------------|-------|---------|-------------------------------|
| `hits`      | `int` | `0`     | Number of cache hits          |
| `misses`    | `int` | `0`     | Number of cache misses        |
| `evictions` | `int` | `0`     | Number of TTL/LRU evictions   |

**Location**: `src/sigil_ml/cache.py`

### ModelLoader (Protocol)

Protocol defining how model weights are loaded from storage.

| Method | Signature                                        | Description                     |
|--------|--------------------------------------------------|---------------------------------|
| `load` | `(tenant_id: str, model_name: str) -> Any | None` | Load model weights; return `None` if not found |

**Location**: `src/sigil_ml/loader.py`

### FilesystemModelLoader

Concrete implementation of `ModelLoader` for local filesystem.

| Field        | Type   | Description                                |
|--------------|--------|--------------------------------------------|
| `models_dir` | `Path` | Base directory for model weights           |

**Location**: `src/sigil_ml/loader.py`
**Storage layout**: `{models_dir}/{tenant_id}/{model_name}.joblib`

## Entity Relationships

```
ServingMode ──selects──> Startup behavior (poller vs stateless)
                         │
                         ├── LOCAL: AppState.load_models() (existing path)
                         └── CLOUD: ModelCache + ModelLoader
                                     │
TenantContext ──routes──> ModelCache.get(tenant_id, model_name)
                                     │
                                     ├── HIT: return cached model
                                     └── MISS: ModelLoader.load(tenant_id, model_name)
                                               │
                                               ├── FOUND: cache + return
                                               └── NOT FOUND: rule-based fallback
```

## Modified Entities

### AppState (Existing - Modified)

New fields added for cloud mode:

| Field          | Type                   | Description                          |
|----------------|------------------------|--------------------------------------|
| `mode`         | `ServingMode`          | Current operating mode               |
| `model_cache`  | `ModelCache | None`    | Set in cloud mode only               |
| `model_loader` | `ModelLoader | None`   | Set in cloud mode only               |

### HealthResponse (Existing - Modified)

New optional field:

| Field  | Type          | Default  | Description                      |
|--------|---------------|----------|----------------------------------|
| `mode` | `str | None`  | `None`   | `"local"` or `"cloud"` if present |

## State Transitions

### ServingMode Selection (Startup)

```
CLI parse --mode flag
    │
    ├── --mode cloud  ──> ServingMode.CLOUD
    ├── --mode local  ──> ServingMode.LOCAL
    └── (no flag)     ──> check SIGIL_ML_MODE env var
                           │
                           ├── "cloud" ──> ServingMode.CLOUD
                           └── else    ──> ServingMode.LOCAL (default)
```

### Request Flow (Cloud Mode)

```
Request arrives
    │
    ├── Extract X-Tenant-ID header
    │     ├── Missing ──> 401 Unauthorized
    │     └── Present ──> TenantContext(tenant_id=<value>)
    │
    ├── Resolve model via cache
    │     ├── Cache HIT (not expired) ──> use cached model
    │     ├── Cache MISS ──> ModelLoader.load()
    │     │     ├── Model found ──> cache it, use it
    │     │     └── Model not found ──> rule-based fallback
    │     └── Cache EXPIRED ──> evict, treat as MISS
    │
    └── Run prediction ──> return response
```
