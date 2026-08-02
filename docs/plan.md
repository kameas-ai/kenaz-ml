# Implementation Plan: kenaz-ml Cloud Enhancements

**Date**: 2026-03-29
**Specs**: `kitty-specs/001` through `kitty-specs/004`
**Architecture**: `docs/SIGIL_CLOUD_ARCHITECTURE.md`

## Summary

This plan covers the four features needed to run kenaz-ml in the cloud as a stateless prediction API and batch training pipeline, while preserving the existing local-first experience unchanged.

The features, in dependency order:

1. **002 - Storage Abstraction**: DataStore protocol decoupling all data access from SQLite
2. **003 - Model Storage Abstraction**: ModelStore protocol decoupling model weight persistence from the filesystem
3. **001 - Cloud Serving Mode**: `--mode cloud` flag for stateless K8s serving
4. **004 - Cloud Training Pipeline**: `kenaz-ml train --mode cloud` for K8s CronJob execution

Features 002 and 003 are foundational abstractions with no cloud dependencies. They can be built and tested entirely in local mode. Features 001 and 004 build on top, adding cloud-specific wiring.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: FastAPI, uvicorn, scikit-learn, numpy, joblib (existing). New cloud-only: psycopg2-binary, boto3, pyjwt (optional via `[cloud]` extra).
**Storage**: SQLite (local, existing), Postgres (cloud, new), S3 (cloud model weights, new)
**Testing**: pytest
**Target Platform**: Linux/macOS laptop (local), K8s pod (cloud)
**Performance Goals**: <50ms prediction cycle, <2s cold start in cloud mode, 50 RPS per replica
**Constraints**: No new required dependencies for local installs. Cloud deps are optional extras.

## Architecture Decisions

### AD-1: Optional dependency extras

Cloud dependencies are installed via `pip install kenaz-ml[cloud]`. The base install remains lightweight.

```toml
[project.optional-dependencies]
cloud = ["psycopg2-binary>=2.9", "boto3>=1.34", "pyjwt>=2.8"]
dev = ["pytest>=8.0", "httpx>=0.27", "ruff>=0.4", "pyre-check>=0.9.18"]
```

Import guards at module level:

```python
# src/kenaz_ml/storage/postgres.py
try:
    import psycopg2
except ImportError:
    psycopg2 = None  # type: ignore

class PostgresStore:
    def __init__(self, dsn: str):
        if psycopg2 is None:
            raise ImportError("psycopg2-binary is required for cloud mode: pip install kenaz-ml[cloud]")
        ...
```

### AD-2: DataStore protocol location and shape

The protocol lives at `src/kenaz_ml/storage/protocol.py`. It defines the minimal interface that the poller, routes, trainer, and scheduler currently use against SQLite.

```python
# src/kenaz_ml/storage/protocol.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class DataStore(Protocol):
    """Abstraction over event/prediction data access.

    SQLite implementation for local mode. Postgres implementation for cloud mode.
    Python only writes to ml_predictions, ml_events, ml_cursor.
    Python only reads from events, tasks, patterns, suggestions.
    """

    def get_events_since(self, cursor: int, limit: int = 100) -> list[dict]:
        """Read events with id > cursor, ordered by id ascending."""
        ...

    def get_active_task(self) -> dict | None:
        """Return the currently active (non-idle, non-completed) task, or None."""
        ...

    def get_completed_tasks(self, limit: int = 100) -> list[dict]:
        """Return completed tasks ordered by completed_at descending."""
        ...

    def get_task(self, task_id: str) -> dict | None:
        """Return a single task by ID."""
        ...

    def get_events_for_task(self, task_id: str, since: int | None = None) -> list[dict]:
        """Return events within a task's time window."""
        ...

    def get_cursor(self) -> int:
        """Return the current poll cursor (last_event_id)."""
        ...

    def set_cursor(self, event_id: int) -> None:
        """Update the poll cursor."""
        ...

    def insert_prediction(
        self, model: str, result: dict, confidence: float, expires_at: int | None
    ) -> None:
        """Write a prediction row to ml_predictions."""
        ...

    def insert_ml_event(
        self, kind: str, endpoint: str, routing: str, latency_ms: int
    ) -> None:
        """Write an audit row to ml_events."""
        ...

    def get_latest_predictions(self) -> list[dict]:
        """Return non-expired predictions for /status."""
        ...

    def get_latest_completed_task_stats(self) -> dict | None:
        """Return test_runs, test_fails, commit_count from most recent completed task."""
        ...

    def count_completed_tasks(self) -> int:
        """Count completed tasks (for training scheduler)."""
        ...
```

### AD-3: Postgres connection strategy

Use a simple connection factory -- no pool. Each method opens a connection, runs its query, and closes. This matches the existing SQLite pattern (open, query, close in `_connect()`). Pooling can be added later as a performance optimization.

```python
class PostgresStore:
    def __init__(self, dsn: str, tenant_id: str | None = None):
        self._dsn = dsn
        self._tenant_id = tenant_id  # None for local-like single-tenant usage

    def _connect(self):
        conn = psycopg2.connect(self._dsn)
        conn.autocommit = False
        return conn
```

For multi-tenant cloud mode, the `tenant_id` is used to set the search_path:

```python
def _connect(self):
    conn = psycopg2.connect(self._dsn)
    if self._tenant_id:
        with conn.cursor() as cur:
            cur.execute("SET search_path TO %s, public", (f"tenant_{self._tenant_id}",))
    return conn
```

### AD-4: ModelStore protocol shape

```python
# src/kenaz_ml/storage/model_store.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class ModelStore(Protocol):
    """Abstraction over model weight persistence."""

    def load(self, model_name: str) -> bytes | None:
        """Load serialized model weights. Returns None if not found."""
        ...

    def save(self, model_name: str, data: bytes) -> None:
        """Persist serialized model weights."""
        ...
```

### AD-5: Config via environment variables for cloud mode

Cloud config uses environment variables. Local mode uses existing TOML/path-based config unchanged.

| Variable | Purpose | Default |
|---|---|---|
| `KENAZ_ML_MODE` | `local` or `cloud` | `local` |
| `DATABASE_URL` | Postgres DSN for cloud mode | None |
| `KENAZ_MODEL_BUCKET` | S3 bucket for model weights | None |
| `KENAZ_MODEL_REGION` | AWS region for S3 | `us-east-1` |
| `KENAZ_MODEL_ENDPOINT` | S3-compatible endpoint (MinIO) | None |
| `KENAZ_TENANT_HEADER` | Request header name for tenant ID | `X-Sigil-Tenant-Id` |
| `KENAZ_MODEL_CACHE_TTL` | Model cache TTL in seconds | `300` |

### AD-6: No ORM, raw SQL only

Both SQLite and Postgres implementations use raw SQL. The Go side owns all DDL. Python only reads Go-owned tables and writes to `ml_*` tables. No schema migrations from Python.

### AD-7: Mode-based app factory

The `create_app()` function accepts a mode parameter that controls which components are initialized:

```python
def create_app(mode: str = "local") -> FastAPI:
    if mode == "cloud":
        # No poller, no SQLite, no training scheduler
        # Initialize from env vars: DATABASE_URL, S3 config
        # Add tenant middleware
        pass
    else:
        # Current behavior: SQLite, poller, training scheduler
        pass
```

## Constitution Check

| Principle | Compliance | Notes |
|---|---|---|
| Minimal dependencies | PASS | Cloud deps are optional extras, not required for base install |
| pytest for tests | PASS | All new tests use pytest |
| Ruff for linting | PASS | Existing ruff config applies to new code |
| No data leaves machine (local) | PASS | Local mode is completely unchanged |
| Cross-platform | PASS | psycopg2-binary provides platform wheels; boto3 is pure Python |
| No GPU required | PASS | Same scikit-learn models |
| Simplicity over complexity | PASS | Simple connection factory, raw SQL, no ORM, no pool |

## Project Structure

### New Source Files

```
src/kenaz_ml/
  storage/
    __init__.py              # Exports DataStore, ModelStore, factory functions
    protocol.py              # DataStore protocol definition
    sqlite_store.py          # SqliteStore - extracted from current direct SQLite calls
    postgres_store.py        # PostgresStore - new, cloud mode
    model_store.py           # ModelStore protocol definition
    local_model_store.py     # LocalModelStore - extracted from current joblib load/save
    s3_model_store.py        # S3ModelStore - new, cloud mode
    model_cache.py           # ModelCache - TTL cache wrapping any ModelStore
  middleware/
    __init__.py
    tenant.py                # TenantContext extraction from request headers
  cloud_config.py            # Environment variable parsing for cloud mode

# Modified files:
  app.py                     # Mode-aware app factory
  cli.py                     # --mode flag on serve and train commands
  config.py                  # Add cloud config helpers
  poller.py                  # Accept DataStore instead of db_path
  routes.py                  # Accept DataStore instead of direct SQLite
  features.py                # Accept DataStore instead of db_path for feature extraction
  training/trainer.py        # Accept DataStore + ModelStore
  training/scheduler.py      # Accept DataStore instead of db_path
  models/stuck.py            # Accept ModelStore for weight persistence
  models/activity.py         # Accept ModelStore for weight persistence
  models/workflow.py         # Accept ModelStore for weight persistence
  models/duration.py         # Accept ModelStore for weight persistence
  models/quality.py          # Accept ModelStore for weight persistence
  schema.py                  # Wrapped into SqliteStore.ensure_tables()

# New tests:
  tests/test_storage.py      # DataStore protocol, SqliteStore, mock store
  tests/test_model_store.py  # ModelStore protocol, LocalModelStore, mock store
  tests/test_cloud_app.py    # Cloud mode app startup, tenant middleware
  tests/test_cloud_train.py  # Cloud training pipeline
```

### Documentation (kitty-specs)

Each feature spec already exists. This plan is the cross-feature implementation guide.

## Implementation Phases

### Phase 1: Storage Abstraction (Feature 002)

**Goal**: Replace all direct SQLite access with the DataStore protocol. Zero behavior change for local users.

**Why first**: This is the deepest refactor -- it touches poller, routes, features, trainer, and scheduler. Everything else builds on it. If done correctly, cloud mode is mostly wiring.

#### Step 1.1: Define DataStore protocol and create SqliteStore

Create `src/kenaz_ml/storage/protocol.py` with the DataStore protocol (see AD-2 above).

Create `src/kenaz_ml/storage/sqlite_store.py` by extracting all SQLite operations currently spread across:
- `poller.py`: `_connect()`, `_poll_once()` (cursor read/write, event queries, prediction inserts, ml_event inserts, task queries)
- `routes.py`: `/status` endpoint (cursor query, prediction query)
- `features.py`: `_query_task()`, `_query_events_for_task()` (task/event reads)
- `training/trainer.py`: task queries for training data
- `training/scheduler.py`: completed task count, ml_event logging
- `schema.py`: `ensure_ml_tables()` (cursor table creation)

The SqliteStore wraps all of these behind the DataStore interface. Every connection opens with WAL mode and busy_timeout=5000.

**Key mapping from current code to DataStore methods**:

| Current code location | Current pattern | DataStore method |
|---|---|---|
| `poller._poll_once()` line 65-66 | `SELECT last_event_id FROM ml_cursor` | `get_cursor()` |
| `poller._poll_once()` line 68-71 | `SELECT ... FROM events WHERE id > ?` | `get_events_since()` |
| `poller._poll_once()` line 97-99 | `UPDATE ml_cursor SET last_event_id = ?` | `set_cursor()` |
| `poller._write()` line 234-247 | `INSERT INTO ml_predictions` | `insert_prediction()` |
| `poller._predict_and_write()` line 179-183 | `INSERT INTO ml_events` | `insert_ml_event()` |
| `poller._active_task_id()` line 249-252 | `SELECT id FROM tasks WHERE ...` | `get_active_task()` |
| `poller._session_info()` line 218-219 | `SELECT started_at, phase, test_fails FROM tasks` | `get_task()` |
| `poller._quality_features()` line 272-279 | `SELECT test_runs, test_fails, commit_count FROM tasks` | `get_latest_completed_task_stats()` |
| `routes.status()` line 128-133 | Direct cursor/prediction queries | `get_cursor()`, `get_latest_predictions()` |
| `features._query_task()` | `SELECT * FROM tasks WHERE id = ?` | `get_task()` |
| `features._query_events_for_task()` | `SELECT * FROM events WHERE ts >= ? AND ts <= ?` | `get_events_for_task()` |
| `trainer._train_stuck()` line 66 | `SELECT id FROM tasks WHERE completed_at IS NOT NULL` | `get_completed_tasks()` |
| `scheduler._count_completed()` | `SELECT COUNT(*) FROM tasks WHERE completed_at IS NOT NULL` | `count_completed_tasks()` |
| `scheduler._log_retrain()` | `INSERT INTO ml_events` | `insert_ml_event()` |
| `schema.ensure_ml_tables()` | `CREATE TABLE IF NOT EXISTS ml_cursor` | `SqliteStore.ensure_tables()` (init-time) |

**Tests**: Write `tests/test_storage.py` with:
- A `MockStore` implementing DataStore for unit tests
- SqliteStore tests against a temporary SQLite database
- Verify all DataStore methods return correct types
- Verify WAL mode and busy_timeout are set on every SqliteStore connection

#### Step 1.2: Refactor poller to use DataStore

Change `EventPoller.__init__` to accept a `DataStore` instead of `db_path: Path`.

Current signature:
```python
def __init__(self, db_path: Path, models: dict) -> None:
```

New signature:
```python
def __init__(self, store: DataStore, models: dict) -> None:
```

Replace all `self._connect()` + raw SQL in `_poll_once()`, `_predict_and_write()`, `_write()`, `_active_task_id()`, `_session_info()`, `_quality_features()` with calls to `self.store.method()`.

The `_connect()` method is removed entirely from the poller. The poller no longer imports `sqlite3`.

**Transaction handling**: The current poller opens one connection per `_poll_once()` and commits at the end. The DataStore's SqliteStore can maintain a context-manager pattern or the poller can call individual methods that each handle their own connection. Given the current pattern (single connection per poll cycle with multiple reads and writes), the simplest approach is:

- Each DataStore method manages its own connection/transaction for reads.
- For the write-heavy `_predict_and_write()` path, the poller calls individual `insert_prediction()` and `insert_ml_event()` methods. Each is its own micro-transaction. This is acceptable because:
  - Predictions are idempotent (Go reads latest by created_at)
  - A partial write (some predictions written, crash before ml_event) is benign
  - This avoids exposing transaction primitives in the protocol

**Tests**: Update existing `tests/test_models.py` and `tests/test_server.py` to use MockStore where they currently create SQLite databases.

#### Step 1.3: Refactor routes and features to use DataStore

**routes.py**: The `/status` endpoint currently opens its own SQLite connection. Refactor `register_routes()` to accept a DataStore via AppState, and use `state.store.get_cursor()` and `state.store.get_latest_predictions()`.

The `/predict/stuck` and `/predict/duration` endpoints call `extract_stuck_features(config.db_path(), task_id)` and `extract_duration_features(config.db_path(), task_id)`. These functions in `features.py` take a `db_path` and open their own SQLite connections.

Refactor `features.py`:
- `extract_stuck_features(store: DataStore, task_id: str)` -- uses `store.get_task()` and `store.get_events_for_task()` instead of `_query_task(db_path, ...)` and `_query_events_for_task(db_path, ...)`
- `extract_duration_features(store: DataStore, task_id: str)` -- same pattern
- Remove `_query_task()` and `_query_events_for_task()` private functions (their logic moves to SqliteStore)

**Training**: Refactor `Trainer.__init__` to accept a DataStore. Replace direct SQLite queries in `_train_stuck()` and `_train_duration()` with DataStore calls.

Refactor `TrainingScheduler.__init__` to accept a DataStore. Replace `_count_completed()` and `_log_retrain()` with DataStore calls.

#### Step 1.4: Create PostgresStore

Create `src/kenaz_ml/storage/postgres_store.py` implementing DataStore.

Key differences from SqliteStore:
- Uses `psycopg2.connect(dsn)` instead of `sqlite3.connect(path)`
- Uses `%s` parameter style instead of `?`
- Supports per-tenant schema via `SET search_path TO tenant_{id}`
- No WAL mode or busy_timeout (Postgres handles concurrency natively)
- The `ensure_tables()` method is a no-op -- Go/ingest service owns DDL

**Tests**: Integration tests in `tests/test_storage.py` that require a Postgres instance. Mark with `@pytest.mark.postgres` so they're skipped in CI without a Postgres container. Use environment variable `TEST_DATABASE_URL` to configure.

#### Step 1.5: Wire DataStore into app.py

Update `create_app()` to construct the appropriate DataStore based on mode:

```python
def create_app(mode: str = "local") -> FastAPI:
    state = AppState()
    if mode == "cloud":
        dsn = os.environ.get("DATABASE_URL")
        if not dsn:
            raise RuntimeError("DATABASE_URL required in cloud mode")
        store = PostgresStore(dsn=dsn)
    else:
        db = config.db_path()
        store = SqliteStore(db)
        store.ensure_tables()

    state.store = store
    ...
```

**Verification**: Run `pytest tests/` -- all existing tests must pass with zero modifications to test code (they use the SQLite backend by default).

---

### Phase 2: Model Storage Abstraction (Feature 003)

**Goal**: Replace direct `joblib.load/dump` and `config.weights_path()` calls in model classes with a pluggable ModelStore interface.

#### Step 2.1: Define ModelStore protocol and create LocalModelStore

Create `src/kenaz_ml/storage/model_store.py` with the ModelStore protocol (see AD-4 above).

Create `src/kenaz_ml/storage/local_model_store.py`:

```python
class LocalModelStore:
    """Loads/saves model weights from the local filesystem (joblib files)."""

    def __init__(self, models_dir: Path | None = None):
        self._dir = models_dir or config.models_dir()

    def load(self, model_name: str) -> bytes | None:
        path = self._dir / f"{model_name}.joblib"
        if not path.exists():
            return None
        return path.read_bytes()

    def save(self, model_name: str, data: bytes) -> None:
        path = self._dir / f"{model_name}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
```

Note: The models currently use `joblib.load(path)` directly which deserializes the sklearn object. With the ModelStore abstraction returning raw bytes, the model classes will use `joblib.load(BytesIO(data))` and `joblib.dump(model, buffer); store.save(name, buffer.getvalue())`.

#### Step 2.2: Refactor model classes to use ModelStore

Each model class (`StuckPredictor`, `ActivityClassifier`, `WorkflowStatePredictor`, `DurationEstimator`, `QualityEstimator`) currently:
1. In `__init__`, calls `config.weights_path(name)` and `joblib.load(path)` if the file exists
2. In `train()` or `_save()`, calls `joblib.dump(model, path)`

Refactor to:
1. In `__init__`, accept a `ModelStore` parameter. Call `store.load(name)` and deserialize with `joblib.load(BytesIO(data))` if data is not None.
2. In `train()` / `_save()`, serialize to bytes with `BytesIO` + `joblib.dump`, then call `store.save(name, bytes)`.

For `QualityEstimator`, which uses `json.load/dump` instead of joblib, the same pattern applies but with `json.loads(data.decode())` and `store.save(name, json.dumps(...).encode())`.

The model classes will no longer import `config.weights_path()` or directly access the filesystem.

Signature changes:

```python
# Before:
class StuckPredictor:
    def __init__(self) -> None:

# After:
class StuckPredictor:
    def __init__(self, model_store: ModelStore | None = None) -> None:
```

The `model_store` parameter defaults to `None` for backward compatibility during the transition. If None, fall back to `LocalModelStore()` to avoid breaking existing code that creates models without a store.

#### Step 2.3: Create S3ModelStore

Create `src/kenaz_ml/storage/s3_model_store.py`:

```python
class S3ModelStore:
    """Loads/saves model weights from S3-compatible object storage."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",  # e.g., "tenant_abc123/"
        region: str = "us-east-1",
        endpoint_url: str | None = None,
    ):
        if boto3 is None:
            raise ImportError("boto3 required for cloud mode: pip install kenaz-ml[cloud]")
        self._bucket = bucket
        self._prefix = prefix
        kwargs = {"region_name": region}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        self._s3 = boto3.client("s3", **kwargs)

    def load(self, model_name: str) -> bytes | None:
        key = f"{self._prefix}{model_name}.joblib"
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
            return resp["Body"].read()
        except self._s3.exceptions.NoSuchKey:
            return None

    def save(self, model_name: str, data: bytes) -> None:
        key = f"{self._prefix}{model_name}.joblib"
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=data)
```

#### Step 2.4: Create ModelCache

Create `src/kenaz_ml/storage/model_cache.py`:

```python
class ModelCache:
    """In-memory TTL cache wrapping any ModelStore.

    Used in cloud mode to avoid re-downloading from S3 on every request.
    """

    def __init__(self, backend: ModelStore, ttl_sec: int = 300):
        self._backend = backend
        self._ttl = ttl_sec
        self._cache: dict[str, tuple[bytes, float]] = {}  # key -> (data, loaded_at)

    def load(self, model_name: str) -> bytes | None:
        if model_name in self._cache:
            data, loaded_at = self._cache[model_name]
            if time.time() - loaded_at < self._ttl:
                return data
            del self._cache[model_name]

        data = self._backend.load(model_name)
        if data is not None:
            self._cache[model_name] = (data, time.time())
        return data

    def save(self, model_name: str, data: bytes) -> None:
        self._backend.save(model_name, data)
        self._cache[model_name] = (data, time.time())
```

For multi-tenant usage, the cache key includes tenant ID. The simplest approach: create one `ModelCache(S3ModelStore(bucket, prefix=f"tenant_{tid}/"))` per tenant, stored in a dict on AppState.

#### Step 2.5: Wire ModelStore into app.py

Update `create_app()` and `AppState`:

```python
class AppState:
    def __init__(self) -> None:
        ...
        self.model_store: ModelStore | None = None

    def load_models(self) -> None:
        store = self.model_store or LocalModelStore()
        self.stuck = StuckPredictor(model_store=store)
        self.activity = ActivityClassifier(model_store=store)
        ...
```

**Verification**: `pytest tests/` passes with no changes to test code. Models use LocalModelStore by default.

---

### Phase 3: Cloud Serving Mode (Feature 001)

**Goal**: Add `--mode cloud` to `kenaz-ml serve` for stateless K8s deployment.

This phase is mostly wiring -- Phases 1 and 2 did the hard refactoring.

#### Step 3.1: Add --mode flag to CLI

Update `src/kenaz_ml/cli.py`:

```python
serve_parser.add_argument("--mode", choices=["local", "cloud"], default="local")
```

Pass the mode to `create_app()`. For cloud mode, construct the app differently:
- Set `KENAZ_ML_MODE` environment variable or pass through
- uvicorn runs `kenaz_ml.app:app` -- need a way to pass mode

Two approaches:
1. Environment variable: `KENAZ_ML_MODE=cloud` is read by the module-level `app = create_app()`.
2. Factory pattern: Use uvicorn's `factory=True` option.

Go with approach 1 (environment variable) because it's simpler and works with the existing module-level `app` instance pattern:

```python
# cli.py
if args.command == "serve":
    os.environ["KENAZ_ML_MODE"] = args.mode
    uvicorn.run(...)

# app.py
app = create_app(mode=os.environ.get("KENAZ_ML_MODE", "local"))
```

#### Step 3.2: Mode-aware create_app()

The full mode-aware app factory:

```python
def create_app(mode: str = "local") -> FastAPI:
    application = FastAPI(title="kenaz-ml", version="0.1.0")
    state = AppState()
    state.mode = mode

    if mode == "cloud":
        # Data store: Postgres or None (cloud serving can be stateless)
        dsn = os.environ.get("DATABASE_URL")
        state.store = PostgresStore(dsn=dsn) if dsn else None

        # Model store: S3 with caching
        bucket = os.environ.get("KENAZ_MODEL_BUCKET")
        if not bucket:
            raise RuntimeError("KENAZ_MODEL_BUCKET required in cloud mode")
        region = os.environ.get("KENAZ_MODEL_REGION", "us-east-1")
        endpoint = os.environ.get("KENAZ_MODEL_ENDPOINT")
        ttl = int(os.environ.get("KENAZ_MODEL_CACHE_TTL", "300"))
        s3_store = S3ModelStore(bucket=bucket, region=region, endpoint_url=endpoint)
        state.model_store = ModelCache(s3_store, ttl_sec=ttl)

        # Tenant middleware
        application.add_middleware(TenantMiddleware)

    else:
        # Local mode: current behavior
        db = config.db_path()
        store = SqliteStore(db)
        store.ensure_tables()
        state.store = store
        state.model_store = LocalModelStore()

    register_routes(application, state)

    @application.on_event("startup")
    async def startup_event():
        if mode == "local":
            state.load_models()
            state.poller = EventPoller(store=state.store, models={...})
            asyncio.create_task(state.poller.run())
            scheduler = TrainingScheduler(state.store, ...)
            asyncio.create_task(_schedule_loop(scheduler))
        else:
            # Cloud mode: no poller, no scheduler
            # Models loaded on-demand per tenant
            logger.info("kenaz-ml: cloud mode, no poller, models loaded per-tenant")

    ...
```

#### Step 3.3: Tenant middleware

Create `src/kenaz_ml/middleware/tenant.py`:

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        header = os.environ.get("KENAZ_TENANT_HEADER", "X-Sigil-Tenant-Id")
        tenant_id = request.headers.get(header)
        request.state.tenant_id = tenant_id
        request.state.tenant_tier = request.headers.get("X-Sigil-Tenant-Tier", "pro")
        response = await call_next(request)
        return response
```

#### Step 3.4: Cloud-aware predict endpoints

In cloud mode, `/predict/*` endpoints:
1. Extract tenant_id from request state
2. Load the tenant's models (from S3 via cache)
3. Accept features in the request body (stateless -- no buffer)
4. Return predictions

The existing endpoint signatures already accept features in the request body (`StuckRequest.features`, `DurationRequest.features`, `WorkflowStateRequest.classified_events`). Cloud mode simply requires these fields to be populated since there's no local buffer.

For tenant-specific model loading, add a helper to AppState:

```python
class AppState:
    def __init__(self):
        ...
        self._tenant_models: dict[str, dict] = {}  # tenant_id -> models dict

    def get_models_for_tenant(self, tenant_id: str) -> dict:
        if tenant_id in self._tenant_models:
            return self._tenant_models[tenant_id]

        # Create tenant-specific model store
        prefix = f"{tenant_id}/"
        s3 = S3ModelStore(bucket=..., prefix=prefix)
        cached = ModelCache(s3, ttl_sec=...)
        models = {
            "stuck": StuckPredictor(model_store=cached),
            "activity": ActivityClassifier(model_store=cached),
            ...
        }
        self._tenant_models[tenant_id] = models
        return models
```

When no model exists for a tenant (store.load returns None), models fall back to rule-based predictions. This matches FR-005 in the spec.

#### Step 3.5: Health endpoint for cloud mode

The `/health` endpoint already exists. In cloud mode, modify to report:
- `mode: "cloud"`
- No poller status
- No SQLite references
- List loaded tenants and their model states

```python
@fastapi_app.get("/health")
async def health():
    if state.mode == "cloud":
        return {
            "status": "ok",
            "mode": "cloud",
            "models": {},  # no pre-loaded models in cloud
            "loaded_tenants": list(state._tenant_models.keys()),
            "uptime_sec": round(time.time() - _start_time, 1),
        }
    # ... existing local health response
```

**Tests**: Create `tests/test_cloud_app.py`:
- Test app starts in cloud mode without SQLite
- Test tenant header extraction
- Test fallback to rule-based when no model weights exist
- Test model caching behavior
- Mock S3 and Postgres for unit tests

---

### Phase 4: Cloud Training Pipeline (Feature 004)

**Goal**: Add `kenaz-ml train --mode cloud` for K8s CronJob batch training.

#### Step 4.1: Add cloud training CLI commands

Update `src/kenaz_ml/cli.py`:

```python
train_parser.add_argument("--mode", choices=["local", "cloud"], default="local")
train_parser.add_argument("--tenant", help="Train models for a specific tenant")
train_parser.add_argument("--all-tenants", action="store_true", help="Train all eligible tenants")
train_parser.add_argument("--aggregate", action="store_true", help="Train aggregate models from pooled data")
```

#### Step 4.2: Create CloudTrainer

Create or extend `training/trainer.py` with cloud-specific logic:

```python
class CloudTrainer:
    """Orchestrates cloud training for one or more tenants."""

    def __init__(self, store_factory, model_store_factory):
        self._store_factory = store_factory      # (tenant_id) -> DataStore
        self._model_store_factory = model_store_factory  # (tenant_id) -> ModelStore

    def train_tenant(self, tenant_id: str) -> TrainingRun:
        store = self._store_factory(tenant_id)
        model_store = self._model_store_factory(tenant_id)
        trainer = Trainer(store=store, model_store=model_store)
        return trainer.train_all()

    def train_all_tenants(self) -> TrainingBatch:
        tenants = self._discover_eligible_tenants()
        results = []
        for tid in tenants:
            try:
                result = self.train_tenant(tid)
                results.append(TrainingRun(tenant_id=tid, status="trained", ...))
            except Exception as e:
                results.append(TrainingRun(tenant_id=tid, status="failed", error=str(e)))
        return TrainingBatch(runs=results)
```

The `Trainer` class (from Phase 1 refactor) already accepts a DataStore. Now it also accepts a ModelStore so it can save weights to S3 for a specific tenant.

#### Step 4.3: Tenant discovery and eligibility

The cloud trainer needs to discover which tenants exist and which are eligible for retraining:

```python
def _discover_eligible_tenants(self) -> list[str]:
    """Query Postgres for tenants with enough new data since last training."""
    # Connect to Postgres without tenant scope (admin-level)
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    # Query all tenant schemas
    # Check each tenant's completed task count and last training timestamp
    # Return list of eligible tenant IDs
    ...
```

Eligibility criteria (from spec):
- Minimum 10 completed tasks for ML training (below that, use synthetic data)
- Minimum 1 hour since last retraining for that tenant
- Skip tenants with no new completed tasks since last training

#### Step 4.4: Training output format

Structured JSON output for monitoring:

```python
@dataclass
class TrainingRun:
    tenant_id: str
    status: str  # "trained", "skipped", "failed"
    models_trained: list[str] = field(default_factory=list)
    sample_count: int = 0
    duration_sec: float = 0.0
    error: str | None = None

@dataclass
class TrainingBatch:
    runs: list[TrainingRun]
    total_duration_sec: float = 0.0

    def summary(self) -> dict:
        return {
            "total_tenants": len(self.runs),
            "trained": len([r for r in self.runs if r.status == "trained"]),
            "skipped": len([r for r in self.runs if r.status == "skipped"]),
            "failed": len([r for r in self.runs if r.status == "failed"]),
            "total_duration_sec": self.total_duration_sec,
            "per_tenant": [asdict(r) for r in self.runs],
        }
```

#### Step 4.5: Aggregate training (P2)

For Team-tier aggregate models:

```python
def train_aggregate(self) -> TrainingRun:
    """Train aggregate models from pooled opted-in tenant data."""
    opted_in = self._discover_opted_in_tenants()
    if len(opted_in) < 2:
        logger.warning("Only %d opted-in tenants, aggregate training may be insufficient", len(opted_in))

    # Pool events and tasks from all opted-in tenants
    all_events = []
    all_tasks = []
    for tid in opted_in:
        store = self._store_factory(tid)
        all_events.extend(store.get_completed_tasks(limit=1000))
        ...

    # Train using pooled data
    model_store = self._model_store_factory("aggregate")  # prefix: aggregate/
    trainer = Trainer(store=PooledStore(all_events, all_tasks), model_store=model_store)
    return trainer.train_all()
```

**Tests**: Create `tests/test_cloud_train.py`:
- Test single-tenant training with mock DataStore and ModelStore
- Test batch training with mixed success/failure
- Test eligibility filtering
- Test structured output format

---

## Implementation Order and Dependencies

```
Week 1:  Phase 1 (Steps 1.1 - 1.3)  -- DataStore protocol, SqliteStore, refactor consumers
Week 2:  Phase 1 (Steps 1.4 - 1.5)  -- PostgresStore, wire into app
         Phase 2 (Steps 2.1 - 2.2)  -- ModelStore protocol, LocalModelStore, refactor models
Week 3:  Phase 2 (Steps 2.3 - 2.5)  -- S3ModelStore, ModelCache, wire into app
         Phase 3 (Steps 3.1 - 3.3)  -- CLI flag, mode-aware factory, tenant middleware
Week 4:  Phase 3 (Steps 3.4 - 3.5)  -- Cloud predict endpoints, health
         Phase 4 (Steps 4.1 - 4.5)  -- Cloud training CLI, batch trainer, aggregate
```

### Dependency Graph

```
Phase 1 (DataStore)  -----> Phase 3 (Cloud Serving)
                      \
                       +--> Phase 4 (Cloud Training)
                      /
Phase 2 (ModelStore) -----> Phase 3 (Cloud Serving)
```

Phases 1 and 2 are independent of each other and could be parallelized.
Phases 3 and 4 require both 1 and 2 to be complete.

## Risk Mitigation

### Risk 1: Refactor breaks existing behavior

**Mitigation**: Phase 1 Step 1.1 writes SqliteStore first, then Step 1.2 swaps the poller. At each step, `pytest tests/` must pass with zero test modifications. Any test that needs changes indicates a regression in the abstraction.

### Risk 2: Postgres query compatibility

**Mitigation**: SQLite and Postgres have slightly different SQL dialects. The queries in kenaz-ml are simple (SELECT, INSERT, basic WHERE). The main risk is parameter syntax (`?` vs `%s`). Both store implementations use their native parameter syntax -- the protocol hides this.

### Risk 3: Cloud mode cold start latency

**Mitigation**: Model cache (Phase 2) ensures S3 downloads happen once per TTL. First request for a new tenant will be slower (S3 download). For critical paths, pre-warm by loading models at startup from a configured tenant list.

### Risk 4: psycopg2-binary platform compatibility

**Mitigation**: psycopg2-binary provides pre-built wheels for Linux, macOS, and Windows on Python 3.10+. If platform issues arise, fall back to `psycopg2` (requires libpq-dev). The optional dependency means local users are unaffected.

## Testing Strategy

### Unit Tests (must pass on every commit)

- `tests/test_storage.py`: DataStore protocol conformance, SqliteStore against temp DB, MockStore
- `tests/test_model_store.py`: ModelStore protocol conformance, LocalModelStore against temp dir, ModelCache TTL behavior
- `tests/test_features.py`: Updated to use DataStore (MockStore) instead of db_path
- `tests/test_models.py`: Updated to use ModelStore (MockStore) instead of filesystem
- `tests/test_server.py`: Updated for cloud mode routes
- `tests/test_cloud_app.py`: Cloud app factory, tenant middleware, fallback predictions
- `tests/test_cloud_train.py`: Cloud trainer, batch processing, structured output

### Integration Tests (require external services)

Marked with `@pytest.mark.integration` or specific markers like `@pytest.mark.postgres`:

- PostgresStore against a real Postgres instance
- S3ModelStore against MinIO or localstack
- Full cloud app startup with Postgres + S3

### Manual Verification

- Start `kenaz-ml serve` (no flags) -- verify identical behavior to current
- Start `kenaz-ml serve --mode cloud` with env vars -- verify stateless serving
- Run `kenaz-ml train --mode cloud --tenant test` -- verify training output

## Open Items Resolved

| Question | Decision |
|---|---|
| Postgres driver | `psycopg2-binary` as optional dep in `[cloud]` extra |
| Connection pooling | Simple connection factory initially, no pool |
| ORM | No. Raw SQL only. Go owns DDL. |
| Protocol location | `src/kenaz_ml/storage/protocol.py` |
| Cloud config | Environment variables. Keep TOML for local mode. |
| Multi-tenant isolation | Per-tenant Postgres schema via `SET search_path` |
| Model cache scope | Per-tenant ModelCache instances, keyed by tenant ID |
| Aggregate model storage | `s3://{bucket}/aggregate/{model}.joblib` |
| Training concurrency guard | Deferred to later -- simple skip-if-running flag for now |
