# Implementation Plan: Model Storage Abstraction

**Branch**: `003-model-storage-abstraction` | **Date**: 2026-03-29 | **Spec**: `kitty-specs/003-model-storage-abstraction/spec.md`
**Input**: Feature specification from `kitty-specs/003-model-storage-abstraction/spec.md`

## Summary

Introduce a `ModelStore` protocol that decouples model weight persistence from direct filesystem access. Provide two implementations: `LocalModelStore` (preserving current `.joblib` file behavior) and `S3ModelStore` (for cloud/K8s deployment with per-tenant model isolation). Add an in-memory TTL cache (`CachedModelStore`) for cloud mode. Refactor all five model classes and the training pipeline to load/save weights exclusively through the `ModelStore` interface.

## Technical Context

**Language/Version**: Python 3.10+ (matching existing `requires-python = ">=3.10"`)
**Primary Dependencies**: `joblib` (existing), `boto3` (new, optional via `kenaz-ml[cloud]`)
**Storage**: Local filesystem (`.joblib` files in `~/.local/share/sigild/ml-models/`), S3-compatible object storage (cloud mode)
**Testing**: pytest (existing framework, with moto for S3 mocking in cloud tests)
**Target Platform**: Linux, macOS, Windows (local); Linux containers (cloud/K8s)
**Project Type**: Single Python package (existing `src/sigil_ml/` layout)
**Performance Goals**: Cached model load adds <1ms overhead vs direct in-memory access; local filesystem performance within 5% of current
**Constraints**: No heavyweight dependencies beyond `boto3`; `boto3` must be optional; local mode must work without any cloud packages installed
**Scale/Scope**: 5 model classes refactored, 2 storage backends, 1 cache wrapper

## Constitution Check

| Gate | Status | Notes |
|------|--------|-------|
| Minimal dependencies | PASS | `boto3` is optional, not added to core deps |
| Local-first | PASS | `LocalModelStore` preserves exact current behavior; default path unchanged |
| Security-first / no data leakage | PASS | S3 is opt-in cloud mode only; local mode unchanged |
| Simplicity over complexity | PASS | Protocol + 2 implementations + 1 cache wrapper; no over-engineering |
| Cross-platform | PASS | `pathlib` for local paths; `boto3` handles S3 cross-platform |
| pytest / Ruff / Pyre | PASS | All new code tested with pytest; Ruff-compliant; typed for Pyre |

## Project Structure

### Documentation (this feature)

```
kitty-specs/003-model-storage-abstraction/
├── plan.md              # This file
├── spec.md              # Feature specification
├── tasks.md             # Work packages (generated separately)
└── tasks/               # Task prompt files
```

### Source Code Changes

```
src/sigil_ml/
├── storage/                    # NEW: storage abstraction package
│   ├── __init__.py             # Exports ModelStore, LocalModelStore, S3ModelStore, CachedModelStore
│   └── model_store.py          # Protocol + all implementations
├── config.py                   # MODIFIED: add cloud config helpers (S3 bucket, endpoint, region)
├── app.py                      # MODIFIED: wire ModelStore into AppState and model loading
├── cli.py                      # MODIFIED: accept --mode flag, pass ModelStore to trainer
├── models/
│   ├── stuck.py                # MODIFIED: accept ModelStore, remove direct filesystem I/O
│   ├── activity.py             # MODIFIED: accept ModelStore, remove direct filesystem I/O
│   ├── workflow.py             # MODIFIED: accept ModelStore, remove direct filesystem I/O
│   ├── duration.py             # MODIFIED: accept ModelStore, remove direct filesystem I/O
│   └── quality.py              # MODIFIED: accept ModelStore, remove direct filesystem I/O
├── training/
│   ├── trainer.py              # MODIFIED: save weights through ModelStore
│   └── scheduler.py            # MODIFIED: pass ModelStore to Trainer
├── poller.py                   # MODIFIED: models already injected; no direct changes needed
└── routes.py                   # MODIFIED: models loaded via ModelStore; train endpoint updated

tests/
├── test_model_store.py         # NEW: unit tests for LocalModelStore, S3ModelStore, CachedModelStore
├── test_models.py              # MODIFIED: ensure existing tests still pass with ModelStore
├── test_server.py              # MODIFIED: verify endpoints work with ModelStore wiring
└── test_features.py            # UNCHANGED
```

**Structure Decision**: New `storage/` package under `src/sigil_ml/` following existing package layout conventions. The `model_store.py` module contains the protocol and all implementations in a single file (total code is small enough to not warrant splitting).

## Design Decisions

### D1: Protocol vs ABC

**Decision**: Use `typing.Protocol` (PEP 544) for `ModelStore`.

**Rationale**: Protocols enable structural subtyping -- implementations don't need to inherit from a base class. This is idiomatic for Python 3.10+ and matches the codebase's typing style. It also allows duck-typed testing with simple mock objects.

**Alternative rejected**: ABC requires explicit inheritance, adds coupling.

### D2: Serialization responsibility

**Decision**: Model classes serialize/deserialize their own weights using `joblib`. The `ModelStore` operates on raw `bytes` -- it stores and retrieves opaque byte blobs.

**Rationale**: This keeps the `ModelStore` interface simple (just `load` and `save` with bytes). Model classes know their own serialization format. The `QualityEstimator` uses JSON for its weights (not joblib), and this approach handles that naturally.

**Interface**:
```python
class ModelStore(Protocol):
    def load(self, model_name: str) -> bytes | None: ...
    def save(self, model_name: str, data: bytes) -> None: ...
    def exists(self, model_name: str) -> bool: ...
```

### D3: CachedModelStore as decorator

**Decision**: `CachedModelStore` wraps any `ModelStore` instance, adding TTL-based caching. It is not a separate protocol implementation but a decorator/proxy.

**Rationale**: Cache behavior is orthogonal to storage backend. Wrapping allows caching S3 reads without modifying `S3ModelStore`. Local mode does not use caching (filesystem reads are fast enough).

**Cache structure**: `dict[str, tuple[bytes, float]]` keyed by model name, value is `(data, timestamp)`. TTL check on read; expired entries re-fetched from wrapped store.

### D4: Tenant ID handling

**Decision**: `S3ModelStore` receives `tenant_id` at construction time (one instance per tenant). The S3 key path is `{tenant_id}/models/{model_name}/{version}/model.joblib`.

**Rationale**: In cloud mode, tenant context is known at request time. The app creates or retrieves an `S3ModelStore` instance per tenant. This keeps the `ModelStore` interface tenant-unaware -- the protocol itself has no tenant parameter, which means `LocalModelStore` works identically.

### D5: Model versioning

**Decision**: Simple timestamp-based versioning. When saving, the version is a UTC ISO-8601 timestamp (e.g., `20260329T143000Z`). A `latest` pointer file is maintained at `{tenant_id}/models/{model_name}/latest` containing the version string. Loading always reads `latest` first, then fetches that version.

**Rationale**: Keeps implementation simple. No version database needed. S3's eventual consistency is acceptable since we use `latest` pointer writes after model upload. Rollback is possible by updating the `latest` pointer.

### D6: boto3 as optional dependency

**Decision**: `boto3` is declared in `[project.optional-dependencies]` as `cloud = ["boto3>=1.34"]`. The `S3ModelStore` class imports `boto3` at class instantiation time (lazy import), not at module level.

**Rationale**: Local users never install boto3. Import errors surface clearly at startup if cloud mode is configured without the cloud extras.

### D7: Configuration approach

**Decision**: Cloud storage config via environment variables: `SIGIL_S3_BUCKET`, `SIGIL_S3_ENDPOINT_URL` (optional, for MinIO), `AWS_REGION` (standard AWS env var). Local mode uses existing `XDG_DATA_HOME` path logic. Mode selection via `--mode local|cloud` CLI flag (from feature 001 spec).

**Rationale**: Environment variables are the standard K8s configuration mechanism. No new config file format needed. The existing `config.py` module is extended with helper functions.

### D8: Model class refactoring approach

**Decision**: Each model class gains an optional `model_store: ModelStore | None` constructor parameter. When `None` (default), the class constructs a `LocalModelStore` internally, preserving 100% backward compatibility. When provided, it uses the injected store.

**Rationale**: This is the least disruptive refactoring path. Existing code that constructs `StuckPredictor()` with no arguments continues to work. The app factory and tests can inject stores as needed.

## Detailed Component Design

### ModelStore Protocol (`src/sigil_ml/storage/model_store.py`)

```python
from typing import Protocol

class ModelStore(Protocol):
    def load(self, model_name: str) -> bytes | None:
        """Load serialized model weights by name. Returns None if not found."""
        ...

    def save(self, model_name: str, data: bytes) -> None:
        """Save serialized model weights."""
        ...

    def exists(self, model_name: str) -> bool:
        """Check if model weights exist."""
        ...
```

### LocalModelStore

- Reads/writes `.joblib` files from `models_dir()` (default `~/.local/share/sigild/ml-models/`)
- `load()`: reads file as bytes, returns `None` if file doesn't exist
- `save()`: writes bytes to file, creates parent directories
- `exists()`: checks `Path.exists()`
- Constructor: `LocalModelStore(base_dir: Path | None = None)` -- defaults to `config.models_dir()`

### S3ModelStore

- Constructor: `S3ModelStore(bucket: str, tenant_id: str, endpoint_url: str | None = None, region: str | None = None)`
- Lazy `boto3` import at construction time
- `load()`: reads from `s3://{bucket}/{tenant_id}/models/{model_name}/latest` to get version, then reads `s3://{bucket}/{tenant_id}/models/{model_name}/{version}/model.joblib`. Returns `None` on `NoSuchKey` or connection error.
- `save()`: generates timestamp version, writes model bytes, then writes/overwrites `latest` pointer
- `exists()`: HEAD request on the `latest` pointer key
- Error handling: catches `botocore.exceptions.ClientError`, logs warning, returns `None`/raises as appropriate

### CachedModelStore

- Constructor: `CachedModelStore(inner: ModelStore, ttl_seconds: float = 300.0)`
- `load()`: checks cache `dict[str, tuple[bytes, float]]`; if present and not expired, returns cached bytes; otherwise delegates to `inner.load()` and caches result
- `save()`: delegates to `inner.save()`, then updates cache entry with fresh timestamp
- `exists()`: checks cache first, then delegates
- Cache eviction: on `load()`, expired entries are replaced. No background eviction thread (simplicity).

### Model Class Changes (all 5 models)

Each model class is refactored following this pattern (using `StuckPredictor` as the example):

**Before** (current):
```python
class StuckPredictor:
    def __init__(self) -> None:
        weights = config.weights_path("stuck")
        if weights.exists():
            self.model = joblib.load(weights)
```

**After**:
```python
class StuckPredictor:
    def __init__(self, model_store: ModelStore | None = None) -> None:
        self._store = model_store or LocalModelStore()
        data = self._store.load("stuck")
        if data is not None:
            self.model = joblib.loads(data)  # Note: joblib.loads() for bytes
```

Key changes per model:
- `__init__` accepts optional `ModelStore`
- `__init__` loads via `self._store.load(name)` instead of `config.weights_path(name)`
- `train()` saves via `self._store.save(name, joblib.dumps(self.model))` instead of `joblib.dump(self.model, path)`
- No model class imports `config.weights_path` or uses `pathlib` for weight I/O
- `QualityEstimator` is special: uses `json.dumps`/`json.loads` for its weights dict, but the `ModelStore` still stores it as bytes (`json.dumps(...).encode()`)

### Config Changes (`src/sigil_ml/config.py`)

New functions:
```python
def s3_bucket() -> str | None:
    """Return the S3 bucket name from SIGIL_S3_BUCKET env var."""

def s3_endpoint_url() -> str | None:
    """Return optional S3 endpoint URL from SIGIL_S3_ENDPOINT_URL env var."""

def aws_region() -> str | None:
    """Return AWS region from AWS_REGION env var."""

def model_cache_ttl() -> float:
    """Return model cache TTL in seconds from SIGIL_MODEL_CACHE_TTL env var (default: 300)."""
```

Existing functions (`models_dir`, `weights_path`) are preserved for backward compatibility but model classes will no longer call them directly.

### App Factory Changes (`src/sigil_ml/app.py`)

The `AppState.load_models()` method gains a `model_store` parameter:
```python
def load_models(self, model_store: ModelStore | None = None) -> None:
    store = model_store or LocalModelStore()
    self.stuck = StuckPredictor(model_store=store)
    self.activity = ActivityClassifier(model_store=store)
    # ... etc
```

The startup event creates the appropriate store based on mode:
- Local mode: `LocalModelStore()` (default)
- Cloud mode: `CachedModelStore(S3ModelStore(bucket, tenant_id, endpoint_url, region), ttl)`

For cloud mode multi-tenancy, model loading is deferred to request time (not startup). The app keeps a store factory that creates per-tenant stores on demand. This aligns with feature 001's `TenantContext` design.

### CLI Changes (`src/sigil_ml/cli.py`)

- The `train` subcommand gains `--mode` flag support
- In cloud mode, the trainer receives an `S3ModelStore` instead of relying on local filesystem
- The `serve` subcommand will be updated to pass store configuration to the app factory

### pyproject.toml Changes

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27", "ruff>=0.4", "pyre-check>=0.9.18", "moto[s3]>=5.0"]
cloud = ["boto3>=1.34"]
```

## Error Handling Strategy

| Scenario | Behavior |
|----------|----------|
| S3 unreachable during model load | Log warning, return `None`, model falls back to rule-based predictions |
| S3 unreachable during model save | Raise exception (training pipeline handles this as a failed training run) |
| Corrupted `.joblib` bytes from S3 | `joblib.loads()` raises, caught in model `__init__`, logs warning, model starts untrained |
| Missing `latest` pointer in S3 | `load()` returns `None`, treated as "model not yet trained" |
| `boto3` not installed in cloud mode | `ImportError` raised at `S3ModelStore` construction with clear message |
| Invalid S3 bucket or credentials | `botocore.exceptions.ClientError` caught at startup, logged with actionable error message |
| Cache TTL expired | Next `load()` re-fetches from inner store; stale cache entry replaced |
| Concurrent saves from multiple trainers | S3 last-writer-wins (acceptable per spec edge cases) |

## Migration Path

The refactoring is backward-compatible at every step:

1. **Step 1**: Add `storage/model_store.py` with `ModelStore`, `LocalModelStore`, `S3ModelStore`, `CachedModelStore`. No existing code changes.
2. **Step 2**: Refactor model classes to accept optional `ModelStore`. Default `None` creates `LocalModelStore` internally. All existing tests pass without modification.
3. **Step 3**: Update `AppState` and `Trainer` to create and inject stores. Existing behavior preserved.
4. **Step 4**: Add cloud config to `config.py` and mode-aware wiring to `cli.py` and `app.py`.
5. **Step 5**: Add new tests for S3 and cache behavior.

## Dependency Graph

```
WP1: ModelStore protocol + LocalModelStore + CachedModelStore
  ↓
WP2: S3ModelStore implementation (depends on WP1 for protocol)
  ↓ (WP2 and WP3 can run in parallel after WP1)
WP3: Refactor model classes to use ModelStore (depends on WP1 for LocalModelStore)
  ↓
WP4: Wire ModelStore into app factory, CLI, and trainer (depends on WP2 + WP3)
  ↓
WP5: Integration tests and verification (depends on WP4)
```

**Parallel opportunities**: WP2 (S3ModelStore) and WP3 (model class refactoring) can proceed independently after WP1 is complete, since WP3 only needs `LocalModelStore` to pass tests.

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `joblib.loads()` incompatibility with bytes from `joblib.dump()` | Low | High | Verify `joblib.dumps()`/`joblib.loads()` round-trip in WP1 tests |
| Existing tests break due to constructor signature change | Low | Medium | Default `model_store=None` preserves backward compatibility |
| S3 latency exceeds cache TTL, causing request stalls | Medium | Medium | Cache TTL default of 5 minutes; async model refresh in future iteration |
| boto3 dependency conflicts with other packages | Low | Low | Optional dependency, pinned to `>=1.34` |
| QualityEstimator's JSON-based weights don't fit bytes interface | Low | Low | `json.dumps(...).encode("utf-8")` / `json.loads(data.decode("utf-8"))` |

## Success Criteria Verification

| Criterion | How Verified |
|-----------|-------------|
| SC-001: All existing tests pass | `pytest tests/` after full refactoring |
| SC-002: No model class imports filesystem I/O for weights | `grep` for `config.weights_path`, `open(`, `pathlib` in model files |
| SC-003: Local load latency within 5% | Benchmark test comparing before/after load times |
| SC-004: Cached load <1ms overhead | Unit test timing cached vs direct dict access |
| SC-005: Round-trip correctness | Save via one store instance, load via another, compare predictions |
