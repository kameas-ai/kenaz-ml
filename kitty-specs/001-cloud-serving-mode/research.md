# Research: Cloud Serving Mode

**Feature**: 001-cloud-serving-mode
**Date**: 2026-03-30
**Status**: Complete

## R1: Serving Mode Branching Strategy

**Decision**: Use a Python `enum.Enum` subclass `ServingMode` with two values (`LOCAL`, `CLOUD`), threaded through `create_app()` as a factory parameter and stored on `AppState`.

**Rationale**: An enum is type-safe, exhaustive in match statements, and trivially testable. Passing it through the factory avoids global state and makes testing different modes straightforward.

**Alternatives considered**:
- Global config singleton: Rejected because it makes testing harder and introduces hidden coupling.
- String literals: Rejected because they lack type safety and IDE support.

## R2: Tenant Identification Mechanism

**Decision**: Read tenant ID from the `X-Tenant-ID` HTTP header, set by the upstream API gateway. No JWT parsing. No authentication in kenaz-ml itself.

**Rationale**: The API gateway handles authentication and authorization. kenaz-ml in cloud mode is an internal service behind the gateway. A simple header is the lightest-weight approach with zero new dependencies.

**Alternatives considered**:
- JWT claim extraction: Rejected because it adds a `PyJWT` dependency and duplicates gateway logic.
- Query parameter: Rejected because it pollutes the API contract and is harder to enforce consistently.

## R3: Model Cache Design

**Decision**: Custom `ModelCache` class using a Python `dict` with `(tenant_id, model_name)` tuple keys, `time.monotonic()` timestamps for TTL, and `threading.Lock` for thread safety. LRU eviction when size exceeds 100 entries. TTL configurable via `MODEL_CACHE_TTL_SECONDS` env var (default 300s).

**Rationale**: Python's `functools.lru_cache` does not support TTL or composite keys cleanly. A custom implementation is ~60 lines and avoids external dependencies. `time.monotonic()` is immune to wall-clock adjustments. A threading lock is sufficient because uvicorn's async workers run on a thread pool.

**Alternatives considered**:
- `cachetools.TTLCache`: Good fit but adds a dependency. Rejected per constitution (minimal deps).
- `functools.lru_cache` with wrapper: Does not support TTL natively. Rejected.
- No cache (load on every request): Unacceptable latency for joblib deserialization on every request.

## R4: Model Loading Protocol

**Decision**: Define a `ModelLoader` typing.Protocol with a single method `load(tenant_id: str, model_name: str) -> Any | None`. Provide `FilesystemModelLoader` as the initial implementation loading from `{models_dir}/{tenant_id}/{model_name}.joblib`.

**Rationale**: A Protocol (structural typing) avoids inheritance and keeps the interface lightweight. The filesystem loader is sufficient for local testing and development. Feature 003 (Model Storage Abstraction) will provide the S3 implementation against this same Protocol.

**Alternatives considered**:
- ABC with `@abstractmethod`: More ceremony than needed for a single-method interface.
- Direct S3 integration now: Premature; S3 client is a heavyweight dependency that belongs in Feature 003.

## R5: Conditional Startup in Cloud Mode

**Decision**: The `create_app(mode: ServingMode)` factory branches at startup:
- **Cloud mode**: Skip `ensure_ml_tables()`, skip `EventPoller`, skip `TrainingScheduler`, skip local `load_models()`. Initialize `ModelCache` and `ModelLoader` instead.
- **Local mode**: Identical to current behavior. No changes to any existing code paths.

**Rationale**: A single branch point in the startup event keeps the change surface minimal. The module-level `app = create_app()` continues to default to `LOCAL` for backward compatibility with `uvicorn sigil_ml.app:app`.

**Alternatives considered**:
- Separate `create_cloud_app()` factory: Rejected because it duplicates route registration and makes the CLI more complex.
- Plugin/strategy pattern for startup: Over-engineered for two modes.

## R6: Endpoint Behavior in Cloud Mode

**Decision**: All five `/predict/*` endpoints continue to accept feature dicts in the request body. In cloud mode:
- `task_id`-only requests return 400 (no SQLite to look up from).
- `features` are required in the request body.
- `/train` returns 405 (training is not supported in cloud mode).
- `/status` returns cache stats and loaded tenants instead of SQLite queries.
- `/plugins` remains available (queries sigild HTTP API, not SQLite).

**Rationale**: The existing request schemas already support inline features. Cloud mode simply enforces that path and disables the SQLite fallback.

**Alternatives considered**:
- Separate cloud-specific route handlers: Rejected because it duplicates prediction logic.
- Accept `task_id` and proxy to a data service: Deferred to Feature 002 (Storage Abstraction).

## R7: Health Endpoint Extension

**Decision**: Add an optional `mode` field to `HealthResponse`. In cloud mode, the health check verifies that the `ModelLoader` is reachable (filesystem exists or future S3 connectivity). In local mode, behavior is unchanged.

**Rationale**: Kubernetes liveness/readiness probes need mode-aware health checks. Adding an optional field is backward-compatible.

**Alternatives considered**:
- Separate `/cloud-health` endpoint: Rejected because it fragments the health check surface.

## R8: Cloud Dependencies and Packaging

**Decision**: Add a `[cloud]` optional extras group in `pyproject.toml`. Initially empty (no new deps needed for filesystem-based cloud mode). Future features (003, 002) will add `boto3`, `asyncpg` etc. here.

**Rationale**: Keeps the base install lightweight for local users. Cloud operators install with `pip install kenaz-ml[cloud]`.

**Alternatives considered**:
- Separate `kenaz-ml-cloud` package: Over-engineered for this stage.
