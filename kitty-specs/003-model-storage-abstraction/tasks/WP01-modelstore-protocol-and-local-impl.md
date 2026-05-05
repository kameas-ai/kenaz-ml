---
work_package_id: "WP01"
title: "ModelStore Protocol & LocalModelStore"
lane: "planned"
dependencies: []
subtasks:
  - "T001"
  - "T002"
  - "T003"
  - "T004"
  - "T005"
  - "T006"
phase: "Phase 1 - Foundation"
assignee: ""
agent: ""
shell_pid: ""
review_status: ""
reviewed_by: ""
history:
  - timestamp: "2026-03-30T01:45:11Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt generated via /spec-kitty.tasks"
requirement_refs:
  - "FR-001"
  - "FR-002"
  - "FR-003"
  - "FR-004"
---

# Work Package Prompt: WP01 -- ModelStore Protocol & LocalModelStore

## Review Feedback

> **Populated by `/spec-kitty.review`** -- Reviewers add detailed feedback here when work needs changes.

*[This section is empty initially.]*

---

## Markdown Formatting
Wrap HTML/XML tags in backticks: `` `<div>` ``, `` `<script>` ``
Use language identifiers in code blocks: ````python`, ````bash`

---

## Objectives & Success Criteria

- Define a `ModelStore` protocol (Python `Protocol` class) with `load()`, `save()`, and `exists()` methods.
- Implement `LocalModelStore` that preserves the exact current filesystem behavior: reading/writing `.joblib` files from `~/.local/share/sigild/ml-models/`.
- Implement a `model_store_factory()` function that selects the correct backend based on configuration.
- Add configuration helpers to `config.py` for mode detection and cloud config (S3 bucket, region, endpoint, cache TTL).
- After this WP, the abstraction layer is in place and local behavior is functionally identical to today.

**Success Criteria**:
- `ModelStore` protocol is importable from `sigil_ml.storage`.
- `LocalModelStore` can save and load raw bytes, producing files at the exact same paths as current `config.weights_path()`.
- `model_store_factory()` returns `LocalModelStore` when mode is `local` (default).
- Missing model files cause `load()` to return `None` (no exception).
- Corrupted/unreadable files cause `load()` to return `None` with a warning log.
- `exists()` returns `True`/`False` matching filesystem state.

## Context & Constraints

- **Spec**: `kitty-specs/003-model-storage-abstraction/spec.md` -- FR-001 through FR-004.
- **Plan**: `kitty-specs/003-model-storage-abstraction/plan.md` -- Design decisions D1 (Protocol vs ABC), D2 (serialization responsibility), D7 (configuration approach).
- **Current code**: All 5 model classes in `src/sigil_ml/models/` currently call `config.weights_path(name)` and use `joblib.load(path)` / `joblib.dump(obj, path)` directly. `QualityEstimator` uses `json.load`/`json.dump` with `open()`.
- **Current config** (`src/sigil_ml/config.py`): Provides `models_dir()` returning `~/.local/share/sigild/ml-models/` and `weights_path(model_name)` returning `models_dir() / f"{model_name}.joblib"`.
- **Constraint**: No heavyweight dependencies. The protocol uses `typing.Protocol` (stdlib).
- **Constraint**: Protocol must be synchronous (all current code is sync; `boto3` is sync).
- **Constraint**: Per plan.md, all storage implementations live in a single `model_store.py` file (codebase is small enough).
- **Constraint**: `LocalModelStore` must create the models directory if missing (matching `models_dir()` behavior).

## Subtasks & Detailed Guidance

### Subtask T001 -- Create `src/sigil_ml/storage/__init__.py` with public exports

- **Purpose**: Establish the `storage` package as the public entry point. Downstream code imports `from sigil_ml.storage import ModelStore, LocalModelStore`.

- **Steps**:
  1. Create directory `src/sigil_ml/storage/`.
  2. Create `src/sigil_ml/storage/__init__.py` with imports:
     ```python
     """Model storage abstraction for kameas-ml."""

     from sigil_ml.storage.model_store import (
         CachedModelStore,
         LocalModelStore,
         ModelStore,
         model_store_factory,
     )

     __all__ = [
         "ModelStore",
         "LocalModelStore",
         "CachedModelStore",
         "model_store_factory",
     ]
     ```
  3. `S3ModelStore` is intentionally NOT exported here -- it requires `boto3` and should be imported directly from `model_store` when needed. The factory handles instantiation.
  4. `CachedModelStore` will be added in WP03 but is listed here for completeness. For now, the import may be omitted or guarded until WP03 is merged. Alternatively, define a placeholder that WP03 replaces.

- **Files**:
  - Create: `src/sigil_ml/storage/__init__.py`

- **Parallel?**: No -- other subtasks depend on this package existing.

- **Notes**:
  - Use `from __future__ import annotations` for forward-reference compatibility.
  - The `__all__` list controls what `from sigil_ml.storage import *` exposes.

### Subtask T002 -- Define ModelStore protocol in `src/sigil_ml/storage/model_store.py`

- **Purpose**: Establish the contract that all storage backends must fulfill. This is the cornerstone interface that decouples model classes from storage implementation details.

- **Steps**:
  1. Create `src/sigil_ml/storage/model_store.py`.
  2. Define the `ModelStore` protocol per plan.md design decision D1:
     ```python
     from __future__ import annotations

     from typing import Protocol


     class ModelStore(Protocol):
         """Protocol for loading and saving serialized model weights.

         Implementations handle the storage backend (filesystem, S3, etc.).
         Model classes handle serialization (joblib, JSON, etc.).
         The protocol operates on raw bytes -- it is model-agnostic.
         """

         def load(self, model_name: str) -> bytes | None:
             """Load serialized model weights by name.

             Args:
                 model_name: Identifier such as "stuck", "activity",
                     "workflow", "duration", "quality".

             Returns:
                 Raw bytes of the model file, or None if the model
                 does not exist or cannot be read.
             """
             ...

         def save(self, model_name: str, data: bytes) -> None:
             """Save serialized model weights by name.

             Args:
                 model_name: Model identifier.
                 data: Raw bytes of the serialized model.
             """
             ...

         def exists(self, model_name: str) -> bool:
             """Check if model weights exist in the store.

             Args:
                 model_name: Model identifier.

             Returns:
                 True if weights exist and are accessible.
             """
             ...
     ```

- **Files**:
  - Create: `src/sigil_ml/storage/model_store.py`

- **Parallel?**: No -- T003 and T005 depend on this definition.

- **Notes**:
  - The protocol intentionally has no `delete()` or `list()` methods -- those can be added later.
  - Model names are simple strings: `"stuck"`, `"activity"`, `"workflow"`, `"duration"`, `"quality"`.
  - The `exists()` method is per plan.md protocol definition and is used for health checks and startup validation.

### Subtask T003 -- Implement LocalModelStore in `src/sigil_ml/storage/model_store.py`

- **Purpose**: Preserve the exact current local filesystem behavior behind the `ModelStore` interface. This is the default backend for local-mode users and must produce identical files at identical paths.

- **Steps**:
  1. Add `LocalModelStore` to `src/sigil_ml/storage/model_store.py` (same file as the protocol):
     ```python
     import logging
     from pathlib import Path

     from sigil_ml import config

     logger = logging.getLogger(__name__)


     class LocalModelStore:
         """ModelStore backed by the local filesystem.

         Reads and writes .joblib files from a configured base directory.
         Default: ~/.local/share/sigild/ml-models/
         """

         def __init__(self, base_dir: Path | None = None) -> None:
             self._dir = base_dir or config.models_dir()
             self._dir.mkdir(parents=True, exist_ok=True)

         def load(self, model_name: str) -> bytes | None:
             path = self._dir / f"{model_name}.joblib"
             if not path.exists():
                 return None
             try:
                 return path.read_bytes()
             except Exception:
                 logger.warning("Failed to read model file: %s", path, exc_info=True)
                 return None

         def save(self, model_name: str, data: bytes) -> None:
             self._dir.mkdir(parents=True, exist_ok=True)
             path = self._dir / f"{model_name}.joblib"
             path.write_bytes(data)
             logger.info("Saved model to %s (%d bytes)", path, len(data))

         def exists(self, model_name: str) -> bool:
             return (self._dir / f"{model_name}.joblib").exists()
     ```
  2. The constructor takes `base_dir: Path | None = None` so it defaults to `config.models_dir()` but can be overridden for testing.

- **Files**:
  - Update: `src/sigil_ml/storage/model_store.py` (add to same file as protocol)

- **Parallel?**: No -- depends on T002 for the protocol definition it must satisfy structurally.

- **Notes**:
  - `load()` returns raw bytes via `Path.read_bytes()`. Deserialization (`joblib.load(BytesIO(data))`) is the model class's responsibility (plan.md D2).
  - `save()` accepts raw bytes via `Path.write_bytes()`. Serialization is the model class's responsibility.
  - The `.joblib` extension is hardcoded to match current convention. `QualityEstimator` currently saves JSON content with a `.joblib` extension -- this is preserved.
  - Directory creation in `__init__` and `save()` matches current `models_dir()` behavior.
  - The `config` import is for the default `base_dir` only. The class itself has no knowledge of `weights_path()`.

### Subtask T004 -- Add cloud config helpers to `src/sigil_ml/config.py`

- **Purpose**: Provide config-driven mode detection and S3 configuration so the factory and S3ModelStore can read settings from environment variables.

- **Steps**:
  1. Add the following functions to `src/sigil_ml/config.py` (below existing functions):
     ```python
     def serving_mode() -> str:
         """Return the serving mode: 'local' (default) or 'cloud'."""
         return os.environ.get("SIGIL_MODE", "local").lower()


     def model_store_backend() -> str:
         """Return the model storage backend: 'local' or 's3'.

         Defaults based on serving mode: local -> 'local', cloud -> 's3'.
         Override explicitly with SIGIL_MODEL_STORE env var.
         """
         explicit = os.environ.get("SIGIL_MODEL_STORE", "").lower()
         if explicit in ("local", "s3"):
             return explicit
         mode = serving_mode()
         return "s3" if mode == "cloud" else "local"


     def s3_bucket() -> str:
         """Return the S3 bucket name from SIGIL_S3_BUCKET env var."""
         return os.environ.get("SIGIL_S3_BUCKET", "")


     def s3_endpoint_url() -> str | None:
         """Return optional S3 endpoint URL from SIGIL_S3_ENDPOINT_URL.

         Used for S3-compatible stores like MinIO during local development.
         """
         url = os.environ.get("SIGIL_S3_ENDPOINT_URL", "")
         return url if url else None


     def aws_region() -> str:
         """Return AWS region from AWS_REGION env var (default: us-east-1)."""
         return os.environ.get("AWS_REGION", "us-east-1")


     def model_cache_ttl() -> int:
         """Return model cache TTL in seconds (default 300 = 5 minutes).

         Read from SIGIL_MODEL_CACHE_TTL env var. Invalid values fall back
         to the default.
         """
         try:
             return int(os.environ.get("SIGIL_MODEL_CACHE_TTL", "300"))
         except ValueError:
             return 300


     def tenant_id() -> str:
         """Return the default tenant ID from SIGIL_TENANT_ID env var.

         In cloud mode, this determines the S3 key prefix for model storage.
         Per-request tenant ID comes from HTTP headers at the route level.
         """
         return os.environ.get("SIGIL_TENANT_ID", "default")
     ```

- **Files**:
  - Update: `src/sigil_ml/config.py`

- **Parallel?**: Yes -- can proceed alongside T002-T003. Only adds new functions; does not modify existing ones.

- **Notes**:
  - `serving_mode()` is the canonical way to check local vs cloud. Feature 001 (Cloud Serving Mode) also uses this.
  - `model_store_backend()` infers from mode but can be overridden explicitly. This allows local-mode S3 testing.
  - Environment variable names use `SIGIL_` prefix except `AWS_REGION` which is a standard AWS convention.
  - `model_cache_ttl()` returns `int` (seconds). Sub-second precision is unnecessary.
  - Existing functions (`models_dir`, `weights_path`, `db_path`, `sigild_plugin_url`) are untouched.

### Subtask T005 -- Implement model_store_factory() in `src/sigil_ml/storage/model_store.py`

- **Purpose**: Centralize backend selection logic so callers (app.py, cli.py) get the right `ModelStore` with a single call.

- **Steps**:
  1. Add the factory function to `src/sigil_ml/storage/model_store.py`:
     ```python
     from sigil_ml import config


     def model_store_factory(
         backend: str | None = None,
         tenant_id: str | None = None,
     ) -> ModelStore:
         """Create a ModelStore instance based on configuration.

         Args:
             backend: Override backend selection ('local' or 's3').
                 If None, reads from config.model_store_backend().
             tenant_id: Tenant ID for S3 key prefix (cloud mode only).
                 If None, reads from config.tenant_id().

         Returns:
             A ModelStore implementation.

         Raises:
             ValueError: If the backend is unknown.
             NotImplementedError: If S3 backend is requested but not yet
                 implemented (placeholder for WP02).
         """
         backend = backend or config.model_store_backend()

         if backend == "local":
             return LocalModelStore()

         if backend == "s3":
             # S3 implementation added in WP02. Placeholder raises.
             raise NotImplementedError(
                 "S3ModelStore not yet implemented. "
                 "See WP02 for the S3 backend implementation."
             )

         raise ValueError(f"Unknown model store backend: {backend!r}")
     ```
  2. The `NotImplementedError` for S3 is intentional -- WP02 replaces it.

- **Files**:
  - Update: `src/sigil_ml/storage/model_store.py` (add function to same file)

- **Parallel?**: No -- depends on T002, T003, T004.

- **Notes**:
  - `tenant_id` parameter is accepted but only used by S3 backend (WP02). Local mode ignores it.
  - The factory returns a `ModelStore` type (protocol), not a concrete class. This ensures callers depend on the abstraction.
  - Imports: `config` is imported at the top of the module (already needed by `LocalModelStore`).

### Subtask T006 -- Verify LocalModelStore edge cases

- **Purpose**: Ensure `LocalModelStore` correctly handles all edge cases from the spec: missing files, corrupted files, directory creation, and byte round-trip integrity.

- **Steps**:
  1. **Missing file**: `load("nonexistent_model")` returns `None` without raising an exception. The `if not path.exists(): return None` guard handles this.
  2. **Unreadable file**: If `path.read_bytes()` raises (e.g., permission error), `load()` catches the exception, logs a warning, and returns `None`.
  3. **Directory creation on save**: `save()` calls `self._dir.mkdir(parents=True, exist_ok=True)` before writing. If the models directory was deleted between startup and save, it is recreated.
  4. **Overwrite existing file**: `save("stuck", new_data)` overwrites the existing `stuck.joblib` file without error. `Path.write_bytes()` handles this.
  5. **Round-trip integrity**: `save("test", data)` followed by `load("test")` returns exactly `data`. Verify with various byte payloads including empty bytes `b""` and large payloads.
  6. **exists() accuracy**: `exists("test")` returns `False` before save, `True` after save.

- **Files**:
  - Verify behavior in: `src/sigil_ml/storage/model_store.py`

- **Parallel?**: No -- requires T003 to be complete.

- **Notes**:
  - This is a verification subtask during implementation. If tests are later requested, these scenarios become formal test cases.
  - The round-trip check is critical: the model classes depend on getting back exactly the bytes they saved.
  - Empty bytes `b""` is an edge case -- `save()` should write an empty file, `load()` should return `b""` (not `None`).

## Risks & Mitigations

- **Risk**: `Protocol` class behavior differs across type checkers. **Mitigation**: `requires-python = ">=3.10"` ensures full `Protocol` support. Both `mypy` and `pyre` handle it correctly.
- **Risk**: New `storage/` package name conflicts with other planned features. **Mitigation**: Feature 002 (DataStore) handles event/prediction data, not model weights. The namespaces are distinct.
- **Risk**: `config.models_dir()` creates the directory on call (side effect). **Mitigation**: `LocalModelStore` also creates the directory, which is harmless and consistent.

## Review Guidance

- Verify the `ModelStore` protocol has exactly `load`, `save`, and `exists` methods with correct signatures matching plan.md.
- Verify `LocalModelStore` produces files at `{models_dir}/{model_name}.joblib` -- same paths as current `config.weights_path()`.
- Verify the factory function defaults to `local` backend when `SIGIL_MODE` is not set.
- Verify all new config functions have sensible defaults and read from documented env vars.
- Verify no existing code is modified -- this WP only adds new files and new functions to `config.py`.
- Verify `__init__.py` exports are correct and importable.

## Implementation Command

```bash
spec-kitty implement WP01
```

## Activity Log

- 2026-03-30T01:45:11Z -- system -- lane=planned -- Prompt created.
