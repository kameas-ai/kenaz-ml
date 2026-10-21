"""ModelStore protocol and implementations for model weight persistence.

Implementations:
    LocalModelStore   — filesystem (local mode, default)
    S3ModelStore      — S3/MinIO (cloud mode)
    CachedModelStore  — in-memory TTL cache wrapping any ModelStore
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ModelStore(Protocol):
    """Protocol for loading and saving model weights as raw bytes."""

    def load(self, model_name: str) -> bytes | None:
        """Load model weights by name. Returns None if not found."""
        ...

    def save(self, model_name: str, data: bytes) -> None:
        """Save model weights by name."""
        ...

    def exists(self, model_name: str) -> bool:
        """Check if model weights exist."""
        ...


class LocalModelStore:
    """Filesystem-based model store using .joblib files."""

    def __init__(self, base_dir: Path | None = None) -> None:
        if base_dir is None:
            from kenaz_ml import config

            base_dir = config.models_dir()
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, model_name: str) -> Path:
        return self._base_dir / f"{model_name}.joblib"

    def load(self, model_name: str) -> bytes | None:
        """Load model weights from disk. Returns None if the file does not exist."""
        path = self._path(model_name)
        if not path.exists():
            return None
        try:
            return path.read_bytes()
        except OSError:
            logger.warning("Failed to read model %s from %s", model_name, path)
            return None

    def save(self, model_name: str, data: bytes) -> None:
        """Save model weights to disk, creating parent directories as needed."""
        path = self._path(model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def exists(self, model_name: str) -> bool:
        """Return True if the model weight file exists on disk."""
        return self._path(model_name).exists()

    def last_modified(self, model_name: str) -> float | None:
        """Return the mtime (epoch seconds) of the persisted weights, or None.

        Used by the /introspect endpoint as an honest last-trained proxy —
        weights are only ever written by the trainer, so the file mtime is
        the last successful training time. Not part of the ModelStore
        protocol; callers must getattr-probe for it (other backends do not
        track this cheaply).
        """
        try:
            return self._path(model_name).stat().st_mtime
        except OSError:
            return None


class S3ModelStore:
    """S3-based model store with per-tenant key prefix and latest pointer versioning.

    Key format: {tenant_id}/models/{model_name}/{version}/model.joblib
    Latest pointer: {tenant_id}/models/{model_name}/latest
    """

    def __init__(
        self,
        bucket: str,
        tenant_id: str = "default",
        endpoint_url: str | None = None,
        region: str | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        """``client``: an S3 client to use instead of building one -- a cloud
        host passes one carrying the tenant's own assumed-role credentials."""
        self._bucket = bucket
        self._tenant_id = tenant_id
        if client is not None:
            self._s3 = client
            self._validate_prefix()
            return
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 model storage. Install with: pip install kenaz-ml[cloud]"
            ) from None

        boto_config = BotoConfig(
            connect_timeout=5,
            read_timeout=10,
            retries={"max_attempts": 2},
        )
        kwargs: dict = {"config": boto_config}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        if region:
            kwargs["region_name"] = region

        self._s3 = boto3.client("s3", **kwargs)
        self._validate_bucket()

    def _validate_prefix(self) -> None:
        """Check access the way a prefix-scoped role can: a tenant role may list
        only under its own prefix, so ``head_bucket`` would be refused."""
        try:
            self._s3.list_objects_v2(Bucket=self._bucket, Prefix=f"{self._tenant_id}/models/", MaxKeys=1)
        except Exception as e:
            raise ValueError(f"S3 prefix '{self._tenant_id}/models/' in '{self._bucket}' is not accessible: {e}") from e

    def _validate_bucket(self) -> None:
        try:
            self._s3.head_bucket(Bucket=self._bucket)
        except Exception as e:
            raise ValueError(f"S3 bucket '{self._bucket}' is not accessible: {e}") from e

    def _latest_key(self, model_name: str) -> str:
        return f"{self._tenant_id}/models/{model_name}/latest"

    def _versioned_key(self, model_name: str, version: str) -> str:
        return f"{self._tenant_id}/models/{model_name}/{version}/model.joblib"

    def load(self, model_name: str) -> bytes | None:
        """Load model weights from S3 by resolving the latest version pointer."""
        try:
            # Read the latest pointer to get the version
            resp = self._s3.get_object(Bucket=self._bucket, Key=self._latest_key(model_name))
            version = resp["Body"].read().decode("utf-8").strip()

            # Load the versioned model
            resp = self._s3.get_object(Bucket=self._bucket, Key=self._versioned_key(model_name, version))
            return resp["Body"].read()
        except self._s3.exceptions.NoSuchKey:
            return None
        except Exception:
            logger.warning("Failed to load model %s from S3", model_name, exc_info=True)
            return None

    def save(self, model_name: str, data: bytes) -> None:
        """Save model weights to S3 with a timestamped version and update the latest pointer."""
        from datetime import datetime

        version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

        # Save versioned model
        self._s3.put_object(
            Bucket=self._bucket,
            Key=self._versioned_key(model_name, version),
            Body=data,
        )

        # Update latest pointer
        self._s3.put_object(
            Bucket=self._bucket,
            Key=self._latest_key(model_name),
            Body=version.encode("utf-8"),
        )

    def exists(self, model_name: str) -> bool:
        """Return True if a latest-version pointer exists for this model in S3."""
        try:
            self._s3.head_object(Bucket=self._bucket, Key=self._latest_key(model_name))
            return True
        except Exception:
            return False

    # --- version lifecycle (member-erasure retrain) ---
    #
    # The bucket is versioned, so overwriting ``latest`` or deleting a key
    # leaves the old bytes readable as a noncurrent version. These methods
    # remove versions for good, the way ``s3_retained.purge_prefix`` does.

    def _model_prefix(self, model_name: str) -> str:
        if not model_name or "/" in model_name or model_name in (".", ".."):
            raise ValueError(f"refusing model name {model_name!r}: one non-empty path segment required")
        return f"{self._tenant_id}/models/{model_name}/"

    def list_models(self) -> list[str]:
        """Names of this tenant's models: every ``<model>/`` holding a ``latest`` pointer.

        Other data kept under the tenant's models prefix (retained sets,
        a host's own records) has no pointer and is not listed.
        """
        prefix = f"{self._tenant_id}/models/"
        names: set[str] = set()
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                parts = obj["Key"][len(prefix) :].split("/")
                if len(parts) == 2 and parts[1] == "latest" and parts[0]:
                    names.add(parts[0])
        return sorted(names)

    def _is_not_found(self, exc: BaseException) -> bool:
        """True only for S3's "no such key" (``NoSuchKey`` or a 404 ``ClientError``)."""
        no_such_key = getattr(getattr(self._s3, "exceptions", None), "NoSuchKey", None)
        if isinstance(no_such_key, type) and isinstance(exc, no_such_key):
            return True
        response = getattr(exc, "response", None)
        code = str((response or {}).get("Error", {}).get("Code", "")) if isinstance(response, dict) else ""
        return code in ("NoSuchKey", "404", "NotFound")

    def current_version(self, model_name: str) -> str | None:
        """The version the ``latest`` pointer names, or ``None`` when there is none.

        ``None`` means S3 positively answered that the pointer does not exist.
        Any other failure (throttling, a timeout, a refused credential) is
        raised: a caller deciding whether a model is gone must not mistake a
        failed read for an absent model.
        """
        self._model_prefix(model_name)
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=self._latest_key(model_name))
        except Exception as e:
            if self._is_not_found(e):
                return None
            raise
        return resp["Body"].read().decode("utf-8").strip() or None

    def delete_model(self, model_name: str) -> int:
        """Permanently delete every version of every object of one model.

        Removes the pointer, all version directories, their noncurrent versions
        and delete markers. Idempotent. Returns the number of versions removed.
        """
        from kenaz_ml.modelstore.s3_retained import purge_prefix

        return purge_prefix(self._s3, self._bucket, self._model_prefix(model_name))

    def purge_versions_before(self, model_name: str, before: datetime) -> int:
        """Permanently delete this model's object versions written before ``before``.

        Covers every key under the model's prefix -- old version directories,
        noncurrent ``latest`` pointers, delete markers -- whose S3
        ``LastModified`` precedes ``before`` (timezone-aware). The current
        version is never deleted, whatever its timestamp: the live ``latest``
        pointer and the live object of the version it names are always kept.

        Raises ``LookupError`` only when S3 positively reports no current
        version (call :meth:`delete_model` to remove a model outright); any
        other read failure propagates and nothing is deleted. Idempotent.
        Returns the number of versions removed.

        Clock skew: ``before`` is the caller's clock, ``LastModified`` is S3's.
        Skew can only shift which *noncurrent* versions near the cutoff are
        removed; the live pointer and artifact are kept by key, not by time.
        A caller that needs every pre-cutoff version gone should pass a cutoff
        taken after its own last pre-cutoff write (the erasure-retrain host
        passes the retrain's start, which follows the erased stream's stop).
        """
        from kenaz_ml.modelstore.s3_retained import _delete_versions, _versions

        if before.tzinfo is None:
            raise ValueError("before must be timezone-aware")
        prefix = self._model_prefix(model_name)
        current = self.current_version(model_name)
        if current is None:
            raise LookupError(f"{model_name!r} has no current version to keep")
        keep = {self._latest_key(model_name), self._versioned_key(model_name, current)}
        doomed = [
            e
            for e in _versions(self._s3, self._bucket, prefix)
            if not (e["IsLatest"] and e["Key"] in keep) and (e["LastModified"] is None or e["LastModified"] < before)
        ]
        removed = _delete_versions(self._s3, self._bucket, doomed)
        logger.info("s3: purged %d superseded version(s) of %s", removed, prefix)
        return removed


class CachedModelStore:
    """In-memory TTL cache wrapping any ModelStore.

    Thread-safe via threading.Lock. Entries are evicted when TTL expires
    or max_entries is reached (LRU by access time).
    """

    def __init__(
        self,
        inner: ModelStore,
        ttl_seconds: float = 300.0,
        max_entries: int = 100,
    ) -> None:
        self._inner = inner
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._cache: dict[str, tuple[bytes, float]] = {}  # name -> (data, access_time)
        self._lock = threading.Lock()

    def load(self, model_name: str) -> bytes | None:
        """Load from cache if fresh, otherwise delegate to the inner store and cache the result."""
        now = time.monotonic()
        with self._lock:
            if model_name in self._cache:
                data, cached_at = self._cache[model_name]
                if now - cached_at < self._ttl:
                    # Update access time for LRU
                    self._cache[model_name] = (data, now)
                    return data
                else:
                    del self._cache[model_name]

        # Cache miss — load from inner store
        data = self._inner.load(model_name)
        if data is not None:
            with self._lock:
                self._evict_if_full()
                self._cache[model_name] = (data, now)
        return data

    def save(self, model_name: str, data: bytes) -> None:
        """Save to the inner store and update the cache."""
        self._inner.save(model_name, data)
        with self._lock:
            self._evict_if_full()
            self._cache[model_name] = (data, time.monotonic())

    def exists(self, model_name: str) -> bool:
        """Check cache first; fall back to inner store if not cached or expired."""
        with self._lock:
            if model_name in self._cache:
                _, cached_at = self._cache[model_name]
                if time.monotonic() - cached_at < self._ttl:
                    return True
        return self._inner.exists(model_name)

    def _evict_if_full(self) -> None:
        """Evict oldest entry if cache is at capacity. Must hold lock."""
        while len(self._cache) >= self._max_entries:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]


def model_store_factory(mode: str | None = None) -> ModelStore:
    """Create the appropriate ModelStore based on operating mode.

    Args:
        mode: "local" or "cloud". Defaults to the KENAZ_MODE env var.

    Returns:
        LocalModelStore for local mode, CachedModelStore(S3ModelStore) for cloud.
    """
    from kenaz_ml import config

    resolved_mode = mode or config.serving_mode()

    if resolved_mode == "cloud":
        bucket = config.s3_bucket()
        if not bucket:
            raise ValueError("KENAZ_S3_BUCKET environment variable is required in cloud mode")
        s3_store = S3ModelStore(
            bucket=bucket,
            tenant_id=config.tenant_id(),
            endpoint_url=config.s3_endpoint_url(),
            region=config.aws_region(),
        )
        ttl = config.model_cache_ttl()
        return CachedModelStore(s3_store, ttl_seconds=ttl)

    return LocalModelStore()
