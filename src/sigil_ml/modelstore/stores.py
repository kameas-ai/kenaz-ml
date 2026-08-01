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
from pathlib import Path
from typing import Protocol, runtime_checkable

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
            from sigil_ml import config

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
    ) -> None:
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 model storage. Install with: pip install kameas-ml[cloud]"
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
        self._bucket = bucket
        self._tenant_id = tenant_id
        self._validate_bucket()

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
        from datetime import datetime, timezone

        version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

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
        mode: "local" or "cloud". Defaults to SIGIL_MODE env var.

    Returns:
        LocalModelStore for local mode, CachedModelStore(S3ModelStore) for cloud.
    """
    from sigil_ml import config

    resolved_mode = mode or config.serving_mode()

    if resolved_mode == "cloud":
        bucket = config.s3_bucket()
        if not bucket:
            raise ValueError("SIGIL_S3_BUCKET environment variable is required in cloud mode")
        s3_store = S3ModelStore(
            bucket=bucket,
            tenant_id=config.tenant_id(),
            endpoint_url=config.s3_endpoint_url(),
            region=config.aws_region(),
        )
        ttl = config.model_cache_ttl()
        return CachedModelStore(s3_store, ttl_seconds=ttl)

    return LocalModelStore()
