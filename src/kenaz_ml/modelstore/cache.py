"""In-memory model cache with TTL expiration and LRU eviction.

Stores loaded model objects keyed by (tenant_id, model_name) with
time-based expiration. Thread-safe for concurrent async access.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Type alias for cache key.
CacheKey = tuple[str, str]  # (tenant_id, model_name)

DEFAULT_TTL_SECONDS = 300.0
DEFAULT_MAX_SIZE = 100


@dataclass
class _CacheEntry:
    """Internal: a single cached model with creation timestamp."""

    model: Any
    loaded_at: float  # time.monotonic() timestamp


class ModelCache:
    """Thread-safe in-memory cache for tenant model objects.

    Args:
        ttl_seconds: Time-to-live for cache entries in seconds.
        max_size: Maximum number of entries before LRU eviction.
    """

    def __init__(
        self,
        ttl_seconds: float = 300.0,
        max_size: int = 100,
    ) -> None:
        self._entries: dict[CacheKey, _CacheEntry] = {}
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, tenant_id: str, model_name: str) -> Any | None:
        """Retrieve a model from cache.

        Returns the model object on hit, None on miss or expiry.
        """
        key = (tenant_id, model_name)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            age = time.monotonic() - entry.loaded_at
            if age >= self._ttl_seconds:
                del self._entries[key]
                self._misses += 1
                self._evictions += 1
                return None
            self._hits += 1
            return entry.model

    def put(self, tenant_id: str, model_name: str, model: Any) -> None:
        """Store a model in the cache.

        If the cache is at capacity and the key is new, the oldest
        entry is evicted first (LRU).
        """
        key = (tenant_id, model_name)
        entry = _CacheEntry(model=model, loaded_at=time.monotonic())
        with self._lock:
            if key not in self._entries and len(self._entries) >= self._max_size:
                self._evict_oldest_unlocked()
            self._entries[key] = entry

    def evict(self, tenant_id: str) -> int:
        """Remove all entries for a tenant. Returns count removed."""
        with self._lock:
            keys = [k for k in self._entries if k[0] == tenant_id]
            for k in keys:
                del self._entries[k]
            self._evictions += len(keys)
            return len(keys)

    def evict_all(self) -> int:
        """Clear the entire cache. Returns count removed."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._evictions += count
            return count

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed.

        This is optional -- expiry is also checked lazily on get().
        Useful for periodic maintenance to free memory.
        """
        now = time.monotonic()
        with self._lock:
            expired = [k for k, v in self._entries.items() if (now - v.loaded_at) >= self._ttl_seconds]
            for k in expired:
                del self._entries[k]
            self._evictions += len(expired)
            return len(expired)

    def stats(self) -> dict[str, Any]:
        """Return cache statistics for observability."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._entries),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            }

    def loaded_tenants(self) -> dict[str, list[str]]:
        """Return mapping of tenant_id -> list of cached model names.

        Only includes non-expired entries.
        """
        now = time.monotonic()
        with self._lock:
            result: dict[str, list[str]] = {}
            for (tenant_id, model_name), entry in self._entries.items():
                if (now - entry.loaded_at) < self._ttl_seconds:
                    result.setdefault(tenant_id, []).append(model_name)
            return result

    def _evict_oldest_unlocked(self) -> None:
        """Evict the entry with the oldest loaded_at timestamp.

        MUST be called while holding self._lock.
        """
        if not self._entries:
            return
        oldest_key = min(self._entries, key=lambda k: self._entries[k].loaded_at)
        del self._entries[oldest_key]
        self._evictions += 1
        logger.debug("cache: evicted oldest entry %s", oldest_key)


def create_model_cache() -> ModelCache:
    """Create a ModelCache with settings from environment variables.

    Env vars:
        MODEL_CACHE_TTL_SECONDS: TTL in seconds (default 300).
        MODEL_CACHE_MAX_SIZE: Maximum entries (default 100).
    """
    ttl = float(os.environ.get("MODEL_CACHE_TTL_SECONDS", str(DEFAULT_TTL_SECONDS)))
    max_size = int(os.environ.get("MODEL_CACHE_MAX_SIZE", str(DEFAULT_MAX_SIZE)))
    logger.info("cache: created with ttl=%.0fs, max_size=%d", ttl, max_size)
    return ModelCache(ttl_seconds=ttl, max_size=max_size)
