"""Tests for ModelStore implementations: LocalModelStore, CachedModelStore."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

from sigil_ml.modelstore import CachedModelStore, LocalModelStore

# ---------------------------------------------------------------------------
# LocalModelStore round-trip
# ---------------------------------------------------------------------------


class TestLocalModelStore:
    """Verify LocalModelStore save/load/exists round-trip."""

    def test_round_trip(self, tmp_path: Path) -> None:
        store = LocalModelStore(base_dir=tmp_path)
        data = b"model-weights-bytes"
        store.save("my_model", data)

        assert store.exists("my_model")
        loaded = store.load("my_model")
        assert loaded == data

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        store = LocalModelStore(base_dir=tmp_path)
        assert store.load("nonexistent") is None

    def test_exists_false_when_missing(self, tmp_path: Path) -> None:
        store = LocalModelStore(base_dir=tmp_path)
        assert not store.exists("nonexistent")

    def test_overwrite(self, tmp_path: Path) -> None:
        store = LocalModelStore(base_dir=tmp_path)
        store.save("m", b"v1")
        store.save("m", b"v2")
        assert store.load("m") == b"v2"

    def test_nested_model_name(self, tmp_path: Path) -> None:
        """Model names with '/' create sub-directories (tenant-scoped keys)."""
        store = LocalModelStore(base_dir=tmp_path)
        store.save("tenant-a/stuck", b"data")
        assert store.exists("tenant-a/stuck")
        assert store.load("tenant-a/stuck") == b"data"


# ---------------------------------------------------------------------------
# CachedModelStore
# ---------------------------------------------------------------------------


def _make_inner() -> MagicMock:
    """Create a mock inner ModelStore."""
    inner = MagicMock()
    inner.load = MagicMock(return_value=b"bytes-from-inner")
    inner.save = MagicMock()
    inner.exists = MagicMock(return_value=True)
    return inner


class TestCachedModelStoreTTL:
    """TTL expiration behavior."""

    def test_cache_hit_within_ttl(self) -> None:
        inner = _make_inner()
        cache = CachedModelStore(inner, ttl_seconds=10.0, max_entries=10)

        first = cache.load("m")
        second = cache.load("m")

        assert first == second == b"bytes-from-inner"
        # Inner store should only be called once (cache hit on second load)
        assert inner.load.call_count == 1

    def test_cache_expired_after_ttl(self) -> None:
        inner = _make_inner()
        cache = CachedModelStore(inner, ttl_seconds=0.05, max_entries=10)

        cache.load("m")
        time.sleep(0.1)  # wait for TTL expiration
        cache.load("m")

        # Should call inner twice: first load + expired reload
        assert inner.load.call_count == 2

    def test_save_updates_cache(self) -> None:
        inner = _make_inner()
        cache = CachedModelStore(inner, ttl_seconds=60.0, max_entries=10)

        cache.save("m", b"new-data")
        # Now load should come from cache, not inner
        result = cache.load("m")

        assert result == b"new-data"
        inner.load.assert_not_called()

    def test_exists_uses_cache(self) -> None:
        inner = _make_inner()
        cache = CachedModelStore(inner, ttl_seconds=60.0, max_entries=10)

        cache.load("m")  # populate cache
        assert cache.exists("m") is True
        inner.exists.assert_not_called()


class TestCachedModelStoreLRU:
    """LRU eviction behavior."""

    def test_eviction_at_capacity(self) -> None:
        inner = _make_inner()
        inner.load = MagicMock(side_effect=lambda name: name.encode())
        cache = CachedModelStore(inner, ttl_seconds=60.0, max_entries=3)

        # Fill to capacity
        cache.load("a")
        cache.load("b")
        cache.load("c")

        # Adding a 4th should evict the oldest
        cache.load("d")

        # 'a' was oldest (least recently used), so inner should be called again
        inner.load.reset_mock()
        cache.load("a")
        inner.load.assert_called_once_with("a")

    def test_access_refreshes_lru_order(self) -> None:
        inner = _make_inner()
        inner.load = MagicMock(side_effect=lambda name: name.encode())
        cache = CachedModelStore(inner, ttl_seconds=60.0, max_entries=3)

        cache.load("a")
        cache.load("b")
        cache.load("c")

        # Access 'a' to refresh its timestamp
        cache.load("a")

        # Now add 'd' - should evict 'b' (oldest after refresh)
        cache.load("d")

        inner.load.reset_mock()
        # 'a' should still be cached (was refreshed)
        cache.load("a")
        inner.load.assert_not_called()

        # 'b' should have been evicted
        cache.load("b")
        inner.load.assert_called_once_with("b")


class TestCachedModelStoreThreadSafety:
    """Thread safety under concurrent access."""

    def test_concurrent_loads(self) -> None:
        inner = _make_inner()
        call_count = 0
        lock = threading.Lock()

        def counting_load(name: str) -> bytes:
            nonlocal call_count
            with lock:
                call_count += 1
            time.sleep(0.01)  # simulate I/O
            return b"data"

        inner.load = MagicMock(side_effect=counting_load)
        cache = CachedModelStore(inner, ttl_seconds=60.0, max_entries=100)

        errors: list[Exception] = []

        def worker(model_name: str) -> None:
            try:
                for _ in range(10):
                    cache.load(model_name)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"model_{i % 5}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_save_and_load(self) -> None:
        inner = _make_inner()
        cache = CachedModelStore(inner, ttl_seconds=60.0, max_entries=100)
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(50):
                    cache.save(f"m_{i % 5}", f"v{i}".encode())
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for i in range(50):
                    cache.load(f"m_{i % 5}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        threads += [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
