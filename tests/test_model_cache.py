"""Characterization tests for ModelCache (FR-008).

Written against the pre-move import path (`sigil_ml.cache`) so that they are
demonstrably independent of the storage-layer reorganization: the assertions
below describe behaviour that must survive the move to
`sigil_ml.modelstore.cache` byte for byte.

Time is controlled by patching ``time.monotonic`` on the stdlib ``time``
module rather than sleeping. ``cache.py`` does ``import time`` and calls
``time.monotonic()`` at call time, so attribute patching on the shared module
object is sufficient -- and, unlike ``mock.patch("sigil_ml.cache.time...")``,
it carries no module path that the move would invalidate.

Mutation testing (T006)
-----------------------
Each mutation was applied to a scratch copy of the package (never to repo
source) and the suite re-run to confirm the tests bite. Observed results:

* **M1 -- TTL ignored.** ``get()`` returns the entry regardless of age
  (``if age >= self._ttl_seconds:`` -> ``if False:``).
  4 failures, all in ``TestTtlExpiry``:
  ``test_entry_expires_once_age_reaches_ttl``,
  ``test_entry_expires_well_past_ttl``,
  ``test_expiry_counts_as_miss_and_eviction``,
  ``test_expired_entry_is_dropped_from_storage``.
  (``TestCleanupExpired`` and ``TestLoadedTenants`` survive M1 by design --
  they compare against the TTL through their own code paths, so they pin the
  eager and inventory expiry checks independently of the lazy one in
  ``get()``.)
* **M2 -- LRU evicts the newest.** ``_evict_oldest_unlocked`` uses ``max``
  instead of ``min`` on ``loaded_at``. 1 failure:
  ``TestLruEviction::test_eviction_at_capacity_drops_the_oldest_entry``.

The pre-move baseline required by T007 -- suite counts, ``importtime``
measurement and the FR-007 artifact fixture -- is recorded in the module
docstring of ``tests/test_model_loader.py``, kept in one place rather than
split across two files.
"""

from __future__ import annotations

import threading
import time

import pytest

from sigil_ml.modelstore import ModelCache, create_model_cache
from sigil_ml.modelstore.cache import DEFAULT_MAX_SIZE, DEFAULT_TTL_SECONDS


class _Clock:
    """A deterministic stand-in for ``time.monotonic``."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """Freeze ``time.monotonic`` so TTL behaviour is testable without sleeping."""
    fake = _Clock()
    monkeypatch.setattr(time, "monotonic", fake)
    return fake


# ---------------------------------------------------------------------------
# TTL expiration
# ---------------------------------------------------------------------------


class TestTtlExpiry:
    """Entries live for exactly ttl_seconds; the boundary is inclusive."""

    def test_entry_is_returned_before_ttl_elapses(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=100.0, max_size=10)
        model = object()
        cache.put("tenant-a", "stuck", model)

        clock.advance(99.9)

        assert cache.get("tenant-a", "stuck") is model

    def test_entry_expires_once_age_reaches_ttl(self, clock: _Clock) -> None:
        """``age >= ttl`` expires -- exactly at the TTL the entry is gone."""
        cache = ModelCache(ttl_seconds=100.0, max_size=10)
        cache.put("tenant-a", "stuck", object())

        clock.advance(100.0)

        assert cache.get("tenant-a", "stuck") is None

    def test_entry_expires_well_past_ttl(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=100.0, max_size=10)
        cache.put("tenant-a", "stuck", object())

        clock.advance(10_000.0)

        assert cache.get("tenant-a", "stuck") is None

    def test_expiry_counts_as_miss_and_eviction(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=10.0, max_size=10)
        cache.put("tenant-a", "stuck", object())
        clock.advance(10.0)

        cache.get("tenant-a", "stuck")

        stats = cache.stats()
        assert stats["misses"] == 1
        assert stats["evictions"] == 1
        assert stats["hits"] == 0
        assert stats["entries"] == 0

    def test_expired_entry_is_dropped_from_storage(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=10.0, max_size=10)
        cache.put("tenant-a", "stuck", object())
        clock.advance(10.0)
        cache.get("tenant-a", "stuck")

        assert cache.stats()["entries"] == 0

    def test_put_refreshes_the_entry_clock(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=100.0, max_size=10)
        cache.put("tenant-a", "stuck", object())

        clock.advance(90.0)
        replacement = object()
        cache.put("tenant-a", "stuck", replacement)
        clock.advance(90.0)

        assert cache.get("tenant-a", "stuck") is replacement


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestLruEviction:
    """At capacity a *new* key evicts the entry with the oldest loaded_at."""

    def test_eviction_at_capacity_drops_the_oldest_entry(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=1_000.0, max_size=3)
        first, second, third, fourth = object(), object(), object(), object()

        cache.put("t", "a", first)
        clock.advance(1.0)
        cache.put("t", "b", second)
        clock.advance(1.0)
        cache.put("t", "c", third)
        clock.advance(1.0)
        cache.put("t", "d", fourth)

        assert cache.get("t", "a") is None, "oldest entry should have been evicted"
        assert cache.get("t", "b") is second
        assert cache.get("t", "c") is third
        assert cache.get("t", "d") is fourth

    def test_eviction_keeps_the_cache_at_max_size(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=1_000.0, max_size=2)
        for name in ("a", "b", "c", "d"):
            cache.put("t", name, object())
            clock.advance(1.0)

        assert cache.stats()["entries"] == 2

    def test_eviction_increments_the_eviction_counter(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=1_000.0, max_size=2)
        cache.put("t", "a", object())
        clock.advance(1.0)
        cache.put("t", "b", object())
        clock.advance(1.0)
        cache.put("t", "c", object())

        assert cache.stats()["evictions"] == 1

    def test_overwriting_an_existing_key_at_capacity_evicts_nothing(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=1_000.0, max_size=2)
        cache.put("t", "a", object())
        clock.advance(1.0)
        cache.put("t", "b", object())
        clock.advance(1.0)

        replacement = object()
        cache.put("t", "a", replacement)

        assert cache.stats()["evictions"] == 0
        assert cache.stats()["entries"] == 2
        assert cache.get("t", "a") is replacement

    def test_eviction_is_keyed_on_tenant_and_model(self, clock: _Clock) -> None:
        """The cache key is the (tenant_id, model_name) pair, not either alone."""
        cache = ModelCache(ttl_seconds=1_000.0, max_size=2)
        for_a = object()
        for_b = object()
        cache.put("tenant-a", "stuck", for_a)
        cache.put("tenant-b", "stuck", for_b)

        assert cache.get("tenant-a", "stuck") is for_a
        assert cache.get("tenant-b", "stuck") is for_b


# ---------------------------------------------------------------------------
# Explicit eviction paths
# ---------------------------------------------------------------------------


class TestExplicitEviction:
    """evict() is tenant-scoped; evict_all() clears everything."""

    def test_evict_removes_only_the_named_tenant(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=1_000.0, max_size=10)
        kept = object()
        cache.put("tenant-a", "stuck", object())
        cache.put("tenant-a", "duration", object())
        cache.put("tenant-b", "stuck", kept)

        removed = cache.evict("tenant-a")

        assert removed == 2
        assert cache.get("tenant-a", "stuck") is None
        assert cache.get("tenant-a", "duration") is None
        assert cache.get("tenant-b", "stuck") is kept

    def test_evict_unknown_tenant_removes_nothing(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=1_000.0, max_size=10)
        cache.put("tenant-a", "stuck", object())

        assert cache.evict("tenant-zzz") == 0
        assert cache.stats()["entries"] == 1

    def test_evict_counts_toward_the_eviction_statistic(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=1_000.0, max_size=10)
        cache.put("tenant-a", "stuck", object())
        cache.put("tenant-a", "duration", object())

        cache.evict("tenant-a")

        assert cache.stats()["evictions"] == 2

    def test_evict_all_clears_every_tenant(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=1_000.0, max_size=10)
        cache.put("tenant-a", "stuck", object())
        cache.put("tenant-b", "stuck", object())

        removed = cache.evict_all()

        assert removed == 2
        assert cache.stats()["entries"] == 0
        assert cache.stats()["evictions"] == 2

    def test_evict_all_on_empty_cache_returns_zero(self, clock: _Clock) -> None:
        assert ModelCache(ttl_seconds=1_000.0, max_size=10).evict_all() == 0


class TestCleanupExpired:
    """cleanup_expired() is the eager counterpart to lazy expiry in get()."""

    def test_removes_only_expired_entries(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=100.0, max_size=10)
        cache.put("t", "old", object())
        clock.advance(60.0)
        fresh = object()
        cache.put("t", "new", fresh)
        clock.advance(50.0)  # "old" is 110s, "new" is 50s

        removed = cache.cleanup_expired()

        assert removed == 1
        assert cache.get("t", "old") is None
        assert cache.get("t", "new") is fresh

    def test_returns_zero_when_nothing_has_expired(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=100.0, max_size=10)
        cache.put("t", "a", object())
        clock.advance(1.0)

        assert cache.cleanup_expired() == 0
        assert cache.stats()["entries"] == 1

    def test_counts_toward_the_eviction_statistic(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=10.0, max_size=10)
        cache.put("t", "a", object())
        cache.put("t", "b", object())
        clock.advance(10.0)

        cache.cleanup_expired()

        assert cache.stats()["evictions"] == 2


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestStatistics:
    """stats() is the /introspect observability surface."""

    def test_fresh_cache_reports_zeroed_counters(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=42.0, max_size=7)

        assert cache.stats() == {
            "entries": 0,
            "max_size": 7,
            "ttl_seconds": 42.0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "hit_rate": 0.0,
        }

    def test_hits_and_misses_are_counted_separately(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=1_000.0, max_size=10)
        cache.put("t", "a", object())

        cache.get("t", "a")
        cache.get("t", "a")
        cache.get("t", "missing")

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1

    def test_hit_rate_is_hits_over_total_rounded_to_four_places(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=1_000.0, max_size=10)
        cache.put("t", "a", object())

        cache.get("t", "a")
        cache.get("t", "b")
        cache.get("t", "c")

        assert cache.stats()["hit_rate"] == pytest.approx(0.3333)

    def test_hit_rate_is_zero_when_no_lookups_happened(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=1_000.0, max_size=10)
        cache.put("t", "a", object())

        assert cache.stats()["hit_rate"] == 0.0

    def test_put_alone_does_not_move_hit_or_miss_counters(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=1_000.0, max_size=10)
        cache.put("t", "a", object())

        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestLoadedTenants:
    """loaded_tenants() reports the live inventory, expired entries excluded."""

    def test_groups_model_names_by_tenant(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=1_000.0, max_size=10)
        cache.put("tenant-a", "stuck", object())
        cache.put("tenant-a", "duration", object())
        cache.put("tenant-b", "quality", object())

        loaded = cache.loaded_tenants()

        assert sorted(loaded) == ["tenant-a", "tenant-b"]
        assert sorted(loaded["tenant-a"]) == ["duration", "stuck"]
        assert loaded["tenant-b"] == ["quality"]

    def test_excludes_expired_entries(self, clock: _Clock) -> None:
        cache = ModelCache(ttl_seconds=100.0, max_size=10)
        cache.put("tenant-a", "stuck", object())
        clock.advance(60.0)
        cache.put("tenant-b", "stuck", object())
        clock.advance(50.0)  # tenant-a is 110s old, tenant-b is 50s

        assert cache.loaded_tenants() == {"tenant-b": ["stuck"]}

    def test_empty_cache_returns_empty_mapping(self, clock: _Clock) -> None:
        assert ModelCache(ttl_seconds=100.0, max_size=10).loaded_tenants() == {}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateModelCache:
    """create_model_cache() reads its configuration from the environment."""

    def test_defaults_when_environment_is_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MODEL_CACHE_TTL_SECONDS", raising=False)
        monkeypatch.delenv("MODEL_CACHE_MAX_SIZE", raising=False)

        stats = create_model_cache().stats()

        assert stats["ttl_seconds"] == DEFAULT_TTL_SECONDS
        assert stats["max_size"] == DEFAULT_MAX_SIZE

    def test_reads_ttl_and_max_size_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL_CACHE_TTL_SECONDS", "12.5")
        monkeypatch.setenv("MODEL_CACHE_MAX_SIZE", "3")

        stats = create_model_cache().stats()

        assert stats["ttl_seconds"] == 12.5
        assert stats["max_size"] == 3


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """The cache documents itself thread-safe for concurrent async access."""

    def test_concurrent_get_and_put_does_not_corrupt_state(self) -> None:
        cache = ModelCache(ttl_seconds=1_000.0, max_size=50)
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(200):
                    cache.put(f"tenant-{i % 5}", f"model-{i % 5}", object())
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        def reader() -> None:
            try:
                for i in range(200):
                    cache.get(f"tenant-{i % 5}", f"model-{i % 5}")
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, f"thread errors: {errors}"
        stats = cache.stats()
        assert stats["entries"] == 5, "five distinct (tenant, model) keys were written"
        assert stats["hits"] + stats["misses"] == 800

    def test_concurrent_eviction_and_put_does_not_corrupt_state(self) -> None:
        cache = ModelCache(ttl_seconds=1_000.0, max_size=10)
        errors: list[Exception] = []

        def churn() -> None:
            try:
                for i in range(200):
                    cache.put(f"tenant-{i % 8}", "stuck", object())
                    if i % 10 == 0:
                        cache.evict(f"tenant-{i % 8}")
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=churn) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, f"thread errors: {errors}"
        assert cache.stats()["entries"] <= 10
