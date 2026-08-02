"""Tests for the cloud training pipeline (Feature 004).

Tests cover:
- Training data models (WP01)
- CloudTrainer per-tenant training (WP02)
- Batch training and tenant discovery (WP03)
- Training lock protocol (WP04)
- Aggregate training (WP05)
- Observability / structured output (WP06)
- CLI cloud training flags
- Feature extraction from data (cloud-compatible extractors)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from kenaz_ml.training.models import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SKIPPED_LOCKED,
    STATUS_TRAINED,
    CloudTrainingConfig,
    TrainingBatch,
    TrainingRun,
    TrainingSummary,
)


@pytest.fixture(autouse=True)
def _isolate_models(tmp_path, monkeypatch):
    """Redirect model weights to a temp directory so tests don't pollute real config."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))


# ---------------------------------------------------------------------------
# Mock DataStore
# ---------------------------------------------------------------------------


class MockDataStore:
    """A mock DataStore that supports cloud training operations."""

    def __init__(
        self,
        tenants: list[str] | None = None,
        opted_in_tenants: list[str] | None = None,
        tasks_per_tenant: dict[str, list[dict]] | None = None,
        events_per_task: dict[str, list[dict]] | None = None,
        last_training_ts: dict[str, int | None] | None = None,
    ) -> None:
        self._tenants = tenants or []
        self._opted_in_tenants = opted_in_tenants or []
        self._tasks = tasks_per_tenant or {}
        self._events = events_per_task or {}
        self._last_training_ts = last_training_ts or {}
        self.recorded_events: list[dict] = []
        self._ml_events: list[dict] = []

    def list_tenants(self) -> list[str]:
        return list(self._tenants)

    def list_opted_in_tenants(self) -> list[str]:
        return list(self._opted_in_tenants)

    def get_last_training_ts(self, tenant_id: str) -> int | None:
        return self._last_training_ts.get(tenant_id)

    def query_completed_tasks(self, tenant_id: str) -> list[dict]:
        return list(self._tasks.get(tenant_id, []))

    def get_completed_tasks_for_tenant(self, tenant_id: str) -> list[dict]:
        return list(self._tasks.get(tenant_id, []))

    def query_events_for_task(self, tenant_id: str, task_id: str) -> list[dict]:
        return list(self._events.get(task_id, []))

    def get_events_for_task_id(self, task_id: str) -> list[dict]:
        return list(self._events.get(task_id, []))

    def get_all_tenant_ids(self) -> list[str]:
        return list(self._tenants)

    def get_opted_in_tenant_ids(self) -> list[str]:
        return list(self._opted_in_tenants)

    def record_training_run(self, tenant_id: str, status: str, duration_ms: int) -> None:
        self._ml_events.append(
            {
                "kind": "training",
                "endpoint": "cloud_trainer",
                "routing": tenant_id,
                "latency_ms": duration_ms,
            }
        )

    # Existing DataStore protocol methods (fallback path)
    def get_completed_task_ids(self) -> list[str]:
        all_ids = []
        for tasks in self._tasks.values():
            for t in tasks:
                all_ids.append(t["id"])
        return all_ids

    def get_task_by_id(self, task_id: str) -> dict | None:
        for tasks in self._tasks.values():
            for t in tasks:
                if t["id"] == task_id:
                    return t
        return None

    def get_events_for_task(self, task_id: str, since: int | None = None) -> list[dict]:
        return list(self._events.get(task_id, []))

    def insert_ml_event(self, kind: str, endpoint: str, routing: str, latency_ms: int) -> None:
        self._ml_events.append(
            {
                "kind": kind,
                "endpoint": endpoint,
                "routing": routing,
                "latency_ms": latency_ms,
            }
        )

    def commit(self) -> None:
        pass

    def ensure_tables(self) -> None:
        pass

    def close(self) -> None:
        pass


class MockModelStore:
    """A mock ModelStore that stores bytes in memory."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def load(self, model_name: str) -> bytes | None:
        return self._store.get(model_name)

    def save(self, model_name: str, data: bytes) -> None:
        self._store[model_name] = data

    def exists(self, model_name: str) -> bool:
        return model_name in self._store


class MockTrainingLock:
    """A mock TrainingLock for testing."""

    def __init__(self, locked_tenants: set[str] | None = None) -> None:
        self._locked = locked_tenants or set()
        self._acquired: set[str] = set()

    def acquire(self, tenant_id: str) -> bool:
        if tenant_id in self._locked:
            return False
        self._acquired.add(tenant_id)
        return True

    def release(self, tenant_id: str) -> None:
        self._acquired.discard(tenant_id)


def _make_tasks(n: int, tenant_id: str = "t1") -> list[dict]:
    """Generate n completed task dicts with varied labels for both classes.

    Half the tasks have high test_fails (stuck heuristic triggers),
    half have low test_fails (not stuck). This ensures the GradientBoosting
    classifier has both classes during training.
    """
    now_ms = int(time.time() * 1000)
    tasks = []
    for i in range(n):
        # Alternate: even tasks are "stuck" (high test_fails), odd are "ok"
        if i % 2 == 0:
            test_fails = 5  # > 3, so stuck heuristic fires
        else:
            test_fails = 0  # not stuck
        tasks.append(
            {
                "id": f"{tenant_id}-task-{i}",
                "repo_root": "/tmp/repo",
                "branch": "feature/test",
                "phase": "done",
                "files": json.dumps({"main.py": 5}),
                "started_at": now_ms - 3600_000,
                "last_active": now_ms - 60_000,
                "completed_at": now_ms - 60_000,
                "commit_count": 2,
                "test_runs": 5,
                "test_fails": test_fails,
            }
        )
    return tasks


def _make_events(task_id: str) -> list[dict]:
    """Generate sample events for a task."""
    now_ms = int(time.time() * 1000)
    return [
        {"kind": "edit", "source": "editor", "payload": {"file": "main.py"}, "ts": now_ms - 3000_000},
        {"kind": "edit", "source": "editor", "payload": {"file": "utils.py"}, "ts": now_ms - 2800_000},
        {"kind": "save", "source": "editor", "payload": {"file": "main.py"}, "ts": now_ms - 2500_000},
        {"kind": "commit", "source": "git", "payload": {"hash": "abc123"}, "ts": now_ms - 1800_000},
        {"kind": "phase_change", "source": "sigild", "payload": {"phase": "coding"}, "ts": now_ms - 1200_000},
    ]


# ===========================================================================
# WP01: Training Data Models
# ===========================================================================


class TestTrainingRun:
    def test_defaults(self) -> None:
        run = TrainingRun(tenant_id="t1", status="trained")
        assert run.tenant_id == "t1"
        assert run.status == "trained"
        assert run.models_trained == []
        assert run.sample_count == 0
        assert run.duration_ms == 0
        assert run.error is None

    def test_to_dict(self) -> None:
        run = TrainingRun(
            tenant_id="t1",
            status="trained",
            models_trained=["stuck", "duration"],
            sample_count=50,
            duration_ms=1200,
        )
        d = run.to_dict()
        assert d["tenant_id"] == "t1"
        assert d["models_trained"] == ["stuck", "duration"]
        assert d["sample_count"] == 50
        assert "error" not in d  # None errors excluded

    def test_to_dict_with_error(self) -> None:
        run = TrainingRun(tenant_id="t1", status="failed", error="boom")
        d = run.to_dict()
        assert d["error"] == "boom"

    def test_to_dict_with_timestamps(self) -> None:
        now = datetime.now(timezone.utc)
        run = TrainingRun(
            tenant_id="t1",
            status="trained",
            started_at=now,
            completed_at=now,
        )
        d = run.to_dict()
        assert "started_at" in d
        assert "completed_at" in d
        assert d["started_at"].endswith("+00:00")

    def test_to_dict_with_data_freshness(self) -> None:
        run = TrainingRun(tenant_id="t1", status="trained", data_freshness_sec=45.2)
        d = run.to_dict()
        assert d["data_freshness_sec"] == 45.2

    def test_to_json(self) -> None:
        run = TrainingRun(tenant_id="t1", status="trained")
        j = run.to_json()
        parsed = json.loads(j)
        assert parsed["tenant_id"] == "t1"

    def test_to_json_compact(self) -> None:
        run = TrainingRun(tenant_id="t1", status="trained")
        j = run.to_json(indent=None)
        assert "\n" not in j

    def test_status_constants(self) -> None:
        assert STATUS_TRAINED == "trained"
        assert STATUS_SKIPPED == "skipped"
        assert STATUS_SKIPPED_LOCKED == "skipped_locked"
        assert STATUS_FAILED == "failed"


class TestTrainingBatch:
    def test_empty(self) -> None:
        batch = TrainingBatch()
        assert batch.total == 0
        assert batch.trained == 0
        assert batch.skipped == 0
        assert batch.failed == 0

    def test_counts(self) -> None:
        batch = TrainingBatch(
            runs=[
                TrainingRun(tenant_id="t1", status="trained"),
                TrainingRun(tenant_id="t2", status="trained"),
                TrainingRun(tenant_id="t3", status="skipped"),
                TrainingRun(tenant_id="t4", status="skipped_locked"),
                TrainingRun(tenant_id="t5", status="failed"),
            ]
        )
        assert batch.total == 5
        assert batch.trained == 2
        assert batch.skipped == 2  # skipped + skipped_locked
        assert batch.failed == 1

    def test_status_breakdown(self) -> None:
        batch = TrainingBatch(
            runs=[
                TrainingRun(tenant_id="t1", status="trained"),
                TrainingRun(tenant_id="t2", status="skipped"),
                TrainingRun(tenant_id="t3", status="skipped_locked"),
                TrainingRun(tenant_id="t4", status="failed"),
                TrainingRun(tenant_id="t5", status="trained"),
            ]
        )
        breakdown = batch.status_breakdown
        assert breakdown["trained"] == 2
        assert breakdown["skipped"] == 1
        assert breakdown["skipped_locked"] == 1
        assert breakdown["failed"] == 1

    def test_to_dict(self) -> None:
        batch = TrainingBatch(
            runs=[TrainingRun(tenant_id="t1", status="trained")],
            total_duration_ms=500,
        )
        d = batch.to_dict()
        assert d["total"] == 1
        assert d["trained"] == 1
        assert d["total_duration_ms"] == 500
        assert "status_breakdown" in d
        assert len(d["runs"]) == 1

    def test_to_json(self) -> None:
        batch = TrainingBatch()
        j = batch.to_json()
        parsed = json.loads(j)
        assert parsed["total"] == 0


class TestCloudTrainingConfig:
    def test_defaults(self) -> None:
        cfg = CloudTrainingConfig()
        assert cfg.min_interval_sec == 3600
        assert cfg.min_tasks == 10
        assert cfg.max_tasks_per_tenant == 1000
        assert cfg.aggregate_min_tenants == 3

    def test_custom_values(self) -> None:
        cfg = CloudTrainingConfig(
            min_interval_sec=1800,
            min_tasks=5,
            max_tasks_per_tenant=500,
            aggregate_min_tenants=2,
        )
        assert cfg.min_interval_sec == 1800
        assert cfg.min_tasks == 5


class TestTrainingSummary:
    def test_to_dict(self) -> None:
        summary = TrainingSummary(
            mode="batch",
            total_tenants=10,
            trained=8,
            skipped=1,
            failed=1,
            total_samples=500,
        )
        d = summary.to_dict()
        assert d["mode"] == "batch"
        assert d["total_tenants"] == 10

    def test_to_json(self) -> None:
        summary = TrainingSummary(mode="single")
        j = summary.to_json()
        parsed = json.loads(j)
        assert parsed["mode"] == "single"


# ===========================================================================
# WP02: Per-Tenant Training Logic
# ===========================================================================


class TestCloudTrainerPerTenant:
    def _make_trainer(
        self,
        tasks: list[dict] | None = None,
        events: dict[str, list[dict]] | None = None,
        last_training_ts: dict[str, int | None] | None = None,
        config: CloudTrainingConfig | None = None,
        lock: Any | None = None,
    ):
        from kenaz_ml.training.cloud_trainer import CloudTrainer

        tenant_tasks = {"t1": tasks or []}
        data_store = MockDataStore(
            tenants=["t1"],
            tasks_per_tenant=tenant_tasks,
            events_per_task=events or {},
            last_training_ts=last_training_ts or {},
        )
        model_store = MockModelStore()
        return CloudTrainer(
            data_store=data_store,
            model_store=model_store,
            config=config or CloudTrainingConfig(),
            training_lock=lock,
        )

    def test_insufficient_data_uses_synthetic(self) -> None:
        """Tenant with < 10 tasks gets synthetic data training."""
        trainer = self._make_trainer(tasks=_make_tasks(5, "t1"))
        run = trainer.train_tenant("t1")
        assert run.status == "trained"
        assert run.sample_count == 500  # synthetic
        assert "stuck" in run.models_trained
        assert "duration" in run.models_trained

    def test_sufficient_data_trains_real(self) -> None:
        """Tenant with >= 10 tasks trains on real data."""
        tasks = _make_tasks(15, "t1")
        events = {t["id"]: _make_events(t["id"]) for t in tasks}
        trainer = self._make_trainer(tasks=tasks, events=events)
        run = trainer.train_tenant("t1")
        assert run.status == "trained"
        assert run.sample_count == 15
        assert "stuck" in run.models_trained
        assert "duration" in run.models_trained

    def test_boundary_exactly_min_tasks(self) -> None:
        """Exactly min_tasks (10) tasks trains on real data, not synthetic."""
        tasks = _make_tasks(10, "t1")
        events = {t["id"]: _make_events(t["id"]) for t in tasks}
        trainer = self._make_trainer(tasks=tasks, events=events)
        run = trainer.train_tenant("t1")
        assert run.status == "trained"
        assert run.sample_count == 10

    def test_custom_threshold(self) -> None:
        """Custom min_tasks=5 lowers the bar."""
        tasks = _make_tasks(7, "t1")
        events = {t["id"]: _make_events(t["id"]) for t in tasks}
        cfg = CloudTrainingConfig(min_tasks=5)
        trainer = self._make_trainer(tasks=tasks, events=events, config=cfg)
        run = trainer.train_tenant("t1")
        assert run.status == "trained"
        assert run.sample_count == 7

    def test_interval_skip_recent(self) -> None:
        """Tenant trained 30 minutes ago is skipped."""
        now_ms = int(time.time() * 1000)
        thirty_min_ago = now_ms - 1800_000  # 30 minutes ago
        trainer = self._make_trainer(
            tasks=_make_tasks(20, "t1"),
            last_training_ts={"t1": thirty_min_ago},
        )
        run = trainer.train_tenant("t1")
        assert run.status == "skipped"

    def test_interval_allow_old(self) -> None:
        """Tenant trained 2 hours ago proceeds."""
        now_ms = int(time.time() * 1000)
        two_hours_ago = now_ms - 7200_000
        tasks = _make_tasks(15, "t1")
        events = {t["id"]: _make_events(t["id"]) for t in tasks}
        trainer = self._make_trainer(
            tasks=tasks,
            events=events,
            last_training_ts={"t1": two_hours_ago},
        )
        run = trainer.train_tenant("t1")
        assert run.status == "trained"

    def test_never_trained_proceeds(self) -> None:
        """Tenant never trained (None timestamp) proceeds."""
        tasks = _make_tasks(15, "t1")
        events = {t["id"]: _make_events(t["id"]) for t in tasks}
        trainer = self._make_trainer(
            tasks=tasks,
            events=events,
            last_training_ts={"t1": None},
        )
        run = trainer.train_tenant("t1")
        assert run.status == "trained"

    def test_duration_ms_captured(self) -> None:
        """TrainingRun captures duration."""
        trainer = self._make_trainer(tasks=_make_tasks(5, "t1"))
        run = trainer.train_tenant("t1")
        assert run.duration_ms >= 0

    def test_error_returns_failed_run(self) -> None:
        """Errors during training return a failed TrainingRun."""
        from kenaz_ml.training.cloud_trainer import CloudTrainer

        # DataStore that raises on get_completed_tasks_for_tenant
        data_store = MockDataStore()
        data_store.get_completed_tasks_for_tenant = MagicMock(side_effect=ValueError("boom"))
        model_store = MockModelStore()
        trainer = CloudTrainer(data_store, model_store)
        run = trainer.train_tenant("t1")
        assert run.status == "failed"
        assert "boom" in run.error

    def test_models_saved_to_model_store(self) -> None:
        """Weights are saved via ModelStore with tenant-scoped prefix."""
        tasks = _make_tasks(15, "t1")
        events = {t["id"]: _make_events(t["id"]) for t in tasks}
        trainer = self._make_trainer(tasks=tasks, events=events)
        model_store = trainer.model_store
        run = trainer.train_tenant("t1")
        assert run.status == "trained"
        # Check model store has tenant-scoped keys
        assert model_store.exists("t1/stuck")
        assert model_store.exists("t1/duration")


# ===========================================================================
# WP03: Batch Training & Tenant Discovery
# ===========================================================================


class TestBatchTraining:
    def test_batch_multiple_tenants(self) -> None:
        from kenaz_ml.training.cloud_trainer import CloudTrainer

        tasks_t1 = _make_tasks(15, "t1")
        tasks_t2 = _make_tasks(15, "t2")
        events_t1 = {t["id"]: _make_events(t["id"]) for t in tasks_t1}
        events_t2 = {t["id"]: _make_events(t["id"]) for t in tasks_t2}

        data_store = MockDataStore(
            tenants=["t1", "t2"],
            tasks_per_tenant={"t1": tasks_t1, "t2": tasks_t2},
            events_per_task={**events_t1, **events_t2},
        )
        model_store = MockModelStore()
        trainer = CloudTrainer(data_store, model_store)
        batch = trainer.train_all_tenants()

        assert batch.total == 2
        assert batch.trained == 2
        assert batch.failed == 0

    def test_batch_zero_tenants(self) -> None:
        from kenaz_ml.training.cloud_trainer import CloudTrainer

        data_store = MockDataStore(tenants=[])
        model_store = MockModelStore()
        trainer = CloudTrainer(data_store, model_store)
        batch = trainer.train_all_tenants()

        assert batch.total == 0
        assert batch.trained == 0
        assert batch.skipped == 0
        assert batch.failed == 0

    def test_batch_fault_isolation(self) -> None:
        """One tenant's failure doesn't prevent others from training."""
        from kenaz_ml.training.cloud_trainer import CloudTrainer

        tasks_t2 = _make_tasks(15, "t2")
        events_t2 = {t["id"]: _make_events(t["id"]) for t in tasks_t2}

        data_store = MockDataStore(
            tenants=["t1", "t2"],
            tasks_per_tenant={"t1": [], "t2": tasks_t2},
            events_per_task=events_t2,
        )
        # Make t1 fail by having get_completed_tasks_for_tenant raise for t1
        original_query = data_store.get_completed_tasks_for_tenant

        def failing_query(tenant_id: str) -> list[dict]:
            if tenant_id == "t1":
                raise ConnectionError("DB unreachable for t1")
            return original_query(tenant_id)

        data_store.get_completed_tasks_for_tenant = failing_query  # type: ignore[assignment]
        model_store = MockModelStore()
        trainer = CloudTrainer(data_store, model_store)
        batch = trainer.train_all_tenants()

        assert batch.total == 2
        assert batch.trained == 1  # t2 succeeded
        assert batch.failed == 1  # t1 failed

        # Verify t1 is the failed one
        t1_run = [r for r in batch.runs if r.tenant_id == "t1"][0]
        assert t1_run.status == "failed"
        t2_run = [r for r in batch.runs if r.tenant_id == "t2"][0]
        assert t2_run.status == "trained"

    def test_batch_all_skipped(self) -> None:
        """All tenants skipped (recently trained) returns clean batch."""
        from kenaz_ml.training.cloud_trainer import CloudTrainer

        now_ms = int(time.time() * 1000)
        recent = now_ms - 600_000  # 10 minutes ago

        data_store = MockDataStore(
            tenants=["t1", "t2"],
            tasks_per_tenant={"t1": _make_tasks(15, "t1"), "t2": _make_tasks(15, "t2")},
            last_training_ts={"t1": recent, "t2": recent},
        )
        model_store = MockModelStore()
        trainer = CloudTrainer(data_store, model_store)
        batch = trainer.train_all_tenants()

        assert batch.total == 2
        assert batch.trained == 0
        assert batch.skipped == 2
        assert batch.failed == 0


class TestTenantDiscovery:
    def test_discover_eligible(self) -> None:
        from kenaz_ml.training.tenant_discovery import discover_eligible_tenants

        ds = MockDataStore(tenants=["a", "b", "c"])
        result = discover_eligible_tenants(ds)
        assert result == ["a", "b", "c"]

    def test_discover_empty(self) -> None:
        from kenaz_ml.training.tenant_discovery import discover_eligible_tenants

        ds = MockDataStore(tenants=[])
        result = discover_eligible_tenants(ds)
        assert result == []

    def test_discover_opted_in(self) -> None:
        from kenaz_ml.training.tenant_discovery import discover_opted_in_tenants

        ds = MockDataStore(opted_in_tenants=["a", "c"])
        result = discover_opted_in_tenants(ds)
        assert result == ["a", "c"]


# ===========================================================================
# WP04: Concurrent Training Lock
# ===========================================================================


class TestTrainingLock:
    def test_lock_protocol(self) -> None:
        from kenaz_ml.training.locking import TrainingLock

        lock = MockTrainingLock()
        assert isinstance(lock, TrainingLock)

    def test_lock_acquire_release(self) -> None:
        lock = MockTrainingLock()
        assert lock.acquire("t1") is True
        lock.release("t1")

    def test_lock_blocked(self) -> None:
        lock = MockTrainingLock(locked_tenants={"t1"})
        assert lock.acquire("t1") is False
        assert lock.acquire("t2") is True  # different tenant OK

    def test_trainer_with_lock_skips_locked(self) -> None:
        """train_tenant returns skipped_locked when lock is held."""
        from kenaz_ml.training.cloud_trainer import CloudTrainer

        lock = MockTrainingLock(locked_tenants={"t1"})
        data_store = MockDataStore(
            tenants=["t1"],
            tasks_per_tenant={"t1": _make_tasks(20, "t1")},
        )
        model_store = MockModelStore()
        trainer = CloudTrainer(data_store, model_store, training_lock=lock)
        run = trainer.train_tenant("t1")
        assert run.status == "skipped_locked"

    def test_trainer_without_lock_works(self) -> None:
        """training_lock=None means no locking overhead."""
        from kenaz_ml.training.cloud_trainer import CloudTrainer

        data_store = MockDataStore(
            tenants=["t1"],
            tasks_per_tenant={"t1": _make_tasks(5, "t1")},
        )
        model_store = MockModelStore()
        trainer = CloudTrainer(data_store, model_store, training_lock=None)
        run = trainer.train_tenant("t1")
        assert run.status == "trained"  # proceeds without lock

    def test_lock_released_on_failure(self) -> None:
        """Lock is released even if training fails."""
        from kenaz_ml.training.cloud_trainer import CloudTrainer

        lock = MockTrainingLock()
        data_store = MockDataStore()
        data_store.get_completed_tasks_for_tenant = MagicMock(side_effect=RuntimeError("db error"))
        model_store = MockModelStore()
        trainer = CloudTrainer(data_store, model_store, training_lock=lock)
        run = trainer.train_tenant("t1")
        assert run.status == "failed"
        # Lock should have been released (not in acquired set)
        assert "t1" not in lock._acquired

    def test_lock_checked_before_interval(self) -> None:
        """Lock check happens before interval check (cheapest first)."""
        from kenaz_ml.training.cloud_trainer import CloudTrainer

        lock = MockTrainingLock(locked_tenants={"t1"})
        now_ms = int(time.time() * 1000)
        # Even though interval would allow training, lock prevents it
        data_store = MockDataStore(
            tenants=["t1"],
            tasks_per_tenant={"t1": _make_tasks(20, "t1")},
            last_training_ts={"t1": now_ms - 7200_000},  # 2 hours ago, would be OK
        )
        model_store = MockModelStore()
        trainer = CloudTrainer(data_store, model_store, training_lock=lock)
        run = trainer.train_tenant("t1")
        assert run.status == "skipped_locked"


class TestDataStoreTrainingLock:
    def test_acquire_delegates_to_data_store(self) -> None:
        from kenaz_ml.training.locking import DataStoreTrainingLock

        ds = MagicMock()
        ds.acquire_training_lock = MagicMock(return_value=True)
        lock = DataStoreTrainingLock(ds, stale_timeout_sec=7200)
        result = lock.acquire("t1")
        assert result is True
        ds.acquire_training_lock.assert_called_once()

    def test_release_delegates_to_data_store(self) -> None:
        from kenaz_ml.training.locking import DataStoreTrainingLock

        ds = MagicMock()
        ds.release_training_lock = MagicMock()
        lock = DataStoreTrainingLock(ds)
        lock.release("t1")
        ds.release_training_lock.assert_called_once_with("t1")

    def test_acquire_graceful_no_method(self) -> None:
        """If DataStore doesn't have lock methods, treat as acquired."""
        from kenaz_ml.training.locking import DataStoreTrainingLock

        ds = MockDataStore()  # doesn't have acquire_training_lock
        # Remove the method if it exists
        if hasattr(ds, "acquire_training_lock"):
            delattr(ds, "acquire_training_lock")
        lock = DataStoreTrainingLock(ds)
        result = lock.acquire("t1")
        assert result is True  # graceful fallback

    def test_release_graceful_no_method(self) -> None:
        """If DataStore doesn't have release method, no-op."""
        from kenaz_ml.training.locking import DataStoreTrainingLock

        ds = MockDataStore()
        if hasattr(ds, "release_training_lock"):
            delattr(ds, "release_training_lock")
        lock = DataStoreTrainingLock(ds)
        # Should not raise
        lock.release("t1")


# ===========================================================================
# WP05: Aggregate Model Training
# ===========================================================================


class TestAggregateTraining:
    def test_aggregate_pools_opted_in_data(self) -> None:
        from kenaz_ml.training.cloud_trainer import AGGREGATE_TENANT_ID, CloudTrainer

        tasks_t1 = _make_tasks(15, "t1")
        tasks_t2 = _make_tasks(15, "t2")
        events_t1 = {t["id"]: _make_events(t["id"]) for t in tasks_t1}
        events_t2 = {t["id"]: _make_events(t["id"]) for t in tasks_t2}

        data_store = MockDataStore(
            opted_in_tenants=["t1", "t2"],
            tasks_per_tenant={"t1": tasks_t1, "t2": tasks_t2},
            events_per_task={**events_t1, **events_t2},
        )
        model_store = MockModelStore()
        trainer = CloudTrainer(data_store, model_store)
        run = trainer.train_aggregate()

        assert run.tenant_id == AGGREGATE_TENANT_ID
        assert run.status == "trained"
        assert run.sample_count == 30  # 15 + 15
        assert "stuck" in run.models_trained
        assert "duration" in run.models_trained

    def test_aggregate_zero_tenants(self) -> None:
        from kenaz_ml.training.cloud_trainer import AGGREGATE_TENANT_ID, CloudTrainer

        data_store = MockDataStore(opted_in_tenants=[])
        model_store = MockModelStore()
        trainer = CloudTrainer(data_store, model_store)
        run = trainer.train_aggregate()

        assert run.tenant_id == AGGREGATE_TENANT_ID
        assert run.status == "skipped"
        assert "No opted-in tenants" in run.error

    def test_aggregate_low_tenant_warning(self) -> None:
        """Fewer than aggregate_min_tenants warns but proceeds."""
        from kenaz_ml.training.cloud_trainer import CloudTrainer

        tasks = _make_tasks(15, "t1")
        events = {t["id"]: _make_events(t["id"]) for t in tasks}

        data_store = MockDataStore(
            opted_in_tenants=["t1"],
            tasks_per_tenant={"t1": tasks},
            events_per_task=events,
        )
        model_store = MockModelStore()
        cfg = CloudTrainingConfig(aggregate_min_tenants=3)
        trainer = CloudTrainer(data_store, model_store, cfg)
        run = trainer.train_aggregate()

        assert run.status == "trained"  # proceeds despite warning
        assert run.error is not None
        assert "recommended minimum" in run.error

    def test_aggregate_enough_tenants_no_warning(self) -> None:
        from kenaz_ml.training.cloud_trainer import CloudTrainer

        all_tasks = {}
        all_events = {}
        tenants = ["t1", "t2", "t3"]
        for t in tenants:
            tasks = _make_tasks(15, t)
            all_tasks[t] = tasks
            for task in tasks:
                all_events[task["id"]] = _make_events(task["id"])

        data_store = MockDataStore(
            opted_in_tenants=tenants,
            tasks_per_tenant=all_tasks,
            events_per_task=all_events,
        )
        model_store = MockModelStore()
        cfg = CloudTrainingConfig(aggregate_min_tenants=3)
        trainer = CloudTrainer(data_store, model_store, cfg)
        run = trainer.train_aggregate()

        assert run.status == "trained"
        assert run.error is None  # no warning

    def test_aggregate_sampling_caps(self) -> None:
        """Per-tenant sampling cap limits contribution."""
        from kenaz_ml.training.cloud_trainer import CloudTrainer

        # t1 has 50 tasks, t2 has 5. Cap at 10.
        tasks_t1 = _make_tasks(50, "t1")
        tasks_t2 = _make_tasks(5, "t2")
        events = {}
        for t in tasks_t1 + tasks_t2:
            events[t["id"]] = _make_events(t["id"])

        data_store = MockDataStore(
            opted_in_tenants=["t1", "t2"],
            tasks_per_tenant={"t1": tasks_t1, "t2": tasks_t2},
            events_per_task=events,
        )
        model_store = MockModelStore()
        cfg = CloudTrainingConfig(max_tasks_per_tenant=10, aggregate_min_tenants=1)
        trainer = CloudTrainer(data_store, model_store, cfg)
        run = trainer.train_aggregate()

        assert run.status == "trained"
        # t1 capped at 10, t2 all 5 = 15 total
        assert run.sample_count == 15

    def test_aggregate_model_saved_to_aggregate_prefix(self) -> None:
        from kenaz_ml.training.cloud_trainer import AGGREGATE_TENANT_ID, CloudTrainer

        tasks = _make_tasks(15, "t1")
        events = {t["id"]: _make_events(t["id"]) for t in tasks}

        data_store = MockDataStore(
            opted_in_tenants=["t1"],
            tasks_per_tenant={"t1": tasks},
            events_per_task=events,
        )
        model_store = MockModelStore()
        cfg = CloudTrainingConfig(aggregate_min_tenants=1)
        trainer = CloudTrainer(data_store, model_store, cfg)
        run = trainer.train_aggregate()

        assert run.status == "trained"
        # Models saved with aggregate prefix
        assert model_store.exists(f"{AGGREGATE_TENANT_ID}/stuck")
        assert model_store.exists(f"{AGGREGATE_TENANT_ID}/duration")


# ===========================================================================
# WP06: Observability
# ===========================================================================


class TestObservability:
    def test_audit_event_on_success(self) -> None:
        """Successful training records an audit event."""
        from kenaz_ml.training.cloud_trainer import CloudTrainer

        tasks = _make_tasks(15, "t1")
        events = {t["id"]: _make_events(t["id"]) for t in tasks}
        data_store = MockDataStore(
            tenants=["t1"],
            tasks_per_tenant={"t1": tasks},
            events_per_task=events,
        )
        model_store = MockModelStore()
        trainer = CloudTrainer(data_store, model_store)
        trainer.train_tenant("t1")

        assert len(data_store._ml_events) > 0
        event = data_store._ml_events[0]
        assert event["kind"] == "training"
        assert event["routing"] == "t1"

    def test_audit_event_on_failure(self) -> None:
        """Failed training records an audit event."""
        from kenaz_ml.training.cloud_trainer import CloudTrainer

        data_store = MockDataStore()
        data_store.get_completed_tasks_for_tenant = MagicMock(side_effect=ValueError("fail"))
        model_store = MockModelStore()
        trainer = CloudTrainer(data_store, model_store)
        run = trainer.train_tenant("t1")
        assert run.status == "failed"
        assert len(data_store._ml_events) > 0

    def test_audit_event_on_skip(self) -> None:
        """Skipped training records an audit event."""
        from kenaz_ml.training.cloud_trainer import CloudTrainer

        now_ms = int(time.time() * 1000)
        data_store = MockDataStore(
            tenants=["t1"],
            tasks_per_tenant={"t1": _make_tasks(15, "t1")},
            last_training_ts={"t1": now_ms - 600_000},
        )
        model_store = MockModelStore()
        trainer = CloudTrainer(data_store, model_store)
        run = trainer.train_tenant("t1")
        assert run.status == "skipped"
        assert len(data_store._ml_events) > 0

    def test_batch_audit_event(self) -> None:
        """Batch training records a batch-level audit event."""
        from kenaz_ml.training.cloud_trainer import CloudTrainer

        data_store = MockDataStore(tenants=["t1"])
        data_store._tasks["t1"] = _make_tasks(5, "t1")
        model_store = MockModelStore()
        trainer = CloudTrainer(data_store, model_store)
        trainer.train_all_tenants()

        # Should have per-tenant + batch-level events
        batch_events = [e for e in data_store._ml_events if e["kind"] == "batch_training"]
        assert len(batch_events) == 1
        assert batch_events[0]["routing"] == "__batch__"

    def test_training_run_json_valid(self) -> None:
        """TrainingRun produces valid JSON."""
        run = TrainingRun(
            tenant_id="t1",
            status="trained",
            models_trained=["stuck", "duration"],
            sample_count=100,
            duration_ms=5000,
        )
        j = run.to_json()
        parsed = json.loads(j)
        assert parsed["tenant_id"] == "t1"
        assert parsed["sample_count"] == 100

    def test_batch_json_valid(self) -> None:
        """TrainingBatch produces valid JSON."""
        batch = TrainingBatch(
            runs=[
                TrainingRun(tenant_id="t1", status="trained", models_trained=["stuck"]),
                TrainingRun(tenant_id="t2", status="failed", error="boom"),
            ],
            total_duration_ms=3000,
        )
        j = batch.to_json()
        parsed = json.loads(j)
        assert parsed["total"] == 2
        assert parsed["trained"] == 1
        assert parsed["failed"] == 1
        assert len(parsed["runs"]) == 2


# ===========================================================================
# Feature Extraction (cloud-compatible)
# ===========================================================================


class TestCloudFeatureExtraction:
    def test_stuck_features_from_data(self) -> None:
        from kenaz_ml.features import extract_stuck_features_from_data

        now_ms = int(time.time() * 1000)
        task = {
            "started_at": now_ms - 3600_000,
            "last_active": now_ms - 60_000,
            "test_fails": 3,
        }
        events = [
            {"kind": "edit", "payload": {"file": "main.py"}, "ts": now_ms - 3000_000},
            {"kind": "edit", "payload": {"file": "utils.py"}, "ts": now_ms - 2800_000},
            {"kind": "commit", "payload": {}, "ts": now_ms - 1800_000},
            {"kind": "phase_change", "payload": {}, "ts": now_ms - 1200_000},
        ]
        feats = extract_stuck_features_from_data(task, events)
        assert feats["test_failure_count"] == 3.0
        assert feats["session_length_sec"] > 0
        assert feats["edit_velocity"] >= 0
        assert 0 <= feats["file_switch_rate"] <= 1.0

    def test_stuck_features_empty_events(self) -> None:
        from kenaz_ml.features import extract_stuck_features_from_data

        now_ms = int(time.time() * 1000)
        task = {
            "started_at": now_ms - 3600_000,
            "last_active": now_ms - 60_000,
            "test_fails": 0,
        }
        feats = extract_stuck_features_from_data(task, [])
        assert feats["test_failure_count"] == 0.0
        assert feats["edit_velocity"] == 0.0

    def test_stuck_features_no_started_at(self) -> None:
        from kenaz_ml.features import extract_stuck_features_from_data

        feats = extract_stuck_features_from_data({}, [])
        assert feats["session_length_sec"] >= 1.0  # min 1.0

    def test_duration_features_from_data(self) -> None:
        from kenaz_ml.features import extract_duration_features_from_data

        now_ms = int(time.time() * 1000)
        task = {
            "started_at": now_ms - 3600_000,
            "branch": "feature/add-widget",
            "files": json.dumps({"main.py": 5, "utils.py": 3}),
        }
        events = [
            {"kind": "edit", "payload": {"file": "main.py"}, "ts": now_ms - 3000_000},
            {"kind": "save", "payload": {"file": "main.py"}, "ts": now_ms - 2500_000},
        ]
        feats = extract_duration_features_from_data(task, events)
        assert feats["file_count"] == 2.0
        assert feats["total_edits"] == 2.0
        assert 0 <= feats["time_of_day_hour"] < 24
        assert feats["branch_name_length"] == len("feature/add-widget")

    def test_duration_features_missing_files(self) -> None:
        from kenaz_ml.features import extract_duration_features_from_data

        feats = extract_duration_features_from_data({}, [])
        assert feats["file_count"] == 0.0
        assert feats["branch_name_length"] == 0.0


# ===========================================================================
# CLI Cloud Training Flags
# ===========================================================================


class TestCLICloudFlags:
    def test_cloud_mode_requires_target(self) -> None:
        """--mode cloud without --tenant/--all-tenants/--aggregate errors."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "kenaz_ml.cli", "train", "--mode", "cloud"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "requires --tenant" in result.stderr

    def test_cloud_mode_mutual_exclusivity(self) -> None:
        """--tenant and --all-tenants together errors."""
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "kenaz_ml.cli",
                "train",
                "--mode",
                "cloud",
                "--tenant",
                "t1",
                "--all-tenants",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "mutually exclusive" in result.stderr

    def test_cloud_mode_missing_env_vars(self) -> None:
        """Missing KENAZ_POSTGRES_URL errors."""
        import subprocess

        env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "kenaz_ml.cli",
                "train",
                "--mode",
                "cloud",
                "--tenant",
                "t1",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0
        assert "KENAZ_POSTGRES_URL" in result.stderr

    def test_local_mode_default(self) -> None:
        """train --help shows --mode flag."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "kenaz_ml.cli", "train", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--mode" in result.stdout
        assert "--tenant" in result.stdout
        assert "--all-tenants" in result.stdout
        assert "--aggregate" in result.stdout
        assert "--min-interval" in result.stdout
        assert "--min-tasks" in result.stdout
        assert "--json" in result.stdout
