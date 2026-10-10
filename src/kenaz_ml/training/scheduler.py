"""Training scheduler — fires retraining when enough new data has accumulated."""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from kenaz_ml.datastore import DataStore
from kenaz_ml.modelstore import ModelStore
from kenaz_ml.training.trainer import Trainer

logger = logging.getLogger(__name__)

MIN_NEW_TASKS = 10  # retrain after 10 new completed tasks
MIN_INTERVAL_SEC = 3600  # no more than once per hour
RETENTION_SWEEP_SEC = 86_400  # expire aged retained examples at least daily
_DAY_MS = 86_400_000


class RetainedMirror(Protocol):
    """Where a stream's retained sets persist when local disk does not (S3)."""

    def pull(self, directory: Path) -> int: ...

    def push(self, directory: Path) -> int: ...


class TrainingScheduler:
    """Monitors completed task count and triggers retraining when due.

    Args:
        store: DataStore instance for data access.
        reload_callback: Called after successful retraining to reload
                         model instances into the running poller.
    """

    def __init__(
        self,
        store: DataStore,
        model_store: ModelStore | None = None,
        reload_callback: Callable[[], None] | None = None,
        *,
        retained_dir: Path | None = None,
        retained_mirror: RetainedMirror | None = None,
        retention_days: Callable[[], int | None] | None = None,
    ) -> None:
        if retained_mirror is not None and retained_dir is None:
            raise ValueError("a retained mirror needs a retained_dir to work in")
        self.store = store
        self._model_store = model_store
        self._retained_dir = retained_dir
        self._mirror = retained_mirror
        self._retention_days = retention_days
        self._last_sweep: float = 0.0
        self._reload = reload_callback
        self._last_retrain: float = 0.0
        self._baseline_tasks: int = self._count_completed()

    def check_and_retrain(self) -> None:
        """Call from a background thread — blocks while training runs."""
        elapsed = time.time() - self._last_retrain
        if elapsed < MIN_INTERVAL_SEC and self._last_retrain > 0:
            return

        current = self._count_completed()
        if (current - self._baseline_tasks) < MIN_NEW_TASKS:
            if time.time() - self._last_sweep >= RETENTION_SWEEP_SEC:
                self.sweep_retention()
            return

        logger.info(
            "scheduler: triggering retrain (%d new tasks)",
            current - self._baseline_tasks,
        )
        try:
            result = self._with_retained(
                lambda: Trainer(self.store, model_store=self._model_store, retained_dir=self._retained_dir).train_all()
            )
            self._last_retrain = time.time()
            self._baseline_tasks = current
            self._log_retrain(result)
            if self._reload is not None:
                self._reload()
            logger.info("scheduler: retrain complete — %s", result)
        except Exception:
            logger.exception("scheduler: retrain failed")

    def _expire(self) -> int:
        """Drop retained examples older than the retention period, if one is set."""
        days = self._retention_days() if self._retention_days is not None else None
        if not days or self._retained_dir is None or not self._retained_dir.exists():
            return 0
        from kenaz_ml.modelstore.registry import expire_examples

        cutoff = int(time.time() * 1000) - int(days) * _DAY_MS
        return sum(
            expire_examples(path.stem, before_ms=cutoff, directory=self._retained_dir)
            for path in sorted(self._retained_dir.glob("*.jsonl"))
        )

    def _with_retained[T](self, run: Callable[[], T]) -> T:
        """Run ``run`` with this stream's retained sets in place and in date.

        With a mirror: pull them into the scratch directory, expire, run,
        expire again, push, and always clear the scratch copy, so the only
        durable copy is the mirror's.
        """
        if self._mirror is None or self._retained_dir is None:
            self._expire()
            result = run()
            self._expire()
            return result
        try:
            self._mirror.pull(self._retained_dir)
            self._expire()
            result = run()
            self._expire()
            self._mirror.push(self._retained_dir)
            return result
        finally:
            shutil.rmtree(self._retained_dir, ignore_errors=True)

    def sweep_retention(self) -> None:
        """Expire aged retained examples without training. Never raises."""
        self._last_sweep = time.time()
        if self._retention_days is None or self._retained_dir is None:
            return
        try:
            self._with_retained(lambda: None)
        except Exception:
            logger.warning("scheduler: retention sweep failed (will retry)", exc_info=True)

    def _count_completed(self) -> int:
        try:
            return self.store.count_completed_tasks()
        except Exception:
            return 0

    def _log_retrain(self, result: dict) -> None:
        try:
            latency_ms = int(result.get("duration_sec", 0) * 1000)
            self.store.insert_ml_event("retrain", "scheduler", "local", latency_ms)
            self.store.commit()
        except Exception:
            logger.warning("scheduler: failed to log retrain event")
