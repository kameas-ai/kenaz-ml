"""Training scheduler — fires retraining when enough new data has accumulated."""

from __future__ import annotations

import logging
import time

from sigil_ml.datastore import DataStore
from sigil_ml.modelstore import ModelStore
from sigil_ml.training.trainer import Trainer

logger = logging.getLogger(__name__)

MIN_NEW_TASKS = 10  # retrain after 10 new completed tasks
MIN_INTERVAL_SEC = 3600  # no more than once per hour


class TrainingScheduler:
    """Monitors completed task count and triggers retraining when due.

    Args:
        store: DataStore instance for data access.
        reload_callback: Called after successful retraining to reload
                         model instances into the running poller.
    """

    def __init__(self, store: DataStore, model_store: ModelStore | None = None, reload_callback=None) -> None:
        self.store = store
        self._model_store = model_store
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
            return

        logger.info(
            "scheduler: triggering retrain (%d new tasks)",
            current - self._baseline_tasks,
        )
        try:
            result = Trainer(self.store, model_store=self._model_store).train_all()
            self._last_retrain = time.time()
            self._baseline_tasks = current
            self._log_retrain(result)
            self._reload()
            logger.info("scheduler: retrain complete — %s", result)
        except Exception:
            logger.exception("scheduler: retrain failed")

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
