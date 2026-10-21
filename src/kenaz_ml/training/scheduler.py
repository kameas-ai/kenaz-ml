"""Training scheduler — fires retraining when enough new data has accumulated."""

from __future__ import annotations

import logging
import shutil
import threading
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
CHECK_INTERVAL_SEC = 600  # how often the serving loop calls check_and_retrain
# Models train_all fits (from local or synthetic data) whose missing weights
# make the very first check train at once instead of waiting for new tasks.
FIRST_RUN_MODELS = ("stuck", "duration")
# The ml_events row every successful retrain writes; its ts persists the
# baseline across restarts (the completed-task count at that moment).
RETRAIN_EVENT_KIND = "retrain"
RETRAIN_EVENT_ENDPOINT = "scheduler"
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
        check_interval_sec: float = CHECK_INTERVAL_SEC,
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
        self.check_interval_sec = check_interval_sec
        self._last_retrain: float = 0.0
        # Set once every first-run model has weights; stops re-probing the
        # model store (an S3 round trip in the cloud) on every check.
        self._weights_present = False
        self._last_check_at: float | None = None
        # "ok" | "failed" | None (no retrain known to this store yet).
        self._last_result: str | None = None
        self._baseline_tasks: int = self._restore_baseline()
        # One training run (or retention sweep) at a time per stream: the
        # scheduled check and a forced retrain share the retained directory.
        self._lock = threading.Lock()
        # Forced retrains called but still waiting for the lock: they count as
        # busy, so a run queued behind another is never invisible between the
        # first releasing the lock and the second taking it.
        self._waiting = 0
        self._waiting_lock = threading.Lock()

    @property
    def busy(self) -> bool:
        """True while a training run or retention sweep is in progress, or a
        forced retrain is queued for one.

        A host that stops a stream uses this to know when the stream's last
        write has landed: a run executes in a thread that cannot be cancelled.
        """
        with self._waiting_lock:
            waiting = self._waiting
        return waiting > 0 or self._lock.locked()

    def completed_task_count(self) -> int:
        """Completed tasks this scheduler's store can see (0 when unreadable)."""
        return self._count_completed()

    def progress(self) -> dict:
        """What the next retrain is waiting for, for ``/introspect``.

        ``completed_tasks - baseline`` new completed tasks out of
        ``threshold`` trigger a retrain at the next check. Times are unix
        seconds; ``next_check_at`` is None until the first check has run.
        """
        next_check = None if self._last_check_at is None else self._last_check_at + self.check_interval_sec
        return {
            "completed_tasks": self._count_completed(),
            "baseline": self._baseline_tasks,
            "threshold": MIN_NEW_TASKS,
            "last_result": self._last_result,
            "last_retrain_at": self._last_retrain or None,
            "next_check_at": next_check,
            "in_progress": self.busy,
        }

    def _restore_baseline(self) -> int:
        """Rebuild the new-task baseline from the last logged retrain.

        The baseline is the completed-task count when the models were last
        trained. Every successful retrain writes an ``ml_events`` row, so the
        baseline survives a restart as "tasks completed at or before that
        row's ts" -- read through the DataStore, on either SQLite shape or
        Postgres. With no such row (never retrained, or the row was pruned by
        activity retention) the current count is the baseline, as before; a
        store with no weights at all trains on the first check regardless.
        """
        try:
            last_ts = self.store.last_ml_event_ts(RETRAIN_EVENT_KIND, RETRAIN_EVENT_ENDPOINT)
        except Exception:
            logger.warning("scheduler: could not read the last retrain event; baseline is the current count")
            last_ts = None
        if last_ts is None:
            return self._count_completed()
        try:
            baseline = int(self.store.count_completed_tasks_before(last_ts))
        except Exception:
            logger.warning("scheduler: could not count tasks before the last retrain; baseline is the current count")
            return self._count_completed()
        self._last_retrain = last_ts / 1000.0
        self._last_result = "ok"
        return baseline

    def _missing_weights(self) -> list[str]:
        """Models in :data:`FIRST_RUN_MODELS` with weights in neither the local nor the base slot."""
        from kenaz_ml.modelstore import LocalModelStore
        from kenaz_ml.modelstore.registry.slots import base_slot_dir, slot_is_empty

        if self._weights_present:
            return []
        ms = self._model_store if self._model_store is not None else LocalModelStore()
        missing = []
        unknown = False
        for name in FIRST_RUN_MODELS:
            try:
                if ms.exists(name):
                    continue
            except Exception:
                # Unreadable is not "missing": never train on a store we
                # cannot see into, but probe again next check.
                logger.warning("scheduler: could not check stored weights for %s", name)
                unknown = True
                continue
            try:
                if not slot_is_empty(base_slot_dir(), name):
                    continue
            except Exception:
                pass
            missing.append(name)
        self._weights_present = not missing and not unknown
        return missing

    def check_and_retrain(self) -> None:
        """Call from a background thread — blocks while training runs.

        Returns at once when another run (a forced retrain) is in progress.
        """
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._check_and_retrain_locked()
        finally:
            self._lock.release()

    def force_retrain(self) -> dict:
        """Retrain every model now, ignoring the new-task threshold and interval.

        For a host that must replace the stored models promptly, e.g. after a
        member's data was erased from the data the models were trained on.
        The run reads only what the store and the retained mirror hold *now*:
        the mirror is pulled afresh (stale local copies are removed first), so
        nothing deleted before this call is read by it.

        Waits for any run already in progress. Raises if training fails, so
        the caller can retry; returns :meth:`Trainer.train_all`'s summary,
        whose ``"synthetic"`` list names the models fitted on synthetic data
        because this stream had too little of its own.
        """
        with self._waiting_lock:
            self._waiting += 1
        try:
            self._lock.acquire()
        finally:
            with self._waiting_lock:
                self._waiting -= 1
        try:
            current = self._count_completed()
            result = self._with_retained(
                lambda: Trainer(self.store, model_store=self._model_store, retained_dir=self._retained_dir).train_all()
            )
            self._last_retrain = time.time()
            self._baseline_tasks = current
            self._last_result = "ok"
            self._log_retrain(result)
            if self._reload is not None:
                try:
                    self._reload()
                except Exception:
                    logger.warning("scheduler: reload after forced retrain failed", exc_info=True)
            logger.info("scheduler: forced retrain complete — %s", result)
            return result
        except Exception:
            self._last_result = "failed"
            raise
        finally:
            self._lock.release()

    def _check_and_retrain_locked(self) -> None:
        self._last_check_at = time.time()
        missing = self._missing_weights()
        # Missing weights skip the hourly interval: until a first run lands
        # (e.g. the data store's tables do not exist yet) it is retried at
        # every check, as a failed retrain always has been.
        elapsed = time.time() - self._last_retrain
        if not missing and elapsed < MIN_INTERVAL_SEC and self._last_retrain > 0:
            return

        current = self._count_completed()
        if missing:
            # First run: nothing to serve but heuristic defaults. train_all
            # fits these from local data, or bootstraps them from synthetic
            # data when there is too little, so train now rather than after
            # MIN_NEW_TASKS more tasks.
            logger.info("scheduler: no weights for %s; training on first run", ", ".join(missing))
        elif (current - self._baseline_tasks) < MIN_NEW_TASKS:
            if time.time() - self._last_sweep >= RETENTION_SWEEP_SEC:
                self._sweep_retention_locked()
            return
        else:
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
            self._last_result = "ok"
            self._log_retrain(result)
            if self._reload is not None:
                self._reload()
            logger.info("scheduler: retrain complete — %s", result)
        except Exception:
            self._last_result = "failed"
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
        """Expire aged retained examples without training. Never raises.

        Skipped (returns at once) while a training run is in progress; that
        run expires the same examples on its way in and out.
        """
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._sweep_retention_locked()
        finally:
            self._lock.release()

    def _sweep_retention_locked(self) -> None:
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
            self.store.insert_ml_event(RETRAIN_EVENT_KIND, RETRAIN_EVENT_ENDPOINT, "local", latency_ms)
            self.store.commit()
        except Exception:
            logger.warning("scheduler: failed to log retrain event")
