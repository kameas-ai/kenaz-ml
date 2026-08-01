"""Training lock protocol and implementations for preventing concurrent training."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sigil_ml.datastore import DataStore

logger = logging.getLogger(__name__)

# Default stale lock timeout: 2 hours (configurable via env var)
STALE_LOCK_TIMEOUT_SEC = int(os.environ.get("SIGIL_ML_LOCK_TIMEOUT_SEC", "7200"))


@runtime_checkable
class TrainingLock(Protocol):
    """Protocol for distributed training locks.

    Prevents concurrent training for the same tenant across
    multiple processes/pods. Implementations must be safe for
    use from multiple K8s pods accessing the same database.
    """

    def acquire(self, tenant_id: str) -> bool:
        """Attempt to acquire a training lock for the given tenant.

        Returns:
            True if the lock was acquired (caller proceeds with training).
            False if another process holds a fresh lock (caller should skip).
        """
        ...

    def release(self, tenant_id: str) -> None:
        """Release the training lock for the given tenant.

        No-op if the lock is not held by this process.
        """
        ...


class DataStoreTrainingLock:
    """Training lock backed by a database table via DataStore.

    Uses an ml_training_locks table:
      - tenant_id TEXT PRIMARY KEY
      - acquired_at BIGINT NOT NULL (epoch milliseconds)
      - pid TEXT NOT NULL (process ID of lock holder)

    Acquire logic (atomic):
      1. Try INSERT -- if succeeds, lock acquired
      2. If conflict (lock exists):
         a. If lock.acquired_at + stale_timeout < now -> override (stale)
         b. If fresh -> return False (lock held)

    The DataStore must support:
      - acquire_training_lock(tenant_id, pid, stale_timeout_sec) -> bool
      - release_training_lock(tenant_id) -> None
    """

    def __init__(
        self,
        data_store: DataStore,
        stale_timeout_sec: int = STALE_LOCK_TIMEOUT_SEC,
    ) -> None:
        self.data_store = data_store
        self.stale_timeout_sec = stale_timeout_sec
        self._pid = str(os.getpid())

    def acquire(self, tenant_id: str) -> bool:
        """Attempt to acquire the lock.

        Handles stale lock detection: if the existing lock is older
        than stale_timeout_sec, it is overridden with a warning.
        """
        try:
            acquired = self.data_store.acquire_training_lock(
                tenant_id=tenant_id,
                pid=self._pid,
                stale_timeout_sec=self.stale_timeout_sec,
            )
        except AttributeError:
            # DataStore doesn't support locking -- treat as acquired
            logger.debug("DataStore does not support training locks, proceeding without lock")
            return True

        if acquired:
            logger.debug(
                "Acquired training lock for tenant %s (pid=%s)",
                tenant_id,
                self._pid,
            )
        else:
            logger.info("Training lock held for tenant %s, skipping", tenant_id)
        return acquired

    def release(self, tenant_id: str) -> None:
        """Release the lock. No-op if not held."""
        try:
            self.data_store.release_training_lock(tenant_id)
            logger.debug("Released training lock for tenant %s", tenant_id)
        except AttributeError:
            # DataStore doesn't support locking -- no-op
            pass
        except Exception:
            logger.warning(
                "Failed to release lock for tenant %s",
                tenant_id,
                exc_info=True,
            )
