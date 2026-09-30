"""Client leases, the pid-liveness sweep, and the self-termination timer.

Design doc §3.5 / repairs R4, R5 (via this mission's research.md):

* **Explicit lease** — ``POST /v1/clients/lease {client, pid, client_version,
  min_contracts}``, renewed by repeat calls. Keyed by ``(client, pid)``. Held
  in memory only: nothing is written into the client-owned ``lease/`` tree.
* **Sweep** — every lease whose pid is no longer alive is discarded
  (``os.kill(pid, 0)``), so a crashed client never pins the engine alive.
  Known limitation, stated rather than engineered around: pid reuse — if the
  OS hands a dead client's pid to an unrelated process, that lease survives
  until the new process exits too.
* **Explicit-lease staleness** — the spec gives explicit leases pid liveness
  *only*. A live pid that stops renewing keeps its lease; there is no time-based
  expiry for explicit leases (a decision recorded for the owner, not a default).
* **Implicit lease** — every ``/health`` poll counts as a lease for
  :data:`IMPLICIT_LEASE_SEC` (90 s): the skew-window bridge for a client that
  predates the lease API. It carries no pid, so it is recency-validated only.
* **Self-termination** — with zero live leases (explicit or implicit) for
  :data:`IDLE_EXIT_SEC` (120 s) the engine exits, **but only when the install
  root's ``lease/`` directory exists** (a managed install; ruled 2026-09-30,
  FR-014 / D-A2). Without it — a developer's ``kenaz-ml serve`` — the engine
  never self-terminates. The countdown starts at process start (the literal
  reading): a freshly spawned engine that never receives a lease or a health
  poll exits after the window.

Timing constants are module-level and overridable by environment variable for
tests (plan D-A6), read when a :class:`LeaseTable` is constructed.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kenaz_ml import config

logger = logging.getLogger(__name__)

#: Shipped defaults (seconds). Overridable for tests via the env vars below.
IMPLICIT_LEASE_SEC = 90.0
IDLE_EXIT_SEC = 120.0
SWEEP_INTERVAL_SEC = 5.0
#: Longest the exit path waits for an in-flight training run before exiting anyway.
TRAINING_DRAIN_SEC = 120.0

IMPLICIT_LEASE_ENV = "KENAZ_ML_IMPLICIT_LEASE_SEC"
IDLE_EXIT_ENV = "KENAZ_ML_IDLE_EXIT_SEC"
SWEEP_INTERVAL_ENV = "KENAZ_ML_LEASE_SWEEP_SEC"

#: Largest pid treated as plausible (Linux ``pid_max`` ceiling). Anything above,
#: or non-positive, is refused rather than passed to ``os.kill`` — ``os.kill(0)``
#: or a negative pid would signal a whole process group.
MAX_PID = 4_194_304

#: The ``/health`` / lease-response marker naming the lifecycle protocol this
#: engine speaks. A cross-repo value: the harness's ``mlsidecar.HealthPayload``
#: decodes it as an integer where 0 (or absence) means a pre-lease legacy engine,
#: so version 1 of the lease protocol is the integer 1.
LIFECYCLE_PROTOCOL = 1


def _env_seconds(name: str, default: float) -> float:
    raw = config.env(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("lifecycle: ignoring non-numeric %s=%r", name, raw)
        return default
    return value if value > 0 else default


def pid_alive(pid: int) -> bool:
    """True when ``pid`` names a live process. Never signals anything (signal 0)."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or pid > MAX_PID:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False
    return True


@dataclass
class Lease:
    client: str
    pid: int
    client_version: str
    min_contracts: dict[str, Any] = field(default_factory=dict)
    renewed_at: float = 0.0


class LeaseTable:
    """In-memory lease bookkeeping. Thread-safe; every time source is injectable."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        pid_alive_fn: Callable[[int], bool] = pid_alive,
        implicit_lease_sec: float | None = None,
        idle_exit_sec: float | None = None,
        sweep_interval_sec: float | None = None,
    ) -> None:
        self._clock = clock
        self._pid_alive = pid_alive_fn
        self.implicit_lease_sec = implicit_lease_sec or _env_seconds(IMPLICIT_LEASE_ENV, IMPLICIT_LEASE_SEC)
        self.idle_exit_sec = idle_exit_sec or _env_seconds(IDLE_EXIT_ENV, IDLE_EXIT_SEC)
        self.sweep_interval_sec = sweep_interval_sec or _env_seconds(SWEEP_INTERVAL_ENV, SWEEP_INTERVAL_SEC)
        self._lock = threading.Lock()
        self._leases: dict[tuple[str, int], Lease] = {}
        self._last_health_poll: float | None = None
        # The countdown starts at process (table) start.
        self._last_live: float = clock()

    # -- recording -------------------------------------------------------------

    def renew(self, client: str, pid: int, client_version: str, min_contracts: dict[str, Any] | None = None) -> Lease:
        now = self._clock()
        with self._lock:
            lease = Lease(client, pid, client_version, dict(min_contracts or {}), now)
            self._leases[(client, pid)] = lease
            self._last_live = now
            return lease

    def note_health_poll(self) -> None:
        now = self._clock()
        with self._lock:
            self._last_health_poll = now
            self._last_live = now

    # -- the sweep -------------------------------------------------------------

    def sweep(self) -> list[Lease]:
        """Discard leases whose pid is dead; refresh the idle clock. Returns the discarded."""
        with self._lock:
            leases = list(self._leases.items())
        dead = [(key, lease) for key, lease in leases if not self._pid_alive(lease.pid)]
        with self._lock:
            for key, _ in dead:
                self._leases.pop(key, None)
            if self._live_locked():
                self._last_live = self._clock()
        for _, lease in dead:
            logger.info("lifecycle: discarded lease client=%s pid=%d (process gone)", lease.client, lease.pid)
        return [lease for _, lease in dead]

    # -- queries ---------------------------------------------------------------

    def _implicit_live_locked(self) -> bool:
        return self._last_health_poll is not None and self._clock() - self._last_health_poll < self.implicit_lease_sec

    def _live_locked(self) -> bool:
        return bool(self._leases) or self._implicit_live_locked()

    def explicit_leases(self) -> list[Lease]:
        with self._lock:
            return list(self._leases.values())

    def implicit_lease_live(self) -> bool:
        with self._lock:
            return self._implicit_live_locked()

    def live_count(self) -> int:
        with self._lock:
            return len(self._leases) + (1 if self._implicit_live_locked() else 0)

    def idle_for(self) -> float:
        """Seconds since the last moment a lease was live (0 while one is).

        An implicit lease stays live for its window after the poll, so idleness
        is measured from the window's end, not from the poll itself.
        """
        with self._lock:
            if self._live_locked():
                return 0.0
            last = self._last_live
            if self._last_health_poll is not None:
                last = max(last, self._last_health_poll + self.implicit_lease_sec)
            return max(0.0, self._clock() - last)

    def should_exit(self, managed: bool) -> bool:
        """True when a managed engine has had zero live leases for the idle window."""
        return managed and self.idle_for() >= self.idle_exit_sec


def is_managed() -> bool:
    """True when the install root's ``lease/`` directory exists (FR-014 / D-A2)."""
    try:
        return config.lease_dir().is_dir()
    except OSError:
        return False
