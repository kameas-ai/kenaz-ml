"""The stateless ``POST /v1/features`` push lane (two-client-engine-01MSK2EN WP09, FR-023).

The design doc's third transport lane: typed feature events for event classes
that are *not* observed by the daemon's event capture — egress denies, run
outcomes, budget pressure, lifecycle telemetry, commit timestamps. They are
deliberately **not** routed through the ``events`` table (Go-owned, C-007):
they live in a bounded **in-process** store with no table, no file and no
outbound call, and are lost on restart — recent-signal hints, not a record.

Closed class registry
---------------------
* ``commit`` — fully specified: ``ts_ms`` is the commit time (positive integer
  milliseconds), ``session_id`` scopes it, ``fields`` must be empty (the only
  consumer, ``feature-vocabulary-refresh-01MSK2VR`` D-D3, needs a timestamp).
* ``egress_deny``, ``run_outcome``, ``budget_pressure``, ``lifecycle`` —
  recognized but **schema-deferred**: refused ``class_schema_not_defined``
  until the consumer mission that needs one defines its fields. No schema is
  invented here.
* anything else — refused ``unknown_class``.

A refused event never blocks the others in its batch.

The store
---------
Keyed by ``(event_class, session_id)`` (``session_id`` ``None`` = global), at
most :data:`PER_KEY_LIMIT` events per key and :data:`TOTAL_LIMIT` overall,
oldest-first eviction, lock-safe. One store per app: ``register_routes``
creates it (so two ``create_app()`` instances never share buffered events) and
:func:`install` makes it the process's current store for in-process
accessors such as :func:`commit_ts_ms`.

Standard library only; no ``sqlite3``, no file I/O, no HTTP client.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

CLASS_COMMIT = "commit"
#: Classes named by the design doc whose field schemas belong to future consumers.
DEFERRED_CLASSES: frozenset[str] = frozenset({"egress_deny", "run_outcome", "budget_pressure", "lifecycle"})
#: Fully specified classes -> the closed set of allowed ``fields`` keys.
SPECIFIED_CLASSES: dict[str, frozenset[str]] = {CLASS_COMMIT: frozenset()}

REASON_UNKNOWN_CLASS = "unknown_class"
REASON_SCHEMA_NOT_DEFINED = "class_schema_not_defined"
REASON_BAD_TIMESTAMP = "invalid_ts_ms"
REASON_BAD_FIELDS = "invalid_fields"
REASON_BAD_SESSION = "invalid_session_id"
REASON_BATCH_TOO_LARGE = "batch_too_large"
REASON_LANE_UNAVAILABLE = "lane_unavailable"

MAX_BATCH = 256
PER_KEY_LIMIT = 64
TOTAL_LIMIT = 4096

Key = tuple[str, "str | None"]


@dataclass(frozen=True)
class StoredEvent:
    client: str
    event_class: str
    ts_ms: int
    session_id: str | None
    fields: dict[str, Any]
    received_ms: int
    seq: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "client": self.client,
            "event_class": self.event_class,
            "ts_ms": self.ts_ms,
            "session_id": self.session_id,
            "fields": dict(self.fields),
            "received_ms": self.received_ms,
        }


@dataclass
class PushOutcome:
    accepted: int = 0
    refusals: list[dict[str, Any]] = field(default_factory=list)


def validate_event(event: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Return ``(reason, detail)`` for a refusal, or ``(None, None)`` when valid."""
    event_class = event.get("event_class")
    if event_class in DEFERRED_CLASSES:
        return REASON_SCHEMA_NOT_DEFINED, f"event class {event_class!r} is recognized but its schema is not defined yet"
    if event_class not in SPECIFIED_CLASSES:
        return REASON_UNKNOWN_CLASS, f"unknown event class {event_class!r}"
    ts = event.get("ts_ms")
    if isinstance(ts, bool) or not isinstance(ts, int) or ts <= 0:
        return REASON_BAD_TIMESTAMP, f"ts_ms must be a positive integer (milliseconds), got {ts!r}"
    session_id = event.get("session_id")
    if session_id is not None and (not isinstance(session_id, str) or not session_id):
        return REASON_BAD_SESSION, "session_id must be a non-empty string when present"
    fields = event.get("fields")
    if fields is None:
        fields = {}
    if not isinstance(fields, Mapping):
        return REASON_BAD_FIELDS, "fields must be an object"
    allowed = SPECIFIED_CLASSES[event_class]
    unexpected = sorted(set(fields) - allowed)
    if unexpected:
        return REASON_BAD_FIELDS, f"fields {unexpected} are not defined for class {event_class!r}"
    return None, None


class FeaturePushStore:
    """Bounded, lock-safe, in-memory event store. Never raises from an accessor."""

    def __init__(
        self,
        *,
        per_key_limit: int = PER_KEY_LIMIT,
        total_limit: int = TOTAL_LIMIT,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._per_key = per_key_limit
        self._total_limit = total_limit
        self._clock = clock
        self._lock = threading.Lock()
        self._by_key: dict[Key, deque[StoredEvent]] = {}
        self._order: deque[tuple[int, Key]] = deque()
        self._total = 0
        self._seq = 0

    def __len__(self) -> int:
        with self._lock:
            return self._total

    def push(self, client: str, events: Sequence[Mapping[str, Any]]) -> PushOutcome:
        outcome = PushOutcome()
        for index, event in enumerate(events):
            reason, detail = validate_event(event)
            if reason is not None:
                outcome.refusals.append({"index": index, "reason": reason, "detail": detail})
                continue
            self._insert(client, event)
            outcome.accepted += 1
        return outcome

    def _insert(self, client: str, event: Mapping[str, Any]) -> None:
        received = int(self._clock() * 1000)
        with self._lock:
            self._seq += 1
            stored = StoredEvent(
                client=client,
                event_class=str(event["event_class"]),
                ts_ms=int(event["ts_ms"]),
                session_id=event.get("session_id"),
                fields=dict(event.get("fields") or {}),
                received_ms=received,
                seq=self._seq,
            )
            key: Key = (stored.event_class, stored.session_id)
            bucket = self._by_key.setdefault(key, deque())
            bucket.append(stored)
            self._order.append((stored.seq, key))
            self._total += 1
            if len(bucket) > self._per_key:
                bucket.popleft()
                self._total -= 1
            while self._total > self._total_limit and self._order:
                seq, oldest_key = self._order.popleft()
                oldest = self._by_key.get(oldest_key)
                if oldest and oldest[0].seq == seq:
                    oldest.popleft()
                    self._total -= 1
                    if not oldest:
                        del self._by_key[oldest_key]
            # Drop order entries already evicted per-key so the deque stays bounded.
            while self._order and self._stale(self._order[0]):
                self._order.popleft()
            if len(self._order) > 2 * self._total_limit:
                # Stale entries can pile up behind a long-lived one; compact.
                live = sorted((e.seq, k) for k, b in self._by_key.items() for e in b)
                self._order = deque(live)

    def _stale(self, entry: tuple[int, Key]) -> bool:
        seq, key = entry
        bucket = self._by_key.get(key)
        return not bucket or bucket[0].seq > seq

    # -- accessors -------------------------------------------------------------

    def latest(self, event_class: str, session_id: str | None = None) -> dict[str, Any] | None:
        """The most recently pushed event for the key, as plain data, or ``None``."""
        try:
            with self._lock:
                bucket = self._by_key.get((event_class, session_id))
                return bucket[-1].as_dict() if bucket else None
        except Exception:  # pragma: no cover - accessors never raise
            return None

    def events(self, event_class: str, session_id: str | None = None) -> list[dict[str, Any]]:
        try:
            with self._lock:
                return [e.as_dict() for e in self._by_key.get((event_class, session_id), ())]
        except Exception:  # pragma: no cover
            return []

    def commit_ts_ms(self, session_id: str | None = None) -> int | None:
        """The latest pushed commit timestamp for ``session_id`` (max ``ts_ms``), or ``None``.

        Max rather than last-arrived, so a replayed older push never moves the
        answer backwards.
        """
        try:
            with self._lock:
                bucket = self._by_key.get((CLASS_COMMIT, session_id))
                return max(e.ts_ms for e in bucket) if bucket else None
        except Exception:  # pragma: no cover
            return None


_CURRENT: FeaturePushStore | None = None


def install(store: FeaturePushStore) -> FeaturePushStore:
    """Make ``store`` the process's current store for the module-level accessors."""
    global _CURRENT
    _CURRENT = store
    return store


def current_store() -> FeaturePushStore | None:
    return _CURRENT


def commit_ts_ms(session_id: str | None = None) -> int | None:
    """In-process accessor for ``feature-vocabulary-refresh-01MSK2VR`` (D-D3). Never raises."""
    store = _CURRENT
    return store.commit_ts_ms(session_id) if store is not None else None


def latest(event_class: str, session_id: str | None = None) -> dict[str, Any] | None:
    store = _CURRENT
    return store.latest(event_class, session_id) if store is not None else None
