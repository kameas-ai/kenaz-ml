"""Serving-path feature resolution for an **active** task.

This module inverts Feast's conventional read path, on purpose (plan.md D-003,
FR-014). For an active task the vector is **computed live** by
``sigil_ml.features`` and returned from that computation; the online store is
written as a byproduct and is *never* consulted.

Why the inversion
-----------------
The models answer questions about *right now* — is this developer stuck, how
much longer will this task take. An online store returns the last materialized
value, so reading it here would trade correctness for microseconds on the exact
path where freshness is the product: a prediction would be stale by up to a poll
interval (``poller.POLL_INTERVAL_SEC`` bounded by ``PREDICT_MIN_INTERVAL_SEC``,
currently 60 s). The April 2026 attempt adopted Feast's online half and stubbed
the offline half, and ended up with a TTL'd cache in front of functions that
already existed. That is the failure this module is shaped to make impossible.

How "never read the store for an active task" is enforced
---------------------------------------------------------
Not by convention — by three independent structural properties, each of which a
future edit has to defeat separately:

1. **The active resolvers take no parameter that could select a stored value.**
   There is no ``as_of_ms``, no ``allow_cached``, no ``prefer_store``. They have
   exactly one behaviour, and its name is the module docstring. Omitting
   ``as_of_ms`` also holds C-006 in place: serving passes nothing, meaning now.
2. **The push helper is typed ``-> None``.** Its result cannot become, modify or
   fall back into the returned vector no matter what happens inside it — the
   caller's ``return`` names the extractor's dict and nothing else.
3. **Online-store reads are refused while an active resolution is in progress.**
   :func:`read_online_features` raises :class:`ActiveEntityStoreReadError` when
   called inside the guard the active resolvers hold. A future edit that inserts
   a "check the store first" — directly, or several helpers deep — fails at
   runtime rather than silently serving a stale vector.

``tests/test_feature_store_local.py`` adds a fourth, independent check: it walks
this module's AST from each active entry point and asserts no reachable call
touches an online-read API, and it seeds the online store with a deliberately
wrong value and asserts the wrong value never reaches a prediction.

Non-active entities may read the store normally — that is what
:func:`read_online_features` is for.

Why the push is asynchronous
----------------------------
Measured against feast 0.65.0 with the SQLite online store: a single
``FeatureStore.push`` costs a median of **1.05 ms**, while
``extract_stuck_features`` costs a median of **4.7 µs** on a 60-event task and
**37 µs** on a 500-event one. A synchronous push would therefore be a 30x-220x
serving-path regression against NFR-002's 20% budget. So the vector is handed to
a background worker and the caller returns immediately, which costs a constant
~0.8 µs — a measured **1.18x** on the smallest tasks and **1.02x** on realistic
ones (``tests/test_feature_store_local.py`` re-measures both on every run). The
pushed row still carries the timestamp of the moment it was *computed*, not the
moment the worker got to it.

The push is best-effort throughout (FR-017): a store problem must not become a
prediction outage. Failures are logged at WARNING, rate-limited so a
persistently broken store cannot flood the log, and counted so that persistent
silent failure is still visible through :func:`push_stats`. Nothing is retried
inline — the next prediction cycle pushes again, and blocking a prediction on
store availability would defeat the point of computing live.
"""

from __future__ import annotations

import contextvars
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sigil_ml.features import (
    extract_duration_features,
    extract_features_from_buffer,
    extract_stuck_features,
)

if TYPE_CHECKING:  # pragma: no cover - import-time cost is the whole point
    from sigil_ml.store import DataStore

logger = logging.getLogger(__name__)

__all__ = [
    "ActiveEntityStoreReadError",
    "FAMILIES",
    "FeatureFamily",
    "flush_pushes",
    "local_feature_views",
    "local_push_source",
    "push_stats",
    "read_online_features",
    "reset_push_state",
    "resolve_duration_features",
    "resolve_stuck_features",
    "resolve_stuck_features_from_buffer",
    "use_feature_store",
]

# Nothing from `feast`, `pandas` or `sigil_ml.feature_store.definitions` is
# imported at module scope. This module is imported by `poller.py` and
# `routes.py`, which `app.py` and therefore the CLI import, and Feast's import
# pulls pyarrow and grpcio (spec.md, "Import cost in short-lived commands"). The
# heavy imports happen on the push worker thread, on first use.


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureFamily:
    """The three names a family needs to be pushed under.

    ``view_name`` and ``push_source_name`` are spelled out rather than derived
    from :mod:`sigil_ml.feature_store.definitions` so that importing this module
    does not import Feast. The test suite asserts they agree with the shipped
    definitions, so the duplication cannot drift silently.
    """

    name: str
    view_name: str
    push_source_name: str
    join_key: str


STUCK = FeatureFamily(
    name="stuck",
    view_name="stuck_features",
    push_source_name="stuck_push",
    join_key="task_id",
)

DURATION = FeatureFamily(
    name="duration",
    view_name="duration_features",
    push_source_name="duration_push",
    join_key="task_id",
)

FAMILIES: dict[str, FeatureFamily] = {STUCK.name: STUCK, DURATION.name: DURATION}


# ---------------------------------------------------------------------------
# Local source binding
# ---------------------------------------------------------------------------

#: The path handed to the batch source every local push source is required to
#: carry. It is not a path, and it is not meant to become one: it is a marker
#: string chosen so that nothing reads as a file that might exist, and so that
#: any code that ever does try to open it fails immediately and legibly.
UNUSED_BATCH_SOURCE_PATH = "UNUSED-the-local-deployment-has-no-offline-store"


def local_push_source(family: FeatureFamily) -> Any:
    """Build the ``PushSource`` the local deployment binds to ``family``'s view.

    About the batch source
    ----------------------
    Feast's ``FeatureView`` refuses a ``PushSource`` that carries no
    ``batch_source`` ("A batch_source needs to be specified for stream source"),
    so one is bound here. It is never read. The local flow only ever pushes into
    the online store and only ever serves from a live computation, and nothing
    in this deployment materializes from, or backfills out of, a batch source.

    This is not the mistake the April 2026 attempt made, and the difference is
    worth being precise about, because the two look similar in a diff:

    * *April*: placeholder ``FileSource`` entries pointing at parquet paths that
      did not exist, in a design that had **no offline story anywhere** — its own
      docstring conceded the paths were placeholders. Nothing could have read
      them, and nothing was ever going to, so the design was unfalsifiable.
    * *Here*: the **local** deployment genuinely has no offline store, by design
      (C-001, D-002) — it has no network path to one, and local training computes
      directly from the shared database rather than replaying a materialized
      table. The **cloud** deployment has a real offline store: a
      ``PostgreSQLSource`` over the ``ml_features`` table, built in WP04, which
      is what ``get_historical_features`` reads for point-in-time-correct
      training retrieval.

    So the batch source here is a *structural requirement of Feast's data model*
    for the local push path, not a stand-in for a source that ought to exist.
    It is named ``unused`` and pointed at :data:`UNUSED_BATCH_SOURCE_PATH`
    precisely so that no reader can mistake it for one — do not "fix" it by
    pointing it at a plausible parquet path, which would restate April's error.
    """
    from feast import FileSource, PushSource

    unused_batch_source = FileSource(
        name=f"{family.view_name}_unused_batch",
        path=UNUSED_BATCH_SOURCE_PATH,
        timestamp_field="event_timestamp",
        description=(
            "Required by Feast for a push source; never read. The local deployment has no "
            "offline store by design (C-001) — the cloud deployment's offline store is the "
            "ml_features PostgreSQL table."
        ),
    )
    return PushSource(name=family.push_source_name, batch_source=unused_batch_source)


def local_feature_views() -> list[Any]:
    """Return the shipped feature views with their local push sources bound.

    This is what the build-time ``feast apply`` (WP05) registers, and what the
    tests apply. Keeping the binding in one function is what makes the registry
    the serving path pushes into and the registry that ships the same object.
    """
    from sigil_ml.feature_store.definitions import duration_feature_view, stuck_feature_view

    return [
        stuck_feature_view(source=local_push_source(STUCK)),
        duration_feature_view(source=local_push_source(DURATION)),
    ]


# ---------------------------------------------------------------------------
# The active-path guard
# ---------------------------------------------------------------------------


class ActiveEntityStoreReadError(RuntimeError):
    """Raised when the online store is read during an active-task resolution.

    Reading the store for an active task is prohibited (D-003, FR-014): it would
    serve a value stale by up to a poll interval on the exact path where
    freshness is the product. This exception is the runtime half of that
    prohibition — it fires in a user's install, not only under test.
    """


#: True for the duration of an active-task resolution. A context variable rather
#: than a plain global so that concurrent resolutions on different threads or
#: asyncio tasks cannot see each other's guard.
#:
#: Set and reset inline in each resolver rather than through a
#: ``@contextmanager`` helper: the generator wrapper costs about 0.8 µs, which
#: against a ~12 µs extraction is most of NFR-002's whole 20% budget spent on
#: syntax.
_active_resolution: contextvars.ContextVar[bool] = contextvars.ContextVar("sigil_ml_active_resolution", default=False)


# ---------------------------------------------------------------------------
# The active resolvers — FR-014
# ---------------------------------------------------------------------------


def resolve_stuck_features(store: DataStore, task_id: str) -> dict[str, float]:
    """Resolve the stuck vector for the task the developer is working on now.

    Computed live and returned from that computation; pushed to the online store
    as a byproduct. There is deliberately no ``as_of_ms`` parameter: the
    extractor's ``None`` default means current wall clock, which is the only
    correct reference time for an active task (C-006), and the absence of the
    parameter is what stops this function from ever being repurposed for a
    historical replay.

    Args:
        store: The shared ``DataStore``.
        task_id: The active task.

    Returns:
        Exactly what ``sigil_ml.features.extract_stuck_features`` returns — same
        keys, same order, same values (C-005). This module changes where values
        go, never what they are.
    """
    token = _active_resolution.set(True)
    try:
        features = extract_stuck_features(store, task_id)
        _enqueue_push(STUCK, task_id, features)
    finally:
        _active_resolution.reset(token)
    return features


def resolve_duration_features(store: DataStore, task_id: str) -> dict[str, float]:
    """Resolve the duration vector for the active task. See :func:`resolve_stuck_features`."""
    token = _active_resolution.set(True)
    try:
        features = extract_duration_features(store, task_id)
        _enqueue_push(DURATION, task_id, features)
    finally:
        _active_resolution.reset(token)
    return features


def resolve_stuck_features_from_buffer(events: list[dict]) -> dict[str, float]:
    """Resolve the stuck vector from the raw event buffer, with no active task.

    The poller falls back to this when ``get_active_task`` returns nothing. It
    computes live like the other resolvers and, unlike them, pushes nothing:
    there is no active task, so there is no entity key to store the row under,
    and inventing one would put a row into the store that no lookup could ever
    legitimately match.

    Returns:
        Exactly what ``sigil_ml.features.extract_features_from_buffer`` returns.
    """
    token = _active_resolution.set(True)
    try:
        return extract_features_from_buffer(events)
    finally:
        _active_resolution.reset(token)


# ---------------------------------------------------------------------------
# The non-active read path
# ---------------------------------------------------------------------------


def read_online_features(family: FeatureFamily, entity_id: str) -> dict[str, float | None]:
    """Read a stored vector for a **non-active** entity.

    Legitimate for anything historical, for a different consumer of the store,
    or for cloud parity checks. Prohibited for an active task, and that
    prohibition is enforced here rather than documented: calling this inside an
    active resolution raises rather than returning.

    Unlike the push, this path is **not** best-effort. FR-017 requires feature
    resolution to fail loudly when its store is unavailable and never to return
    partial results, so an unopenable store raises rather than yielding a
    sentinel that a caller could carry into a model as if it were data. The
    asymmetry is deliberate and follows the direction the values flow: a failed
    *push* loses a byproduct the next cycle recreates, while a failed *read* is
    the caller's whole answer.

    Args:
        family: Which feature family to read.
        entity_id: The join-key value.

    Returns:
        The stored vector, whole. Every value is ``None`` for an entity the store
        has never seen — that is Feast reporting no row, not a partial result;
        one row carries every feature of a view, so a mixed vector cannot arise.

    Raises:
        ActiveEntityStoreReadError: If called during an active-task resolution.
        Exception: Whatever opening or querying the store raised.
    """
    if _active_resolution.get():
        raise ActiveEntityStoreReadError(
            f"The online store was read for {family.name!r} entity {entity_id!r} during an "
            "active-task resolution. Active-task predictions compute live and never read the "
            "store (D-003, FR-014): a stored value is stale by up to a poll interval, on the "
            "exact path where freshness is the product. Compute the vector with "
            f"resolve_{family.name}_features() instead."
        )

    store = _feature_store()

    from sigil_ml.feature_store.definitions import REGISTERED_FEATURE_NAMES

    names = REGISTERED_FEATURE_NAMES[family.name]
    resolved = store.get_online_features(
        features=[f"{family.view_name}:{name}" for name in names],
        entity_rows=[{family.join_key: entity_id}],
    ).to_dict()
    return {name: resolved[name][0] for name in names}


# ---------------------------------------------------------------------------
# Best-effort push — FR-017
# ---------------------------------------------------------------------------

#: Bounded so a store that has stopped draining cannot grow the queue without
#: limit inside a long-lived daemon. Overflow drops the *new* row and counts it:
#: the rows already queued are older, and the next cycle recomputes anyway.
#:
#: 1024 is far above anything the serving path can produce — the poller predicts
#: at most once per ``PREDICT_MIN_INTERVAL_SEC`` (60 s), so this is about 17
#: hours of backlog — and the entries are 6-float dicts. It is set well clear of
#: the working range on purpose: overflow is the expensive path (an exception
#: plus the rate limiter's bookkeeping), so a queue sized close to normal load
#: would put that cost on the serving path exactly when the store is struggling.
PUSH_QUEUE_MAXSIZE = 1024

#: At most one WARNING per this many seconds, however many pushes fail. A
#: persistently broken store logs once a minute with a suppression count rather
#: than once per prediction.
PUSH_LOG_INTERVAL_SEC = 60.0

#: How long to wait before rebuilding a feature store that failed to build. Not
#: a retry of a push — the failed row is gone — just a bound on how often an
#: unavailable store is re-opened.
STORE_REOPEN_INTERVAL_SEC = 60.0

_PushItem = tuple[FeatureFamily, str, dict[str, float], int]

_state_lock = threading.Lock()
_queue: queue.Queue[_PushItem] | None = None
_worker: threading.Thread | None = None

#: Counters are maintained by the push worker, not by the serving thread. That
#: is partly hygiene — the worker is single, so its increments need no lock — and
#: partly NFR-002: an uncontended lock acquire costs about 0.3 µs, which against
#: a ~5 µs extraction on a short task is a quarter of the whole 20% budget spent
#: counting. The only counter the serving thread touches is ``dropped``, which by
#: definition only moves when the queue is already saturated.
_stats: dict[str, int] = {
    "dropped": 0,
    "succeeded": 0,
    "failed": 0,
    "suppressed_warnings": 0,
}
_last_warning_at: float | None = None
_pending_suppressed = 0

# The cached store is built on the push worker and read by the non-active read
# path; the explicit one is installed by whoever calls `use_feature_store`. All
# three are single plain assignments, so the worst a race can do is open one
# store twice — not worth a lock on a path this hot.
_store: Any | None = None
_store_unavailable_until = 0.0
_explicit_store: Any | None = None


def use_feature_store(store: Any | None) -> None:
    """Install an explicit Feast ``FeatureStore`` for the push path.

    Lets a caller that already holds an opened store — the tests, and the
    build-time apply step — hand it over rather than have this module resolve
    one from the deployment configuration. ``None`` restores lazy resolution.
    """
    global _explicit_store, _store, _store_unavailable_until
    with _state_lock:
        _explicit_store = store
        _store = None
        _store_unavailable_until = 0.0


def _feature_store() -> Any:
    """Return an opened Feast store.

    Raises rather than returning ``None`` so that a failure to open the store
    and a failure to push produce exactly one logged, counted failure between
    them, carrying the real diagnostic. Both callers are on best-effort paths
    and handle it.

    Raises:
        Exception: Whatever opening the store raised, or a plain ``RuntimeError``
            while the reopen backoff is in force.
    """
    global _store, _store_unavailable_until

    explicit = _explicit_store
    if explicit is not None:
        return explicit
    if _store is not None:
        return _store
    if time.monotonic() < _store_unavailable_until:
        raise RuntimeError(f"feature store could not be opened; not retrying for {STORE_REOPEN_INTERVAL_SEC:.0f}s")

    try:
        from feast import FeatureStore

        from sigil_ml.feature_store.config import load_repo_config

        _store = FeatureStore(config=load_repo_config())
    except Exception:
        _store_unavailable_until = time.monotonic() + STORE_REOPEN_INTERVAL_SEC
        raise
    return _store


def _warn_push_failure(what: str, exc: BaseException) -> None:
    """Log a push failure at WARNING, at most once per :data:`PUSH_LOG_INTERVAL_SEC`.

    Rate limiting is deliberately by wall-clock window rather than by count: a
    store that fails every cycle should say so periodically forever, not go
    quiet after the first N.
    """
    global _last_warning_at, _pending_suppressed

    now = time.monotonic()
    with _state_lock:
        _stats["failed"] += 1
        if _last_warning_at is not None and now - _last_warning_at < PUSH_LOG_INTERVAL_SEC:
            _pending_suppressed += 1
            _stats["suppressed_warnings"] += 1
            return
        suppressed = _pending_suppressed
        _pending_suppressed = 0
        _last_warning_at = now

    suffix = ""
    if suppressed:
        suffix = f" ({suppressed} further failures suppressed in the last {PUSH_LOG_INTERVAL_SEC:.0f}s)"
    logger.warning(
        "feature-store push failed: %s: %s%s. The prediction was served from the live "
        "computation and is unaffected; the next cycle will push again.",
        what,
        exc,
        suffix,
    )


def _enqueue_push(family: FeatureFamily, entity_id: str, features: dict[str, float]) -> None:
    """Hand a freshly computed vector to the push worker.

    Returns ``None``, always, by signature and by construction: the push cannot
    contribute to — or fall back into — the vector the caller returns, whatever
    happens to it downstream (FR-017). This function does not raise.

    The computation time is captured *here* rather than in the worker, so the
    stored row carries the instant the vector describes rather than the instant
    the queue drained.
    """
    computed_at_ms = int(time.time() * 1000)
    # Fast path: read the queue without taking the state lock. The worker never
    # exits — its loop catches everything — so once the queue exists there is
    # nothing to re-check, and an uncontended lock acquire here is measurable
    # against a ~12 µs extraction.
    work = _queue
    try:
        if work is None:
            work = _ensure_worker()
        work.put_nowait((family, entity_id, dict(features), computed_at_ms))
    except queue.Full:
        with _state_lock:
            _stats["dropped"] += 1
        _warn_push_failure(f"push queue full for {family.name}/{entity_id}", RuntimeError("queue full"))
    except Exception as exc:  # pragma: no cover - defensive; enqueue is trivial
        _warn_push_failure(f"push could not be queued for {family.name}/{entity_id}", exc)


def _ensure_worker() -> queue.Queue[_PushItem]:
    """Start the push worker on first use and return its queue."""
    global _queue, _worker
    with _state_lock:
        if _queue is None:
            _queue = queue.Queue(maxsize=PUSH_QUEUE_MAXSIZE)
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(
                target=_push_worker,
                args=(_queue,),
                name="sigil-ml-feature-push",
                daemon=True,
            )
            _worker.start()
        return _queue


def _push_worker(work: queue.Queue[_PushItem]) -> None:
    """Drain the push queue forever. Nothing here can reach a caller.

    This loop does not exit: every failure mode inside it is caught, which is
    what lets :func:`_enqueue_push` skip a liveness check on the hot path.
    """
    while True:
        family, entity_id, features, computed_at_ms = work.get()
        try:
            _push_one(family, entity_id, features, computed_at_ms)
        except Exception as exc:
            # No inline retry: the row is dropped and the next prediction cycle
            # pushes a fresher one. Retrying here would hold the queue against a
            # store that is, by hypothesis, not working.
            _warn_push_failure(f"{family.name} vector for {entity_id!r}", exc)
        else:
            with _state_lock:
                _stats["succeeded"] += 1
        finally:
            work.task_done()


def _push_one(family: FeatureFamily, entity_id: str, features: dict[str, float], computed_at_ms: int) -> None:
    """Write one computed vector into the online store. Worker thread only."""
    store = _feature_store()

    import pandas as pd
    from feast.data_source import PushMode

    timestamp = pd.Timestamp(computed_at_ms, unit="ms", tz="UTC")
    frame = pd.DataFrame(
        {
            family.join_key: [entity_id],
            "event_timestamp": [timestamp],
            **{name: [value] for name, value in features.items()},
        }
    )
    store.push(family.push_source_name, frame, to=PushMode.ONLINE)


# ---------------------------------------------------------------------------
# Observability and test hooks
# ---------------------------------------------------------------------------


def push_stats() -> dict[str, int]:
    """Return cumulative push counters.

    Persistent silent failure is the failure mode best-effort semantics create,
    so it is counted: ``failed`` climbing while ``succeeded`` does not is a
    broken online store, whatever the log's rate limiter decided to print.
    """
    with _state_lock:
        return dict(_stats)


def flush_pushes(timeout: float = 5.0) -> bool:
    """Block until the push queue drains, or ``timeout`` elapses.

    For tests and for orderly shutdown. Never used on the serving path — waiting
    for the store there is exactly what this module exists to avoid.

    Returns:
        True if the queue drained within the timeout.
    """
    with _state_lock:
        work = _queue
    if work is None:
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if work.unfinished_tasks == 0:
            return True
        time.sleep(0.005)
    return work.unfinished_tasks == 0


def reset_push_state() -> None:
    """Clear counters, log rate limiting and any cached store. Test hook."""
    global _last_warning_at, _pending_suppressed, _store, _store_unavailable_until
    flush_pushes()
    with _state_lock:
        for key in _stats:
            _stats[key] = 0
        _last_warning_at = None
        _pending_suppressed = 0
    _store = None
    _store_unavailable_until = 0.0
