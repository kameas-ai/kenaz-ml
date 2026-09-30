"""Per-kind training and the retained-set-driven retrain trigger (harness-recommendation-models-01MSK2RM WP01).

What this module owns
---------------------
* :func:`load_training_set` — the kind's retained set, read strictly against
  the kind's hand-authored contract (``advice/contracts.py``), complete rows
  only.
* :func:`train_kind` — fit, then write artifact and manifest into the local
  slot keeping the pair consistent, or leave the previous pair byte-identical.
* :class:`AdviceTrainingScheduler` — the ``≥50 new labels AND ≥24 h`` trigger
  (FR-005, plan D-B2), a *sibling* of the workbench ``TrainingScheduler``,
  which is not modified.

Where the data comes from (T001, verified against Mission A's merged code)
--------------------------------------------------------------------------
Mission A's ``/v1/labels/{kind}`` ingest (``advice/label_log.py``) keeps every
label row, highest ``revision`` per ``(client, kind, features_hash, ts)`` key,
in ``<retained_data_dir>/<kind>.labels.jsonl`` with the propensity fields
(``features_complete``, ``shown``, ``model_id``, ``rung``, ``user_action``,
``confidence``, ``session_id``, ``snapshot_ms``). Only trainable rows reach the
retained set ``<retained_data_dir>/<kind>.jsonl`` — ``features_complete=true``
and ``user_action`` in ``accepted``/``auto_acted`` (``y=1``) or ``dismissed``
(``y=0``); ``ignored`` is recorded, not trained. ``retained.Example`` stays
``x``, ``y``, ``as_of_ms`` (Amendment A3.3), with ``as_of_ms`` = the row's
``snapshot_ms`` (or ``ts`` when absent), carried and never recomputed.

The defensive ``features_complete`` re-check (FR-002)
------------------------------------------------------
A retained example carries no ``(features_hash, ts)``, so it cannot be joined
to its label row by key. It is matched by what it *does* carry: an example
leaks when an incomplete label row has the same ``as_of_ms`` and — when that
row's features are all present — the same positional vector. An incomplete row
missing features can only be matched on ``as_of_ms``, and is then treated as a
leak only when no *complete* row shares that ``as_of_ms`` (otherwise the
complete row is the one retained, and excluding it would discard real data).
Leaks are counted, logged and excluded.

Manifest-beside-artifact
------------------------
The fit happens entirely in memory first. Only then: read previous provenance
(``trainer._next_provenance``, reused, not modified) → remove the stale
manifest → write the artifact (temp file + ``os.replace``) → write the
manifest. The previous pair's bytes are held in memory across that window and
restored on any write failure, so a failed run leaves the prior pair exactly as
it was (FR-012's registry guarantee: replace or restore).

"Generation" (D-B6) is ``Manifest.version`` at fit time — the extension count
— recorded as ``metrics["trained_generation"]``.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import platform
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kenaz_ml.advice.contracts import KIND_IDS, contract_for
from kenaz_ml.advice.models import MIN_TRAINING_ROWS, CalibrationDecline, fit_calibrated

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trigger constants (FR-005) — module-level, env-overridable for tests
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    from kenaz_ml.config import env

    raw = env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("advice training: %s=%r is not an integer; using %d", name, raw, default)
        return default


#: Retrain only after this many new complete labels since the last retrain.
MIN_NEW_LABELS = _env_int("KENAZ_ML_ADVICE_MIN_NEW_LABELS", 50)
#: ...and only when at least this long has passed since the last retrain.
MIN_INTERVAL_SEC = _env_int("KENAZ_ML_ADVICE_MIN_INTERVAL_SEC", 24 * 3600)
#: How often the app's loop calls :meth:`AdviceTrainingScheduler.check_and_retrain`.
TICK_SEC = _env_int("KENAZ_ML_ADVICE_TICK_SEC", 600)
#: After a failed or declined attempt, how long before the same kind is tried
#: again (in-memory backoff: never a hot retry loop; a restart retries once).
RETRY_BACKOFF_SEC = _env_int("KENAZ_ML_ADVICE_RETRY_BACKOFF_SEC", 3600)

TRAINING_SOURCE_LOCAL = "local"
ARTIFACT_SUFFIX = ".joblib"
MANIFEST_SUFFIX = ".json"

# Manifest ``metrics`` keys this module writes.
METRIC_TRAINED_GENERATION = "trained_generation"
METRIC_RUNG = "rung"
METRIC_LEAKED_INCOMPLETE = "n_leaked_incomplete_excluded"

#: Manifest ``metrics`` keys that describe the kind's *serving state* rather than
#: one fit — the persisted flip (WP04) and demotion (WP05). A retrain carries
#: them forward so a retrain never silently un-flips or un-demotes a kind.
CARRY_FORWARD_METRIC_KEYS: tuple[str, ...] = (
    "serving_backend",
    "flipped_at_ms",
    "flipped_generation",
    "demoted_generation",
    "demoted_at_ms",
)

# Rungs (design doc §5.4) as this mission reports them in ``metrics["rung"]``.
RUNG_SHADOW = "R2"  # trained model scores in shadow; the client's heuristic answers
RUNG_LIVE = "R3"  # classic model serves

# Decline / failure reasons — stable strings, safe to branch on and to log.
REASON_UNKNOWN_KIND = "unknown_kind"
REASON_NO_RETAINED = "no_retained_set"
REASON_CONTRACT_MISMATCH = "contract_mismatch"
REASON_MIRROR_DIRTY = "retained_mirror_dirty"
REASON_INSUFFICIENT_COMPLETE = "insufficient_complete_labels"
REASON_BELOW_MINIMUM = "below_minimum_rows"
REASON_FIT_FAILED = "fit_failed"
REASON_WRITE_FAILED = "write_failed"
REASON_BASE_SLOT = "base_slot_refused"

STATUS_TRAINED = "trained"
STATUS_DECLINED = "declined"
STATUS_FAILED = "failed"

Clock = Callable[[], int]


def wall_clock_ms() -> int:
    """The default clock (epoch ms). Injected everywhere so tests control time."""
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Per-kind serialization
# ---------------------------------------------------------------------------

_KIND_LOCKS: dict[str, threading.RLock] = {}
_KIND_LOCKS_GUARD = threading.Lock()


def kind_lock(kind: str) -> threading.RLock:
    """The one lock serializing retrain, reload, flip and evaluation for ``kind``."""
    with _KIND_LOCKS_GUARD:
        lock = _KIND_LOCKS.get(kind)
        if lock is None:
            lock = threading.RLock()
            _KIND_LOCKS[kind] = lock
        return lock


# ---------------------------------------------------------------------------
# T002 — the training set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decline:
    """A typed refusal to train. Never an exception."""

    kind: str
    reason: str
    detail: str
    counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingSet:
    """Complete, strictly-positional training rows for one kind."""

    kind: str
    contract: Any
    X: np.ndarray
    y: np.ndarray
    as_of_ms: tuple[int, ...]
    n_retained: int
    n_leaked: int
    retained_generation: str | None

    @property
    def n_complete(self) -> int:
        return int(self.X.shape[0])


def _incomplete_signatures(
    kind: str, names: tuple[str, ...], directory: Path | None
) -> tuple[set[tuple[int, tuple[float, ...]]], set[int]]:
    """``(as_of, vector)`` of incomplete label rows, and ``as_of`` of those with missing features."""
    from kenaz_ml.advice.label_log import read_label_log

    with_vector: set[tuple[int, tuple[float, ...]]] = set()
    as_of_only: set[int] = set()
    complete_as_of: set[int] = set()
    for record in read_label_log(kind, directory=directory).labels().values():
        snapshot = record.get("snapshot_ms")
        try:
            as_of = int(snapshot) if snapshot is not None else int(record["ts"])
        except (KeyError, TypeError, ValueError):
            continue
        if record.get("features_complete", False):
            complete_as_of.add(as_of)
            continue
        features = record.get("features") or {}
        if all(name in features for name in names):
            try:
                with_vector.add((as_of, tuple(float(features[name]) for name in names)))
            except (TypeError, ValueError):
                as_of_only.add(as_of)
        else:
            as_of_only.add(as_of)
    return with_vector, as_of_only - complete_as_of


def load_training_set(kind: str, *, retained_dir: Path | str | None = None) -> TrainingSet | Decline:
    """Read ``kind``'s retained set, strictly and complete-only (FR-001, FR-002). Never raises."""
    from kenaz_ml.modelstore.registry import read_retained

    contract = contract_for(kind)
    if contract is None:
        return Decline(kind, REASON_UNKNOWN_KIND, f"{kind!r} has no published contract (advice/contracts.py)")
    names = tuple(contract.names)
    directory = Path(retained_dir) if retained_dir is not None else None

    from kenaz_ml.advice.label_log import is_dirty

    # Mission A (Amendment A4) marks the retained mirror dirty when it lags the
    # label log (a failed append/rebuild); it may then hold a stale label for a
    # row a later revision changed. Wait for the next ingest batch's rebuild.
    if is_dirty(kind, directory=directory):
        return Decline(kind, REASON_MIRROR_DIRTY, f"retained mirror for {kind!r} is marked dirty; awaiting its rebuild")

    retained = read_retained(kind, directory=directory)
    if not retained.ok:
        return Decline(kind, REASON_NO_RETAINED, f"no usable retained set for {kind!r}: {retained.reason}")

    # Ordered, element-by-element — never a set comparison.
    if tuple(retained.names) != names or retained.contract_version != contract.service_version:
        return Decline(
            kind,
            REASON_CONTRACT_MISMATCH,
            f"retained header names {list(retained.names)} (contract {retained.contract_version}) "
            f"!= contract {list(names)} ({contract.service_version})",
        )

    leaked_vectors, leaked_as_of = _incomplete_signatures(kind, names, directory)
    rows_x: list[tuple[float, ...]] = []
    rows_y: list[float] = []
    rows_as_of: list[int] = []
    leaked = 0
    for example in retained.examples:
        if example.as_of_ms is None or len(example.x) != len(names):
            leaked += 1  # not a complete, honestly-timed row: never train on it
            continue
        vector = tuple(float(v) for v in example.x)  # already positional (header order)
        as_of = int(example.as_of_ms)
        if (as_of, vector) in leaked_vectors or as_of in leaked_as_of:
            leaked += 1
            continue
        rows_x.append(vector)
        rows_y.append(float(example.y))
        rows_as_of.append(as_of)

    if leaked:
        logger.warning(
            "advice training: %d retained row(s) for %r are features_complete=false in the label log "
            "(or unusable); excluded from training",
            leaked,
            kind,
        )

    counts = {"retained": len(retained.examples), "complete": len(rows_x), "leaked": leaked}
    if not rows_x:
        return Decline(kind, REASON_INSUFFICIENT_COMPLETE, f"insufficient complete labels for {kind!r}", counts)
    if len(rows_x) < MIN_TRAINING_ROWS:
        return Decline(
            kind,
            REASON_BELOW_MINIMUM,
            f"{len(rows_x)} complete row(s) for {kind!r}; at least {MIN_TRAINING_ROWS} are needed to fit",
            counts,
        )
    return TrainingSet(
        kind=kind,
        contract=contract,
        X=np.asarray(rows_x, dtype=float),
        y=np.asarray(rows_y, dtype=float),
        as_of_ms=tuple(rows_as_of),
        n_retained=len(retained.examples),
        n_leaked=leaked,
        retained_generation=retained.generation,
    )


# ---------------------------------------------------------------------------
# T003/T004 — fit and write
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitResult:
    """A fitted model plus the fit's own manifest metrics."""

    model: Any
    metrics: dict[str, Any] = field(default_factory=dict)


def fit_model(training_set: TrainingSet) -> FitResult | Decline:
    """Fit ``training_set``'s kind: a fresh ``CalibratedClassifierCV``-wrapped GBDT (WP02).

    Never raises. A calibration precondition (one class, thin class, NaN) is a
    typed decline; any sklearn failure is ``fit_failed``.
    """
    kind = training_set.kind
    try:
        fitted = fit_calibrated(kind, training_set.X, training_set.y)
    except Exception as exc:
        logger.warning("advice training: fit failed for %r", kind, exc_info=True)
        return Decline(kind, REASON_FIT_FAILED, f"{type(exc).__name__}: {exc}")
    if isinstance(fitted, CalibrationDecline):
        logger.info("advice training: %r declined before fitting: %s", kind, fitted.detail)
        return Decline(kind, fitted.reason, fitted.detail, dict(fitted.counts))
    return FitResult(model=fitted.model, metrics=fitted.metrics())


@dataclass(frozen=True)
class TrainOutcome:
    """What one :func:`train_kind` call did."""

    kind: str
    status: str
    reason: str | None = None
    detail: str | None = None
    manifest: Any = None
    counts: dict[str, int] = field(default_factory=dict)
    duration_ms: int = 0

    @property
    def trained(self) -> bool:
        return self.status == STATUS_TRAINED

    @property
    def generation(self) -> str | None:
        return str(self.manifest.version) if self.manifest is not None else None


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _read_or_none(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _restore(path: Path, previous: bytes | None) -> None:
    try:
        if previous is None:
            path.unlink(missing_ok=True)
        else:
            _atomic_write_bytes(path, previous)
    except OSError:
        logger.error("advice training: could not restore %s after a failed write", path, exc_info=True)


def previous_manifest(kind: str, models_dir: Path | str | None = None) -> Any:
    """The local slot's current manifest for ``kind``, or ``None``."""
    from kenaz_ml import config
    from kenaz_ml.modelstore.registry import read_manifest

    directory = Path(models_dir) if models_dir is not None else config.models_dir()
    read = read_manifest(directory / f"{kind}{MANIFEST_SUFFIX}")
    return read.manifest if read.ok else None


def train_kind(
    kind: str,
    *,
    models_dir: Path | str | None = None,
    retained_dir: Path | str | None = None,
    clock: Clock = wall_clock_ms,
) -> TrainOutcome:
    """Train ``kind`` end to end (FR-001). Never raises; the prior pair survives any failure."""
    with kind_lock(kind):
        started = clock()
        try:
            return _train_kind(kind, models_dir, retained_dir, clock, started)
        except Exception as exc:  # defensive: nothing raises to the scheduler
            logger.exception("advice training: unexpected failure training %r", kind)
            return TrainOutcome(kind, STATUS_FAILED, REASON_FIT_FAILED, f"{type(exc).__name__}: {exc}")


def _train_kind(
    kind: str, models_dir: Path | str | None, retained_dir: Path | str | None, clock: Clock, started: int
) -> TrainOutcome:
    import joblib

    from kenaz_ml import config
    from kenaz_ml.modelstore.registry import Manifest, Runtime, Training, running_sklearn_version, write_manifest
    from kenaz_ml.training.trainer import _is_base_slot, _next_provenance

    loaded = load_training_set(kind, retained_dir=retained_dir)
    if isinstance(loaded, Decline):
        return TrainOutcome(kind, STATUS_DECLINED, loaded.reason, loaded.detail, counts=loaded.counts)
    counts = {"retained": loaded.n_retained, "complete": loaded.n_complete, "leaked": loaded.n_leaked}

    fitted = fit_model(loaded)
    if isinstance(fitted, Decline):
        status = STATUS_FAILED if fitted.reason == REASON_FIT_FAILED else STATUS_DECLINED
        return TrainOutcome(kind, status, fitted.reason, fitted.detail, counts={**counts, **fitted.counts})

    directory = Path(models_dir) if models_dir is not None else config.models_dir()
    if _is_base_slot(directory):
        return TrainOutcome(kind, STATUS_FAILED, REASON_BASE_SLOT, f"{directory} is the read-only base slot")
    directory.mkdir(parents=True, exist_ok=True)

    buffer = io.BytesIO()
    joblib.dump(fitted.model, buffer)
    payload = buffer.getvalue()

    artifact = directory / f"{kind}{ARTIFACT_SUFFIX}"
    manifest_file = directory / f"{kind}{MANIFEST_SUFFIX}"

    prior = previous_manifest(kind, directory)
    provenance = _next_provenance(directory, kind, TRAINING_SOURCE_LOCAL)
    version = str(provenance.n_local_extensions)

    metrics: dict[str, Any] = {}
    if prior is not None:
        for key in CARRY_FORWARD_METRIC_KEYS:
            if key in prior.metrics:
                metrics[key] = prior.metrics[key]
    metrics.update(fitted.metrics)
    metrics[METRIC_TRAINED_GENERATION] = int(version)
    metrics[METRIC_LEAKED_INCOMPLETE] = loaded.n_leaked
    metrics[METRIC_RUNG] = RUNG_LIVE if metrics.get("serving_backend") == "classic" else RUNG_SHADOW

    manifest = Manifest(
        name=kind,
        version=version,
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        created_at=clock(),
        provenance=provenance,
        runtime=Runtime(
            estimator=type(fitted.model).__name__,
            sklearn_version=running_sklearn_version() or "",
            python_version=platform.python_version(),
        ),
        feature_contract=loaded.contract,
        training=Training(
            n_samples=loaded.n_complete,
            retained_generation=loaded.retained_generation,
            as_of_ms=max(loaded.as_of_ms),
        ),
        metrics=metrics,
    )

    previous_artifact = _read_or_none(artifact)
    previous_manifest_bytes = _read_or_none(manifest_file)
    try:
        manifest_file.unlink(missing_ok=True)  # stale manifest first: never a mismatched pair
        _atomic_write_bytes(artifact, payload)
        write_manifest(manifest_file, manifest)
    except OSError as exc:
        logger.warning("advice training: write failed for %r; restoring the previous pair", kind, exc_info=True)
        _restore(artifact, previous_artifact)
        _restore(manifest_file, previous_manifest_bytes)
        return TrainOutcome(kind, STATUS_FAILED, REASON_WRITE_FAILED, f"{type(exc).__name__}: {exc}", counts=counts)

    logger.info(
        "advice training: trained %r generation %s on %d complete row(s) (%d leaked excluded)",
        kind,
        version,
        loaded.n_complete,
        loaded.n_leaked,
    )
    return TrainOutcome(kind, STATUS_TRAINED, manifest=manifest, counts=counts, duration_ms=max(0, clock() - started))


# ---------------------------------------------------------------------------
# T005 — the retrain trigger (FR-005, D-B2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DueCheck:
    """Whether a kind is due, and the numbers that decided it."""

    kind: str
    due: bool
    reason: str
    new_labels: int = 0
    elapsed_sec: float | None = None


def count_new_labels(kind: str, prior: Any, *, retained_dir: Path | str | None = None) -> int:
    """New complete labels since the manifest ``prior`` was trained — from durable state only.

    An in-memory baseline would reset on restart (retraining at once, or never),
    so this reads the retained set and the last manifest:

    ``max(complete_now - prior.training.n_samples, rows with as_of_ms > prior.training.as_of_ms)``

    Trade-offs, documented rather than hidden: a late-arriving label whose
    ``as_of_ms`` predates the last fit is invisible to the second term but
    counted by the first; eviction or a retained rebuild can make the first term
    shrink (non-monotonic), which the second term covers. Either term alone
    under-counts in one of those cases; the maximum under-counts only when both
    happen at once, and never over-counts by more than the evicted rows.
    """
    loaded = load_training_set(kind, retained_dir=retained_dir)
    if isinstance(loaded, Decline):
        return int(loaded.counts["complete"]) if "complete" in loaded.counts else 0
    if prior is None:
        return loaded.n_complete
    last_n = prior.training.n_samples or 0
    last_as_of = prior.training.as_of_ms
    by_count = loaded.n_complete - last_n
    by_time = sum(1 for a in loaded.as_of_ms if last_as_of is None or a > last_as_of)
    return max(by_count, by_time, 0)


OnTrained = Callable[[str, TrainOutcome], None]
OnTick = Callable[[int], None]


class AdviceTrainingScheduler:
    """Per-kind retrain trigger: ≥ :data:`MIN_NEW_LABELS` new labels AND ≥ :data:`MIN_INTERVAL_SEC`.

    Sibling of ``training.scheduler.TrainingScheduler`` (D-B2): same
    ``check_and_retrain`` shape, called from a background thread. Each kind is
    decided independently; two kinds due in the same tick both train (their
    files never collide — each writes only ``<kind>.joblib``/``<kind>.json``).

    Seams, all injectable:

    * ``on_trained(kind, outcome)`` — after a successful retrain: the audit
      row, hot reload, graduation and flip (WP04 binds it).
    * ``on_tick(now_ms)`` — once per tick before the retrain checks: the
      post-flip evaluation (WP05 binds it).
    """

    def __init__(
        self,
        kinds: Iterable[str] = KIND_IDS,
        *,
        models_dir: Path | str | None = None,
        retained_dir: Path | str | None = None,
        clock: Clock = wall_clock_ms,
        on_trained: OnTrained | None = None,
        on_tick: OnTick | None = None,
        min_new_labels: int | None = None,
        min_interval_sec: int | None = None,
    ) -> None:
        self.kinds = tuple(kinds)
        self.models_dir = models_dir
        self.retained_dir = retained_dir
        self.clock = clock
        self.on_trained = on_trained
        self.on_tick = on_tick
        self.min_new_labels = MIN_NEW_LABELS if min_new_labels is None else min_new_labels
        self.min_interval_sec = MIN_INTERVAL_SEC if min_interval_sec is None else min_interval_sec
        self._backoff_until: dict[str, int] = {}

    def due(self, kind: str, now_ms: int | None = None) -> DueCheck:
        now = self.clock() if now_ms is None else now_ms
        until = self._backoff_until.get(kind)
        if until is not None and now < until:
            return DueCheck(kind, False, "backoff after a failed attempt")
        prior = previous_manifest(kind, self.models_dir)
        elapsed: float | None = None
        if prior is not None and prior.created_at is not None:
            elapsed = (now - prior.created_at) / 1000.0
            if elapsed < self.min_interval_sec:
                return DueCheck(kind, False, "interval not elapsed", elapsed_sec=elapsed)
        new = count_new_labels(kind, prior, retained_dir=self.retained_dir)
        if new < self.min_new_labels:
            return DueCheck(kind, False, "not enough new labels", new_labels=new, elapsed_sec=elapsed)
        return DueCheck(kind, True, "due", new_labels=new, elapsed_sec=elapsed)

    def check_and_retrain(self) -> dict[str, DueCheck | TrainOutcome]:
        """One tick. Blocks while training runs; call it off the event loop. Never raises."""
        now = self.clock()
        if self.on_tick is not None:
            try:
                self.on_tick(now)
            except Exception:
                logger.exception("advice scheduler: tick callback failed")
        results: dict[str, DueCheck | TrainOutcome] = {}
        for kind in self.kinds:
            try:
                results[kind] = self._check_one(kind, now)
            except Exception:
                logger.exception("advice scheduler: check for %r failed", kind)
        return results

    def _check_one(self, kind: str, now: int) -> DueCheck | TrainOutcome:
        check = self.due(kind, now)
        if not check.due:
            # Quiet by design: escalate_model has no labels until the harness
            # ships model.switched — expected, not an error (spec Edge Cases).
            logger.debug("advice scheduler: %r not due (%s, %d new)", kind, check.reason, check.new_labels)
            return check
        logger.info("advice scheduler: retraining %r (%d new labels)", kind, check.new_labels)
        outcome = train_kind(kind, models_dir=self.models_dir, retained_dir=self.retained_dir, clock=self.clock)
        if not outcome.trained:
            self._backoff_until[kind] = now + RETRY_BACKOFF_SEC * 1000
            logger.warning(
                "advice scheduler: %r not retrained (%s: %s); previous generation keeps serving",
                kind,
                outcome.reason,
                outcome.detail,
            )
            return outcome
        self._backoff_until.pop(kind, None)
        if self.on_trained is not None:
            try:
                self.on_trained(kind, outcome)
            except Exception:
                logger.exception("advice scheduler: post-retrain callback failed for %r", kind)
        return outcome
