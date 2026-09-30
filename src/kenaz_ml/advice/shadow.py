"""Shadow records, graduation verdicts and their manifest write (harness-recommendation-models-01MSK2RM WP03).

Promotion is **measured, never asserted** (design doc §5.4). This module:

* defines the **shadow record** appended to Mission A's per-kind label log
  (``<retained_data_dir>/<kind>.labels.jsonl``, D-B3 as revised by A3.3) under
  ``"record": "shadow"`` through ``label_log.append_records`` — the same file,
  the same bound, the same eviction; never a second store;
* offers the non-raising, non-blocking write-site interface
  (:func:`enqueue_shadow`) that the dispatch layer calls (WP04 lands the call);
* computes the per-kind graduation verdict (:func:`graduation_check`) and
  merges it into the manifest's existing ``metrics`` dict
  (:func:`update_manifest_metrics`) — never a new top-level manifest key.

The join key: exact first, the vector-hash heuristic only as a fallback
-----------------------------------------------------------------------
The planning documents join shadow to label on the harness's
``(features_hash, ts)``. Mission A's ``/v1/recommend`` request now carries
both as **optional** fields (``RecommendRequest.features_hash`` / ``.ts``,
two-client-engine-01MSK2EN 9461214). A shadow record stores them as
``features_hash`` and ``label_ts`` — ``label_ts``, not ``ts``, because the
record's ``ts`` is the engine's own receipt time and other code reads it.

:func:`join_shadow_to_labels` pairs in two passes:

1. **Exact** — a shadow record carrying both ``features_hash`` and
   ``label_ts`` pairs with the label row whose ``(features_hash, ts)`` equals
   them, or with nothing. It never falls back: a keyed record whose label has
   not arrived (or was evicted) stays unpaired rather than being guessed.
2. **Fallback, superseded when the exact key is present** — only for a
   record missing either field (a client that predates the wire fields, or a
   record written before 2026-09-30): an engine-computed ``vector_hash`` over
   the kind and its ordered feature vector, the engine's receipt ``ts`` and
   ``session_id``. The join recomputes ``vector_hash`` from each label row's
   full ``features`` JSON and takes the nearest row of the same hash (and the
   same ``session_id`` when both carry one) within
   :data:`SHADOW_JOIN_TOLERANCE_MS`. Delete this pass once every supported
   harness sends the exact key (owner: the harness release that makes
   ``features_hash``/``ts`` unconditional on ``/v1/recommend``).

Each label row pairs at most once across both passes, and a row the exact pass
took is never offered to the fallback. ``user_action`` is never stored on a
shadow record; it is read from the label row (highest ``revision``) at
evaluation time.

Who served a label row
----------------------
The engine logs no heuristic decision (A3.2: it has none). A label row's
``model_id``/``rung`` says who served it: the harness's heuristic when
``rung == "heuristic"`` or ``model_id`` starts with ``"heuristic"``; this
engine's classic backend when ``model_id`` starts with ``"classic/<kind>@"``
(the ``RecommendResponse.model`` label the engine returns) — see
:func:`served_by_heuristic` / :func:`served_by_classic`.

Tunable parameters (D-B7)
-------------------------
Every number below is an **initial, tunable engineering parameter** (ruled
2026-09-30), not an owner-ruled constant. They live here, in one place, and
:func:`tunables` records them beside every verdict they produced.

Accept rate, and its known bias
-------------------------------
*Accept rate* = ``accepted / (accepted + dismissed)`` over **shown** label rows
(``shown=true`` at or above :data:`SHOWN_CONFIDENCE_THRESHOLD`), with
``auto_acted`` counted as accepted and ``ignored`` in neither numerator nor
denominator. **Known, accepted measurement caveat (2026-09-30):** counting
``auto_acted`` as accepted inflates the accept rate for reversible kinds at the
Autonomous tier, but the bias applies to both comparison windows
symmetrically, so the flip-back comparison stays valid.

Flip-back (WP05, spec User Story F)
-----------------------------------
:func:`evaluate_post_flip` compares the flipped classic model's accept rate
since ``metrics["flipped_at_ms"]`` with the heuristic's over its own last
:data:`BASELINE_WINDOW_DAYS` of label rows. Verdicts
(``metrics["flipback_verdict"]``): ``pending`` (fewer than
:data:`FLIPBACK_MIN_SHOWN` shown decisions, or no heuristic baseline, inside
the window), ``holding`` (not worse than baseline minus
:data:`DEMOTION_MARGIN_PP`), ``demoted`` (worse beyond it, or a sustained p95
over :data:`LATENCY_BUDGET_MS`), ``insufficient_data`` (the window passed
:data:`MAX_EVAL_WINDOW_DAYS` without the sample — the kind stays flipped).
"""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kenaz_ml.advice.contracts import contract_for

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable parameters (D-B7) — the one constants site for graduation and flip-back
# ---------------------------------------------------------------------------

#: The harness renders a chip only at or above this confidence (cross-repo).
SHOWN_CONFIDENCE_THRESHOLD = 75
#: FR-007: fewer shadow-labelled decisions than this is never ``eligible``.
MIN_GRADUATION_LABELS = 200
#: FR-008: eligible needs shadow precision this many points above the heuristic's.
MIN_PRECISION_DELTA_PP = 5.0
#: Trailing window over label rows (all-time when the log is younger).
TRAILING_WINDOW_DAYS = 90
#: Mission A NFR-002: per-kind ``/v1/recommend`` serving budget.
LATENCY_BUDGET_MS = 50.0
#: "Sustained" p95: the ring must hold at least this many serving samples
#: before a p95 over budget can demote. The ring (512) is the sustain window.
LATENCY_MIN_SAMPLES = 100
#: Shadow-to-label join: a label row within this many ms of the shadow record.
SHADOW_JOIN_TOLERANCE_MS = 120_000

# -- flip-back (WP05, spec User Story F) --
#: Post-flip evaluation renders no verdict before this many **shown** decided
#: (accepted/auto_acted/dismissed) rows served by the flipped classic model.
FLIPBACK_MIN_SHOWN = 30
#: Noise guard: demote only when the flipped rate < baseline - this many points.
DEMOTION_MARGIN_PP = 5.0
#: Maximum post-flip window. Past it without the sample: ``insufficient_data``,
#: the kind stays flipped, and the state is recorded and logged.
MAX_EVAL_WINDOW_DAYS = 21
#: The heuristic baseline: its last this-many days of label rows (all-time if fewer).
BASELINE_WINDOW_DAYS = TRAILING_WINDOW_DAYS

DAY_MS = 24 * 3600 * 1000

RECORD_SHADOW = "shadow"

VERDICT_NOT_ELIGIBLE = "not_eligible"
VERDICT_ELIGIBLE = "eligible"
VERDICT_DEMOTE = "demote"

# Flip-back verdicts (manifest ``metrics["flipback_verdict"]``). A demotion also
# sets the graduation ``metrics["verdict"]`` to :data:`VERDICT_DEMOTED`.
FLIPBACK_PENDING = "pending"  # window open, minimum sample not yet reached
FLIPBACK_HOLDING = "holding"  # sample reached, not worse than baseline beyond the margin
FLIPBACK_DEMOTED = "demoted"
FLIPBACK_INSUFFICIENT_DATA = "insufficient_data"
VERDICT_DEMOTED = "demoted"

ACCEPT_ACTIONS = frozenset({"accepted", "auto_acted"})  # auto_acted counts as accepted (ruled)
DISMISS_ACTIONS = frozenset({"dismissed"})  # ignored: in neither numerator nor denominator


def tunables() -> dict[str, Any]:
    """The graduation parameters in force, recorded with every verdict (D-B7)."""
    return {
        "shown_confidence_threshold": SHOWN_CONFIDENCE_THRESHOLD,
        "min_graduation_labels": MIN_GRADUATION_LABELS,
        "min_precision_delta_pp": MIN_PRECISION_DELTA_PP,
        "trailing_window_days": TRAILING_WINDOW_DAYS,
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "latency_min_samples": LATENCY_MIN_SAMPLES,
        "shadow_join_tolerance_ms": SHADOW_JOIN_TOLERANCE_MS,
        "flipback_min_shown": FLIPBACK_MIN_SHOWN,
        "demotion_margin_pp": DEMOTION_MARGIN_PP,
        "max_eval_window_days": MAX_EVAL_WINDOW_DAYS,
        "baseline_window_days": BASELINE_WINDOW_DAYS,
    }


# ---------------------------------------------------------------------------
# T012 — the shadow record, writer and reader
# ---------------------------------------------------------------------------


def vector_hash(kind: str, vector: Sequence[float]) -> str:
    """Engine-side join key over ``kind`` and its ordered feature vector (16 hex)."""
    payload = kind + "|" + "|".join(repr(float(v)) for v in vector)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def label_vector_hash(kind: str, record: Mapping[str, Any], names: Sequence[str]) -> str | None:
    """:func:`vector_hash` of a label row's features, or ``None`` when any is missing."""
    features = record.get("features") or {}
    try:
        return vector_hash(kind, [float(features[name]) for name in names])
    except (KeyError, TypeError, ValueError):
        return None


def shadow_record(
    kind: str,
    vector: Sequence[float],
    *,
    ts_ms: int,
    p: float,
    decision: bool,
    confidence: int,
    model_id_sha8: str | None,
    generation: str,
    rung: str,
    session_id: str | None = None,
    features_hash: str | None = None,
    label_ts: int | None = None,
) -> dict[str, Any]:
    """One shadow record. No ``user_action`` (not yet known) and no heuristic decision.

    ``features_hash``/``label_ts`` are the request's exact join key (the label
    row's ``(features_hash, ts)``) when the client sent it; ``ts`` stays the
    engine's receipt time.
    """
    return {
        "record": RECORD_SHADOW,
        "kind": kind,
        "vector_hash": vector_hash(kind, vector),
        "ts": int(ts_ms),
        "as_of_ms": int(ts_ms),
        "session_id": session_id,
        "features_hash": features_hash,
        "label_ts": int(label_ts) if label_ts is not None else None,
        "predicted": {"decision": bool(decision), "confidence": int(confidence), "p": float(p)},
        "model_id_sha8": model_id_sha8,
        "generation": str(generation),
        "rung": rung,
    }


@dataclass(frozen=True)
class ShadowWrite:
    """Outcome of a shadow write. ``ok=False`` carries the reason; nothing is raised."""

    ok: bool
    reason: str | None = None


def write_shadow_records(
    kind: str, records: Iterable[Mapping[str, Any]], *, directory: Path | str | None = None
) -> ShadowWrite:
    """Append shadow records to the kind's label log (Mission A's writer and bound). Never raises."""
    from kenaz_ml.advice.label_log import append_records

    try:
        append_records(kind, list(records), directory=directory)
    except Exception as exc:  # the log is best effort; recommendations never fail on it
        logger.warning("shadow: could not append to %r's label log (%s)", kind, exc)
        return ShadowWrite(False, f"{type(exc).__name__}: {exc}")
    return ShadowWrite(True)


@dataclass
class ShadowLog:
    """A tolerant read of one kind's label log, split by record type."""

    kind: str
    exists: bool
    shadows: list[dict[str, Any]] = field(default_factory=list)
    labels: list[dict[str, Any]] = field(default_factory=list)
    skipped_lines: int = 0
    truncated_final_line: bool = False


def read_shadow_log(kind: str, *, directory: Path | str | None = None) -> ShadowLog:
    """Read shadow and label records (highest revision per key). A missing file is "no log", not an error."""
    from kenaz_ml.advice.label_log import label_log_path, read_label_log

    path = label_log_path(kind, directory=directory)
    read = read_label_log(kind, directory=directory)
    log = ShadowLog(
        kind=kind,
        exists=path.exists(),
        labels=list(read.labels().values()),
        skipped_lines=read.skipped_lines,
        truncated_final_line=read.truncated_final_line,
    )
    for record in read.records:
        if record.get("record") != RECORD_SHADOW:
            continue
        predicted = record.get("predicted")
        if not isinstance(predicted, dict) or "vector_hash" not in record or "ts" not in record:
            log.skipped_lines += 1
            continue
        log.shadows.append(record)
    return log


@dataclass(frozen=True)
class JoinedDecision:
    shadow: dict[str, Any]
    label: dict[str, Any]


def _exact_key(shadow: Mapping[str, Any]) -> tuple[str, int] | None:
    """The shadow record's exact join key, or ``None`` when either half is absent."""
    features_hash, label_ts = shadow.get("features_hash"), shadow.get("label_ts")
    if not features_hash or label_ts is None:
        return None
    try:
        return str(features_hash), int(label_ts)
    except (TypeError, ValueError):
        return None


def join_shadow_to_labels(kind: str, log: ShadowLog) -> list[JoinedDecision]:
    """Pair each shadow record with its label row (see the module docstring). Deterministic.

    Exact ``(features_hash, ts)`` pass first; the vector-hash fallback only for
    records without the exact key, and only over label rows the exact pass did
    not take.
    """
    contract = contract_for(kind)
    if contract is None:
        return []
    names = tuple(contract.names)
    shadows = sorted(log.shadows, key=lambda s: int(s["ts"]))

    used: set[int] = set()
    joined: list[JoinedDecision] = []

    # Pass 1 — exact key. Several label rows can share (features_hash, ts)
    # across clients; the first unused one (log order) is taken.
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for label in log.labels:
        try:
            by_key.setdefault((str(label["features_hash"]), int(label["ts"])), []).append(label)
        except (KeyError, TypeError, ValueError):
            continue
    fallback: list[dict[str, Any]] = []
    for shadow in shadows:
        key = _exact_key(shadow)
        if key is None:
            fallback.append(shadow)
            continue
        for label in by_key.get(key, ()):
            if id(label) not in used:
                used.add(id(label))
                joined.append(JoinedDecision(shadow=shadow, label=label))
                break

    # Pass 2 — the vector-hash heuristic, superseded wherever the exact key exists.
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for label in log.labels:
        h = label_vector_hash(kind, label, names)
        if h is not None:
            by_hash.setdefault(h, []).append(label)
    for shadow in fallback:
        best: dict[str, Any] | None = None
        best_gap: int | None = None
        for label in by_hash.get(str(shadow["vector_hash"]), ()):
            if id(label) in used:
                continue
            s_session, l_session = shadow.get("session_id"), label.get("session_id")
            if s_session is not None and l_session is not None and s_session != l_session:
                continue
            gap = abs(int(label["ts"]) - int(shadow["ts"]))
            if gap <= SHADOW_JOIN_TOLERANCE_MS and (best_gap is None or gap < best_gap):
                best, best_gap = label, gap
        if best is not None:
            used.add(id(best))
            joined.append(JoinedDecision(shadow=shadow, label=best))
    return joined


# ---------------------------------------------------------------------------
# T013 — the write-site interface (the call lands in dispatch.py, WP04)
# ---------------------------------------------------------------------------

#: Bounded queue between the request path and the shadow writer thread. A full
#: queue drops the record (counted) rather than delaying a response.
SHADOW_QUEUE_MAX = 4096


@dataclass
class _Pending:
    kind: str
    vector: tuple[float, ...]
    model: Any
    manifest: Any
    ts_ms: int
    session_id: str | None
    directory: Path | None
    features_hash: str | None = None
    label_ts: int | None = None


class ShadowWriter:
    """Scores and appends shadow records off the request path (one daemon thread)."""

    def __init__(self, maxsize: int = SHADOW_QUEUE_MAX) -> None:
        self._queue: queue.Queue[_Pending | None] = queue.Queue(maxsize=maxsize)
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()
        self.dropped = 0
        self.failed = 0
        self.written = 0

    def _ensure_started(self) -> None:
        with self._start_lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="advice-shadow-writer", daemon=True)
                self._thread.start()

    def submit(self, pending: _Pending) -> ShadowWrite:
        try:
            self._ensure_started()
            self._queue.put_nowait(pending)
        except queue.Full:
            self.dropped += 1
            return ShadowWrite(False, "shadow queue full; record dropped")
        except Exception as exc:  # never propagate to the request path
            self.failed += 1
            return ShadowWrite(False, f"{type(exc).__name__}: {exc}")
        return ShadowWrite(True)

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until every submitted record is handled (tests and shutdown)."""
        if self._thread is None:
            return True
        done = threading.Event()
        worker = threading.Thread(target=lambda: (self._queue.join(), done.set()), daemon=True)
        worker.start()
        return done.wait(timeout)

    def _run(self) -> None:
        while True:
            pending = self._queue.get()
            try:
                if pending is None:
                    return
                result = self._write(pending)
                if result.ok:
                    self.written += 1
                else:
                    self.failed += 1
            except Exception:
                self.failed += 1
                logger.warning("shadow: writer failed", exc_info=True)
            finally:
                self._queue.task_done()

    @staticmethod
    def _write(pending: _Pending) -> ShadowWrite:
        from kenaz_ml.advice.models import decision_and_confidence, positive_probability

        p = positive_probability(pending.model, pending.vector)
        decision, confidence = decision_and_confidence(p)
        manifest = pending.manifest
        record = shadow_record(
            pending.kind,
            pending.vector,
            ts_ms=pending.ts_ms,
            p=p,
            decision=decision,
            confidence=confidence,
            model_id_sha8=(getattr(manifest, "artifact_sha256", "") or "")[:8] or None,
            generation=str(getattr(manifest, "version", "0")),
            rung=str((getattr(manifest, "metrics", None) or {}).get("rung") or "R2"),
            session_id=pending.session_id,
            features_hash=pending.features_hash,
            label_ts=pending.label_ts,
        )
        return write_shadow_records(pending.kind, [record], directory=pending.directory)


SHADOW_WRITER = ShadowWriter()


def enqueue_shadow(
    kind: str,
    vector: Sequence[float],
    model: Any,
    manifest: Any,
    *,
    ts_ms: int,
    session_id: str | None = None,
    directory: Path | str | None = None,
    writer: ShadowWriter | None = None,
    features_hash: str | None = None,
    ts: int | None = None,
) -> ShadowWrite:
    """Record one shadow-rung prediction. Non-raising, non-blocking: O(1) on the request path.

    ``features_hash``/``ts`` are the request's optional exact join key (the
    label row's ``(features_hash, ts)``); ``ts`` is stored as ``label_ts``
    because the record's own ``ts`` is ``ts_ms``, the engine's receipt time.

    Scoring (calibrated probability, decision, confidence) and the append
    happen on :data:`SHADOW_WRITER`'s thread. The returned value says only
    whether the record was queued.
    """
    try:
        pending = _Pending(
            kind=kind,
            vector=tuple(float(v) for v in vector),
            model=model,
            manifest=manifest,
            ts_ms=int(ts_ms),
            session_id=session_id,
            directory=Path(directory) if directory is not None else None,
            features_hash=features_hash or None,
            label_ts=int(ts) if ts is not None else None,
        )
        return (writer or SHADOW_WRITER).submit(pending)
    except Exception as exc:
        return ShadowWrite(False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# T014 — accept rates and the graduation check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateStats:
    """``accepted / (accepted + dismissed)``; ``ignored`` counted but outside both."""

    accepted: int = 0
    dismissed: int = 0
    ignored: int = 0

    @property
    def decided(self) -> int:
        return self.accepted + self.dismissed

    @property
    def rate(self) -> float | None:
        return self.accepted / self.decided if self.decided else None


def accept_rate(actions: Iterable[str | None]) -> RateStats:
    accepted = dismissed = ignored = 0
    for action in actions:
        if action in ACCEPT_ACTIONS:
            accepted += 1
        elif action in DISMISS_ACTIONS:
            dismissed += 1
        else:
            ignored += 1
    return RateStats(accepted, dismissed, ignored)


def is_shown(label: Mapping[str, Any]) -> bool:
    """A label row the harness showed: ``shown=true`` and confidence at/above the threshold."""
    if not label.get("shown"):
        return False
    confidence = label.get("confidence")
    return confidence is None or int(confidence) >= SHOWN_CONFIDENCE_THRESHOLD


def served_by_heuristic(label: Mapping[str, Any]) -> bool:
    """The harness's heuristic served this row (``model_id`` ``heuristic...``).

    Exclusive with :func:`served_by_classic`: when ``model_id`` is present it
    alone decides (the harness writes the engine's ``model`` verbatim), so a
    ``classic/``/``laya/``/unknown ``model_id`` is never counted as the
    heuristic because of its ``rung``. ``rung == "heuristic"`` decides only for
    a row with no ``model_id``. Anything else (laya, a future backend) counts
    on **neither** side of any comparison. (Review fix 2026-09-30.)
    """
    model_id = str(label.get("model_id") or "")
    if model_id:
        return model_id.startswith("heuristic")
    return label.get("rung") == "heuristic"


def served_by_classic(label: Mapping[str, Any], kind: str) -> bool:
    """This engine's classic backend served this row: ``model_id`` is its ``classic/<kind>@<generation>`` label."""
    return str(label.get("model_id") or "").startswith(f"classic/{kind}@")


def in_window(label: Mapping[str, Any], start_ms: int, end_ms: int) -> bool:
    try:
        ts = int(label["ts"])
    except (KeyError, TypeError, ValueError):
        return False
    return start_ms <= ts <= end_ms


def trailing_window(now_ms: int, days: int = TRAILING_WINDOW_DAYS) -> tuple[int, int]:
    """``(start, end)`` of the trailing window. All-time when the log is younger than it."""
    return now_ms - days * DAY_MS, now_ms


def heuristic_baseline(labels: Iterable[Mapping[str, Any]], start_ms: int, end_ms: int) -> RateStats:
    """The heuristic's accept rate over shown label rows it served, in the window."""
    return accept_rate(
        label.get("user_action")
        for label in labels
        if in_window(label, start_ms, end_ms) and is_shown(label) and served_by_heuristic(label)
    )


def classic_live_rate(labels: Iterable[Mapping[str, Any]], kind: str, start_ms: int, end_ms: int) -> RateStats:
    """The engine's classic backend's accept rate over shown label rows it served, in the window."""
    return accept_rate(
        label.get("user_action")
        for label in labels
        if in_window(label, start_ms, end_ms) and is_shown(label) and served_by_classic(label, kind)
    )


def shadow_would_show(shadow: Mapping[str, Any]) -> bool:
    """The model would have recommended: decision true at/above the shown threshold."""
    predicted = shadow.get("predicted") or {}
    return bool(predicted.get("decision")) and int(predicted.get("confidence", 0)) >= SHOWN_CONFIDENCE_THRESHOLD


def delta_pp(a: float | None, b: float | None) -> float | None:
    """``100 * (a - b)`` in percentage points, rounded to 6 dp so a boundary is exact."""
    if a is None or b is None:
        return None
    return round(100.0 * (a - b), 6)


@dataclass
class GraduationResult:
    """Every number behind a verdict, never only the verdict (FR-006)."""

    kind: str
    verdict: str
    reason: str
    evaluated_at_ms: int
    generation: str | None = None
    promoted: bool = False
    label_count: int = 0
    shadow_shown: int = 0
    shadow_precision: float | None = None
    heuristic_shown: int = 0
    heuristic_precision: float | None = None
    precision_delta_pp: float | None = None
    live_shown: int | None = None
    live_precision: float | None = None
    calibration_ece: float | None = None
    calibration_fitted: bool = False
    latency_p95_ms: float | None = None
    latency_samples: int = 0
    latency_evaluated: bool = False
    window_start_ms: int = 0
    window_end_ms: int = 0
    shadow_log_skipped_lines: int = 0

    def to_metrics(self) -> dict[str, Any]:
        """The manifest ``metrics`` entries (FR-009)."""
        return {
            "verdict": self.verdict,
            "verdict_reason": self.reason,
            "verdict_at_ms": self.evaluated_at_ms,
            "verdict_generation": int(self.generation) if self.generation and self.generation.isdigit() else None,
            "shadow_window_labels": self.label_count,
            "shadow_shown": self.shadow_shown,
            "precision_at_shown_threshold": self.shadow_precision,
            "heuristic_precision": self.heuristic_precision,
            "heuristic_shown": self.heuristic_shown,
            "precision_delta": self.precision_delta_pp,
            "live_precision": self.live_precision,
            "live_shown": self.live_shown,
            "window_size_days": TRAILING_WINDOW_DAYS,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "calibration_ece": self.calibration_ece,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_samples": self.latency_samples,
            "latency_evaluated": self.latency_evaluated,
            "graduation_params": tunables(),
        }


def _latency_over_budget(p95: float | None, samples: int) -> bool:
    return p95 is not None and samples >= LATENCY_MIN_SAMPLES and p95 > LATENCY_BUDGET_MS


def graduation_check(
    kind: str,
    *,
    now_ms: int,
    manifest: Any = None,
    models_dir: Path | str | None = None,
    retained_dir: Path | str | None = None,
    latency_p95_ms: float | None = None,
    latency_samples: int = 0,
) -> GraduationResult:
    """Compute ``kind``'s verdict (FR-006..FR-008, FR-010). Never raises.

    ``manifest`` defaults to the local slot's; the latency arguments are the
    dispatch table's per-kind ring (Mission A FR-025) and are evaluated only
    for a kind already served by ``classic``.
    """
    try:
        return _graduation_check(kind, now_ms, manifest, models_dir, retained_dir, latency_p95_ms, latency_samples)
    except Exception as exc:
        logger.exception("graduation: check for %r failed", kind)
        return GraduationResult(kind, VERDICT_NOT_ELIGIBLE, f"check failed: {type(exc).__name__}", now_ms)


def _graduation_check(
    kind: str,
    now_ms: int,
    manifest: Any,
    models_dir: Path | str | None,
    retained_dir: Path | str | None,
    latency_p95_ms: float | None,
    latency_samples: int,
) -> GraduationResult:
    if manifest is None:
        from kenaz_ml.advice.training import previous_manifest

        manifest = previous_manifest(kind, models_dir)
    start, end = trailing_window(now_ms)
    result = GraduationResult(kind, VERDICT_NOT_ELIGIBLE, "", now_ms, window_start_ms=start, window_end_ms=end)
    if manifest is None:
        result.reason = "no trained model"
        return result

    metrics = manifest.metrics or {}
    # Review fix 2026-09-30 (D-B5 "a fresh eligible verdict"): after a
    # demotion, evidence from before it does not count. The shadow records
    # that earned the demoted flip were contradicted by live accept rates;
    # letting the next generation graduate on them re-flipped at the very next
    # retrain (24 h / 50 labels later) — a flap cycle. The window therefore
    # opens at ``demoted_at_ms`` (carried forward by retrains, cleared by the
    # next successful flip) for both the shadow side and the heuristic side.
    demoted_at = metrics.get("demoted_at_ms")
    if demoted_at is not None and int(demoted_at) > start:
        start = int(demoted_at)
        result.window_start_ms = start
    result.generation = str(manifest.version)
    result.calibration_fitted = metrics.get("calibration_fitted") is True
    result.calibration_ece = metrics.get("calibration_ece")
    result.promoted = metrics.get("serving_backend") == "classic"

    log = read_shadow_log(kind, directory=retained_dir)
    result.shadow_log_skipped_lines = log.skipped_lines
    in_range = [label for label in log.labels if in_window(label, start, end)]
    baseline = heuristic_baseline(in_range, start, end)
    result.heuristic_shown = baseline.decided
    result.heuristic_precision = baseline.rate

    if result.promoted:
        return _promoted_verdict(result, log.labels, metrics, now_ms, latency_p95_ms, latency_samples)

    if not log.exists or not log.shadows:
        result.reason = "no shadow log"
        return result

    decided = [
        j
        for j in join_shadow_to_labels(kind, log)
        if in_window(j.label, start, end) and j.label.get("user_action") in ACCEPT_ACTIONS | DISMISS_ACTIONS
    ]
    result.label_count = len(decided)
    shown = accept_rate(j.label.get("user_action") for j in decided if shadow_would_show(j.shadow))
    result.shadow_shown = shown.decided
    result.shadow_precision = shown.rate
    result.precision_delta_pp = delta_pp(shown.rate, baseline.rate)

    if result.label_count < MIN_GRADUATION_LABELS:
        result.reason = f"{result.label_count} shadow-labelled decisions < {MIN_GRADUATION_LABELS}"
        if demoted_at is not None and start == int(demoted_at):
            result.reason += " since the demotion (earlier evidence does not count)"
    elif shown.rate is None:
        result.reason = "no shadow prediction at or above the shown threshold"
    elif baseline.rate is None:
        result.reason = "no heuristic baseline in the window"
    elif not result.calibration_fitted:
        result.reason = "calibration not fitted"
    elif result.precision_delta_pp is None or result.precision_delta_pp < MIN_PRECISION_DELTA_PP:
        result.reason = f"precision delta {result.precision_delta_pp} pp < {MIN_PRECISION_DELTA_PP} pp"
    else:
        result.verdict = VERDICT_ELIGIBLE
        result.reason = f"precision delta {result.precision_delta_pp} pp >= {MIN_PRECISION_DELTA_PP} pp"
    return result


def _promoted_verdict(
    result: GraduationResult,
    labels: list[dict[str, Any]],
    metrics: Mapping[str, Any],
    now_ms: int,
    latency_p95_ms: float | None,
    latency_samples: int,
) -> GraduationResult:
    """An already-promoted kind: the demotion criteria are exactly the flip-back's (D-B5).

    One evaluator, not two: FR-010's "trailing precision below the heuristic's"
    is measured by :func:`evaluate_post_flip` with its minimum sample and noise
    guard, and the sustained-p95 criterion is the same one.
    """
    flip = _evaluate_post_flip(result.kind, labels, metrics, result.generation, now_ms, latency_p95_ms, latency_samples)
    result.live_shown = flip.flipped.decided
    result.live_precision = flip.flipped.rate
    result.heuristic_precision = flip.baseline.rate
    result.heuristic_shown = flip.baseline.decided
    result.latency_p95_ms = latency_p95_ms
    result.latency_samples = latency_samples
    result.latency_evaluated = True
    if flip.verdict == FLIPBACK_DEMOTED:
        result.verdict = VERDICT_DEMOTE
    else:
        result.verdict = VERDICT_ELIGIBLE
    result.reason = flip.reason
    return result


# ---------------------------------------------------------------------------
# WP05 — post-flip evaluation (spec User Story F, D-B5, FR-015..FR-017)
# ---------------------------------------------------------------------------


@dataclass
class FlipbackResult:
    """The post-flip comparison with both windows' numbers (never only a verdict)."""

    kind: str
    verdict: str
    reason: str
    evaluated_at_ms: int
    generation: str | None
    flipped_at_ms: int | None
    flipped: RateStats = field(default_factory=RateStats)
    flipped_window: tuple[int, int] = (0, 0)
    baseline: RateStats = field(default_factory=RateStats)
    baseline_window: tuple[int, int] | None = None
    delta_pp: float | None = None
    latency_p95_ms: float | None = None
    latency_samples: int = 0

    def to_metrics(self) -> dict[str, Any]:
        return {
            "flipback_verdict": self.verdict,
            "flipback_reason": self.reason,
            "flipback_evaluated_at_ms": self.evaluated_at_ms,
            "flipback_generation": int(self.generation) if self.generation and self.generation.isdigit() else None,
            "flipped_accept_rate": self.flipped.rate,
            "flipped_accepted": self.flipped.accepted,
            "flipped_dismissed": self.flipped.dismissed,
            "flipped_ignored": self.flipped.ignored,
            "flipped_window_start_ms": self.flipped_window[0],
            "flipped_window_end_ms": self.flipped_window[1],
            "baseline_accept_rate": self.baseline.rate,
            "baseline_accepted": self.baseline.accepted,
            "baseline_dismissed": self.baseline.dismissed,
            "baseline_ignored": self.baseline.ignored,
            "baseline_window_start_ms": self.baseline_window[0] if self.baseline_window else None,
            "baseline_window_end_ms": self.baseline_window[1] if self.baseline_window else None,
            "flipback_delta_pp": self.delta_pp,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_samples": self.latency_samples,
            "flipback_params": tunables(),
        }


def heuristic_baseline_window(labels: Iterable[Mapping[str, Any]]) -> tuple[RateStats, tuple[int, int] | None]:
    """The heuristic's accept rate over **its** last :data:`BASELINE_WINDOW_DAYS` of label rows.

    The window ends at the newest label row the heuristic served (after a flip
    the heuristic may stop producing rows, so "now" would slide its window
    empty) and spans ``BASELINE_WINDOW_DAYS`` back — all-time when its rows are
    younger than that. ``(empty stats, None)`` when the heuristic served none.
    """
    served = [label for label in labels if served_by_heuristic(label) and "ts" in label]
    if not served:
        return RateStats(), None
    end = max(int(label["ts"]) for label in served)
    start = end - BASELINE_WINDOW_DAYS * DAY_MS
    return heuristic_baseline(served, start, end), (start, end)


def evaluate_post_flip(
    kind: str,
    *,
    now_ms: int,
    manifest: Any = None,
    models_dir: Path | str | None = None,
    retained_dir: Path | str | None = None,
    latency_p95_ms: float | None = None,
    latency_samples: int = 0,
) -> FlipbackResult:
    """Compare the flipped model's live accept rate with the heuristic baseline (FR-015). Never raises.

    Reads ``user_action`` from the kind's label log (highest ``revision`` per
    key) at evaluation time; the post-flip window runs from
    ``metrics["flipped_at_ms"]`` to ``now_ms``.
    """
    try:
        if manifest is None:
            from kenaz_ml.advice.training import previous_manifest

            manifest = previous_manifest(kind, models_dir)
        if manifest is None:
            return FlipbackResult(kind, FLIPBACK_PENDING, "no trained model", now_ms, None, None)
        labels = read_shadow_log(kind, directory=retained_dir).labels
        return _evaluate_post_flip(
            kind, labels, manifest.metrics or {}, str(manifest.version), now_ms, latency_p95_ms, latency_samples
        )
    except Exception as exc:
        logger.exception("flip-back: evaluation of %r failed", kind)
        return FlipbackResult(kind, FLIPBACK_PENDING, f"evaluation failed: {type(exc).__name__}", now_ms, None, None)


def _evaluate_post_flip(
    kind: str,
    labels: list[dict[str, Any]],
    metrics: Mapping[str, Any],
    generation: str | None,
    now_ms: int,
    latency_p95_ms: float | None,
    latency_samples: int,
) -> FlipbackResult:
    flipped_at = metrics.get("flipped_at_ms")
    start = int(flipped_at) if flipped_at is not None else trailing_window(now_ms)[0]
    flipped = classic_live_rate(labels, kind, start, now_ms)
    baseline, baseline_window = heuristic_baseline_window(labels)
    result = FlipbackResult(
        kind=kind,
        verdict=FLIPBACK_PENDING,
        reason="",
        evaluated_at_ms=now_ms,
        generation=generation,
        flipped_at_ms=int(flipped_at) if flipped_at is not None else None,
        flipped=flipped,
        flipped_window=(start, now_ms),
        baseline=baseline,
        baseline_window=baseline_window,
        delta_pp=delta_pp(flipped.rate, baseline.rate),
        latency_p95_ms=latency_p95_ms,
        latency_samples=latency_samples,
    )
    if _latency_over_budget(latency_p95_ms, latency_samples):
        result.verdict = FLIPBACK_DEMOTED
        result.reason = (
            f"sustained p95 {latency_p95_ms:.1f} ms over the {LATENCY_BUDGET_MS:.0f} ms budget "
            f"({latency_samples} samples)"
        )
        return result

    window_elapsed = now_ms - start >= MAX_EVAL_WINDOW_DAYS * DAY_MS
    if flipped.decided < FLIPBACK_MIN_SHOWN or baseline.rate is None:
        missing = (
            f"{flipped.decided} post-flip shown decisions < {FLIPBACK_MIN_SHOWN}"
            if flipped.decided < FLIPBACK_MIN_SHOWN
            else "no heuristic baseline rows"
        )
        if window_elapsed:
            result.verdict = FLIPBACK_INSUFFICIENT_DATA
            result.reason = f"{missing} after the {MAX_EVAL_WINDOW_DAYS}-day window; the kind stays flipped"
        else:
            result.reason = f"{missing}; window open"
        return result

    if result.delta_pp is not None and result.delta_pp < -DEMOTION_MARGIN_PP:
        result.verdict = FLIPBACK_DEMOTED
        result.reason = (
            f"flipped accept rate {flipped.rate:.3f} < baseline {baseline.rate:.3f} - {DEMOTION_MARGIN_PP} pp "
            f"(delta {result.delta_pp} pp)"
        )
    else:
        result.verdict = FLIPBACK_HOLDING
        result.reason = (
            f"flipped accept rate within {DEMOTION_MARGIN_PP} pp of baseline or better (delta {result.delta_pp} pp)"
        )
    return result


# ---------------------------------------------------------------------------
# T015 — the verdict into manifest ``metrics``
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricsWrite:
    ok: bool
    manifest: Any = None
    reason: str | None = None


def update_manifest_metrics(
    kind: str,
    updates: Mapping[str, Any],
    *,
    models_dir: Path | str | None = None,
    expected_version: str | None = None,
    remove: Sequence[str] = (),
) -> MetricsWrite:
    """Merge ``updates`` into the local manifest's ``metrics`` (atomic; FR-009). Never raises.

    Every other key survives; no top-level key is added; the artifact and
    ``artifact_sha256`` are not touched. ``expected_version`` refuses the write
    when a retrain replaced the manifest in between. A failed write leaves the
    previous manifest intact.
    """
    import dataclasses

    from kenaz_ml import config
    from kenaz_ml.modelstore.registry import read_manifest, write_manifest

    directory = Path(models_dir) if models_dir is not None else config.models_dir()
    path = directory / f"{kind}.json"
    read = read_manifest(path)
    if not read.ok or read.manifest is None:
        return MetricsWrite(False, reason=f"no readable manifest for {kind!r}")
    current = read.manifest
    if expected_version is not None and str(current.version) != str(expected_version):
        return MetricsWrite(False, current, f"manifest is version {current.version}, expected {expected_version}")
    metrics = {k: v for k, v in dict(current.metrics).items() if k not in set(remove)}
    metrics.update(dict(updates))
    updated = dataclasses.replace(current, metrics=metrics)
    try:
        write_manifest(path, updated)
    except Exception as exc:
        logger.warning("graduation: could not write %s (%s); previous manifest kept", path, exc)
        return MetricsWrite(False, current, f"{type(exc).__name__}: {exc}")
    return MetricsWrite(True, updated)
