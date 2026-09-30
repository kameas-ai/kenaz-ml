"""The ``/v1/recommend/{kind}`` dispatch layer (two-client-engine-01MSK2EN WP03).

One in-process **dispatch table** maps each recommendation kind to the single
backend that serves it — ``classic`` (a registry-validated scikit-learn
artifact) or ``laya`` (an in-process ONNXAgent; refuses unless a checkpoint is
installed and the host eligible) — or to *no*
backend, in which case the kind refuses with a typed ``kind_not_served``.

The engine ships **no heuristic backend** (Amendment A3.2, plan D-A5). The
three harness kinds (``branch_now``, ``compact_now``, ``escalate_model``) are
registered seed kinds that refuse ``kind_not_served`` until a graduated model
serves them; falling back to a rule is the *client's* job. ``heuristic`` stays a
legal value of the backend enum and the response schema, reserved for test
fixtures, which install themselves through :meth:`DispatchTable.replace_entry`
from ``tests/fixtures/advice_backends.py`` — nothing under ``src/`` implements
one (:data:`SHIPPED_BACKENDS` is the complete list of shipped implementations,
and a structural test pins it).

Population (plan D-A3)
----------------------
At startup :func:`build_table` seeds the three harness kinds, then resolves
each through the registry (local slot, then base slot, sha256 before
deserialize, ordered contract equality against ``advice/contracts.py``). It
also scans both slots for any other manifest that declares a serving backend
(``metrics["serving_backend"]``) under a name not claimed by the workbench
models. A workbench name can never become a recommend kind:
:meth:`DispatchTable.replace_entry` raises on one, so ``/v1/recommend/stuck``
cannot alias ``/predict/stuck``.

Where a manifest names its backend
----------------------------------
``Manifest.metrics["serving_backend"]`` — the same open ``metrics`` dict
``harness-recommendation-models-01MSK2RM`` writes its verdicts and its
persisted flip into (its WP04 T019). Values:

* ``"classic"`` — served by the validated artifact (:class:`ClassicBackend`).
* ``"laya"`` — served by :class:`LayaBackend` **in-process** (never over HTTP to
  ``/v1/systemone``; C-002) when a verified checkpoint directory is installed and
  the host is eligible; otherwise refuses ``laya_backend_not_installed`` (no
  checkpoint, or laya itself absent from the build) or ``host_ineligible`` --
  never substituted (A3.2).
* absent or ``"heuristic"`` — the kind is *not flipped*: the client's heuristic
  answers, so the engine refuses ``kind_not_served``.
* anything else — refuses ``backend_unknown`` rather than defaulting.

``Manifest.metrics["rung"]`` is the reported ``rung`` when present.

Snapshot consistency
--------------------
Readers take :meth:`DispatchTable.snapshot` once per request — an immutable
mapping — and use only that. :meth:`DispatchTable.replace_entry` builds a new
mapping and swaps the reference under a lock, so a request sees the whole
pre-swap or the whole post-swap entry, never half of each. That method is the
seam the sibling mission extends for hot reload and the classic flip.

Refusals
--------
Every refusal has the same shape and the same HTTP status (see
:data:`REFUSAL_STATUS_CODE` and :class:`RecommendRefusal`) with a stable
machine ``reason``. The harness maps any of them to ``ErrNoAdvice``.

This module makes no network call of any kind (C-002, C-003).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, model_validator

from kenaz_ml.advice.contracts import KIND_IDS, contract_for

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------

BACKEND_HEURISTIC = "heuristic"  # reserved for test fixtures (A3.2)
BACKEND_CLASSIC = "classic"
BACKEND_LAYA = "laya"
#: The closed backend set. Anything else in a manifest refuses that kind.
BACKENDS: tuple[str, ...] = (BACKEND_HEURISTIC, BACKEND_CLASSIC, BACKEND_LAYA)

#: Manifest ``metrics`` keys this layer reads.
BACKEND_METRIC_KEY = "serving_backend"
RUNG_METRIC_KEY = "rung"
BENCHMARKED_METRIC_KEY = "benchmarked"

#: The workbench model names (``/predict/*``, ``ml_predictions.model``). The two
#: namespaces are disjoint by construction — see :meth:`DispatchTable.replace_entry`.
WORKBENCH_MODEL_NAMES: frozenset[str] = frozenset({"stuck", "activity", "workflow", "duration", "quality", "suggest"})

# Typed refusal reasons — stable, machine-readable, safe to branch on.
REASON_UNKNOWN_KIND = "unknown_kind"
REASON_KIND_NOT_SERVED = "kind_not_served"
REASON_LAYA_NOT_INSTALLED = "laya_backend_not_installed"
REASON_CONTRACT_MISMATCH = "contract_mismatch"
REASON_FEATURES_MISMATCH = "features_mismatch"
REASON_FEATURES_INVALID = "features_invalid"
REASON_KIND_ID_MISMATCH = "kind_id_mismatch"
REASON_BACKEND_UNKNOWN = "backend_unknown"
REASON_NO_CONTRACT = "no_contract"
REASON_CONFIDENCE_OUT_OF_RANGE = "confidence_out_of_range"
REASON_BACKEND_ERROR = "backend_error"
REASON_HOST_INELIGIBLE = "host_ineligible"
REASON_CHECKPOINT_REFUSED = "checkpoint_refused"
LAYA_NOT_INSTALLED_DETAIL = "kind unavailable: laya backend not installed"

#: The one HTTP status every typed refusal uses. 422 is in the harness's
#: ``ErrNoAdvice`` mapping (503 / timeout / connect-refused / 422 / refusal);
#: the body's ``refusal.reason`` distinguishes the cause.
REFUSAL_STATUS_CODE = 422

#: Latency ring size per kind (FR-025).
LATENCY_RING_SIZE = 512


# ---------------------------------------------------------------------------
# Wire models
# ---------------------------------------------------------------------------


class RecommendRequest(BaseModel):
    """``POST /v1/recommend/{kind}`` body (design doc §3.2)."""

    features: dict[str, float] = Field(..., description="Feature values keyed by name; must match the kind's contract.")
    feature_contract_version: str = Field(..., description="The contract version the client computed features under.")
    session_id: str | None = Field(None, description="Opaque client session id; not interpreted by the engine.")
    kind_id: str | None = Field(None, description="Optional; when present it must equal the path's kind.")
    # Exact shadow-join key (ruled 2026-09-30 from Mission B's contradiction):
    # the same (features_hash, ts) the harness later pushes on the label row.
    # Optional and absent-tolerated; the engine never interprets them for
    # serving -- they are carried on the parsed request for the shadow write
    # site (harness-recommendation-models-01MSK2RM owns the join).
    features_hash: str | None = Field(
        None, min_length=1, description="Optional: the label row's features_hash for this decision (exact shadow join)."
    )
    ts: int | None = Field(
        None, gt=0, description="Optional: the label row's ts (decision time, epoch ms) for this decision."
    )


class RecommendResponse(BaseModel):
    """A served recommendation (FR-004). Exactly one of ``decision``/``score``.

    Field semantics for values the specs leave open:

    * ``rung`` — the serving manifest's ``metrics["rung"]`` when present,
      otherwise the backend name (a test fixture reports ``"heuristic"``).
    * ``checkpoint_provenance`` — the registry slot that served the artifact
      (``local`` / ``base``); ``org`` is reserved for an org-distributed pack.
      A test fixture has no checkpoint and reports ``local`` (in-process).
    * ``model_id_sha8`` — first eight hex characters of the serving artifact's
      ``artifact_sha256``; ``null`` when there is no artifact (a fixture).
    * ``generation`` — the serving manifest's ``Manifest.version``; ``"0"`` when
      there is no artifact.
    * ``unbenchmarked`` — ``true`` unless the manifest records
      ``metrics["benchmarked"] is True``; always ``true`` for a fixture.
    """

    decision: bool | None = None
    score: int | None = None
    confidence: int = Field(..., ge=0, le=100, description="0-100, validated, never clamped.")
    kind_id: str
    feature_contract_version: str
    model: str
    rung: str
    backend: Literal["heuristic", "classic", "laya"]
    model_id_sha8: str | None
    checkpoint_provenance: Literal["local", "org", "base"]
    generation: str
    unbenchmarked: bool

    @model_validator(mode="after")
    def _exactly_one_answer(self) -> RecommendResponse:
        if (self.decision is None) == (self.score is None):
            raise ValueError("exactly one of decision or score must be set")
        return self


class RefusalBody(BaseModel):
    kind_id: str
    reason: str = Field(..., description="Stable machine reason, e.g. kind_not_served, contract_mismatch.")
    detail: str = Field(..., description="Human-readable diagnostic, e.g. 'kind unavailable: contract v3 != v2'.")


class RecommendRefusal(BaseModel):
    """Every typed refusal: HTTP :data:`REFUSAL_STATUS_CODE`, this body.

    ``error`` is the stable machine code at the top level — the envelope the
    harness's ``mlsidecar`` client parses (``{"error": "<code>"}``; a code of
    ``kind_not_served`` satisfies its ``ErrKindNotServed``). ``refusal`` repeats
    it with the kind and a human-readable detail.
    """

    error: str = Field(..., description="Stable typed code, e.g. kind_not_served, contract_mismatch.")
    refusal: RefusalBody


class ContractEntry(BaseModel):
    """One kind in ``GET /v1/contracts``, keyed by kind id in :class:`ContractsResponse`.

    ``features``/``backend``/``available`` are the fields the harness's
    ``KindContract`` reads; ``available`` is true only when a backend actually
    serves the kind (the harness gates routing on it). ``version`` is the
    kind's 16-hex contract version (D-A4).
    """

    features: list[str] = Field(..., description="Ordered feature names (the vector layout).")
    dtypes: list[str]
    version: str
    supported_versions: list[str] = Field(
        ..., description="Versions accepted by /v1/recommend. Only the current one today (no N-1 defined)."
    )
    available: bool
    backend: str | None = None
    reason: str | None = None
    detail: str | None = None


class ContractsResponse(BaseModel):
    kinds: dict[str, ContractEntry] = Field(..., description="Every registered kind, keyed by kind id.")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendAnswer:
    """What a backend returns. ``confidence`` is validated by the dispatcher."""

    decision: bool | None = None
    score: int | None = None
    confidence: Any = None


class Backend(Protocol):
    """A serving backend. ``name`` must be one of :data:`BACKENDS`."""

    name: str
    model_label: str
    rung: str
    checkpoint_provenance: str
    model_id_sha8: str | None
    generation: str
    unbenchmarked: bool

    def answer(self, vector: tuple[float, ...]) -> BackendAnswer: ...


@dataclass
class ClassicBackend:
    """A registry-validated scikit-learn classifier.

    Answers ``decision = P(positive) >= 0.5`` with ``confidence`` the rounded
    probability of the decided class, ``round(100 * max(p, 1 - p))`` — the
    minimal honest mapping; ``harness-recommendation-models-01MSK2RM`` owns
    calibration and may redefine it.
    """

    model: Any
    kind_id: str
    manifest: Any
    slot: str
    name: str = BACKEND_CLASSIC

    @property
    def model_label(self) -> str:
        return f"classic/{self.kind_id}@{self.manifest.version}"

    @property
    def rung(self) -> str:
        return str(self.manifest.metrics.get(RUNG_METRIC_KEY) or BACKEND_CLASSIC)

    @property
    def checkpoint_provenance(self) -> str:
        return self.slot if self.slot in ("local", "base") else "local"

    @property
    def model_id_sha8(self) -> str | None:
        return (self.manifest.artifact_sha256 or "")[:8] or None

    @property
    def generation(self) -> str:
        return str(self.manifest.version)

    @property
    def unbenchmarked(self) -> bool:
        return self.manifest.metrics.get(BENCHMARKED_METRIC_KEY) is not True

    def answer(self, vector: tuple[float, ...]) -> BackendAnswer:
        # harness-recommendation-models-01MSK2RM WP04: the served artifact is a
        # CalibratedClassifierCV, so ``p`` is the *calibrated* P(positive); the
        # ruled mapping (R4) is models.decision_and_confidence, which raises
        # rather than clamps on an out-of-range probability.
        from kenaz_ml.advice.models import decision_and_confidence, positive_probability

        decision, confidence = decision_and_confidence(positive_probability(self.model, vector))
        return BackendAnswer(decision=decision, confidence=confidence)


@dataclass
class LayaBackend:
    """An installed laya checkpoint, called **in-process** through the ``ONNXAgent`` wrapper.

    Never an HTTP call to this process's own ``/v1/systemone`` (spec C-002;
    ``tests/test_laya_dispatch_boundary.py`` scans this module for one). The
    refusal semantics are Amendment A3.2's: :meth:`preflight` refuses on an
    ineligible host (``host_ineligible``), :meth:`answer` refuses
    ``laya_backend_not_installed`` when the laya package itself is absent, and
    neither ever falls back to another backend -- the client does.

    ``confidence`` is ``round(100 * answer_confidence)``, left **unclamped** so
    :func:`dispatch` refuses an out-of-range value; the entropy ``confidence``
    field laya also reports is never read (``kenaz_ml.laya.agent.LayaAnswer``
    cannot carry it).
    """

    kind_id: str
    checkpoint: Any  # kenaz_ml.laya.agent.CheckpointRef
    names: tuple[str, ...]
    runtime: Any = None  # kenaz_ml.laya.agent.LayaRuntime; the process-wide one when None
    name: str = BACKEND_LAYA

    @property
    def manifest(self) -> Any:
        return self.checkpoint.manifest

    @property
    def model_label(self) -> str:
        return f"laya/{self.kind_id}@{self.manifest.version}"

    @property
    def rung(self) -> str:
        return str(self.manifest.metrics.get(RUNG_METRIC_KEY) or BACKEND_LAYA)

    @property
    def checkpoint_provenance(self) -> str:
        return self.checkpoint.provenance if self.checkpoint.provenance in ("local", "org", "base") else "local"

    @property
    def model_id_sha8(self) -> str | None:
        return self.checkpoint.sha8

    @property
    def generation(self) -> str:
        return str(self.manifest.version)

    @property
    def unbenchmarked(self) -> bool:
        return self.manifest.metrics.get(BENCHMARKED_METRIC_KEY) is not True

    def _runtime(self) -> Any:
        from kenaz_ml.laya.agent import get_runtime

        return self.runtime or get_runtime()

    def preflight(self) -> None:
        """Consult the per-host eligibility gate; raise :class:`Refused` if the host is ineligible."""
        from kenaz_ml.laya import eligibility

        verdict = eligibility.current_verdict()
        if not verdict.eligible:
            raise Refused(
                self.kind_id,
                REASON_HOST_INELIGIBLE,
                f"kind unavailable: host ineligible: {verdict.reason} ({verdict.detail})",
            )

    def standing_refusal(self) -> tuple[str, str] | None:
        """``(reason, detail)`` if the host is *already known* ineligible this boot; never triggers a measurement."""
        from kenaz_ml.laya import eligibility

        verdict = eligibility.cached_verdict()
        if verdict is not None and not verdict.eligible:
            return REASON_HOST_INELIGIBLE, f"kind unavailable: host ineligible: {verdict.reason} ({verdict.detail})"
        return None

    def answer(self, vector: tuple[float, ...]) -> BackendAnswer:
        from kenaz_ml.laya.agent import LayaNotInstalledError

        runtime = self._runtime()
        try:
            runtime.ensure_loaded(self.checkpoint)
        except LayaNotInstalledError as exc:
            logger.warning("dispatch: laya unavailable for %r: %s", self.kind_id, exc)
            raise Refused(self.kind_id, REASON_LAYA_NOT_INSTALLED, LAYA_NOT_INSTALLED_DETAIL) from exc
        laya_answer = runtime.predict(self.kind_id, self.names, vector)
        # round() of a non-finite value raises -> backend_error; an out-of-range
        # integer is refused by dispatch() as confidence_out_of_range. Never clamped.
        confidence = round(100 * laya_answer.answer_confidence)
        return BackendAnswer(decision=laya_answer.decision, score=laya_answer.score, confidence=confidence)


#: The complete set of backend *implementations* shipped in ``src/``. There is
#: deliberately no heuristic implementation here (A3.2). ``laya`` joined with
#: ``laya-serving-and-packs-01MSK2SP`` WP02.
SHIPPED_BACKENDS: dict[str, type] = {BACKEND_CLASSIC: ClassicBackend, BACKEND_LAYA: LayaBackend}


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchEntry:
    """One kind's dispatch state. ``server`` is ``None`` when the kind refuses."""

    kind_id: str
    contract: Any  # FeatureContract | None
    backend: str | None = None
    server: Backend | None = None
    manifest: Any = None
    reason: str | None = None
    detail: str | None = None
    #: A trained-but-unflipped model (rung R2): scored in shadow on every
    #: request for the kind, never authoritative (harness-recommendation-models WP04).
    shadow_model: Any = None

    @property
    def available(self) -> bool:
        return self.server is not None

    @property
    def supported_versions(self) -> tuple[str, ...]:
        """Accepted ``feature_contract_version`` values. N-1 is not defined yet."""
        version = getattr(self.contract, "service_version", None)
        return (version,) if version else ()


def not_served(kind_id: str, contract: Any, detail: str | None = None, **extra: Any) -> DispatchEntry:
    return DispatchEntry(
        kind_id=kind_id,
        contract=contract,
        reason=REASON_KIND_NOT_SERVED,
        detail=detail or f"kind unavailable: no backend serves {kind_id!r} (the client's heuristic answers)",
        **extra,
    )


class DispatchTable:
    """The authoritative kind -> entry mapping. Reads are snapshots; swaps are atomic."""

    def __init__(self, entries: Mapping[str, DispatchEntry] | None = None) -> None:
        self._lock = threading.Lock()
        self._snapshot: Mapping[str, DispatchEntry] = MappingProxyType({})
        self._latency: dict[str, deque[float]] = {}
        self._latency_lock = threading.Lock()
        for entry in (entries or {}).values():
            self.replace_entry(entry)

    def snapshot(self) -> Mapping[str, DispatchEntry]:
        """An immutable view of the whole table; take it once per request."""
        return self._snapshot

    def replace_entry(self, entry: DispatchEntry) -> None:
        """Install or replace one kind's entry atomically (the hot-reload/flip seam).

        Raises ``ValueError`` for a workbench model name (D-A3 namespace
        disjointness) or a backend outside :data:`BACKENDS`.
        """
        if entry.kind_id in WORKBENCH_MODEL_NAMES:
            raise ValueError(
                f"{entry.kind_id!r} is a workbench model name; /v1/recommend and /predict are namespace-disjoint"
            )
        if entry.backend is not None and entry.backend not in BACKENDS:
            raise ValueError(f"backend {entry.backend!r} is not one of {BACKENDS}")
        with self._lock:
            updated = dict(self._snapshot)
            updated[entry.kind_id] = entry
            self._snapshot = MappingProxyType(updated)

    def replace_all(self, entries: Mapping[str, DispatchEntry]) -> None:
        """Swap the whole table at once (startup population, full refresh)."""
        for entry in entries.values():
            if entry.kind_id in WORKBENCH_MODEL_NAMES:
                raise ValueError(f"{entry.kind_id!r} is a workbench model name")
        with self._lock:
            self._snapshot = MappingProxyType(dict(entries))

    # -- FR-025: per-kind serving latency ------------------------------------

    def record_latency(self, kind_id: str, elapsed_ms: float) -> None:
        with self._latency_lock:
            ring = self._latency.get(kind_id)
            if ring is None:
                ring = deque(maxlen=LATENCY_RING_SIZE)
                self._latency[kind_id] = ring
            ring.append(elapsed_ms)

    def latency_samples(self, kind_id: str) -> list[float]:
        with self._latency_lock:
            return list(self._latency.get(kind_id, ()))

    def latency_p95_ms(self, kind_id: str) -> float | None:
        """p95 of the last :data:`LATENCY_RING_SIZE` serving times for ``kind_id``, or ``None``."""
        samples = sorted(self.latency_samples(kind_id))
        if not samples:
            return None
        index = math.ceil(0.95 * len(samples)) - 1  # n >= 1, so index >= 0
        return samples[index]


# ---------------------------------------------------------------------------
# Startup population (D-A3)
# ---------------------------------------------------------------------------


def _manifest_backend(manifest: Any) -> str | None:
    value = (getattr(manifest, "metrics", None) or {}).get(BACKEND_METRIC_KEY)
    return str(value) if value is not None else None


def _entry_for(kind_id: str, local_dir: Path, base_dir: Path) -> DispatchEntry:
    """Resolve one kind through the registry and decide its entry. Never raises."""
    from kenaz_ml.modelstore.registry import REASON_SLOT_EMPTY, read_manifest, resolve_model

    contract = contract_for(kind_id)
    if contract is None:
        return DispatchEntry(
            kind_id=kind_id,
            contract=None,
            reason=REASON_NO_CONTRACT,
            detail=f"kind unavailable: {kind_id!r} has no published feature contract (advice/contracts.py)",
        )

    resolution = resolve_model(kind_id, local_dir=local_dir, base_dir=base_dir, expected_contract=contract)
    if resolution.served:
        manifest = resolution.manifest
        backend = _manifest_backend(manifest)
        if getattr(manifest, "artifact_kind", "file") == "directory" and backend != BACKEND_LAYA:
            # A directory artifact is a laya checkpoint (FR-011); its "model" is a path,
            # which no other backend can serve. Refuse rather than hand a Path to sklearn.
            return DispatchEntry(
                kind_id=kind_id,
                contract=contract,
                manifest=manifest,
                reason=REASON_BACKEND_UNKNOWN,
                detail=f"kind unavailable: manifest is a directory artifact (a laya checkpoint) but names "
                f"backend {backend!r}, not {BACKEND_LAYA!r}",
            )
        if backend == BACKEND_CLASSIC:
            server = ClassicBackend(model=resolution.model, kind_id=kind_id, manifest=manifest, slot=resolution.slot)
            return DispatchEntry(kind_id=kind_id, contract=contract, backend=backend, server=server, manifest=manifest)
        if backend == BACKEND_LAYA:
            return _laya_entry(kind_id, contract, manifest, local_dir, base_dir, resolution)
        if backend in (None, BACKEND_HEURISTIC):
            return not_served(
                kind_id,
                contract,
                f"kind unavailable: {kind_id!r} has a trained model (generation {manifest.version}) "
                "that has not graduated; the client's heuristic answers",
                manifest=manifest,
                shadow_model=resolution.model,
            )
        return _unknown_backend(kind_id, contract, backend, manifest)

    # No servable joblib pair. A laya checkpoint is not a joblib pair, so a
    # manifest naming laya refuses as laya, never as "not served".
    for slot_dir in (local_dir, base_dir):
        read = read_manifest(slot_dir / f"{kind_id}.json")
        if read.ok and read.manifest is not None:
            backend = _manifest_backend(read.manifest)
            if backend == BACKEND_LAYA:
                entry = _laya_entry(kind_id, contract, read.manifest, local_dir, base_dir)
                refused = [str(r) for r in resolution.refusals if r.reason != REASON_SLOT_EMPTY]
                if entry.server is None and refused and getattr(read.manifest, "artifact_kind", "file") == "directory":
                    # The checkpoint IS installed but the registry refused it (tampered, wrong contract...):
                    # say so, rather than the misleading "laya backend not installed".
                    return DispatchEntry(
                        kind_id=kind_id,
                        contract=contract,
                        backend=BACKEND_LAYA,
                        manifest=read.manifest,
                        reason=REASON_CHECKPOINT_REFUSED,
                        detail=f"kind unavailable: {kind_id!r} laya checkpoint refused by the registry — "
                        + "; ".join(refused),
                    )
                return entry
            if backend is not None and backend not in BACKENDS:
                return _unknown_backend(kind_id, contract, backend, read.manifest)

    refused = [str(r) for r in resolution.refusals if r.reason != REASON_SLOT_EMPTY]
    detail = None
    if refused:
        detail = f"kind unavailable: {kind_id!r} artifact refused by the registry — " + "; ".join(refused)
    return not_served(kind_id, contract, detail)


def _laya_refusal(kind_id: str, contract: Any, manifest: Any) -> DispatchEntry:
    _note_checkpoint(kind_id, None)
    return DispatchEntry(
        kind_id=kind_id,
        contract=contract,
        backend=BACKEND_LAYA,
        manifest=manifest,
        reason=REASON_LAYA_NOT_INSTALLED,
        detail=LAYA_NOT_INSTALLED_DETAIL,
    )


def _note_checkpoint(kind_id: str, identity: str | None) -> None:
    """Keep the eligibility gate's view of the installed-checkpoint set current (a change recomputes the verdict)."""
    try:
        from kenaz_ml.laya import eligibility

        if identity is None:
            eligibility.unregister_checkpoint(kind_id)
        else:
            eligibility.register_checkpoint(kind_id, identity)
    except Exception:  # pragma: no cover - bookkeeping must never break table population
        logger.debug("dispatch: could not update the eligibility checkpoint set", exc_info=True)


def _laya_entry(
    kind_id: str, contract: Any, manifest: Any, local_dir: Path, base_dir: Path, resolution: Any = None
) -> DispatchEntry:
    """A laya-configured kind: served by an in-process :class:`LayaBackend` only if a checkpoint is installed.

    With no verified checkpoint directory -- the day-one state of every install,
    whether or not the host is eligible (FR-007) -- the kind refuses
    ``laya_backend_not_installed``. Eligibility is a *request-time* gate
    (:meth:`LayaBackend.preflight`), because it is measured lazily at the first
    dispatch of a laya-configured kind (FR-005), not at table build.
    """
    try:
        from kenaz_ml.laya.agent import find_checkpoint, ref_from_resolution

        # A directory the registry already verified this pass is reused, not re-hashed.
        ref = ref_from_resolution(kind_id, resolution) if resolution is not None else None
        if ref is None:
            ref = find_checkpoint(kind_id, local_dir=local_dir, base_dir=base_dir, expected_contract=contract)
    except Exception:
        logger.warning("dispatch: laya checkpoint lookup failed for %r", kind_id, exc_info=True)
        ref = None
    if ref is None:
        return _laya_refusal(kind_id, contract, manifest)
    _note_checkpoint(kind_id, f"{ref.sha8}@{ref.provenance}")
    server = LayaBackend(kind_id=kind_id, checkpoint=ref, names=tuple(contract.names) if contract else ())
    return DispatchEntry(kind_id=kind_id, contract=contract, backend=BACKEND_LAYA, server=server, manifest=manifest)


def _unknown_backend(kind_id: str, contract: Any, backend: str, manifest: Any) -> DispatchEntry:
    return DispatchEntry(
        kind_id=kind_id,
        contract=contract,
        manifest=manifest,
        reason=REASON_BACKEND_UNKNOWN,
        detail=f"kind unavailable: manifest names backend {backend!r}, not one of {list(BACKENDS)}",
    )


def _declared_kinds(*slot_dirs: Path) -> list[str]:
    """Manifest names in the slots that declare a serving backend, excluding workbench names."""
    from kenaz_ml.modelstore.registry import read_manifest

    found: list[str] = []
    for slot_dir in slot_dirs:
        try:
            candidates = sorted(slot_dir.glob("*.json"))
        except OSError:
            continue
        for path in candidates:
            name = path.stem
            if name in WORKBENCH_MODEL_NAMES or name in found:
                continue
            read = read_manifest(path)
            if read.ok and read.manifest is not None and _manifest_backend(read.manifest) is not None:
                found.append(name)
    return found


def build_table(local_dir: Path | None = None, base_dir: Path | None = None) -> DispatchTable:
    """Populate the dispatch table from the seed kinds and the slots (D-A3). Never raises per kind."""
    from kenaz_ml import config

    assert not (set(KIND_IDS) & WORKBENCH_MODEL_NAMES), "seed kinds collide with workbench names"
    local = Path(local_dir) if local_dir is not None else config.models_dir()
    base = Path(base_dir) if base_dir is not None else config.base_models_dir()

    kinds = list(KIND_IDS)
    for extra in _declared_kinds(local, base):
        if extra not in kinds:
            kinds.append(extra)

    entries: dict[str, DispatchEntry] = {}
    for kind_id in kinds:
        try:
            entries[kind_id] = _entry_for(kind_id, local, base)
        except Exception as exc:  # a broken kind must never take the table down
            logger.warning("dispatch: could not resolve %r", kind_id, exc_info=True)
            entries[kind_id] = not_served(kind_id, contract_for(kind_id), f"kind unavailable: {exc}")
    for entry in entries.values():
        logger.info(
            "dispatch: kind=%s backend=%s available=%s reason=%s",
            entry.kind_id,
            entry.backend,
            entry.available,
            entry.reason,
        )
    table = DispatchTable()
    table.replace_all(entries)
    return table


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------


class Refused(Exception):
    """A typed refusal. The route renders it as :class:`RecommendRefusal`."""

    def __init__(self, kind_id: str, reason: str, detail: str) -> None:
        super().__init__(f"{kind_id}: {reason}: {detail}")
        self.kind_id = kind_id
        self.reason = reason
        self.detail = detail

    def body(self) -> dict[str, Any]:
        return RecommendRefusal(
            error=self.reason, refusal=RefusalBody(kind_id=self.kind_id, reason=self.reason, detail=self.detail)
        ).model_dump()


def _ordered_diagnostic(expected: tuple[str, ...], posted: set[str]) -> str:
    missing = [n for n in expected if n not in posted]
    unexpected = sorted(n for n in posted if n not in set(expected))
    parts = []
    if missing:
        parts.append(f"missing {missing}")
    if unexpected:
        parts.append(f"unexpected {unexpected}")
    return "; ".join(parts)


def _record_shadow(entry: DispatchEntry, kind_id: str, contract: Any, request: RecommendRequest) -> None:
    """The shadow write site (harness-recommendation-models WP04 T017, D-B3). Never raises.

    The kind is at the shadow rung: a trained model exists but has not been
    flipped, so the engine still refuses ``kind_not_served`` and the client's
    heuristic answers. The request's features are queued for scoring; scoring
    and the append happen off the request path (``shadow.enqueue_shadow``).
    A request whose features do not match the contract is simply not shadowed.
    """
    try:
        from kenaz_ml.advice.models import vector_for
        from kenaz_ml.advice.shadow import enqueue_shadow

        vector = vector_for(request.features, tuple(contract.names))
        enqueue_shadow(
            kind_id,
            vector,
            entry.shadow_model,
            entry.manifest,
            ts_ms=int(time.time() * 1000),
            session_id=request.session_id,
            # The exact shadow-join key (two-client-engine 9461214), when sent.
            features_hash=request.features_hash,
            ts=request.ts,
        )
    except Exception:
        logger.debug("dispatch: shadow record skipped for %r", kind_id, exc_info=True)


def dispatch(snapshot: Mapping[str, DispatchEntry], kind_id: str, request: RecommendRequest) -> RecommendResponse:
    """Serve one request against one table snapshot. Raises :class:`Refused` only."""
    entry = snapshot.get(kind_id)
    if entry is None:
        raise Refused(kind_id, REASON_UNKNOWN_KIND, f"unknown kind {kind_id!r}")
    if request.kind_id is not None and request.kind_id != kind_id:
        raise Refused(kind_id, REASON_KIND_ID_MISMATCH, f"body kind_id {request.kind_id!r} != path kind {kind_id!r}")

    contract = entry.contract
    if contract is not None and request.feature_contract_version not in entry.supported_versions:
        raise Refused(
            kind_id,
            REASON_CONTRACT_MISMATCH,
            f"kind unavailable: contract {request.feature_contract_version} != {contract.service_version} "
            f"(supported: {list(entry.supported_versions)})",
        )

    if entry.server is None:
        if entry.shadow_model is not None and contract is not None:
            _record_shadow(entry, kind_id, contract, request)
        raise Refused(kind_id, entry.reason or REASON_KIND_NOT_SERVED, entry.detail or "kind unavailable")

    names = tuple(contract.names)
    posted = set(request.features)
    if posted != set(names):
        raise Refused(kind_id, REASON_FEATURES_MISMATCH, _ordered_diagnostic(names, posted))
    vector = tuple(float(request.features[name]) for name in names)
    if not all(math.isfinite(v) for v in vector):
        raise Refused(kind_id, REASON_FEATURES_INVALID, "features must be finite numbers")

    server = entry.server
    preflight = getattr(server, "preflight", None)
    if preflight is not None:
        preflight()  # a backend that can refuse before answering (laya: host eligibility); raises Refused
    try:
        answer = server.answer(vector)
    except Refused:
        raise  # a typed refusal from the backend itself (laya: not installed) keeps its own reason
    except Exception as exc:
        logger.warning("dispatch: backend %s failed for %r", server.name, kind_id, exc_info=True)
        raise Refused(kind_id, REASON_BACKEND_ERROR, f"{type(exc).__name__}: {exc}") from exc

    confidence = answer.confidence
    # Validated, never clamped and never rounded into range.
    if isinstance(confidence, bool) or not isinstance(confidence, int) or not 0 <= confidence <= 100:
        raise Refused(
            kind_id,
            REASON_CONFIDENCE_OUT_OF_RANGE,
            f"backend {server.name} produced confidence {confidence!r}; must be an integer in 0-100",
        )
    if (answer.decision is None) == (answer.score is None):
        raise Refused(kind_id, REASON_BACKEND_ERROR, "backend must answer exactly one of decision or score")

    return RecommendResponse(
        decision=answer.decision,
        score=answer.score,
        confidence=confidence,
        kind_id=kind_id,
        feature_contract_version=contract.service_version,
        model=server.model_label,
        rung=str(server.rung),
        backend=server.name,  # type: ignore[arg-type]
        model_id_sha8=server.model_id_sha8,
        checkpoint_provenance=server.checkpoint_provenance,  # type: ignore[arg-type]
        generation=str(server.generation),
        unbenchmarked=bool(server.unbenchmarked),
    )


def contracts_payload(snapshot: Mapping[str, DispatchEntry]) -> ContractsResponse:
    """``GET /v1/contracts``: every registered kind, its ordered contract, and availability."""
    kinds: dict[str, ContractEntry] = {}
    for kind_id, entry in snapshot.items():
        contract = entry.contract
        available, reason, detail = entry.available, entry.reason, entry.detail
        standing = getattr(entry.server, "standing_refusal", None)
        refusal = standing() if callable(standing) else None
        if refusal is not None:  # a laya kind on a host already measured ineligible: honest, not "available"
            available, (reason, detail) = False, refusal
        kinds[kind_id] = ContractEntry(
            features=list(getattr(contract, "names", ())),
            dtypes=list(getattr(contract, "dtypes", ())),
            version=getattr(contract, "service_version", None) or "",
            supported_versions=list(entry.supported_versions),
            available=available,
            backend=entry.server.name if entry.server is not None else entry.backend,
            reason=reason,
            detail=detail,
        )
    return ContractsResponse(kinds=kinds)


def timed_dispatch(table: DispatchTable, kind_id: str, request: RecommendRequest) -> RecommendResponse:
    """:func:`dispatch` against one snapshot, recording serving latency (FR-025) either way.

    Latency is recorded only for kinds registered in that snapshot: the path
    segment is caller-controlled, and a ring per arbitrary unknown kind would
    make the "bounded" measurement unbounded in the number of rings.
    """
    snapshot = table.snapshot()
    started = time.perf_counter()
    try:
        return dispatch(snapshot, kind_id, request)
    finally:
        if kind_id in snapshot:
            table.record_latency(kind_id, (time.perf_counter() - started) * 1000)


# ---------------------------------------------------------------------------
# Hot reload, audit and the classic flip (harness-recommendation-models-01MSK2RM WP04)
# ---------------------------------------------------------------------------
#
# Mirrors ``AppState.reload_models_into_poller`` (app.py) for the advice kinds:
# after a retrain the new generation is resolved through the registry
# (integrity, ordered contract, runtime) and swapped into the table through
# :meth:`DispatchTable.replace_entry` — the atomic seam Mission A built. A
# request sees the whole old entry or the whole new one.
#
# Failed-retrain safety (FR-012): a retrain that declines or fails writes
# nothing, so there is nothing to reload; a new artifact the registry refuses
# is never swapped in — the in-memory entry keeps serving. What is *not*
# covered is an evaluation regression of a successfully-fitted generation
# (e.g. a much worse held-out ECE): no such gate exists; graduation and the
# flip-back (WP05) are the quality gates.

#: ``ml_events`` rows for the advice loop. ``kind`` is prefixed ``advice_`` so
#: none can be mistaken for the workbench scheduler's ``"retrain"`` row;
#: ``endpoint`` names the kind and generation (the table has no generation
#: column); ``routing`` is ``"local"`` like every local-mode row.
AUDIT_RETRAIN = "advice_retrain"
AUDIT_FLIP = "advice_flip"
AUDIT_DEMOTE = "advice_demote"
AUDIT_INSUFFICIENT_DATA = "advice_insufficient_data"

#: Manifest ``metrics`` keys the flip persists (read back by :func:`_entry_for`).
FLIPPED_AT_METRIC_KEY = "flipped_at_ms"
FLIPPED_GENERATION_METRIC_KEY = "flipped_generation"
DEMOTED_GENERATION_METRIC_KEY = "demoted_generation"
DEMOTED_AT_METRIC_KEY = "demoted_at_ms"
RUNG_LIVE = "R3"


def audit_event(store: Any, event: str, kind_id: str, generation: Any, latency_ms: int = 0) -> None:
    """One best-effort ``ml_events`` row (mirrors ``TrainingScheduler._log_retrain``). Never raises."""
    if store is None:
        return
    try:
        store.insert_ml_event(event, f"advice/{kind_id}@{generation}", "local", int(latency_ms))
        store.commit()
    except Exception:
        logger.warning("dispatch: failed to write the %s audit row for %r", event, kind_id)


@dataclass(frozen=True)
class ReloadResult:
    kind_id: str
    reloaded: bool
    generation: str | None = None
    reason: str | None = None


def reload_kind(
    table: DispatchTable,
    kind_id: str,
    *,
    expected_generation: str | None = None,
    local_dir: Path | None = None,
    base_dir: Path | None = None,
) -> ReloadResult:
    """Resolve ``kind_id`` afresh and swap it in, or leave the current entry serving. Never raises."""
    from kenaz_ml import config

    local = Path(local_dir) if local_dir is not None else config.models_dir()
    base = Path(base_dir) if base_dir is not None else config.base_models_dir()
    try:
        entry = _entry_for(kind_id, local, base)
    except Exception as exc:  # pragma: no cover - _entry_for never raises by contract
        logger.warning("dispatch: reload of %r failed; previous entry keeps serving", kind_id, exc_info=True)
        return ReloadResult(kind_id, False, reason=f"{type(exc).__name__}: {exc}")

    has_model = entry.server is not None or entry.shadow_model is not None
    generation = str(entry.manifest.version) if entry.manifest is not None else None
    if not has_model or (expected_generation is not None and generation != str(expected_generation)):
        logger.warning(
            "dispatch: reload of %r did not yield generation %s (%s); previous entry keeps serving",
            kind_id,
            expected_generation,
            entry.detail or entry.reason,
        )
        return ReloadResult(kind_id, False, generation, entry.detail or entry.reason or "artifact refused")
    table.replace_entry(entry)
    logger.info(
        "advice model reloaded into dispatch: kind=%s trained_generation=%s backend=%s",
        kind_id,
        generation,
        entry.server.name if entry.server is not None else "shadow",
    )
    return ReloadResult(kind_id, True, generation)


@dataclass(frozen=True)
class FlipOutcome:
    kind_id: str
    flipped: bool
    reason: str
    generation: str | None = None


def apply_flip(
    table: DispatchTable,
    kind_id: str,
    result: Any,
    *,
    now_ms: int,
    store: Any = None,
    local_dir: Path | None = None,
    base_dir: Path | None = None,
) -> FlipOutcome:
    """D-B1: flip ``kind_id`` to ``classic`` on an ``eligible`` verdict — a separable call.

    One sequence: write the verdict *plus* the persisted serving backend, rung
    and flip time into manifest ``metrics`` (pinned to the generation the
    verdict was computed for), then swap the dispatch entry. If the manifest
    write fails there is no flip. If the swap fails after the write, the
    manifest's serving keys are rolled back and the inconsistency is logged at
    ERROR — never "classic in the manifest, not served" silently. Refuses the
    generation recorded as ``demoted_generation`` (WP05's marker).
    """
    from kenaz_ml.advice.shadow import VERDICT_ELIGIBLE, update_manifest_metrics

    generation = result.generation
    if result.verdict != VERDICT_ELIGIBLE or generation is None:
        return FlipOutcome(kind_id, False, f"verdict {result.verdict}", generation)
    if result.promoted:
        return FlipOutcome(kind_id, False, "already served by classic", generation)
    manifest = _current_manifest(kind_id, local_dir)
    if manifest is None or str(manifest.version) != str(generation):
        return FlipOutcome(kind_id, False, "manifest changed since the verdict", generation)
    demoted = manifest.metrics.get(DEMOTED_GENERATION_METRIC_KEY)
    if demoted is not None and str(demoted) == str(generation):
        logger.info("dispatch: %r generation %s was demoted; it may not re-flip", kind_id, generation)
        return FlipOutcome(kind_id, False, "generation demoted; awaiting a new generation", generation)

    serving = {
        BACKEND_METRIC_KEY: BACKEND_CLASSIC,
        RUNG_METRIC_KEY: RUNG_LIVE,
        FLIPPED_AT_METRIC_KEY: int(now_ms),
        FLIPPED_GENERATION_METRIC_KEY: int(generation),
    }
    # A later generation's flip clears the demotion marker (WP05); the rollback
    # below restores it along with the serving keys.
    touched = (*serving, DEMOTED_GENERATION_METRIC_KEY, DEMOTED_AT_METRIC_KEY)
    previous = {k: manifest.metrics[k] for k in touched if k in manifest.metrics}
    written = update_manifest_metrics(
        kind_id,
        {**result.to_metrics(), **serving},
        models_dir=local_dir,
        expected_version=generation,
        remove=(DEMOTED_GENERATION_METRIC_KEY, DEMOTED_AT_METRIC_KEY),
    )
    if not written.ok:
        logger.warning("dispatch: flip of %r aborted — manifest write failed (%s)", kind_id, written.reason)
        return FlipOutcome(kind_id, False, f"manifest write failed: {written.reason}", generation)

    reloaded = reload_kind(table, kind_id, expected_generation=generation, local_dir=local_dir, base_dir=base_dir)
    entry = table.snapshot().get(kind_id)
    if not reloaded.reloaded or entry is None or entry.server is None:
        rollback = update_manifest_metrics(
            kind_id,
            {**previous, "verdict_reason": f"flip rolled back: {reloaded.reason}"},
            models_dir=local_dir,
            remove=[k for k in touched if k not in previous],
        )
        logger.error(
            "dispatch: flip of %r generation %s wrote the manifest but could not serve it (%s); manifest rollback %s",
            kind_id,
            generation,
            reloaded.reason,
            "succeeded" if rollback.ok else f"FAILED ({rollback.reason}) — manifest says classic, table does not",
        )
        return FlipOutcome(kind_id, False, f"swap failed: {reloaded.reason}", generation)

    audit_event(store, AUDIT_FLIP, kind_id, generation)
    logger.info(
        "dispatch: FLIPPED %r to classic at generation %s (shadow precision %s vs heuristic %s, delta %s pp, "
        "%d labels, ECE %s)",
        kind_id,
        generation,
        result.shadow_precision,
        result.heuristic_precision,
        result.precision_delta_pp,
        result.label_count,
        result.calibration_ece,
    )
    return FlipOutcome(kind_id, True, "eligible", generation)


def _current_manifest(kind_id: str, local_dir: Path | None) -> Any:
    from kenaz_ml.advice.training import previous_manifest

    return previous_manifest(kind_id, local_dir)


def after_retrain(
    table: DispatchTable,
    kind_id: str,
    outcome: Any,
    *,
    now_ms: int,
    store: Any = None,
    local_dir: Path | None = None,
    base_dir: Path | None = None,
    flip: bool = True,
) -> dict[str, Any]:
    """The retrain job's post-fit sequence: audit row, hot reload, graduation, verdict, flip.

    ``flip=False`` (or deleting the :func:`apply_flip` call) leaves the reload
    path untouched — the separability D-B1 asks for. Never raises.
    """
    from kenaz_ml.advice.shadow import VERDICT_ELIGIBLE, graduation_check, update_manifest_metrics
    from kenaz_ml.advice.training import kind_lock

    report: dict[str, Any] = {"kind": kind_id}
    generation = outcome.generation
    with kind_lock(kind_id):
        audit_event(store, AUDIT_RETRAIN, kind_id, generation, getattr(outcome, "duration_ms", 0))
        report["reload"] = reload_kind(
            table, kind_id, expected_generation=generation, local_dir=local_dir, base_dir=base_dir
        )
        result = graduation_check(
            kind_id,
            now_ms=now_ms,
            models_dir=local_dir,
            latency_p95_ms=table.latency_p95_ms(kind_id),
            latency_samples=len(table.latency_samples(kind_id)),
        )
        report["graduation"] = result
        if result.promoted:
            # Already served by classic: record the verdict, then let the one
            # flip-back evaluator decide (and act on) any demotion (WP05).
            update_manifest_metrics(kind_id, result.to_metrics(), models_dir=local_dir, expected_version=generation)
            report["flipback"] = evaluate_flipped_kind(
                table, kind_id, now_ms=now_ms, store=store, local_dir=local_dir, base_dir=base_dir
            )
            return report
        if flip and result.verdict == VERDICT_ELIGIBLE:
            report["flip"] = apply_flip(
                table, kind_id, result, now_ms=now_ms, store=store, local_dir=local_dir, base_dir=base_dir
            )
            if report["flip"].flipped:
                return report
        # No flip: the verdict and its numbers are still recorded.
        written = update_manifest_metrics(
            kind_id, result.to_metrics(), models_dir=local_dir, expected_version=generation
        )
        if not written.ok:
            logger.warning("dispatch: could not record %r's verdict (%s)", kind_id, written.reason)
    return report


def retrain_hook(table: DispatchTable, store: Any = None, clock: Any = None) -> Any:
    """Bind :func:`after_retrain` as ``AdviceTrainingScheduler(on_trained=...)``."""
    from kenaz_ml.advice.training import wall_clock_ms

    now = clock or wall_clock_ms

    def _hook(kind_id: str, outcome: Any) -> None:
        after_retrain(table, kind_id, outcome, now_ms=now(), store=store)

    return _hook


# ---------------------------------------------------------------------------
# Flip-back and demotion (harness-recommendation-models-01MSK2RM WP05, D-B5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemotionOutcome:
    kind_id: str
    demoted: bool
    reason: str
    generation: str | None = None


def apply_demotion(
    table: DispatchTable,
    kind_id: str,
    result: Any,
    *,
    now_ms: int,
    store: Any = None,
    local_dir: Path | None = None,
    base_dir: Path | None = None,
) -> DemotionOutcome:
    """FR-016: revert ``kind_id`` to the not-served (``heuristic``) state on a ``demoted`` flip-back verdict.

    Manifest first, pinned to the generation evaluated: the ``demoted``
    verdict with both windows' numbers, ``serving_backend = "heuristic"`` (so a
    restart agrees) and ``demoted_generation`` (so this generation can never
    re-flip — :func:`apply_flip` refuses it). Then the dispatch entry is
    swapped. If the manifest write fails nothing is reverted and nothing is
    claimed: the failure is logged at ERROR and the next tick retries — the
    table and the manifest never disagree silently.
    """
    from kenaz_ml.advice.contracts import contract_for
    from kenaz_ml.advice.shadow import FLIPBACK_DEMOTED, VERDICT_DEMOTED, update_manifest_metrics

    generation = result.generation
    if result.verdict != FLIPBACK_DEMOTED or generation is None:
        return DemotionOutcome(kind_id, False, f"flip-back verdict {result.verdict}", generation)
    written = update_manifest_metrics(
        kind_id,
        {
            **result.to_metrics(),
            "verdict": VERDICT_DEMOTED,
            "verdict_reason": result.reason,
            "verdict_at_ms": int(now_ms),
            BACKEND_METRIC_KEY: BACKEND_HEURISTIC,
            RUNG_METRIC_KEY: "R2",
            DEMOTED_GENERATION_METRIC_KEY: int(generation),
            DEMOTED_AT_METRIC_KEY: int(now_ms),
        },
        models_dir=local_dir,
        expected_version=generation,
        remove=(FLIPPED_AT_METRIC_KEY, FLIPPED_GENERATION_METRIC_KEY),
    )
    if not written.ok:
        logger.error(
            "dispatch: DEMOTION of %r generation %s NOT applied — manifest write failed (%s); "
            "classic keeps serving and the next evaluation retries",
            kind_id,
            generation,
            written.reason,
        )
        return DemotionOutcome(kind_id, False, f"manifest write failed: {written.reason}", generation)

    reloaded = reload_kind(table, kind_id, expected_generation=generation, local_dir=local_dir, base_dir=base_dir)
    entry = table.snapshot().get(kind_id)
    if not reloaded.reloaded or entry is None or entry.server is not None:
        # The manifest already says heuristic; the table must agree even if the
        # model cannot be re-read for shadow scoring.
        table.replace_entry(
            not_served(kind_id, contract_for(kind_id), f"kind unavailable: {kind_id!r} demoted ({result.reason})")
        )
    audit_event(store, AUDIT_DEMOTE, kind_id, generation)
    logger.warning(
        "dispatch: DEMOTED %r generation %s back to the client's heuristic: %s "
        "(flipped %s over %d shown decisions vs baseline %s over %d)",
        kind_id,
        generation,
        result.reason,
        result.flipped.rate,
        result.flipped.decided,
        result.baseline.rate,
        result.baseline.decided,
    )
    return DemotionOutcome(kind_id, True, result.reason, generation)


def evaluate_flipped_kind(
    table: DispatchTable,
    kind_id: str,
    *,
    now_ms: int,
    store: Any = None,
    local_dir: Path | None = None,
    base_dir: Path | None = None,
) -> Any:
    """Evaluate one flipped kind (FR-015..FR-017) and act on the verdict. Never raises.

    ``demoted`` reverts (:func:`apply_demotion`); ``holding``/``pending``
    record the measured numbers for the next comparison; ``insufficient_data``
    records itself, leaves the kind flipped, and emits an audit row and a
    WARNING the first time it is reached (not on every tick).
    """
    from kenaz_ml.advice.shadow import (
        FLIPBACK_DEMOTED,
        FLIPBACK_INSUFFICIENT_DATA,
        evaluate_post_flip,
        update_manifest_metrics,
    )
    from kenaz_ml.advice.training import kind_lock

    with kind_lock(kind_id):
        manifest = _current_manifest(kind_id, local_dir)
        if manifest is None or manifest.metrics.get(BACKEND_METRIC_KEY) != BACKEND_CLASSIC:
            return None
        result = evaluate_post_flip(
            kind_id,
            now_ms=now_ms,
            manifest=manifest,
            latency_p95_ms=table.latency_p95_ms(kind_id),
            latency_samples=len(table.latency_samples(kind_id)),
        )
        if result.verdict == FLIPBACK_DEMOTED:
            apply_demotion(table, kind_id, result, now_ms=now_ms, store=store, local_dir=local_dir, base_dir=base_dir)
            return result
        previous = manifest.metrics.get("flipback_verdict")
        written = update_manifest_metrics(
            kind_id, result.to_metrics(), models_dir=local_dir, expected_version=result.generation
        )
        if not written.ok:
            logger.warning("dispatch: could not record %r's flip-back numbers (%s)", kind_id, written.reason)
        if result.verdict == FLIPBACK_INSUFFICIENT_DATA and previous != FLIPBACK_INSUFFICIENT_DATA:
            audit_event(store, AUDIT_INSUFFICIENT_DATA, kind_id, result.generation)
            logger.warning(
                "dispatch: %r flip-back evaluation has insufficient data (%s); the kind stays flipped",
                kind_id,
                result.reason,
            )
        return result


def evaluate_flipped_kinds(
    table: DispatchTable,
    *,
    now_ms: int,
    store: Any = None,
    kinds: Any = None,
    local_dir: Path | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate every currently flipped kind (the scheduler-tick job)."""
    out: dict[str, Any] = {}
    for kind_id in kinds if kinds is not None else KIND_IDS:
        try:
            result = evaluate_flipped_kind(
                table, kind_id, now_ms=now_ms, store=store, local_dir=local_dir, base_dir=base_dir
            )
        except Exception:
            logger.exception("dispatch: flip-back evaluation of %r failed", kind_id)
            continue
        if result is not None:
            out[kind_id] = result
    return out


def evaluation_hook(table: DispatchTable, store: Any = None) -> Any:
    """Bind :func:`evaluate_flipped_kinds` as ``AdviceTrainingScheduler(on_tick=...)``."""

    def _tick(now_ms: int) -> None:
        evaluate_flipped_kinds(table, now_ms=now_ms, store=store)

    return _tick
