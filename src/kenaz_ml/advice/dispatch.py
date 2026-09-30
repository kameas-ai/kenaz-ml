"""The ``/v1/recommend/{kind}`` dispatch layer (two-client-engine-01MSK2EN WP03).

One in-process **dispatch table** maps each recommendation kind to the single
backend that serves it — ``classic`` (a registry-validated scikit-learn
artifact) or ``laya`` (dormant: always refuses in this mission) — or to *no*
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
* ``"laya"`` — refuses ``laya_backend_not_installed`` (no laya runtime ships
  in this mission; never substituted).
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
        proba = self.model.predict_proba([list(vector)])[0]
        classes = list(getattr(self.model, "classes_", range(len(proba))))
        positive = classes.index(1) if 1 in classes else len(proba) - 1
        p = float(proba[positive])
        decision = p >= 0.5
        return BackendAnswer(decision=decision, confidence=int(round(100 * (p if decision else 1 - p))))


#: The complete set of backend *implementations* shipped in ``src/``. There is
#: deliberately no heuristic implementation here (A3.2); ``laya`` has none until
#: ``laya-serving-and-packs-01MSK2SP`` lands one.
SHIPPED_BACKENDS: dict[str, type] = {BACKEND_CLASSIC: ClassicBackend}


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
        if backend == BACKEND_CLASSIC:
            server = ClassicBackend(model=resolution.model, kind_id=kind_id, manifest=manifest, slot=resolution.slot)
            return DispatchEntry(kind_id=kind_id, contract=contract, backend=backend, server=server, manifest=manifest)
        if backend == BACKEND_LAYA:
            return _laya_refusal(kind_id, contract, manifest)
        if backend in (None, BACKEND_HEURISTIC):
            return not_served(
                kind_id,
                contract,
                f"kind unavailable: {kind_id!r} has a trained model (generation {manifest.version}) "
                "that has not graduated; the client's heuristic answers",
                manifest=manifest,
            )
        return _unknown_backend(kind_id, contract, backend, manifest)

    # No servable joblib pair. A laya checkpoint is not a joblib pair, so a
    # manifest naming laya refuses as laya, never as "not served".
    for slot_dir in (local_dir, base_dir):
        read = read_manifest(slot_dir / f"{kind_id}.json")
        if read.ok and read.manifest is not None:
            backend = _manifest_backend(read.manifest)
            if backend == BACKEND_LAYA:
                return _laya_refusal(kind_id, contract, read.manifest)
            if backend is not None and backend not in BACKENDS:
                return _unknown_backend(kind_id, contract, backend, read.manifest)

    refused = [str(r) for r in resolution.refusals if r.reason != REASON_SLOT_EMPTY]
    detail = None
    if refused:
        detail = f"kind unavailable: {kind_id!r} artifact refused by the registry — " + "; ".join(refused)
    return not_served(kind_id, contract, detail)


def _laya_refusal(kind_id: str, contract: Any, manifest: Any) -> DispatchEntry:
    return DispatchEntry(
        kind_id=kind_id,
        contract=contract,
        backend=BACKEND_LAYA,
        manifest=manifest,
        reason=REASON_LAYA_NOT_INSTALLED,
        detail="kind unavailable: laya backend not installed",
    )


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
        raise Refused(kind_id, entry.reason or REASON_KIND_NOT_SERVED, entry.detail or "kind unavailable")

    names = tuple(contract.names)
    posted = set(request.features)
    if posted != set(names):
        raise Refused(kind_id, REASON_FEATURES_MISMATCH, _ordered_diagnostic(names, posted))
    vector = tuple(float(request.features[name]) for name in names)
    if not all(math.isfinite(v) for v in vector):
        raise Refused(kind_id, REASON_FEATURES_INVALID, "features must be finite numbers")

    server = entry.server
    try:
        answer = server.answer(vector)
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
        kinds[kind_id] = ContractEntry(
            features=list(getattr(contract, "names", ())),
            dtypes=list(getattr(contract, "dtypes", ())),
            version=getattr(contract, "service_version", None) or "",
            supported_versions=list(entry.supported_versions),
            available=entry.available,
            backend=entry.server.name if entry.server is not None else entry.backend,
            reason=entry.reason,
            detail=entry.detail,
        )
    return ContractsResponse(kinds=kinds)


def timed_dispatch(table: DispatchTable, kind_id: str, request: RecommendRequest) -> RecommendResponse:
    """:func:`dispatch` against one snapshot, recording serving latency (FR-025) either way."""
    started = time.perf_counter()
    try:
        return dispatch(table.snapshot(), kind_id, request)
    finally:
        table.record_latency(kind_id, (time.perf_counter() - started) * 1000)
