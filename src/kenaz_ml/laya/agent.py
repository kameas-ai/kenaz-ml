"""The in-process laya ``ONNXAgent`` wrapper (WP02).

What this module is, and is not
-------------------------------
It gives the dispatch table's ``laya`` backend an implementation that calls a
loaded agent **in-process** -- never over HTTP to this process's own
``/v1/systemone`` (spec C-002; ``tests/test_laya_dispatch_boundary.py`` proves
that structurally). It performs no network access and no signature verification
(C-005: verification is the client's Go code; here a checkpoint arrives already
verified by the registry's directory integrity check, WP04).

**Distribution blocker (read this before assuming laya works).** The ``laya``
package is *not* a declared dependency and is imported lazily, inside
:func:`default_agent_factory`. PyPI ``laya`` 0.3.22 declares ``torch`` and
``transformers`` as unconditional requirements and ``laya.onnx_agent`` imports
torch at module scope, so ``ONNXAgent`` cannot run without torch -- which spec
C-004 forbids in the base install and the frozen bundle. With ``laya`` absent,
:func:`default_agent_factory` raises :class:`LayaNotInstalledError` and the
dispatch layer refuses ``laya_backend_not_installed``: truthful, typed, never a
crash. Everything else here is written against laya 0.3.22's documented API
(``ONNXAgent(model_id_or_path, onnx_path=, expected_sha256=)``, ``system_one``,
``predict_batch``) and exercised only through fixture agents -- **it has never
run against the real package**.

Design rules (design doc R3/R8 via research.md 3.2)
----------------------------------------------------
* No laya Router: explicit **per-kind checkpoint selection** sits above
  ``laya.load()`` (all harness states are machine-rendered English, so language
  routing is structurally out of the path).
* ``max_loaded=1``: loading a second kind's checkpoint **unloads the first**.
  One laya kind is resident at a time; alternating kinds therefore reloads on
  every switch. That cost is accepted for the day-one footprint (see
  :class:`LayaRuntime`).
* The calibrated quantity is ``answer_confidence`` (max-p). The entropy-based
  ``confidence`` field is **never read** and :class:`LayaAnswer` has nowhere to
  carry it. The recommend confidence is ``round(100 * answer_confidence)``
  (computed by the dispatch backend), validated by the dispatcher into 0-100 and
  **never clamped** -- an out-of-range value refuses.
* ONNX Runtime threads are pinned deliberately (:func:`pinned_threads`): an
  unpinned CPU session measured ~12x slower (design doc R3).
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: The ONNX graph's file name inside a checkpoint directory (laya's own default).
ONNX_FILENAME = "laya.onnx"

#: Environment override for the pinned intra-op thread count.
THREADS_ENV = "KENAZ_ML_LAYA_THREADS"
#: Upper bound of the default pin. "Keep <= physical cores; oversubscribing the
#: logical/hyperthread count is a large regression" (laya-serve's own LAYA_THREADS
#: note), and a laptop should not spend every core on a background advisor.
DEFAULT_MAX_THREADS = 4

#: The one question id the recommend path asks; answers are read back by it.
DECISION_QID = "decision"


class LayaNotInstalledError(RuntimeError):
    """``laya`` (or a dependency it needs) is not importable in this build."""


class NoCheckpointError(RuntimeError):
    """``predict`` on an unloaded agent: there is nothing to predict with (never a hang)."""


class LayaAnswerInvalid(ValueError):
    """The agent answered in a shape this wrapper will not interpret."""


# ---------------------------------------------------------------------------
# Answers -- the return type deliberately cannot carry the entropy confidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayaAnswer:
    """One typed answer: exactly one of ``decision`` / ``score``, plus ``answer_confidence``.

    ``answer_confidence`` is laya's calibrated max-p in ``[0, 1]`` as reported --
    **not** clamped here. There is intentionally no ``confidence`` field: the
    entropy-based number laya also reports is a different quantity on a
    label-count-dependent scale and must never reach ``/v1/recommend``.
    """

    decision: bool | None = None
    score: int | None = None
    answer_confidence: float = 0.0


@dataclass(frozen=True)
class KindQuestion:
    """How one kind's decision is put to laya as a single typed question.

    ``answer_type`` is ``"noul"`` (a yes/no statement, answered with P(true)) or
    ``"score"`` (``levels`` ordered criteria). The instruction wording is an
    **assumption**: there is no checkpoint to evaluate any phrasing against, so
    these are placeholders a later mission tunes against real data.
    """

    answer_type: str
    instructions: str
    levels: tuple[str, ...] = ()

    def question(self) -> dict[str, Any]:
        if self.answer_type == "noul":
            return {"type": "noul", "instructions": self.instructions}
        if self.answer_type == "score":
            return {"type": "score", "instructions": self.instructions, "criteria": list(self.levels)}
        raise ValueError(f"unknown answer_type {self.answer_type!r}")


#: Per-kind question definitions. A kind with no entry gets :func:`default_question`.
KIND_QUESTIONS: dict[str, KindQuestion] = {
    "branch_now": KindQuestion("noul", "The state lists signals of an agent conversation. Should it branch now?"),
    "compact_now": KindQuestion("noul", "The state lists signals of an agent conversation. Should it compact now?"),
    "escalate_model": KindQuestion(
        "noul", "The state lists signals of an agent conversation. Should it escalate to a stronger model now?"
    ),
}


def default_question(kind_id: str) -> KindQuestion:
    return KIND_QUESTIONS.get(kind_id) or KindQuestion(
        "noul", f"The state lists signals. Should the action {kind_id!r} be taken now?"
    )


def render_state(names: tuple[str, ...], vector: tuple[float, ...]) -> str:
    """Machine-render a feature vector as English ``name: value`` lines (deterministic)."""
    if len(names) != len(vector):
        raise ValueError(f"{len(names)} feature names but {len(vector)} values")
    return "\n".join(f"{name}: {value:.6g}" for name, value in zip(names, vector))


def extract_answer(result: dict[str, Any], kind: KindQuestion, qid: str = DECISION_QID) -> LayaAnswer:
    """Pull the typed answer and ``answer_confidence`` out of a laya ``system_one`` result.

    Reads ``answer_confidence`` only. A result without it is **invalid** -- never
    silently replaced with the entropy ``confidence`` field.
    """
    try:
        answer = result["answers"][qid]
    except (KeyError, TypeError) as exc:
        raise LayaAnswerInvalid(f"laya result carries no answer for question {qid!r}") from exc
    if not isinstance(answer, dict):
        raise LayaAnswerInvalid("laya answer is not an object")
    raw = answer.get("answer_confidence")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        raise LayaAnswerInvalid(f"laya answer carries no usable answer_confidence (got {raw!r})")
    confidence = float(raw)
    if kind.answer_type == "noul":
        p_true = answer.get("noul")
        if isinstance(p_true, bool) or not isinstance(p_true, (int, float)) or not math.isfinite(float(p_true)):
            raise LayaAnswerInvalid(f"noul answer carries no usable probability (got {p_true!r})")
        return LayaAnswer(decision=float(p_true) >= 0.5, answer_confidence=confidence)
    if kind.answer_type == "score":
        value = answer.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise LayaAnswerInvalid(f"score answer carries no usable score (got {value!r})")
        return LayaAnswer(score=int(round(float(value))), answer_confidence=confidence)
    raise LayaAnswerInvalid(f"unknown answer_type {kind.answer_type!r}")


# ---------------------------------------------------------------------------
# ONNX Runtime thread pinning
# ---------------------------------------------------------------------------


def pinned_threads() -> int:
    """The intra-op thread count serving *and* the eligibility benchmark use.

    ``KENAZ_ML_LAYA_THREADS`` when it is a positive integer, otherwise half the
    logical CPUs (a proxy for physical cores) clamped to ``[1, 4]``. One constant
    for both paths: a benchmark measured under a different pinning than serving
    is a wrong number.
    """
    raw = os.environ.get(THREADS_ENV, "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return max(1, min(DEFAULT_MAX_THREADS, (os.cpu_count() or 2) // 2))


def session_options() -> Any:
    """``onnxruntime.SessionOptions`` with the deliberate thread pinning applied."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = pinned_threads()
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def pin_agent_threads(agent: Any, onnx_path: str | os.PathLike[str], expected_sha256: str | None = None) -> bool:
    """Rebuild a loaded ``ONNXAgent``'s session with pinned threads. True when applied.

    laya 0.3.22's ``ONNXAgent`` builds its own ``SessionOptions`` and exposes no
    thread parameter, so the only seam is replacing its public ``session``
    attribute. CPU provider only: an explicit provider list also keeps
    ``AzureExecutionProvider`` (present in the stock wheel) out of the process.
    Never raises; an agent with no ``session`` is left alone.

    The graph is read **once, as bytes**, and -- when the manifest declared the
    member's digest -- verified before the session is built from those same
    bytes. Rebuilding from the path would re-read a file laya already verified,
    reopening exactly the verify-then-load window ``expected_sha256`` exists to
    close. A mismatch leaves laya's own (verified) session serving; so does a
    graph whose weights live in an ONNX external-data file (not loadable from
    bytes) -- unpinned, never unverified.
    """
    if not hasattr(agent, "session"):
        return False
    try:
        import hashlib

        import onnxruntime as ort

        payload = Path(onnx_path).read_bytes()
        if expected_sha256 and hashlib.sha256(payload).hexdigest() != expected_sha256.lower():
            logger.warning("laya: %s changed after verification; not re-reading it to pin threads", onnx_path)
            return False
        agent.session = ort.InferenceSession(
            payload, sess_options=session_options(), providers=["CPUExecutionProvider"]
        )
        return True
    except Exception:
        logger.warning("laya: could not pin ONNX Runtime threads; serving on the agent's own session", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Checkpoints and the agent factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointRef:
    """A verified laya checkpoint directory and what the registry knows about it."""

    kind_id: str
    path: Path
    provenance: str  # "local" | "org" | "base"
    manifest: Any
    #: Per-member digests from the manifest, handed to laya's own ``expected_sha256``
    #: so laya re-verifies each file as it parses it (closes the verify-then-load window).
    expected_sha256: dict[str, str] | None = None

    @property
    def sha8(self) -> str | None:
        # A directory digest carries the canonical "sha256:" prefix (tree_digest); the id is the hex.
        digest = (getattr(self.manifest, "artifact_sha256", "") or "").removeprefix("sha256:")
        return digest[:8] or None


def ref_from_resolution(kind_id: str, resolution: Any) -> CheckpointRef | None:
    """A :class:`CheckpointRef` from a registry ``Resolution`` that served a *directory* artifact, else ``None``."""
    manifest = getattr(resolution, "manifest", None)
    if not getattr(resolution, "served", False) or manifest is None:
        return None
    if getattr(manifest, "artifact_kind", "file") != "directory":
        return None
    members = dict(getattr(manifest, "artifact_members", {}) or {})
    return CheckpointRef(
        kind_id=kind_id,
        path=Path(resolution.model),
        provenance=resolution.slot,
        manifest=manifest,
        expected_sha256=members or None,
    )


def find_checkpoint(
    kind_id: str,
    *,
    local_dir: Path | None = None,
    base_dir: Path | None = None,
    org_dir: Path | None = None,
    expected_contract: Any = None,
) -> CheckpointRef | None:
    """The lookup seam: kind -> verified checkpoint directory, or ``None``.

    Resolves through the registry's ladder (``local -> org -> base``) for a
    **directory** artifact (WP04, FR-011): ``{slot}/{kind}.json`` declaring
    ``artifact_kind: "directory"`` beside ``{slot}/{kind}.ckpt/``. Packs live in
    the client-owned install root and reach the slots already verified by the
    client's Go code; this engine re-verifies the directory digest and never
    fetches anything. ``org_dir`` stays ``None`` in production: there is no org
    location until a later mission delivers one (the org producer is unscheduled).
    With nothing installed -- every shipped build -- it returns ``None``.
    """
    from kenaz_ml.modelstore.registry import resolve_model

    resolution = resolve_model(
        kind_id,
        local_dir=local_dir,
        base_dir=base_dir,
        org_dir=org_dir,
        expected_contract=expected_contract,
    )
    return ref_from_resolution(kind_id, resolution)


class OnnxAgentLike(Protocol):
    def system_one(self, state: Any, questions: dict[str, Any], **kwargs: Any) -> dict[str, Any]: ...

    def predict_batch(self, states: list[Any], questions: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]: ...


AgentFactory = Callable[[CheckpointRef], OnnxAgentLike]


def _enforce_offline() -> None:
    """Make a model or tokenizer download impossible (C-005, C-007).

    ``ONNXAgent`` only calls ``snapshot_download`` for a path that does not
    exist locally; these switches make that path (and any tokenizer fetch) fail
    loudly instead of reaching a hub.
    """
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[var] = "1"


def default_agent_factory(ref: CheckpointRef) -> OnnxAgentLike:
    """Build laya's real ``ONNXAgent`` for ``ref`` -- or refuse truthfully if laya is absent."""
    _enforce_offline()
    try:
        from laya.onnx_agent import ONNXAgent
    except ImportError as exc:  # laya, torch or transformers not installed (the shipped state)
        raise LayaNotInstalledError(f"laya is not importable in this build: {exc}") from exc
    onnx_path = ref.path / ONNX_FILENAME
    agent = ONNXAgent(str(ref.path), onnx_path=str(onnx_path), expected_sha256=ref.expected_sha256)
    pin_agent_threads(agent, onnx_path, (ref.expected_sha256 or {}).get(ONNX_FILENAME))
    return agent


# ---------------------------------------------------------------------------
# The runtime: at most one loaded agent (max_loaded=1)
# ---------------------------------------------------------------------------


class LayaRuntime:
    """Holds at most one loaded agent and runs one forward pass at a time.

    ``max_loaded=1``: :meth:`ensure_loaded` for a different checkpoint unloads
    the resident one first. An *unloaded* runtime is a normal, constructible
    state -- day one of every install -- and :meth:`predict` on it raises
    :class:`NoCheckpointError`.
    """

    def __init__(self, agent_factory: AgentFactory | None = None) -> None:
        self._factory: AgentFactory = agent_factory or default_agent_factory
        self._lock = threading.RLock()
        self._agent: OnnxAgentLike | None = None
        self._ref: CheckpointRef | None = None

    # -- state ---------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._agent is not None

    @property
    def loaded_kind(self) -> str | None:
        return self._ref.kind_id if self._ref is not None else None

    def runtime_present(self) -> bool:
        """Whether an agent could be built at all -- answered **without importing laya** (or torch).

        With the real factory that means the ``laya`` package is findable; an
        injected factory (tests, a future runtime-pack loader) answers for itself.
        Lets ``/v1/contracts`` stop advertising a laya kind as available in a
        build where every request would refuse ``laya_backend_not_installed``.
        """
        if self._factory is not default_agent_factory:
            return True
        import importlib.util

        try:
            return importlib.util.find_spec("laya") is not None
        except ImportError, ValueError:
            return False

    # -- lifecycle -----------------------------------------------------------

    def load(self, ref: CheckpointRef) -> None:
        """Load ``ref``'s checkpoint, unloading any other first. Raises on failure (nothing stays half-loaded)."""
        with self._lock:
            if self._ref is not None and self._ref.path == ref.path and self._ref.kind_id == ref.kind_id:
                return
            self.unload()
            self._agent = self._factory(ref)
            self._ref = ref

    def ensure_loaded(self, ref: CheckpointRef) -> None:
        self.load(ref)

    def unload(self) -> None:
        with self._lock:
            self._agent = None
            self._ref = None

    # -- inference -----------------------------------------------------------

    def predict(self, kind_id: str, names: tuple[str, ...], vector: tuple[float, ...]) -> LayaAnswer:
        """One in-process forward pass for ``kind_id``'s rendered state."""
        with self._lock:
            agent = self._agent
            if agent is None or self.loaded_kind != kind_id:
                raise NoCheckpointError(f"no laya checkpoint is loaded for {kind_id!r}")
            question = default_question(kind_id)
            result = agent.system_one(render_state(names, vector), {DECISION_QID: question.question()})
            return extract_answer(result, question)

    # -- the raw /v1/systemone seam -------------------------------------------

    def systemone_backend(self) -> _SystemOneAdapter | None:
        """The loaded agent as a :class:`~kenaz_ml.laya.systemone_mount.SystemOneBackend`, or ``None``."""
        with self._lock:
            return _SystemOneAdapter(self, self._agent) if self._agent is not None else None


@dataclass
class _SystemOneAdapter:
    """Raw pass-through of laya's own result dicts for ``/v1/systemone`` (never used by recommend)."""

    runtime: LayaRuntime
    agent: OnnxAgentLike | None = field(repr=False, default=None)

    def predict(self, state: Any, questions: dict[str, Any], **controls: Any) -> dict[str, Any]:
        with self.runtime._lock:
            return self.agent.system_one(state, questions, **controls)  # type: ignore[union-attr]

    def predict_batch(self, states: list[Any], questions: dict[str, Any], **controls: Any) -> list[dict[str, Any]]:
        with self.runtime._lock:
            return self.agent.predict_batch(states, questions, **controls)  # type: ignore[union-attr]


_runtime: LayaRuntime | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> LayaRuntime:
    """The process-wide runtime (created unloaded, on first use)."""
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = LayaRuntime()
        return _runtime


def set_runtime(runtime: LayaRuntime | None) -> None:
    """Replace the process-wide runtime (tests; ``None`` resets to a fresh unloaded one)."""
    global _runtime
    with _runtime_lock:
        _runtime = runtime
