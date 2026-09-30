"""laya-serving-and-packs-01MSK2SP WP02 -- in-process laya dispatch and the structural boundary.

There is no real laya checkpoint (none ships, C-001) and the ``laya`` package is
not installable under C-004 (see ``kenaz_ml.laya.agent``), so every served case
drives a fixture agent. What is proven here is the *dispatch* contract: the
``laya`` backend is an in-process call, refusals are typed and truthful, the
confidence mapping is ``round(100 * answer_confidence)`` and never clamped, and
``/v1/recommend`` can never reach ``/v1/systemone`` -- checked structurally (an
AST scan that is itself shown able to fail) and behaviourally.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kenaz_ml.advice import dispatch as dispatch_mod
from kenaz_ml.advice.dispatch import (
    BACKEND_LAYA,
    SHIPPED_BACKENDS,
    DispatchEntry,
    DispatchTable,
    LayaBackend,
    RecommendRequest,
    Refused,
    dispatch,
)
from kenaz_ml.laya import agent as agent_mod
from kenaz_ml.laya import eligibility
from kenaz_ml.laya.agent import (
    CheckpointRef,
    LayaAnswerInvalid,
    LayaNotInstalledError,
    LayaRuntime,
    NoCheckpointError,
    default_agent_factory,
    extract_answer,
    pinned_threads,
    render_state,
    session_options,
)
from kenaz_ml.modelstore.registry import Manifest
from tests.fixtures.advice_backends import FIXTURE_FEATURES, fixture_contract, fixture_features

KIND = "fixture_laya"


# ---------------------------------------------------------------------------
# The structural boundary (C-002, SC-002)
# ---------------------------------------------------------------------------

# `subprocess` is deliberately not banned here: the eligibility gate reads macOS free memory
# through /usr/bin/vm_stat (a local, non-network read; psutil is not a dependency, WP03).
BANNED_MODULES = {"httpx", "requests", "urllib", "urllib3", "aiohttp", "http", "socket"}
BANNED_STRINGS = ("/v1/systemone", "7774")


def scan_for_http(source: str, filename: str = "<src>") -> list[str]:
    """Flag HTTP-client imports and references to the systemone route or this process's port.

    Docstrings are prose and are skipped; every other string constant counts.
    """
    tree = ast.parse(source, filename)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BANNED_MODULES:
                    findings.append(f"{filename}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in BANNED_MODULES:
                findings.append(f"{filename}:{node.lineno}: from {node.module} import ...")
            if node.module == "kenaz_ml.laya.systemone_mount":
                findings.append(f"{filename}:{node.lineno}: imports the systemone mount")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            if any(token in node.value for token in BANNED_STRINGS):
                findings.append(f"{filename}:{node.lineno}: string {node.value!r}")
    return findings


def _first_party_closure(start: list[Path]) -> list[Path]:
    """Files reachable from ``start`` through imports of kenaz_ml.advice.* / kenaz_ml.laya.* (never the mount)."""
    root = Path(dispatch_mod.__file__).resolve().parent.parent  # src/kenaz_ml
    seen: dict[Path, None] = {}
    queue = list(start)
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen[path] = None
        for node in ast.walk(ast.parse(path.read_text())):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
                modules += [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                modules += [a.name for a in node.names]
            for module in modules:
                if module.startswith(("kenaz_ml.advice", "kenaz_ml.laya")):
                    candidate = root.parent / Path(*module.split("."))
                    for option in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                        if option.is_file() and option.resolve() not in seen:
                            queue.append(option.resolve())
    return list(seen)


def test_scanner_flags_a_planted_http_call_to_systemone() -> None:
    planted = 'import httpx\n\ndef answer(v):\n    return httpx.post("http://127.0.0.1:7774/v1/systemone", json=v)\n'
    findings = scan_for_http(planted)
    assert any("import httpx" in f for f in findings)
    assert any("/v1/systemone" in f for f in findings)
    assert scan_for_http("import socket\n")
    assert scan_for_http("from urllib.request import urlopen\n")
    assert scan_for_http("URL = 'http://127.0.0.1:7774/health'\n")
    assert scan_for_http("from kenaz_ml.laya.systemone_mount import mount_systemone\n")
    # ...and prose that merely mentions the route is not a violation.
    assert scan_for_http('def f():\n    """never calls /v1/systemone"""\n') == []


def test_real_dispatch_and_agent_modules_pass_the_boundary() -> None:
    start = [Path(dispatch_mod.__file__).resolve(), Path(agent_mod.__file__).resolve()]
    files = _first_party_closure(start)
    names = {p.name for p in files}
    assert {"dispatch.py", "agent.py"} <= names
    assert "systemone_mount.py" not in names
    findings = [f for path in files for f in scan_for_http(path.read_text(), str(path.name))]
    assert findings == []


def test_no_permission_confirm_specific_code() -> None:
    """C-006: the generic ungraduated-checkpoint refusal covers it; nothing kind-specific was added."""
    for path in (Path(dispatch_mod.__file__), Path(agent_mod.__file__)):
        assert "permission_confirm" not in path.read_text()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeAgent:
    """A stand-in ``ONNXAgent``: records calls, answers in laya's native result shape."""

    def __init__(self, noul: float = 0.9, answer_confidence: Any = 0.83, entropy_confidence: float = 0.1) -> None:
        self.noul = noul
        self.answer_confidence = answer_confidence
        self.entropy_confidence = entropy_confidence
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.error: Exception | None = None

    def system_one(self, state: Any, questions: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if self.error is not None:
            raise self.error
        self.calls.append((state, questions))
        answer: dict[str, Any] = {"type": "noul", "noul": self.noul, "confidence": self.entropy_confidence}
        if self.answer_confidence is not None:
            answer["answer_confidence"] = self.answer_confidence
        return {"model": "fixture", "answers": {"decision": answer}, "usage": {"input_tokens": 1, "output_tokens": 0}}

    def predict_batch(self, states: list[Any], questions: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        return [self.system_one(s, questions) for s in states]


def _ref(kind_id: str = KIND, path: str = "/ckpt/a", provenance: str = "base") -> CheckpointRef:
    manifest = Manifest(name=kind_id, version="7", artifact_sha256="ab" * 32, metrics={"benchmarked": False})
    return CheckpointRef(kind_id=kind_id, path=Path(path), provenance=provenance, manifest=manifest)


def _runtime(agent: FakeAgent | None = None) -> tuple[LayaRuntime, FakeAgent, list[CheckpointRef]]:
    fake = agent or FakeAgent()
    built: list[CheckpointRef] = []

    def factory(ref: CheckpointRef) -> FakeAgent:
        built.append(ref)
        return fake

    return LayaRuntime(factory), fake, built


@pytest.fixture
def eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eligibility, "current_verdict", lambda key="": eligibility.Verdict(True, "eligible", "test"))


def _entry(runtime: LayaRuntime, ref: CheckpointRef | None = None) -> DispatchEntry:
    server = LayaBackend(kind_id=KIND, checkpoint=ref or _ref(), names=FIXTURE_FEATURES, runtime=runtime)
    return DispatchEntry(kind_id=KIND, contract=fixture_contract(KIND), backend=BACKEND_LAYA, server=server)


def _request() -> RecommendRequest:
    return RecommendRequest(
        features=fixture_features(), feature_contract_version=fixture_contract(KIND).service_version
    )


def _serve(entry: DispatchEntry) -> Any:
    return dispatch({KIND: entry}, KIND, _request())


# ---------------------------------------------------------------------------
# Serving through the in-process agent
# ---------------------------------------------------------------------------


def test_fixture_checkpoint_answers_in_process_with_truthful_envelope(eligible: None) -> None:
    runtime, fake, built = _runtime()
    response = _serve(_entry(runtime, _ref(provenance="org")))
    assert response.backend == "laya"
    assert response.decision is True  # noul 0.9 >= 0.5
    assert response.confidence == 83  # round(100 * answer_confidence)
    assert response.checkpoint_provenance == "org"
    assert response.model_id_sha8 == "abababab"
    assert response.generation == "7"
    assert response.model == f"laya/{KIND}@7"
    assert response.unbenchmarked is True
    assert len(built) == 1 and len(fake.calls) == 1
    state, questions = fake.calls[0]
    assert state == render_state(FIXTURE_FEATURES, (0.9, 0.1))  # machine-rendered English state
    assert questions["decision"]["type"] == "noul"


def test_confidence_is_answer_confidence_never_the_entropy_field(eligible: None) -> None:
    runtime, _, _ = _runtime(FakeAgent(answer_confidence=0.91, entropy_confidence=0.12))
    assert _serve(_entry(runtime)).confidence == 91
    # A result with no answer_confidence is invalid -- it is NOT replaced by `confidence`.
    runtime2, _, _ = _runtime(FakeAgent(answer_confidence=None, entropy_confidence=0.77))
    with pytest.raises(Refused) as refused:
        _serve(_entry(runtime2))
    assert refused.value.reason == "backend_error"
    assert "answer_confidence" in refused.value.detail


@pytest.mark.parametrize("bad", [1.2, -0.1])
def test_out_of_range_confidence_refuses_and_is_never_clamped(eligible: None, bad: float) -> None:
    runtime, _, _ = _runtime(FakeAgent(answer_confidence=bad))
    with pytest.raises(Refused) as refused:
        _serve(_entry(runtime))
    assert refused.value.reason == "confidence_out_of_range"


def test_non_finite_confidence_refuses(eligible: None) -> None:
    runtime, _, _ = _runtime(FakeAgent(answer_confidence=float("nan")))
    with pytest.raises(Refused) as refused:
        _serve(_entry(runtime))
    assert refused.value.reason == "backend_error"


def test_wrapper_return_type_cannot_carry_the_entropy_field() -> None:
    answer = extract_answer(
        {"answers": {"decision": {"noul": 0.2, "confidence": 0.5, "answer_confidence": 0.8}}},
        agent_mod.default_question("branch_now"),
    )
    assert answer.decision is False and answer.answer_confidence == 0.8
    assert set(answer.__dataclass_fields__) == {"decision", "score", "answer_confidence"}


def test_score_question_answers_an_integer_score() -> None:
    kind = agent_mod.KindQuestion("score", "rate it", ("low", "mid", "high"))
    answer = extract_answer({"answers": {"decision": {"score": 1.6, "answer_confidence": 0.7}}}, kind)
    assert answer.score == 2 and answer.decision is None
    with pytest.raises(LayaAnswerInvalid):
        extract_answer({"answers": {}}, kind)


# ---------------------------------------------------------------------------
# Refusals -- typed, truthful, never a crash, never a substitution
# ---------------------------------------------------------------------------


def test_ineligible_host_refuses_with_its_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        eligibility, "current_verdict", lambda key="": eligibility.Verdict(False, "insufficient_memory", "1.0 GiB free")
    )
    runtime, _, built = _runtime()
    with pytest.raises(Refused) as refused:
        _serve(_entry(runtime))
    assert refused.value.reason == "host_ineligible"
    assert refused.value.detail.startswith("kind unavailable: host ineligible: insufficient_memory")
    assert not built


def test_no_checkpoint_refuses_even_on_an_eligible_host(eligible: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_mod, "find_checkpoint", lambda *a, **k: None)
    entry = dispatch_mod._laya_entry(KIND, fixture_contract(KIND), None, Path("/nope"), Path("/nope"))
    assert not entry.available
    assert entry.reason == "laya_backend_not_installed"
    assert entry.detail == "kind unavailable: laya backend not installed"
    with pytest.raises(Refused) as refused:
        _serve(entry)
    assert refused.value.reason == "laya_backend_not_installed"


def test_installed_checkpoint_makes_the_entry_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_mod, "find_checkpoint", lambda *a, **k: _ref())
    entry = dispatch_mod._laya_entry(KIND, fixture_contract(KIND), None, Path("/x"), Path("/y"))
    assert entry.available and isinstance(entry.server, LayaBackend)


def test_laya_package_absent_refuses_truthfully(eligible: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped state: a checkpoint exists, eligibility passes, but laya cannot be imported."""
    monkeypatch.setitem(sys.modules, "laya", None)  # `import laya...` -> ImportError
    monkeypatch.setitem(sys.modules, "laya.onnx_agent", None)
    runtime = LayaRuntime()  # the real default factory
    with pytest.raises(Refused) as refused:
        _serve(_entry(runtime))
    assert refused.value.reason == "laya_backend_not_installed"
    assert not runtime.is_loaded


def test_default_factory_forces_offline_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setitem(sys.modules, "laya", None)
    monkeypatch.setitem(sys.modules, "laya.onnx_agent", None)
    with pytest.raises(LayaNotInstalledError):
        default_agent_factory(_ref())
    import os

    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_erroring_agent_is_a_typed_refusal_not_a_500(eligible: None) -> None:
    runtime, fake, _ = _runtime()
    fake.error = RuntimeError("onnx session blew up")
    table = DispatchTable()
    table.replace_entry(_entry(runtime))
    app = FastAPI()
    from kenaz_ml.app import AppState
    from kenaz_ml.routes import register_routes

    state = AppState()
    state.dispatch_table = table
    register_routes(app, state)
    response = TestClient(app).post(
        f"/v1/recommend/{KIND}",
        json={"features": fixture_features(), "feature_contract_version": fixture_contract(KIND).service_version},
    )
    assert response.status_code == 422
    assert response.json()["refusal"]["reason"] == "backend_error"


# ---------------------------------------------------------------------------
# The runtime: unloaded state and max_loaded=1
# ---------------------------------------------------------------------------


def test_unloaded_runtime_is_constructible_and_refuses_cleanly() -> None:
    runtime = LayaRuntime()
    assert not runtime.is_loaded and runtime.loaded_kind is None
    assert runtime.systemone_backend() is None
    with pytest.raises(NoCheckpointError):
        runtime.predict(KIND, FIXTURE_FEATURES, (1.0, 2.0))


def test_loading_a_second_checkpoint_unloads_the_first() -> None:
    runtime, _, built = _runtime()
    a, b = _ref("branch_now", "/ckpt/a"), _ref("compact_now", "/ckpt/b")
    runtime.load(a)
    runtime.load(a)  # idempotent for the resident checkpoint
    assert len(built) == 1 and runtime.loaded_kind == "branch_now"
    runtime.load(b)
    assert len(built) == 2 and runtime.loaded_kind == "compact_now"  # max_loaded=1: a is gone
    with pytest.raises(NoCheckpointError):
        runtime.predict("branch_now", FIXTURE_FEATURES, (1.0, 2.0))


def test_a_failed_load_leaves_nothing_half_loaded() -> None:
    def boom(ref: CheckpointRef) -> Any:
        raise LayaNotInstalledError("nope")

    runtime = LayaRuntime(boom)
    with pytest.raises(LayaNotInstalledError):
        runtime.load(_ref())
    assert not runtime.is_loaded


def test_systemone_adapter_passes_laya_results_through_unchanged() -> None:
    runtime, fake, _ = _runtime()
    runtime.load(_ref())
    backend = runtime.systemone_backend()
    assert backend is not None
    questions = {"q": {"type": "noul", "instructions": "x"}}
    assert backend.predict("state", questions) == fake.system_one("state", questions)
    assert len(backend.predict_batch(["a", "b"], questions)) == 2


def test_thread_pinning_is_deliberate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KENAZ_ML_LAYA_THREADS", "3")
    assert pinned_threads() == 3
    assert session_options().intra_op_num_threads == 3
    monkeypatch.delenv("KENAZ_ML_LAYA_THREADS")
    assert 1 <= pinned_threads() <= 4
    monkeypatch.setenv("KENAZ_ML_LAYA_THREADS", "zero")
    assert 1 <= pinned_threads() <= 4
    assert agent_mod.pin_agent_threads(object(), "/nonexistent.onnx") is False  # no session attribute: left alone


def test_laya_is_a_shipped_backend_implementation() -> None:
    assert {"classic", "laya"} == set(SHIPPED_BACKENDS)


# ---------------------------------------------------------------------------
# Behavioural boundary: no socket, never enters the systemone handler (T009 step 3)
# ---------------------------------------------------------------------------


def test_laya_backed_call_makes_no_socket(eligible: None) -> None:
    from tests.test_no_egress import no_network

    runtime, fake, _ = _runtime()
    entry = _entry(runtime)
    with no_network() as recorder:
        response = _serve(entry)
    assert response.backend == "laya" and len(fake.calls) == 1
    assert recorder.socket_attempts == []


def test_recommend_never_enters_the_systemone_handler(eligible: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from kenaz_ml.app import AppState
    from kenaz_ml.laya import systemone_mount
    from kenaz_ml.routes import register_routes

    entered: list[str] = []
    original = systemone_mount.SystemOneService._run

    async def spy(self: Any, fn: Any, what: str) -> Any:
        entered.append(what)
        return await original(self, fn, what)

    monkeypatch.setattr(systemone_mount.SystemOneService, "_run", spy)
    runtime, fake, _ = _runtime()
    table = DispatchTable()
    table.replace_entry(_entry(runtime))
    state = AppState()
    state.dispatch_table = table
    app = FastAPI()
    register_routes(app, state)
    systemone_mount.mount_systemone(app, runtime.systemone_backend)
    client = TestClient(app)
    body = {"features": fixture_features(), "feature_contract_version": fixture_contract(KIND).service_version}
    response = client.post(f"/v1/recommend/{KIND}", json=body)
    assert response.status_code == 200 and response.json()["backend"] == "laya"
    assert entered == []  # the raw route was never entered by the recommend path
    # ...and the raw route, when called directly, is entered and is laya's own shape.
    raw = client.post("/v1/systemone", json={"state": "s", "questions": {"q": {"type": "noul", "instructions": "i"}}})
    assert raw.status_code == 200 and "backend" not in raw.json()
    assert entered == ["systemone inference"]
    assert len(fake.calls) == 2
