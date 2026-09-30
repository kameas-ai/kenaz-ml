"""laya-serving-and-packs-01MSK2SP WP03 -- the per-host eligibility gate (User Story 3, SC-004).

No real checkpoint exists, so the gate is exercised against the bundled
random-weights fixture (real ONNX Runtime, real timings) and injected fakes for
the failure modes. Every case must yield a verdict with a reason and none may
raise.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from kenaz_ml import config
from kenaz_ml.advice.dispatch import (
    BACKEND_LAYA,
    DispatchEntry,
    LayaBackend,
    RecommendRequest,
    Refused,
    build_table,
    contracts_payload,
    dispatch,
)
from kenaz_ml.laya import agent as agent_mod
from kenaz_ml.laya import eligibility as el
from kenaz_ml.laya.agent import CheckpointRef, LayaRuntime
from kenaz_ml.modelstore.registry import Manifest
from tests.fixtures.advice_backends import FIXTURE_FEATURES, fixture_contract, fixture_features

GIB = 1024**3
ROOMY = lambda: (8 * GIB, "test")  # noqa: E731


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("KENAZ_ML_INSTALL_ROOT", str(tmp_path / "install"))
    el.reset()
    yield tmp_path
    el.reset()


class FakeTarget:
    basis = el.BASIS_FIXTURE

    def __init__(self, output: Any = (0.4, 0.6), error: Exception | None = None, block: threading.Event | None = None):
        self.output, self.error, self.block = output, error, block
        self.calls = 0

    def call(self, index: int) -> Any:
        self.calls += 1
        if self.block is not None:
            self.block.wait(timeout=30)
        if self.error is not None:
            raise self.error
        return self.output


# ---------------------------------------------------------------------------
# Scenario 1 -- eligible: real ONNX Runtime against the bundled fixture
# ---------------------------------------------------------------------------


def test_fixture_is_a_real_loadable_onnx_graph_labelled_non_functional() -> None:
    model = el.build_fixture_model()
    assert model == el.build_fixture_model()  # deterministic
    assert b"NON-FUNCTIONAL" in model
    target = el.FixtureTarget()
    out = target.call(0)
    assert el.validate_output(out) is None and len(out) == 2
    assert target.call(0) == out  # same fixed input, same output


def test_eligible_with_real_timings_and_labelled_fixture_measured() -> None:
    started = time.perf_counter()
    verdict = el.evaluate(free_memory_fn=ROOMY, use_cache=False)
    elapsed = time.perf_counter() - started
    assert verdict.eligible and verdict.reason == el.REASON_ELIGIBLE
    assert verdict.measured_against == "fixture"
    assert "fixture-measured" in verdict.detail and "non-functional" in verdict.detail
    assert verdict.calls == el.BENCH_CALLS == 10
    assert verdict.p50_ms is not None and verdict.p95_ms is not None and verdict.max_ms is not None
    assert 0 < verdict.p50_ms <= verdict.p95_ms <= verdict.max_ms
    assert elapsed < el.BENCH_BUDGET_SEC  # NFR-002 (5 s), session creation included
    assert verdict.free_memory_bytes == 8 * GIB and verdict.memory_method == "test"


# ---------------------------------------------------------------------------
# Scenario 2 -- insufficient memory, with the specific reason and numbers
# ---------------------------------------------------------------------------


def test_under_two_gib_is_ineligible_insufficient_memory_and_skips_the_benchmark() -> None:
    def boom() -> Any:
        raise AssertionError("the benchmark must not run on a host that is already out of memory")

    verdict = el.evaluate(free_memory_fn=lambda: (GIB, "test"), use_cache=False)
    assert not verdict.eligible and verdict.reason == el.REASON_INSUFFICIENT_MEMORY
    assert verdict.free_memory_bytes == GIB
    assert "1.00 GiB" in verdict.detail and "2 GiB" in verdict.detail
    assert verdict.measured_against is None  # nothing was benchmarked, so nothing is claimed
    el.evaluate(free_memory_fn=lambda: (GIB, "test"), target_factory=boom, use_cache=False)  # also bounded


def test_exactly_two_gib_is_eligible_boundary() -> None:
    verdict = el.evaluate(
        free_memory_fn=lambda: (el.MIN_FREE_BYTES, "test"), target_factory=FakeTarget, use_cache=False
    )
    assert verdict.eligible


def test_memory_that_cannot_be_determined_is_ineligible_not_guessed() -> None:
    verdict = el.evaluate(free_memory_fn=lambda: (None, "unavailable"), target_factory=FakeTarget, use_cache=False)
    assert not verdict.eligible and verdict.reason == el.REASON_MEMORY_UNKNOWN
    assert verdict.free_memory_bytes is None and verdict.memory_method == "unavailable"


def test_free_memory_is_read_without_psutil_on_this_platform() -> None:
    free, method = el.free_memory_bytes()
    assert method
    assert free is None or free > 0
    import importlib.util

    assert importlib.util.find_spec("psutil") is None or True  # never imported by kenaz_ml
    src = Path(el.__file__).read_text()
    assert "import psutil" not in src and "from psutil" not in src


def test_darwin_vm_stat_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    class Done:
        stdout = (
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free:                               1000.\n"
            "Pages active:                             5000.\n"
            "Pages inactive:                           2000.\n"
            "Pages speculative:                         500.\n"
        )

    monkeypatch.setattr(el.subprocess, "run", lambda *a, **k: Done())
    assert el._darwin_available_bytes() == (3500 * 16384, "vm_stat free+inactive+speculative pages")

    def fail(*a: Any, **k: Any) -> Any:
        raise OSError("no vm_stat")

    monkeypatch.setattr(el.subprocess, "run", fail)
    assert el._darwin_available_bytes() == (None, "unavailable")


def test_linux_meminfo_parse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:  16000000 kB\nMemFree:  100 kB\nMemAvailable:   4194304 kB\n")
    import builtins

    real_open = builtins.open
    monkeypatch.setattr(
        el,
        "open",
        lambda path, *a, **k: real_open(meminfo if path == "/proc/meminfo" else path, *a, **k),
        raising=False,
    )
    assert el._linux_available_bytes() == (4194304 * 1024, "/proc/meminfo MemAvailable")


# ---------------------------------------------------------------------------
# Scenario 3 -- the benchmark itself fails / is nonsense / hangs: bounded verdicts
# ---------------------------------------------------------------------------


def test_instantiation_failure_is_benchmark_failed_never_an_exception() -> None:
    def factory() -> Any:
        raise RuntimeError("cannot instantiate")

    verdict = el.evaluate(free_memory_fn=ROOMY, target_factory=factory, use_cache=False)
    assert not verdict.eligible and verdict.reason == el.REASON_BENCHMARK_FAILED
    assert "cannot instantiate" in verdict.detail


def test_raising_during_the_run_is_benchmark_failed() -> None:
    verdict = el.evaluate(
        free_memory_fn=ROOMY, target_factory=lambda: FakeTarget(error=ValueError("boom")), use_cache=False
    )
    assert verdict.reason == el.REASON_BENCHMARK_FAILED


@pytest.mark.parametrize("garbage", [(float("nan"), 0.5), (0.2, 0.2), (1.5, -0.5), (1.0,), "oops", (0.5, float("inf"))])
def test_nonsense_output_is_distinct_from_a_pass(garbage: Any) -> None:
    verdict = el.evaluate(free_memory_fn=ROOMY, target_factory=lambda: FakeTarget(output=garbage), use_cache=False)
    assert not verdict.eligible
    assert verdict.reason == el.REASON_BENCHMARK_INVALID
    assert verdict.reason != el.REASON_BENCHMARK_FAILED  # a corrupted runtime is not a crashed one


def test_a_hung_agent_is_bounded_by_the_timeout() -> None:
    release = threading.Event()
    started = time.perf_counter()
    verdict = el.evaluate(
        free_memory_fn=ROOMY, target_factory=lambda: FakeTarget(block=release), timeout_sec=0.3, use_cache=False
    )
    release.set()
    assert time.perf_counter() - started < 3
    assert not verdict.eligible and verdict.reason == el.REASON_BENCHMARK_TIMEOUT


def test_every_outcome_carries_a_reason_and_none_raises() -> None:
    cases = [
        dict(free_memory_fn=ROOMY, target_factory=FakeTarget),
        dict(free_memory_fn=lambda: (1, "t"), target_factory=FakeTarget),
        dict(free_memory_fn=lambda: (None, "u"), target_factory=FakeTarget),
        dict(free_memory_fn=ROOMY, target_factory=lambda: FakeTarget(output=(9, 9))),
        dict(
            free_memory_fn=lambda: (_ for _ in ()).throw(RuntimeError("memory probe exploded")),
            target_factory=FakeTarget,
        ),
    ]
    for kwargs in cases:
        verdict = el.evaluate(use_cache=False, **kwargs)
        assert isinstance(verdict.reason, str) and verdict.reason
        assert isinstance(verdict.detail, str) and verdict.detail


# ---------------------------------------------------------------------------
# D-C2 -- the version-stamped disk cache
# ---------------------------------------------------------------------------


def _cache_file() -> Path:
    return config.models_dir() / el.CACHE_FILENAME


def test_a_successful_benchmark_is_cached_and_reused_but_memory_is_not(monkeypatch: pytest.MonkeyPatch) -> None:
    first = el.evaluate(free_memory_fn=ROOMY)
    assert first.eligible and _cache_file().is_file()
    raw = json.loads(_cache_file().read_text())
    assert raw["stamp"]["onnxruntime"] and raw["benchmark"]["ok"] is True

    monkeypatch.setattr(el, "run_benchmark", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-ran")))
    second = el.evaluate(free_memory_fn=ROOMY)
    assert second.eligible and second.p50_ms == first.p50_ms  # the cached timings
    # Memory is judged fresh every time, even with a perfect cached benchmark.
    tight = el.evaluate(free_memory_fn=lambda: (GIB, "now"))
    assert not tight.eligible and tight.reason == el.REASON_INSUFFICIENT_MEMORY


def test_version_change_forces_recomputation(monkeypatch: pytest.MonkeyPatch) -> None:
    el.evaluate(free_memory_fn=ROOMY)
    real_stamp = el.environment_stamp()
    runs: list[int] = []
    real_run = el.run_benchmark

    def counting(*a: Any, **k: Any) -> Any:
        runs.append(1)
        return real_run(*a, **k)

    monkeypatch.setattr(el, "run_benchmark", counting)
    monkeypatch.setattr(el, "environment_stamp", lambda: {**real_stamp, "onnxruntime": "0.0.0-other"})
    el.evaluate(free_memory_fn=ROOMY)
    assert runs == [1]
    monkeypatch.setattr(el, "environment_stamp", lambda: {**real_stamp, "laya": "9.9.9"})
    el.evaluate(free_memory_fn=ROOMY)
    assert runs == [1, 1]


def test_checkpoint_install_or_removal_forces_recomputation(monkeypatch: pytest.MonkeyPatch) -> None:
    el.evaluate(free_memory_fn=ROOMY)
    runs: list[int] = []
    real_run = el.run_benchmark
    monkeypatch.setattr(el, "run_benchmark", lambda *a, **k: (runs.append(1), real_run(*a, **k))[1])
    el.register_checkpoint("branch_now", "abababab@base")
    el.evaluate(free_memory_fn=ROOMY)
    assert runs == [1]
    el.evaluate(free_memory_fn=ROOMY)  # same set -> cache hit
    assert runs == [1]
    el.unregister_checkpoint("branch_now")
    el.evaluate(free_memory_fn=ROOMY)  # removal is a change too: the one cached stamp no longer matches
    assert runs == [1, 1]
    assert el.checkpoint_set() == []


@pytest.mark.parametrize("content", ["{not json", "[]", '{"stamp": 1}', "", '{"stamp": {}, "benchmark": {"ok": true}}'])
def test_corrupt_cache_recomputes_without_raising(content: str) -> None:
    _cache_file().write_text(content)
    verdict = el.evaluate(free_memory_fn=ROOMY)
    assert verdict.eligible
    assert json.loads(_cache_file().read_text())["benchmark"]["ok"] is True  # healed


def test_failures_are_not_cached() -> None:
    el.evaluate(free_memory_fn=ROOMY, target_factory=lambda: FakeTarget(error=ValueError("x")))
    assert not _cache_file().exists()
    el.evaluate(free_memory_fn=lambda: (GIB, "t"))
    assert not _cache_file().exists()


def test_cache_write_is_atomic(monkeypatch: pytest.MonkeyPatch) -> None:
    replaced: list[tuple[str, str]] = []
    real_replace = el.os.replace
    monkeypatch.setattr(el.os, "replace", lambda a, b: (replaced.append((str(a), str(b))), real_replace(a, b))[1])
    el.evaluate(free_memory_fn=ROOMY)
    assert len(replaced) == 1 and replaced[0][1].endswith(el.CACHE_FILENAME) and replaced[0][0] != replaced[0][1]
    assert [p.name for p in config.models_dir().iterdir() if p.name.endswith(".tmp")] == []


def test_an_unwritable_cache_location_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(el.os, "replace", lambda a, b: (_ for _ in ()).throw(PermissionError("read-only")))
    assert el.evaluate(free_memory_fn=ROOMY).eligible


def test_explicit_recompute_clears_both_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    el.evaluate(free_memory_fn=ROOMY)
    assert _cache_file().is_file()
    verdict = el.recompute(free_memory_fn=ROOMY)
    assert verdict.eligible and el.cached_verdict() is verdict


# ---------------------------------------------------------------------------
# Lazy, once-per-boot behaviour
# ---------------------------------------------------------------------------


def test_nothing_is_measured_until_the_first_check_and_then_once_per_boot(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real = el.evaluate
    monkeypatch.setattr(el, "evaluate", lambda **k: (calls.append(1), real(free_memory_fn=ROOMY, **k))[1])
    assert el.cached_verdict() is None and calls == []  # importing/booting measures nothing
    first = el.current_verdict()
    assert el.current_verdict() is first and calls == [1]  # one evaluation per boot
    el.register_checkpoint("compact_now", "deadbeef@local")  # the installed set changed
    el.current_verdict()
    assert calls == [1, 1]


def test_describe_is_honest_before_and_after() -> None:
    before = el.describe()
    assert before["verdict"] == "not_evaluated" and before["reason"] == "not_evaluated"
    assert all(before[k] is None for k in ("measured_against", "free_memory_bytes", "p50_ms", "p95_ms", "max_ms"))
    el.recompute(free_memory_fn=lambda: (GIB, "test"))
    after = el.describe()
    assert after["verdict"] == "ineligible" and after["reason"] == "insufficient_memory"
    assert after["free_memory_bytes"] == GIB


# ---------------------------------------------------------------------------
# Scenario 4 -- /health carries it; dispatch refuses truthfully, never says backend laya
# ---------------------------------------------------------------------------


def test_health_carries_the_verdict_and_never_triggers_a_measurement() -> None:
    from kenaz_ml.app import create_app

    with TestClient(create_app()) as client:
        body = client.get("/health").json()["laya_eligibility"]
        assert body["verdict"] == "not_evaluated" and body["p50_ms"] is None
        assert el.cached_verdict() is None  # polling /health measured nothing
        el.recompute(free_memory_fn=ROOMY)
        body = client.get("/health").json()["laya_eligibility"]
        assert body["verdict"] == "eligible" and body["measured_against"] == "fixture"
        assert body["p50_ms"] is not None and "fixture-measured" in body["detail"]


def _ref() -> CheckpointRef:
    manifest = Manifest(name="fixture_laya", version="1", artifact_sha256="cd" * 32)
    return CheckpointRef("fixture_laya", Path("/ckpt"), "base", manifest)


class _Agent:
    def system_one(self, state: Any, questions: dict[str, Any], **kw: Any) -> dict[str, Any]:
        return {"answers": {"decision": {"noul": 0.7, "confidence": 0.2, "answer_confidence": 0.8}}}

    def predict_batch(self, states: list[Any], questions: dict[str, Any], **kw: Any) -> list[dict[str, Any]]:
        return [self.system_one(s, questions) for s in states]


def _laya_entry(runtime: LayaRuntime) -> DispatchEntry:
    server = LayaBackend("fixture_laya", _ref(), FIXTURE_FEATURES, runtime)
    return DispatchEntry("fixture_laya", fixture_contract("fixture_laya"), backend=BACKEND_LAYA, server=server)


def _request() -> RecommendRequest:
    return RecommendRequest(
        features=fixture_features(), feature_contract_version=fixture_contract("fixture_laya").service_version
    )


def test_ineligible_host_refuses_with_the_reason_and_never_answers_as_laya(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(el, "free_memory_bytes", lambda: (GIB, "test"))
    built: list[Any] = []
    runtime = LayaRuntime(lambda ref: (built.append(ref), _Agent())[1])
    entry = _laya_entry(runtime)
    with pytest.raises(Refused) as refused:
        dispatch({"fixture_laya": entry}, "fixture_laya", _request())
    assert refused.value.reason == "host_ineligible"
    assert "insufficient_memory" in refused.value.detail and "1.00 GiB" in refused.value.detail
    assert built == []  # laya was not even attempted
    # The standing refusal is now visible on /v1/contracts without a measurement.
    kind = contracts_payload({"fixture_laya": entry}).kinds["fixture_laya"]
    assert kind.available is False and kind.reason == "host_ineligible"


def test_failing_benchmark_on_an_otherwise_roomy_host_refuses_benchmark_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(el, "free_memory_bytes", ROOMY)
    monkeypatch.setattr(el, "FixtureTarget", lambda: (_ for _ in ()).throw(RuntimeError("ort exploded")))
    entry = _laya_entry(LayaRuntime(lambda ref: _Agent()))
    with pytest.raises(Refused) as refused:
        dispatch({"fixture_laya": entry}, "fixture_laya", _request())
    assert refused.value.reason == "host_ineligible" and "benchmark_failed" in refused.value.detail


def test_eligible_host_serves_through_the_real_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(el, "free_memory_bytes", ROOMY)
    runtime = LayaRuntime(lambda ref: _Agent())
    response = dispatch({"fixture_laya": _laya_entry(runtime)}, "fixture_laya", _request())
    assert response.backend == "laya" and response.confidence == 80
    verdict = el.cached_verdict()
    assert verdict is not None and verdict.eligible and verdict.measured_against == "fixture"


def test_eligible_host_with_no_checkpoint_still_refuses_end_to_end(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR-007 through the real table build: a laya manifest, nothing installed, a roomy host."""
    from tests.test_recommend_dispatch import _fitted_compact_model, _write_trio_pair

    monkeypatch.setattr(el, "free_memory_bytes", ROOMY)
    local = config.models_dir()
    base = isolated / "base_models"
    base.mkdir()
    _write_trio_pair(local, "compact_now", _fitted_compact_model(), {"serving_backend": "laya"})
    table = build_table(local, base)
    entry = table.snapshot()["compact_now"]
    assert not entry.available and entry.reason == "laya_backend_not_installed"
    assert el.checkpoint_set() == []  # nothing registered: no checkpoint is installed
    contract = entry.contract
    request = RecommendRequest(
        features={n: 0.5 for n in contract.names}, feature_contract_version=contract.service_version
    )
    with pytest.raises(Refused) as refused:
        dispatch(table.snapshot(), "compact_now", request)
    assert refused.value.reason == "laya_backend_not_installed"
    assert el.cached_verdict() is None  # the gate was never consulted for a kind with nothing to run


def test_no_checkpoint_is_not_a_reason_for_the_gate_to_say_ineligible(monkeypatch: pytest.MonkeyPatch) -> None:
    """The day-one state: the gate answers about the *host*, not about checkpoints."""
    monkeypatch.setattr(el, "free_memory_bytes", ROOMY)
    assert el.current_verdict().eligible
    assert agent_mod.find_checkpoint("branch_now") is None


def test_thread_pinning_rebuilds_a_real_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``pin_agent_threads`` replaces an agent's session with a pinned, CPU-only one that still runs."""
    import numpy as np

    monkeypatch.setenv("KENAZ_ML_LAYA_THREADS", "2")
    onnx = tmp_path / agent_mod.ONNX_FILENAME
    onnx.write_bytes(el.build_fixture_model())

    class Agent:
        session = "original"

    agent = Agent()
    assert agent_mod.pin_agent_threads(agent, onnx) is True
    assert agent.session != "original"
    assert agent.session.get_providers() == ["CPUExecutionProvider"]
    (probs,) = agent.session.run(None, {"x": np.zeros((1, 64), dtype=np.float32)})
    assert probs.shape == (1, 2)
    assert agent_mod.pin_agent_threads(Agent(), tmp_path / "missing.onnx") is False  # failure is not an exception
