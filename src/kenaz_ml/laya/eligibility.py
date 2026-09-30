"""Per-host eligibility gate for laya-backed serving (WP03, spec FR-005/FR-006, User Story 3).

A machine is *eligible* to attempt laya-bearing kinds only if (a) it has at
least :data:`MIN_FREE_BYTES` of available memory and (b) a :data:`BENCH_CALLS`-call
ONNX Runtime micro-benchmark runs cleanly, produces valid output, and finishes
inside its timeout. The result is always a :class:`Verdict` -- an explicit
``reason`` code, a human ``detail`` and the measurements behind it -- never a
bare boolean, and **nothing here raises to a caller**: every failure is itself a
verdict.

What an ineligible verdict does to serving (ruled: Amendment A3.2)
------------------------------------------------------------------
Laya is **not attempted**. Every laya-bearing kind **refuses** with the typed
``host_ineligible`` reason and the specific cause (``host ineligible:
insufficient_memory ...``); the engine never falls back to another backend --
**the client falls back**. The response's ``backend`` field is never ``laya`` on
an ineligible host because no response is produced at all. An *eligible* host
with no checkpoint installed also refuses (``laya_backend_not_installed``,
FR-007); eligibility is a precondition for attempting laya, not a promise that a
checkpoint exists.

What the benchmark measures, honestly
-------------------------------------
With no checkpoint installed -- every shipped build's day-one state -- the
benchmark runs against a **bundled tiny random-weights ONNX fixture** generated
in memory by :func:`build_fixture_model` (never written to disk, so no ``.onnx``
file ships; see ``tests/test_onnx_freeze_no_weights.py``). It is **explicitly
non-functional**: its weights are seeded noise, it is not a laya checkpoint, and
its latency says little about a real checkpoint's. The verdict records
``measured_against="fixture"`` and says so in its ``detail``; the real
MPS/INT8 number (OQ-N6) comes with the first real checkpoint through the same
harness (:func:`run_benchmark` takes the target as a parameter). The fixture
runs under the same ONNX Runtime thread pinning as production serving
(:func:`kenaz_ml.laya.agent.session_options`), otherwise the number would be
wrong.

When it runs (R8) and what is cached (D-C2)
-------------------------------------------
Lazily, at the **first eligibility check per engine boot** -- the first dispatch
of a laya-configured kind, or a change in the installed-checkpoint set (both are
engine-side events; "first enablement" is client-side and unobservable here).
Never at process start. The expensive part (the benchmark) is cached to
``{models_dir()}/laya_eligibility.json``, version-stamped with the onnxruntime
and laya versions, platform, architecture, and the identity of every installed
checkpoint; any stamp difference, an unreadable file, or a corrupt one forces a
recompute. Only **successful** measurements are cached (a transient failure must
not stick). **Free memory is never cached**: it is re-read at the first check of
every boot and folded into a fresh verdict.

Free memory without ``psutil`` (recorded decision: it is not a dependency)
-------------------------------------------------------------------------
"Free" means **available** memory -- what the system could give a new
allocation without swapping -- not raw free pages. Linux: ``MemAvailable`` from
``/proc/meminfo``, falling back to ``sysconf(SC_AVPHYS_PAGES)``. macOS:
``/usr/bin/vm_stat`` (free + inactive + speculative pages) through the standard
library, bounded by a short timeout. The method used is recorded in the
verdict. Where it cannot be determined the verdict says ``memory_unknown`` and
the host is treated as **ineligible** (we do not guess we have 2 GiB).
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import re
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: Minimum available memory, in bytes (design doc: ">= 2 GiB free RSS"). The one named constant.
MIN_FREE_BYTES = 2 * 1024**3
#: Calls in the micro-benchmark (design doc: 10).
BENCH_CALLS = 10
#: Hard bound on the whole benchmark, session creation included. A hung runtime is a verdict, not a hang.
BENCHMARK_TIMEOUT_SEC = 10.0
#: NFR-002: an eligible host finishes the benchmark inside this many seconds (asserted by the tests).
BENCH_BUDGET_SEC = 5.0

CACHE_FILENAME = "laya_eligibility.json"

# Reason codes -- stable, machine-readable.
REASON_ELIGIBLE = "eligible"
REASON_NOT_EVALUATED = "not_evaluated"
REASON_INSUFFICIENT_MEMORY = "insufficient_memory"
REASON_MEMORY_UNKNOWN = "memory_unknown"
REASON_BENCHMARK_FAILED = "benchmark_failed"
REASON_BENCHMARK_INVALID = "benchmark_output_invalid"
REASON_BENCHMARK_TIMEOUT = "benchmark_timeout"

BASIS_FIXTURE = "fixture"
BASIS_CHECKPOINT = "checkpoint"

FIXTURE_DOC = (
    "NON-FUNCTIONAL random-weights fixture for kenaz-ml's laya host-eligibility benchmark. Not a laya checkpoint."
)
_FIXTURE_IN = 64
_FIXTURE_HIDDEN = 128
_FIXTURE_OUT = 2


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """An explicit eligibility verdict. ``reason`` is always set; measurements are ``None`` when unmeasured."""

    eligible: bool
    reason: str
    detail: str
    measured_against: str | None = None  # "fixture" | "checkpoint" | None
    free_memory_bytes: int | None = None
    memory_method: str | None = None
    calls: int | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    max_ms: float | None = None
    stamp: dict[str, Any] = field(default_factory=dict)
    evaluated_at_ms: int | None = None

    def describe(self) -> dict[str, Any]:
        """Plain data for ``/health``: honest nulls for whatever was not measured."""
        return {
            "verdict": "eligible" if self.eligible else "ineligible",
            "reason": self.reason,
            "detail": self.detail,
            "measured_against": self.measured_against,
            "free_memory_bytes": self.free_memory_bytes,
            "memory_method": self.memory_method,
            "benchmark_calls": self.calls,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "max_ms": self.max_ms,
            "evaluated_at_ms": self.evaluated_at_ms,
        }


@dataclass(frozen=True)
class Benchmark:
    """The outcome of one benchmark run (cacheable when ``ok``)."""

    ok: bool
    reason: str
    detail: str
    basis: str | None = None
    calls: int = 0
    p50_ms: float | None = None
    p95_ms: float | None = None
    max_ms: float | None = None


# ---------------------------------------------------------------------------
# Free memory (no psutil)
# ---------------------------------------------------------------------------


def _linux_available_bytes() -> tuple[int | None, str]:
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024, "/proc/meminfo MemAvailable"
    except OSError, ValueError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"), "sysconf SC_AVPHYS_PAGES"
    except ValueError, OSError, AttributeError:
        return None, "unavailable"


def _darwin_available_bytes() -> tuple[int | None, str]:
    try:
        out = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True, timeout=2.0, check=True).stdout
        page = re.search(r"page size of (\d+) bytes", out)
        if page is None:
            return None, "unavailable"
        pages = 0
        for label in ("Pages free", "Pages inactive", "Pages speculative"):
            found = re.search(rf"{label}:\s+(\d+)\.", out)
            if found is None:
                return None, "unavailable"
            pages += int(found.group(1))
        return pages * int(page.group(1)), "vm_stat free+inactive+speculative pages"
    except OSError, subprocess.SubprocessError, ValueError:
        return None, "unavailable"


def free_memory_bytes() -> tuple[int | None, str]:
    """``(available bytes, method)``; ``(None, "unavailable")`` where it cannot be determined. Never raises."""
    try:
        if sys.platform.startswith("linux"):
            return _linux_available_bytes()
        if sys.platform == "darwin":
            return _darwin_available_bytes()
    except Exception:  # pragma: no cover - defensive; the helpers already catch
        logger.debug("laya eligibility: free-memory read failed", exc_info=True)
    return None, "unavailable"


# ---------------------------------------------------------------------------
# The bundled fixture: a tiny random-weights ONNX graph, hand-encoded in memory
# ---------------------------------------------------------------------------
#
# onnxruntime (the only ONNX dependency) cannot *write* a model and the `onnx`
# package is not a dependency, so the protobuf is encoded by hand. ONNX's wire
# format is small and stable: ModelProto{ir_version=1, opset_import=8, graph=7}.


def _varint(value: int) -> bytes:
    if value < 0:
        value += 1 << 64
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _field_varint(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


def _field_bytes(number: int, payload: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _field_str(number: int, text: str) -> bytes:
    return _field_bytes(number, text.encode("utf-8"))


def _tensor(name: str, dims: tuple[int, ...], values: Sequence[float]) -> bytes:
    body = b"".join(_field_varint(1, d) for d in dims)
    body += _field_varint(2, 1)  # FLOAT
    body += _field_str(8, name)
    body += _field_bytes(9, struct.pack(f"<{len(values)}f", *values))
    return body


def _node(op: str, inputs: Sequence[str], output: str, attrs: bytes = b"") -> bytes:
    body = b"".join(_field_str(1, i) for i in inputs) + _field_str(2, output) + _field_str(3, f"{op}_{output}")
    return body + _field_str(4, op) + attrs


def _value_info(name: str, dims: tuple[int, ...]) -> bytes:
    shape = b"".join(_field_bytes(1, _field_varint(1, d)) for d in dims)
    tensor_type = _field_varint(1, 1) + _field_bytes(2, shape)
    return _field_str(1, name) + _field_bytes(2, _field_bytes(1, tensor_type))


def build_fixture_model(seed: int = 0) -> bytes:
    """Serialized ONNX bytes: ``x[1,64] -> MatMul -> Relu -> MatMul -> Softmax -> probs[1,2]``.

    Weights are seeded noise from a fixed-seed LCG (no numpy needed, fully
    deterministic across platforms). It computes nothing meaningful; it exists
    to make ONNX Runtime do real, repeatable work for the timing harness.
    """
    state = (seed * 2654435761 + 1) & 0xFFFFFFFF

    def noise(n: int) -> list[float]:
        nonlocal state
        out = []
        for _ in range(n):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            out.append(((state >> 8) / float(1 << 24) - 0.5) * 0.2)
        return out

    axis = _field_bytes(5, _field_str(1, "axis") + _field_varint(3, -1) + _field_varint(20, 2))
    graph = b"".join(
        (
            _field_bytes(1, _node("MatMul", ["x", "w1"], "h")),
            _field_bytes(1, _node("Relu", ["h"], "a")),
            _field_bytes(1, _node("MatMul", ["a", "w2"], "z")),
            _field_bytes(1, _node("Softmax", ["z"], "probs", axis)),
            _field_str(2, "kenaz_ml_laya_eligibility_fixture"),
            _field_bytes(5, _tensor("w1", (_FIXTURE_IN, _FIXTURE_HIDDEN), noise(_FIXTURE_IN * _FIXTURE_HIDDEN))),
            _field_bytes(5, _tensor("w2", (_FIXTURE_HIDDEN, _FIXTURE_OUT), noise(_FIXTURE_HIDDEN * _FIXTURE_OUT))),
            _field_str(10, FIXTURE_DOC),
            _field_bytes(11, _value_info("x", (1, _FIXTURE_IN))),
            _field_bytes(12, _value_info("probs", (1, _FIXTURE_OUT))),
        )
    )
    opset = _field_str(1, "") + _field_varint(2, 13)
    return b"".join(
        (
            _field_varint(1, 8),  # ir_version
            _field_str(2, "kenaz-ml-eligibility-fixture"),
            _field_str(6, FIXTURE_DOC),
            _field_bytes(7, graph),
            _field_bytes(8, opset),
        )
    )


class BenchTarget(Protocol):
    """Something the harness can time: a fixture today, a real checkpoint's agent later."""

    basis: str

    def call(self, index: int) -> Sequence[float]:
        """One inference on the ``index``-th fixed input; returns a probability vector."""
        ...


class FixtureTarget:
    """The bundled random-weights fixture behind ONNX Runtime (CPU only, production thread pinning)."""

    basis = BASIS_FIXTURE

    def __init__(self) -> None:
        import onnxruntime as ort

        from kenaz_ml.laya.agent import session_options

        self._session = ort.InferenceSession(
            build_fixture_model(), sess_options=session_options(), providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name

    def call(self, index: int) -> Sequence[float]:
        import numpy as np

        x = np.random.RandomState(index).standard_normal((1, _FIXTURE_IN)).astype(np.float32)
        (probs,) = self._session.run(None, {self._input: x})
        return [float(v) for v in probs.reshape(-1)]


def validate_output(probs: Any) -> str | None:
    """``None`` when ``probs`` is a sane probability vector, else why it is not."""
    try:
        values = [float(v) for v in probs]
    except TypeError, ValueError:
        return "output is not a numeric vector"
    if len(values) < 2:
        return f"expected a probability vector of at least 2 entries, got {len(values)}"
    if not all(math.isfinite(v) for v in values):
        return "output contains a non-finite value"
    if any(v < 0.0 or v > 1.0 for v in values):
        return "output probability outside [0, 1]"
    if abs(sum(values) - 1.0) > 1e-3:
        return f"output does not sum to 1 (sum={sum(values):.4f})"
    return None


def _percentile(sorted_ms: list[float], q: float) -> float:
    return sorted_ms[max(0, math.ceil(q * len(sorted_ms)) - 1)]


def run_benchmark(
    target_factory: Callable[[], BenchTarget] | None = None,
    *,
    calls: int = BENCH_CALLS,
    timeout_sec: float = BENCHMARK_TIMEOUT_SEC,
) -> Benchmark:
    """Time ``calls`` inferences on fixed inputs; return p50/p95/max. Bounded, never raises.

    ``target_factory`` builds the thing to time (default: the bundled fixture), so
    tests inject fakes and a real checkpoint can be substituted without touching
    the harness. Everything -- building the target included -- runs in a daemon
    thread bounded by ``timeout_sec``; a hang is a ``benchmark_timeout`` verdict.
    """
    box: dict[str, Benchmark] = {}

    def work() -> None:
        try:
            target = (target_factory or FixtureTarget)()
            timings: list[float] = []
            for index in range(calls):
                started = time.perf_counter()
                out = target.call(index)
                timings.append((time.perf_counter() - started) * 1000.0)
                problem = validate_output(out)
                if problem is not None:
                    box["result"] = Benchmark(
                        False, REASON_BENCHMARK_INVALID, f"call {index}: {problem}", basis=target.basis, calls=index + 1
                    )
                    return
            ordered = sorted(timings)
            box["result"] = Benchmark(
                True,
                REASON_ELIGIBLE,
                f"{calls} calls",
                basis=target.basis,
                calls=calls,
                p50_ms=round(_percentile(ordered, 0.50), 3),
                p95_ms=round(_percentile(ordered, 0.95), 3),
                max_ms=round(ordered[-1], 3),
            )
        except Exception as exc:
            logger.warning("laya eligibility: benchmark failed", exc_info=True)
            box["result"] = Benchmark(False, REASON_BENCHMARK_FAILED, f"{type(exc).__name__}: {exc}")

    thread = threading.Thread(target=work, name="laya-eligibility-benchmark", daemon=True)
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive():
        return Benchmark(False, REASON_BENCHMARK_TIMEOUT, f"benchmark did not finish within {timeout_sec:g}s")
    return box.get("result") or Benchmark(False, REASON_BENCHMARK_FAILED, "benchmark produced no result")


# ---------------------------------------------------------------------------
# Stamp, cache, and the lazy verdict
# ---------------------------------------------------------------------------

_checkpoints: dict[str, str] = {}
_lock = threading.RLock()
_verdict: Verdict | None = None
_verdict_stamp: dict[str, Any] | None = None


def register_checkpoint(kind_id: str, identity: str) -> None:
    """Note that ``kind_id`` has an installed checkpoint (called as the dispatch table is built)."""
    with _lock:
        _checkpoints[kind_id] = identity


def unregister_checkpoint(kind_id: str) -> None:
    with _lock:
        _checkpoints.pop(kind_id, None)


def checkpoint_set() -> list[str]:
    """The sorted identity of every installed checkpoint (part of the stamp)."""
    with _lock:
        return sorted(f"{kind}:{identity}" for kind, identity in _checkpoints.items())


def _version(dist: str) -> str | None:
    try:
        from importlib import metadata

        return metadata.version(dist)
    except Exception:
        return None


def environment_stamp() -> dict[str, Any]:
    """Everything whose change must invalidate a cached measurement (D-C2)."""
    from kenaz_ml.laya.agent import pinned_threads

    return {
        "onnxruntime": _version("onnxruntime"),
        "laya": _version("laya"),
        "platform": sys.platform,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "threads": pinned_threads(),
        "checkpoints": checkpoint_set(),
    }


def cache_path() -> Any:
    from kenaz_ml import config

    return config.models_dir() / CACHE_FILENAME


def _read_cache(stamp: dict[str, Any]) -> Benchmark | None:
    """A cached *successful* benchmark whose stamp equals ``stamp``; ``None`` otherwise. Never raises."""
    try:
        raw = json.loads(cache_path().read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("stamp") != stamp:
            return None
        b = raw["benchmark"]
        if not b.get("ok") or b.get("basis") not in (BASIS_FIXTURE, BASIS_CHECKPOINT):
            return None
        bench = Benchmark(**b)
        if bench.p50_ms is None or bench.p95_ms is None or bench.max_ms is None:
            return None
        return bench
    except OSError, ValueError, KeyError, TypeError:
        return None


def _write_cache(stamp: dict[str, Any], bench: Benchmark) -> None:
    """Atomic write (temp file + ``os.replace``) into the writable model dir. Best effort."""
    try:
        path = cache_path()
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"stamp": stamp, "benchmark": asdict(bench)}, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        logger.debug("laya eligibility: could not write the cache", exc_info=True)


def _assemble(bench: Benchmark, free: int | None, method: str, stamp: dict[str, Any], *, memory_min: int) -> Verdict:
    """Fold a benchmark and a fresh memory reading into the final verdict (memory is judged first)."""
    now = int(time.time() * 1000)
    common: dict[str, Any] = {
        "free_memory_bytes": free,
        "memory_method": method,
        "stamp": stamp,
        "evaluated_at_ms": now,
    }
    gib = 1024**3
    if free is None:
        return Verdict(
            False,
            REASON_MEMORY_UNKNOWN,
            f"available memory could not be determined on this platform ({sys.platform}); "
            f"not assuming the required {memory_min / gib:g} GiB",
            **common,
        )
    if free < memory_min:
        return Verdict(
            False,
            REASON_INSUFFICIENT_MEMORY,
            f"{free / gib:.2f} GiB available < the required {memory_min / gib:g} GiB ({method})",
            **common,
        )
    if not bench.ok:
        return Verdict(False, bench.reason, bench.detail, measured_against=bench.basis, calls=bench.calls, **common)
    basis_note = (
        "no checkpoint (fixture-measured): timings come from the bundled non-functional random-weights fixture, "
        "not a laya checkpoint"
        if bench.basis == BASIS_FIXTURE
        else "measured against the installed checkpoint"
    )
    return Verdict(
        True,
        REASON_ELIGIBLE,
        f"{free / gib:.2f} GiB available; {bench.calls}-call benchmark p50 {bench.p50_ms} ms, "
        f"p95 {bench.p95_ms} ms, max {bench.max_ms} ms; {basis_note}",
        measured_against=bench.basis,
        calls=bench.calls,
        p50_ms=bench.p50_ms,
        p95_ms=bench.p95_ms,
        max_ms=bench.max_ms,
        **common,
    )


def evaluate(
    *,
    free_memory_fn: Callable[[], tuple[int | None, str]] | None = None,
    target_factory: Callable[[], BenchTarget] | None = None,
    calls: int = BENCH_CALLS,
    timeout_sec: float = BENCHMARK_TIMEOUT_SEC,
    memory_min: int = MIN_FREE_BYTES,
    use_cache: bool = True,
) -> Verdict:
    """Compute a fresh verdict (memory now; benchmark from cache when the stamp matches). Never raises."""
    try:
        stamp = environment_stamp()
        free, method = (free_memory_fn or free_memory_bytes)()
        bench = _read_cache(stamp) if use_cache and target_factory is None else None
        if bench is None:
            if free is not None and free < memory_min:
                # Insufficient memory decides the verdict; do not spend seconds (and memory) benchmarking.
                bench = Benchmark(False, REASON_INSUFFICIENT_MEMORY, "benchmark skipped: insufficient memory")
            else:
                bench = run_benchmark(target_factory, calls=calls, timeout_sec=timeout_sec)
                if bench.ok and use_cache and target_factory is None:
                    _write_cache(stamp, bench)
        return _assemble(bench, free, method, stamp, memory_min=memory_min)
    except Exception as exc:  # pragma: no cover - every step above already catches; belt and braces
        logger.warning("laya eligibility: evaluation failed", exc_info=True)
        return Verdict(False, REASON_BENCHMARK_FAILED, f"{type(exc).__name__}: {exc}")


def current_verdict(checkpoint_key: str = "") -> Verdict:
    """The verdict the dispatch layer consults -- computed lazily, once per boot and per stamp.

    ``checkpoint_key`` is accepted for compatibility with the WP02 seam and
    ignored: the installed-checkpoint set is tracked through
    :func:`register_checkpoint`. A change in that set (or in any stamp element)
    recomputes; otherwise the in-memory verdict stands for the rest of the boot.
    """
    global _verdict, _verdict_stamp
    with _lock:
        try:
            stamp = environment_stamp()
        except Exception:  # pragma: no cover
            stamp = {}
        if _verdict is not None and _verdict_stamp == stamp:
            return _verdict
        verdict = evaluate()
        _verdict, _verdict_stamp = verdict, stamp
        return verdict


def recompute(**kwargs: Any) -> Verdict:
    """Drop every cached measurement (memory and disk) and evaluate afresh."""
    global _verdict, _verdict_stamp
    with _lock:
        try:
            cache_path().unlink()
        except Exception:
            pass
        verdict = evaluate(**{"use_cache": False, **kwargs})
        try:
            _verdict_stamp = environment_stamp()
        except Exception:  # pragma: no cover
            _verdict_stamp = {}
        _verdict = verdict
        return verdict


def cached_verdict() -> Verdict | None:
    """The in-process verdict if one has been computed this boot; never triggers a measurement."""
    with _lock:
        return _verdict


def reset() -> None:
    """Forget the in-process verdict and registered checkpoints (tests)."""
    global _verdict, _verdict_stamp
    with _lock:
        _verdict = None
        _verdict_stamp = None
        _checkpoints.clear()


def describe() -> dict[str, Any]:
    """``/health`` payload: the cached verdict as plain data, or an honest ``not_evaluated``.

    Never triggers the benchmark -- health is polled constantly, and the
    measurement happens at the first dispatch of a laya-configured kind.
    """
    verdict = cached_verdict()
    if verdict is None:
        return {
            "verdict": "not_evaluated",
            "reason": REASON_NOT_EVALUATED,
            "detail": "measured lazily at the first dispatch of a laya-configured kind; none has happened this boot",
            "measured_against": None,
            "free_memory_bytes": None,
            "memory_method": None,
            "benchmark_calls": None,
            "p50_ms": None,
            "p95_ms": None,
            "max_ms": None,
            "evaluated_at_ms": None,
        }
    return verdict.describe()


def selfcheck() -> dict[str, Any]:
    """Frozen-bundle probe (``onnx-selfcheck``): fixture benchmark, unloaded laya runtime. Never touches the disk cache."""
    from kenaz_ml.laya.agent import LayaRuntime, NoCheckpointError

    bench = run_benchmark()
    runtime = LayaRuntime()
    try:
        runtime.predict("branch_now", ("a",), (1.0,))
        refused = False
    except NoCheckpointError:
        refused = True
    return {
        "fixture_benchmark": asdict(bench),
        "unloaded_runtime_refuses": refused and not runtime.is_loaded,
        "fixture_bytes": len(build_fixture_model()),
    }
