"""laya-serving-and-packs-01MSK2SP WP01 -- ONNX Runtime ships, no weights, no torch (User Story 2).

Two layers, like the rest of the freeze tests:

* **Source-level** (always run): the declarations that make the bundle what it
  must be -- ``torch`` never in the base dependency set (C-004), ``onnxruntime``
  pinned exactly, the freeze spec refusing checkpoint artifacts and excluding
  torch, and ``kenaz_ml.laya`` importing nothing heavy at module scope.
* **Frozen** (skipped unless ``KENAZ_ML_FROZEN_BIN`` points at a built artifact):
  the real ``onnx-selfcheck``, a walk of the bundle for checkpoint artifacts, the
  engine starting with no checkpoint, and the cold-start budget.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN_BIN = os.environ.get("KENAZ_ML_FROZEN_BIN")

CHECKPOINT_NAMES = {"rl_agent_config.json", "tokenizer.json", "tokenizer_config.json"}


def _pyproject() -> dict:
    tomllib = pytest.importorskip("tomllib")  # stdlib from 3.11; the suite runs on 3.12
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def _names(requirements: list[str]) -> set[str]:
    return {re.split(r"[\s\[<>=!~;]", r, maxsplit=1)[0].lower().replace("_", "-") for r in requirements}


# ---------------------------------------------------------------------------
# Source-level
# ---------------------------------------------------------------------------


def test_torch_is_never_a_declared_dependency() -> None:
    project = _pyproject()["project"]
    everything = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        everything += extra
    declared = _names(everything)
    # C-004, plus the packages that would drag torch in (laya requires torch + transformers).
    assert not declared & {"torch", "transformers", "laya", "laya-serve", "pytorch"}


def test_onnxruntime_is_declared_and_pinned_exactly() -> None:
    deps = _pyproject()["project"]["dependencies"]
    pins = [d for d in deps if d.lower().startswith("onnxruntime")]
    assert len(pins) == 1 and re.fullmatch(r"onnxruntime==\d+\.\d+\.\d+", pins[0]), pins


def test_freeze_spec_refuses_checkpoint_artifacts_and_torch() -> None:
    spec = (REPO_ROOT / "freeze" / "kenaz-ml.spec").read_text()
    assert 'collect_dynamic_libs("onnxruntime")' in spec
    assert '"torch"' in spec and '"transformers"' in spec  # in `excludes`
    assert "_is_checkpoint_artifact" in spec
    for marker in (".onnx", "rl_agent_config.json", "tokenizer"):
        assert marker in spec


def test_freeze_spec_checkpoint_filter_behaves() -> None:
    """Exec just the filter out of the spec and feed it real-shaped entries."""
    spec = (REPO_ROOT / "freeze" / "kenaz-ml.spec").read_text()
    tree = ast.parse(spec)
    namespace: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_CHECKPOINT_MARKERS" for t in node.targets):
            exec(compile(ast.Module([node], []), "spec", "exec"), namespace)
        if isinstance(node, ast.FunctionDef) and node.name == "_is_checkpoint_artifact":
            exec(compile(ast.Module([node], []), "spec", "exec"), namespace)
    is_ckpt = namespace["_is_checkpoint_artifact"]
    assert is_ckpt(("/x/laya.onnx", "laya"))
    assert is_ckpt(("/x/rl_agent_config.json", "."))
    assert is_ckpt(("/x/tokenizer/vocab.txt", "laya/tokenizer"))
    assert not is_ckpt(("/x/onnxruntime/capi/libonnxruntime.1.30.0.dylib", "onnxruntime/capi"))
    assert not is_ckpt(("/x/sklearn/data.csv", "sklearn"))


def test_laya_package_imports_nothing_heavy_at_module_scope() -> None:
    heavy = {"laya", "torch", "onnxruntime", "transformers", "onnx"}
    for path in (REPO_ROOT / "src" / "kenaz_ml" / "laya").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:  # module scope only; function bodies may import lazily
            if isinstance(node, ast.Import):
                assert not {a.name.split(".")[0] for a in node.names} & heavy, path
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in heavy, path


def test_importing_the_app_does_not_import_onnxruntime_or_torch() -> None:
    code = (
        "import sys; import kenaz_ml.app, kenaz_ml.laya.systemone_mount; "
        "bad = sorted(m for m in ('onnxruntime', 'torch', 'laya', 'transformers') if m in sys.modules); "
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# Frozen
# ---------------------------------------------------------------------------

frozen = pytest.mark.skipif(
    not FROZEN_BIN,
    reason="KENAZ_ML_FROZEN_BIN not set; build the frozen binary first (make freeze)",
)


def _bundle_root() -> Path:
    assert FROZEN_BIN is not None
    return Path(FROZEN_BIN).resolve().parent


@frozen
def test_frozen_onnxruntime_runs_and_torch_is_absent() -> None:
    proc = subprocess.run([FROZEN_BIN, "onnx-selfcheck"], capture_output=True, text=True, timeout=180)
    report = json.loads(proc.stdout)
    assert proc.returncode == 0 and report["ok"], report
    assert report["frozen"] is True
    assert report["onnxruntime_version"]
    assert "CPUExecutionProvider" in report["providers"]
    assert report["torch_importable"] is False  # C-004
    assert report["laya_importable"] is False  # not bundled (see systemone_mount.py)
    # The bundled fixture ran through the *frozen* ONNX Runtime (WP03), and an unloaded runtime refuses.
    bench = report["fixture_benchmark"]
    assert bench["ok"] is True and bench["basis"] == "fixture" and bench["calls"] == 10
    assert bench["p50_ms"] is not None and bench["p95_ms"] is not None
    assert report["unloaded_runtime_refuses"] is True


@frozen
def test_frozen_bundle_carries_onnxruntime_natives_and_no_checkpoint_artifact() -> None:
    root = _bundle_root()
    natives = {p.name for p in root.rglob("*") if p.suffix in (".so", ".dylib") and "onnxruntime" in p.name}
    assert natives, "onnxruntime's native libraries are missing from the bundle"
    offenders = [
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.suffix == ".onnx" or p.name in CHECKPOINT_NAMES or "tokenizer" in p.parts
    ]
    assert offenders == []  # FR-003 / C-001: zero checkpoint artifacts of any kind
    # C-004. `excludes` only prunes the *module* graph; a torch native library contributed as a binary by
    # some other package's hook would pass it, and a bare `rglob("torch")` sees only a directory named
    # exactly "torch". So: the packages by name, and torch's native libraries by file name (review fix).
    # (Feast ships `torch_wrapper.py` and a `pytorch_nlp` template: Python text, deliberately not matched.)
    torch_native = re.compile(r"^(lib)?(torch|c10|shm)([_.-][\w.-]*)?\.(so|dylib|dll|pyd)$", re.IGNORECASE)
    torch_like = [
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.name in {"torch", "transformers", "laya"} or torch_native.match(p.name)
    ]
    assert torch_like == [], f"torch/transformers/laya must not be in the bundle (C-004): {torch_like}"


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(url: str) -> tuple[int, dict]:
    from urllib.error import HTTPError
    from urllib.request import urlopen

    try:
        with urlopen(url, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def _post(url: str, payload: dict) -> tuple[int, dict, dict]:
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    req = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read()), dict(resp.headers)
    except HTTPError as err:
        return err.code, json.loads(err.read() or b"{}"), dict(err.headers)


@frozen
def test_frozen_engine_starts_with_no_checkpoint_and_reports_laya_unavailable() -> None:
    """User Story 2 scenario 3: day-one state -- starts, never hangs or crashes on a missing weight file."""
    import time
    from urllib.error import URLError

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [FROZEN_BIN, "serve", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        health: dict = {}
        while time.monotonic() < deadline:
            try:
                status, health = _get(f"{base}/health")
                if status == 200:
                    break
            except (URLError, ConnectionError, OSError):
                time.sleep(0.5)
        assert health.get("status") == "ok", "frozen engine did not become healthy"
        # /health carries the lazily-measured verdict: honest nulls before anything dispatched.
        assert health["laya_eligibility"]["verdict"] == "not_evaluated"
        # Every laya-bearing kind is honestly unavailable; none is served, none hangs.
        status, contracts = _get(f"{base}/v1/contracts")
        assert status == 200
        assert all(not k["available"] for k in contracts["kinds"].values())
        # The raw laya route answers with its "model not loaded" 503 (no Retry-After: retrying cannot help).
        status, body, headers = _post(
            f"{base}/v1/systemone", {"state": "x", "questions": {"q": {"type": "noul", "instructions": "i"}}}
        )
        assert status == 503 and body["detail"].startswith("model not loaded")
        assert "Retry-After" not in headers
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
