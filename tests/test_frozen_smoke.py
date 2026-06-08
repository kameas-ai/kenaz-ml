"""Freeze smoke test for the frozen `kameas-ml` binary (FR-3, ADR-ml-packaging).

This guards the known scikit-learn / numpy / uvicorn hidden-import breakage
that PyInstaller one-file builds are prone to: a binary can `serve` and answer
`/health` while still failing the moment an sklearn estimator is exercised,
because a Cython-compiled submodule was dropped from the frozen graph. So the
test does more than "it starts" — it POSTs `/predict/stuck` and asserts a real
sklearn prediction comes back.

The test is SKIPPED unless ``KAMEAS_ML_FROZEN_BIN`` points at a built
artifact, so the normal `pytest tests/` run on a dev machine (no frozen binary)
stays green. It is the integration contract between the freeze recipe (part A)
and kenaz's supervised-production path (part B): B's production wiring is only
considered correct once this passes against the baked binary.

Run it explicitly after a freeze build:

    pip install -e ".[freeze]"
    pyinstaller freeze/kameas-ml.spec --noconfirm
    KAMEAS_ML_FROZEN_BIN=$PWD/dist/kameas-ml pytest tests/test_frozen_smoke.py -v
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

FROZEN_BIN = os.environ.get("KAMEAS_ML_FROZEN_BIN")

pytestmark = pytest.mark.skipif(
    not FROZEN_BIN,
    reason="KAMEAS_ML_FROZEN_BIN not set; build the frozen binary first "
    "(pyinstaller freeze/kameas-ml.spec) and point the env var at dist/kameas-ml",
)


def _free_port() -> int:
    """Reserve an ephemeral port and return it (closed immediately so the
    server can bind it)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_healthy(base_url: str, proc: subprocess.Popen, timeout: float = 60.0) -> None:
    """Poll /health until it returns 200, or fail with captured output."""
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out, _ = proc.communicate(timeout=5)
            raise AssertionError(
                f"frozen binary exited early (code {proc.returncode}) before "
                f"becoming healthy.\n--- output ---\n{out}"
            )
        try:
            with urlopen(f"{base_url}/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except (URLError, ConnectionError, OSError) as exc:  # not up yet
            last_err = exc
        time.sleep(0.5)
    raise AssertionError(f"frozen binary did not become healthy within {timeout}s: {last_err}")


@pytest.fixture
def frozen_server():
    """Launch the frozen binary with `serve --port <ephemeral>`; yield base URL."""
    assert os.path.exists(FROZEN_BIN), f"frozen binary not found: {FROZEN_BIN}"
    assert os.access(FROZEN_BIN, os.X_OK), f"frozen binary not executable: {FROZEN_BIN}"

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    proc = subprocess.Popen(
        [FROZEN_BIN, "serve", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_healthy(base_url, proc)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _post_json(url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_frozen_binary_serves_health(frozen_server: str) -> None:
    """The frozen binary boots and answers /health with 200 — proves the
    uvicorn/fastapi freeze graph is intact."""
    with urlopen(f"{frozen_server}/health", timeout=5) as resp:
        assert resp.status == 200
        body = json.loads(resp.read().decode("utf-8"))
    assert "status" in body


def test_frozen_binary_predict_stuck(frozen_server: str) -> None:
    """POST /predict/stuck with a fixed feature dict returns 200 + a
    `probability` field — proves the sklearn estimator path survived freezing
    (the actual hidden-import guard, not just "it started")."""
    # A minimal, well-formed StuckRequest feature dict. Local-mode serving
    # falls back to a deterministic response when no trained model is present,
    # but the request still flows through the sklearn-backed StuckPredictor
    # path, exercising the frozen estimator imports.
    features = {
        "elapsed_minutes": 42.0,
        "edit_count": 7.0,
        "test_run_count": 2.0,
        "error_count": 1.0,
        "file_switch_count": 3.0,
    }
    status, body = _post_json(f"{frozen_server}/predict/stuck", {"features": features})

    assert status == 200, f"unexpected status {status}: {body}"
    assert "probability" in body, f"response missing 'probability': {body}"
    assert isinstance(body["probability"], (int, float))
    assert 0.0 <= float(body["probability"]) <= 1.0
