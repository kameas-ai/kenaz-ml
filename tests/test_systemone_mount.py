"""laya-serving-and-packs-01MSK2SP WP01 -- the raw ``/v1/systemone`` pass-through.

laya itself cannot be installed (it hard-requires torch; see
``src/kenaz_ml/laya/systemone_mount.py``), so every test drives the vendored
in-process route with a fake backend that returns laya's own Jev-shaped result.
The contract under test is the *route's* (status codes, envelopes, admission,
single forward pass), never laya's inference.
"""

from __future__ import annotations

import ast
import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kenaz_ml.laya import systemone_mount
from kenaz_ml.laya.systemone_mount import mount_systemone

QUESTIONS = {"q1": {"type": "noul", "instructions": "Is the sky blue?"}}
LAYA_RESULT = {
    "model": "laya-fixture",
    "answers": {"q1": {"type": "noul", "noul": 0.9, "confidence": 0.7, "answer_confidence": 0.9, "action": {}}},
    "usage": {"input_tokens": 12, "output_tokens": 0},
}


class FakeBackend:
    """Returns laya's native shape; records concurrency."""

    def __init__(self, hold: threading.Event | None = None, delay: float = 0.0) -> None:
        self.hold = hold
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls: list[dict[str, Any]] = []
        self.entered = threading.Event()
        self._lock = threading.Lock()
        self.error: Exception | None = None

    def _enter(self) -> None:
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.entered.set()
        if self.hold is not None:
            self.hold.wait(timeout=10)
        if self.delay:
            time.sleep(self.delay)

    def _exit(self) -> None:
        with self._lock:
            self.in_flight -= 1

    def predict(self, state: Any, questions: dict[str, Any], **controls: Any) -> dict[str, Any]:
        self._enter()
        try:
            if self.error is not None:
                raise self.error
            self.calls.append({"state": state, "questions": questions, **controls})
            return dict(LAYA_RESULT)
        finally:
            self._exit()

    def predict_batch(self, states: list[Any], questions: dict[str, Any], **controls: Any) -> list[dict[str, Any]]:
        self._enter()
        try:
            return [dict(LAYA_RESULT) for _ in states]
        finally:
            self._exit()


def _app(backend: FakeBackend | None) -> FastAPI:
    app = FastAPI()
    mount_systemone(app, lambda: backend)
    return app


def test_single_passthrough_is_laya_shaped_and_unmodified() -> None:
    backend = FakeBackend()
    client = TestClient(_app(backend))
    resp = client.post("/v1/systemone", json={"state": "it is noon", "questions": QUESTIONS, "max_len": 128})
    assert resp.status_code == 200
    assert resp.json() == LAYA_RESULT  # byte-for-byte: no injected backend field, no confidence remap
    assert "backend" not in resp.json()
    assert "Server-Timing" in resp.headers and "X-Inference-Time-Ms" in resp.headers
    assert backend.calls == [{"state": "it is noon", "questions": QUESTIONS, "max_len": 128}]


def test_batch_returns_per_item_single_shape() -> None:
    client = TestClient(_app(FakeBackend()))
    resp = client.post("/v1/systemone/batch", json={"states": ["a", "b", "c"], "questions": QUESTIONS})
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"] == [LAYA_RESULT] * 3
    assert body["total_usage"] == {"input_tokens": 36, "output_tokens": 0}


def test_no_checkpoint_is_a_detail_envelope_503_without_retry_after() -> None:
    client = TestClient(_app(None))
    for path, body in (
        ("/v1/systemone", {"state": "x", "questions": QUESTIONS}),
        ("/v1/systemone/batch", {"states": ["x"], "questions": QUESTIONS}),
    ):
        resp = client.post(path, json=body)
        assert resp.status_code == 503
        assert set(resp.json()) == {"detail"}
        assert resp.json()["detail"].startswith("model not loaded")
        assert "Retry-After" not in resp.headers


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"state": "x"}, 400),
        ({"questions": QUESTIONS}, 400),
        ({"state": "x", "questions": []}, 400),
        ({"state": "x" * 60000, "questions": QUESTIONS}, 413),
        ({"state": "x", "questions": {f"q{i}": {} for i in range(65)}}, 413),
        ({"state": "x", "questions": QUESTIONS, "hooks": ["nope"]}, 422),
    ],
)
def test_request_validation_uses_laya_status_codes(body: dict[str, Any], status: int) -> None:
    resp = TestClient(_app(FakeBackend())).post("/v1/systemone", json=body)
    assert resp.status_code == status
    assert set(resp.json()) == {"detail"}


def test_non_json_body_is_400() -> None:
    resp = TestClient(_app(FakeBackend())).post("/v1/systemone", content=b"{not json")
    assert resp.status_code == 400


def test_value_error_is_422_and_other_errors_are_a_fixed_500() -> None:
    backend = FakeBackend()
    client = TestClient(_app(backend), raise_server_exceptions=False)
    backend.error = ValueError("question 'q1': bad type")
    resp = client.post("/v1/systemone", json={"state": "x", "questions": QUESTIONS})
    assert (resp.status_code, resp.json()) == (422, {"detail": "question 'q1': bad type"})
    backend.error = RuntimeError("secret path /tmp/weights")
    resp = client.post("/v1/systemone", json={"state": "x", "questions": QUESTIONS})
    assert (resp.status_code, resp.json()) == (500, {"detail": "inference failed"})


def test_api_key_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAYA_API_KEY", "s3cret")
    client = TestClient(_app(FakeBackend()))
    body = {"state": "x", "questions": QUESTIONS}
    assert client.post("/v1/systemone", json=body).status_code == 401
    assert client.post("/v1/systemone", json=body, headers={"Authorization": "Bearer s3cret"}).status_code == 200


async def _burst(
    app: FastAPI, n: int, backend: FakeBackend, release: threading.Event | None = None
) -> list[httpx.Response]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = {"state": "x", "questions": QUESTIONS}
        tasks = [asyncio.create_task(client.post("/v1/systemone", json=body)) for _ in range(n)]
        if release is not None:
            # Let the first request reach the backend and the rest be refused before releasing it.
            for _ in range(200):
                if backend.entered.is_set():
                    break
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.2)
            release.set()
        return await asyncio.gather(*tasks)


def test_admission_overflow_is_503_with_retry_after_never_queued(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAYA_MAX_CONCURRENT", "1")
    release = threading.Event()
    backend = FakeBackend(hold=release)
    responses = asyncio.run(_burst(_app(backend), 3, backend, release))
    statuses = sorted(r.status_code for r in responses)
    assert statuses == [200, 503, 503]
    busy = next(r for r in responses if r.status_code == 503)
    assert busy.headers["Retry-After"] == "1"
    assert busy.json() == {"detail": "server busy, try again later"}


def test_exactly_one_forward_pass_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    """NFR-003: admission admits many, the single worker + gate run one at a time."""
    monkeypatch.setenv("LAYA_MAX_CONCURRENT", "16")
    backend = FakeBackend(delay=0.03)
    responses = asyncio.run(_burst(_app(backend), 6, backend))
    assert [r.status_code for r in responses] == [200] * 6
    assert backend.max_in_flight == 1


def test_routes_are_absent_from_openapi() -> None:
    schema = _app(FakeBackend()).openapi()
    assert not [p for p in schema["paths"] if p.startswith("/v1/systemone")]


def test_create_app_mounts_locally_and_reports_model_not_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    from kenaz_ml.app import create_app

    with TestClient(create_app()) as client:
        resp = client.post("/v1/systemone", json={"state": "x", "questions": QUESTIONS})
        assert resp.status_code == 503
        assert resp.json()["detail"].startswith("model not loaded")


def test_cloud_mode_does_not_mount() -> None:
    from kenaz_ml.app import create_app
    from kenaz_ml.config import ServingMode

    app = create_app(ServingMode.CLOUD)
    assert not [r for r in app.routes if getattr(r, "path", "").startswith("/v1/systemone")]


def test_request_cycle_opens_no_socket() -> None:
    """C-007: the mount's whole request cycle under the no-egress guard."""
    from tests.test_no_egress import no_network

    backend = FakeBackend()
    app = _app(backend)
    loop = asyncio.new_event_loop()  # built before the guard: loop creation itself makes a socketpair
    try:
        with no_network() as recorder:
            resp = loop.run_until_complete(_one(app))
        assert resp.status_code == 200
        assert recorder.socket_attempts == []
    finally:
        loop.close()


async def _one(app: FastAPI) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/v1/systemone", json={"state": "x", "questions": QUESTIONS})


def test_mount_module_imports_no_network_client() -> None:
    tree = ast.parse(Path(systemone_mount.__file__).read_text())
    banned = {"httpx", "requests", "urllib", "aiohttp", "http", "socket", "subprocess"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & banned), imported & banned
    # laya / torch / onnxruntime are never imported at module scope either (cold start).
    assert not (imported & {"laya", "torch", "onnxruntime", "transformers"})
