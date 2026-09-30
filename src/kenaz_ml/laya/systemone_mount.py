"""``POST /v1/systemone`` and ``POST /v1/systemone/batch`` -- laya's native wire API.

**This is never the client contract.** ``/v1/recommend/{kind}`` does not call it
(spec C-002; ``tests/test_laya_dispatch_boundary.py`` proves that structurally).
It exists because an owner ruling requires raw laya to be reachable for direct
callers -- parity testing today. The response shapes are *laya's*: no ``backend``
field is injected, no confidence is remapped, and errors keep laya-serve's own
``{"detail": ...}`` envelope (FR-001).

Why this is a vendored route and not laya-serve's own app (plan D-C1 fallback)
------------------------------------------------------------------------------
D-C1 prefers mounting laya-serve's own ASGI app object. That is **not possible
here**: verified 2026-09-30 against PyPI, ``laya`` (0.3.22, Apache-2.0) declares
``torch>=2.0.0`` and ``transformers>=4.48.0`` as *unconditional* requirements and
its ``laya.onnx_agent`` -> ``laya.common`` imports torch at module top level; the
standalone ``laya-serve`` 0.2.1 is deprecated/archived and upstream's
``laya[serve]`` is the same ``laya.serve.create_app`` over the same torch
Router. Spec C-004 forbids torch in the base dependency set and the frozen
bundle, so neither package can be declared. The sanctioned fallback (R8, D-C1)
is what this module is: a thin in-process route **whose request validation,
status codes, admission control and response envelope are vendored from
``laya/serve.py`` (laya 0.3.22, Apache-2.0)**, backed by a pluggable in-process
backend. Never a subprocess proxy, never a socket -- the no-egress fence
(C-007) covers it.

Behaviour deliberately matching laya-serve
------------------------------------------
* ``LAYA_MAX_CONCURRENT`` (default 16) bounds requests past auth; the excess is
  refused with ``503`` + ``Retry-After: 1``, never queued (spec scenario 4). This
  is laya's own admission layer, reproduced -- kenaz-ml adds no second one.
* One forward pass at a time: inference runs on a single-worker pool behind an
  ``asyncio.Lock`` (NFR-003).
* ``LAYA_API_KEY``, when set, requires ``Authorization: Bearer <it>`` (401).
* Validation: 400 (malformed body / missing ``state`` or ``questions``), 413
  (size limits), 422 (``ValueError`` from the backend -- a question-shape
  problem), a fixed 500 ``inference failed`` for anything else. Responses carry
  ``Server-Timing`` and ``X-Inference-Time-Ms``.

The one thing laya-serve has no shape for
-----------------------------------------
laya-serve always has a Router that loads on demand, so it has no native "model
not loaded" response. With no checkpoint installed (every install's day-one
state) this mount answers ``503 {"detail": "model not loaded: ..."}`` in the same
``detail`` envelope and **without** ``Retry-After`` (retrying cannot help). That
single envelope is kenaz-ml's own and is the one place this route is not a
verbatim laya shape; it is reported in the mission hand-off.

The mount is local-mode only (spec is silent on cloud; the safe default).
The routes are excluded from kenaz-ml's OpenAPI document: they are laya's
contract, not ours, so ``make openapi-check`` neither sees nor pins them.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Protocol

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

SYSTEMONE_PATH = "/v1/systemone"
SYSTEMONE_BATCH_PATH = "/v1/systemone/batch"

# Guardrails vendored from laya/serve.py (0.3.22).
MAX_QUESTIONS = 64
MAX_STATE_CHARS = 50000
MAX_BATCH_STATES = 64
MAX_BODY_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_CONCURRENT = 16

#: Controls a JSON body may carry through to the backend (laya's ``BODY_CONTROLS``).
BODY_CONTROLS = ("max_len", "head_max_len", "lang", "min_confidence")
#: Sent by a caller, refused with 422 (laya's ``BODY_REFUSALS``): callables that cannot cross JSON.
BODY_REFUSALS = ("hooks", "on_predict_start", "on_predict_end", "hooks_raise", "hooks_timeout")


class SystemOneBackend(Protocol):
    """What the mount calls. Implemented in-process by :mod:`kenaz_ml.laya.agent`.

    ``predict`` returns laya's Jev-shaped result dict unmodified
    (``{"model", "answers", "usage", ...}``); ``predict_batch`` returns one such
    dict per state. A ``ValueError`` means the *request* was bad (422).
    """

    def predict(self, state: Any, questions: dict[str, Any], **controls: Any) -> dict[str, Any]: ...

    def predict_batch(self, states: list[Any], questions: dict[str, Any], **controls: Any) -> list[dict[str, Any]]: ...


#: Returns the loaded backend, or ``None`` when no checkpoint is installed.
BackendProvider = Callable[[], SystemOneBackend | None]


def _resolve_max_concurrent() -> int:
    raw = os.environ.get("LAYA_MAX_CONCURRENT")
    if not raw:
        return DEFAULT_MAX_CONCURRENT
    try:
        n = int(raw)
    except ValueError:
        return DEFAULT_MAX_CONCURRENT
    return n if n > 0 else DEFAULT_MAX_CONCURRENT


def _state_length(state: Any) -> int:
    try:
        return len(state) if isinstance(state, str) else len(json.dumps(state, ensure_ascii=False))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="'state' must be JSON-serializable") from None


def _check_request_limits(state: Any, questions: Any) -> None:
    if state is None:
        raise HTTPException(status_code=400, detail="'state' is required")
    if not isinstance(questions, dict) or not questions:
        raise HTTPException(status_code=400, detail="'questions' must be an object")
    if len(questions) > MAX_QUESTIONS:
        raise HTTPException(status_code=413, detail=f"too many questions (max {MAX_QUESTIONS})")
    if _state_length(state) > MAX_STATE_CHARS:
        raise HTTPException(status_code=413, detail=f"'state' too large (max {MAX_STATE_CHARS} characters)")


def _check_batch_limits(states: Any, questions: Any) -> None:
    if not isinstance(states, list) or not states:
        raise HTTPException(status_code=400, detail="'states' must be a non-empty list")
    if len(states) > MAX_BATCH_STATES:
        raise HTTPException(status_code=413, detail=f"too many states (max {MAX_BATCH_STATES})")
    for state in states:
        _check_request_limits(state, questions)


async def _read_json_body(request: Request) -> Any:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="request body too large")
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="request body too large")
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="request body must be valid JSON") from None


def _controls(body: dict[str, Any]) -> dict[str, Any]:
    refused = [k for k in BODY_REFUSALS if k in body]
    if refused:
        raise HTTPException(
            status_code=422, detail=f"{', '.join(refused)} cannot be sent over HTTP (a hook is a callable)"
        )
    return {k: body[k] for k in BODY_CONTROLS if body.get(k) is not None}


class SystemOneService:
    """One mount's state: the admission semaphore, the inference gate and the single worker."""

    def __init__(self, provider: BackendProvider) -> None:
        self._provider = provider
        self._max_concurrent = _resolve_max_concurrent()
        # Created on first use: asyncio primitives bind to the running loop.
        self._admission: asyncio.Semaphore | None = None
        self._gate: asyncio.Lock | None = None
        # One worker: one forward pass at a time is what a single CPU wants (NFR-003).
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-infer")
        self.router = APIRouter()
        self.router.add_api_route(SYSTEMONE_PATH, self._systemone, methods=["POST"], include_in_schema=False)
        self.router.add_api_route(SYSTEMONE_BATCH_PATH, self._batch, methods=["POST"], include_in_schema=False)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # -- auth / admission ---------------------------------------------------

    @staticmethod
    def _check_auth(authorization: str | None) -> None:
        api_key = os.environ.get("LAYA_API_KEY") or None
        if api_key is None:
            return
        supplied = (authorization or "").encode("utf-8", "surrogateescape")
        expected = ("Bearer " + api_key).encode("utf-8", "surrogateescape")
        if not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @asynccontextmanager
    async def _admit(self):
        if self._admission is None:
            self._admission = asyncio.Semaphore(self._max_concurrent)
        if self._admission.locked():
            # Refused, never queued -- laya-serve's own behaviour (spec scenario 4).
            raise HTTPException(status_code=503, detail="server busy, try again later", headers={"Retry-After": "1"})
        await self._admission.acquire()
        try:
            yield
        finally:
            self._admission.release()

    def _backend(self) -> SystemOneBackend:
        backend = self._provider()
        if backend is None:
            raise HTTPException(status_code=503, detail="model not loaded: no laya checkpoint is installed")
        return backend

    async def _run(self, fn: Callable[[], Any], what: str) -> JSONResponse:
        if self._gate is None:
            self._gate = asyncio.Lock()
        try:
            async with self._gate:
                loop = asyncio.get_running_loop()
                started = time.perf_counter()
                content = await loop.run_in_executor(self._pool, fn)
                infer_ms = (time.perf_counter() - started) * 1000.0
                return JSONResponse(
                    content=content,
                    headers={
                        "Server-Timing": f"inference;dur={infer_ms:.2f}",
                        "X-Inference-Time-Ms": f"{infer_ms:.2f}",
                    },
                )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception:
            logger.exception("%s failed", what)
            raise HTTPException(status_code=500, detail="inference failed") from None

    # -- routes -------------------------------------------------------------

    async def _systemone(self, request: Request, authorization: str | None = Header(default=None)):
        self._check_auth(authorization)
        async with self._admit():
            body = await _read_json_body(request)
            if not isinstance(body, dict) or "questions" not in body:
                raise HTTPException(status_code=400, detail="request body must be an object with a 'questions' field")
            state, questions = body.get("state"), body["questions"]
            _check_request_limits(state, questions)
            controls = _controls(body)
            backend = self._backend()
            return await self._run(lambda: backend.predict(state, questions, **controls), "systemone inference")

    async def _batch(self, request: Request, authorization: str | None = Header(default=None)):
        self._check_auth(authorization)
        async with self._admit():
            body = await _read_json_body(request)
            if not isinstance(body, dict) or "questions" not in body or "states" not in body:
                raise HTTPException(
                    status_code=400, detail="request body must be an object with 'states' and 'questions' fields"
                )
            states, questions = body["states"], body["questions"]
            _check_batch_limits(states, questions)
            backend = self._backend()

            def _do() -> dict[str, Any]:
                results = backend.predict_batch(states, questions)
                total = sum(r.get("usage", {}).get("input_tokens", 0) for r in results)
                return {"results": results, "total_usage": {"input_tokens": total, "output_tokens": 0}}

            return await self._run(_do, "systemone batch inference")


def mount_systemone(app: FastAPI, provider: BackendProvider) -> SystemOneService:
    """Attach ``/v1/systemone`` and ``/v1/systemone/batch`` to ``app`` (in-process; local mode).

    Registered additively, before nothing it could shadow: these are exact
    paths under ``/v1/systemone`` and no other route lives there. Laya is not
    imported here; the ``provider`` hands over whatever backend is loaded.
    """
    service = SystemOneService(provider)
    app.include_router(service.router)
    app.state.systemone = service
    return service
