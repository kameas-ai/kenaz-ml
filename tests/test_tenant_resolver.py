"""The tenant-resolver seam (engine phase 1 of hosted inference).

Cloud mode resolves the tenant through a resolver the hosting shell supplies;
the engine's own header resolver is only the default. Whatever the resolver
returns is validated; local mode never consults one.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from kenaz_ml.app import AppState, init_cloud_state, mount_engine
from kenaz_ml.config import ServingMode
from kenaz_ml.tenant import LOCAL_TENANT_ID, TenantContext, make_tenant_dependency

STUCK = {"features": {"session_length_sec": 120.0}}


def _cloud_app(resolver=None) -> tuple[FastAPI, AppState]:
    app = FastAPI()
    state = AppState(ServingMode.CLOUD)
    init_cloud_state(state)
    mount_engine(app, state, tenant_resolver=resolver)
    return app, state


def test_the_default_cloud_resolver_is_the_header() -> None:
    app, state = _cloud_app()
    with TestClient(app) as c:
        r = c.post("/predict/stuck", json=STUCK, headers={"X-Tenant-ID": "org-a"})
    assert r.status_code == 200, r.text
    assert state.request_counters == {"org-a": 1}


def test_a_missing_header_is_refused_before_the_engine_runs() -> None:
    app, state = _cloud_app()
    with TestClient(app) as c:
        r = c.post("/predict/stuck", json=STUCK)
    assert r.status_code == 401
    assert state.request_counters == {}


def test_a_shell_resolver_replaces_the_header_entirely() -> None:
    async def from_verified_token(request: Request) -> TenantContext:
        return TenantContext(tenant_id="org-from-token", tier="paid")

    app, state = _cloud_app(from_verified_token)
    with TestClient(app) as c:
        # A spoofed header is ignored: only the resolver vouches for the tenant.
        r = c.post("/predict/stuck", json=STUCK, headers={"X-Tenant-ID": "org-spoofed"})
    assert r.status_code == 200, r.text
    assert state.request_counters == {"org-from-token": 1}


def test_what_a_resolver_returns_is_still_validated() -> None:
    async def sloppy(request: Request) -> TenantContext:
        return TenantContext(tenant_id="Not A Valid Tenant!")

    app, state = _cloud_app(sloppy)
    with TestClient(app) as c:
        r = c.post("/predict/stuck", json=STUCK)
    assert r.status_code == 400
    assert state.request_counters == {}


def test_a_resolver_may_refuse_with_its_own_status() -> None:
    async def not_entitled(request: Request) -> TenantContext:
        raise HTTPException(status_code=403, detail="hosted_inference not granted")

    app, _ = _cloud_app(not_entitled)
    with TestClient(app) as c:
        r = c.post("/predict/stuck", json=STUCK)
    assert r.status_code == 403
    assert r.json()["detail"] == "hosted_inference not granted"


def test_local_mode_never_consults_a_resolver() -> None:
    async def boom(request: Request) -> TenantContext:
        raise AssertionError("local mode must not resolve tenants")

    dependency = make_tenant_dependency(AppState(ServingMode.LOCAL), boom)
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": [], "query_string": b""})
    tenant = asyncio.run(dependency(request))
    assert tenant.tenant_id == LOCAL_TENANT_ID
    assert tenant.is_local


@pytest.mark.parametrize("tenant_id", ["org-a", "org_b", "0123456789abcdef0123456789abcdef"])
def test_uuid_shaped_and_slug_shaped_ids_pass_validation(tenant_id: str) -> None:
    async def resolve(request: Request) -> TenantContext:
        return TenantContext(tenant_id=tenant_id)

    app, state = _cloud_app(resolve)
    with TestClient(app) as c:
        r = c.post("/predict/stuck", json=STUCK)
    assert r.status_code == 200
    assert tenant_id in state.request_counters
