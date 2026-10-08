"""Tenant context extraction for multi-tenant cloud serving."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request

from kenaz_ml.config import env

if TYPE_CHECKING:
    from kenaz_ml.app import AppState

logger = logging.getLogger(__name__)

# Sentinel for local mode -- not used for model routing.
LOCAL_TENANT_ID = "local"


@dataclass(frozen=True)
class TenantContext:
    """Per-request tenant context extracted from the authenticated request.

    In cloud mode, populated from the X-Tenant-ID header (or configured header).
    In local mode, returns a sentinel with tenant_id="local".
    """

    tenant_id: str
    tier: str = "default"

    @property
    def is_local(self) -> bool:
        """True when running in local mode (no real tenancy)."""
        return self.tenant_id == LOCAL_TENANT_ID


def tenant_header_name() -> str:
    """Return the configured tenant header name."""
    return env("KENAZ_TENANT_HEADER", "X-Tenant-ID") or "X-Tenant-ID"


#: A tenant resolver turns an authenticated request into the tenant it acts
#: for. The engine ships one, :func:`header_tenant_resolver`, which trusts a
#: header and is only safe behind a gateway that sets it; a hosting shell
#: that verifies tokens itself passes its own resolver to
#: :func:`kenaz_ml.app.create_app` or :func:`kenaz_ml.app.mount_engine`, and
#: the engine never sees a tenant it did not vouch for.
TenantResolver = Callable[[Request], Awaitable[TenantContext]]


def header_tenant_resolver(header: str | None = None) -> TenantResolver:
    """The engine's default cloud resolver: the tenant is a request header.

    401 when the header is missing, 400 when its value is not a valid tenant
    id. ``header`` defaults to :func:`tenant_header_name`.
    """

    async def resolve(request: Request) -> TenantContext:
        name = header or tenant_header_name()
        tenant_id = request.headers.get(name)
        if not tenant_id or not tenant_id.strip():
            raise HTTPException(
                status_code=401,
                detail=f"Missing required header '{name}' for cloud mode.",
            )
        return TenantContext(tenant_id=tenant_id.strip())

    return resolve


def make_tenant_dependency(state: AppState, resolver: TenantResolver | None = None):
    """Create a FastAPI dependency that extracts tenant context.

    Local mode always yields the local sentinel and never consults a
    resolver. Cloud mode runs ``resolver`` (default: the header resolver) and
    then validates the tenant id it returned, whoever produced it, so a
    shell's resolver cannot hand the engine an id the stores would mishandle.

    Args:
        state: Application state containing the serving mode.
        resolver: The cloud-mode tenant resolver; ``None`` means the header.

    Returns:
        An async callable suitable for FastAPI Depends().
    """
    from kenaz_ml.config import ServingMode, validate_tenant_id

    resolve = resolver or header_tenant_resolver()

    async def get_tenant_context(request: Request) -> TenantContext:
        if state.mode == ServingMode.LOCAL:
            return TenantContext(tenant_id=LOCAL_TENANT_ID, tier="local")

        tenant = await resolve(request)
        if not validate_tenant_id(tenant.tenant_id):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid tenant ID '{tenant.tenant_id}'. "
                    "Must be 1-63 characters of lowercase alphanumeric, hyphens, or underscores."
                ),
            )
        return tenant

    return get_tenant_context
