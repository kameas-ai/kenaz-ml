"""Tenant context extraction for multi-tenant cloud serving."""

from __future__ import annotations

import logging
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


def make_tenant_dependency(state: AppState):
    """Create a FastAPI dependency that extracts tenant context.

    Uses a closure to capture the application state so the dependency
    can check the serving mode.

    Args:
        state: Application state containing the serving mode.

    Returns:
        An async callable suitable for FastAPI Depends().
    """
    from kenaz_ml.config import ServingMode

    async def get_tenant_context(request: Request) -> TenantContext:
        """Extract tenant context from the request.

        In cloud mode: reads X-Tenant-ID header, returns 401 if missing.
        In local mode: returns sentinel context without checking headers.
        """
        if state.mode == ServingMode.LOCAL:
            return TenantContext(tenant_id=LOCAL_TENANT_ID, tier="local")

        header = tenant_header_name()
        tenant_id = request.headers.get(header)
        if not tenant_id or not tenant_id.strip():
            raise HTTPException(
                status_code=401,
                detail=f"Missing required header '{header}' for cloud mode.",
            )
        tenant_id = tenant_id.strip()

        from kenaz_ml.config import validate_tenant_id

        if not validate_tenant_id(tenant_id):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid tenant ID '{tenant_id}'. "
                    "Must be 1-63 characters of lowercase alphanumeric, hyphens, or underscores."
                ),
            )
        return TenantContext(tenant_id=tenant_id)

    return get_tenant_context
