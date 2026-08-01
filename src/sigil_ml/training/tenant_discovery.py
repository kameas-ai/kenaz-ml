"""Tenant discovery for cloud training pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sigil_ml.datastore import DataStore

logger = logging.getLogger(__name__)


def discover_eligible_tenants(data_store: DataStore) -> list[str]:
    """Discover all tenant IDs with synced data.

    Returns all tenant IDs from the DataStore. Eligibility filtering
    (threshold, interval) is handled by train_tenant() for each tenant.

    Returns:
        List of tenant ID strings.
    """
    tenants = data_store.list_tenants()
    logger.info("Discovered %d tenants with synced data", len(tenants))
    return tenants


def discover_opted_in_tenants(data_store: DataStore) -> list[str]:
    """Discover tenants that have opted in to aggregate data pooling.

    Returns only tenant IDs where data_pooling_opted_in flag is True.
    The query is always fresh (not cached) to respect opt-out changes.

    Returns:
        List of opted-in tenant ID strings.
    """
    tenants = data_store.list_opted_in_tenants()
    logger.info("Found %d opted-in tenants for aggregate training", len(tenants))
    return tenants
