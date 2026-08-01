"""Feast feature store integration for ``sigil_ml``.

This package registers, stores, and retrieves features. It does **not** compute
them — ``sigil_ml.features`` remains the sole authority for feature arithmetic
(plan.md D-002).
"""

from sigil_ml.feature_store.definitions import (
    DURATION_TTL,
    ENTITIES,
    FEATURE_SERVICES,
    FEATURE_VIEWS,
    LOCAL_FEATURE_VIEWS,
    REGISTERED_FEATURE_NAMES,
    STUCK_TTL,
    duration_feature_view,
    duration_features,
    duration_service,
    node,
    stuck_feature_view,
    stuck_features,
    stuck_service,
    task,
    tenant,
)

__all__ = [
    "DURATION_TTL",
    "ENTITIES",
    "FEATURE_SERVICES",
    "FEATURE_VIEWS",
    "LOCAL_FEATURE_VIEWS",
    "REGISTERED_FEATURE_NAMES",
    "STUCK_TTL",
    "duration_feature_view",
    "duration_features",
    "duration_service",
    "node",
    "stuck_feature_view",
    "stuck_features",
    "stuck_service",
    "task",
    "tenant",
]
