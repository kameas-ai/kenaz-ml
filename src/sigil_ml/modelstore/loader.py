"""Model loading interface for pluggable storage backends.

Defines the ModelLoader protocol that storage backends must implement.
Feature 003 (Model Storage Abstraction) will provide an S3 implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ModelLoader(Protocol):
    """Protocol for loading model objects from a storage backend.

    Implementations must:
    - Handle tenant-specific model resolution.
    - Return None when no model exists (not raise exceptions).
    - Be thread-safe (may be called concurrently).
    """

    def load(self, tenant_id: str, model_name: str) -> Any | None:
        """Load a model for the given tenant and model name.

        Args:
            tenant_id: Tenant identifier.
            model_name: One of "stuck", "suggest", "workflow",
                        "duration", "activity", "quality".

        Returns:
            The loaded model object, or None if not found.
        """
        ...


class FilesystemModelLoader:
    """Loads model weights from the local filesystem.

    Directory layout:
        {models_dir}/{tenant_id}/{model_name}.joblib  (tenant-specific)
        {models_dir}/{model_name}.joblib               (shared fallback)
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize with the base directory for model weights.

        Args:
            base_dir: Root directory. Defaults to config.models_dir().
        """
        if base_dir is None:
            from sigil_ml import config

            base_dir = config.models_dir()
        self._base_dir = base_dir

    def load(self, tenant_id: str, model_name: str) -> Any | None:
        """Load a model from the filesystem.

        Tries tenant-specific path first, then shared path.
        Returns None if neither exists or if loading fails.
        """
        # Tenant-specific path
        tenant_path = self._base_dir / tenant_id / f"{model_name}.joblib"
        if tenant_path.exists():
            return self._safe_load(tenant_path, tenant_id, model_name)

        # Shared model fallback (no tenant directory)
        shared_path = self._base_dir / f"{model_name}.joblib"
        if shared_path.exists():
            logger.info(
                "loader: using shared model for %s/%s",
                tenant_id,
                model_name,
            )
            return self._safe_load(shared_path, tenant_id, model_name)

        logger.debug("loader: no model found for %s/%s", tenant_id, model_name)
        return None

    def _safe_load(self, path: Path, tenant_id: str, model_name: str) -> Any | None:
        """Load a joblib file with error handling."""
        import joblib

        try:
            model = joblib.load(path)
            logger.info("loader: loaded %s/%s from %s", tenant_id, model_name, path)
            return model
        except Exception:
            logger.warning(
                "loader: failed to load %s/%s from %s",
                tenant_id,
                model_name,
                path,
                exc_info=True,
            )
            return None
