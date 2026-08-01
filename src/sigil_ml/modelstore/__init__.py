"""Model-artifact layer for kameas-ml.

Byte-level persistence (`ModelStore`), object-level loading (`ModelLoader`)
and the in-memory serving cache (`ModelCache`). Import from this package, not
from its submodules -- the submodule split is an implementation detail that
may change again (D-003, FR-010).

Note the factory is `model_store_factory`, not `create_model_store`.
"""

from sigil_ml.modelstore.cache import ModelCache, create_model_cache
from sigil_ml.modelstore.loader import FilesystemModelLoader, ModelLoader
from sigil_ml.modelstore.stores import (
    CachedModelStore,
    LocalModelStore,
    ModelStore,
    S3ModelStore,
    model_store_factory,
)

__all__ = [
    "CachedModelStore",
    "FilesystemModelLoader",
    "LocalModelStore",
    "ModelCache",
    "ModelLoader",
    "ModelStore",
    "S3ModelStore",
    "create_model_cache",
    "model_store_factory",
]
