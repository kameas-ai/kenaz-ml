"""Verification of the storage-layer reorganization (WP03).

These tests assert the *shape* of the move rather than the behaviour of the
moved code -- behaviour is covered by ``tests/test_model_store.py`` (Stack A)
and ``tests/test_model_cache.py`` + ``tests/test_model_loader.py`` (Stack B).

What is pinned here:

* **FR-004 / SC-007** -- no module survives at its old path. A shim left
  behind would let stale imports keep working and reintroduce the "store"
  ambiguity in a new form (D-004), so absence is asserted, not assumed.
* **FR-010** -- every name in the plan's public surface resolves *from the
  package*, not merely from a submodule. The follow-on
  ``model-registry-and-base-refresh`` mission imports this surface, and D-003
  reserves the right to re-split the submodules underneath it.
* **FR-005** -- ``model_store_factory`` keeps its name. Three planning
  artifacts called it ``create_model_store``; that symbol never existed
  (occurrence map finding F1) and must not be invented now.
* **FR-007** -- a ``.joblib`` artifact serialized by the *pre-move* tree
  loads through the relocated loader and yields identical values.
* **NFR-003** -- the two new re-exporting ``__init__.py`` files do not make
  ``import sigil_ml.app`` materially more expensive.

Measurements taken for WP03 (Python 3.12.13, this hardware)
-----------------------------------------------------------
``python -X importtime -c "import sigil_ml.app"`` cumulative, 5 runs:
1,809,373 / 1,792,680 / 1,780,143 / 1,775,333 / 1,785,166 us,
**mean 1,788,539 us**. WP02's pre-move baseline (recorded in the
``tests/test_model_loader.py`` docstring) is **1,774,269 us**, so the move
costs **+0.80%** against NFR-003's 10% budget -- cap 1,951,696 us. The two
new packages account for 744 us of that total (``sigil_ml.datastore`` 186 us
cumulative, ``sigil_ml.modelstore`` 558 us), of which the re-exporting
``__init__.py`` bodies are 132 us.

Those absolute figures are machine-specific and deliberately *not* asserted:
pinning a microsecond baseline into the suite makes it fail on other
hardware. ``TestImportCostOfTheNewPackages`` asserts the hardware-independent
property the budget actually depends on instead.
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import joblib
import numpy as np
import pytest

import sigil_ml
import sigil_ml.datastore
import sigil_ml.modelstore
from sigil_ml.datastore import DataStore, create_store
from sigil_ml.modelstore import (
    CachedModelStore,
    FilesystemModelLoader,
    LocalModelStore,
    ModelCache,
    ModelLoader,
    ModelStore,
    S3ModelStore,
    create_model_cache,
    model_store_factory,
)
from tests.test_model_loader import PRE_MOVE_ARTIFACT_B64

# Every path the six moved modules used to occupy, plus the package that
# disappeared entirely. quickstart.md §2 lists six; ``sigil_ml.storage`` is
# included as a seventh because D-004 removes the package, not just the
# module inside it.
OLD_MODULE_PATHS = (
    "sigil_ml.store",
    "sigil_ml.store_sqlite",
    "sigil_ml.store_postgres",
    "sigil_ml.storage",
    "sigil_ml.storage.model_store",
    "sigil_ml.loader",
    "sigil_ml.cache",
)

OLD_SOURCE_FILES = (
    "store.py",
    "store_sqlite.py",
    "store_postgres.py",
    "loader.py",
    "cache.py",
    "storage/model_store.py",
    "storage/__init__.py",
)

# plan.md §"Public import surface (FR-010)", verbatim.
DATASTORE_SURFACE = ("DataStore", "create_store")
MODELSTORE_SURFACE = (
    "ModelStore",
    "LocalModelStore",
    "S3ModelStore",
    "CachedModelStore",
    "model_store_factory",
    "ModelLoader",
    "FilesystemModelLoader",
    "ModelCache",
    "create_model_cache",
)


def _package_root() -> Path:
    """The installed ``sigil_ml/`` directory."""
    assert sigil_ml.__file__ is not None
    return Path(sigil_ml.__file__).parent


# ===========================================================================
# T014 / FR-004 / SC-007 -- nothing survives at an old path
# ===========================================================================


class TestOldPathsAreGone:
    """No moved module is importable where it used to live."""

    @pytest.mark.parametrize("old_path", OLD_MODULE_PATHS)
    def test_old_import_path_raises_module_not_found(self, old_path: str) -> None:
        # A previously-imported module lingering in sys.modules would make
        # import_module succeed from cache and hide a surviving shim.
        assert old_path not in sys.modules, f"{old_path} is in sys.modules -- something still imports it"

        importlib.invalidate_caches()
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(old_path)

    @pytest.mark.parametrize("old_path", OLD_MODULE_PATHS)
    def test_old_path_has_no_import_spec(self, old_path: str) -> None:
        """``find_spec`` is the check a shim would have to defeat.

        ``import_module`` alone can be satisfied by a cached module object;
        the finder answering "nothing here" is what proves nothing is
        discoverable on the path. ``find_spec`` signals that two ways --
        ``None`` when the parent package exists, ``ModuleNotFoundError`` when
        it does not -- and both count.
        """
        importlib.invalidate_caches()
        try:
            spec = importlib.util.find_spec(old_path)
        except ModuleNotFoundError:
            return
        assert spec is None, f"{old_path} still has an import spec: {spec}"

    @pytest.mark.parametrize("relative_path", OLD_SOURCE_FILES)
    def test_old_source_file_is_gone_from_the_tree(self, relative_path: str) -> None:
        """The file itself, not just its importability.

        A ``.py`` restored by a bad merge would silently resurrect the old
        path even though the import test above passed at the time it was
        written.
        """
        stale = _package_root() / relative_path
        assert not stale.exists(), f"{stale} still exists"

    def test_the_storage_package_directory_is_gone(self) -> None:
        """D-004: ``sigil_ml/storage/`` disappears entirely, shim included."""
        assert not (_package_root() / "storage").exists()


# ===========================================================================
# T014 / FR-010 -- the public surface resolves from the package
# ===========================================================================


class TestPublicSurfaceResolvesFromThePackage:
    """Each documented name is reachable as a package attribute."""

    @pytest.mark.parametrize("name", DATASTORE_SURFACE)
    def test_datastore_name_is_a_package_attribute(self, name: str) -> None:
        assert hasattr(sigil_ml.datastore, name), f"sigil_ml.datastore has no attribute {name!r}"

    @pytest.mark.parametrize("name", MODELSTORE_SURFACE)
    def test_modelstore_name_is_a_package_attribute(self, name: str) -> None:
        assert hasattr(sigil_ml.modelstore, name), f"sigil_ml.modelstore has no attribute {name!r}"

    def test_the_plan_snippet_imports_verbatim(self) -> None:
        """The exact ``from ... import ...`` block plan.md publishes.

        The module-level imports at the top of this file *are* that snippet,
        so this test would never run if the surface had regressed. It names
        the requirement and checks that each binding is the same object the
        package exports, rather than something shadowed locally.
        """
        bound = {
            "DataStore": DataStore,
            "create_store": create_store,
        }
        for name, obj in bound.items():
            assert getattr(sigil_ml.datastore, name) is obj

        bound = {
            "ModelStore": ModelStore,
            "LocalModelStore": LocalModelStore,
            "S3ModelStore": S3ModelStore,
            "CachedModelStore": CachedModelStore,
            "model_store_factory": model_store_factory,
            "ModelLoader": ModelLoader,
            "FilesystemModelLoader": FilesystemModelLoader,
            "ModelCache": ModelCache,
            "create_model_cache": create_model_cache,
        }
        assert set(bound) == set(MODELSTORE_SURFACE)
        for name, obj in bound.items():
            assert getattr(sigil_ml.modelstore, name) is obj

    @pytest.mark.parametrize(
        ("package", "submodule", "names"),
        [
            ("sigil_ml.datastore", "sigil_ml.datastore.protocol", DATASTORE_SURFACE),
            (
                "sigil_ml.modelstore",
                "sigil_ml.modelstore.stores",
                ("ModelStore", "LocalModelStore", "S3ModelStore", "CachedModelStore", "model_store_factory"),
            ),
            ("sigil_ml.modelstore", "sigil_ml.modelstore.loader", ("ModelLoader", "FilesystemModelLoader")),
            ("sigil_ml.modelstore", "sigil_ml.modelstore.cache", ("ModelCache", "create_model_cache")),
        ],
    )
    def test_package_export_is_the_submodule_object(self, package: str, submodule: str, names: tuple[str, ...]) -> None:
        """The package re-exports, it does not redefine.

        Identity matters because ``isinstance`` against the runtime-checkable
        protocols and ``mock.patch`` targets both depend on there being one
        object, not a package-level copy.
        """
        pkg = importlib.import_module(package)
        sub = importlib.import_module(submodule)
        for name in names:
            assert getattr(pkg, name) is getattr(sub, name), f"{package}.{name} is not {submodule}.{name}"

    def test_submodule_paths_remain_importable(self) -> None:
        """D-003: submodules are not the supported surface, but they work.

        Stated so that a later decision to hide them is a deliberate change
        rather than an accident.
        """
        for submodule in (
            "sigil_ml.datastore.protocol",
            "sigil_ml.datastore.sqlite",
            "sigil_ml.modelstore.stores",
            "sigil_ml.modelstore.loader",
            "sigil_ml.modelstore.cache",
        ):
            assert importlib.import_module(submodule) is not None


class TestFactoryNameWasNotInvented:
    """FR-005 / occurrence-map finding F1.

    plan.md:92, quickstart.md:49 and WP02-tests-then-move.md:169 all instruct
    exporting ``create_model_store``. No such symbol has ever existed -- the
    real factory is ``model_store_factory``. Renaming it to match the prose
    would be the behaviour change C-001 forbids, so the *absence* of the
    invented name is pinned alongside the presence of the real one.
    """

    def test_model_store_factory_is_exported(self) -> None:
        assert callable(model_store_factory)
        assert "model_store_factory" in sigil_ml.modelstore.__all__

    def test_create_model_store_does_not_exist(self) -> None:
        assert not hasattr(sigil_ml.modelstore, "create_model_store")
        assert "create_model_store" not in sigil_ml.modelstore.__all__


# ===========================================================================
# T014 -- __all__ is declared and matches what the package exports
# ===========================================================================


class TestAllMatchesTheImportableSurface:
    """``__all__`` is the contract, so it must be neither short nor stale."""

    @pytest.mark.parametrize("package", ["sigil_ml.datastore", "sigil_ml.modelstore"])
    def test_all_is_declared(self, package: str) -> None:
        pkg = importlib.import_module(package)
        assert hasattr(pkg, "__all__"), f"{package} declares no __all__"
        assert isinstance(pkg.__all__, list)
        assert pkg.__all__, f"{package}.__all__ is empty"

    @pytest.mark.parametrize("package", ["sigil_ml.datastore", "sigil_ml.modelstore"])
    def test_every_name_in_all_resolves(self, package: str) -> None:
        pkg = importlib.import_module(package)
        missing = [name for name in pkg.__all__ if not hasattr(pkg, name)]
        assert not missing, f"{package}.__all__ names nothing: {missing}"

    @pytest.mark.parametrize(
        ("package", "expected"),
        [
            ("sigil_ml.datastore", set(DATASTORE_SURFACE)),
            ("sigil_ml.modelstore", set(MODELSTORE_SURFACE)),
        ],
    )
    def test_all_equals_the_documented_surface(self, package: str, expected: set[str]) -> None:
        pkg = importlib.import_module(package)
        assert set(pkg.__all__) == expected

    @pytest.mark.parametrize("package", ["sigil_ml.datastore", "sigil_ml.modelstore"])
    def test_all_omits_nothing_the_package_exports(self, package: str) -> None:
        """No public non-submodule name escapes ``__all__``.

        Catches the failure where a name is added to the package body and the
        list is not updated, which would make it exported by attribute access
        but invisible to ``from ... import *`` and to readers.
        """
        pkg = importlib.import_module(package)
        exported = {
            name for name, value in vars(pkg).items() if not name.startswith("_") and not isinstance(value, ModuleType)
        }
        assert exported == set(pkg.__all__), f"{package}: attribute surface and __all__ disagree"


# ===========================================================================
# T016 / FR-007 -- an artifact written before the move still loads
# ===========================================================================


class TestPreMoveArtifactLoadsThroughTheNewPaths:
    """``PRE_MOVE_ARTIFACT_B64`` predates the move (WP02 Part 1, T007).

    It is loaded here through both model-artifact entry points -- the
    object-level ``FilesystemModelLoader`` and the byte-level
    ``LocalModelStore`` -- because FR-007 is about artifacts on disk, and the
    two stacks reach them differently.
    """

    EXPECTED_COEFFICIENTS = (0.5, -1.25, 3.0)
    FIXED_INPUT = (1.0, 2.0, 3.0)
    EXPECTED_SCORE = 7.0  # 0.5*1 + -1.25*2 + 3.0*3

    @staticmethod
    def _materialize(directory: Path, name: str = "stuck") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.joblib"
        path.write_bytes(base64.b64decode(PRE_MOVE_ARTIFACT_B64))
        return path

    def test_loader_returns_an_object_of_the_original_type(self, tmp_path: Path) -> None:
        self._materialize(tmp_path)
        loaded = FilesystemModelLoader(base_dir=tmp_path).load("tenant-a", "stuck")

        assert isinstance(loaded, dict)
        assert loaded["model_name"] == "stuck"
        assert loaded["tenant_id"] == "tenant-a"
        assert isinstance(loaded["coefficients"], np.ndarray)
        assert loaded["coefficients"].dtype == np.dtype("float64")

    def test_loader_reproduces_the_same_prediction_on_a_fixed_input(self, tmp_path: Path) -> None:
        """The point of FR-007: same bytes in, same numbers out."""
        self._materialize(tmp_path)
        loaded = FilesystemModelLoader(base_dir=tmp_path).load("tenant-a", "stuck")

        assert loaded is not None
        assert tuple(loaded["coefficients"]) == self.EXPECTED_COEFFICIENTS
        score = float(np.dot(loaded["coefficients"], np.array(self.FIXED_INPUT)))
        assert score == pytest.approx(self.EXPECTED_SCORE)

    def test_model_store_returns_the_original_bytes(self, tmp_path: Path) -> None:
        self._materialize(tmp_path)
        raw = LocalModelStore(base_dir=tmp_path).load("stuck")

        assert raw == base64.b64decode(PRE_MOVE_ARTIFACT_B64)

    def test_bytes_from_the_model_store_deserialize_identically(self, tmp_path: Path) -> None:
        """The byte-level stack reaches the same object as the object-level one."""
        artifact = self._materialize(tmp_path)
        raw = LocalModelStore(base_dir=tmp_path).load("stuck")
        assert raw is not None

        via_store = joblib.load(artifact)
        loaded = FilesystemModelLoader(base_dir=tmp_path).load("tenant-a", "stuck")

        assert loaded is not None
        assert via_store["feature_names"] == loaded["feature_names"]
        assert via_store["trained_at_ms"] == loaded["trained_at_ms"]
        np.testing.assert_array_equal(via_store["coefficients"], loaded["coefficients"])

    def test_the_artifact_carries_no_sigil_ml_module_path(self, tmp_path: Path) -> None:
        """A pickle naming a moved module would break on load, not silently.

        ``grep`` over every ``joblib.dump`` call site in ``src/`` says no
        shipped artifact pickles a ``sigil_ml`` class. This asserts it of the
        fixture so the FR-007 guarantee rests on evidence rather than on that
        grep staying true.
        """
        self._materialize(tmp_path)
        blob = base64.b64decode(PRE_MOVE_ARTIFACT_B64)
        assert b"sigil_ml" not in blob


# ===========================================================================
# T016 / NFR-003 -- the re-export packages are not an import-time tax
# ===========================================================================


class TestImportCostOfTheNewPackages:
    """The risk NFR-003 names, asserted without pinning a wall-clock number.

    NFR-003's 10% budget is measured against a pre-move baseline that only
    existed on WP02's tree, on one machine. Rather than freeze that number
    into the suite -- where it would fail on any slower host -- this measures
    the share of ``import sigil_ml.app`` that the two new packages account
    for. That share is what two extra ``__init__.py`` files could plausibly
    inflate, and it is hardware-independent.
    """

    NEW_PACKAGES = ("sigil_ml.datastore", "sigil_ml.modelstore")
    BUDGET_SHARE = 0.01  # 1% of total import time; measured at 0.04%

    @staticmethod
    def _importtime_cumulative() -> dict[str, int]:
        """Map module name -> cumulative microseconds from ``-X importtime``."""
        proc = subprocess.run(
            [sys.executable, "-X", "importtime", "-c", "import sigil_ml.app"],
            capture_output=True,
            text=True,
            check=True,
        )
        cumulative: dict[str, int] = {}
        for line in proc.stderr.splitlines():
            if not line.startswith("import time:") or "cumulative" in line:
                continue
            _, _, payload = line.partition("import time:")
            parts = [field.strip() for field in payload.split("|")]
            if len(parts) != 3:
                continue
            try:
                cumulative[parts[2]] = int(parts[1])
            except ValueError:
                continue
        return cumulative

    def test_the_new_packages_are_a_negligible_share_of_import_time(self) -> None:
        cumulative = self._importtime_cumulative()

        assert "sigil_ml.app" in cumulative, "importtime output did not name sigil_ml.app"
        total = cumulative["sigil_ml.app"]
        assert total > 0

        measured = {name: cumulative.get(name, 0) for name in self.NEW_PACKAGES}
        assert all(value > 0 for value in measured.values()), (
            f"expected both new packages on the import path of sigil_ml.app, got {measured}"
        )

        share = sum(measured.values()) / total
        assert share < self.BUDGET_SHARE, (
            f"new storage packages cost {share:.2%} of `import sigil_ml.app` "
            f"({measured}, total {total} us) -- NFR-003 is at risk; consider lazy re-exports"
        )

    def test_the_postgres_backend_is_not_imported_eagerly(self) -> None:
        """``psycopg2`` is a cloud-only dependency (see datastore/__init__).

        If the package re-exported ``PostgresStore``, importing
        ``sigil_ml.app`` would drag the optional driver into the local
        deployment and into the frozen bundle's signable surface.
        """
        cumulative = self._importtime_cumulative()
        assert "sigil_ml.datastore.postgres" not in cumulative
