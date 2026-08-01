"""Characterization tests for FilesystemModelLoader (FR-009).

Written against the pre-move import path (`sigil_ml.loader`) so that they are
demonstrably independent of the storage-layer reorganization: the assertions
below describe behaviour that must survive the move to
`sigil_ml.modelstore.loader` byte for byte.

Every test writes its artifacts under ``tmp_path``. The real
``~/.local/share/sigild/`` tree is never read or written -- the one test that
exercises the ``base_dir=None`` default monkeypatches ``config.models_dir``.

Resolution order under test (from the class docstring):
    {models_dir}/{tenant_id}/{model_name}.joblib   tenant-specific
    {models_dir}/{model_name}.joblib               shared fallback

Mutation testing (T006)
-----------------------
Each mutation was applied to a scratch copy of the package (never to repo
source) and the suite re-run to confirm the tests bite. Observed results:

* **M3 -- loader ignores ``tenant_id``.** The tenant-specific probe becomes
  ``self._base_dir / f"{model_name}.joblib"``, i.e. every tenant resolves to
  the shared path. 10 failures:
  ``TestResolutionOrder::test_tenant_specific_artifact_is_loaded``,
  ``::test_tenant_specific_artifact_wins_over_shared``,
  ``::test_resolution_is_per_model_name``,
  ``TestCrossTenantIsolation::test_two_tenants_get_their_own_artifact_by_content``,
  ``::test_isolation_holds_when_a_shared_artifact_also_exists``,
  ``::test_one_tenant_specific_artifact_does_not_leak_to_another_tenant``,
  ``::test_repeated_loads_are_stable_and_independent``,
  ``TestCorruptArtifact::test_a_corrupt_tenant_artifact_does_not_fall_through_to_shared``,
  ``::test_corrupt_artifact_is_logged_as_a_warning``,
  ``TestDefaultBaseDir::test_default_base_dir_comes_from_config``.
* **M4 -- loader raises instead of returning ``None``.** The trailing
  ``return None`` in ``load()`` becomes ``raise FileNotFoundError(...)``.
  6 failures:
  ``TestMissingArtifact::test_returns_none_when_nothing_matches``,
  ``::test_does_not_raise_when_nothing_matches``,
  ``::test_returns_none_when_only_another_tenants_model_exists``,
  ``::test_returns_none_when_only_another_model_name_exists``,
  ``::test_returns_none_when_the_base_directory_does_not_exist``,
  ``TestCrossTenantIsolation::test_one_tenant_specific_artifact_does_not_leak_to_another_tenant``.
  (``TestCorruptArtifact`` survives M4: a corrupt artifact returns ``None``
  from ``_safe_load``'s ``except`` branch, never reaching the fall-through.)

Pre-move baseline (T007)
------------------------
Recorded on the pre-move tree so WP03 can compare without re-deriving it.
Toolchain: Python 3.12.13, joblib 1.5.3, numpy 2.5.1, scikit-learn 1.9.0.

* ``pytest tests/`` before this work package:
  **486 passed, 1 failed, 9 skipped** in ~17s.
  The single failure is pre-existing and unrelated:
  ``tests/test_feature_store_local.py::TestNfr002ServingLatency::
  test_resolution_stays_within_twenty_percent_of_the_direct_extractor[500]``
  -- a Feast-vs-direct-extractor latency ratio that misses its 20% budget on
  this hardware. It fails identically on the unmodified pre-move tree
  (reproduced three consecutive times) and is therefore the baseline, not a
  regression. WP03 should expect it to keep failing.
* ``tests/test_model_cache.py`` + ``tests/test_model_loader.py`` add
  **56 tests** (31 cache, 25 loader), so the post-move expectation is
  **542 passed, 1 failed, 9 skipped**.
* ``python -X importtime -c "import sigil_ml.app"`` cumulative total,
  mean of 5 runs: **1,774,269 us** (min 1,762,105, max 1,782,823).
  NFR-003's 10% budget therefore caps the post-move figure at
  **1,951,696 us**.

FR-007 artifact fixture
-----------------------
``PRE_MOVE_ARTIFACT_B64`` below is a real ``joblib.dump`` output produced by
the *pre-move* tree at this commit. ``TestPreMoveArtifactStillLoads`` decodes
it to ``tmp_path`` and loads it through ``FilesystemModelLoader``; after the
move the same bytes must still load through the relocated loader, which is
what FR-007 asks for.

The payload is numpy + plain containers rather than a fitted estimator on
purpose: ``scikit-learn`` is specified as ``>=1.4`` (unpinned), so freezing an
estimator pickle would turn this into a scikit-learn version test instead of a
module-move test. That loses nothing here -- ``grep`` over every
``joblib.dump`` call site in ``src/`` confirms that no shipped artifact
pickles a ``sigil_ml`` class (they are bare estimators, or dicts of estimators
and numpy arrays), so no artifact carries a module path that this move could
invalidate.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import joblib
import numpy as np
import pytest

from sigil_ml.modelstore import FilesystemModelLoader, ModelLoader

# A genuine joblib artifact serialized by the pre-move tree. Do not regenerate:
# its value is that it predates the move (FR-007).
PRE_MOVE_ARTIFACT_B64 = (
    "gASVDwEAAAAAAAB9lCiMCm1vZGVsX25hbWWUjAVzdHVja5SMCXRlbmFudF9pZJSMCHRlbmFudC1hlIwMY29lZmZp"
    "Y2llbnRzlIwTam9ibGliLm51bXB5X3BpY2tsZZSMEU51bXB5QXJyYXlXcmFwcGVylJOUKYGUfZQojAhzdWJjbGFz"
    "c5SMBW51bXB5lIwHbmRhcnJheZSTlIwFc2hhcGWUSwOFlIwFb3JkZXKUjAFDlIwFZHR5cGWUaAyMBWR0eXBllJOU"
    "jAJmOJSJiIeUUpQoSwOMATyUTk5OSv////9K/////0sAdJRijAphbGxvd19tbWFwlIiMG251bXB5X2FycmF5X2Fs"
    "aWdubWVudF9ieXRlc5RLEHViBf//////AAAAAAAA4D8AAAAAAAD0vwAAAAAAAAhAlVsAAAAAAAAAjA1mZWF0dXJl"
    "X25hbWVzlF2UKIwIaWRsZV9zZWOUjAplZGl0X2NvdW50lIwSc2Vzc2lvbl9sZW5ndGhfc2VjlGWMDXRyYWluZWRf"
    "YXRfbXOUigYAwCzImQF1Lg=="
)


def _write_artifact(path: Path, payload: object) -> Path:
    """Persist ``payload`` as a joblib artifact, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    return path


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """ModelLoader is runtime_checkable; the filesystem loader satisfies it."""

    def test_filesystem_loader_satisfies_the_protocol(self, tmp_path: Path) -> None:
        assert isinstance(FilesystemModelLoader(base_dir=tmp_path), ModelLoader)


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------


class TestResolutionOrder:
    """Tenant-specific path first, shared path second."""

    def test_tenant_specific_artifact_is_loaded(self, tmp_path: Path) -> None:
        _write_artifact(tmp_path / "tenant-a" / "stuck.joblib", {"origin": "tenant-a"})
        loader = FilesystemModelLoader(base_dir=tmp_path)

        assert loader.load("tenant-a", "stuck") == {"origin": "tenant-a"}

    def test_shared_fallback_applies_when_tenant_artifact_is_absent(self, tmp_path: Path) -> None:
        _write_artifact(tmp_path / "stuck.joblib", {"origin": "shared"})
        loader = FilesystemModelLoader(base_dir=tmp_path)

        assert loader.load("tenant-a", "stuck") == {"origin": "shared"}

    def test_tenant_specific_artifact_wins_over_shared(self, tmp_path: Path) -> None:
        """Both exist -- the tenant-specific one must be the one returned."""
        _write_artifact(tmp_path / "stuck.joblib", {"origin": "shared"})
        _write_artifact(tmp_path / "tenant-a" / "stuck.joblib", {"origin": "tenant-a"})
        loader = FilesystemModelLoader(base_dir=tmp_path)

        assert loader.load("tenant-a", "stuck") == {"origin": "tenant-a"}

    def test_shared_fallback_is_shared_by_every_tenant(self, tmp_path: Path) -> None:
        _write_artifact(tmp_path / "quality.joblib", {"origin": "shared"})
        loader = FilesystemModelLoader(base_dir=tmp_path)

        assert loader.load("tenant-a", "quality") == {"origin": "shared"}
        assert loader.load("tenant-b", "quality") == {"origin": "shared"}

    def test_resolution_is_per_model_name(self, tmp_path: Path) -> None:
        """A tenant override of one model does not affect another model."""
        _write_artifact(tmp_path / "stuck.joblib", {"origin": "shared-stuck"})
        _write_artifact(tmp_path / "quality.joblib", {"origin": "shared-quality"})
        _write_artifact(tmp_path / "tenant-a" / "stuck.joblib", {"origin": "tenant-a-stuck"})
        loader = FilesystemModelLoader(base_dir=tmp_path)

        assert loader.load("tenant-a", "stuck") == {"origin": "tenant-a-stuck"}
        assert loader.load("tenant-a", "quality") == {"origin": "shared-quality"}


# ---------------------------------------------------------------------------
# Cross-tenant isolation -- asserted by CONTENT, not by path
# ---------------------------------------------------------------------------


class TestCrossTenantIsolation:
    """Two tenants, same model name, different artifacts: no bleed-through."""

    def test_two_tenants_get_their_own_artifact_by_content(self, tmp_path: Path) -> None:
        _write_artifact(tmp_path / "tenant-a" / "stuck.joblib", {"weights": [1, 2, 3], "owner": "a"})
        _write_artifact(tmp_path / "tenant-b" / "stuck.joblib", {"weights": [9, 8, 7], "owner": "b"})
        loader = FilesystemModelLoader(base_dir=tmp_path)

        for_a = loader.load("tenant-a", "stuck")
        for_b = loader.load("tenant-b", "stuck")

        assert for_a == {"weights": [1, 2, 3], "owner": "a"}
        assert for_b == {"weights": [9, 8, 7], "owner": "b"}
        assert for_a != for_b

    def test_isolation_holds_when_a_shared_artifact_also_exists(self, tmp_path: Path) -> None:
        _write_artifact(tmp_path / "stuck.joblib", {"owner": "shared"})
        _write_artifact(tmp_path / "tenant-a" / "stuck.joblib", {"owner": "a"})
        _write_artifact(tmp_path / "tenant-b" / "stuck.joblib", {"owner": "b"})
        loader = FilesystemModelLoader(base_dir=tmp_path)

        assert loader.load("tenant-a", "stuck") == {"owner": "a"}
        assert loader.load("tenant-b", "stuck") == {"owner": "b"}
        assert loader.load("tenant-c", "stuck") == {"owner": "shared"}

    def test_one_tenant_specific_artifact_does_not_leak_to_another_tenant(self, tmp_path: Path) -> None:
        """Only tenant-a has an artifact; tenant-b must not receive it."""
        _write_artifact(tmp_path / "tenant-a" / "stuck.joblib", {"owner": "a"})
        loader = FilesystemModelLoader(base_dir=tmp_path)

        assert loader.load("tenant-a", "stuck") == {"owner": "a"}
        assert loader.load("tenant-b", "stuck") is None

    def test_repeated_loads_are_stable_and_independent(self, tmp_path: Path) -> None:
        """The loader holds no per-call state; order of access is irrelevant."""
        _write_artifact(tmp_path / "tenant-a" / "stuck.joblib", {"owner": "a"})
        _write_artifact(tmp_path / "tenant-b" / "stuck.joblib", {"owner": "b"})
        loader = FilesystemModelLoader(base_dir=tmp_path)

        observed = [
            loader.load("tenant-b", "stuck"),
            loader.load("tenant-a", "stuck"),
            loader.load("tenant-b", "stuck"),
            loader.load("tenant-a", "stuck"),
        ]

        assert observed == [{"owner": "b"}, {"owner": "a"}, {"owner": "b"}, {"owner": "a"}]


# ---------------------------------------------------------------------------
# Missing artifacts -- None, never an exception
# ---------------------------------------------------------------------------


class TestMissingArtifact:
    """The protocol docstring promises None on miss; callers depend on it."""

    def test_returns_none_when_nothing_matches(self, tmp_path: Path) -> None:
        loader = FilesystemModelLoader(base_dir=tmp_path)

        assert loader.load("tenant-a", "stuck") is None

    def test_does_not_raise_when_nothing_matches(self, tmp_path: Path) -> None:
        loader = FilesystemModelLoader(base_dir=tmp_path)

        try:
            result = loader.load("tenant-a", "stuck")
        except Exception as exc:  # pragma: no cover - failure path
            pytest.fail(f"load() must not raise on a missing model, raised {exc!r}")

        assert result is None

    def test_returns_none_when_only_another_tenants_model_exists(self, tmp_path: Path) -> None:
        _write_artifact(tmp_path / "tenant-a" / "stuck.joblib", {"owner": "a"})
        loader = FilesystemModelLoader(base_dir=tmp_path)

        assert loader.load("tenant-b", "stuck") is None

    def test_returns_none_when_only_another_model_name_exists(self, tmp_path: Path) -> None:
        _write_artifact(tmp_path / "tenant-a" / "quality.joblib", {"owner": "a"})
        loader = FilesystemModelLoader(base_dir=tmp_path)

        assert loader.load("tenant-a", "stuck") is None

    def test_returns_none_when_the_base_directory_does_not_exist(self, tmp_path: Path) -> None:
        loader = FilesystemModelLoader(base_dir=tmp_path / "definitely-absent")

        assert loader.load("tenant-a", "stuck") is None

    def test_a_directory_at_the_artifact_path_is_not_treated_as_a_model(self, tmp_path: Path) -> None:
        """`Path.exists()` is true for directories; joblib.load then fails."""
        (tmp_path / "tenant-a" / "stuck.joblib").mkdir(parents=True)
        loader = FilesystemModelLoader(base_dir=tmp_path)

        assert loader.load("tenant-a", "stuck") is None


# ---------------------------------------------------------------------------
# Corrupt artifacts -- documenting the CURRENT behaviour
# ---------------------------------------------------------------------------


class TestCorruptArtifact:
    """_safe_load() swallows every exception, logs a warning and returns None.

    These tests record what the implementation does today rather than what it
    arguably ought to do; the move must not change either answer.
    """

    def test_corrupt_tenant_artifact_returns_none_rather_than_raising(self, tmp_path: Path) -> None:
        path = tmp_path / "tenant-a" / "stuck.joblib"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"this is not a joblib pickle")
        loader = FilesystemModelLoader(base_dir=tmp_path)

        assert loader.load("tenant-a", "stuck") is None

    def test_corrupt_shared_artifact_also_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "stuck.joblib").write_bytes(b"garbage")
        loader = FilesystemModelLoader(base_dir=tmp_path)

        assert loader.load("tenant-a", "stuck") is None

    def test_empty_artifact_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "tenant-a" / "stuck.joblib"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"")
        loader = FilesystemModelLoader(base_dir=tmp_path)

        assert loader.load("tenant-a", "stuck") is None

    def test_a_corrupt_tenant_artifact_does_not_fall_through_to_shared(self, tmp_path: Path) -> None:
        """Existence, not loadability, decides the branch -- so no fallback."""
        _write_artifact(tmp_path / "stuck.joblib", {"origin": "shared"})
        path = tmp_path / "tenant-a" / "stuck.joblib"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"garbage")
        loader = FilesystemModelLoader(base_dir=tmp_path)

        assert loader.load("tenant-a", "stuck") is None

    def test_corrupt_artifact_is_logged_as_a_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        path = tmp_path / "tenant-a" / "stuck.joblib"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"garbage")
        loader = FilesystemModelLoader(base_dir=tmp_path)

        with caplog.at_level(logging.WARNING):
            loader.load("tenant-a", "stuck")

        assert any("failed to load" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Default base directory
# ---------------------------------------------------------------------------


class TestDefaultBaseDir:
    """base_dir=None defers to config.models_dir() -- lazily, at construction."""

    def test_default_base_dir_comes_from_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from sigil_ml import config

        monkeypatch.setattr(config, "models_dir", lambda: tmp_path)
        _write_artifact(tmp_path / "tenant-a" / "stuck.joblib", {"origin": "from-config"})

        loader = FilesystemModelLoader()

        assert loader.load("tenant-a", "stuck") == {"origin": "from-config"}


# ---------------------------------------------------------------------------
# FR-007 -- a pre-move artifact must still load
# ---------------------------------------------------------------------------


class TestPreMoveArtifactStillLoads:
    """The bytes in PRE_MOVE_ARTIFACT_B64 were serialized before the move."""

    def test_pre_move_artifact_loads_through_the_tenant_path(self, tmp_path: Path) -> None:
        path = tmp_path / "tenant-a" / "stuck.joblib"
        path.parent.mkdir(parents=True)
        path.write_bytes(base64.b64decode(PRE_MOVE_ARTIFACT_B64))
        loader = FilesystemModelLoader(base_dir=tmp_path)

        model = loader.load("tenant-a", "stuck")

        assert model is not None, "a pre-move artifact must still load (FR-007)"
        assert model["model_name"] == "stuck"
        assert model["tenant_id"] == "tenant-a"
        assert model["feature_names"] == ["idle_sec", "edit_count", "session_length_sec"]
        assert model["trained_at_ms"] == 1_760_000_000_000
        np.testing.assert_allclose(model["coefficients"], np.array([0.5, -1.25, 3.0]))

    def test_pre_move_artifact_loads_through_the_shared_path(self, tmp_path: Path) -> None:
        (tmp_path / "stuck.joblib").write_bytes(base64.b64decode(PRE_MOVE_ARTIFACT_B64))
        loader = FilesystemModelLoader(base_dir=tmp_path)

        model = loader.load("tenant-zzz", "stuck")

        assert model is not None
        assert model["model_name"] == "stuck"

    def test_pre_move_artifact_matches_a_freshly_written_one(self, tmp_path: Path) -> None:
        """Round-trip equivalence: frozen bytes == what joblib writes today."""
        frozen = tmp_path / "a" / "stuck.joblib"
        frozen.parent.mkdir(parents=True)
        frozen.write_bytes(base64.b64decode(PRE_MOVE_ARTIFACT_B64))

        fresh_payload = {
            "model_name": "stuck",
            "tenant_id": "tenant-a",
            "coefficients": np.array([0.5, -1.25, 3.0], dtype=np.float64),
            "feature_names": ["idle_sec", "edit_count", "session_length_sec"],
            "trained_at_ms": 1_760_000_000_000,
        }
        fresh = tmp_path / "b" / "stuck.joblib"
        fresh.parent.mkdir(parents=True)
        joblib.dump(fresh_payload, fresh)

        loader = FilesystemModelLoader(base_dir=tmp_path)
        from_frozen = loader.load("a", "stuck")
        from_fresh = loader.load("b", "stuck")

        assert from_frozen is not None and from_fresh is not None
        assert from_frozen.keys() == from_fresh.keys()
        np.testing.assert_allclose(from_frozen["coefficients"], from_fresh["coefficients"])
