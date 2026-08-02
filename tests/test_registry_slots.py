"""Tests for two-slot resolution: local, then base, then cold start.

The first class is the important one. **No base models exist.** None have ever
been built, so every install in the field today resolves every model with an
empty base slot and an often-empty local slot. That is the default case, not an
edge case, and if it regresses every install breaks and nothing else in this
mission matters. So it is asserted first, and asserted as *behavior* — the
predictor stays untrained and returns its documented heuristic, the log stays
free of anything that would alarm a user — rather than as the mere absence of a
traceback.

The rest covers the resolution order, the six unusable conditions that must fall
through rather than raise (FR-017), the read-only-ness of the base slot
(FR-015, C-004), and both distribution-form branches of
``config.base_models_dir()``, which the packaged build never exercises under
test and would otherwise ship wrong undetected.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import sys
from pathlib import Path

import joblib
import pytest

from kenaz_ml import config
from kenaz_ml.modelstore.registry import (
    CHECK_CONTRACT,
    CHECK_INTEGRITY,
    CHECK_MANIFEST,
    CHECK_RUNTIME,
    FeatureContract,
    Manifest,
    Provenance,
    Runtime,
    Training,
    local_feature_contract,
    running_sklearn_version,
    write_manifest,
)
from kenaz_ml.modelstore.registry import slots as slots_module
from kenaz_ml.modelstore.registry.slots import (
    CHECK_SLOT,
    REASON_ARTIFACT_NOT_FOUND,
    REASON_SLOT_EMPTY,
    SLOT_BASE,
    SLOT_COLD_START,
    SLOT_LOCAL,
    Resolution,
    artifact_path,
    base_slot_dir,
    local_slot_dir,
    manifest_path,
    resolve_model,
    slot_is_empty,
)

SLOTS_LOGGER = "kenaz_ml.modelstore.registry.slots"

#: The full roster the service serves. Only ``stuck`` and ``duration`` have a
#: registered Feast feature service; the rest are deliberately included so the
#: default-state assertions cover them too.
MODEL_ROSTER = (
    "stuck",
    "duration",
    "suggest",
    "quality",
    "profile",
    "workflow",
    "activity",
    "fleet_focus",
    "fleet_meeting",
    "fleet_onboarding",
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _contract(names: tuple[str, ...] = ("a", "b", "c")) -> FeatureContract:
    """A self-consistent contract, passed explicitly so tests stay hermetic.

    Resolution is what is under test here, not contract derivation; the tests
    that exercise the real Feast-sourced contract are grouped separately.
    """
    return FeatureContract(
        service="test-service",
        service_version="00000000cafe",
        names=names,
        dtypes=tuple("float64" for _ in names),
    )


def _install(
    slot_dir: Path,
    model_name: str,
    *,
    payload: object,
    contract: FeatureContract | None = None,
    sklearn_version: str | None = None,
    with_artifact: bool = True,
    with_manifest: bool = True,
    tamper: bool = False,
    corrupt_manifest: bool = False,
) -> Path:
    """Place an artifact/manifest pair in a slot, optionally broken in one way.

    Returns the artifact path. The digest is computed over the bytes actually
    written, so the pair is valid unless a keyword asks for it not to be.
    """
    slot_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_path(slot_dir, model_name)
    manifest_file = manifest_path(slot_dir, model_name)

    digest = ""
    if with_artifact:
        joblib.dump(payload, artifact)
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    if with_manifest:
        if corrupt_manifest:
            manifest_file.write_text("{ this is not json", encoding="utf-8")
        else:
            write_manifest(
                manifest_file,
                Manifest(
                    name=model_name,
                    version="1",
                    artifact_sha256=digest or "0" * 64,
                    created_at=1753900000000,
                    provenance=Provenance(training_source="base"),
                    runtime=Runtime(
                        estimator="Dict",
                        sklearn_version=sklearn_version or running_sklearn_version() or "1.5.2",
                        python_version="3.12",
                    ),
                    feature_contract=contract if contract is not None else _contract(),
                    training=Training(n_samples=10),
                ),
            )

    if tamper and with_artifact:
        # Flip the payload after the digest was recorded. The manifest now
        # describes bytes that are no longer on disk.
        joblib.dump({"tampered": True, "original": payload}, artifact)

    return artifact


@pytest.fixture
def empty_slots(tmp_path: Path) -> tuple[Path, Path]:
    """A local and a base directory, both absent from disk.

    Absent rather than empty on purpose: the base slot does not exist at all in
    a real install today, and a resolver that assumed the directory existed
    would pass against an empty directory and fail in the field.
    """
    return tmp_path / "ml-models", tmp_path / "ml-base"


# ---------------------------------------------------------------------------
# T009 — the state every install is actually in (SC-008)
# ---------------------------------------------------------------------------


class TestNoBaseModelsDefaultState:
    """Empty base slot + empty local slot. Production, today, everywhere."""

    def test_reaches_cold_start(self, empty_slots: tuple[Path, Path]) -> None:
        local, base = empty_slots

        resolution = resolve_model("stuck", local_dir=local, base_dir=base)

        assert resolution.slot == SLOT_COLD_START
        assert resolution.cold_start is True
        assert resolution.served is False
        assert resolution.model is None
        assert resolution.manifest is None
        assert resolution.artifact is None

    def test_cold_start_is_reached_for_every_model_on_the_roster(self, empty_slots: tuple[Path, Path]) -> None:
        local, base = empty_slots

        for name in MODEL_ROSTER:
            resolution = resolve_model(name, local_dir=local, base_dir=base)
            assert resolution.cold_start is True, f"{name} did not reach cold start"
            assert resolution.model is None, f"{name} produced a model from nowhere"

    def test_both_slots_report_empty_rather_than_broken(self, empty_slots: tuple[Path, Path]) -> None:
        """The refusals are still recorded — quiet is not the same as silent."""
        local, base = empty_slots

        resolution = resolve_model("stuck", local_dir=local, base_dir=base)

        assert [r.slot for r in resolution.refusals] == [SLOT_LOCAL, SLOT_BASE]
        assert resolution.reasons == (REASON_SLOT_EMPTY, REASON_SLOT_EMPTY)
        assert all(r.check == CHECK_SLOT for r in resolution.refusals)

    def test_nothing_at_warning_or_above_is_logged(
        self, empty_slots: tuple[Path, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        """The default state must not put a permanent warning in front of users.

        A warning nobody can act on — there is no base model to install — trains
        people to ignore the level that a genuinely broken release needs.
        """
        local, base = empty_slots

        with caplog.at_level(logging.DEBUG, logger=SLOTS_LOGGER):
            resolve_model("stuck", local_dir=local, base_dir=base)

        alarming = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert alarming == [], f"the default no-models state logged {[r.getMessage() for r in alarming]}"

    def test_the_empty_slots_are_still_visible_at_debug(
        self, empty_slots: tuple[Path, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Quiet, but diagnosable: the reason is in the log at DEBUG."""
        local, base = empty_slots

        with caplog.at_level(logging.DEBUG, logger=SLOTS_LOGGER):
            resolve_model("stuck", local_dir=local, base_dir=base)

        messages = " ".join(r.getMessage() for r in caplog.records)
        assert REASON_SLOT_EMPTY in messages
        assert SLOT_BASE in messages
        assert "stuck" in messages

    def test_no_missing_file_exception_when_neither_directory_exists(self, empty_slots: tuple[Path, Path]) -> None:
        local, base = empty_slots
        assert not local.exists()
        assert not base.exists()

        resolve_model("stuck", local_dir=local, base_dir=base)  # must not raise

    def test_resolution_does_not_create_the_missing_directories(self, empty_slots: tuple[Path, Path]) -> None:
        local, base = empty_slots

        resolve_model("stuck", local_dir=local, base_dir=base)

        assert not base.exists(), "resolution created the read-only base slot"
        assert not local.exists(), "resolution created the local slot as a side effect of reading"

    def test_the_real_shipped_base_slot_resolves_to_cold_start(self, tmp_path: Path) -> None:
        """The genuine production path, not a temp-directory stand-in.

        Uses the real ``config.base_models_dir()`` — which ships empty today —
        with only the local slot redirected away from the user's home.
        """
        assert not (base_slot_dir() / "stuck.joblib").exists(), (
            "a base artifact has appeared in the distribution; this test's premise, "
            "and the default-state assumptions around it, need revisiting"
        )

        resolution = resolve_model("stuck", local_dir=tmp_path / "empty")

        assert resolution.cold_start is True

    def test_predictor_cold_start_behavior_is_unchanged(self, tmp_path: Path) -> None:
        """The behavior cold start actually has to preserve.

        Resolution reporting ``cold_start`` is only meaningful if the thing on
        the other side of it still works. An untrained ``StuckPredictor``
        answers 0.5/weak, and that is what the service returns today for every
        install with no model on disk.
        """
        from kenaz_ml.models.stuck import StuckPredictor

        class _EmptyStore:
            def load(self, name: str) -> bytes | None:
                return None

            def save(self, name: str, data: bytes) -> None:  # pragma: no cover - never called
                raise AssertionError("cold start must not write")

        resolution = resolve_model("stuck", local_dir=tmp_path / "ml-models", base_dir=tmp_path / "ml-base")
        predictor = StuckPredictor(model_store=_EmptyStore())

        assert resolution.cold_start is True
        assert predictor.is_trained is False
        assert predictor.predict({"test_failure_count": 3.0}) == {"probability": 0.5, "confidence": "weak"}


# ---------------------------------------------------------------------------
# T007 — resolution order
# ---------------------------------------------------------------------------


class TestResolutionOrder:
    def test_local_is_preferred_when_usable(self, empty_slots: tuple[Path, Path]) -> None:
        local, base = empty_slots
        contract = _contract()
        _install(local, "stuck", payload={"from": "local"}, contract=contract)
        _install(base, "stuck", payload={"from": "base"}, contract=contract)

        resolution = resolve_model("stuck", local_dir=local, base_dir=base, expected_contract=contract)

        assert resolution.slot == SLOT_LOCAL
        assert resolution.model == {"from": "local"}
        assert resolution.refusals == (), "the base slot should not even be consulted"

    def test_base_serves_when_local_is_absent(self, empty_slots: tuple[Path, Path]) -> None:
        local, base = empty_slots
        contract = _contract()
        _install(base, "stuck", payload={"from": "base"}, contract=contract)

        resolution = resolve_model("stuck", local_dir=local, base_dir=base, expected_contract=contract)

        assert resolution.slot == SLOT_BASE
        assert resolution.model == {"from": "base"}
        assert resolution.reasons == (REASON_SLOT_EMPTY,)

    def test_base_serves_when_local_is_present_but_unusable(self, empty_slots: tuple[Path, Path]) -> None:
        local, base = empty_slots
        contract = _contract()
        _install(local, "stuck", payload={"from": "local"}, contract=contract, tamper=True)
        _install(base, "stuck", payload={"from": "base"}, contract=contract)

        resolution = resolve_model("stuck", local_dir=local, base_dir=base, expected_contract=contract)

        assert resolution.slot == SLOT_BASE
        assert resolution.model == {"from": "base"}
        assert resolution.refusals[0].slot == SLOT_LOCAL
        assert resolution.refusals[0].check == CHECK_INTEGRITY

    def test_local_only_serves_local(self, empty_slots: tuple[Path, Path]) -> None:
        local, base = empty_slots
        contract = _contract()
        _install(local, "stuck", payload={"from": "local"}, contract=contract)

        resolution = resolve_model("stuck", local_dir=local, base_dir=base, expected_contract=contract)

        assert resolution.slot == SLOT_LOCAL
        assert resolution.served is True

    def test_neither_usable_falls_all_the_way_through(self, empty_slots: tuple[Path, Path]) -> None:
        local, base = empty_slots
        contract = _contract()
        _install(local, "stuck", payload={"from": "local"}, contract=contract, tamper=True)
        _install(base, "stuck", payload={"from": "base"}, contract=contract, tamper=True)

        resolution = resolve_model("stuck", local_dir=local, base_dir=base, expected_contract=contract)

        assert resolution.slot == SLOT_COLD_START
        assert [r.slot for r in resolution.refusals] == [SLOT_LOCAL, SLOT_BASE]

    def test_the_answering_slot_and_its_provenance_are_reported(self, empty_slots: tuple[Path, Path]) -> None:
        local, base = empty_slots
        contract = _contract()
        artifact = _install(base, "duration", payload={"from": "base"}, contract=contract)

        resolution = resolve_model("duration", local_dir=local, base_dir=base, expected_contract=contract)

        assert resolution.name == "duration"
        assert resolution.slot == SLOT_BASE
        assert resolution.artifact == artifact
        assert resolution.manifest is not None
        assert resolution.manifest.name == "duration"
        assert resolution.manifest.provenance.is_base is True

    def test_serving_is_logged_with_the_slot(
        self, empty_slots: tuple[Path, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        local, base = empty_slots
        contract = _contract()
        _install(local, "stuck", payload={"from": "local"}, contract=contract)

        with caplog.at_level(logging.INFO, logger=SLOTS_LOGGER):
            resolve_model("stuck", local_dir=local, base_dir=base, expected_contract=contract)

        assert any("serving" in r.getMessage() and SLOT_LOCAL in r.getMessage() for r in caplog.records)


class TestSlotHelpers:
    def test_paths_follow_the_flat_layout(self, tmp_path: Path) -> None:
        assert artifact_path(tmp_path, "stuck") == tmp_path / "stuck.joblib"
        assert manifest_path(tmp_path, "stuck") == tmp_path / "stuck.json"

    def test_slot_is_empty_distinguishes_absent_from_half_populated(self, tmp_path: Path) -> None:
        assert slot_is_empty(tmp_path, "stuck") is True

        _install(tmp_path, "stuck", payload={"x": 1}, with_manifest=False)

        assert slot_is_empty(tmp_path, "stuck") is False

    def test_local_slot_is_tenant_scopable_and_base_is_not(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "models_dir", lambda: Path("/models"))

        assert local_slot_dir() == Path("/models")
        assert local_slot_dir("acme") == Path("/models/acme")
        assert base_slot_dir() == config.base_models_dir()


# ---------------------------------------------------------------------------
# T008 — unusable-artifact fallthrough
# ---------------------------------------------------------------------------


def _refuse(
    slot_dir: Path,
    other_dir: Path,
    slot: str,
    caplog: pytest.LogCaptureFixture,
    **install_kwargs: bool,
) -> tuple[Resolution, list[logging.LogRecord]]:
    """Install a broken pair in one slot, resolve, and return result plus records."""
    contract = _contract()
    _install(slot_dir, "stuck", payload={"from": slot}, contract=contract, **install_kwargs)
    local, base = (slot_dir, other_dir) if slot == SLOT_LOCAL else (other_dir, slot_dir)
    with caplog.at_level(logging.DEBUG, logger=SLOTS_LOGGER):
        resolution = resolve_model("stuck", local_dir=local, base_dir=base, expected_contract=contract)
    return resolution, list(caplog.records)


class TestUnusableFallthrough:
    """Each of the six conditions falls through, and says why."""

    def test_missing_artifact_falls_through(
        self, empty_slots: tuple[Path, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        local, base = empty_slots
        resolution, records = _refuse(local, base, SLOT_LOCAL, caplog, with_artifact=False)

        assert resolution.slot == SLOT_COLD_START
        assert resolution.reasons[0] == REASON_ARTIFACT_NOT_FOUND
        assert any("stuck.joblib" in r.getMessage() for r in records)

    def test_missing_manifest_falls_through(
        self, empty_slots: tuple[Path, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        """A pre-registry artifact lands here: found, but with no digest to verify."""
        local, base = empty_slots
        resolution, records = _refuse(local, base, SLOT_LOCAL, caplog, with_manifest=False)

        assert resolution.slot == SLOT_COLD_START
        assert resolution.refusals[0].check == CHECK_MANIFEST
        assert resolution.reasons[0] == "not_found"

    def test_corrupt_manifest_falls_through(
        self, empty_slots: tuple[Path, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        local, base = empty_slots
        resolution, records = _refuse(local, base, SLOT_LOCAL, caplog, corrupt_manifest=True)

        assert resolution.slot == SLOT_COLD_START
        assert resolution.refusals[0].check == CHECK_MANIFEST
        assert resolution.reasons[0] == "unparseable"

    def test_integrity_failure_falls_through(
        self, empty_slots: tuple[Path, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        local, base = empty_slots
        resolution, records = _refuse(local, base, SLOT_LOCAL, caplog, tamper=True)

        assert resolution.slot == SLOT_COLD_START
        assert resolution.refusals[0].check == CHECK_INTEGRITY
        assert any("stuck" in r.getMessage() for r in records)

    def test_contract_mismatch_falls_through(self, empty_slots: tuple[Path, Path]) -> None:
        local, base = empty_slots
        recorded = _contract(("a", "b", "c"))
        _install(local, "stuck", payload={"from": "local"}, contract=recorded)

        # Same names, different order — the case a set comparison would accept.
        resolution = resolve_model(
            "stuck", local_dir=local, base_dir=base, expected_contract=_contract(("c", "b", "a"))
        )

        assert resolution.slot == SLOT_COLD_START
        assert resolution.refusals[0].check == CHECK_CONTRACT

    def test_runtime_mismatch_falls_through(self, empty_slots: tuple[Path, Path]) -> None:
        local, base = empty_slots
        contract = _contract()
        _install(local, "stuck", payload={"from": "local"}, contract=contract, sklearn_version="0.1.0")

        resolution = resolve_model("stuck", local_dir=local, base_dir=base, expected_contract=contract)

        assert resolution.slot == SLOT_COLD_START
        assert resolution.refusals[0].check == CHECK_RUNTIME

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"with_artifact": False},
            {"with_manifest": False},
            {"corrupt_manifest": True},
            {"tamper": True},
        ],
    )
    def test_every_refusal_names_model_slot_reason_and_detail(
        self, empty_slots: tuple[Path, Path], caplog: pytest.LogCaptureFixture, kwargs: dict[str, bool]
    ) -> None:
        """T008's logging bar: diagnosable from the line alone."""
        local, base = empty_slots
        resolution, records = _refuse(local, base, SLOT_LOCAL, caplog, **kwargs)

        refusal = resolution.refusals[0]
        assert refusal.model_name == "stuck"
        assert refusal.slot == SLOT_LOCAL
        assert refusal.reason
        assert refusal.detail, "a refusal with no detail is not actionable"

        line = next(r for r in records if r.levelno >= logging.WARNING).getMessage()
        assert "stuck" in line
        assert SLOT_LOCAL in line
        assert refusal.reason in line
        assert refusal.detail in line

    def test_a_base_refusal_is_louder_than_the_same_local_refusal(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A bad local model self-heals next training run; a bad base one cannot."""
        contract = _contract()

        local_only = tmp_path / "a-local"
        _install(local_only, "stuck", payload={"x": 1}, contract=contract, tamper=True)
        with caplog.at_level(logging.DEBUG, logger=SLOTS_LOGGER):
            resolve_model("stuck", local_dir=local_only, base_dir=tmp_path / "a-base", expected_contract=contract)
        local_level = max(r.levelno for r in caplog.records if "local" in r.getMessage())

        caplog.clear()
        base_only = tmp_path / "b-base"
        _install(base_only, "stuck", payload={"x": 1}, contract=contract, tamper=True)
        with caplog.at_level(logging.DEBUG, logger=SLOTS_LOGGER):
            resolve_model("stuck", local_dir=tmp_path / "b-local", base_dir=base_only, expected_contract=contract)
        base_level = max(r.levelno for r in caplog.records if "base" in r.getMessage())

        assert local_level == logging.WARNING
        assert base_level == logging.ERROR
        assert base_level > local_level

    def test_an_undeserializable_artifact_falls_through(self, empty_slots: tuple[Path, Path]) -> None:
        """Digest matches, bytes are still not a model. Refusal, not a traceback."""
        local, base = empty_slots
        local.mkdir(parents=True)
        artifact = artifact_path(local, "stuck")
        artifact.write_bytes(b"not a joblib payload at all")
        contract = _contract()
        write_manifest(
            manifest_path(local, "stuck"),
            Manifest(
                name="stuck",
                version="1",
                artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                runtime=Runtime(sklearn_version=running_sklearn_version() or "1.5.2"),
                feature_contract=contract,
            ),
        )

        resolution = resolve_model("stuck", local_dir=local, base_dir=base, expected_contract=contract)

        assert resolution.slot == SLOT_COLD_START
        assert resolution.reasons[0] == "undeserializable"

    def test_an_unreadable_slot_does_not_raise(self, empty_slots: tuple[Path, Path]) -> None:
        """A directory where a file is expected: OSError, refused not raised."""
        local, base = empty_slots
        local.mkdir(parents=True)
        artifact_path(local, "stuck").mkdir()
        manifest_path(local, "stuck").mkdir()

        resolution = resolve_model("stuck", local_dir=local, base_dir=base)

        assert resolution.slot == SLOT_COLD_START

    def test_no_exception_escapes_for_any_broken_shape(self, tmp_path: Path) -> None:
        """The prediction path must never see a raise (FR-017)."""
        shapes: list[dict[str, bool]] = [
            {"with_artifact": False},
            {"with_manifest": False},
            {"corrupt_manifest": True},
            {"tamper": True},
        ]
        for index, kwargs in enumerate(shapes):
            slot_dir = tmp_path / f"case-{index}"
            _install(slot_dir, "stuck", payload={"x": 1}, **kwargs)
            resolution = resolve_model("stuck", local_dir=slot_dir, base_dir=tmp_path / "none")
            assert resolution.cold_start is True


# ---------------------------------------------------------------------------
# T007.4 / FR-015 / C-004 — the base slot is read-only
# ---------------------------------------------------------------------------


class TestBaseSlotIsReadOnly:
    #: Anything that could mutate the filesystem. Matched against call targets
    #: in the AST rather than the source text, so the module's own prose about
    #: not calling ``mkdir`` does not trip its own check.
    FORBIDDEN_CALLS = frozenset(
        {
            "mkdir",
            "makedirs",
            "touch",
            "write_text",
            "write_bytes",
            "unlink",
            "rmdir",
            "rmtree",
            "remove",
            "rename",
            "replace",
            "copy",
            "copy2",
            "copyfile",
            "copytree",
            "write_manifest",
            "dump",
            "save",
        }
    )

    def _called_names(self) -> set[str]:
        tree = ast.parse(Path(slots_module.__file__).read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                names.add(node.func.id)
        return names

    def test_the_module_contains_no_write_call_at_all(self) -> None:
        """Structural, not conventional. A helper added 'for tests' fails here."""
        offenders = self._called_names() & self.FORBIDDEN_CALLS
        assert offenders == set(), f"slots.py can mutate the filesystem via {sorted(offenders)}"

    def test_the_module_opens_no_file_for_writing(self) -> None:
        assert "open" not in self._called_names()

    def test_the_public_surface_exposes_no_writer(self) -> None:
        exported = set(slots_module.__all__)
        assert not {n for n in exported if n.startswith(("write", "save", "install", "put", "copy"))}

    def test_resolving_never_touches_the_base_directory(self, tmp_path: Path) -> None:
        base = tmp_path / "ml-base"
        contract = _contract()
        _install(base, "stuck", payload={"from": "base"}, contract=contract)
        before = {p.name: p.read_bytes() for p in base.iterdir()}

        for _ in range(3):
            resolve_model("stuck", local_dir=tmp_path / "ml-models", base_dir=base, expected_contract=contract)

        after = {p.name: p.read_bytes() for p in base.iterdir()}
        assert after == before, "the base slot changed as a side effect of being read"

    def test_deserialization_goes_through_wp01_only(self) -> None:
        """No second ``joblib.load`` call site: verification is the only route."""
        source = Path(slots_module.__file__).read_text(encoding="utf-8")
        assert "import joblib" not in source
        assert "joblib.load" not in source


# ---------------------------------------------------------------------------
# T006 — config.base_models_dir() and config.retained_data_dir()
# ---------------------------------------------------------------------------


class TestBaseModelsDir:
    def test_source_install_resolves_beside_the_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delattr(sys, "frozen", raising=False)

        resolved = config.base_models_dir()

        assert resolved == Path(config.__file__).resolve().parent / config.BASE_MODELS_DIRNAME
        assert resolved.parent.name == "kenaz_ml"
        assert resolved.name == "ml-base"

    def test_frozen_bundle_resolves_under_meipass(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The branch the test suite never runs for real, asserted directly.

        The packaged build is not exercised here, so a wrong path in this branch
        would ship undetected and fail only in a notarized bundle.
        """
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

        assert config.base_models_dir() == tmp_path / "kenaz_ml" / "ml-base"

    def test_frozen_layout_matches_the_feature_store_precedent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """One convention for bundled package data, not two.

        ``feature_store.config.bundle_dir()`` already resolves
        ``<_MEIPASS>/kenaz_ml/feature_store``; the base slot must be its sibling,
        because the PyInstaller collection rules place both the same way.
        """
        from kenaz_ml.feature_store import config as fs_config

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

        assert config.base_models_dir().parent == fs_config.bundle_dir().parent

    def test_frozen_without_meipass_anchors_on_the_executable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)
        monkeypatch.setattr(sys, "executable", str(tmp_path / "kenaz-ml"))

        assert config.base_models_dir() == tmp_path / "kenaz_ml" / "ml-base"

    def test_it_does_not_create_the_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The prohibition that matters: mkdir inside a signed bundle fails.

        Asserted against a path guaranteed not to exist, so the check is that
        nothing was created rather than that something already existed.
        """
        bundle = tmp_path / "nonexistent-bundle"
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)

        resolved = config.base_models_dir()

        assert not resolved.exists()
        assert not resolved.parent.exists()
        assert not bundle.exists()

    def test_it_creates_nothing_on_the_source_branch_either(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The source branch, redirected so the assertion is order-independent.

        The frozen branch can be pointed at a path guaranteed not to exist; the
        source branch anchors on the module's own ``__file__``, so that is what
        gets redirected. Asserting against the real package directory instead
        would be order-dependent — any earlier test that called the function
        would already have created the directory a mutation was meant to reveal.
        """
        monkeypatch.delattr(sys, "frozen", raising=False)
        monkeypatch.setattr(config, "__file__", str(tmp_path / "pkg" / "config.py"))

        resolved = config.base_models_dir()
        config.base_models_dir()

        assert resolved == tmp_path / "pkg" / config.BASE_MODELS_DIRNAME
        assert not resolved.exists()
        assert not resolved.parent.exists()

    def test_no_branch_of_it_can_create_anything(self) -> None:
        """Structural, so it holds for every branch including unreached ones.

        A behavioral check can only cover the branch the test happens to take.
        Reading the function's own AST covers all of them, which matters because
        the frozen branch is the one that would fail — inside a notarized
        read-only bundle a ``mkdir`` raises rather than helps.
        """
        tree = ast.parse(Path(config.__file__).read_text(encoding="utf-8"))
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "base_models_dir")
        called = {
            node.func.attr
            for node in ast.walk(func)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

        assert not (called & {"mkdir", "makedirs", "touch", "write_text", "write_bytes"}), (
            f"base_models_dir() can create the read-only base slot via {sorted(called)}"
        )

    def test_it_is_idempotent_and_still_creates_nothing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)

        first = config.base_models_dir()
        second = config.base_models_dir()

        assert first == second
        assert not first.exists()

    def test_it_returns_a_path_even_when_nothing_is_shipped(self) -> None:
        """The real install today: the directory is absent and that is fine."""
        resolved = config.base_models_dir()

        assert isinstance(resolved, Path)
        assert resolved.is_absolute()


class TestRetainedDataDir:
    def test_it_is_created_on_demand_under_models_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

        resolved = config.retained_data_dir()

        assert resolved == tmp_path / "sigild" / "ml-models" / "retained"
        assert resolved.is_dir()

    def test_calling_it_twice_is_harmless(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

        assert config.retained_data_dir() == config.retained_data_dir()

    def test_it_is_inside_the_writable_slot_not_the_base_slot(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

        assert config.retained_data_dir().parent == config.models_dir()
        assert config.base_models_dir() not in config.retained_data_dir().parents


class TestExistingPathsUnchanged:
    """``models_dir()`` and ``weights_path()`` keep the flat pre-registry layout."""

    def test_models_dir_is_unchanged_and_still_created(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

        resolved = config.models_dir()

        assert resolved == tmp_path / "sigild" / "ml-models"
        assert resolved.is_dir()

    def test_weights_path_is_unchanged(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

        assert config.weights_path("stuck") == tmp_path / "sigild" / "ml-models" / "stuck.joblib"

    def test_the_two_slots_are_different_roots(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """D-001: the base slot is not a subdirectory of the writable one."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

        assert config.base_models_dir() != config.models_dir()
        assert config.models_dir() not in config.base_models_dir().parents


# ---------------------------------------------------------------------------
# The real Feast-sourced contract, i.e. the default `expected_contract=None`
# ---------------------------------------------------------------------------


class TestDefaultContractSourcing:
    def test_a_registered_model_validates_against_its_feature_service(self, empty_slots: tuple[Path, Path]) -> None:
        """``stuck`` has a Feast service, so a matching manifest is served."""
        local, base = empty_slots
        contract = local_feature_contract("stuck")
        assert contract is not None, "stuck should have a registered feature service"

        _install(local, "stuck", payload={"from": "local"}, contract=contract)

        resolution = resolve_model("stuck", local_dir=local, base_dir=base)

        assert resolution.slot == SLOT_LOCAL

    def test_an_unregistered_model_falls_through_to_cold_start(
        self, empty_slots: tuple[Path, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Fail closed, and land exactly where these models already are.

        ``quality``, ``workflow``, ``activity`` and the ``fleet_*`` family
        declare no feature service, so there is no contract to validate against
        and WP01 refuses. Cold start is where they run today anyway — none of
        them has an artifact on disk — so this is the existing behavior reached
        by a stricter route, not a regression.
        """
        local, base = empty_slots
        assert local_feature_contract("quality") is None

        _install(local, "quality", payload={"from": "local"})

        with caplog.at_level(logging.DEBUG, logger=SLOTS_LOGGER):
            resolution = resolve_model("quality", local_dir=local, base_dir=base)

        assert resolution.slot == SLOT_COLD_START
        assert resolution.refusals[0].check == CHECK_CONTRACT
        assert resolution.reasons[0] == "contract_unregistered"

    def test_unregistered_models_never_raise(self, empty_slots: tuple[Path, Path]) -> None:
        local, base = empty_slots
        for name in ("quality", "workflow", "activity", "fleet_focus"):
            _install(local, name, payload={"from": "local"})
            assert resolve_model(name, local_dir=local, base_dir=base).cold_start is True


def test_manifest_json_stays_human_readable(tmp_path: Path) -> None:
    """D-004's point: the sidecar can be opened and read without tooling."""
    _install(tmp_path, "stuck", payload={"x": 1})

    parsed = json.loads(manifest_path(tmp_path, "stuck").read_text(encoding="utf-8"))

    assert parsed["name"] == "stuck"
    assert len(parsed["artifact_sha256"]) == 64
