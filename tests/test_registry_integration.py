"""Integration tests for WP05 — the registry wired into loading and training.

Four things are asserted here, and one of them is a regression guard rather than
a feature.

**The pre-registry migration (T022).** This mission makes a local artifact with
no manifest unservable. Every install that exists today has exactly that: this
machine carries five trained artifacts — `stuck`, `duration`, `activity`,
`workflow`, `quality` — and not one manifest. Retraining is scheduled on
accumulated *new* data rather than on a clock, so the gap does not reliably
close on its own; without a migration the mission would silently discard every
existing install's trained models, possibly indefinitely.
`TestPreRegistryMigration` builds that exact fixture and asserts all five still
serve, and `TestTheMigrationIsConfinedToTheLocalSlot` asserts the *base* slot is
not given the same treatment — a base artifact without a manifest is still
refused, because that is the case the integrity guarantee exists for.

**`load()` still returns None and never raises (T018).** The `ModelLoader`
protocol docstring promises it and `ModelCache` and every predictor depend on
it. `TestLoaderRefusalsLookLikeAbsence` walks each refusal the registry can
produce — bad digest, wrong contract, wrong runtime, unparseable manifest,
undeserializable bytes, a feature store that raises — and asserts `None` out and
a reason in the log, for every one.

**Strict vector construction (T019, T020).** `[features[f] for f in names]`, in
both trainers, with no `.get(f, 0.0)` anywhere near it. The test greps the
source as well as exercising the behaviour, because the failure mode being
guarded against is somebody adding the default back to silence a `KeyError` that
was telling them something true.

**A manifest on every local run, never in the base slot (T021, FR-015,
FR-016).** Including the property that ties the whole mission together: after a
local training run, the loader can serve the artifact that run just wrote. A
manifest that disagreed with the artifact beside it would fail that, which is
exactly the silent downgrade the paired-write ordering exists to prevent.

Isolation
---------
Every test redirects `XDG_DATA_HOME` at `tmp_path` (so `models_dir()` and
`retained_data_dir()` are per-test) and monkeypatches `config.base_models_dir`
at a `tmp_path` subdirectory. The real `~/.local/share/sigild/` tree and the
real shipped base slot are never read or written.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from kenaz_ml import config
from kenaz_ml.models.duration import FEATURE_NAMES as DURATION_FEATURES
from kenaz_ml.models.stuck import FEATURE_NAMES as STUCK_FEATURES
from kenaz_ml.modelstore import FilesystemModelLoader, LocalModelStore
from kenaz_ml.modelstore.registry import (
    Manifest,
    Provenance,
    Runtime,
    local_feature_contract,
    manifest_to_dict,
    read_manifest,
    read_retained,
    running_sklearn_version,
    write_manifest,
)
from kenaz_ml.training import trainer as trainer_mod
from kenaz_ml.training.trainer import Trainer

HOUR_MS = 3_600_000
BASE_MS = 1_760_000_000_000
TENANT = "default"

#: The five artifacts a real install carries today. Two have a registered Feast
#: feature service; three do not, and neither does the `fleet_*` family.
PRE_REGISTRY_MODELS = ("stuck", "duration", "activity", "workflow", "quality")
REGISTERED_MODELS = ("stuck", "duration")
UNREGISTERED_MODELS = ("activity", "workflow", "quality", "fleet_throughput")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def slots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Isolate both slots at `tmp_path`, and return them.

    `local` is what `config.models_dir()` resolves to once `XDG_DATA_HOME`
    points here; `base` replaces the read-only slot that ships in the
    distribution, so a test can put an artifact there without touching the
    installed one.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    base = tmp_path / "base_models"
    monkeypatch.setattr(config, "base_models_dir", lambda: base)
    return {"local": config.models_dir(), "base": base}


def _artifact_bytes(payload: Any) -> bytes:
    """Serialize the way `LocalModelStore` receives bytes from a model class."""
    buf = io.BytesIO()
    joblib.dump(payload, buf)
    return buf.getvalue()


def _model_payload(name: str) -> dict[str, Any]:
    """A stand-in artifact body.

    Deliberately a plain container rather than a fitted estimator: nothing in
    the registry inspects the *contents* of an artifact — integrity hashes the
    bytes, the runtime check reads the manifest's recorded version, and the
    contract check reads the manifest's recorded names. Pickling a real
    estimator would turn these into scikit-learn version tests.
    """
    return {"model_name": name, "coefficients": np.array([0.5, -1.25, 3.0]), "trained_at_ms": BASE_MS}


def _write_pre_registry(slot_dir: Path, name: str) -> Path:
    """Write an artifact with *no* manifest beside it — the pre-registry state."""
    store = LocalModelStore(base_dir=slot_dir)
    store.save(name, _artifact_bytes(_model_payload(name)))
    path = slot_dir / f"{name}.joblib"
    assert not (slot_dir / f"{name}.json").exists()
    return path


def _manifest_for(path: Path, name: str, **overrides: Any) -> Manifest:
    """A manifest that would validate for `path`, before overrides are applied."""
    contract = local_feature_contract(name)
    fields: dict[str, Any] = {
        "name": name,
        "version": "1",
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "provenance": Provenance(n_local_extensions=1, training_source="local"),
        "runtime": Runtime(sklearn_version=running_sklearn_version() or "1.5", python_version="3.12"),
    }
    if contract is not None:
        fields["feature_contract"] = contract
    fields.update(overrides)
    return Manifest(**fields)


def _events_for(task_id: str, index: int, reference_ms: int) -> list[dict]:
    """Events for one task, measured against its own reference time.

    Even-indexed tasks get a long time-in-phase and, paired with `test_fails`,
    come out labelled stuck; odd-indexed ones do not. Both classes have to be
    present or `GradientBoostingClassifier.fit` refuses the matrix, which would
    make every assertion downstream about sklearn rather than about the
    registry.
    """
    phase_age_ms = 700_000 if index % 2 == 0 else 60_000
    events: list[dict] = [
        {
            "kind": "phase_change",
            "source": "sigild",
            "payload": {"phase": "coding"},
            "ts": reference_ms - phase_age_ms,
        },
    ]
    for n in range(index + 1):
        events.append(
            {"kind": "edit", "source": "editor", "payload": {"file": f"{task_id}-{n}.py"}, "ts": reference_ms - 1_000}
        )
    return events


class FakeStore:
    """Minimal DataStore covering exactly what the local trainer calls."""

    def __init__(self, tasks: list[dict], events: dict[str, list[dict]]) -> None:
        self._tasks = {t["id"]: t for t in tasks}
        self._events = events

    def get_completed_task_ids(self) -> list[str]:
        return list(self._tasks)

    def get_task_by_id(self, task_id: str) -> dict | None:
        return self._tasks.get(task_id)

    def get_events_for_task(self, task_id: str, since: int | None = None) -> list[dict]:
        return list(self._events.get(task_id, []))

    def get_completed_tasks_with_timestamps(self) -> list[dict]:
        return [
            {"id": t["id"], "started_at": t["started_at"], "completed_at": t["completed_at"]}
            for t in self._tasks.values()
        ]


def _training_store(n: int = 14) -> FakeStore:
    """`n` completed tasks, enough to clear the trainer's 10-example threshold."""
    tasks: list[dict] = []
    events: dict[str, list[dict]] = {}
    for i in range(n):
        completed_at = BASE_MS + i * HOUR_MS
        task_id = f"task-{i}"
        tasks.append(
            {
                "id": task_id,
                "repo_root": "/tmp/repo",
                "branch": "main",
                "phase": "done",
                "files": json.dumps({"main.py": 5}),
                "started_at": completed_at - 2 * HOUR_MS,
                "last_active": completed_at,
                "completed_at": completed_at,
                "test_fails": 5 if i % 2 == 0 else 0,
            }
        )
        events[task_id] = _events_for(task_id, i, completed_at)
    return FakeStore(tasks, events)


# ---------------------------------------------------------------------------
# T022 — the pre-registry migration
# ---------------------------------------------------------------------------


class TestPreRegistryMigration:
    """Five artifacts, no manifests, and all five must still serve.

    This is the shape of every install that exists today. If any assertion in
    this class fails, shipping the mission un-trains people.
    """

    @pytest.fixture
    def install(self, slots: dict[str, Path]) -> dict[str, Path]:
        """An install as it looks the moment before this mission lands."""
        return {name: _write_pre_registry(slots["local"], name) for name in PRE_REGISTRY_MODELS}

    def test_real_fitted_estimators_survive_the_migration(self, slots: dict[str, Path]) -> None:
        """The same five, as the estimators they actually are on disk.

        Everywhere else in this file the artifact body is a plain container,
        because nothing in the registry inspects it. Here it is the real thing —
        the two boosted models, an `IsolationForest`, and the two plain
        containers the signal models write — so that the deserialization half of
        the path is exercised against objects with the pickle shape a real
        install carries, not just against dicts.
        """
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest

        rng = np.random.default_rng(0)
        x6, x4 = rng.random((40, 6)), rng.random((40, 4))
        bodies: dict[str, Any] = {
            "stuck": GradientBoostingClassifier(n_estimators=5).fit(x6, (rng.random(40) > 0.5).astype(int)),
            "duration": GradientBoostingRegressor(n_estimators=5).fit(x4, rng.random(40) * 100),
            "activity": IsolationForest(n_estimators=5).fit(x6),
            "workflow": {"transitions": {"coding": {"testing": 3}}, "n": 3},
            "quality": {"weights": np.array([0.1, 0.9])},
        }
        store = LocalModelStore(base_dir=slots["local"])
        for name, body in bodies.items():
            store.save(name, _artifact_bytes(body))
        loader = FilesystemModelLoader(base_dir=slots["local"])

        served = {name: loader.load(TENANT, name) for name in bodies}

        unserved = sorted(name for name, model in served.items() if model is None)
        assert not unserved, f"the migration dropped {unserved}"
        assert isinstance(served["stuck"], GradientBoostingClassifier)
        assert isinstance(served["duration"], GradientBoostingRegressor)
        assert isinstance(served["activity"], IsolationForest)
        # And the model still predicts, which is the only thing that matters.
        assert served["stuck"].predict_proba(x6[:1]).shape == (1, 2)

    def test_all_five_pre_registry_artifacts_still_serve(
        self, install: dict[str, Path], slots: dict[str, Path]
    ) -> None:
        loader = FilesystemModelLoader(base_dir=slots["local"])

        served = {name: loader.load(TENANT, name) for name in PRE_REGISTRY_MODELS}

        unserved = sorted(name for name, model in served.items() if model is None)
        assert not unserved, f"the migration dropped {unserved} — these installs would lose their trained models"
        for name, model in served.items():
            assert model["model_name"] == name

    def test_each_artifact_gains_a_manifest_beside_it(self, install: dict[str, Path], slots: dict[str, Path]) -> None:
        loader = FilesystemModelLoader(base_dir=slots["local"])

        for name in PRE_REGISTRY_MODELS:
            loader.load(TENANT, name)

        for name in PRE_REGISTRY_MODELS:
            assert (slots["local"] / f"{name}.json").is_file(), f"no manifest synthesized for {name}"

    def test_the_recorded_digest_is_the_digest_of_the_bytes_on_disk(
        self, install: dict[str, Path], slots: dict[str, Path]
    ) -> None:
        """The manifest must describe the artifact, not assert something about it."""
        loader = FilesystemModelLoader(base_dir=slots["local"])

        for name, artifact in install.items():
            loader.load(TENANT, name)
            read = read_manifest(slots["local"] / f"{name}.json")

            assert read.ok and read.manifest is not None
            assert read.manifest.artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()

    def test_provenance_claims_no_base_and_no_extensions(
        self, install: dict[str, Path], slots: dict[str, Path]
    ) -> None:
        """A pre-registry artifact has no evidenced lineage, so none is invented."""
        FilesystemModelLoader(base_dir=slots["local"]).load(TENANT, "stuck")

        read = read_manifest(slots["local"] / "stuck.json")

        assert read.manifest is not None
        assert read.manifest.provenance.training_source == "local"
        assert read.manifest.provenance.base_version is None
        assert read.manifest.provenance.base_sha256 is None
        assert read.manifest.provenance.n_local_extensions == 0

    def test_the_runtime_recorded_is_this_environment(self, install: dict[str, Path], slots: dict[str, Path]) -> None:
        FilesystemModelLoader(base_dir=slots["local"]).load(TENANT, "stuck")

        read = read_manifest(slots["local"] / "stuck.json")

        assert read.manifest is not None
        assert read.manifest.runtime.sklearn_version == running_sklearn_version()

    @pytest.mark.parametrize("name", REGISTERED_MODELS)
    def test_a_registered_model_records_the_feast_contract(
        self, name: str, install: dict[str, Path], slots: dict[str, Path]
    ) -> None:
        FilesystemModelLoader(base_dir=slots["local"]).load(TENANT, name)

        read = read_manifest(slots["local"] / f"{name}.json")
        expected = local_feature_contract(name)

        assert read.manifest is not None
        assert expected is not None
        assert read.manifest.feature_contract.service == expected.service
        assert read.manifest.feature_contract.service_version == expected.service_version
        assert tuple(read.manifest.feature_contract.names) == tuple(expected.names)

    @pytest.mark.parametrize("name", UNREGISTERED_MODELS)
    def test_a_model_with_no_feature_service_records_no_contract(self, name: str, slots: dict[str, Path]) -> None:
        """Recorded as unregistered rather than invented — and still servable.

        `quality`, `workflow`, `activity` and the `fleet_*` family declare no
        Feast feature service. Fabricating a contract for them would put a
        fiction into the manifest; leaving the manifest to be refused would
        un-train them. So the contract is recorded empty, and the loader
        validates against an empty contract for exactly these models.
        """
        _write_pre_registry(slots["local"], name)
        loader = FilesystemModelLoader(base_dir=slots["local"])

        assert local_feature_contract(name) is None, f"{name} unexpectedly has a feature service now"
        assert loader.load(TENANT, name) is not None

        read = read_manifest(slots["local"] / f"{name}.json")
        assert read.manifest is not None
        assert read.manifest.feature_contract.service is None
        assert read.manifest.feature_contract.names == ()

    @pytest.mark.parametrize("name", UNREGISTERED_MODELS)
    def test_an_unregistered_model_serves_again_from_the_persisted_manifest(
        self, name: str, slots: dict[str, Path]
    ) -> None:
        """The synthesized manifest must not cause a refusal on the *next* load.

        The first load migrates; the second reads what the first wrote. If the
        recorded contract and the expected contract disagreed, the second load
        is where it would show up.
        """
        _write_pre_registry(slots["local"], name)
        loader = FilesystemModelLoader(base_dir=slots["local"])

        first = loader.load(TENANT, name)
        second = loader.load(TENANT, name)

        assert first is not None
        assert second is not None
        assert second["model_name"] == name

    def test_all_five_still_serve_on_a_second_pass(self, install: dict[str, Path], slots: dict[str, Path]) -> None:
        loader = FilesystemModelLoader(base_dir=slots["local"])
        for name in PRE_REGISTRY_MODELS:
            loader.load(TENANT, name)

        again = {name: loader.load(TENANT, name) for name in PRE_REGISTRY_MODELS}

        assert sorted(name for name, model in again.items() if model is None) == []

    def test_migration_leaves_an_existing_manifest_alone(self, slots: dict[str, Path]) -> None:
        """Only a *missing* manifest is synthesized; a present one is authoritative."""
        artifact = _write_pre_registry(slots["local"], "stuck")
        manifest_file = slots["local"] / "stuck.json"
        write_manifest(manifest_file, _manifest_for(artifact, "stuck", version="7"))
        before = manifest_file.read_bytes()

        FilesystemModelLoader(base_dir=slots["local"]).load(TENANT, "stuck")

        assert manifest_file.read_bytes() == before

    def test_a_manifest_that_disagrees_is_still_refused_after_migration_exists(self, slots: dict[str, Path]) -> None:
        """Migration is not a bypass: once a manifest is there, it is enforced.

        The artifact is tampered with *after* migration, which is the case the
        integrity check is for. Serving it would mean the sidecar had become
        decorative.
        """
        artifact = _write_pre_registry(slots["local"], "stuck")
        loader = FilesystemModelLoader(base_dir=slots["local"])
        assert loader.load(TENANT, "stuck") is not None

        artifact.write_bytes(_artifact_bytes({"model_name": "stuck", "swapped": True}))

        assert loader.load(TENANT, "stuck") is None

    def test_migration_does_not_run_for_a_model_with_no_artifact(self, slots: dict[str, Path]) -> None:
        loader = FilesystemModelLoader(base_dir=slots["local"])

        assert loader.load(TENANT, "stuck") is None
        assert not (slots["local"] / "stuck.json").exists()

    def test_a_directory_at_the_artifact_path_is_not_migrated(self, slots: dict[str, Path]) -> None:
        (slots["local"] / "stuck.joblib").mkdir(parents=True)
        loader = FilesystemModelLoader(base_dir=slots["local"])

        assert loader.load(TENANT, "stuck") is None
        assert not (slots["local"] / "stuck.json").exists()

    def test_a_tenant_scoped_pre_registry_artifact_is_migrated_in_place(self, slots: dict[str, Path]) -> None:
        """Tenant resolution still applies; migration follows the slot it picked."""
        tenant_dir = slots["local"] / "tenant-a"
        _write_pre_registry(tenant_dir, "stuck")
        loader = FilesystemModelLoader(base_dir=slots["local"])

        assert loader.load("tenant-a", "stuck") is not None
        assert (tenant_dir / "stuck.json").is_file()
        assert not (slots["local"] / "stuck.json").exists()

    def test_an_unwritable_slot_still_serves(self, slots: dict[str, Path]) -> None:
        """A read-only model directory must not un-train the install either.

        The synthesized manifest cannot be persisted, so it is validated in
        memory instead — the same three checks, in the same order, from the same
        functions. The only thing lost is that the next load re-does the work.
        """
        _write_pre_registry(slots["local"], "stuck")
        slots["local"].chmod(0o555)
        try:
            model = FilesystemModelLoader(base_dir=slots["local"]).load(TENANT, "stuck")
        finally:
            slots["local"].chmod(0o755)

        assert model is not None
        assert model["model_name"] == "stuck"


class TestTheMigrationIsConfinedToTheLocalSlot:
    """A base artifact with no manifest stays refused (FR-005).

    The integrity guarantee exists to catch a *shipped* artifact that was
    tampered with. A local artifact was written by this install into a directory
    it owns and rewrites on every training run, so synthesizing a manifest for
    it grants an attacker nothing they did not already have. That reasoning does
    not carry over to the base slot, and neither does the migration.
    """

    def test_a_base_artifact_without_a_manifest_is_refused(self, slots: dict[str, Path]) -> None:
        _write_pre_registry(slots["base"], "stuck")
        loader = FilesystemModelLoader(base_dir=slots["local"])

        assert loader.load(TENANT, "stuck") is None

    def test_no_manifest_is_synthesized_into_the_base_slot(self, slots: dict[str, Path]) -> None:
        _write_pre_registry(slots["base"], "stuck")

        FilesystemModelLoader(base_dir=slots["local"]).load(TENANT, "stuck")

        assert not (slots["base"] / "stuck.json").exists()

    def test_a_base_artifact_with_a_valid_manifest_does_serve(self, slots: dict[str, Path]) -> None:
        """The refusal is about the missing manifest, not about the base slot."""
        artifact = _write_pre_registry(slots["base"], "stuck")
        write_manifest(slots["base"] / "stuck.json", _manifest_for(artifact, "stuck"))

        assert FilesystemModelLoader(base_dir=slots["local"]).load(TENANT, "stuck") is not None

    def test_the_local_slot_still_wins_over_the_base_slot(self, slots: dict[str, Path]) -> None:
        base_artifact = _write_pre_registry(slots["base"], "stuck")
        write_manifest(slots["base"] / "stuck.json", _manifest_for(base_artifact, "stuck"))
        LocalModelStore(base_dir=slots["local"]).save(
            "stuck", _artifact_bytes({"model_name": "stuck", "slot": "local"})
        )

        model = FilesystemModelLoader(base_dir=slots["local"]).load(TENANT, "stuck")

        assert model is not None
        assert model.get("slot") == "local"


# ---------------------------------------------------------------------------
# T018 — refusals look like absence
# ---------------------------------------------------------------------------


class TestLoaderRefusalsLookLikeAbsence:
    """`load()` returns None on every refusal path and never raises.

    The `ModelLoader` protocol docstring says so and `ModelCache` and every
    predictor rely on it: a raise here turns a bad file on disk into a failed
    prediction request rather than a fallback to the heuristic (FR-017).
    """

    @staticmethod
    def _corrupt(slots: dict[str, Path], how: str) -> None:
        local = slots["local"]
        artifact = _write_pre_registry(local, "stuck")
        manifest_file = local / "stuck.json"

        if how == "digest_mismatch":
            write_manifest(manifest_file, _manifest_for(artifact, "stuck"))
            artifact.write_bytes(_artifact_bytes({"model_name": "stuck", "tampered": True}))
        elif how == "contract_mismatch":
            broken = _manifest_for(artifact, "stuck")
            names = ("not_a_real_feature", *broken.feature_contract.names[1:])
            write_manifest(
                manifest_file,
                _manifest_for(
                    artifact,
                    "stuck",
                    feature_contract=type(broken.feature_contract)(
                        service=broken.feature_contract.service,
                        service_version=broken.feature_contract.service_version,
                        names=names,
                        dtypes=broken.feature_contract.dtypes,
                    ),
                ),
            )
        elif how == "runtime_mismatch":
            write_manifest(
                manifest_file,
                _manifest_for(artifact, "stuck", runtime=Runtime(sklearn_version="0.1", python_version="3.12")),
            )
        elif how == "undeserializable":
            artifact.write_bytes(b"this is not a joblib pickle")
            write_manifest(manifest_file, _manifest_for(artifact, "stuck"))
        elif how == "unparseable_manifest":
            manifest_file.write_text("{not json", encoding="utf-8")
        elif how == "manifest_without_artifact":
            write_manifest(manifest_file, _manifest_for(artifact, "stuck"))
            artifact.unlink()
        elif how == "empty_artifact":
            artifact.write_bytes(b"")
            write_manifest(manifest_file, _manifest_for(artifact, "stuck"))
        else:  # pragma: no cover - guards the parametrize list
            raise AssertionError(f"unknown corruption {how!r}")

    REFUSALS = (
        "digest_mismatch",
        "contract_mismatch",
        "runtime_mismatch",
        "undeserializable",
        "unparseable_manifest",
        "manifest_without_artifact",
        "empty_artifact",
    )

    @pytest.mark.parametrize("how", REFUSALS)
    def test_a_refused_artifact_returns_none(self, how: str, slots: dict[str, Path]) -> None:
        self._corrupt(slots, how)

        assert FilesystemModelLoader(base_dir=slots["local"]).load(TENANT, "stuck") is None

    @pytest.mark.parametrize("how", REFUSALS)
    def test_a_refused_artifact_never_raises(self, how: str, slots: dict[str, Path]) -> None:
        self._corrupt(slots, how)
        loader = FilesystemModelLoader(base_dir=slots["local"])

        try:
            loader.load(TENANT, "stuck")
        except Exception as exc:  # pragma: no cover - the assertion is the point
            pytest.fail(f"load() raised {type(exc).__name__} on {how}: {exc}")

    @pytest.mark.parametrize("how", REFUSALS)
    def test_the_reason_reaches_the_log(
        self, how: str, slots: dict[str, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        """A refusal is invisible to the caller, so the log is the only report."""
        self._corrupt(slots, how)
        loader = FilesystemModelLoader(base_dir=slots["local"])

        with caplog.at_level(logging.WARNING):
            loader.load(TENANT, "stuck")

        assert any("failed to load" in record.message for record in caplog.records), (
            f"{how} was refused without saying why"
        )

    def test_a_feature_store_that_raises_does_not_propagate(
        self, slots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The contract is sourced from Feast, which is code this module does not own."""
        _write_pre_registry(slots["local"], "stuck")

        def explode(_name: str) -> Any:
            raise RuntimeError("the feature registry is a version-coupled protobuf")

        monkeypatch.setattr("kenaz_ml.modelstore.registry.manifest.local_feature_contract", explode)
        monkeypatch.setattr("kenaz_ml.modelstore.registry.local_feature_contract", explode)

        assert FilesystemModelLoader(base_dir=slots["local"]).load(TENANT, "stuck") is None

    def test_a_feature_store_that_raises_migrates_nothing(
        self, slots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fail closed: an install that cannot say what the contract is stamps nothing."""
        _write_pre_registry(slots["local"], "stuck")
        monkeypatch.setattr(
            "kenaz_ml.modelstore.registry.local_feature_contract",
            lambda _name: (_ for _ in ()).throw(RuntimeError("no")),
        )

        FilesystemModelLoader(base_dir=slots["local"]).load(TENANT, "stuck")

        assert not (slots["local"] / "stuck.json").exists()

    def test_an_empty_slot_is_not_reported_as_a_failure(
        self, slots: dict[str, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Nothing on disk is the ordinary state, not something to warn about."""
        loader = FilesystemModelLoader(base_dir=slots["local"])

        with caplog.at_level(logging.WARNING):
            assert loader.load(TENANT, "stuck") is None

        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_a_corrupt_tenant_artifact_does_not_fall_through_to_the_shared_one(self, slots: dict[str, Path]) -> None:
        """Serving one tenant another tenant's model is worse than serving none."""
        shared = _write_pre_registry(slots["local"], "stuck")
        write_manifest(slots["local"] / "stuck.json", _manifest_for(shared, "stuck"))
        tenant_dir = slots["local"] / "tenant-a"
        tenant_artifact = _write_pre_registry(tenant_dir, "stuck")
        write_manifest(tenant_dir / "stuck.json", _manifest_for(tenant_artifact, "stuck"))
        tenant_artifact.write_bytes(_artifact_bytes({"model_name": "stuck", "tampered": True}))

        assert FilesystemModelLoader(base_dir=slots["local"]).load("tenant-a", "stuck") is None


# ---------------------------------------------------------------------------
# T019 / T020 — strict vector construction
# ---------------------------------------------------------------------------


class TestStrictVectorConstruction:
    """`[features[f] for f in names]`, and no defaulted `.get` anywhere near it.

    A defaulted zero is indistinguishable from a real one, so a renamed or
    dropped feature used to train the model on a column of silence with nothing
    raised anywhere (FR-007). The source is grepped as well as exercised because
    the thing being guarded against is somebody restoring the default to silence
    a `KeyError` that was telling them something true.
    """

    @staticmethod
    def _comprehensions(module: str) -> list[ast.ListComp]:
        """Every list comprehension in `module`, read from the syntax tree.

        Parsed rather than grepped so that prose about `.get(f, 0.0)` -- of
        which there is a fair amount, since explaining why it is gone is half
        the point -- cannot fail or pass this test. Comments do not survive the
        parse and docstrings become `Constant` nodes, so what is left is
        executable code.
        """
        tree = ast.parse((Path(trainer_mod.__file__).parent / f"{module}.py").read_text(encoding="utf-8"))
        return [node for node in ast.walk(tree) if isinstance(node, ast.ListComp)]

    @pytest.mark.parametrize("module", ["trainer", "cloud_trainer"])
    def test_no_defaulted_feature_lookup_remains(self, module: str) -> None:
        offenders = [
            ast.unparse(comp)
            for comp in self._comprehensions(module)
            for call in ast.walk(comp.elt)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "get"
            and len(call.args) == 2
        ]

        assert offenders == [], f"{module}.py still defaults a feature lookup: {offenders}"

    @pytest.mark.parametrize("module", ["trainer", "cloud_trainer"])
    def test_the_strict_form_is_the_one_present(self, module: str) -> None:
        strict = [
            ast.unparse(comp)
            for comp in self._comprehensions(module)
            if isinstance(comp.elt, ast.Subscript) and isinstance(comp.elt.slice, ast.Name) and comp.elt.slice.id == "f"
        ]

        assert len(strict) == 2, f"{module}.py should build two vectors by strict lookup, found {strict}"

    def test_the_contract_agrees_with_the_names_the_extractor_emits(self) -> None:
        """The precondition that makes strict lookup safe, asserted directly."""
        for name, feature_names in (("stuck", STUCK_FEATURES), ("duration", DURATION_FEATURES)):
            contract = local_feature_contract(name)
            assert contract is not None
            assert tuple(contract.names) == tuple(feature_names)

    def test_a_missing_feature_is_not_absorbed(self, slots: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
        """It surfaces, rather than training on a zero nobody asked for.

        Reaching this state means validation was skipped, which is a defect. The
        test pins the reaction to it: raise, do not quietly fit.
        """
        monkeypatch.setattr(
            trainer_mod,
            "extract_stuck_features",
            lambda *a, **k: {f: 1.0 for f in STUCK_FEATURES if f != "edit_velocity"},
        )

        with pytest.raises(KeyError):
            Trainer(_training_store())._train_stuck()

    def test_vectors_are_built_in_contract_order(self, slots: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[np.ndarray] = []
        monkeypatch.setattr(
            trainer_mod,
            "extract_stuck_features",
            lambda *a, **k: {name: float(i) for i, name in enumerate(STUCK_FEATURES)},
        )
        monkeypatch.setattr("kenaz_ml.models.stuck.StuckPredictor.train", lambda self, X, y: captured.append(X))

        Trainer(_training_store())._train_stuck()

        assert captured
        np.testing.assert_array_equal(captured[0][0], np.arange(len(STUCK_FEATURES), dtype=float))

    def test_cold_start_works_with_no_contract_and_no_artifact(
        self, slots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing is loaded on cold start, so there is nothing to validate against."""
        monkeypatch.setattr("kenaz_ml.modelstore.registry.local_feature_contract", lambda _name: None)

        samples = Trainer(FakeStore([], {}))._train_stuck()

        assert samples == 500


class TestCloudPathHasNoRetention:
    """T020: strict lookup only. Retention is a local-personalization concept.

    Retained examples are the user's own behaviour, kept on the user's own
    machine (FR-011, C-003). Nothing in the cloud path may start accumulating
    them, and nothing there writes a sidecar manifest either -- cloud artifacts
    live in S3 and record provenance through MLflow (C-002).
    """

    @staticmethod
    def _names_used(module: str) -> set[str]:
        """Every identifier the module's executable code references."""
        tree = ast.parse((Path(trainer_mod.__file__).parent / f"{module}.py").read_text(encoding="utf-8"))
        return {
            node.id if isinstance(node, ast.Name) else node.attr
            for node in ast.walk(tree)
            if isinstance(node, (ast.Name, ast.Attribute))
        } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}

    @pytest.mark.parametrize(
        "forbidden", ["append_examples", "read_retained", "reset_retained", "retained_path", "write_manifest"]
    )
    def test_the_cloud_trainer_reaches_for_no_local_only_machinery(self, forbidden: str) -> None:
        assert forbidden not in self._names_used("cloud_trainer")

    def test_the_local_trainer_does_reach_for_it(self) -> None:
        """The mirror of the above -- otherwise the assertion above proves nothing."""
        used = self._names_used("trainer")

        assert "append_examples" in used
        assert "write_manifest" in used


# ---------------------------------------------------------------------------
# T019 — retention (FR-009)
# ---------------------------------------------------------------------------


class TestRetention:
    """Every vector a local run trains on is kept, so a base refresh can replay it."""

    def test_a_successful_run_retains_its_examples(self, slots: dict[str, Path]) -> None:
        store = _training_store(n=14)

        samples = Trainer(store)._train_stuck()

        retained = read_retained("stuck")
        assert retained.ok
        assert len(retained) == samples

    def test_each_example_carries_its_own_reference_time(self, slots: dict[str, Path]) -> None:
        """`as_of_ms` is recorded, never recomputed — the whole point of it."""
        Trainer(_training_store(n=14))._train_stuck()

        retained = read_retained("stuck")

        expected = {BASE_MS + i * HOUR_MS for i in range(14)}
        assert {e.as_of_ms for e in retained.examples} == expected

    def test_the_header_stamps_the_contract_version(self, slots: dict[str, Path]) -> None:
        Trainer(_training_store(n=14))._train_stuck()

        retained = read_retained("stuck")
        contract = local_feature_contract("stuck")

        assert contract is not None
        assert retained.contract_version == contract.service_version
        assert retained.names == tuple(contract.names)

    def test_a_failed_run_retains_nothing(self, slots: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
        """A partial run must not leave examples the model was never fitted on."""

        def explode(self: Any, X: Any, y: Any) -> None:
            raise RuntimeError("fit failed")

        monkeypatch.setattr("kenaz_ml.models.stuck.StuckPredictor.train", explode)

        with pytest.raises(RuntimeError):
            Trainer(_training_store(n=14))._train_stuck()

        assert not read_retained("stuck").ok

    def test_synthetic_cold_start_retains_nothing(self, slots: dict[str, Path]) -> None:
        """Generated data is not this user's behaviour; replaying it teaches nobody."""
        assert Trainer(FakeStore([], {}))._train_stuck() == 500

        assert not read_retained("stuck").ok

    def test_a_second_run_appends_rather_than_replacing(self, slots: dict[str, Path]) -> None:
        trainer = Trainer(_training_store(n=14))

        first = trainer._train_stuck()
        second = trainer._train_stuck()

        assert len(read_retained("stuck")) == first + second

    def test_retention_failure_does_not_fail_the_training_run(
        self, slots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The model was fitted and saved; losing the retained copy is not fatal."""

        def explode(*args: Any, **kwargs: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr("kenaz_ml.modelstore.registry.append_examples", explode)

        assert Trainer(_training_store(n=14))._train_stuck() == 14


# ---------------------------------------------------------------------------
# T021 — a manifest on every local training run (FR-015, FR-016)
# ---------------------------------------------------------------------------


class TestLocalTrainingWritesAManifest:
    def test_a_manifest_is_written_beside_the_artifact(self, slots: dict[str, Path]) -> None:
        Trainer(_training_store(n=14))._train_stuck()

        assert (slots["local"] / "stuck.json").is_file()

    def test_the_digest_matches_the_bytes_actually_written(self, slots: dict[str, Path]) -> None:
        Trainer(_training_store(n=14))._train_stuck()

        read = read_manifest(slots["local"] / "stuck.json")
        artifact = (slots["local"] / "stuck.joblib").read_bytes()

        assert read.manifest is not None
        assert read.manifest.artifact_sha256 == hashlib.sha256(artifact).hexdigest()

    def test_the_loader_serves_back_what_training_just_wrote(self, slots: dict[str, Path]) -> None:
        """The property the whole mission rests on.

        If the pair disagreed by so much as a byte the artifact would be refused
        and the install would drop to the base model or to cold start — silently,
        because a refusal is indistinguishable from an absence at the call site.
        """
        Trainer(_training_store(n=14))._train_stuck()

        served = FilesystemModelLoader(base_dir=slots["local"]).load(TENANT, "stuck")

        assert served is not None

    def test_the_contract_and_sample_count_are_recorded(self, slots: dict[str, Path]) -> None:
        samples = Trainer(_training_store(n=14))._train_stuck()

        read = read_manifest(slots["local"] / "stuck.json")
        contract = local_feature_contract("stuck")

        assert read.manifest is not None
        assert contract is not None
        assert read.manifest.training.n_samples == samples
        assert read.manifest.feature_contract.service_version == contract.service_version
        assert read.manifest.runtime.estimator == "GradientBoostingClassifier"

    def test_the_retained_generation_is_recorded(self, slots: dict[str, Path]) -> None:
        Trainer(_training_store(n=14))._train_stuck()

        read = read_manifest(slots["local"] / "stuck.json")

        assert read.manifest is not None
        assert read.manifest.training.retained_generation == read_retained("stuck").generation

    def test_the_extension_count_increments_across_runs(self, slots: dict[str, Path]) -> None:
        trainer = Trainer(_training_store(n=14))

        counts = []
        for _ in range(3):
            trainer._train_stuck()
            read = read_manifest(slots["local"] / "stuck.json")
            assert read.manifest is not None
            counts.append(read.manifest.provenance.n_local_extensions)

        assert counts == [1, 2, 3]

    def test_a_migrated_artifact_starts_the_count_at_its_first_real_run(self, slots: dict[str, Path]) -> None:
        """Migration records 0 extensions because it can evidence none; the first
        run after it records 1."""
        _write_pre_registry(slots["local"], "stuck")
        FilesystemModelLoader(base_dir=slots["local"]).load(TENANT, "stuck")
        assert read_manifest(slots["local"] / "stuck.json").manifest.provenance.n_local_extensions == 0

        Trainer(_training_store(n=14))._train_stuck()

        read = read_manifest(slots["local"] / "stuck.json")
        assert read.manifest is not None
        assert read.manifest.provenance.n_local_extensions == 1

    def test_the_base_being_extended_is_recorded_on_the_first_run(self, slots: dict[str, Path]) -> None:
        base_artifact = _write_pre_registry(slots["base"], "stuck")
        base_manifest = _manifest_for(
            base_artifact, "stuck", version="3", provenance=Provenance(training_source="base")
        )
        write_manifest(slots["base"] / "stuck.json", base_manifest)

        Trainer(_training_store(n=14))._train_stuck()

        read = read_manifest(slots["local"] / "stuck.json")
        assert read.manifest is not None
        assert read.manifest.provenance.base_version == "3"
        assert read.manifest.provenance.base_sha256 == base_manifest.artifact_sha256
        assert read.manifest.provenance.n_local_extensions == 1

    def test_the_base_identity_is_carried_forward_on_later_runs(self, slots: dict[str, Path]) -> None:
        base_artifact = _write_pre_registry(slots["base"], "stuck")
        write_manifest(slots["base"] / "stuck.json", _manifest_for(base_artifact, "stuck", version="3"))
        trainer = Trainer(_training_store(n=14))

        trainer._train_stuck()
        trainer._train_stuck()

        read = read_manifest(slots["local"] / "stuck.json")
        assert read.manifest is not None
        assert read.manifest.provenance.base_version == "3"
        assert read.manifest.provenance.n_local_extensions == 2

    def test_a_synthetic_run_says_so(self, slots: dict[str, Path]) -> None:
        Trainer(FakeStore([], {}))._train_stuck()

        read = read_manifest(slots["local"] / "stuck.json")

        assert read.manifest is not None
        assert read.manifest.provenance.training_source == "synthetic"
        assert read.manifest.training.n_samples == 500

    def test_the_duration_estimator_gets_one_too(self, slots: dict[str, Path]) -> None:
        Trainer(_training_store(n=14))._train_duration()

        read = read_manifest(slots["local"] / "duration.json")

        assert read.manifest is not None
        assert read.manifest.name == "duration"
        assert read.manifest.runtime.estimator == "GradientBoostingRegressor"

    def test_a_failed_run_leaves_no_manifest_behind(
        self, slots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(self: Any, X: Any, y: Any) -> None:
            raise RuntimeError("fit failed")

        monkeypatch.setattr("kenaz_ml.models.stuck.StuckPredictor.train", explode)

        with pytest.raises(RuntimeError):
            Trainer(_training_store(n=14))._train_stuck()

        assert not (slots["local"] / "stuck.json").exists()

    def test_a_failed_run_never_leaves_a_mismatched_pair(
        self, slots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The window between the two writes holds an artifact with no manifest.

        That state is migrated on the next load. A *mismatched* pair would be
        refused instead, which is the silent downgrade the write ordering exists
        to prevent — so the previous run's manifest must be gone, not stale.
        """
        trainer = Trainer(_training_store(n=14))
        trainer._train_stuck()
        assert (slots["local"] / "stuck.json").is_file()

        def explode(self: Any, X: Any, y: Any) -> None:
            raise RuntimeError("fit failed")

        monkeypatch.setattr("kenaz_ml.models.stuck.StuckPredictor.train", explode)
        with pytest.raises(RuntimeError):
            trainer._train_stuck()

        assert not (slots["local"] / "stuck.json").exists()
        assert FilesystemModelLoader(base_dir=slots["local"]).load(TENANT, "stuck") is not None

    def test_no_manifest_is_written_when_the_contract_cannot_be_vouched_for(
        self, slots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A manifest recording an empty contract would discard this very model.

        The loader validates against the contract the *install* declares. An
        empty one recorded for a model that has a registered service refuses on
        the next load — so writing nothing, and letting the loader migrate the
        artifact into a manifest that is true, is the safe direction.
        """
        monkeypatch.setattr("kenaz_ml.modelstore.registry.local_feature_contract", lambda _name: None)

        assert Trainer(_training_store(n=14))._train_stuck() == 14

        assert not (slots["local"] / "stuck.json").exists()

    def test_that_artifact_is_then_recovered_by_the_migration(
        self, slots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two halves meet: what training declined to stamp, loading reconstructs."""
        with monkeypatch.context() as broken:
            broken.setattr("kenaz_ml.modelstore.registry.local_feature_contract", lambda _name: None)
            Trainer(_training_store(n=14))._train_stuck()

        served = FilesystemModelLoader(base_dir=slots["local"]).load(TENANT, "stuck")

        assert served is not None
        read = read_manifest(slots["local"] / "stuck.json")
        assert read.manifest is not None
        assert read.manifest.feature_contract.service == "stuck"

    def test_the_manifest_round_trips_through_json_unchanged(self, slots: dict[str, Path]) -> None:
        Trainer(_training_store(n=14))._train_stuck()
        path = slots["local"] / "stuck.json"

        read = read_manifest(path)

        assert read.manifest is not None
        assert json.loads(path.read_text(encoding="utf-8")) == manifest_to_dict(read.manifest)


class TestLocalTrainingNeverWritesTheBaseSlot:
    """FR-015. The base slot is read-only and covered by the binary's signature."""

    @staticmethod
    def _snapshot(directory: Path) -> dict[str, bytes]:
        return {p.name: p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}

    def test_the_base_slot_is_byte_identical_after_repeated_training(self, slots: dict[str, Path]) -> None:
        artifact = _write_pre_registry(slots["base"], "stuck")
        write_manifest(slots["base"] / "stuck.json", _manifest_for(artifact, "stuck", version="3"))
        before = self._snapshot(slots["base"])
        trainer = Trainer(_training_store(n=14))

        for _ in range(3):
            trainer._train_stuck()
            trainer._train_duration()

        assert self._snapshot(slots["base"]) == before

    def test_an_empty_base_slot_stays_empty(self, slots: dict[str, Path]) -> None:
        Trainer(_training_store(n=14))._train_stuck()

        assert not slots["base"].exists() or list(slots["base"].rglob("*")) == []

    def test_a_store_pointing_at_the_base_slot_is_refused(
        self, slots: dict[str, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Not reachable through any supported configuration — asserted so it stays that way.

        A training run that wrote here would replace a signed artifact with an
        unsigned one and destroy the only pristine copy the install can fall
        back to, so the guard is a check rather than an assumption.
        """
        slots["base"].mkdir(parents=True, exist_ok=True)
        trainer = Trainer(_training_store(n=14), model_store=LocalModelStore(base_dir=slots["base"]))

        with caplog.at_level(logging.ERROR):
            trainer._train_stuck()

        assert not (slots["base"] / "stuck.json").exists()
        assert any("base slot" in record.message for record in caplog.records)

    def test_the_base_slot_check_recognises_a_subdirectory(self, slots: dict[str, Path]) -> None:
        assert trainer_mod._is_base_slot(slots["base"])
        assert trainer_mod._is_base_slot(slots["base"] / "tenant-a")
        assert not trainer_mod._is_base_slot(slots["local"])


class TestArtifactDirectoryResolution:
    """Where the trainer looks for the artifact it just caused to be written."""

    def test_no_store_means_the_default_local_slot(self, slots: dict[str, Path]) -> None:
        assert trainer_mod._artifact_dir(None) == config.models_dir()

    def test_a_local_store_names_its_own_directory(self, slots: dict[str, Path]) -> None:
        store = LocalModelStore(base_dir=slots["local"] / "elsewhere")

        assert trainer_mod._artifact_dir(store) == slots["local"] / "elsewhere"

    def test_a_cache_in_front_of_a_local_store_writes_through_to_it(self, slots: dict[str, Path]) -> None:
        from kenaz_ml.modelstore import CachedModelStore

        store = CachedModelStore(LocalModelStore(base_dir=slots["local"]))

        assert trainer_mod._artifact_dir(store) == slots["local"]

    def test_a_store_that_is_not_a_filesystem_gets_no_manifest(self, slots: dict[str, Path]) -> None:
        """An S3-backed or fake store has no sidecar to write, and that is fine."""

        class NotAFilesystem:
            def load(self, model_name: str) -> bytes | None:
                return None

            def save(self, model_name: str, data: bytes) -> None:
                pass

            def exists(self, model_name: str) -> bool:
                return False

        assert trainer_mod._artifact_dir(NotAFilesystem()) is None

    def test_training_through_a_non_filesystem_store_still_succeeds(self, slots: dict[str, Path]) -> None:
        saved: dict[str, bytes] = {}

        class InMemory:
            def load(self, model_name: str) -> bytes | None:
                return saved.get(model_name)

            def save(self, model_name: str, data: bytes) -> None:
                saved[model_name] = data

            def exists(self, model_name: str) -> bool:
                return model_name in saved

        assert Trainer(_training_store(n=14), model_store=InMemory())._train_stuck() == 14
        assert "stuck" in saved
        assert list(slots["local"].glob("*.json")) == []
