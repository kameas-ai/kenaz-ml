"""End-to-end: a contract change resets retained data through ``refresh_all()`` (WP02, SC-003).

Retained vectors computed under the old (degenerate) contracts are discarded,
not replayed -- with or without a shipped base -- the reset is recorded where an
operator can see it, retention resumes under the new contract, and the models
keep serving (cold start / rules) rather than erroring.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier

from kenaz_ml import config
from kenaz_ml.models.activity import ACTIVITY_FEATURE_NAMES, ActivityClassifier, activity_feature_contract
from kenaz_ml.modelstore import LocalModelStore
from kenaz_ml.modelstore.registry import (
    FeatureContract,
    Manifest,
    Provenance,
    Runtime,
    local_feature_contract,
    refresh_all,
    running_sklearn_version,
    write_manifest,
)
from kenaz_ml.modelstore.registry.refresh import (
    ACTION_NONE,
    ACTION_RESET,
    REASON_LOCAL_CONTRACT_CHANGED,
    REASON_NO_BASE,
    RESET_REASON_CONTRACT_CHANGED,
    reset_on_local_contract_change,
)
from kenaz_ml.modelstore.registry.retained import Example, append_examples, read_retained
from kenaz_ml.modelstore.registry.slots import resolve_model

STUCK_NAMES = (
    "test_failure_count",
    "time_in_phase_sec",
    "edit_velocity",
    "file_switch_rate",
    "session_length_sec",
    "time_since_last_commit_sec",
)


def _pre_salt_stuck_contract() -> FeatureContract:
    """The stuck contract as every existing install recorded it, before the salt."""
    refs = [f"stuck_features:{n}" for n in STUCK_NAMES]
    version = hashlib.sha256("|".join(["stuck", *refs]).encode()).hexdigest()[:16]
    return FeatureContract(
        service="stuck", service_version=version, names=STUCK_NAMES, dtypes=("float64",) * len(STUCK_NAMES)
    )


def _dump(model: Any) -> bytes:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()


def _stuck_model() -> Any:
    return DummyClassifier(strategy="constant", constant=1).fit(np.zeros((4, 6)), np.array([0, 1, 0, 1]))


def _install(slot: Path, name: str, model: Any, contract: FeatureContract, **prov: Any) -> Manifest:
    slot.mkdir(parents=True, exist_ok=True)
    payload = _dump(model)
    (slot / f"{name}.joblib").write_bytes(payload)
    manifest = Manifest(
        name=name,
        version=prov.pop("version", "1"),
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        created_at=1_753_900_000_000,
        provenance=Provenance(training_source=prov.pop("source", "local"), **prov),
        runtime=Runtime(
            estimator=type(model).__name__,
            sklearn_version=running_sklearn_version() or "",
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        ),
        feature_contract=contract,
    )
    write_manifest(slot / f"{name}.json", manifest)
    return manifest


def _retain(directory: Path, name: str, contract: FeatureContract, n: int = 5) -> None:
    width = len(contract.names)
    result = append_examples(
        name,
        [Example(x=tuple(float(i) for _ in range(width)), y=float(i % 2)) for i in range(n)],
        contract,
        directory=directory,
    )
    assert result.ok, result.reason


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    base = tmp_path / "base_models"
    base.mkdir()
    monkeypatch.setattr(config, "base_models_dir", lambda: base)
    local = tmp_path / "local"
    local.mkdir()
    retained = tmp_path / "retained"
    retained.mkdir()
    return {"local": local, "base": base, "retained": retained}


# ---------------------------------------------------------------------------
# No shipped base -- the state of every install today
# ---------------------------------------------------------------------------


class TestNoShippedBase:
    def _seed(self, world: dict[str, Path]) -> FeatureContract:
        old = _pre_salt_stuck_contract()
        assert old.service_version != local_feature_contract("stuck").service_version
        _install(world["local"], "stuck", _stuck_model(), old, n_local_extensions=1)
        _retain(world["retained"], "stuck", old, n=5)
        return old

    def _run(self, world: dict[str, Path]) -> Any:
        return refresh_all(["stuck"], local_dir=world["local"], base_dir=world["base"], retained_dir=world["retained"])[
            0
        ]

    def test_the_retained_set_is_discarded_and_the_reset_recorded(
        self, world: dict[str, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        old = self._seed(world)
        assert len(read_retained("stuck", directory=world["retained"]).examples) == 5
        current = local_feature_contract("stuck")

        with caplog.at_level(logging.WARNING, logger="kenaz_ml.modelstore.registry.refresh"):
            result = self._run(world)

        assert result.action == ACTION_RESET and result.ok
        assert result.reset_reason == RESET_REASON_CONTRACT_CHANGED
        assert result.change.reason == REASON_LOCAL_CONTRACT_CHANGED
        assert (result.previous_generation, result.retained_generation) == ("1", "2")

        after = read_retained("stuck", directory=world["retained"])
        assert after.examples == (), "discarded, not replayed"
        assert after.contract_version == current.service_version
        assert after.generation == "2"

        # Recorded: the manifest an operator can read, and the log line.
        raw = json.loads((world["local"] / "stuck.json").read_text(encoding="utf-8"))
        assert raw["provenance"]["reset_reason"] == "contract_version_changed"
        assert any(
            "personalization reset" in r.getMessage() and old.service_version in r.getMessage() for r in caplog.records
        )

    def test_the_old_artifact_is_refused_and_nothing_raises(self, world: dict[str, Path]) -> None:
        self._seed(world)
        self._run(world)

        resolution = resolve_model(
            "stuck", local_dir=world["local"], base_dir=world["base"], expected_contract=local_feature_contract("stuck")
        )
        assert not resolution.served
        assert resolution.slot == "cold_start"
        assert any("service_version" in str(r) or "contract" in str(r) for r in resolution.refusals)

    def test_retention_resumes_under_the_new_contract(self, world: dict[str, Path]) -> None:
        self._seed(world)
        self._run(world)
        current = local_feature_contract("stuck")

        appended = append_examples(
            "stuck", [Example(x=tuple(0.5 for _ in current.names), y=1.0)], current, directory=world["retained"]
        )
        assert appended.ok, appended.reason
        assert appended.generation == "2"

    def test_a_second_run_is_a_no_op(self, world: dict[str, Path]) -> None:
        self._seed(world)
        self._run(world)
        second = self._run(world)
        assert second.action == ACTION_NONE
        assert second.change.reason == REASON_NO_BASE
        assert read_retained("stuck", directory=world["retained"]).generation == "2"

    def test_a_matching_or_absent_retained_set_is_left_alone(self, world: dict[str, Path]) -> None:
        current = local_feature_contract("stuck")
        assert reset_on_local_contract_change("stuck", local_dir=world["local"], retained_dir=world["retained"]) is None
        _retain(world["retained"], "stuck", current, n=3)
        assert reset_on_local_contract_change("stuck", local_dir=world["local"], retained_dir=world["retained"]) is None
        assert len(read_retained("stuck", directory=world["retained"]).examples) == 3

    def test_an_unregistered_model_is_never_touched(self, world: dict[str, Path]) -> None:
        odd = FeatureContract(service="workflow", service_version="whatever", names=("a",), dtypes=("float64",))
        _retain(world["retained"], "workflow", odd, n=2)
        assert (
            reset_on_local_contract_change("workflow", local_dir=world["local"], retained_dir=world["retained"]) is None
        )
        assert len(read_retained("workflow", directory=world["retained"]).examples) == 2


# ---------------------------------------------------------------------------
# With a shipped base -- the existing path runs, and is not double-applied
# ---------------------------------------------------------------------------


class TestWithShippedBase:
    def test_the_base_path_resets_once_and_the_new_trigger_does_not_double_apply(self, world: dict[str, Path]) -> None:
        old = _pre_salt_stuck_contract()
        current = local_feature_contract("stuck")
        _install(world["local"], "stuck", _stuck_model(), old, version="3", base_version="0", base_sha256="0" * 64)
        _retain(world["retained"], "stuck", old, n=4)
        _install(world["base"], "stuck", _stuck_model(), current, version="1", source="base")

        (result,) = refresh_all(
            ["stuck"], local_dir=world["local"], base_dir=world["base"], retained_dir=world["retained"]
        )

        assert result.action == ACTION_RESET
        assert result.change.reason != REASON_LOCAL_CONTRACT_CHANGED, "the existing base-diff path handled it"
        assert (result.previous_generation, result.retained_generation) == ("1", "2"), "reset exactly once"
        after = read_retained("stuck", directory=world["retained"])
        assert after.generation == "2" and after.examples == ()
        assert after.contract_version == current.service_version


# ---------------------------------------------------------------------------
# The activity classifier
# ---------------------------------------------------------------------------


def _old_activity_model() -> SGDClassifier:
    """An artifact fitted on the pre-refresh vector width (6 kind dims -> 18 inputs)."""
    rng = np.random.default_rng(0)
    model = SGDClassifier(loss="log_loss", random_state=0)
    model.partial_fit(rng.random((20, 18)), np.array(["editing", "idle"] * 10), classes=np.array(["editing", "idle"]))
    return model


class TestActivity:
    EVENT = {"kind": "file", "payload": {"path": "/a.py"}}

    def test_an_old_contract_artifact_is_refused_and_rules_answer(self, world: dict[str, Path]) -> None:
        _install(world["local"], "activity", _old_activity_model(), FeatureContract())  # unregistered-era manifest
        clf = ActivityClassifier(model_store=LocalModelStore(base_dir=world["local"]), registry=True)

        assert not clf.is_trained
        assert clf.resolution is not None and not clf.resolution.served
        assert any("contract" in str(r) or "service" in str(r) for r in clf.resolution.refusals)
        assert clf.classify(self.EVENT)["method"] == "rules"

    def test_a_pre_registry_old_width_artifact_is_refused_by_the_width_guard(self, world: dict[str, Path]) -> None:
        # No manifest: the loader would stamp it with the CURRENT contract, so the width guard is what refuses it.
        (world["local"] / "activity.joblib").write_bytes(_dump(_old_activity_model()))
        clf = ActivityClassifier(model_store=LocalModelStore(base_dir=world["local"]), registry=True)

        assert not clf.is_trained
        assert clf.classify(self.EVENT)["method"] == "rules"  # never raises

    def test_a_current_width_artifact_is_served(self, world: dict[str, Path]) -> None:
        rng = np.random.default_rng(1)
        model = SGDClassifier(loss="log_loss", random_state=0)
        model.partial_fit(
            rng.random((20, len(ACTIVITY_FEATURE_NAMES))),
            np.array(["editing", "idle"] * 10),
            classes=np.array(["editing", "idle"]),
        )
        _install(world["local"], "activity", model, activity_feature_contract())
        clf = ActivityClassifier(model_store=LocalModelStore(base_dir=world["local"]), registry=True)
        assert clf.is_trained
        assert clf.classify(self.EVENT)["method"] == "ml"

    def test_its_retained_data_is_reset_and_retention_resumes(self, world: dict[str, Path]) -> None:
        old = FeatureContract(
            service="activity", service_version="pre-refresh", names=ACTIVITY_FEATURE_NAMES[:-1], dtypes=()
        )
        _retain(world["retained"], "activity", old, n=3)
        current = activity_feature_contract()

        (result,) = refresh_all(
            ["activity"], local_dir=world["local"], base_dir=world["base"], retained_dir=world["retained"]
        )
        assert result.action == ACTION_RESET and result.reset_reason == RESET_REASON_CONTRACT_CHANGED
        after = read_retained("activity", directory=world["retained"])
        assert after.examples == () and after.contract_version == current.service_version

        ok = append_examples(
            "activity", [Example(x=tuple(0.0 for _ in current.names), y=0.0)], current, directory=world["retained"]
        )
        assert ok.ok, ok.reason

    def test_the_positional_vector_is_built_from_the_registered_names(self) -> None:
        tree = ast.parse(Path(sys.modules[ActivityClassifier.__module__].__file__).read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_classify_ml")
        calls = {c.func.id for c in ast.walk(method) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        assert "sorted" not in calls, "the vector is ordered by the registered names, not sorted(keys)"
        assert "ACTIVITY_FEATURE_NAMES" in {n.id for n in ast.walk(method) if isinstance(n, ast.Name)}


# ---------------------------------------------------------------------------
# /introspect and the production caller
# ---------------------------------------------------------------------------


class TestIntrospectAndWiring:
    def test_startup_resets_and_introspect_reports_the_refusal(self, world: dict[str, Path]) -> None:
        from kenaz_ml.app import create_app

        models = config.models_dir()
        old = _pre_salt_stuck_contract()
        _install(models, "stuck", _stuck_model(), old, n_local_extensions=1)
        retained_dir = Path(config.retained_data_dir()) if hasattr(config, "retained_data_dir") else models / "retained"
        retained_dir.mkdir(parents=True, exist_ok=True)
        _retain(retained_dir, "stuck", old, n=3)

        with TestClient(create_app()) as client:
            body = client.get("/introspect").json()
            info = next(m for m in body["models"] if m["name"] == "stuck")
            assert info["serving_slot"] == "cold_start"
            assert info["refusal"], "the refused old artifact is visible to the operator"

        after = read_retained("stuck", directory=retained_dir)
        assert after.examples == () and after.contract_version == local_feature_contract("stuck").service_version
        raw = json.loads((models / "stuck.json").read_text(encoding="utf-8"))
        assert raw["provenance"]["reset_reason"] == "contract_version_changed"

    def test_refresh_all_has_a_production_call_site(self) -> None:
        """If the sibling mission's startup wiring is removed, this fix turns invisible -- loudly."""
        app_py = Path(sys.modules["kenaz_ml.app"].__file__)
        tree = ast.parse(app_py.read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
        }
        assert "refresh_all" in names
        assert "refresh_registry_roster" in names
