"""two-client-engine-01MSK2EN WP01 — local serving goes through the registry.

Before this mission, local serving loaded ``{name}.joblib`` raw: no manifest,
no integrity check, no contract, no base slot. These tests assert, against the
real app started in local mode, that:

* a base-slot-only model is served by ``/predict/*`` and reported by
  ``/introspect`` as ``training_source: "base"`` (spec User Story 1);
* a bare pre-registry ``.joblib`` (every existing install) is migrated in place
  and the *personalized* model keeps serving (FR-022);
* no artifact anywhere is today's cold start, unchanged;
* a tampered local artifact is refused, the base serves instead, and the app
  still boots;
* models with no Feast service (``activity``, ``workflow``) still serve a bare
  artifact (the ``UNREGISTERED_CONTRACT`` path, not bare ``resolve_model``);
* ``refresh_all()`` runs at startup and again after the retrain callback, with
  an audit line (SC-002).

Isolation: ``XDG_DATA_HOME`` points at ``tmp_path`` and
``config.base_models_dir`` is monkeypatched to a ``tmp_path`` subdirectory.
"""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from kenaz_ml import config
from kenaz_ml.modelstore import LocalModelStore
from kenaz_ml.modelstore.registry import (
    Manifest,
    Provenance,
    Runtime,
    local_feature_contract,
    read_manifest,
    running_sklearn_version,
    write_manifest,
)

STUCK_FEATURES = {
    "test_failure_count": 5,
    "time_in_phase_sec": 1200,
    "edit_velocity": 4.0,
    "file_switch_rate": 0.7,
    "session_length_sec": 3600,
    "time_since_last_commit_sec": 1800,
}


@pytest.fixture
def slots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    base = tmp_path / "base_models"
    base.mkdir()
    monkeypatch.setattr(config, "base_models_dir", lambda: base)
    return {"local": config.models_dir(), "base": base}


def _dump(obj: Any) -> bytes:
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.getvalue()


def _stuck_model(always: int) -> Any:
    """A fitted classifier whose answer identifies which artifact is serving."""
    from sklearn.dummy import DummyClassifier

    x = np.zeros((4, 6))
    y = np.array([0, 1, 0, 1])
    model = DummyClassifier(strategy="constant", constant=always)
    return model.fit(x, y)


def _write_pair(slot: Path, name: str, body: Any, *, version: str = "1", source: str = "base", **prov: Any) -> Path:
    artifact = slot / f"{name}.joblib"
    artifact.write_bytes(_dump(body))
    contract = local_feature_contract(name)
    fields: dict[str, Any] = {
        "name": name,
        "version": version,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "provenance": Provenance(training_source=source, **prov),
        "runtime": Runtime(sklearn_version=running_sklearn_version() or "", python_version="3.12"),
    }
    if contract is not None:
        fields["feature_contract"] = contract
    write_manifest(slot / f"{name}.json", Manifest(**fields))
    return artifact


def _descends_from(base_artifact: Path, version: str = "1") -> dict[str, Any]:
    """Provenance of a local extension of the shipped base -- so startup's
    refresh_all() sees it as up to date rather than adopting the base over it."""
    return {
        "base_version": version,
        "base_sha256": hashlib.sha256(base_artifact.read_bytes()).hexdigest(),
        "n_local_extensions": 1,
    }


def _client() -> TestClient:
    from kenaz_ml.app import create_app

    return TestClient(create_app())


def _introspect(client: TestClient, name: str) -> dict[str, Any]:
    body = client.get("/introspect").json()
    return next(m for m in body["models"] if m["name"] == name)


class TestBaseSlotServing:
    def test_base_only_stuck_is_served_and_reported(self, slots: dict[str, Path]) -> None:
        _write_pair(slots["base"], "stuck", _stuck_model(1))

        with _client() as client:
            resp = client.post("/predict/stuck", json={"features": STUCK_FEATURES})
            assert resp.status_code == 200
            # The constant-1 dummy says "stuck" with probability 1.0: the base
            # artifact answered, not the untrained 0.5 fallback.
            assert resp.json()["probability"] == 1.0

            info = _introspect(client, "stuck")
            assert info["serving_slot"] == "base"
            assert info["training_source"] == "base"
            assert info["manifest_version"] == "1"
            assert info["refusal"] is None

    @pytest.mark.parametrize("name", ["stuck", "duration", "activity", "workflow"])
    def test_each_predictor_serves_a_base_only_model(self, slots: dict[str, Path], name: str) -> None:
        _write_pair(slots["base"], name, {"stand_in_for": name})

        from kenaz_ml.app import AppState

        state = AppState()
        state.load_models(LocalModelStore(base_dir=slots["local"]))

        predictor = getattr(state, name)
        assert predictor.is_trained, f"{name} ignored the base slot"
        assert state.resolutions[name].slot == "base"


class TestLocalPairAndMigration:
    def test_local_pair_is_served_from_the_local_slot(self, slots: dict[str, Path]) -> None:
        base = _write_pair(slots["base"], "stuck", _stuck_model(0))
        _write_pair(slots["local"], "stuck", _stuck_model(1), source="local", **_descends_from(base))

        with _client() as client:
            assert client.post("/predict/stuck", json={"features": STUCK_FEATURES}).json()["probability"] == 1.0
            assert _introspect(client, "stuck")["serving_slot"] == "local"

    def test_bare_pre_registry_joblib_is_migrated_and_keeps_serving(self, slots: dict[str, Path]) -> None:
        """Every existing install: stuck.joblib, no stuck.json. Personalization must survive."""
        LocalModelStore(base_dir=slots["local"]).save("stuck", _dump(_stuck_model(1)))
        assert not (slots["local"] / "stuck.json").exists()

        with _client() as client:
            # The personalized model serves, not cold start.
            assert client.post("/predict/stuck", json={"features": STUCK_FEATURES}).json()["probability"] == 1.0
            info = _introspect(client, "stuck")
            assert info["serving_slot"] == "local"
            assert info["training_source"] == "local"

        read = read_manifest(slots["local"] / "stuck.json")
        assert read.ok and read.manifest is not None, "no manifest synthesized on disk"
        assert read.manifest.extra.get("synthesized_by") == "pre_registry_migration"
        assert (
            read.manifest.artifact_sha256 == hashlib.sha256((slots["local"] / "stuck.joblib").read_bytes()).hexdigest()
        )

    @pytest.mark.parametrize("name", ["activity", "workflow"])
    def test_unregistered_service_models_still_serve_a_bare_artifact(self, slots: dict[str, Path], name: str) -> None:
        """Guards the UNREGISTERED_CONTRACT path against a bare-resolve_model regression."""
        LocalModelStore(base_dir=slots["local"]).save(name, _dump({"stand_in_for": name}))

        from kenaz_ml.app import AppState

        state = AppState()
        state.load_models(LocalModelStore(base_dir=slots["local"]))

        assert getattr(state, name).is_trained
        assert state.resolutions[name].slot == "local"
        assert (slots["local"] / f"{name}.json").is_file()


class TestColdStartAndRefusal:
    def test_no_artifact_anywhere_is_unchanged_cold_start(self, slots: dict[str, Path]) -> None:
        with _client() as client:
            assert client.post("/predict/stuck", json={"features": STUCK_FEATURES}).json() == {
                "probability": 0.5,
                "confidence": "weak",
            }
            health = client.get("/health").json()["models"]
            assert health["stuck"] == "untrained"
            info = _introspect(client, "stuck")
            assert info["serving_slot"] == "cold_start"
            assert info["training_source"] is None
            assert info["manifest_version"] is None
            assert info["refusal"] is None  # an empty slot is not a refusal

    def test_tampered_local_artifact_falls_through_to_base(self, slots: dict[str, Path]) -> None:
        base = _write_pair(slots["base"], "stuck", _stuck_model(0))
        artifact = _write_pair(slots["local"], "stuck", _stuck_model(1), source="local", **_descends_from(base))
        artifact.write_bytes(artifact.read_bytes() + b"tampered")

        with _client() as client:
            assert client.post("/predict/stuck", json={"features": STUCK_FEATURES}).json()["probability"] == 0.0
            info = _introspect(client, "stuck")
            assert info["serving_slot"] == "base"
            assert info["refusal"] is not None and "[local]" in info["refusal"]

    def test_tampered_artifacts_in_both_slots_cold_start_and_boot(self, slots: dict[str, Path]) -> None:
        for slot in ("local", "base"):
            artifact = _write_pair(slots[slot], "stuck", _stuck_model(1))
            artifact.write_bytes(b"not the bytes the manifest describes")

        with _client() as client:
            assert client.get("/health").status_code == 200
            info = _introspect(client, "stuck")
            assert info["serving_slot"] == "cold_start"
            assert "[local]" in info["refusal"] and "[base]" in info["refusal"]

    def test_non_filesystem_store_keeps_the_legacy_path(self) -> None:
        class BytesStore:
            def __init__(self) -> None:
                self.data = {"stuck": _dump(_stuck_model(1))}

            def load(self, name: str) -> bytes | None:
                return self.data.get(name)

            def save(self, name: str, data: bytes) -> None:
                self.data[name] = data

            def exists(self, name: str) -> bool:
                return name in self.data

        from kenaz_ml.models.stuck import StuckPredictor

        predictor = StuckPredictor(model_store=BytesStore(), registry=True)
        assert predictor.is_trained
        assert predictor.resolution is None


class TestRefreshAll:
    def test_refresh_all_runs_at_startup_with_an_audit_line(
        self, slots: dict[str, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="kenaz_ml"), _client():
            pass

        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("refresh_all: model=")]
        for name in ("stuck", "activity", "workflow", "duration"):
            assert any(f"model={name} " in line for line in lines), f"no refresh_all line for {name}"
        assert any("reason=no_base" in line for line in lines)

    def test_refresh_all_runs_before_models_load(self, slots: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
        from kenaz_ml.app import AppState

        order: list[str] = []
        real_load = AppState.load_models
        monkeypatch.setattr(AppState, "refresh_registry", lambda self, ms=None: order.append("refresh"))
        monkeypatch.setattr(AppState, "load_models", lambda self, ms=None: (order.append("load"), real_load(self, ms)))

        with _client():
            pass
        assert order[:2] == ["refresh", "load"]

    def test_retrain_callback_reruns_refresh(self, slots: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
        import kenaz_ml.modelstore.registry as registry
        from kenaz_ml.app import AppState

        calls: list[tuple[str, ...]] = []
        real = registry.refresh_all
        monkeypatch.setattr(
            registry, "refresh_all", lambda names, **kw: (calls.append(tuple(names)), real(names, **kw))[1]
        )

        state = AppState()
        state.model_store = LocalModelStore(base_dir=slots["local"])
        state.reload_models_into_poller()

        assert calls == [("stuck", "activity", "workflow", "duration")]

    def test_a_shipped_base_version_change_adopts_the_base(self, slots: dict[str, Path]) -> None:
        """Local descends from base v1; v2 ships; nothing retained -> adopt_base (registry mission policy)."""
        old_base = _dump(_stuck_model(0))
        _write_pair(
            slots["local"],
            "stuck",
            _stuck_model(0),
            source="local",
            base_version="1",
            base_sha256=hashlib.sha256(old_base).hexdigest(),
            n_local_extensions=1,
        )
        _write_pair(slots["base"], "stuck", _stuck_model(1), version="2")

        from kenaz_ml.app import AppState

        state = AppState()
        state.refresh_registry(LocalModelStore(base_dir=slots["local"]))

        result = state.refresh_results["stuck"]
        assert result["reason"] == "base_version_changed"
        assert result["action"] == "adopt_base"
        assert result["ok"] is True

    def test_refresh_failure_does_not_stop_startup(
        self, slots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import kenaz_ml.modelstore.registry as registry

        def boom(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("refresh exploded")

        monkeypatch.setattr(registry, "refresh_all", boom)
        with _client() as client:
            assert client.get("/health").status_code == 200
