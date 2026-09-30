"""two-client-engine-01MSK2EN WP03 — GET /v1/contracts (FR-010)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kenaz_ml import config
from kenaz_ml.advice.contracts import KIND_IDS, contract_for
from kenaz_ml.advice.dispatch import build_table
from tests.fixtures.advice_backends import FIXTURE_KIND, FIXTURE_LAYA_KIND, register_fixtures


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from kenaz_ml.app import AppState
    from kenaz_ml.routes import register_routes

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    base = tmp_path / "base"
    monkeypatch.setattr(config, "base_models_dir", lambda: base)
    state = AppState()
    state.dispatch_table = build_table(config.models_dir(), base)
    register_fixtures(state.dispatch_table)
    app = FastAPI()
    register_routes(app, state)
    return TestClient(app)


def test_every_registered_kind_is_published_in_order(client: TestClient) -> None:
    kinds = client.get("/v1/contracts").json()["kinds"]
    ids = [k["kind_id"] for k in kinds]
    assert ids[: len(KIND_IDS)] == list(KIND_IDS)
    assert FIXTURE_KIND in ids and FIXTURE_LAYA_KIND in ids


@pytest.mark.parametrize("kind", KIND_IDS)
def test_trio_contract_shape_and_status(client: TestClient, kind: str) -> None:
    entry = next(k for k in client.get("/v1/contracts").json()["kinds"] if k["kind_id"] == kind)
    contract = contract_for(kind)
    assert entry["names"] == list(contract.names)  # ordered, not a set
    assert entry["dtypes"] == list(contract.dtypes)
    assert entry["version"] == contract.service_version
    assert entry["supported_versions"] == [contract.service_version]
    assert entry["available"] is False
    assert entry["reason"] == "kind_not_served"
    assert entry["detail"]


def test_served_and_laya_status(client: TestClient) -> None:
    kinds = {k["kind_id"]: k for k in client.get("/v1/contracts").json()["kinds"]}
    assert kinds[FIXTURE_KIND]["available"] is True
    assert kinds[FIXTURE_KIND]["backend"] == "heuristic"
    assert kinds[FIXTURE_KIND]["reason"] is None
    assert kinds[FIXTURE_LAYA_KIND]["available"] is False
    assert kinds[FIXTURE_LAYA_KIND]["backend"] == "laya"
    assert kinds[FIXTURE_LAYA_KIND]["reason"] == "laya_backend_not_installed"


def test_cloud_mode_publishes_no_kinds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kenaz_ml.app import create_app
    from kenaz_ml.config import ServingMode

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    with TestClient(create_app(ServingMode.CLOUD)) as c:
        assert c.get("/v1/contracts").json() == {"kinds": []}
        assert (
            c.post("/v1/recommend/branch_now", json={"features": {}, "feature_contract_version": "x"}).json()[
                "refusal"
            ]["reason"]
            == "unknown_kind"
        )
