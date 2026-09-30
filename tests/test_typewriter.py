"""The typewriter spec is the single source of the engine's wire contract.

These tests pin the three things generation alone cannot: that the checked-in
generated files match the spec, that the engine's own constants agree with the
spec's vocabularies, and that the shared fixtures (which the Go module's tests
also decode) are what a running engine actually sends.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from kenaz_ml import features_push
from kenaz_ml.advice import contracts, dispatch, label_log
from kenaz_ml.typewriter import models, vocab

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = sorted((ROOT / "typewriter" / "testdata").glob("*.json"))


def test_generated_files_match_the_spec() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_typewriter.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_contract_versions_match_the_engine_recipe() -> None:
    for kind_id, names in vocab.KIND_FEATURES.items():
        contract = contracts.contract_for(kind_id)
        assert contract is not None
        assert contract.names == names
        assert contract.service_version == vocab.KIND_CONTRACT_VERSIONS[kind_id]


def test_engine_constants_agree_with_the_spec_vocabularies() -> None:
    def constants(module: Any, prefix: str) -> set[str]:
        return {v for k, v in vars(module).items() if k.startswith(prefix) and isinstance(v, str)}

    assert tuple(dispatch.BACKENDS) == vocab.BACKENDS
    assert constants(dispatch, "REASON_") == set(vocab.RECOMMEND_REFUSAL_REASONS)
    assert set(label_log.USER_ACTION_LABELS) == set(vocab.USER_ACTIONS)
    assert constants(label_log, "REASON_") == set(vocab.LABEL_BATCH_REFUSAL_REASONS)
    assert constants(label_log, "ROW_") == set(vocab.LABEL_ROW_REFUSAL_REASONS)
    assert constants(features_push, "REASON_") == set(vocab.FEATURES_REFUSAL_REASONS)


def test_feature_models_carry_the_contract_names_in_order() -> None:
    assert set(models.KIND_FEATURE_MODELS) == set(vocab.KIND_FEATURES)
    for kind_id, model in models.KIND_FEATURE_MODELS.items():
        assert tuple(model.model_fields) == vocab.KIND_FEATURES[kind_id]


def _without_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_nulls(v) for v in value]
    return value


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_is_exactly_its_schema(path: Path) -> None:
    """Validates, and carries no key the schema does not declare."""
    payload = json.loads(path.read_text())
    model = models.SCHEMAS[path.name.split(".", 1)[0]]
    dumped = model.model_validate(payload).model_dump(mode="json")
    assert _without_nulls(dumped) == _without_nulls(payload)


def _keys(value: Any) -> Any:
    """The key structure of a payload, ignoring values (lists by first element)."""
    if isinstance(value, dict):
        return {k: _keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_keys(value[0])] if value else []
    return None


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("KENAZ_ML_INSTALL_ROOT", str(tmp_path / "install"))
    from kenaz_ml.app import create_app

    with TestClient(create_app()) as c:
        yield c


def test_fixtures_have_the_shape_a_running_engine_sends(client: TestClient) -> None:
    def fixture(name: str) -> Any:
        return json.loads((ROOT / "typewriter" / "testdata" / f"{name}.json").read_text())

    assert _keys(client.get("/health").json()) == _keys(fixture("HealthResponse.local"))
    assert _keys(client.get("/v1/contracts").json()) == _keys(fixture("ContractsResponse.unserved"))
    lease = client.post("/v1/clients/lease", json=fixture("LeaseRequest.harness"))
    assert _keys(lease.json()) == _keys(fixture("LeaseResponse.harness"))
    refusal = client.post("/v1/recommend/branch_now", json=fixture("RecommendRequest.branch_now"))
    assert refusal.status_code == dispatch.REFUSAL_STATUS_CODE
    assert refusal.json() == fixture("RecommendRefusal.kind_not_served")
    labels = client.post("/v1/labels/branch_now", json=fixture("LabelBatchRequest.two_rows"))
    assert labels.json() == fixture("LabelBatchResponse.one_refused")
