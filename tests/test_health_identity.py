"""two-client-engine-01MSK2EN WP05 — /health is honest enough to be verified (FR-016, A1).

The "every proper subset fails" assertion is generic over the model's declared
required fields, so a later required field (``laya-serving-and-packs-01MSK2SP``
WP03 may add exactly one) extends it automatically instead of breaking it.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from kenaz_ml import config
from kenaz_ml.advice.contracts import KIND_IDS, contract_for
from kenaz_ml.lifecycle.leases import LIFECYCLE_PROTOCOL
from kenaz_ml.modelstore.registry import Manifest, Provenance, Runtime, running_sklearn_version, write_manifest
from kenaz_ml.routes import HealthResponse

IDENTITY_FIELDS = {
    "product",
    "sidecar_version",
    "contract_versions",
    "exe_path",
    "model_details",
    "device",
    "lifecycle_protocol",
}


@pytest.fixture
def slots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("KENAZ_ML_INSTALL_ROOT", str(tmp_path / "install"))
    base = tmp_path / "base"
    base.mkdir()
    monkeypatch.setattr(config, "base_models_dir", lambda: base)
    return {"local": config.models_dir(), "base": base}


@pytest.fixture
def client(slots: dict[str, Path]) -> TestClient:
    from kenaz_ml.app import create_app

    with TestClient(create_app()) as c:
        yield c


def _valid_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "mode": "local",
        "models": {"stuck": "untrained"},
        "uptime_sec": 1.0,
        "product": "kenaz-ml",
        "sidecar_version": "0.1.0",
        "contract_versions": {},
        "exe_path": "/x",
        "model_details": {"stuck": {"status": "untrained", "slot": "cold_start", "refusal": None}},
        "device": "cpu",
        "lifecycle_protocol": LIFECYCLE_PROTOCOL,
    }


def test_full_identity_field_set_is_present(client: TestClient) -> None:
    body = client.get("/health").json()
    assert set(body) >= IDENTITY_FIELDS
    assert body["product"] == "kenaz-ml"
    assert body["lifecycle_protocol"] == LIFECYCLE_PROTOCOL
    assert body["device"] == "cpu"
    from kenaz_ml import __version__

    assert body["sidecar_version"] == __version__
    for kind in KIND_IDS:
        assert body["contract_versions"][kind] == [contract_for(kind).service_version]  # type: ignore[union-attr]


def test_existing_models_field_is_unchanged(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["mode"] == "local"
    assert body["models"] == {
        "stuck": "untrained",
        "activity": "untrained",
        "workflow": "untrained",
        "duration": "untrained",
        "quality": "ready",
    }
    assert all(isinstance(v, str) for v in body["models"].values())


def test_exe_path_is_this_process(client: TestClient) -> None:
    exe = client.get("/health").json()["exe_path"]
    candidates = {os.path.realpath(sys.executable)}
    if sys.argv and sys.argv[0] and os.path.exists(sys.argv[0]):
        candidates.add(os.path.realpath(sys.argv[0]))
    assert exe in candidates


def test_empty_payload_cannot_be_constructed() -> None:
    with pytest.raises(ValidationError):
        HealthResponse.model_validate({})


def test_every_proper_subset_of_required_fields_fails() -> None:
    payload = _valid_payload()
    required = [name for name, info in HealthResponse.model_fields.items() if info.is_required()]
    assert set(required) >= IDENTITY_FIELDS
    HealthResponse.model_validate(payload)  # the full set is valid
    for size in range(len(required)):
        for kept in itertools.combinations(required, size):
            partial = {k: v for k, v in payload.items() if k in kept or k not in required}
            with pytest.raises(ValidationError):
                HealthResponse.model_validate(partial)


def test_failing_model_carries_the_registry_refusal_text(slots: dict[str, Path]) -> None:
    buf = io.BytesIO()
    joblib.dump({"stand_in": True}, buf)
    artifact = slots["base"] / "stuck.joblib"
    artifact.write_bytes(buf.getvalue())
    write_manifest(
        slots["base"] / "stuck.json",
        Manifest(
            name="stuck",
            version="1",
            artifact_sha256=hashlib.sha256(buf.getvalue()).hexdigest(),
            provenance=Provenance(training_source="base"),
            runtime=Runtime(sklearn_version=running_sklearn_version() or "", python_version="3.12"),
        ),
    )
    artifact.write_bytes(b"tampered after the manifest was written")

    from kenaz_ml.app import create_app

    with TestClient(create_app()) as c:
        body = c.get("/health").json()
    detail = body["model_details"]["stuck"]
    assert body["models"]["stuck"] == "untrained"
    assert detail["slot"] == "cold_start"
    assert detail["refusal"] and "[base]" in detail["refusal"] and "integrity" in detail["refusal"]


def test_health_poll_is_an_implicit_lease(client: TestClient) -> None:
    client.get("/health")
    resp = client.post("/v1/clients/lease", json={"client": "harness", "pid": os.getpid(), "client_version": "0.84.0"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["live_leases"] == 2  # the explicit lease + the implicit health-poll lease
    assert body["lifecycle_protocol"] == LIFECYCLE_PROTOCOL
    assert body["managed"] is False


def test_lease_reports_contract_incompatibility_per_kind(client: TestClient) -> None:
    good = contract_for("branch_now").service_version  # type: ignore[union-attr]
    resp = client.post(
        "/v1/clients/lease",
        json={
            "client": "harness",
            "pid": os.getpid(),
            "client_version": "0.84.0",
            "min_contracts": {"branch_now": good, "compact_now": "ffffffffffffffff", "nope": "x"},
        },
    )
    incompatible = resp.json()["incompatible_kinds"]
    assert set(incompatible) == {"compact_now", "nope"}
    # Incompatibility is reported, never enforced: the kind still dispatches (and refuses only as unserved).
    r = client.post(
        "/v1/recommend/branch_now",
        json={"features": {n: 0.0 for n in contract_for("branch_now").names}, "feature_contract_version": good},  # type: ignore[union-attr]
    )
    assert r.json()["refusal"]["reason"] == "kind_not_served"


def test_lease_rejects_bad_pids(client: TestClient) -> None:
    for pid in (0, -5):
        assert (
            client.post("/v1/clients/lease", json={"client": "h", "pid": pid, "client_version": "1"}).status_code == 422
        )


def test_cloud_mode_health_is_fully_shaped(slots: dict[str, Path]) -> None:
    from kenaz_ml.app import create_app
    from kenaz_ml.config import ServingMode

    with TestClient(create_app(ServingMode.CLOUD)) as c:
        body = c.get("/health").json()
        assert set(body) >= IDENTITY_FIELDS
        assert body["lifecycle_protocol"] == "none"
        assert c.post("/v1/clients/lease", json={"client": "h", "pid": 1, "client_version": "1"}).status_code == 404
