"""two-client-engine-01MSK2EN WP03 — /v1/recommend/{kind} dispatch.

Covers spec User Story 2's scenarios: a served (fixture) kind answers with the
full FR-004 envelope naming its backend truthfully; unserved trio kinds refuse
``kind_not_served``; a laya-configured kind refuses without substitution; a
contract mismatch refuses that kind only; confidence is validated, never
clamped; reads are snapshot-consistent under concurrent swaps; and the
workbench namespace is never aliased.
"""

from __future__ import annotations

import ast
import hashlib
import io
import threading
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kenaz_ml import config
from kenaz_ml.advice import dispatch as dispatch_mod
from kenaz_ml.advice.contracts import KIND_IDS, contract_for
from kenaz_ml.advice.dispatch import (
    SHIPPED_BACKENDS,
    ClassicBackend,
    DispatchEntry,
    DispatchTable,
    build_table,
)
from kenaz_ml.modelstore.registry import Manifest, Provenance, Runtime, running_sklearn_version, write_manifest
from tests.fixtures.advice_backends import (
    FIXTURE_KIND,
    FIXTURE_LAYA_KIND,
    ThresholdFixtureBackend,
    fixture_contract,
    fixture_features,
    register_fixtures,
    served_entry,
)

FR004_FIELDS = {
    "decision",
    "score",
    "confidence",
    "kind_id",
    "feature_contract_version",
    "model",
    "rung",
    "backend",
    "model_id_sha8",
    "checkpoint_provenance",
    "generation",
    "unbenchmarked",
}


@pytest.fixture
def slots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    base = tmp_path / "base_models"
    base.mkdir()
    monkeypatch.setattr(config, "base_models_dir", lambda: base)
    return {"local": config.models_dir(), "base": base}


def _app(table: DispatchTable) -> tuple[TestClient, Any]:
    from kenaz_ml.app import AppState
    from kenaz_ml.routes import register_routes

    state = AppState()
    state.dispatch_table = table
    app = FastAPI()
    register_routes(app, state)
    return TestClient(app), state


@pytest.fixture
def table(slots: dict[str, Path]) -> DispatchTable:
    t = build_table(slots["local"], slots["base"])
    return t


def _post(client: TestClient, kind: str, features: dict[str, float], version: str) -> Any:
    return client.post(f"/v1/recommend/{kind}", json={"features": features, "feature_contract_version": version})


def _trio_features(kind: str) -> dict[str, float]:
    return {name: 0.0 for name in contract_for(kind).names}  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Scenario 1 — a served kind answers, fully shaped and honest
# ---------------------------------------------------------------------------


def test_fixture_kind_returns_the_full_envelope(table: DispatchTable) -> None:
    register_fixtures(table)
    client, _ = _app(table)

    resp = _post(client, FIXTURE_KIND, fixture_features(0.9), fixture_contract().service_version)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == FR004_FIELDS
    assert body["decision"] is True and body["score"] is None
    assert body["confidence"] == 80
    assert body["backend"] == "heuristic"
    assert body["model"] == "heuristic/threshold-fixture@1"
    assert body["unbenchmarked"] is True
    assert body["kind_id"] == FIXTURE_KIND
    assert body["feature_contract_version"] == fixture_contract().service_version
    assert body["generation"] == "0"
    assert body["model_id_sha8"] is None


# ---------------------------------------------------------------------------
# Scenario 2 — unserved trio kinds refuse kind_not_served
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KIND_IDS)
def test_unserved_trio_kind_refuses_kind_not_served(table: DispatchTable, kind: str) -> None:
    client, _ = _app(table)
    resp = _post(client, kind, _trio_features(kind), contract_for(kind).service_version)  # type: ignore[union-attr]
    assert resp.status_code == 422
    body = resp.json()
    # The harness's mlsidecar client keys ErrKindNotServed off a top-level
    # {"error": "kind_not_served"} — the stable, typed, distinguishable shape.
    assert body["error"] == "kind_not_served"
    refusal = body["refusal"]
    assert refusal["reason"] == "kind_not_served"
    assert refusal["kind_id"] == kind
    assert kind in refusal["detail"]


def test_real_app_startup_registers_the_trio_unserved(slots: dict[str, Path]) -> None:
    from kenaz_ml.app import create_app

    with TestClient(create_app()) as client:
        for kind in KIND_IDS:
            resp = _post(client, kind, _trio_features(kind), contract_for(kind).service_version)  # type: ignore[union-attr]
            assert resp.status_code == 422
            assert resp.json()["refusal"]["reason"] == "kind_not_served"


# ---------------------------------------------------------------------------
# Scenario 3 — laya-configured kind refuses, never substitutes
# ---------------------------------------------------------------------------


def test_laya_kind_refuses_without_substitution(table: DispatchTable) -> None:
    register_fixtures(table)
    client, _ = _app(table)
    resp = _post(client, FIXTURE_LAYA_KIND, fixture_features(), fixture_contract(FIXTURE_LAYA_KIND).service_version)
    assert resp.status_code == 422
    refusal = resp.json()["refusal"]
    assert refusal["reason"] == "laya_backend_not_installed"
    assert refusal["detail"] == "kind unavailable: laya backend not installed"


# ---------------------------------------------------------------------------
# Scenario 4 — contract mismatch refuses only that kind
# ---------------------------------------------------------------------------


def test_contract_mismatch_is_per_kind(table: DispatchTable) -> None:
    register_fixtures(table)
    table.replace_entry(served_entry(ThresholdFixtureBackend(fixture_name="other"), kind_id="fixture_other"))
    client, _ = _app(table)

    bad = _post(client, FIXTURE_KIND, fixture_features(), "0000000000000000")
    assert bad.status_code == 422
    refusal = bad.json()["refusal"]
    assert refusal["reason"] == "contract_mismatch"
    assert "0000000000000000" in refusal["detail"]
    assert fixture_contract().service_version in refusal["detail"]

    ok = _post(client, "fixture_other", fixture_features(), fixture_contract("fixture_other").service_version)
    assert ok.status_code == 200
    again = _post(client, FIXTURE_KIND, fixture_features(), fixture_contract().service_version)
    assert again.status_code == 200


def test_features_must_match_the_contract_by_name(table: DispatchTable) -> None:
    register_fixtures(table)
    client, _ = _app(table)
    resp = _post(client, FIXTURE_KIND, {"signal": 1.0, "surplus": 2.0}, fixture_contract().service_version)
    assert resp.status_code == 422
    refusal = resp.json()["refusal"]
    assert refusal["reason"] == "features_mismatch"
    assert "noise" in refusal["detail"] and "surplus" in refusal["detail"]


def test_unknown_kind_and_body_kind_mismatch(table: DispatchTable) -> None:
    client, _ = _app(table)
    resp = _post(client, "no_such_kind", {}, "x")
    assert resp.status_code == 422
    assert resp.json()["refusal"]["reason"] == "unknown_kind"

    register_fixtures(table)
    resp = client.post(
        f"/v1/recommend/{FIXTURE_KIND}",
        json={
            "features": fixture_features(),
            "feature_contract_version": fixture_contract().service_version,
            "kind_id": "branch_now",
        },
    )
    assert resp.json()["refusal"]["reason"] == "kind_id_mismatch"


# ---------------------------------------------------------------------------
# Scenario 6 — confidence validated, never clamped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [-1, 101, 100.5, 99.0, True, None])
def test_out_of_range_confidence_fails_the_request(table: DispatchTable, confidence: Any) -> None:
    register_fixtures(table, ThresholdFixtureBackend(confidence=confidence))
    client, _ = _app(table)
    resp = _post(client, FIXTURE_KIND, fixture_features(), fixture_contract().service_version)
    assert resp.status_code == 422
    assert resp.json()["refusal"]["reason"] == "confidence_out_of_range"


@pytest.mark.parametrize("confidence", [0, 100])
def test_boundary_confidence_is_served_verbatim(table: DispatchTable, confidence: int) -> None:
    register_fixtures(table, ThresholdFixtureBackend(confidence=confidence))
    client, _ = _app(table)
    resp = _post(client, FIXTURE_KIND, fixture_features(), fixture_contract().service_version)
    assert resp.status_code == 200
    assert resp.json()["confidence"] == confidence


def test_no_clamping_in_the_dispatch_module() -> None:
    source = Path(dispatch_mod.__file__).read_text()
    assert "clamp(" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("min", "max"):
            # the only min/max allowed is the classic backend's p vs 1-p choice (not confidence clamping)
            pytest.fail(f"min/max call at line {node.lineno} — confidence must never be clamped")


# ---------------------------------------------------------------------------
# Snapshot consistency under concurrent swaps
# ---------------------------------------------------------------------------


def test_reads_are_snapshot_consistent_under_concurrent_swaps(table: DispatchTable) -> None:
    a = ThresholdFixtureBackend(fixture_name="gen-a", version="1", generation="1", confidence=11)
    b = ThresholdFixtureBackend(fixture_name="gen-b", version="2", generation="2", confidence=22)
    table.replace_entry(served_entry(a))
    stop = threading.Event()

    def swapper() -> None:
        flip = False
        while not stop.is_set():
            table.replace_entry(served_entry(b if flip else a))
            flip = not flip

    thread = threading.Thread(target=swapper)
    thread.start()
    try:
        request = dispatch_mod.RecommendRequest(
            features=fixture_features(), feature_contract_version=fixture_contract().service_version
        )
        seen = set()
        for _ in range(2000):
            r = dispatch_mod.dispatch(table.snapshot(), FIXTURE_KIND, request)
            seen.add((r.model, r.generation, r.confidence))
    finally:
        stop.set()
        thread.join()
    assert seen <= {("heuristic/gen-a@1", "1", 11), ("heuristic/gen-b@2", "2", 22)}


# ---------------------------------------------------------------------------
# Namespace disjointness (D-A3)
# ---------------------------------------------------------------------------


def test_workbench_names_are_not_recommend_kinds(table: DispatchTable) -> None:
    client, _ = _app(table)
    resp = _post(client, "stuck", {}, "x")
    assert resp.status_code == 422
    assert resp.json()["refusal"]["reason"] == "unknown_kind"

    for name in ("stuck", "activity", "workflow", "duration", "quality", "suggest"):
        with pytest.raises(ValueError):
            table.replace_entry(DispatchEntry(kind_id=name, contract=None))


def test_startup_scan_never_registers_a_workbench_name(slots: dict[str, Path]) -> None:
    (slots["local"] / "stuck.json").write_text('{"name": "stuck", "metrics": {"serving_backend": "classic"}}')
    table = build_table(slots["local"], slots["base"])
    assert "stuck" not in table.snapshot()


# ---------------------------------------------------------------------------
# No shipped heuristics (A3.2) — structural
# ---------------------------------------------------------------------------


def test_only_classic_ships_as_a_backend_implementation() -> None:
    assert {"classic": ClassicBackend} == SHIPPED_BACKENDS


def test_no_heuristic_backend_under_src_advice() -> None:
    advice_dir = Path(dispatch_mod.__file__).parent
    for path in advice_dir.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign | ast.Assign):
                        value = stmt.value
                        if isinstance(value, ast.Name) and value.id == "BACKEND_HEURISTIC":
                            pytest.fail(f"{path.name}:{node.name} declares itself a heuristic backend")
                        if isinstance(value, ast.Constant) and value.value == "heuristic":
                            pytest.fail(f"{path.name}:{node.name} declares itself a heuristic backend")
        assert '"heuristic/' not in path.read_text(), f"{path.name} builds a heuristic model label"


# ---------------------------------------------------------------------------
# Classic backend through the registry (FR-003), and manifest-named backends
# ---------------------------------------------------------------------------


def _write_trio_pair(slot: Path, kind: str, model: Any, metrics: dict[str, Any], version: str = "3") -> str:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    (slot / f"{kind}.joblib").write_bytes(buf.getvalue())
    digest = hashlib.sha256(buf.getvalue()).hexdigest()
    write_manifest(
        slot / f"{kind}.json",
        Manifest(
            name=kind,
            version=version,
            artifact_sha256=digest,
            provenance=Provenance(training_source="local", n_local_extensions=1),
            runtime=Runtime(sklearn_version=running_sklearn_version() or "", python_version="3.12"),
            feature_contract=contract_for(kind),
            metrics=metrics,
        ),
    )
    return digest


def _fitted_compact_model() -> Any:
    from sklearn.linear_model import LogisticRegression

    n = len(contract_for("compact_now").names)  # type: ignore[union-attr]
    rng = np.random.default_rng(0)
    x = rng.random((40, n))
    y = (x[:, 0] > 0.5).astype(int)
    return LogisticRegression().fit(x, y)


def test_classic_manifest_is_served_by_the_classic_backend(slots: dict[str, Path]) -> None:
    digest = _write_trio_pair(
        slots["local"], "compact_now", _fitted_compact_model(), {"serving_backend": "classic", "rung": "classic-v1"}
    )
    table = build_table(slots["local"], slots["base"])
    client, _ = _app(table)

    features = {name: 0.9 for name in contract_for("compact_now").names}  # type: ignore[union-attr]
    resp = _post(client, "compact_now", features, contract_for("compact_now").service_version)  # type: ignore[union-attr]

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["backend"] == "classic"
    assert body["model"] == "classic/compact_now@3"
    assert body["generation"] == "3"
    assert body["model_id_sha8"] == digest[:8]
    assert body["checkpoint_provenance"] == "local"
    assert body["rung"] == "classic-v1"
    assert body["unbenchmarked"] is True
    assert isinstance(body["decision"], bool)
    assert 0 <= body["confidence"] <= 100


@pytest.mark.parametrize(
    ("metrics", "reason"),
    [
        ({"serving_backend": "laya"}, "laya_backend_not_installed"),
        ({"serving_backend": "heuristic"}, "kind_not_served"),
        ({}, "kind_not_served"),
        ({"serving_backend": "quantum"}, "backend_unknown"),
    ],
)
def test_manifest_named_backend_decides_the_entry(slots: dict[str, Path], metrics: dict, reason: str) -> None:
    _write_trio_pair(slots["local"], "compact_now", _fitted_compact_model(), metrics)
    entry = build_table(slots["local"], slots["base"]).snapshot()["compact_now"]
    assert not entry.available
    assert entry.reason == reason


def test_tampered_trio_artifact_refuses_that_kind_only(slots: dict[str, Path]) -> None:
    _write_trio_pair(slots["local"], "compact_now", _fitted_compact_model(), {"serving_backend": "classic"})
    (slots["local"] / "compact_now.joblib").write_bytes(b"tampered")
    snapshot = build_table(slots["local"], slots["base"]).snapshot()
    assert snapshot["compact_now"].reason == "kind_not_served"
    assert "refused by the registry" in (snapshot["compact_now"].detail or "")
    assert set(KIND_IDS) <= set(snapshot)


# ---------------------------------------------------------------------------
# NFR-002 / FR-025 — latency
# ---------------------------------------------------------------------------


def _p95(samples: list[float]) -> float:
    ordered = sorted(samples)
    return ordered[max(0, int(len(ordered) * 0.95) - 1)]


def test_fixture_and_refusal_p95_under_50ms_and_ring_is_readable(table: DispatchTable) -> None:
    register_fixtures(table)
    client, state = _app(table)
    version = fixture_contract().service_version

    served, refused = [], []
    for _ in range(200):
        t0 = time.perf_counter()
        assert _post(client, FIXTURE_KIND, fixture_features(), version).status_code == 200
        served.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        assert (
            _post(
                client, "branch_now", _trio_features("branch_now"), contract_for("branch_now").service_version
            ).status_code
            == 422
        )  # type: ignore[union-attr]
        refused.append((time.perf_counter() - t0) * 1000)

    assert _p95(served) < 50
    assert _p95(refused) < 50

    p95 = state.dispatch_table.latency_p95_ms(FIXTURE_KIND)
    assert p95 is not None and 0 <= p95 < 50
    assert len(state.dispatch_table.latency_samples(FIXTURE_KIND)) == 200
    assert state.dispatch_table.latency_p95_ms("branch_now") is not None
    assert state.dispatch_table.latency_p95_ms("never_called") is None


def test_latency_ring_is_bounded() -> None:
    t = DispatchTable()
    for i in range(dispatch_mod.LATENCY_RING_SIZE + 50):
        t.record_latency("k", float(i))
    assert len(t.latency_samples("k")) == dispatch_mod.LATENCY_RING_SIZE


def test_unknown_kinds_never_grow_the_latency_rings(table: DispatchTable) -> None:
    # Review fix: the {kind} path segment is caller-controlled; recording a
    # ring per unknown kind made the FR-025 measurement unbounded in ring count.
    client, state = _app(table)
    for i in range(20):
        assert _post(client, f"junk_{i}", {}, "x").status_code == 422
    assert all(state.dispatch_table.latency_p95_ms(f"junk_{i}") is None for i in range(20))
    assert not state.dispatch_table._latency


# ---------------------------------------------------------------------------
# Optional exact shadow-join key on the request (ruled 2026-09-30)
# ---------------------------------------------------------------------------


def test_join_key_fields_are_optional_and_carried_on_the_request(table: DispatchTable) -> None:
    register_fixtures(table)
    client, _ = _app(table)
    version = fixture_contract().service_version
    base = {"features": fixture_features(), "feature_contract_version": version}

    assert client.post(f"/v1/recommend/{FIXTURE_KIND}", json=base).status_code == 200  # absent-tolerated
    keyed = {**base, "features_hash": "h-abc", "ts": 1_700_000_000_000}
    assert client.post(f"/v1/recommend/{FIXTURE_KIND}", json=keyed).status_code == 200

    parsed = dispatch_mod.RecommendRequest.model_validate(keyed)
    assert (parsed.features_hash, parsed.ts) == ("h-abc", 1_700_000_000_000)
    bare = dispatch_mod.RecommendRequest.model_validate(base)
    assert (bare.features_hash, bare.ts) == (None, None)


@pytest.mark.parametrize("bad", [{"ts": 0}, {"ts": -5}, {"features_hash": ""}])
def test_malformed_join_key_is_rejected(table: DispatchTable, bad: dict) -> None:
    register_fixtures(table)
    client, _ = _app(table)
    body = {"features": fixture_features(), "feature_contract_version": fixture_contract().service_version, **bad}
    assert client.post(f"/v1/recommend/{FIXTURE_KIND}", json=body).status_code == 422
