"""harness-recommendation-models-01MSK2RM WP04 — hot reload, shadow write site, and the flip on eligible.

Spec User Story 4 (scenarios 1-3), FR-011..FR-013, NFR-003, SC-005/SC-006,
and D-B1's automatic, separable, persisted, audit-visible flip.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kenaz_ml import config
from kenaz_ml.advice import dispatch as dispatch_mod
from kenaz_ml.advice import shadow
from kenaz_ml.advice.contracts import contract_for
from kenaz_ml.advice.dispatch import (
    AUDIT_FLIP,
    AUDIT_RETRAIN,
    DispatchTable,
    after_retrain,
    apply_flip,
    build_table,
    reload_kind,
)
from kenaz_ml.advice.shadow import SHADOW_WRITER, MetricsWrite, graduation_check, read_shadow_log
from kenaz_ml.advice.training import AdviceTrainingScheduler, train_kind
from kenaz_ml.modelstore.registry import read_manifest
from tests.fixtures.advice_labels import DAY_MS, T0, push, seed_decisions, separable_rows

KIND = "compact_now"
CONTRACT = contract_for(KIND)
assert CONTRACT is not None
FEATURES = {name: 0.5 for name in CONTRACT.names}
NOW = T0 + 30 * DAY_MS
ELIGIBLE = [(120, "accepted", True), (30, "dismissed", True), (90, "accepted", False), (60, "dismissed", False)]


class FakeStore:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str, int]] = []
        self._lock = threading.Lock()

    def insert_ml_event(self, kind: str, endpoint: str, routing: str, latency_ms: int) -> None:
        with self._lock:
            self.events.append((kind, endpoint, routing, latency_ms))

    def commit(self) -> None:
        pass


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    base = tmp_path / "base"
    monkeypatch.setattr(config, "base_models_dir", lambda: base)
    return {"models": config.models_dir(), "retained": config.retained_data_dir(), "base": base}


def _client(table: DispatchTable) -> TestClient:
    from kenaz_ml.app import AppState
    from kenaz_ml.routes import register_routes

    state = AppState()
    state.dispatch_table = table
    app = FastAPI()
    register_routes(app, state)
    return TestClient(app)


def _ask(client: TestClient) -> Any:
    return client.post(
        f"/v1/recommend/{KIND}",
        json={"features": FEATURES, "feature_contract_version": CONTRACT.service_version, "session_id": "s1"},
    )


def _train(now: int, n: int = 80, seed: int = 0, start: int | None = None) -> Any:
    push(
        KIND,
        separable_rows(KIND, n, start_ts=start or now - DAY_MS, seed=seed, shown=False),
        config.retained_data_dir(),
    )
    outcome = train_kind(KIND, clock=lambda: now)
    assert outcome.trained, outcome
    return outcome


def _mark_classic() -> None:
    assert shadow.update_manifest_metrics(KIND, {"serving_backend": "classic", "rung": "R3"}).ok


# ---------------------------------------------------------------------------
# Scenario 1 / SC-006 / NFR-003 — the next request is the new generation, nothing dropped
# ---------------------------------------------------------------------------


def test_next_request_after_retrain_is_the_new_generation(dirs: dict[str, Path]) -> None:
    _train(T0)
    _mark_classic()
    table = build_table(dirs["models"], dirs["base"])
    client = _client(table)
    assert _ask(client).json()["generation"] == "1"

    outcome = _train(T0 + 2 * DAY_MS, seed=1, start=T0 + DAY_MS)
    assert outcome.manifest.metrics["serving_backend"] == "classic"  # carried forward by the retrain
    report = after_retrain(table, KIND, outcome, now_ms=T0 + 2 * DAY_MS, base_dir=dirs["base"])
    assert report["reload"].reloaded
    body = _ask(client).json()
    assert body["generation"] == "2" and body["backend"] == "classic" and body["rung"] == "R3"


def test_zero_dropped_requests_across_a_reload(dirs: dict[str, Path]) -> None:
    _train(T0)
    _mark_classic()
    table = build_table(dirs["models"], dirs["base"])
    client = _client(table)
    outcome_box: dict[str, Any] = {}

    def retrain() -> None:
        outcome = _train(T0 + 2 * DAY_MS, seed=2, start=T0 + DAY_MS)
        after_retrain(table, KIND, outcome, now_ms=T0 + 2 * DAY_MS, base_dir=dirs["base"])
        outcome_box["done"] = True

    worker = threading.Thread(target=retrain)
    worker.start()
    statuses, generations = [], []
    while worker.is_alive() or len(statuses) < 20:
        resp = _ask(client)
        statuses.append(resp.status_code)
        generations.append(resp.json().get("generation"))
    worker.join()
    assert outcome_box.get("done")
    assert set(statuses) == {200}, statuses
    assert set(generations) <= {"1", "2"}
    assert generations == sorted(generations)  # never back to the old one once swapped
    assert _ask(client).json()["generation"] == "2"


# ---------------------------------------------------------------------------
# Scenario 2 / FR-012 / SC-005 — failed retrains leave the prior generation serving
# ---------------------------------------------------------------------------


def test_failed_write_keeps_serving_and_fires_nothing(dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    _train(T0)
    _mark_classic()
    table = build_table(dirs["models"], dirs["base"])
    client = _client(table)
    push(KIND, separable_rows(KIND, 80, start_ts=T0 + DAY_MS, seed=3, shown=False), dirs["retained"])

    monkeypatch.setattr(
        "kenaz_ml.modelstore.registry.write_manifest", lambda *a, **k: (_ for _ in ()).throw(OSError("full"))
    )
    fired: list[str] = []
    sched = AdviceTrainingScheduler(
        kinds=(KIND,), clock=lambda: T0 + 3 * DAY_MS, on_trained=lambda k, o: fired.append(k)
    )
    result = sched.check_and_retrain()[KIND]
    assert not result.trained  # type: ignore[union-attr]
    assert fired == []
    assert all(_ask(client).json()["generation"] == "1" for _ in range(5))


def test_single_class_decline_keeps_serving(dirs: dict[str, Path]) -> None:
    _train(T0)
    _mark_classic()
    table = build_table(dirs["models"], dirs["base"])
    client = _client(table)
    # Enough new labels, but a retrain must still be a *fresh* fit: make it fail by
    # removing the retained set's second class in a rebuilt set.
    from kenaz_ml.modelstore.registry import reset_retained

    reset_retained(KIND, contract=CONTRACT)
    push(
        KIND,
        [dict(r, user_action="accepted") for r in separable_rows(KIND, 60, start_ts=T0 + DAY_MS, seed=4)],
        dirs["retained"],
    )
    outcome = train_kind(KIND, clock=lambda: T0 + 3 * DAY_MS)
    assert outcome.reason == "single_class"
    assert _ask(client).json()["generation"] == "1"


def test_refused_artifact_is_never_swapped_in(dirs: dict[str, Path]) -> None:
    _train(T0)
    _mark_classic()
    table = build_table(dirs["models"], dirs["base"])
    client = _client(table)
    outcome = _train(T0 + 2 * DAY_MS, seed=5, start=T0 + DAY_MS)
    artifact = dirs["models"] / f"{KIND}.joblib"
    artifact.write_bytes(artifact.read_bytes()[:-10] + b"tampered!!")  # digest no longer matches
    result = reload_kind(table, KIND, expected_generation=outcome.generation)
    assert not result.reloaded
    statuses = [_ask(client) for _ in range(5)]
    assert all(r.status_code == 200 and r.json()["generation"] == "1" for r in statuses)


# ---------------------------------------------------------------------------
# Scenario 3 / FR-013 — audit row naming the generation
# ---------------------------------------------------------------------------


def test_every_retrain_writes_an_audit_row(dirs: dict[str, Path]) -> None:
    store = FakeStore()
    table = build_table(dirs["models"], dirs["base"])
    outcome = _train(T0)
    after_retrain(table, KIND, outcome, now_ms=T0, store=store, base_dir=dirs["base"])
    assert store.events[0][:3] == (AUDIT_RETRAIN, f"advice/{KIND}@1", "local")
    assert all(e[0].startswith("advice_") for e in store.events)  # never the workbench "retrain" kind


# ---------------------------------------------------------------------------
# D-B1 — the flip on eligible
# ---------------------------------------------------------------------------


def _eligible_generation(dirs: dict[str, Path]) -> Any:
    outcome = _train(T0)
    seed_decisions(KIND, dirs["retained"], ELIGIBLE)
    return outcome


def test_eligible_flips_and_the_next_response_is_classic(dirs: dict[str, Path]) -> None:
    store = FakeStore()
    table = build_table(dirs["models"], dirs["base"])
    client = _client(table)
    outcome = _eligible_generation(dirs)
    assert _ask(client).json()["refusal"]["reason"] == "kind_not_served"

    report = after_retrain(table, KIND, outcome, now_ms=NOW, store=store, base_dir=dirs["base"])
    assert report["graduation"].verdict == "eligible"
    assert report["flip"].flipped
    body = _ask(client).json()
    assert body["backend"] == "classic" and body["rung"] == "R3" and body["generation"] == "1"

    metrics = read_manifest(dirs["models"] / f"{KIND}.json").manifest.metrics
    assert metrics["serving_backend"] == "classic"
    assert metrics["verdict"] == "eligible"
    assert metrics["flipped_at_ms"] == NOW and metrics["flipped_generation"] == 1
    assert [e[0] for e in store.events] == [AUDIT_RETRAIN, AUDIT_FLIP]
    assert store.events[1][1] == f"advice/{KIND}@1"


def test_flip_survives_a_restart(dirs: dict[str, Path]) -> None:
    table = build_table(dirs["models"], dirs["base"])
    outcome = _eligible_generation(dirs)
    after_retrain(table, KIND, outcome, now_ms=NOW, base_dir=dirs["base"])
    restarted = build_table(dirs["models"], dirs["base"])  # a fresh process populating from manifests
    assert _ask(_client(restarted)).json()["backend"] == "classic"


def test_manifest_write_failure_means_no_flip(dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    table = build_table(dirs["models"], dirs["base"])
    outcome = _eligible_generation(dirs)
    result = graduation_check(KIND, now_ms=NOW)
    assert result.verdict == "eligible"
    monkeypatch.setattr(shadow, "update_manifest_metrics", lambda *a, **k: MetricsWrite(False, reason="disk"))
    flip = apply_flip(table, KIND, result, now_ms=NOW, base_dir=dirs["base"])
    assert not flip.flipped and "manifest write failed" in flip.reason
    assert _ask(_client(table)).json()["refusal"]["reason"] == "kind_not_served"
    assert "serving_backend" not in read_manifest(dirs["models"] / f"{KIND}.json").manifest.metrics
    assert outcome.trained


def test_swap_failure_after_write_rolls_the_manifest_back(
    dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    table = build_table(dirs["models"], dirs["base"])
    _eligible_generation(dirs)
    result = graduation_check(KIND, now_ms=NOW)
    monkeypatch.setattr(
        dispatch_mod, "reload_kind", lambda *a, **k: dispatch_mod.ReloadResult(KIND, False, reason="simulated")
    )
    flip = apply_flip(table, KIND, result, now_ms=NOW, base_dir=dirs["base"])
    assert not flip.flipped
    metrics = read_manifest(dirs["models"] / f"{KIND}.json").manifest.metrics
    assert "serving_backend" not in metrics and "flipped_at_ms" not in metrics


def test_the_flip_is_separable_from_reload(dirs: dict[str, Path]) -> None:
    table = build_table(dirs["models"], dirs["base"])
    outcome = _eligible_generation(dirs)
    report = after_retrain(table, KIND, outcome, now_ms=NOW, base_dir=dirs["base"], flip=False)
    assert report["reload"].reloaded and "flip" not in report
    assert table.snapshot()[KIND].shadow_model is not None  # reloaded into the shadow rung
    metrics = read_manifest(dirs["models"] / f"{KIND}.json").manifest.metrics
    assert metrics["verdict"] == "eligible" and "serving_backend" not in metrics


def test_not_eligible_is_recorded_without_a_flip(dirs: dict[str, Path]) -> None:
    table = build_table(dirs["models"], dirs["base"])
    outcome = _train(T0)
    report = after_retrain(table, KIND, outcome, now_ms=NOW, base_dir=dirs["base"])
    assert report["graduation"].verdict == "not_eligible" and "flip" not in report
    metrics = read_manifest(dirs["models"] / f"{KIND}.json").manifest.metrics
    assert metrics["verdict"] == "not_eligible" and metrics["verdict_reason"] == "no shadow log"


# ---------------------------------------------------------------------------
# T017 step 4 — the shadow write site
# ---------------------------------------------------------------------------


def test_shadow_rung_request_writes_one_record_and_still_refuses(dirs: dict[str, Path]) -> None:
    manifest = _train(T0).manifest
    table = build_table(dirs["models"], dirs["base"])
    resp = _ask(_client(table))
    assert resp.status_code == 422 and resp.json()["refusal"]["reason"] == "kind_not_served"
    assert SHADOW_WRITER.flush()
    shadows = read_shadow_log(KIND).shadows
    assert len(shadows) == 1
    assert shadows[0]["generation"] == manifest.version and shadows[0]["session_id"] == "s1"
    assert shadows[0]["vector_hash"] == shadow.vector_hash(KIND, [FEATURES[n] for n in CONTRACT.names])


def test_flipped_and_untrained_kinds_write_no_shadow(dirs: dict[str, Path]) -> None:
    table = build_table(dirs["models"], dirs["base"])
    assert _ask(_client(table)).status_code == 422  # no model at all: heuristic-only
    _train(T0)
    _mark_classic()
    table = build_table(dirs["models"], dirs["base"])
    assert _ask(_client(table)).status_code == 200  # flipped: authoritative, not shadow
    assert SHADOW_WRITER.flush()
    assert read_shadow_log(KIND).shadows == []


def test_shadow_failure_never_fails_the_request(dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    _train(T0)
    table = build_table(dirs["models"], dirs["base"])

    def explode(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("log unavailable")

    monkeypatch.setattr(shadow, "enqueue_shadow", explode)
    resp = _ask(_client(table))
    assert resp.status_code == 422 and resp.json()["refusal"]["reason"] == "kind_not_served"
