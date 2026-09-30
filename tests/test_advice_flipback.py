"""harness-recommendation-models-01MSK2RM WP05 — flip-back and demotion (spec User Story F, D-B5).

All four User Story F scenarios, the ruled parameter boundaries (29/30 shown,
exactly baseline - 5 pp vs just below, the 21-day window), latency demotion,
no same-generation re-flip, a new generation re-flipping, restart agreement,
and the no-flapping property.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kenaz_ml import config
from kenaz_ml.advice import shadow
from kenaz_ml.advice.contracts import contract_for
from kenaz_ml.advice.dispatch import (
    AUDIT_DEMOTE,
    AUDIT_INSUFFICIENT_DATA,
    DispatchTable,
    after_retrain,
    apply_flip,
    build_table,
    evaluate_flipped_kinds,
    evaluation_hook,
)
from kenaz_ml.advice.shadow import (
    DEMOTION_MARGIN_PP,
    FLIPBACK_MIN_SHOWN,
    MAX_EVAL_WINDOW_DAYS,
    MetricsWrite,
    graduation_check,
)
from kenaz_ml.advice.training import AdviceTrainingScheduler, train_kind
from kenaz_ml.modelstore.registry import read_manifest
from tests.fixtures.advice_labels import DAY_MS, T0, features_for, label_row, push, seed_decisions, separable_rows

KIND = "compact_now"
CONTRACT = contract_for(KIND)
assert CONTRACT is not None
FLIP_AT = T0 + 30 * DAY_MS
#: Heuristic baseline from these rows: 210 / 300 = 0.70.
ELIGIBLE = [(120, "accepted", True), (30, "dismissed", True), (90, "accepted", False), (60, "dismissed", False)]
CLASSIC_ID = f"classic/{KIND}@1"


class FakeStore:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def insert_ml_event(self, kind: str, endpoint: str, routing: str, latency_ms: int) -> None:
        self.events.append((kind, endpoint))

    def commit(self) -> None:
        pass


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Generation 1 trained, graduated ``eligible`` and flipped at FLIP_AT."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    base = tmp_path / "base"
    monkeypatch.setattr(config, "base_models_dir", lambda: base)
    retained = config.retained_data_dir()
    push(KIND, separable_rows(KIND, 80, start_ts=T0 - DAY_MS, shown=False), retained)
    outcome = train_kind(KIND, clock=lambda: T0)
    assert outcome.trained
    seed_decisions(KIND, retained, ELIGIBLE)
    table = build_table(config.models_dir(), base)
    store = FakeStore()
    report = after_retrain(table, KIND, outcome, now_ms=FLIP_AT, store=store, base_dir=base)
    assert report["flip"].flipped
    return {"table": table, "store": store, "retained": retained, "base": base, "models": config.models_dir()}


def _client(table: DispatchTable) -> TestClient:
    from kenaz_ml.app import AppState
    from kenaz_ml.routes import register_routes

    state = AppState()
    state.dispatch_table = table
    app = FastAPI()
    register_routes(app, state)
    return TestClient(app)


def _ask(table: DispatchTable) -> Any:
    features = {name: 0.5 for name in CONTRACT.names}
    return _client(table).post(
        f"/v1/recommend/{KIND}", json={"features": features, "feature_contract_version": CONTRACT.service_version}
    )


def _metrics(env: dict[str, Any]) -> dict[str, Any]:
    return read_manifest(env["models"] / f"{KIND}.json").manifest.metrics


def classic_rows(
    env: dict[str, Any],
    accepted: int,
    dismissed: int,
    *,
    ignored: int = 0,
    start: int = FLIP_AT + 3_600_000,
    seed: int = 7,
    shown: bool = True,
    as_revisions: bool = True,
) -> None:
    """Label rows served by the flipped classic model.

    With ``as_revisions`` each row first lands as revision 1 (``ignored``: no
    action known yet) and then as revision 2 carrying the real ``user_action`` —
    how a post-push action change arrives under the frozen contract (A3.3).
    """
    rng = np.random.default_rng(seed)
    actions = ["accepted"] * accepted + ["dismissed"] * dismissed + ["ignored"] * ignored
    final = []
    for i, action in enumerate(actions):
        feats = features_for(KIND, rng, float(rng.normal()))
        final.append(
            label_row(KIND, start + i * 60_000, feats, action=action, model_id=CLASSIC_ID, rung="R3", shown=shown)
        )
    if as_revisions:
        push(KIND, [dict(r, user_action="ignored", revision=1) for r in final], env["retained"])
        push(KIND, [dict(r, revision=2) for r in final], env["retained"])
    else:
        push(KIND, final, env["retained"])


def _evaluate(env: dict[str, Any], now: int) -> Any:
    return evaluate_flipped_kinds(env["table"], now_ms=now, store=env["store"], base_dir=env["base"]).get(KIND)


# ---------------------------------------------------------------------------
# Scenario 1 — worse beyond the noise guard reverts, recorded and audit-visible
# ---------------------------------------------------------------------------


def test_worse_flipped_model_is_demoted_with_both_windows_numbers(env: dict[str, Any]) -> None:
    assert _ask(env["table"]).json()["backend"] == "classic"
    classic_rows(env, accepted=10, dismissed=30)  # 0.25 vs baseline 0.70

    result = _evaluate(env, FLIP_AT + 2 * DAY_MS)
    assert result.verdict == "demoted"

    resp = _ask(env["table"])
    assert resp.status_code == 422 and resp.json()["refusal"]["reason"] == "kind_not_served"
    m = _metrics(env)
    assert m["verdict"] == "demoted" and m["flipback_verdict"] == "demoted"
    assert m["serving_backend"] == "heuristic"
    assert m["demoted_generation"] == 1
    assert m["flipped_accept_rate"] == pytest.approx(0.25)
    assert (m["flipped_accepted"], m["flipped_dismissed"]) == (10, 30)
    assert m["baseline_accept_rate"] == pytest.approx(0.70)
    assert (m["baseline_accepted"], m["baseline_dismissed"]) == (210, 90)
    assert m["flipped_window_start_ms"] == FLIP_AT
    assert m["baseline_window_end_ms"] is not None
    assert m["flipback_params"]["flipback_min_shown"] == FLIPBACK_MIN_SHOWN
    assert m["flipback_params"]["demotion_margin_pp"] == DEMOTION_MARGIN_PP
    assert "flipped_at_ms" not in m
    assert (AUDIT_DEMOTE, f"advice/{KIND}@1") in env["store"].events


def test_restart_agrees_with_the_demotion(env: dict[str, Any]) -> None:
    classic_rows(env, accepted=5, dismissed=35)
    _evaluate(env, FLIP_AT + DAY_MS)
    restarted = build_table(env["models"], env["base"])
    assert _ask(restarted).json()["refusal"]["reason"] == "kind_not_served"


# ---------------------------------------------------------------------------
# Scenario 2 — equal or better: nothing changes, the rates are recorded
# ---------------------------------------------------------------------------


def test_equal_or_better_changes_nothing_and_records_rates(env: dict[str, Any]) -> None:
    classic_rows(env, accepted=24, dismissed=6)  # 0.80 >= 0.70
    result = _evaluate(env, FLIP_AT + DAY_MS)
    assert result.verdict == "holding"
    assert _ask(env["table"]).json()["backend"] == "classic"
    m = _metrics(env)
    assert m["serving_backend"] == "classic" and m["flipback_verdict"] == "holding"
    assert m["flipped_accept_rate"] == pytest.approx(0.80) and m["baseline_accept_rate"] == pytest.approx(0.70)
    assert "demoted_generation" not in m


# ---------------------------------------------------------------------------
# Scenario 3 — no same-generation re-flip; a new generation flips normally
# ---------------------------------------------------------------------------


def test_same_generation_never_reflips_a_new_one_does(env: dict[str, Any]) -> None:
    classic_rows(env, accepted=5, dismissed=35)
    _evaluate(env, FLIP_AT + DAY_MS)
    assert _metrics(env)["demoted_generation"] == 1

    # The same generation still measures eligible on its shadow log...
    later = FLIP_AT + 2 * DAY_MS
    again = graduation_check(KIND, now_ms=later)
    assert again.verdict == "eligible" and again.generation == "1"
    # ...and is refused the flip.
    refused = apply_flip(env["table"], KIND, again, now_ms=later, base_dir=env["base"])
    assert not refused.flipped and "demoted" in refused.reason
    assert _ask(env["table"]).json()["refusal"]["reason"] == "kind_not_served"

    # A later generation with a fresh eligible verdict flips, clearing the marker.
    push(KIND, separable_rows(KIND, 60, start_ts=FLIP_AT + 3 * DAY_MS, seed=21, shown=False), env["retained"])
    gen2 = train_kind(KIND, clock=lambda: FLIP_AT + 4 * DAY_MS)
    assert gen2.trained and gen2.generation == "2"
    assert gen2.manifest.metrics["demoted_generation"] == 1  # carried forward by the retrain
    report = after_retrain(env["table"], KIND, gen2, now_ms=FLIP_AT + 4 * DAY_MS, base_dir=env["base"])
    assert report["flip"].flipped
    body = _ask(env["table"]).json()
    assert body["backend"] == "classic" and body["generation"] == "2"
    m = _metrics(env)
    assert "demoted_generation" not in m and m["flipped_generation"] == 2


def test_recovered_rate_does_not_reflip_without_a_new_generation(env: dict[str, Any]) -> None:
    """No flapping: after a demotion, good numbers alone never re-flip generation 1."""
    classic_rows(env, accepted=5, dismissed=35)
    _evaluate(env, FLIP_AT + DAY_MS)
    classic_rows(env, accepted=200, dismissed=0, start=FLIP_AT + 2 * DAY_MS, seed=9)
    scheduler = AdviceTrainingScheduler(
        kinds=(KIND,),
        clock=lambda: FLIP_AT + 3 * DAY_MS,
        on_tick=evaluation_hook(env["table"], env["store"]),
        min_new_labels=10**9,  # the tick runs; no retrain
    )
    for _ in range(3):
        scheduler.check_and_retrain()
    assert _ask(env["table"]).json()["refusal"]["reason"] == "kind_not_served"
    assert _metrics(env)["serving_backend"] == "heuristic"


# ---------------------------------------------------------------------------
# Scenario 4 — the starving window
# ---------------------------------------------------------------------------


def test_starving_window_is_pending_then_insufficient_data_and_stays_flipped(env: dict[str, Any]) -> None:
    classic_rows(env, accepted=0, dismissed=5)
    early = _evaluate(env, FLIP_AT + (MAX_EVAL_WINDOW_DAYS - 1) * DAY_MS)
    assert early.verdict == "pending"
    assert _metrics(env)["flipback_verdict"] == "pending"

    late = _evaluate(env, FLIP_AT + MAX_EVAL_WINDOW_DAYS * DAY_MS)
    assert late.verdict == "insufficient_data"
    assert _ask(env["table"]).json()["backend"] == "classic"  # never demoted on absence of evidence
    m = _metrics(env)
    assert m["flipback_verdict"] == "insufficient_data" and m["serving_backend"] == "classic"
    assert "stays flipped" in m["flipback_reason"]
    _evaluate(env, FLIP_AT + (MAX_EVAL_WINDOW_DAYS + 1) * DAY_MS)
    audits = [e for e in env["store"].events if e[0] == AUDIT_INSUFFICIENT_DATA]
    assert audits == [(AUDIT_INSUFFICIENT_DATA, f"advice/{KIND}@1")]  # visible once, not every tick


# ---------------------------------------------------------------------------
# Boundaries — sample and margin
# ---------------------------------------------------------------------------


def test_29_shown_renders_no_verdict_30_does(env: dict[str, Any]) -> None:
    classic_rows(env, accepted=0, dismissed=29)
    assert _evaluate(env, FLIP_AT + DAY_MS).verdict == "pending"
    classic_rows(env, accepted=0, dismissed=1, start=FLIP_AT + DAY_MS + 3_600_000, seed=3)
    assert _evaluate(env, FLIP_AT + 2 * DAY_MS).verdict == "demoted"


def test_exactly_baseline_minus_margin_holds_just_below_demotes(env: dict[str, Any]) -> None:
    classic_rows(env, accepted=65, dismissed=35)  # 0.65 = 0.70 - 5.0 pp exactly
    result = _evaluate(env, FLIP_AT + DAY_MS)
    assert result.delta_pp == -5.0 and result.verdict == "holding"
    classic_rows(env, accepted=0, dismissed=1, start=FLIP_AT + DAY_MS + 3_600_000, seed=4)  # 65/101
    result = _evaluate(env, FLIP_AT + 2 * DAY_MS)
    assert result.delta_pp < -5.0 and result.verdict == "demoted"


def test_unshown_and_ignored_rows_are_excluded(env: dict[str, Any]) -> None:
    classic_rows(env, accepted=0, dismissed=40, shown=False)  # propensity captures: not shown
    classic_rows(env, accepted=0, dismissed=0, ignored=40, start=FLIP_AT + 5 * 3_600_000, seed=5)
    result = _evaluate(env, FLIP_AT + DAY_MS)
    assert result.flipped.decided == 0 and result.flipped.ignored == 40
    assert result.verdict == "pending"


def test_heuristic_rows_after_the_flip_do_not_count_as_classic(env: dict[str, Any]) -> None:
    seed_decisions(KIND, env["retained"], [(0, "accepted", False), (40, "dismissed", False)],
                   start=FLIP_AT + 3_600_000, seed=12, with_shadow=False)  # fmt: skip
    result = _evaluate(env, FLIP_AT + DAY_MS)
    assert result.flipped.decided == 0 and result.verdict == "pending"


# ---------------------------------------------------------------------------
# Latency — the engine's own per-kind serving measurement
# ---------------------------------------------------------------------------


def test_sustained_p95_over_budget_demotes(env: dict[str, Any]) -> None:
    for _ in range(50):
        env["table"].record_latency(KIND, 80.0)
    assert _evaluate(env, FLIP_AT + DAY_MS).verdict == "pending"  # 50 samples: not yet sustained
    for _ in range(150):
        env["table"].record_latency(KIND, 80.0)
    result = _evaluate(env, FLIP_AT + DAY_MS)
    assert result.verdict == "demoted" and "p95" in result.reason
    assert _metrics(env)["latency_p95_ms"] == 80.0
    assert _ask(env["table"]).json()["refusal"]["reason"] == "kind_not_served"


# ---------------------------------------------------------------------------
# Wiring and failure safety
# ---------------------------------------------------------------------------


def test_the_scheduler_tick_runs_the_evaluation(env: dict[str, Any]) -> None:
    classic_rows(env, accepted=5, dismissed=35)
    scheduler = AdviceTrainingScheduler(
        kinds=(KIND,),
        clock=lambda: FLIP_AT + DAY_MS,
        on_tick=evaluation_hook(env["table"], env["store"]),
        min_new_labels=10**9,  # isolate the tick: no retrain this time
    )
    scheduler.check_and_retrain()
    assert _metrics(env)["verdict"] == "demoted"


def test_demotion_manifest_write_failure_reverts_nothing(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    classic_rows(env, accepted=5, dismissed=35)
    monkeypatch.setattr(shadow, "update_manifest_metrics", lambda *a, **k: MetricsWrite(False, reason="disk"))
    _evaluate(env, FLIP_AT + DAY_MS)
    assert _ask(env["table"]).json()["backend"] == "classic"  # table and manifest still agree
    monkeypatch.undo()
    assert _metrics(env)["serving_backend"] == "classic"
    assert not [e for e in env["store"].events if e[0] == AUDIT_DEMOTE]


def test_unflipped_kinds_are_not_evaluated(env: dict[str, Any]) -> None:
    results = evaluate_flipped_kinds(env["table"], now_ms=FLIP_AT + DAY_MS, base_dir=env["base"])
    assert set(results) == {KIND}  # branch_now / escalate_model have no flipped model
