"""harness-recommendation-models-01MSK2RM WP03 — shadow records and the graduation verdict.

Spec User Story 3 (scenarios 1-4), FR-006..FR-010, and the "no shadow log"
edge case. Label rows go through Mission A's real ingest; shadow records through
this mission's writer onto the same per-kind file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kenaz_ml import config
from kenaz_ml.advice import shadow
from kenaz_ml.advice.contracts import contract_for
from kenaz_ml.advice.label_log import append_records, label_log_path
from kenaz_ml.advice.shadow import (
    MIN_GRADUATION_LABELS,
    VERDICT_DEMOTE,
    VERDICT_ELIGIBLE,
    VERDICT_NOT_ELIGIBLE,
    ShadowWriter,
    enqueue_shadow,
    graduation_check,
    read_shadow_log,
    shadow_record,
    update_manifest_metrics,
    write_shadow_records,
)
from kenaz_ml.advice.training import train_kind
from kenaz_ml.modelstore.registry import read_manifest, verify_artifact_file
from tests.fixtures.advice_labels import DAY_MS, T0, push, seed_decisions, separable_rows

KIND = "compact_now"
NAMES = contract_for(KIND).names  # type: ignore[union-attr]
NOW = T0 + 30 * DAY_MS


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(config, "base_models_dir", lambda: tmp_path / "base")
    return {"models": config.models_dir(), "retained": config.retained_data_dir()}


@pytest.fixture
def trained(dirs: dict[str, Path]) -> Any:
    # Training rows are propensity captures (shown=false): they train, but do not
    # enter any accept rate, so each test's precision numbers are its own.
    push(KIND, separable_rows(KIND, 80, start_ts=T0 - 5 * DAY_MS, shown=False), dirs["retained"])
    outcome = train_kind(KIND, clock=lambda: T0)
    assert outcome.trained
    return outcome.manifest


def seed(retained: Path, groups: list[tuple[int, str, bool]], *, seed_: int = 0, **kw: Any) -> None:
    seed_decisions(KIND, retained, groups, seed=seed_, **kw)


# ---------------------------------------------------------------------------
# T012 — records, reader, bound
# ---------------------------------------------------------------------------


def test_shadow_records_round_trip_and_carry_no_user_action(dirs: dict[str, Path]) -> None:
    rec = shadow_record(
        KIND, [1.0] * len(NAMES), ts_ms=T0, p=0.8, decision=True, confidence=80,
        model_id_sha8="deadbeef", generation="3", rung="R2", session_id="s9",
    )  # fmt: skip
    assert "user_action" not in rec and "heuristic_predicted" not in rec
    assert write_shadow_records(KIND, [rec], directory=dirs["retained"]).ok
    log = read_shadow_log(KIND, directory=dirs["retained"])
    assert log.exists and log.shadows == [json.loads(json.dumps(rec))]


def test_missing_file_is_no_shadow_log(dirs: dict[str, Path]) -> None:
    log = read_shadow_log(KIND, directory=dirs["retained"])
    assert not log.exists and log.shadows == [] and log.labels == []


def test_corrupt_and_truncated_lines_are_skipped_and_counted(dirs: dict[str, Path]) -> None:
    seed(dirs["retained"], [(3, "accepted", True)])
    path = label_log_path(KIND, directory=dirs["retained"])
    with path.open("ab") as fh:
        fh.write(b"{not json\n")
        fh.write(b'{"record":"shadow","kind":"compact_now"}\n')  # shadow without its fields
        fh.write(b'{"record":"shadow","vector_hash"')  # truncated final line
    log = read_shadow_log(KIND, directory=dirs["retained"])
    assert len(log.shadows) == 3 and len(log.labels) == 3
    assert log.skipped_lines == 2 and log.truncated_final_line


def test_bound_and_eviction_are_mission_as(dirs: dict[str, Path]) -> None:
    rec = shadow_record(
        KIND, [0.0] * len(NAMES), ts_ms=T0, p=0.9, decision=True, confidence=90,
        model_id_sha8=None, generation="1", rung="R2",
    )  # fmt: skip
    records = [dict(rec, ts=T0 + i) for i in range(50)]
    line = len(json.dumps(records[0], sort_keys=True, separators=(",", ":"))) + 1
    evicted = append_records(KIND, records, directory=dirs["retained"], max_bytes=line * 10)
    assert evicted == 40
    kept = read_shadow_log(KIND, directory=dirs["retained"]).shadows
    assert [r["ts"] for r in kept] == [T0 + i for i in range(40, 50)]  # contiguous, oldest-first eviction


# ---------------------------------------------------------------------------
# T013 — the write-site interface
# ---------------------------------------------------------------------------


def test_enqueue_writes_one_scored_record(dirs: dict[str, Path], trained: Any) -> None:
    from kenaz_ml.modelstore.registry import resolve_model

    model = resolve_model(KIND, local_dir=dirs["models"], expected_contract=contract_for(KIND)).model
    writer = ShadowWriter()
    result = enqueue_shadow(
        KIND, [0.5] * len(NAMES), model, trained, ts_ms=T0, session_id="s1", directory=dirs["retained"], writer=writer
    )
    assert result.ok and writer.flush()
    shadows = read_shadow_log(KIND, directory=dirs["retained"]).shadows
    assert len(shadows) == 1
    predicted = shadows[0]["predicted"]
    assert predicted["confidence"] == round(100 * max(predicted["p"], 1 - predicted["p"]))
    assert shadows[0]["generation"] == trained.version
    assert shadows[0]["model_id_sha8"] == trained.artifact_sha256[:8]


def test_enqueue_never_raises_and_contains_failures(dirs: dict[str, Path], trained: Any) -> None:
    class Broken:
        classes_ = [0, 1]

        def predict_proba(self, _x: Any) -> Any:
            raise RuntimeError("model exploded")

    writer = ShadowWriter()
    assert enqueue_shadow(KIND, [0.0] * len(NAMES), Broken(), trained, ts_ms=T0, writer=writer).ok
    assert writer.flush() and writer.failed == 1 and writer.written == 0
    # Unhashable garbage in: a typed failure, not an exception.
    assert not enqueue_shadow(KIND, ["x"], Broken(), trained, ts_ms=T0, writer=writer).ok  # type: ignore[list-item]


def test_full_queue_drops_instead_of_blocking(trained: Any) -> None:
    writer = ShadowWriter(maxsize=1)
    writer._ensure_started = lambda: None  # type: ignore[method-assign]  # no consumer: the queue fills
    assert enqueue_shadow(KIND, [0.0] * len(NAMES), object(), trained, ts_ms=T0, writer=writer).ok
    second = enqueue_shadow(KIND, [0.0] * len(NAMES), object(), trained, ts_ms=T0, writer=writer)
    assert not second.ok and writer.dropped == 1


# ---------------------------------------------------------------------------
# T014 — verdicts (scenarios 1-3, FR-007/FR-008, edge cases)
# ---------------------------------------------------------------------------


def test_under_five_points_is_not_eligible_with_the_delta_reported(dirs: dict[str, Path], trained: Any) -> None:
    # shadow-shown subset: 108/150 = 0.72; heuristic overall: 210/300 = 0.70 -> 2 pp.
    seed(
        dirs["retained"],
        [(108, "accepted", True), (42, "dismissed", True), (102, "accepted", False), (48, "dismissed", False)],
    )
    result = graduation_check(KIND, now_ms=NOW)
    assert result.verdict == VERDICT_NOT_ELIGIBLE
    assert result.label_count == 300
    assert result.shadow_precision == pytest.approx(0.72)
    assert result.heuristic_precision == pytest.approx(0.70)
    assert result.precision_delta_pp == pytest.approx(2.0)
    assert "< 5.0 pp" in result.reason


def test_five_points_or_more_is_eligible_and_written_to_metrics(dirs: dict[str, Path], trained: Any) -> None:
    seed(
        dirs["retained"],
        [(120, "accepted", True), (30, "dismissed", True), (90, "accepted", False), (60, "dismissed", False)],
    )
    result = graduation_check(KIND, now_ms=NOW)
    assert result.verdict == VERDICT_ELIGIBLE
    assert result.precision_delta_pp == pytest.approx(10.0)
    assert result.calibration_fitted and result.calibration_ece is not None

    before = read_manifest(dirs["models"] / f"{KIND}.json").manifest
    written = update_manifest_metrics(KIND, result.to_metrics(), expected_version=before.version)
    assert written.ok
    after = read_manifest(dirs["models"] / f"{KIND}.json").manifest
    for key in ("verdict", "precision_at_shown_threshold", "heuristic_precision", "precision_delta",
                "shadow_window_labels", "window_size_days", "calibration_ece", "graduation_params"):  # fmt: skip
        assert key in after.metrics, key
    assert after.metrics["verdict"] == "eligible"
    assert after.metrics["trained_generation"] == before.metrics["trained_generation"]  # other keys survive
    assert after.metrics["graduation_params"]["min_graduation_labels"] == MIN_GRADUATION_LABELS
    # Only metrics changed: identity, digest and artifact untouched.
    assert (after.name, after.version, after.artifact_sha256) == (before.name, before.version, before.artifact_sha256)
    assert verify_artifact_file(dirs["models"] / f"{KIND}.joblib", after).ok
    raw = json.loads((dirs["models"] / f"{KIND}.json").read_text())
    assert set(raw) == {
        "schema_version", "name", "version", "created_at", "provenance", "runtime",
        "feature_contract", "training", "metrics", "artifact_sha256",
    }  # fmt: skip


def test_199_labels_never_eligible_200_at_exactly_five_points_is(dirs: dict[str, Path], trained: Any) -> None:
    # 199 decisions, a perfect shadow precision against a 50% heuristic: still not eligible.
    seed(dirs["retained"], [(100, "accepted", True), (99, "dismissed", False)])
    result = graduation_check(KIND, now_ms=NOW)
    assert result.label_count == 199
    assert result.shadow_precision == 1.0
    assert result.verdict == VERDICT_NOT_ELIGIBLE and "< 200" in result.reason


def test_exactly_200_at_exactly_five_points_is_eligible(dirs: dict[str, Path], trained: Any) -> None:
    # shadow-shown: 75/100 = 0.75; heuristic overall: 140/200 = 0.70 -> delta exactly 5.0 pp.
    seed(
        dirs["retained"],
        [(75, "accepted", True), (25, "dismissed", True), (65, "accepted", False), (35, "dismissed", False)],
    )
    result = graduation_check(KIND, now_ms=NOW)
    assert result.label_count == 200
    assert result.precision_delta_pp == 5.0
    assert result.verdict == VERDICT_ELIGIBLE


def test_ignored_and_unshown_rows_stay_out_of_precision(dirs: dict[str, Path], trained: Any) -> None:
    seed(dirs["retained"], [(150, "accepted", True), (50, "dismissed", True), (40, "ignored", True)])
    result = graduation_check(KIND, now_ms=NOW)
    assert result.label_count == 200  # ignored joined but not a decision
    assert result.shadow_shown == 200


def test_no_shadow_log_is_a_reasoned_not_eligible(dirs: dict[str, Path], trained: Any) -> None:
    result = graduation_check(KIND, now_ms=NOW)
    assert result.verdict == VERDICT_NOT_ELIGIBLE and result.reason == "no shadow log"
    seed(dirs["retained"], [(10, "accepted", True)], with_shadow=False)
    assert graduation_check(KIND, now_ms=NOW).reason == "no shadow log"


def test_no_trained_model_is_not_eligible(dirs: dict[str, Path]) -> None:
    assert graduation_check(KIND, now_ms=NOW).reason == "no trained model"


def test_one_class_window_is_handled(dirs: dict[str, Path], trained: Any) -> None:
    seed(dirs["retained"], [(220, "dismissed", True)])
    result = graduation_check(KIND, now_ms=NOW)
    assert result.verdict == VERDICT_NOT_ELIGIBLE
    assert result.shadow_precision == 0.0 and result.heuristic_precision == 0.0


def test_rows_outside_the_trailing_window_do_not_count(dirs: dict[str, Path], trained: Any) -> None:
    seed(
        dirs["retained"],
        [(120, "accepted", True), (30, "dismissed", True), (90, "accepted", False), (60, "dismissed", False)],
    )
    assert graduation_check(KIND, now_ms=NOW).verdict == VERDICT_ELIGIBLE
    late = graduation_check(KIND, now_ms=T0 + DAY_MS + 91 * DAY_MS)
    assert late.label_count == 0 and late.verdict == VERDICT_NOT_ELIGIBLE


# ---------------------------------------------------------------------------
# Scenario 3 / FR-010 — demotion of an already-promoted kind is explicit
# ---------------------------------------------------------------------------


def _promote(dirs: dict[str, Path]) -> None:
    assert update_manifest_metrics(KIND, {"serving_backend": "classic", "rung": "R3"}).ok


def test_promoted_kind_with_worse_live_precision_is_demote(dirs: dict[str, Path], trained: Any) -> None:
    seed(dirs["retained"], [(80, "accepted", False), (20, "dismissed", False)], with_shadow=False)  # heuristic 0.8
    seed(
        dirs["retained"], [(60, "accepted", False), (40, "dismissed", False)], start=T0 + 5 * DAY_MS, seed_=1,
        model_id=f"classic/{KIND}@1", rung="R3", with_shadow=False,
    )  # fmt: skip
    _promote(dirs)
    result = graduation_check(KIND, now_ms=NOW)
    assert result.promoted and result.verdict == VERDICT_DEMOTE
    assert result.live_precision == pytest.approx(0.6) and result.heuristic_precision == pytest.approx(0.8)
    metrics = result.to_metrics()
    assert metrics["verdict"] == "demote" and metrics["live_precision"] == pytest.approx(0.6)


def test_promoted_kind_over_latency_budget_is_demote_only_when_sustained(dirs: dict[str, Path], trained: Any) -> None:
    _promote(dirs)
    assert graduation_check(KIND, now_ms=NOW, latency_p95_ms=80.0, latency_samples=10).verdict == VERDICT_ELIGIBLE
    result = graduation_check(KIND, now_ms=NOW, latency_p95_ms=80.0, latency_samples=200)
    assert result.verdict == VERDICT_DEMOTE and "p95" in result.reason and result.latency_evaluated


def test_unflipped_kind_does_not_evaluate_latency(dirs: dict[str, Path], trained: Any) -> None:
    result = graduation_check(KIND, now_ms=NOW, latency_p95_ms=500.0, latency_samples=500)
    assert not result.latency_evaluated and result.verdict == VERDICT_NOT_ELIGIBLE


# ---------------------------------------------------------------------------
# T015 — the manifest write
# ---------------------------------------------------------------------------


def test_metrics_write_refuses_a_replaced_manifest(dirs: dict[str, Path], trained: Any) -> None:
    result = update_manifest_metrics(KIND, {"verdict": "eligible"}, expected_version="99")
    assert not result.ok and "expected 99" in (result.reason or "")


def test_failed_metrics_write_keeps_the_old_manifest(
    dirs: dict[str, Path], trained: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = dirs["models"] / f"{KIND}.json"
    before = path.read_bytes()

    def broken(*_: Any, **__: Any) -> None:
        raise OSError("read-only")

    monkeypatch.setattr("kenaz_ml.modelstore.registry.write_manifest", broken)
    assert not update_manifest_metrics(KIND, {"verdict": "eligible"}).ok
    assert path.read_bytes() == before


def test_tunables_are_the_documented_initial_values() -> None:
    assert shadow.tunables() == {
        "shown_confidence_threshold": 75,
        "min_graduation_labels": 200,
        "min_precision_delta_pp": 5.0,
        "trailing_window_days": 90,
        "latency_budget_ms": 50.0,
        "latency_min_samples": 100,
        "shadow_join_tolerance_ms": 120_000,
        "flipback_min_shown": 30,
        "demotion_margin_pp": 5.0,
        "max_eval_window_days": 21,
        "baseline_window_days": 90,
    }
