"""harness-recommendation-models-01MSK2RM WP01 — per-kind GBDT training and the retrain trigger.

Spec User Story 1 (scenarios 1-4) and its edge cases: complete rows only (with
the defensive label-log re-check), honest ``n_samples``, the 50-labels-and-24h
trigger from durable state, two kinds in one tick, quiet ``escalate_model``,
and a failed run leaving the prior pair byte-identical.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import pytest

from kenaz_ml import config
from kenaz_ml.advice import training
from kenaz_ml.advice.contracts import contract_for
from kenaz_ml.advice.training import (
    REASON_CONTRACT_MISMATCH,
    REASON_INSUFFICIENT_COMPLETE,
    AdviceTrainingScheduler,
    TrainingSet,
    load_training_set,
    train_kind,
)
from kenaz_ml.modelstore.registry import Example, append_examples, read_manifest, resolve_model, verify_artifact_file
from tests.fixtures.advice_labels import DAY_MS, T0, features_for, label_row, push, separable_rows

KIND = "compact_now"
SRC = Path(training.__file__)


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    base = tmp_path / "base"
    monkeypatch.setattr(config, "base_models_dir", lambda: base)
    return {"models": config.models_dir(), "retained": config.retained_data_dir(), "base": base}


class Clock:
    def __init__(self, now: int) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def _train(dirs: dict[str, Path], kind: str = KIND, now: int = T0) -> training.TrainOutcome:
    return train_kind(kind, models_dir=dirs["models"], retained_dir=dirs["retained"], clock=Clock(now))


# ---------------------------------------------------------------------------
# Scenario 1 / SC-001 — complete rows only, honest n_samples
# ---------------------------------------------------------------------------


def test_mixed_set_trains_on_complete_rows_only(dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    complete = separable_rows(KIND, 60)
    rng = np.random.default_rng(9)
    incomplete = [
        label_row(KIND, T0 + 10_000_000 + i, features_for(KIND, rng, 1.0), complete=False, action="accepted")
        for i in range(40)
    ]
    ignored = [label_row(KIND, T0 + 20_000_000 + i, features_for(KIND, rng, 1.0), action="ignored") for i in range(5)]
    push(KIND, complete + incomplete + ignored, dirs["retained"])

    # One incomplete row leaks into the retained set (the defensive re-check's target).
    leak = incomplete[0]
    contract = contract_for(KIND)
    leaked_x = tuple(float(leak["features"].get(n, 0.0)) for n in contract.names)
    append_examples(KIND, [Example(x=leaked_x, y=1.0, as_of_ms=leak["ts"])], contract, directory=dirs["retained"])

    loaded = load_training_set(KIND, retained_dir=dirs["retained"])
    assert isinstance(loaded, TrainingSet)
    assert loaded.n_retained == 61
    assert loaded.n_leaked == 1
    assert loaded.n_complete == 60

    fitted_rows: list[int] = []
    real_fit = training.fit_model

    def spy(training_set: TrainingSet) -> Any:
        fitted_rows.append(int(training_set.X.shape[0]))
        return real_fit(training_set)

    monkeypatch.setattr(training, "fit_model", spy)
    outcome = _train(dirs)
    assert outcome.trained, outcome
    assert fitted_rows == [60]  # the fitted model saw exactly the complete rows
    manifest = outcome.manifest
    assert manifest.training.n_samples == 60
    assert manifest.metrics["trained_generation"] == int(manifest.version) == 1
    assert manifest.metrics["n_leaked_incomplete_excluded"] == 1
    model = resolve_model(KIND, local_dir=dirs["models"], base_dir=dirs["base"], expected_contract=contract).model
    assert model is not None
    assert manifest.training.as_of_ms == max(r["ts"] for r in complete)


def test_all_incomplete_declines_and_writes_nothing(dirs: dict[str, Path]) -> None:
    rows = [dict(r, features_complete=False) for r in separable_rows(KIND, 30)]
    push(KIND, rows, dirs["retained"])
    # Plant the incomplete rows directly in retained, as a pathological capture bug would.
    contract = contract_for(KIND)
    append_examples(
        KIND,
        [Example(x=tuple(r["features"][n] for n in contract.names), y=1.0, as_of_ms=r["ts"]) for r in rows],
        contract,
        directory=dirs["retained"],
    )
    outcome = _train(dirs)
    assert not outcome.trained
    assert outcome.reason == REASON_INSUFFICIENT_COMPLETE
    assert "insufficient complete labels" in (outcome.detail or "")
    assert not (dirs["models"] / f"{KIND}.joblib").exists()
    assert not (dirs["models"] / f"{KIND}.json").exists()


def test_contract_names_mismatch_declines(dirs: dict[str, Path]) -> None:
    from kenaz_ml.advice.contracts import build_contract

    names = tuple(reversed(contract_for(KIND).names))
    wrong = build_contract(KIND, names)
    append_examples(
        KIND,
        [Example(x=tuple(float(i) for i in range(len(names))), y=float(i % 2), as_of_ms=T0 + i) for i in range(30)],
        wrong,
        directory=dirs["retained"],
    )
    outcome = _train(dirs)
    assert outcome.reason == REASON_CONTRACT_MISMATCH
    assert not (dirs["models"] / f"{KIND}.joblib").exists()


def test_a_local_laya_checkpoint_is_never_overwritten_by_a_classic_retrain(dirs: dict[str, Path]) -> None:
    """Review fix (laya-serving-and-packs-01MSK2SP): {kind}.json is one namespace per slot.

    A laya checkpoint (FR-011) and the classic pair both answer to ``{kind}.json``. Without this
    decline a retrain would replace the directory manifest with a joblib one, carry
    ``serving_backend=laya`` onto it, and leave the kind claiming no laya checkpoint is installed.
    """
    from kenaz_ml.advice.training import REASON_LAYA_CHECKPOINT_PRESENT
    from kenaz_ml.modelstore.registry import Manifest, directory_digest, write_manifest

    push(KIND, separable_rows(KIND, 60), dirs["retained"])
    ckpt = dirs["models"] / f"{KIND}.ckpt"
    ckpt.mkdir(parents=True)
    (ckpt / "laya.onnx").write_bytes(b"graph")
    manifest_file = dirs["models"] / f"{KIND}.json"
    write_manifest(
        manifest_file,
        Manifest(
            name=KIND,
            version="3",
            artifact_sha256=directory_digest(ckpt).digest,
            feature_contract=contract_for(KIND),
            metrics={"serving_backend": "laya"},
            artifact_kind="directory",
        ),
    )
    before = manifest_file.read_bytes()
    outcome = _train(dirs)
    assert not outcome.trained and outcome.reason == REASON_LAYA_CHECKPOINT_PRESENT
    assert manifest_file.read_bytes() == before
    assert not (dirs["models"] / f"{KIND}.joblib").exists()


def test_manifest_digest_matches_artifact_and_estimator_is_recorded(dirs: dict[str, Path]) -> None:
    push(KIND, separable_rows(KIND, 60), dirs["retained"])
    outcome = _train(dirs)
    manifest = read_manifest(dirs["models"] / f"{KIND}.json").manifest
    assert manifest is not None
    assert verify_artifact_file(dirs["models"] / f"{KIND}.joblib", manifest).ok
    assert (
        manifest.runtime.estimator
        == type(
            resolve_model(
                KIND, local_dir=dirs["models"], base_dir=dirs["base"], expected_contract=contract_for(KIND)
            ).model
        ).__name__
    )
    assert manifest.provenance.training_source == "local"
    assert outcome.generation == "1"


def test_training_is_deterministic(dirs: dict[str, Path]) -> None:
    push(KIND, separable_rows(KIND, 60), dirs["retained"])
    first = _train(dirs).manifest.artifact_sha256
    (dirs["models"] / f"{KIND}.json").unlink()
    (dirs["models"] / f"{KIND}.joblib").unlink()
    second = _train(dirs).manifest.artifact_sha256
    assert first == second


def test_a_failed_write_leaves_the_prior_pair_byte_identical(
    dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    push(KIND, separable_rows(KIND, 60), dirs["retained"])
    assert _train(dirs).trained
    artifact = dirs["models"] / f"{KIND}.joblib"
    manifest = dirs["models"] / f"{KIND}.json"
    before = (artifact.read_bytes(), manifest.read_bytes())

    push(KIND, separable_rows(KIND, 60, start_ts=T0 + 5 * DAY_MS, seed=3), dirs["retained"])

    def broken(*_: Any, **__: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("kenaz_ml.modelstore.registry.write_manifest", broken)
    outcome = _train(dirs, now=T0 + 6 * DAY_MS)
    assert outcome.status == training.STATUS_FAILED
    assert outcome.reason == training.REASON_WRITE_FAILED
    assert (artifact.read_bytes(), manifest.read_bytes()) == before


def test_a_failed_fit_leaves_the_prior_pair_byte_identical(
    dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    push(KIND, separable_rows(KIND, 60), dirs["retained"])
    assert _train(dirs).trained
    artifact = dirs["models"] / f"{KIND}.joblib"
    manifest = dirs["models"] / f"{KIND}.json"
    before = (artifact.read_bytes(), manifest.read_bytes())

    def exploding(kind: str) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr("kenaz_ml.advice.models.make_estimator", exploding)
    outcome = _train(dirs, now=T0 + 2 * DAY_MS)
    assert outcome.status == training.STATUS_FAILED
    assert outcome.reason == training.REASON_FIT_FAILED
    assert (artifact.read_bytes(), manifest.read_bytes()) == before


# ---------------------------------------------------------------------------
# Scenarios 2-4 — the trigger, from durable state, with an injected clock
# ---------------------------------------------------------------------------


def _scheduler(dirs: dict[str, Path], clock: Clock, **kw: Any) -> AdviceTrainingScheduler:
    return AdviceTrainingScheduler(
        kinds=(KIND,), models_dir=dirs["models"], retained_dir=dirs["retained"], clock=clock, **kw
    )


def test_scheduler_first_fire_then_below_threshold_then_interval(dirs: dict[str, Path]) -> None:
    clock = Clock(T0)
    push(KIND, separable_rows(KIND, 49), dirs["retained"])
    sched = _scheduler(dirs, clock)

    # Scenario 2: fewer than 50 new labels declines.
    result = sched.check_and_retrain()[KIND]
    assert isinstance(result, training.DueCheck) and not result.due and result.new_labels == 49

    push(KIND, separable_rows(KIND, 1, start_ts=T0 + DAY_MS, seed=1), dirs["retained"])
    result = sched.check_and_retrain()[KIND]
    assert isinstance(result, training.TrainOutcome) and result.trained
    assert result.manifest.version == "1"

    # 50 more labels but under 24 h: scenario 3 declines.
    push(KIND, separable_rows(KIND, 60, start_ts=T0 + 2 * DAY_MS, seed=2), dirs["retained"])
    clock.now = T0 + 23 * 3600 * 1000
    result = sched.check_and_retrain()[KIND]
    assert isinstance(result, training.DueCheck) and result.reason == "interval not elapsed"

    # Scenario 4: both met — trains a new generation.
    clock.now = T0 + 24 * 3600 * 1000
    result = sched.check_and_retrain()[KIND]
    assert isinstance(result, training.TrainOutcome) and result.trained
    assert result.manifest.version == "2"
    assert result.manifest.metrics["trained_generation"] == 2


def test_new_label_count_survives_a_restart(dirs: dict[str, Path]) -> None:
    clock = Clock(T0)
    push(KIND, separable_rows(KIND, 60), dirs["retained"])
    assert _scheduler(dirs, clock).check_and_retrain()[KIND].trained  # type: ignore[union-attr]
    # A fresh scheduler (a restarted process) must not retrain on the same data.
    clock.now = T0 + 3 * DAY_MS
    result = _scheduler(dirs, clock).check_and_retrain()[KIND]
    assert isinstance(result, training.DueCheck) and result.new_labels == 0


def test_two_kinds_same_tick_both_train_without_collision(dirs: dict[str, Path]) -> None:
    push("compact_now", separable_rows("compact_now", 60), dirs["retained"])
    push("branch_now", separable_rows("branch_now", 60, seed=5), dirs["retained"])
    sched = AdviceTrainingScheduler(
        kinds=("compact_now", "branch_now"), models_dir=dirs["models"], retained_dir=dirs["retained"], clock=Clock(T0)
    )
    results = sched.check_and_retrain()
    assert all(isinstance(r, training.TrainOutcome) and r.trained for r in results.values())
    files = sorted(p.name for p in dirs["models"].iterdir() if p.is_file())
    assert files == ["branch_now.joblib", "branch_now.json", "compact_now.joblib", "compact_now.json"]
    for kind in ("compact_now", "branch_now"):
        m = read_manifest(dirs["models"] / f"{kind}.json").manifest
        assert m.name == kind and tuple(m.feature_contract.names) == contract_for(kind).names
        assert verify_artifact_file(dirs["models"] / f"{kind}.joblib", m).ok


def test_escalate_model_without_labels_is_quiet(dirs: dict[str, Path], caplog: pytest.LogCaptureFixture) -> None:
    sched = AdviceTrainingScheduler(
        kinds=("escalate_model",), models_dir=dirs["models"], retained_dir=dirs["retained"], clock=Clock(T0)
    )
    with caplog.at_level(logging.DEBUG, logger="kenaz_ml"):
        for _ in range(3):
            result = sched.check_and_retrain()["escalate_model"]
            assert isinstance(result, training.DueCheck) and not result.due
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_on_trained_callback_fires_only_on_success(dirs: dict[str, Path]) -> None:
    seen: list[tuple[str, str | None]] = []
    push(KIND, separable_rows(KIND, 60), dirs["retained"])
    sched = _scheduler(dirs, Clock(T0), on_trained=lambda k, o: seen.append((k, o.generation)))
    sched.check_and_retrain()
    assert seen == [(KIND, "1")]


# ---------------------------------------------------------------------------
# Structural guarantees
# ---------------------------------------------------------------------------


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def test_no_defaulting_get_and_no_wall_clock_in_example_handling() -> None:
    tree = ast.parse(SRC.read_text())
    for call in _calls(tree):
        if isinstance(call.func, ast.Attribute) and call.func.attr == "get" and len(call.args) == 2:
            default = call.args[1]
            is_zero = isinstance(default, ast.Constant) and type(default.value) in (int, float) and default.value == 0
            assert not is_zero, ast.unparse(call)
    handlers = {"load_training_set", "_incomplete_signatures", "count_new_labels"}
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in handlers):
        text = ast.unparse(fn)
        assert "time.time" not in text and "wall_clock_ms" not in text, fn.name


def test_workbench_scheduler_and_trainer_are_untouched() -> None:
    """Compared by syntax tree, so a formatter-only change is not a modification."""
    import subprocess

    root = SRC.parents[3]
    for path in (
        "src/kenaz_ml/training/scheduler.py",
        "src/kenaz_ml/training/trainer.py",
        "src/kenaz_ml/modelstore/registry/retained.py",
    ):
        base = subprocess.run(
            ["/usr/bin/git", "show", f"kitty/mission-two-client-engine-01MSK2EN:{path}"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if base.returncode != 0:
            pytest.skip("base branch not available")
        assert ast.dump(ast.parse(base.stdout)) == ast.dump(ast.parse((root / path).read_text())), path


def test_app_starts_and_stops_the_advice_scheduler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The local-mode lifespan constructs the scheduler and cancels its loop on shutdown (R5)."""
    import asyncio

    from kenaz_ml import app as app_mod

    ticks: list[int] = []

    class FakeScheduler:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def check_and_retrain(self) -> dict:
            ticks.append(1)
            return {}

    async def run() -> None:
        loop_task = asyncio.create_task(app_mod.advice_schedule_loop(FakeScheduler()))  # type: ignore[arg-type]
        await asyncio.sleep(0.05)
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop_task

    asyncio.run(run())
    assert ticks == [1]
    source = Path(app_mod.__file__).read_text()
    assert "state.advice_scheduler = AdviceTrainingScheduler(" in source
    assert "state_tasks.append(asyncio.create_task(advice_schedule_loop(" in source


def test_a_dirty_retained_mirror_declines(dirs: dict[str, Path]) -> None:
    from kenaz_ml.advice.label_log import dirty_marker_path

    push(KIND, separable_rows(KIND, 60), dirs["retained"])
    dirty_marker_path(KIND, directory=dirs["retained"]).write_text("{}\n")
    outcome = _train(dirs)
    assert outcome.reason == training.REASON_MIRROR_DIRTY
    assert not (dirs["models"] / f"{KIND}.joblib").exists()
