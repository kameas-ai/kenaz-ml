"""harness-recommendation-models-01MSK2RM WP02 — CalibratedClassifierCV wrapping.

Spec User Story 2 (scenarios 1-4), SC-002, FR-003/004/014, NFR-002.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from kenaz_ml import config
from kenaz_ml.advice import models
from kenaz_ml.advice.models import (
    DECLINE_NON_FINITE,
    DECLINE_SINGLE_CLASS,
    DECLINE_THIN_CLASS,
    ISOTONIC_MIN_LABELS,
    CalibratedFit,
    CalibrationDecline,
    calibration_method,
    decision_and_confidence,
    expected_calibration_error,
    fit_calibrated,
    make_estimator,
)
from kenaz_ml.advice.training import STATUS_DECLINED, train_kind
from tests.fixtures.advice_labels import T0, push, separable_rows

KIND = "compact_now"
WIDTH = 6  # compact_now's contract width


def _overconfident_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """True P(y=1) is 0.3 or 0.7 (a step on x0) plus pure-noise features.

    A 100-tree GBDT memorizes the noise and pushes its raw probabilities
    towards 0 and 1 — a known, systematic overconfidence.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, WIDTH))
    p_true = np.where(X[:, 0] > 0, 0.7, 0.3)
    y = (rng.random(n) < p_true).astype(int)
    return X, y


def _separable(n: int, seed: int, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, WIDTH))
    y = ((X[:, 0] + 0.3 * rng.normal(size=n)) > 0).astype(int)
    return X, (1 - y) if flip else y


def _pos(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


# ---------------------------------------------------------------------------
# SC-002 — calibration lowers ECE on unseen rows
# ---------------------------------------------------------------------------


def test_calibrated_ece_beats_uncalibrated_on_unseen_rows() -> None:
    X, y = _overconfident_data(800, seed=1)
    X_test, y_test = _overconfident_data(4000, seed=2)  # never seen by either fit

    fit = fit_calibrated(KIND, X, y)
    assert isinstance(fit, CalibratedFit)
    raw = make_estimator(KIND).fit(X, y)

    calibrated_ece = expected_calibration_error(y_test, _pos(fit.model, X_test))
    raw_ece = expected_calibration_error(y_test, _pos(raw, X_test))
    assert calibrated_ece < raw_ece
    # The fit records its own held-out numbers (on a 20% split — 160 rows here,
    # so noisier than the 4000-row comparison above, which is the SC-002 proof).
    assert fit.calibration_ece is not None and fit.uncalibrated_ece is not None
    assert fit.holdout_rows == 160


def test_ece_helper_extremes() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02  # perfectly calibrated by construction
    assert expected_calibration_error(y, np.where(p > 0.5, 1.0, 0.0)) > 0.2  # wildly overconfident
    assert expected_calibration_error([], []) == 0.0


def test_small_sets_record_no_ece_rather_than_training_ece() -> None:
    X, y = _separable(40, seed=3)
    fit = fit_calibrated(KIND, X, y)
    assert isinstance(fit, CalibratedFit)
    assert fit.calibration_ece is None and fit.holdout_rows == 0
    assert "held-out ECE not measured" in (fit.ece_note or "")


# ---------------------------------------------------------------------------
# Scenarios 1-2 — sigmoid below 1000 fit-time labels, isotonic at or above
# ---------------------------------------------------------------------------


def test_method_boundary_constant() -> None:
    assert ISOTONIC_MIN_LABELS == 1000
    assert calibration_method(999) == "sigmoid"
    assert calibration_method(1000) == "isotonic"


@pytest.mark.parametrize(("n", "method"), [(999, "sigmoid"), (1000, "isotonic")])
def test_method_boundary_recorded_by_the_fit(n: int, method: str) -> None:
    X, y = _separable(n, seed=4)
    fit = fit_calibrated(KIND, X, y)
    assert isinstance(fit, CalibratedFit)
    assert fit.method == method == fit.model.method
    assert fit.metrics()["calibration_method"] == method
    assert fit.metrics()["calibration_n_labels"] == n


def test_manifest_records_calibrated_estimator_and_method(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(config, "base_models_dir", lambda: tmp_path / "base")
    push(KIND, separable_rows(KIND, 80), config.retained_data_dir())
    outcome = train_kind(KIND, clock=lambda: T0)
    assert outcome.trained, outcome
    m = outcome.manifest
    assert m.runtime.estimator == "CalibratedClassifierCV"
    assert m.metrics["calibration_fitted"] is True
    assert m.metrics["calibration_method"] == "sigmoid"
    assert m.metrics["calibration_n_labels"] == m.training.n_samples == 80
    assert m.metrics["calibration_ece"] is not None


# ---------------------------------------------------------------------------
# Scenario 3 / FR-004 — refit fresh on every retrain
# ---------------------------------------------------------------------------


def test_refit_reflects_only_the_new_data() -> None:
    XA, yA = _separable(300, seed=5)
    XB, yB = _separable(300, seed=6, flip=True)  # the opposite bias
    fit_a = fit_calibrated(KIND, XA, yA)
    fit_b = fit_calibrated(KIND, XB, yB)
    assert isinstance(fit_a, CalibratedFit) and isinstance(fit_b, CalibratedFit)
    assert fit_a.model is not fit_b.model

    probe = np.zeros((2, WIDTH))
    probe[0, 0], probe[1, 0] = 2.0, -2.0
    pa, pb = _pos(fit_a.model, probe), _pos(fit_b.model, probe)
    assert pa[0] > 0.75 and pa[1] < 0.25
    assert pb[0] < 0.25 and pb[1] > 0.75  # nothing of A survived into B


def test_every_retrain_constructs_a_new_calibrator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(config, "base_models_dir", lambda: tmp_path / "base")
    built: list[object] = []
    real = models._calibrated

    def spy(kind: str, method: str, folds: int):
        est = real(kind, method, folds)
        assert not hasattr(est, "calibrated_classifiers_")  # unfitted when handed over
        built.append(est)
        return est

    monkeypatch.setattr(models, "_calibrated", spy)
    push(KIND, separable_rows(KIND, 60), config.retained_data_dir())
    assert train_kind(KIND, clock=lambda: T0).trained
    push(KIND, separable_rows(KIND, 60, start_ts=T0 + 10**9, seed=8), config.retained_data_dir())
    assert train_kind(KIND, clock=lambda: T0 + 10**9).trained
    assert len({id(e) for e in built}) == len(built) >= 2


# ---------------------------------------------------------------------------
# FR-014 — decline before sklearn
# ---------------------------------------------------------------------------


def test_single_class_declines_without_sklearn(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_a, **_k):
        raise AssertionError("sklearn must not be reached")

    monkeypatch.setattr(models, "_calibrated", forbidden)
    X = np.random.default_rng(0).normal(size=(100, WIDTH))
    result = fit_calibrated(KIND, X, np.ones(100))
    assert isinstance(result, CalibrationDecline)
    assert result.reason == DECLINE_SINGLE_CLASS
    assert result.detail == "only one label class present: cannot calibrate"
    assert result.counts == {"positive": 100, "negative": 0}


def test_thin_minority_declines() -> None:
    X = np.random.default_rng(0).normal(size=(100, WIDTH))
    y = np.ones(100)
    y[0] = 0
    result = fit_calibrated(KIND, X, y)
    assert isinstance(result, CalibrationDecline) and result.reason == DECLINE_THIN_CLASS


def test_nan_inf_and_empty_decline() -> None:
    X, y = _separable(50, seed=1)
    X[3, 2] = np.nan
    assert fit_calibrated(KIND, X, y).reason == DECLINE_NON_FINITE  # type: ignore[union-attr]
    assert isinstance(fit_calibrated(KIND, np.empty((0, WIDTH)), np.empty(0)), CalibrationDecline)


def test_single_class_retrain_declines_and_writes_no_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(config, "base_models_dir", lambda: tmp_path / "base")
    rows = [dict(r, user_action="accepted") for r in separable_rows(KIND, 40)]
    push(KIND, rows, config.retained_data_dir())
    outcome = train_kind(KIND, clock=lambda: T0)
    assert outcome.status == STATUS_DECLINED and outcome.reason == DECLINE_SINGLE_CLASS
    assert not (config.models_dir() / f"{KIND}.joblib").exists()


# ---------------------------------------------------------------------------
# Scenario 4 — confidence from the calibrated probability, validated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "decision", "confidence"),
    [(0.0, False, 100), (0.25, False, 75), (0.4999, False, 50), (0.5, True, 50), (0.75, True, 75), (1.0, True, 100)],
)
def test_decision_and_confidence_mapping(p: float, decision: bool, confidence: int) -> None:
    assert decision_and_confidence(p) == (decision, confidence)


@pytest.mark.parametrize("p", [-0.01, 1.01, float("nan"), float("inf")])
def test_out_of_range_probability_raises_never_clamps(p: float) -> None:
    with pytest.raises(ValueError):
        decision_and_confidence(p)


def test_confidence_comes_from_the_calibrated_probability() -> None:
    X, y = _overconfident_data(600, seed=11)
    fit = fit_calibrated(KIND, X, y)
    assert isinstance(fit, CalibratedFit)
    raw = make_estimator(KIND).fit(X, y)
    probe = X[:50]
    cal_p = _pos(fit.model, probe)
    raw_p = _pos(raw, probe)
    got = [decision_and_confidence(float(p))[1] for p in cal_p]
    assert got == [round(100 * max(p, 1 - p)) for p in cal_p]
    assert got != [round(100 * max(p, 1 - p)) for p in raw_p]


def test_prediction_rejects_wrong_width_or_names() -> None:
    names = ("a", "b")
    with pytest.raises(models.FeatureVectorError):
        models.vector_for({"a": 1.0}, names)
    with pytest.raises(models.FeatureVectorError):
        models.vector_for({"a": 1.0, "b": 2.0, "c": 3.0}, names)
    with pytest.raises(models.FeatureVectorError):
        models.check_width((1.0,), names)
    assert models.vector_for({"b": 2.0, "a": 1.0}, names) == (1.0, 2.0)


# ---------------------------------------------------------------------------
# NFR-002 — timing, generous bound
# ---------------------------------------------------------------------------


def test_retrain_and_calibrate_2000_labels_well_under_30s() -> None:
    X, y = _overconfident_data(2000, seed=12)
    started = time.perf_counter()
    fit = fit_calibrated(KIND, X, y)
    elapsed = time.perf_counter() - started
    assert isinstance(fit, CalibratedFit) and fit.method == "isotonic"
    print(f"NFR-002: fit_calibrated at 2000 labels took {elapsed:.2f}s")
    assert elapsed < 30.0
