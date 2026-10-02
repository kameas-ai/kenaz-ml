"""Per-kind classic estimators for the harness advice trio (harness-recommendation-models-01MSK2RM).

WP01 defines the unfitted estimator per kind and the strict vector discipline;
WP02 wraps it in ``CalibratedClassifierCV``.

Strict features, always
-----------------------
A vector is built by ``[features[name] for name in contract.names]`` — never
``features.get(name, 0.0)``. A missing or unexpected feature name raises
:class:`FeatureVectorError` instead of defaulting to a silent zero, which would
be indistinguishable from a genuine zero and would teach (or query) the model
on a column of silence. The retained set already stores ``x`` positionally in
the header's order, so training never needs a dict at all; the dict form exists
for scoring a posted request.

No laya, no LLM (C-001, C-005); no dependency beyond scikit-learn/numpy
(NFR-001).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from kenaz_ml.advice.contracts import KIND_IDS

#: Hyperparameters per kind, in one place. Start from the ``stuck`` precedent
#: (``models/stuck.py``) — 100 shallow trees — which is conservative for the
#: trio's expected v1 volumes (hundreds of labels, design doc §4). A fixed
#: ``random_state`` makes a fit deterministic given identical data.
GBDT_PARAMS: dict[str, dict[str, Any]] = {
    kind: {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42} for kind in KIND_IDS
}

#: The fewest complete rows a fit is attempted on. Below it a GBDT is fitting
#: noise; the scheduler's own 50-new-labels trigger normally keeps a kind well
#: clear of it.
MIN_TRAINING_ROWS = 20


class FeatureVectorError(ValueError):
    """A vector is not exactly the contract's names, in order, as finite numbers."""


def make_estimator(kind: str) -> GradientBoostingClassifier:
    """A fresh, **unfitted** ``GradientBoostingClassifier`` for ``kind``.

    Raises ``KeyError`` for a kind with no registered hyperparameters — there
    is no generic default, so a new kind cannot train by accident.
    """
    return GradientBoostingClassifier(**GBDT_PARAMS[kind])


def vector_for(features: Mapping[str, Any], names: Sequence[str]) -> tuple[float, ...]:
    """The strict positional vector for ``features`` under ``names``.

    Raises :class:`FeatureVectorError` when a name is missing, a name is
    unexpected, or a value is not a finite number.
    """
    expected = tuple(names)
    posted = set(features)
    missing = [n for n in expected if n not in posted]
    unexpected = sorted(posted - set(expected))
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if unexpected:
            parts.append(f"unexpected {unexpected}")
        raise FeatureVectorError("; ".join(parts))
    vector = tuple(float(features[name]) for name in expected)
    if not all(math.isfinite(v) for v in vector):
        raise FeatureVectorError("features must be finite numbers")
    return vector


def check_width(vector: Sequence[float], names: Sequence[str]) -> tuple[float, ...]:
    """Reject a positional vector whose width is not the contract's."""
    if len(vector) != len(names):
        raise FeatureVectorError(f"vector has {len(vector)} value(s); the contract names {len(names)}")
    out = tuple(float(v) for v in vector)
    if not all(math.isfinite(v) for v in out):
        raise FeatureVectorError("features must be finite numbers")
    return out


# ---------------------------------------------------------------------------
# WP02 — calibration (FR-003, FR-004, FR-014)
# ---------------------------------------------------------------------------

#: D-B4: sigmoid (Platt) below this many labels **used by this fit**, isotonic
#: at or above. One constant, one explicit branch in :func:`calibration_method`.
ISOTONIC_MIN_LABELS = 1000
CalibrationMethod = Literal["sigmoid", "isotonic"]
METHOD_SIGMOID: CalibrationMethod = "sigmoid"
METHOD_ISOTONIC: CalibrationMethod = "isotonic"

#: Cross-validation folds for ``CalibratedClassifierCV``: as many as the data
#: allows up to this cap. ``cv=k`` needs at least ``k`` examples of each class
#: (stratified folds), so ``k = min(CALIBRATION_MAX_FOLDS, minority count)``.
#: Five is sklearn's default and keeps the fit cost at ``k+1`` GBDT fits.
CALIBRATION_MAX_FOLDS = 5
#: Below this minority-class count there is no meaningful cross-validation, and
#: the fit declines ("thin class") instead of letting sklearn raise.
CALIBRATION_MIN_FOLDS = 2

#: Equal-width reliability bins for :func:`expected_calibration_error`.
ECE_BINS = 10
#: The held-out share the reported ECE is measured on (stratified, fixed seed).
ECE_HOLDOUT_FRACTION = 0.2
#: Fewest rows for which a held-out ECE is measured. Below it the manifest
#: records ``calibration_ece: null`` with the reason, rather than a number
#: measured on a handful of rows or — worse — on the rows the calibrator fit.
ECE_HOLDOUT_MIN_ROWS = 50
_SPLIT_SEED = 42

DECLINE_EMPTY = "empty_training_set"
DECLINE_NON_FINITE = "non_finite_values"
DECLINE_SINGLE_CLASS = "single_class"
DECLINE_THIN_CLASS = "thin_minority_class"


@dataclass(frozen=True)
class CalibrationDecline:
    """Why a calibrated fit was not attempted. Returned, never raised."""

    reason: str
    detail: str
    counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibratedFit:
    """A freshly fitted ``CalibratedClassifierCV`` and the evidence about it."""

    model: Any
    method: str
    folds: int
    n_labels: int
    calibration_ece: float | None
    uncalibrated_ece: float | None
    holdout_rows: int
    ece_note: str | None = None

    def metrics(self) -> dict[str, Any]:
        """The manifest ``metrics`` entries this fit contributes (FR-009's open dict)."""
        out: dict[str, Any] = {
            "calibration_fitted": True,
            "calibration_method": self.method,
            "calibration_cv_folds": self.folds,
            "calibration_n_labels": self.n_labels,
            "calibration_ece": self.calibration_ece,
            "uncalibrated_ece": self.uncalibrated_ece,
            "calibration_holdout_rows": self.holdout_rows,
            "ece_bins": ECE_BINS,
        }
        if self.ece_note:
            out["calibration_ece_note"] = self.ece_note
        return out


def calibration_method(n_labels: int) -> CalibrationMethod:
    """D-B4: ``sigmoid`` below :data:`ISOTONIC_MIN_LABELS` fit-time labels, ``isotonic`` at or above."""
    if n_labels < ISOTONIC_MIN_LABELS:
        return METHOD_SIGMOID
    return METHOD_ISOTONIC


def check_trainable(X: Any, y: Any) -> CalibrationDecline | None:
    """Everything that must be true before sklearn is called (FR-014). ``None`` when fine."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.size == 0 or y.size == 0 or X.shape[0] != y.shape[0]:
        return CalibrationDecline(DECLINE_EMPTY, "no rows to fit", {"rows": int(y.size)})
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        return CalibrationDecline(DECLINE_NON_FINITE, "training data contains NaN or infinity")
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    counts = {"positive": positives, "negative": negatives}
    if positives + negatives != y.size:
        return CalibrationDecline(DECLINE_NON_FINITE, "labels must be 0 or 1", counts)
    if positives == 0 or negatives == 0:
        return CalibrationDecline(DECLINE_SINGLE_CLASS, "only one label class present: cannot calibrate", counts)
    if min(positives, negatives) < CALIBRATION_MIN_FOLDS:
        return CalibrationDecline(
            DECLINE_THIN_CLASS,
            f"minority class has {min(positives, negatives)} example(s); "
            f"cross-validated calibration needs at least {CALIBRATION_MIN_FOLDS}",
            counts,
        )
    return None


def _folds(y: np.ndarray) -> int:
    minority = int(min((y == 1).sum(), (y == 0).sum()))
    return max(CALIBRATION_MIN_FOLDS, min(CALIBRATION_MAX_FOLDS, minority))


def _calibrated(kind: str, method: CalibrationMethod, folds: int) -> Any:
    from sklearn.calibration import CalibratedClassifierCV

    # A fresh, unfitted estimator every time: nothing from a prior generation,
    # another kind or another install is ever refit (FR-004).
    return CalibratedClassifierCV(estimator=make_estimator(kind), method=method, cv=folds)


def _holdout_ece(
    kind: str, X: np.ndarray, y: np.ndarray, method: CalibrationMethod
) -> tuple[float | None, float | None, int, str | None]:
    """ECE of the same fitting procedure, measured on rows it did not fit."""
    from sklearn.model_selection import train_test_split

    if len(y) < ECE_HOLDOUT_MIN_ROWS:
        return None, None, 0, f"fewer than {ECE_HOLDOUT_MIN_ROWS} rows: held-out ECE not measured"
    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X, y, test_size=ECE_HOLDOUT_FRACTION, stratify=y, random_state=_SPLIT_SEED
    )
    if check_trainable(X_fit, y_fit) is not None:
        return None, None, 0, "fit split too thin for cross-validated calibration: held-out ECE not measured"
    calibrated = _calibrated(kind, method, _folds(y_fit)).fit(X_fit, y_fit)
    raw = make_estimator(kind).fit(X_fit, y_fit)
    cal_p = calibrated.predict_proba(X_hold)[:, list(calibrated.classes_).index(1)]
    raw_p = raw.predict_proba(X_hold)[:, list(raw.classes_).index(1)]
    return (
        expected_calibration_error(y_hold, cal_p),
        expected_calibration_error(y_hold, raw_p),
        int(len(y_hold)),
        None,
    )


def fit_calibrated(kind: str, X: Any, y: Any) -> CalibratedFit | CalibrationDecline:
    """Fit ``kind``'s GBDT inside ``CalibratedClassifierCV`` on exactly ``X, y`` (FR-003).

    The method is chosen from ``len(y)`` — the labels used by *this* fit (D-B4).
    The reported ECE is measured on a stratified held-out split by fitting the
    identical procedure on the rest; the served model is then refit on all rows,
    so no label is withheld from it. Declines (never raises) on the FR-014 cases.
    """
    decline = check_trainable(X, y)
    if decline is not None:
        return decline
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n = int(len(y))
    method = calibration_method(n)
    cal_ece, raw_ece, holdout, note = _holdout_ece(kind, X, y, method)
    folds = _folds(y)
    model = _calibrated(kind, method, folds).fit(X, y)
    return CalibratedFit(
        model=model,
        method=method,
        folds=folds,
        n_labels=n,
        calibration_ece=cal_ece,
        uncalibrated_ece=raw_ece,
        holdout_rows=holdout,
        ece_note=note,
    )


def expected_calibration_error(y_true: Any, p: Any, n_bins: int = ECE_BINS) -> float:
    """Expected calibration error of positive-class probabilities ``p`` against 0/1 ``y_true``.

    Binning (shared with WP03 and the tests): ``n_bins`` equal-width bins over
    ``[0, 1]``; bin ``i`` holds ``i/n <= p < (i+1)/n``, the last bin also takes
    ``p == 1``. ECE is ``sum_b (|b| / N) * |mean(y in b) - mean(p in b)|`` —
    weighted by bin population; empty bins contribute nothing. Plain numpy.
    """
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    if y_arr.size == 0:
        return 0.0
    bins = np.minimum((p_arr * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        count = int(mask.sum())
        if count:
            total += (count / y_arr.size) * abs(float(y_arr[mask].mean()) - float(p_arr[mask].mean()))
    return float(total)


def decision_and_confidence(p: float) -> tuple[bool, int]:
    """The ruled binary mapping (R4 / D-B7) from a **calibrated** ``P(y=1)``.

    ``decision = p >= 0.5``; ``confidence = round(100 * max(p, 1 - p))`` — the
    confidence in the chosen side, so it is never below 50. Validated and never
    clamped: a probability outside ``[0, 1]`` (or non-finite) raises
    ``ValueError``, which Mission A's dispatch turns into a typed refusal.
    """
    p = float(p)
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        raise ValueError(f"calibrated probability {p!r} is outside [0, 1]")
    confidence = round(100 * max(p, 1.0 - p))
    if not 0 <= confidence <= 100:  # pragma: no cover - unreachable for p in [0, 1]
        raise ValueError(f"confidence {confidence} is outside 0-100")
    return p >= 0.5, int(confidence)


def positive_probability(model: Any, vector: Sequence[float]) -> float:
    """``P(y=1)`` for one positional vector from a fitted binary classifier."""
    proba = model.predict_proba([list(vector)])[0]
    classes = [int(c) for c in getattr(model, "classes_", range(len(proba)))]
    if 1 not in classes:
        raise ValueError(f"model classes {classes} carry no positive class")
    return float(proba[classes.index(1)])
