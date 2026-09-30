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
from typing import Any

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


def positive_probability(model: Any, vector: Sequence[float]) -> float:
    """``P(y=1)`` for one positional vector from a fitted binary classifier."""
    proba = model.predict_proba([list(vector)])[0]
    classes = [int(c) for c in getattr(model, "classes_", range(len(proba)))]
    if 1 not in classes:
        raise ValueError(f"model classes {classes} carry no positive class")
    return float(proba[classes.index(1)])
