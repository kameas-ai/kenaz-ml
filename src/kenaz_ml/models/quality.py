"""Work quality estimator — rolling 30-minute quality score.

Computes a 0-100 score from recent workflow signals. The score is
rule-based initially but the component weights can be learned from
task outcome data. The LLM interprets scores and generates suggestions.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from kenaz_ml.modelstore import LocalModelStore, ModelStore

logger = logging.getLogger(__name__)

# Default component weights (sum to 100).
DEFAULT_WEIGHTS = {
    "test_pass_rate": 30,
    "edit_focus": 20,
    "velocity_vs_baseline": 20,
    "commit_frequency": 15,
    "no_revert_penalty": 15,
}

# Quality thresholds.
THRESHOLD_LOW = 40  # suggest break / switch task
THRESHOLD_HIGH = 70  # leave engineer alone or positive reinforcement


class QualityEstimator:
    """Computes a rolling work quality score from recent workflow signals.

    The score is a weighted sum of 5 components, each normalized to [0, 1].
    Component weights start at defaults and can be learned from task outcome
    data (tasks completed with high speed scores → weight the signals that
    were strong during those tasks).
    """

    def __init__(self, model_store: ModelStore | None = None) -> None:
        self._store = model_store or LocalModelStore()
        self.weights = dict(DEFAULT_WEIGHTS)
        self._load_weights()

    @classmethod
    def from_trained_model(
        cls, weights: dict[str, int] | None = None, store: ModelStore | None = None
    ) -> QualityEstimator:
        """Create an instance from pre-configured weights.

        Use this instead of ``__new__`` to avoid bypassing ``__init__``.

        Args:
            weights: Component weights dict. Uses DEFAULT_WEIGHTS if None.
            store: Optional ModelStore for persistence.
        """
        instance = object.__new__(cls)
        instance._store = store or LocalModelStore()
        instance.weights = dict(weights if weights is not None else DEFAULT_WEIGHTS)
        return instance

    def _load_weights(self) -> None:
        data = self._store.load("quality")
        if data is not None:
            try:
                saved = json.loads(data.decode("utf-8"))
                self.weights = saved.get("weights", self.weights)
                logger.info("quality: loaded learned weights from %s", type(self._store).__name__)
            except Exception as e:
                logger.warning("quality: failed to load weights: %s", e)

    def _save_weights(self) -> None:
        data = json.dumps({"weights": self.weights}).encode("utf-8")
        self._store.save("quality", data)

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        """Compute quality score from rolling window features.

        Features:
            test_pass_rate: float 0-1 (tests passed / tests run in window)
            test_total: int (total tests in window, 0 = no data)
            edit_focus: float 0-1 (1 - file_switch_rate; low switching = focused)
            velocity_ratio: float (current edit velocity / baseline average, capped at 2.0)
            commits_in_window: int (commits in the rolling window)
            expected_commits: float (typical commits in this window duration)
            revert_count: int (files edited back to previous state)
            edits_in_window: int (total edits)

        Returns:
            score: int 0-100
            components: dict of component name → subscore
            status: "degraded" | "normal" | "strong"
        """
        # Component 1: Test pass rate (0-1).
        test_total = features.get("test_total", 0)
        if test_total > 0:
            test_score = features.get("test_pass_rate", 0.5)
        else:
            test_score = 0.7  # no tests = neutral, slight benefit of doubt

        # Component 2: Edit focus (0-1). High = focused on few files.
        edit_focus = features.get("edit_focus", 0.5)

        # Component 3: Velocity vs baseline (0-1). Ratio capped at 2.0, normalized.
        velocity_ratio = min(features.get("velocity_ratio", 1.0), 2.0)
        velocity_score = velocity_ratio / 2.0  # 1.0 ratio = 0.5 score, 2.0 = 1.0

        # Component 4: Commit frequency (0-1).
        commits = features.get("commits_in_window", 0)
        expected = features.get("expected_commits", 1.0)
        if expected > 0:
            commit_score = min(commits / expected, 1.0)
        else:
            commit_score = 0.5  # no baseline

        # Component 5: No-revert penalty (0-1). Reverts reduce score.
        reverts = features.get("revert_count", 0)
        edits = max(features.get("edits_in_window", 1), 1)
        revert_ratio = reverts / edits
        no_revert_score = max(1.0 - revert_ratio * 5, 0.0)  # 20% reverts = 0 score

        # Weighted sum.
        components = {
            "test_pass_rate": round(test_score, 3),
            "edit_focus": round(edit_focus, 3),
            "velocity_vs_baseline": round(velocity_score, 3),
            "commit_frequency": round(commit_score, 3),
            "no_revert_penalty": round(no_revert_score, 3),
        }

        score = 0.0
        for name, subscore in components.items():
            score += subscore * self.weights.get(name, 0)

        score = int(round(score))
        score = max(0, min(100, score))

        # Status.
        if score < THRESHOLD_LOW:
            status = "degraded"
        elif score >= THRESHOLD_HIGH:
            status = "strong"
        else:
            status = "normal"

        return {
            "score": score,
            "components": components,
            "status": status,
            "threshold_low": THRESHOLD_LOW,
            "threshold_high": THRESHOLD_HIGH,
        }

    def train(self, task_outcomes: list[dict]) -> None:
        """Learn component weights from task outcome data.

        Each outcome dict should have:
            components: dict of component name → subscore (from predict())
            speed_score: float (task speed score — higher = better outcome)

        Learns weights that maximize correlation between weighted score
        and task speed score.
        """
        if len(task_outcomes) < 5:
            logger.info("quality: not enough outcomes to learn weights (%d)", len(task_outcomes))
            return

        # Build matrix: rows = tasks, cols = component subscores.
        names = list(DEFAULT_WEIGHTS.keys())
        X = np.zeros((len(task_outcomes), len(names)))
        y = np.zeros(len(task_outcomes))

        for i, outcome in enumerate(task_outcomes):
            comps = outcome.get("components", {})
            for j, name in enumerate(names):
                X[i, j] = comps.get(name, 0.5)
            y[i] = outcome.get("speed_score", 0.0)

        # Simple: weight each component by its correlation with speed score.
        correlations = np.zeros(len(names))
        for j in range(len(names)):
            col = X[:, j]
            if col.std() > 0 and y.std() > 0:
                correlations[j] = np.corrcoef(col, y)[0, 1]
            else:
                correlations[j] = 0.0

        # Convert correlations to weights (positive only, sum to 100).
        pos_corr = np.maximum(correlations, 0.01)  # floor at 0.01 to keep all components
        weight_sum = pos_corr.sum()
        if weight_sum > 0:
            learned = (pos_corr / weight_sum * 100).astype(int)
            # Adjust rounding to sum to exactly 100.
            diff = 100 - learned.sum()
            learned[0] += diff

            for j, name in enumerate(names):
                self.weights[name] = int(learned[j])

            self._save_weights()
            logger.info("quality: learned weights %s", self.weights)
