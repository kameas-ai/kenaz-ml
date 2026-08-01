"""Trainer: reads from the DataStore and retrains all models."""

from __future__ import annotations

import json
import logging
import time

import numpy as np

from sigil_ml.datastore import DataStore
from sigil_ml.features import extract_duration_features, extract_stuck_features
from sigil_ml.models.duration import FEATURE_NAMES as DURATION_FEATURES
from sigil_ml.models.duration import DurationEstimator
from sigil_ml.models.stuck import FEATURE_NAMES as STUCK_FEATURES
from sigil_ml.models.stuck import StuckPredictor
from sigil_ml.modelstore import ModelStore
from sigil_ml.training.synthetic import generate_duration_data, generate_stuck_data

logger = logging.getLogger(__name__)


def _reference_time_for(task: dict) -> int | None:
    """Epoch ms describing the moment a completed task's features should reflect.

    A training example describes an instant in the past, so its features must be
    measured against that instant. For a completed task that is its completion;
    `last_active` is the closest honest fallback when `completed_at` is unset.

    Returns None when the task cannot yield an honest reference time, in which
    case it must be excluded from training. There is deliberately no wall-clock
    fallback: substituting "now" for a historical example is precisely the defect
    this resolver exists to remove, and it would bite hardest on the malformed
    rows most likely to be pathological.

    `0` and `None` are treated alike -- a zero epoch is not a real timestamp in
    this data.
    """
    for field in ("completed_at", "last_active"):
        value = task.get(field)
        if value:
            return int(value)
    return None


class Trainer:
    """Orchestrates training of all kameas-ml models from local data."""

    def __init__(self, store: DataStore, model_store: ModelStore | None = None) -> None:
        self.store = store
        self._model_store = model_store

    def train_all(self) -> dict:
        """Train all models and return a summary.

        Returns:
            {"trained": [model_names], "samples": int, "duration_sec": float}
        """
        start = time.time()
        trained = []
        total_samples = 0

        stuck_result = self._train_stuck()
        if stuck_result:
            trained.append("stuck")
            total_samples += stuck_result

        duration_result = self._train_duration()
        if duration_result:
            trained.append("duration")
            total_samples += duration_result

        # Signal model training (additive)
        try:
            pattern_result = self._train_pattern_detector()
            if pattern_result:
                trained.append("pattern_detector")
                total_samples += pattern_result
        except Exception:
            logger.warning("Signal model training failed: pattern_detector", exc_info=True)

        try:
            next_action_result = self._train_next_action()
            if next_action_result:
                trained.append("next_action")
                total_samples += next_action_result
        except Exception:
            logger.warning("Signal model training failed: next_action", exc_info=True)

        try:
            recommender_result = self._train_file_recommender()
            if recommender_result:
                trained.append("file_recommender")
                total_samples += recommender_result
        except Exception:
            logger.warning("Signal model training failed: file_recommender", exc_info=True)

        elapsed = time.time() - start
        return {
            "trained": trained,
            "samples": total_samples,
            "duration_sec": round(elapsed, 2),
        }

    def _train_stuck(self) -> int:
        """Train the stuck predictor from completed tasks.

        Returns:
            Number of samples used, or 0 if insufficient data.
        """
        task_ids = self.store.get_completed_task_ids()

        X_list = []
        y_list = []
        skipped = 0
        for task_id in task_ids:
            task = self.store.get_task_by_id(task_id)
            if task is None:
                skipped += 1
                continue
            as_of = _reference_time_for(task)
            if as_of is None:
                skipped += 1
                continue
            features = extract_stuck_features(self.store, task_id, as_of_ms=as_of)
            x = [features.get(f, 0.0) for f in STUCK_FEATURES]
            X_list.append(x)
            # Heuristic label: stuck if high test failures and long time in phase
            stuck = features["test_failure_count"] > 3 and features["time_in_phase_sec"] > 600
            y_list.append(1.0 if stuck else 0.0)

        if skipped:
            logger.info("stuck training: skipped %d task(s) with no resolvable reference time", skipped)

        # Threshold is evaluated against the usable count, not the raw count --
        # a skipped task is not training data.
        if len(X_list) < 10:
            logger.info(
                "Not enough completed tasks for stuck training (%d usable of %d)",
                len(X_list),
                len(task_ids),
            )
            X, y = generate_stuck_data(500)
            predictor = StuckPredictor(model_store=self._model_store)
            predictor.train(X, y)
            return 500

        X = np.array(X_list)
        y = np.array(y_list)

        predictor = StuckPredictor(model_store=self._model_store)
        predictor.train(X, y)
        return len(X)

    def _train_duration(self) -> int:
        """Train the duration estimator from completed tasks.

        Returns:
            Number of samples used, or 0 if insufficient data.
        """
        rows = self.store.get_completed_tasks_with_timestamps()

        X_list = []
        y_list = []
        skipped = 0
        for row in rows:
            task_id = row["id"]
            as_of = _reference_time_for(row)
            # The label needs both timestamps, so a row missing either is
            # unusable for the same reason an unresolvable reference time is.
            if as_of is None or not row.get("started_at") or not row.get("completed_at"):
                skipped += 1
                continue
            features = extract_duration_features(self.store, task_id, as_of_ms=as_of)
            x = [features.get(f, 0.0) for f in DURATION_FEATURES]
            X_list.append(x)
            # Duration in minutes
            duration_min = (row["completed_at"] - row["started_at"]) / 60000.0
            y_list.append(max(duration_min, 1.0))

        if skipped:
            logger.info(
                "duration training: skipped %d task(s) with no resolvable reference time or duration label",
                skipped,
            )

        # Threshold is evaluated against the usable count, not the raw count.
        if len(X_list) < 10:
            logger.info(
                "Not enough completed tasks for duration training (%d usable of %d)",
                len(X_list),
                len(rows),
            )
            X, y = generate_duration_data(500)
            estimator = DurationEstimator(model_store=self._model_store)
            estimator.train(X, y)
            return 500

        X = np.array(X_list)
        y = np.array(y_list)

        estimator = DurationEstimator(model_store=self._model_store)
        estimator.train(X, y)
        return len(X)

    # --- Signal model training (WP07) ---

    def _train_pattern_detector(self) -> int:
        """Train the PatternDetector's IsolationForest from feedback labels.

        Returns:
            Number of samples used, or 0 if insufficient data.
        """
        feedback = self.store.get_signal_feedback(since_ms=0)

        if len(feedback) < 500:
            logger.info(
                "Not enough feedback for pattern detector training (%d, need 500)",
                len(feedback),
            )
            return 0

        # Build feature matrix from signal evidence
        X_list: list[list[float]] = []
        for fb in feedback:
            try:
                evidence = fb.get("evidence") or {}
                if isinstance(evidence, str):
                    evidence = json.loads(evidence)
                features = self._extract_pattern_features(evidence)
                if features is not None:
                    X_list.append(features)
            except Exception:
                continue

        if len(X_list) < 100:
            logger.info("Not enough valid pattern features (%d)", len(X_list))
            return 0

        X = np.array(X_list)

        from sigil_ml.signals.pattern_detector import PatternDetector

        detector = PatternDetector()
        detector.train(X)  # IsolationForest is unsupervised
        detector.save(self._model_store)

        return len(X)

    def _extract_pattern_features(self, evidence: dict) -> list[float] | None:
        """Extract a feature vector from signal evidence for IsolationForest training."""
        source = evidence.get("source_model")
        if source != "pattern_detector":
            return None

        observed = evidence.get("observed")
        baseline_mean = evidence.get("baseline_mean")
        baseline_std = evidence.get("baseline_std")
        z_score = evidence.get("z_score")

        if any(v is None for v in [observed, baseline_mean, baseline_std, z_score]):
            return None

        return [float(observed), float(baseline_mean), float(baseline_std), float(z_score)]

    def _train_next_action(self) -> int:
        """Rebuild n-gram model from completed task event sequences.

        Returns:
            Number of tokens processed, or 0 if insufficient data.
        """
        task_ids = self.store.get_completed_task_ids()

        if len(task_ids) < 10:
            logger.info(
                "Not enough completed tasks for next-action training (%d, need 10)",
                len(task_ids),
            )
            return 0

        from sigil_ml.features import extract_action_token
        from sigil_ml.signals.next_action import NextActionPredictor

        predictor = NextActionPredictor()
        predictor.reset()  # Start fresh for full rebuild
        total_tokens = 0

        # Create classifier once before the loop
        from sigil_ml.models.activity import ActivityClassifier

        classifier = ActivityClassifier(model_store=self._model_store)

        for task_id in task_ids:
            events = self.store.get_events_for_task(task_id)
            if not events:
                continue

            # Classify events (needed for composite tokens)
            for e in events:
                if "_category" not in e:
                    result = classifier.classify(e)
                    e["_category"] = result["category"]

            tokens = [extract_action_token(e) for e in events]
            predictor.train_incremental(tokens)
            total_tokens += len(tokens)

        if total_tokens > 0:
            predictor.save(self._model_store)

        return total_tokens

    def _train_file_recommender(self) -> int:
        """Rebuild co-occurrence matrix from completed task file sets.

        Returns:
            Number of tasks processed, or 0 if insufficient data.
        """
        from sigil_ml.signals.file_recommender import FileRecommender

        recommender = FileRecommender()
        task_count = recommender.train_from_tasks(self.store)

        if task_count < 5:
            logger.info(
                "Not enough tasks with file data for recommender training (%d, need 5)",
                task_count,
            )
            return 0

        recommender.save(self._model_store)
        return task_count
