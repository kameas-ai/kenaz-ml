"""Cloud training orchestrator using DataStore and ModelStore abstractions.

Supports per-tenant, batch (all-tenants), and aggregate training modes.
Does NOT subclass or modify the local Trainer -- it is a parallel
implementation that reuses model classes' .train() methods directly.

Two ways to obtain a training set
---------------------------------
Without a feature store, this trainer *replays* the extractors over history,
passing each example its own reference time. That is what it has always done and
what the local trainer still does.

With a feature store injected, it *retrieves* instead: it builds an entity frame
of ``task_id`` + ``event_timestamp`` + label, one row per example, and asks Feast
for each row's features as of that row's own moment (FR-007). The values come
from ``ml_features``, stamped at materialization time with the moment they
describe (FR-009), so no value recorded after an example can appear in it
(FR-008).

The labels are identical in both paths, expression for expression. This changes
where features come from, not what is being learned. What it buys is that the
training set is now a retrieval whose point-in-time correctness can be asserted,
rather than a recomputation whose correctness rests on every caller remembering
to pass ``as_of_ms``.
"""

from __future__ import annotations

import io
import logging
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np

from sigil_ml.datastore import DataStore
from sigil_ml.feature_store.materialize import (
    OFFLINE_FEATURE_VIEWS,
    OfflineFeatureView,
    OfflineStoreUnavailableError,
    PartialRetrievalError,
    feature_service_version,
)
from sigil_ml.features import (
    extract_duration_features_from_data,
    extract_stuck_features_from_data,
)
from sigil_ml.models.duration import FEATURE_NAMES as DURATION_FEATURES
from sigil_ml.models.duration import DurationEstimator
from sigil_ml.models.stuck import FEATURE_NAMES as STUCK_FEATURES
from sigil_ml.models.stuck import StuckPredictor
from sigil_ml.modelstore import ModelStore
from sigil_ml.training.models import (
    CloudTrainingConfig,
    TrainingBatch,
    TrainingRun,
)
from sigil_ml.training.synthetic import generate_duration_data, generate_stuck_data

# Imported rather than reimplemented: local and cloud training must resolve a
# training example's reference time by exactly the same rule, or they will drift.
from sigil_ml.training.trainer import _reference_time_for

logger = logging.getLogger(__name__)

AGGREGATE_TENANT_ID = "__aggregate__"

#: Maps a model name to the feature view whose values it consumes. The feature
#: *service* carries the same name as the model (definitions.py), because that
#: name is what Go already queries in `ml_predictions.model` and is not
#: renameable here.
RETRIEVED_MODELS: dict[str, OfflineFeatureView] = {
    "stuck": OFFLINE_FEATURE_VIEWS["stuck_features"],
    "duration": OFFLINE_FEATURE_VIEWS["duration_features"],
}


def _vector_names(model_name: str, feature_names: Sequence[str]) -> tuple[str, ...]:
    """Return the ordered names a replayed vector is indexed by (FR-007).

    The registered feature service's ordered contract when it agrees with the
    module's own ``FEATURE_NAMES``, and ``FEATURE_NAMES`` otherwise. Establishing
    that they agree is what makes the strict lookup at the construction sites
    safe: every name indexed there is a name this extractor emits.

    Falling back rather than refusing is the right call here specifically
    because vector *construction* is unaffected either way -- both lists name the
    same features when nothing is wrong, and when something is wrong the
    extractor's own list is the one its output actually satisfies. What the
    fallback loses is the ability to stamp the run with a contract version, and
    the cloud path records its feature set through
    :class:`FeatureSetRecord`/``ml_events`` rather than through a manifest, so
    the disagreement is reported there rather than silently absorbed.

    **No retained set and no manifest are written from the cloud path** (T020,
    C-003): retention is a local-personalization concept and the artifacts do
    not live on a filesystem this process owns.
    """
    from sigil_ml.modelstore.registry import local_feature_contract

    try:
        contract = local_feature_contract(model_name)
    except Exception:
        logger.warning("cloud training: cannot read the feature contract for %r", model_name, exc_info=True)
        return tuple(feature_names)

    if contract is None:
        return tuple(feature_names)

    if tuple(contract.names) != tuple(feature_names):
        logger.error(
            "cloud training: the %r feature service declares %s but this build extracts %s",
            model_name,
            list(contract.names),
            list(feature_names),
        )
        return tuple(feature_names)

    return tuple(contract.names)


@dataclass(frozen=True)
class FeatureSetRecord:
    """Which feature contract a training run consumed (FR-010).

    ``version`` is a content hash of the service's ordered feature references,
    so it moves exactly when the contract does. Recorded on the trainer, logged,
    and written to ``ml_events`` -- the audit table Python owns -- so a model can
    be traced back to the feature set that produced it after the fact.
    """

    model: str
    feature_service: str
    version: str
    feature_names: tuple[str, ...]
    rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "feature_service": self.feature_service,
            "version": self.version,
            "feature_names": list(self.feature_names),
            "rows": self.rows,
        }


@dataclass(frozen=True)
class _RetrievedFeatures:
    """A point-in-time training set for one model, in entity-frame row order."""

    task_ids: list[str]
    #: Positional vectors, built against the model's FEATURE_NAMES ordering.
    vectors: list[list[float]]
    #: The same values keyed by name, for the label expressions.
    features: list[dict[str, float]]
    record: FeatureSetRecord


class CloudTrainer:
    """Orchestrates model training for cloud deployments.

    Uses DataStore for reading training data and ModelStore for
    persisting trained model weights. Supports per-tenant, batch,
    and aggregate training modes.
    """

    def __init__(
        self,
        data_store: DataStore,
        model_store: ModelStore,
        config: CloudTrainingConfig | None = None,
        training_lock: Any | None = None,
        feature_store: Any | None = None,
    ) -> None:
        """Construct a cloud trainer.

        Args:
            feature_store: A Feast ``FeatureStore``. When given, training sets
                are retrieved point-in-time from the offline store instead of
                being recomputed. Build one with
                :func:`sigil_ml.feature_store.materialize.feature_store_for_cloud`.
                Left as ``None``, behaviour is unchanged from before this
                migration.
        """
        self.data_store = data_store
        self.model_store = model_store
        self.config = config or CloudTrainingConfig()
        self.training_lock = training_lock
        self.feature_store = feature_store
        #: Feature contracts consumed by the most recent training run (FR-010).
        self.last_feature_sets: list[FeatureSetRecord] = []

    # ------------------------------------------------------------------
    # Per-tenant training (WP02)
    # ------------------------------------------------------------------

    def train_tenant(self, tenant_id: str) -> TrainingRun:
        """Train all models for a single tenant.

        Flow: lock -> interval check -> threshold check -> train -> save -> audit.
        """
        start = time.time()
        started_at = datetime.now(timezone.utc)

        # Lock check (if configured) -- before ANY other operation (WP04)
        if self.training_lock is not None:
            if not self.training_lock.acquire(tenant_id):
                return TrainingRun(
                    tenant_id=tenant_id,
                    status="skipped_locked",
                    duration_ms=int((time.time() - start) * 1000),
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                )

        try:
            return self._train_tenant_inner(tenant_id, start, started_at)
        except Exception as e:
            elapsed = time.time() - start
            logger.error(
                "Training failed for tenant %s: %s",
                tenant_id,
                e,
                exc_info=True,
            )
            run = TrainingRun(
                tenant_id=tenant_id,
                status="failed",
                error=str(e)[:500],
                duration_ms=int(elapsed * 1000),
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
            self._record_audit_event(tenant_id, run)
            return run
        finally:
            if self.training_lock is not None:
                self.training_lock.release(tenant_id)

    def _train_tenant_inner(self, tenant_id: str, start: float, started_at: datetime) -> TrainingRun:
        """Core per-tenant training logic (interval, threshold, train, audit)."""
        # 1. Interval check (cheapest -- avoids all data queries if too recent)
        last_ts = self._get_last_training_ts(tenant_id)
        if last_ts is not None:
            elapsed_sec = time.time() - (last_ts / 1000.0)
            if elapsed_sec < self.config.min_interval_sec:
                logger.info(
                    "Skipping tenant %s: trained %d sec ago (interval: %d sec)",
                    tenant_id,
                    int(elapsed_sec),
                    self.config.min_interval_sec,
                )
                run = TrainingRun(
                    tenant_id=tenant_id,
                    status="skipped",
                    duration_ms=int((time.time() - start) * 1000),
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                )
                self._record_audit_event(tenant_id, run)
                return run

        # 2. Data threshold check
        tasks = self._query_completed_tasks(tenant_id)
        has_sufficient_data = len(tasks) >= self.config.min_tasks

        # 3. Synthetic fallback if insufficient data
        if not has_sufficient_data:
            logger.info(
                "Tenant %s has %d tasks (< %d), using synthetic data",
                tenant_id,
                len(tasks),
                self.config.min_tasks,
            )
            models_trained = self._train_synthetic(tenant_id)
            elapsed_ms = int((time.time() - start) * 1000)
            run = TrainingRun(
                tenant_id=tenant_id,
                status="trained",
                sample_count=500,
                models_trained=models_trained,
                duration_ms=elapsed_ms,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
            self._record_audit_event(tenant_id, run)
            return run

        # 4. Real data: extract features and build task_events map
        task_events: dict[str, list[dict]] = {}
        for task in tasks:
            events = self._query_events_for_task(tenant_id, task["id"])
            task_events[task["id"]] = events

        # 5. Train all models from real data
        models_trained = self._train_models_from_tasks(tasks, task_events, tenant_id)

        elapsed_ms = int((time.time() - start) * 1000)
        run = TrainingRun(
            tenant_id=tenant_id,
            status="trained",
            sample_count=len(tasks),
            models_trained=models_trained,
            duration_ms=elapsed_ms,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )
        self._record_audit_event(tenant_id, run)
        return run

    def _train_synthetic(self, tenant_id: str) -> list[str]:
        """Train stuck and duration models with synthetic data (cold-start)."""
        models_trained: list[str] = []

        # Stuck predictor -- synthetic
        try:
            X_stuck, y_stuck = generate_stuck_data(500)
            stuck = StuckPredictor(model_store=self.model_store)
            stuck.train(X_stuck, y_stuck)
            self._save_model_to_store("stuck", stuck, tenant_id)
            models_trained.append("stuck")
        except Exception:
            logger.warning(
                "Failed to train synthetic stuck model for tenant %s",
                tenant_id,
                exc_info=True,
            )

        # Duration estimator -- synthetic
        try:
            X_dur, y_dur = generate_duration_data(500)
            duration = DurationEstimator(model_store=self.model_store)
            duration.train(X_dur, y_dur)
            self._save_model_to_store("duration", duration, tenant_id)
            models_trained.append("duration")
        except Exception:
            logger.warning(
                "Failed to train synthetic duration model for tenant %s",
                tenant_id,
                exc_info=True,
            )

        # Next-action predictor -- synthetic
        try:
            from sigil_ml.signals.next_action import NextActionPredictor
            from sigil_ml.training.synthetic import generate_next_action_data

            sequences = generate_next_action_data(500)
            predictor = NextActionPredictor()
            for seq in sequences:
                predictor.train_incremental(seq)

            buf = io.BytesIO()
            data = {
                "ngrams": dict(predictor._ngrams),
                "total_tokens": predictor._total_tokens,
                "n": predictor._n,
            }
            joblib.dump(data, buf)
            scoped_name = f"{tenant_id}/next_action"
            self.model_store.save(scoped_name, buf.getvalue())
            models_trained.append("next_action")
        except Exception:
            logger.warning(
                "Failed to train synthetic next_action for tenant %s",
                tenant_id,
                exc_info=True,
            )

        return models_trained

    def _train_models_from_tasks(
        self,
        tasks: list[dict],
        task_events: dict[str, list[dict]],
        tenant_id: str,
    ) -> list[str]:
        """Train all model types from provided tasks and events.

        Shared by train_tenant() (single-tenant) and train_aggregate() (pooled).

        Returns:
            List of model names that were successfully trained.
        """
        models_trained: list[str] = []

        # Retrieval happens up front and deliberately OUTSIDE the per-model
        # try/except blocks below. Those blocks log-and-continue, which is right
        # for a model that fails to fit but catastrophic for a store that failed
        # to answer: it would leave the run reporting success having trained on
        # nothing, or on a subset. FR-017 wants the opposite -- let it propagate.
        retrieved: dict[str, _RetrievedFeatures] | None = None
        if self.feature_store is not None:
            retrieved = self._retrieve_offline_features(tasks, tenant_id)

        # Resolved once per run, not once per task: the contract cannot change
        # mid-run, and this is the validation the strict lookups below rely on.
        stuck_names = _vector_names("stuck", STUCK_FEATURES)
        duration_names = _vector_names("duration", DURATION_FEATURES)

        # --- Stuck predictor ---
        try:
            X_stuck_list: list[list[float]] = []
            y_stuck_list: list[float] = []
            stuck_skipped = 0
            if retrieved is not None:
                stuck_retrieved = retrieved["stuck"]
                X_stuck_list = list(stuck_retrieved.vectors)
                for feats in stuck_retrieved.features:
                    # Heuristic label: stuck if high test failures AND long time in phase
                    stuck = feats["test_failure_count"] > 3 and feats["time_in_phase_sec"] > 600
                    y_stuck_list.append(1.0 if stuck else 0.0)
            else:
                for task in tasks:
                    as_of = _reference_time_for(task)
                    if as_of is None:
                        stuck_skipped += 1
                        continue
                    events = task_events.get(task["id"], [])
                    feats = extract_stuck_features_from_data(task, events, as_of_ms=as_of)
                    # Strict lookup (FR-007), safe only because
                    # _vector_names() validated `stuck_names` against this
                    # extractor's own feature list before the loop. A KeyError
                    # here means that validation was skipped, which is a defect
                    # to fix -- not a data condition to hide behind a defaulted
                    # zero indistinguishable from a real one.
                    x = [feats[f] for f in stuck_names]
                    X_stuck_list.append(x)
                    # Heuristic label: stuck if high test failures AND long time in phase
                    stuck = feats["test_failure_count"] > 3 and feats["time_in_phase_sec"] > 600
                    y_stuck_list.append(1.0 if stuck else 0.0)

            if stuck_skipped:
                logger.info(
                    "stuck training: skipped %d task(s) with no resolvable reference time (tenant %s)",
                    stuck_skipped,
                    tenant_id,
                )

            if X_stuck_list:
                X_stuck = np.array(X_stuck_list)
                y_stuck = np.array(y_stuck_list)
                stuck_predictor = StuckPredictor(model_store=self.model_store)
                stuck_predictor.train(X_stuck, y_stuck)
                self._save_model_to_store("stuck", stuck_predictor, tenant_id)
                models_trained.append("stuck")
        except Exception:
            logger.warning(
                "Failed to train stuck model for tenant %s",
                tenant_id,
                exc_info=True,
            )

        # --- Duration estimator ---
        try:
            X_dur_list: list[list[float]] = []
            y_dur_list: list[float] = []
            dur_skipped = 0
            if retrieved is not None:
                duration_retrieved = retrieved["duration"]
                X_dur_list = list(duration_retrieved.vectors)
                tasks_by_id = {t["id"]: t for t in tasks}
                for task_id in duration_retrieved.task_ids:
                    task = tasks_by_id[task_id]
                    # Duration label: (completed_at - started_at) in minutes, min 1.0
                    started = task.get("started_at", 0)
                    completed = task.get("completed_at", 0)
                    if started and completed:
                        duration_min = (completed - started) / 60000.0
                        y_dur_list.append(max(duration_min, 1.0))
                    else:
                        y_dur_list.append(60.0)  # default 60 min if timestamps missing
            else:
                for task in tasks:
                    as_of = _reference_time_for(task)
                    if as_of is None:
                        dur_skipped += 1
                        continue
                    events = task_events.get(task["id"], [])
                    feats = extract_duration_features_from_data(task, events, as_of_ms=as_of)
                    # Strict lookup (FR-007) -- see the note in the stuck branch
                    # above. Validation ran first; a KeyError is a defect.
                    x = [feats[f] for f in duration_names]
                    X_dur_list.append(x)
                    # Duration label: (completed_at - started_at) in minutes, min 1.0
                    started = task.get("started_at", 0)
                    completed = task.get("completed_at", 0)
                    if started and completed:
                        duration_min = (completed - started) / 60000.0
                        y_dur_list.append(max(duration_min, 1.0))
                    else:
                        y_dur_list.append(60.0)  # default 60 min if timestamps missing

            if dur_skipped:
                logger.info(
                    "duration training: skipped %d task(s) with no resolvable reference time (tenant %s)",
                    dur_skipped,
                    tenant_id,
                )

            if X_dur_list:
                X_dur = np.array(X_dur_list)
                y_dur = np.array(y_dur_list)
                dur_estimator = DurationEstimator(model_store=self.model_store)
                dur_estimator.train(X_dur, y_dur)
                self._save_model_to_store("duration", dur_estimator, tenant_id)
                models_trained.append("duration")
        except Exception:
            logger.warning(
                "Failed to train duration model for tenant %s",
                tenant_id,
                exc_info=True,
            )

        # --- Activity classifier ---
        # Rule-based by default (no ML training needed for cold-start).
        # ActivityClassifier uses heuristic classification without trained weights.
        # Skip training; default rules apply.

        # --- Workflow state predictor ---
        # Also rule-based by default. Skip training; default rules apply.

        # --- Quality estimator ---
        # Uses weight-based scoring. Skip training; default weights apply.

        # --- Signal models (additive) ---

        # Next-Action Predictor: rebuild n-grams from task event sequences
        try:
            from sigil_ml.features import extract_action_token
            from sigil_ml.models.activity import ActivityClassifier
            from sigil_ml.signals.next_action import NextActionPredictor

            predictor = NextActionPredictor()
            predictor.reset()
            total_tokens = 0

            classifier = ActivityClassifier(model_store=self.model_store)
            for task in tasks:
                events = task_events.get(task["id"], [])
                for e in events:
                    if "_category" not in e:
                        result = classifier.classify(e)
                        e["_category"] = result["category"]
                tokens = [extract_action_token(e) for e in events]
                predictor.train_incremental(tokens)
                total_tokens += len(tokens)

            if total_tokens > 0:
                buf = io.BytesIO()
                data = {
                    "ngrams": dict(predictor._ngrams),
                    "total_tokens": predictor._total_tokens,
                    "n": predictor._n,
                }
                joblib.dump(data, buf)
                scoped_name = f"{tenant_id}/next_action"
                self.model_store.save(scoped_name, buf.getvalue())
                models_trained.append("next_action")
        except Exception:
            logger.warning(
                "Failed to train next_action model for tenant %s",
                tenant_id,
                exc_info=True,
            )

        # File Recommender: rebuild co-occurrence from task file sets
        try:
            from sigil_ml.signals.file_recommender import FileRecommender

            recommender = FileRecommender()
            for task in tasks:
                events = task_events.get(task["id"], [])
                files = recommender._extract_files_from_events(events)
                if len(files) < 2:
                    continue
                recommender._task_count += 1
                for f in files:
                    recommender._file_counts[f] += 1
                    for g in files:
                        if f != g:
                            recommender._cooccurrence[f][g] += 1

            if recommender._task_count >= 5:
                buf = io.BytesIO()
                data = {
                    "cooccurrence": dict(recommender._cooccurrence),
                    "file_counts": dict(recommender._file_counts),
                    "task_count": recommender._task_count,
                }
                joblib.dump(data, buf)
                scoped_name = f"{tenant_id}/file_recommender"
                self.model_store.save(scoped_name, buf.getvalue())
                models_trained.append("file_recommender")
        except Exception:
            logger.warning(
                "Failed to train file_recommender model for tenant %s",
                tenant_id,
                exc_info=True,
            )

        return models_trained

    # ------------------------------------------------------------------
    # Batch training (WP03)
    # ------------------------------------------------------------------

    def train_all_tenants(self) -> TrainingBatch:
        """Discover and train all eligible tenants in batch.

        Each tenant is trained independently. Failures for individual
        tenants do not interrupt the batch (FR-007).

        Returns:
            TrainingBatch with per-tenant TrainingRun results.
        """
        start = time.time()
        tenants = self._discover_tenants()

        batch = TrainingBatch(
            started_at=datetime.now(timezone.utc),
        )

        for i, tenant_id in enumerate(tenants, 1):
            logger.info("Processing tenant %d/%d: %s", i, len(tenants), tenant_id)
            run = self._train_tenant_safe(tenant_id)
            batch.runs.append(run)

        batch.total_duration_ms = int((time.time() - start) * 1000)
        batch.completed_at = datetime.now(timezone.utc)

        logger.info(
            "Batch complete: %d trained, %d skipped, %d failed (of %d total) in %dms",
            batch.trained,
            batch.skipped,
            batch.failed,
            batch.total,
            batch.total_duration_ms,
        )

        # Record batch-level audit event (WP06)
        try:
            self.data_store.insert_ml_event(
                kind="batch_training",
                endpoint="cloud_trainer",
                routing="__batch__",
                latency_ms=batch.total_duration_ms,
            )
            self.data_store.commit()
        except Exception:
            logger.warning("Failed to record batch audit event", exc_info=True)

        return batch

    def _train_tenant_safe(self, tenant_id: str) -> TrainingRun:
        """Train a tenant with full error isolation.

        Catches all exceptions, logs them, and returns a failed TrainingRun
        instead of propagating the error.
        """
        try:
            return self.train_tenant(tenant_id)
        except Exception as e:
            logger.error(
                "Training failed for tenant %s: %s",
                tenant_id,
                str(e),
                exc_info=True,
            )
            error_msg = str(e)[:500] if len(str(e)) > 500 else str(e)
            return TrainingRun(
                tenant_id=tenant_id,
                status="failed",
                error=error_msg,
            )

    def _discover_tenants(self) -> list[str]:
        """Discover all tenants with synced data."""
        from sigil_ml.training.tenant_discovery import discover_eligible_tenants

        return discover_eligible_tenants(self.data_store)

    # ------------------------------------------------------------------
    # Aggregate training (WP05)
    # ------------------------------------------------------------------

    def train_aggregate(self) -> TrainingRun:
        """Train aggregate models from pooled opted-in tenant data.

        Returns a TrainingRun with tenant_id="__aggregate__".
        """
        start = time.time()
        started_at = datetime.now(timezone.utc)

        try:
            return self._train_aggregate_inner(start, started_at)
        except Exception as e:
            elapsed = time.time() - start
            logger.error("Aggregate training failed: %s", e, exc_info=True)
            run = TrainingRun(
                tenant_id=AGGREGATE_TENANT_ID,
                status="failed",
                error=str(e)[:500],
                duration_ms=int(elapsed * 1000),
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
            self._record_audit_event(AGGREGATE_TENANT_ID, run)
            return run

    def _train_aggregate_inner(self, start: float, started_at: datetime) -> TrainingRun:
        """Core aggregate training logic."""
        # 1. Discover opted-in tenants
        tenant_ids = self._discover_opted_in_tenants()

        # 2. Minimum opt-in threshold warning (T028)
        warning_msg: str | None = None
        if len(tenant_ids) < self.config.aggregate_min_tenants:
            warning_msg = (
                f"Only {len(tenant_ids)} opted-in tenants "
                f"(recommended minimum: {self.config.aggregate_min_tenants}). "
                f"Aggregate model may be unreliable."
            )
            logger.warning(warning_msg)

        if not tenant_ids:
            run = TrainingRun(
                tenant_id=AGGREGATE_TENANT_ID,
                status="skipped",
                error="No opted-in tenants found",
                duration_ms=int((time.time() - start) * 1000),
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
            self._record_audit_event(AGGREGATE_TENANT_ID, run)
            return run

        # 3. Pool data from all opted-in tenants
        all_tasks, task_events, tenant_counts = self._pool_training_data(tenant_ids)

        # 4. Apply sampling strategy
        sampled_tasks = self._sample_pooled_data(all_tasks, tenant_counts)
        total_samples = len(sampled_tasks)

        if total_samples == 0:
            run = TrainingRun(
                tenant_id=AGGREGATE_TENANT_ID,
                status="skipped",
                error="No tasks found across opted-in tenants",
                duration_ms=int((time.time() - start) * 1000),
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
            self._record_audit_event(AGGREGATE_TENANT_ID, run)
            return run

        # 5. Extract features and train all models
        models_trained = self._train_models_from_tasks(
            sampled_tasks,
            task_events,
            tenant_id=AGGREGATE_TENANT_ID,
        )

        # 6. Build result
        elapsed_ms = int((time.time() - start) * 1000)
        run = TrainingRun(
            tenant_id=AGGREGATE_TENANT_ID,
            status="trained",
            sample_count=total_samples,
            models_trained=models_trained,
            duration_ms=elapsed_ms,
            error=warning_msg,  # Non-fatal warning included in output
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )
        self._record_audit_event(AGGREGATE_TENANT_ID, run)
        return run

    def _discover_opted_in_tenants(self) -> list[str]:
        """Discover tenants opted in to aggregate data pooling."""
        from sigil_ml.training.tenant_discovery import discover_opted_in_tenants

        return discover_opted_in_tenants(self.data_store)

    def _pool_training_data(self, tenant_ids: list[str]) -> tuple[list[dict], dict[str, list[dict]], dict[str, int]]:
        """Pool training data from multiple opted-in tenants.

        Returns:
            (all_tasks, task_events, tenant_task_counts)
            - all_tasks: list of task dicts, each tagged with _tenant_id
            - task_events: dict mapping task_id -> list of event dicts
            - tenant_task_counts: dict mapping tenant_id -> task count
        """
        all_tasks: list[dict] = []
        task_events: dict[str, list[dict]] = {}
        tenant_counts: dict[str, int] = {}

        for tenant_id in tenant_ids:
            tasks = self._query_completed_tasks(tenant_id)
            tenant_counts[tenant_id] = len(tasks)

            for task in tasks:
                task["_tenant_id"] = tenant_id  # Tag with source tenant
                events = self._query_events_for_task(tenant_id, task["id"])
                all_tasks.append(task)
                task_events[task["id"]] = events

        logger.info(
            "Pooled %d total tasks from %d tenants (before sampling)",
            len(all_tasks),
            len(tenant_ids),
        )
        return all_tasks, task_events, tenant_counts

    def _sample_pooled_data(
        self,
        all_tasks: list[dict],
        tenant_counts: dict[str, int],
    ) -> list[dict]:
        """Apply per-tenant sampling caps to pooled data.

        Each tenant contributes at most max_tasks_per_tenant tasks.
        Sampling is deterministic (seeded RNG) for reproducibility.

        Returns:
            Sampled task list.
        """
        max_per = self.config.max_tasks_per_tenant
        rng = random.Random(42)  # deterministic for reproducibility

        sampled: list[dict] = []
        for tenant_id in tenant_counts:
            tenant_tasks = [t for t in all_tasks if t.get("_tenant_id") == tenant_id]

            if len(tenant_tasks) > max_per:
                logger.info(
                    "Sampling %d/%d tasks from tenant %s (cap: %d)",
                    max_per,
                    len(tenant_tasks),
                    tenant_id,
                    max_per,
                )
                tenant_tasks = rng.sample(tenant_tasks, max_per)

            sampled.extend(tenant_tasks)

        logger.info(
            "Aggregate dataset after sampling: %d tasks from %d tenants",
            len(sampled),
            len(tenant_counts),
        )
        return sampled

    # ------------------------------------------------------------------
    # Point-in-time retrieval (WP04)
    # ------------------------------------------------------------------

    def _build_entity_frame(self, tasks: list[dict], tenant_id: str) -> Any:
        """Build the entity frame: one row per example, with its own moment.

        The ``event_timestamp`` of each row is that example's reference time,
        resolved by the same rule the replay path uses. It is what the as-of join
        joins against, so an example carrying "now" here would retrieve the
        newest values in the store regardless of when it happened -- the leakage
        FR-008 forbids, introduced on the retrieval side rather than the write
        side.

        Tasks with no resolvable reference time are dropped, matching the replay
        path exactly; the count is logged once.

        Raises:
            OfflineStoreUnavailableError: If no example survives. A retrieval
                that would return an empty set is reported, never trained on.
        """
        import pandas as pd

        task_ids: list[str] = []
        timestamps: list[datetime] = []
        skipped = 0
        for task in tasks:
            as_of = _reference_time_for(task)
            if as_of is None:
                skipped += 1
                continue
            task_ids.append(task["id"])
            timestamps.append(datetime.fromtimestamp(as_of / 1000.0, tz=timezone.utc))

        if skipped:
            logger.info(
                "offline retrieval: skipped %d task(s) with no resolvable reference time (tenant %s)",
                skipped,
                tenant_id,
            )

        if not task_ids:
            raise OfflineStoreUnavailableError(
                f"No training example for tenant {tenant_id} has a resolvable reference time, so "
                "no entity frame can be built. Refusing to train on an empty retrieval (FR-017)."
            )

        return pd.DataFrame({"task_id": task_ids, "event_timestamp": timestamps})

    def _retrieve_offline_features(
        self,
        tasks: list[dict],
        tenant_id: str,
    ) -> dict[str, _RetrievedFeatures]:
        """Retrieve every model's training set point-in-time from the offline store.

        Tenant scoping is carried by ``tasks``, which the caller has already
        scoped -- per-tenant runs retrieve one tenant's task ids, the aggregate
        run retrieves the pooled set. The join key is the task id, so no query
        can span a tenant that was not asked for.

        Raises:
            OfflineStoreUnavailableError: The store, or a feature service in it,
                could not be reached.
            PartialRetrievalError: The store answered, but incompletely.
        """
        entity_df = self._build_entity_frame(tasks, tenant_id)
        retrieved: dict[str, _RetrievedFeatures] = {}
        records: list[FeatureSetRecord] = []

        for model_name, view in RETRIEVED_MODELS.items():
            retrieved[model_name] = self._retrieve_one(entity_df, model_name, view, tenant_id)
            records.append(retrieved[model_name].record)

        self.last_feature_sets = records
        for record in records:
            logger.info(
                "tenant %s: retrieved %d row(s) for %s from feature service %s@%s",
                tenant_id,
                record.rows,
                record.model,
                record.feature_service,
                record.version,
            )
            self._record_feature_set_audit(tenant_id, record)
        return retrieved

    def _retrieve_one(
        self,
        entity_df: Any,
        model_name: str,
        view: Any,
        tenant_id: str,
    ) -> _RetrievedFeatures:
        """Retrieve one model's feature set and verify it is complete."""
        try:
            service = self.feature_store.get_feature_service(model_name)
        except Exception as exc:  # noqa: BLE001 - normalized to the retrieval contract
            raise OfflineStoreUnavailableError(
                f"Feature service {model_name!r} could not be resolved from the registry: {exc}"
            ) from exc

        try:
            frame = self.feature_store.get_historical_features(
                entity_df=entity_df,
                features=service,
            ).to_df()
        except Exception as exc:  # noqa: BLE001 - normalized to the retrieval contract
            raise OfflineStoreUnavailableError(
                f"Point-in-time retrieval for {model_name!r} (tenant {tenant_id}) failed: {exc}. "
                "Training is abandoned rather than continued on whatever did come back (FR-017)."
            ) from exc

        expected = list(entity_df["task_id"])
        missing_columns = [name for name in view.feature_names if name not in frame.columns]
        if missing_columns:
            raise PartialRetrievalError(
                f"Retrieval for {model_name!r} omitted feature column(s) {missing_columns}. "
                f"The feature service names {list(view.feature_names)}; a vector built from a "
                "partial answer would be silently wrong in a fixed position."
            )

        if len(frame) != len(expected):
            raise PartialRetrievalError(
                f"Retrieval for {model_name!r} returned {len(frame)} row(s) for {len(expected)} "
                "example(s). A short training set trains successfully and is worse than an error."
            )

        # get_historical_features gives no row-order guarantee, so rows are
        # re-keyed rather than zipped positionally -- a silent reordering would
        # pair every vector with the wrong label.
        indexed = frame.set_index("task_id")
        if indexed.index.has_duplicates:
            raise PartialRetrievalError(
                f"Retrieval for {model_name!r} returned duplicate rows for the same task id, so "
                "no vector can be matched to its example unambiguously."
            )
        try:
            ordered = indexed.loc[expected]
        except KeyError as exc:
            raise PartialRetrievalError(
                f"Retrieval for {model_name!r} is missing at least one requested task id: {exc}."
            ) from exc

        vectors: list[list[float]] = []
        feature_dicts: list[dict[str, float]] = []
        for position, (task_id, row) in enumerate(zip(expected, ordered.to_dict("records"))):
            values: dict[str, float] = {}
            for name in view.feature_names:
                value = row[name]
                if value is None or value != value:  # NaN is the only value unequal to itself
                    raise PartialRetrievalError(
                        f"Retrieval for {model_name!r} returned no value for {name!r} on task "
                        f"{task_id!r} (row {position}). An empty as-of window means the offline "
                        "store holds nothing for that moment; training on the default would "
                        "quietly teach the model that the feature was zero."
                    )
                values[name] = float(value)
            feature_dicts.append(values)
            vectors.append([values[name] for name in view.feature_names])

        return _RetrievedFeatures(
            task_ids=list(expected),
            vectors=vectors,
            features=feature_dicts,
            record=FeatureSetRecord(
                model=model_name,
                feature_service=service.name,
                version=feature_service_version(service),
                feature_names=tuple(view.feature_names),
                rows=len(vectors),
            ),
        )

    def _record_feature_set_audit(self, tenant_id: str, record: FeatureSetRecord) -> None:
        """Write the consumed feature contract to ``ml_events`` (FR-010).

        Best-effort, like the other audit writes here: a training run that
        succeeded should not be reported as failed because its audit row could
        not be written.
        """
        try:
            self.data_store.insert_ml_event(
                kind="feature_retrieval",
                endpoint=f"feature_service:{record.feature_service}@{record.version}",
                routing=tenant_id,
                latency_ms=0,
            )
            self.data_store.commit()
        except Exception:
            logger.warning(
                "Failed to record feature set audit for tenant %s",
                tenant_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _save_model_to_store(self, model_name: str, model: Any, tenant_id: str) -> None:
        """Serialize a trained model and save via ModelStore with tenant prefix."""
        buf = io.BytesIO()
        joblib.dump(model.model, buf)
        # Save with tenant-scoped key: {tenant_id}/{model_name}
        scoped_name = f"{tenant_id}/{model_name}"
        self.model_store.save(scoped_name, buf.getvalue())

    def _record_audit_event(self, tenant_id: str, run: TrainingRun) -> None:
        """Record a training audit event via DataStore."""
        try:
            kind = "training"
            if tenant_id == AGGREGATE_TENANT_ID:
                kind = "aggregate_training"
            self.data_store.insert_ml_event(
                kind=kind,
                endpoint="cloud_trainer",
                routing=tenant_id,
                latency_ms=run.duration_ms,
            )
            self.data_store.commit()
        except Exception:
            logger.warning(
                "Failed to record training event for tenant %s",
                tenant_id,
                exc_info=True,
            )

    def _get_last_training_ts(self, tenant_id: str) -> float | None:
        """Get the last training timestamp for a tenant.

        Returns epoch milliseconds, or None if never trained.
        """
        return self.data_store.get_last_training_ts(tenant_id)

    def _query_completed_tasks(self, tenant_id: str) -> list[dict]:
        """Query completed tasks for a tenant via the DataStore protocol."""
        return self.data_store.get_completed_tasks_for_tenant(tenant_id)

    def _query_events_for_task(self, tenant_id: str, task_id: str) -> list[dict]:
        """Query events for a specific task via the DataStore protocol."""
        return self.data_store.get_events_for_task_id(task_id)
