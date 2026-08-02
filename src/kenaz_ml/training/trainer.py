"""Trainer: reads from the DataStore and retrains all models.

Three things this module now owes the registry, beyond fitting estimators.

**Vectors are built by strict lookup (FR-007).** ``[features[f] for f in names]``,
not ``features.get(f, 0.0)``. A silent zero for a missing feature is
indistinguishable from a genuine zero, so a renamed or dropped feature used to
train a model on a column of zeros with nothing raised anywhere. The strict form
is safe *only because* :func:`_training_contract` has already established that
the names being indexed are the ones the extractor produces — see the comments
at the two construction sites. A ``KeyError`` there is a defect to fix, not a
data condition to absorb.

**The examples are retained (FR-009).** Every vector a local run trains on is
appended to that model's retained set, stamped with the contract version it was
computed under and with its own ``as_of_ms``. Base refresh replays them; without
them, adopting a new base would throw away everything the install has learned.
Synthetic cold-start data is deliberately *not* retained — it is not this user's
behaviour and replaying it would teach the next model nothing.

**A manifest is written beside the artifact (FR-016), never into the base slot
(FR-015).** Local training writes under ``models_dir()`` only; the base slot is
read-only and is where a signed, shipped artifact lives.

Manifest and artifact are kept consistent by *ordering*, not by hope
------------------------------------------------------------------
A manifest whose ``artifact_sha256`` does not match the artifact beside it is
refused at load, which would silently drop the install back to the base model or
to cold start. Since the artifact write happens inside the model classes' own
``train()``, the two files cannot be replaced in a single operation, so the
stale manifest is **removed before** the artifact is written and the new one is
written after:

    read previous provenance → remove stale manifest → write artifact → write manifest

At every instant the local slot therefore holds either a matching pair or an
artifact with no manifest — never a mismatched pair. An artifact with no
manifest is exactly the pre-registry state that ``modelstore/loader.py``
migrates, so a run interrupted in that window costs the recorded provenance on
the next load, not the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from kenaz_ml.datastore import DataStore
from kenaz_ml.features import extract_duration_features, extract_stuck_features
from kenaz_ml.models.duration import FEATURE_NAMES as DURATION_FEATURES
from kenaz_ml.models.duration import DurationEstimator
from kenaz_ml.models.stuck import FEATURE_NAMES as STUCK_FEATURES
from kenaz_ml.models.stuck import StuckPredictor
from kenaz_ml.modelstore import ModelStore
from kenaz_ml.training.synthetic import generate_duration_data, generate_stuck_data

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kenaz_ml.modelstore.registry import AppendResult, FeatureContract, Provenance

logger = logging.getLogger(__name__)

ARTIFACT_SUFFIX = ".joblib"
MANIFEST_SUFFIX = ".json"

#: `provenance.training_source`, per data-model.md §Manifest. `base` is written
#: only by the cloud export job; local training writes the other two.
TRAINING_SOURCE_LOCAL = "local"
TRAINING_SOURCE_SYNTHETIC = "synthetic"


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


# ---------------------------------------------------------------------------
# T019 — the contract vectors are built against
# ---------------------------------------------------------------------------


def _training_contract(model_name: str, feature_names: Sequence[str]) -> FeatureContract | None:
    """Return the local feature contract to build ``model_name``'s vectors from.

    This is the validation that makes strict lookup safe. It answers one
    question — *are the names the feature service declares, in order, the names
    this code's extractor produces?* — and returns the contract only when they
    are. Two failure modes, both returning ``None``:

    * **No registered service, or the feature store cannot answer.** There is no
      contract, so there is nothing to validate against and nothing to stamp. The
      caller falls back to the module's own ``FEATURE_NAMES``, which is what it
      indexed before the registry existed. Vector construction is unaffected;
      what is lost is the manifest and the retained-set stamp, both of which
      would otherwise record a contract version this run cannot vouch for.
    * **The service disagrees with ``FEATURE_NAMES``.** A real defect: the
      extractor emits one set of keys and the registry declares another. Refusing
      to stamp is the fail-closed answer — a manifest recording the *declared*
      contract for vectors built from the *emitted* one would validate cleanly at
      load and be wrong.
    """
    from kenaz_ml.modelstore.registry import local_feature_contract

    try:
        contract = local_feature_contract(model_name)
    except Exception:
        logger.warning(
            "training: cannot read the feature contract for %r; "
            "no manifest or retained-set stamp will be written for this run",
            model_name,
            exc_info=True,
        )
        return None

    if contract is None:
        logger.debug("training: %r has no registered feature service; using FEATURE_NAMES", model_name)
        return None

    if tuple(contract.names) != tuple(feature_names):
        logger.error(
            "training: the %r feature service declares %s but this build extracts %s; "
            "refusing to stamp this run with a contract it did not use",
            model_name,
            list(contract.names),
            list(feature_names),
        )
        return None

    return contract


# ---------------------------------------------------------------------------
# T021 — where the artifact landed, and the manifest that must sit beside it
# ---------------------------------------------------------------------------


def _artifact_dir(model_store: ModelStore | None) -> Path | None:
    """Return the directory the model classes will write the artifact into.

    ``None`` when the artifact does not land on this filesystem at all — an
    ``S3ModelStore``, or a test double. A sidecar manifest is a filesystem
    concept (D-004); the cloud path records provenance through MLflow instead
    (C-002), so there is nothing to write here and nothing wrong with that.
    """
    from kenaz_ml import config
    from kenaz_ml.modelstore import CachedModelStore, LocalModelStore

    store: Any = model_store
    # A cache in front of a filesystem store still writes through to it.
    while isinstance(store, CachedModelStore):
        store = getattr(store, "_inner", None)

    if store is None:
        # What StuckPredictor/DurationEstimator construct when passed no store.
        return config.models_dir()
    if isinstance(store, LocalModelStore):
        return Path(getattr(store, "_base_dir", None) or config.models_dir())
    return None


def _is_base_slot(directory: Path) -> bool:
    """True when ``directory`` is, or is inside, the read-only base slot (FR-015).

    Checked rather than assumed. The base slot ships inside the distribution and
    is covered by the same signing as the binary; a training run that wrote
    there would replace a signed artifact with an unsigned one and destroy the
    only pristine copy the install has to fall back to.
    """
    from kenaz_ml import config

    try:
        base = config.base_models_dir().resolve()
        target = directory.resolve()
    except OSError:  # pragma: no cover - resolution of a path we just built
        return True
    return target == base or base in target.parents


def _next_provenance(directory: Path, model_name: str, training_source: str) -> Provenance:
    """Compute the provenance the *next* manifest for ``model_name`` should carry.

    Read before the stale manifest is removed, because it is the source of the
    extension count and of the base this artifact descends from.

    ``n_local_extensions`` counts local training runs on top of a base. It
    continues from the previous local manifest where there is one — including a
    manifest synthesized by the pre-registry migration, which records ``0``
    precisely because it cannot evidence any — and otherwise starts at ``1``,
    whether or not a base is being extended. The base identity is carried
    forward from the previous local manifest if it names one, and taken from the
    base slot's own manifest on the first extension.
    """
    from kenaz_ml import config
    from kenaz_ml.modelstore.registry import Provenance, read_manifest

    previous = read_manifest(directory / f"{model_name}{MANIFEST_SUFFIX}")
    if previous.ok and previous.manifest is not None:
        prior = previous.manifest.provenance
        return Provenance(
            base_version=prior.base_version,
            base_sha256=prior.base_sha256,
            n_local_extensions=prior.n_local_extensions + 1,
            training_source=training_source,
        )

    shipped = read_manifest(config.base_models_dir() / f"{model_name}{MANIFEST_SUFFIX}")
    if shipped.ok and shipped.manifest is not None:
        return Provenance(
            base_version=shipped.manifest.version,
            base_sha256=shipped.manifest.artifact_sha256,
            n_local_extensions=1,
            training_source=training_source,
        )

    return Provenance(n_local_extensions=1, training_source=training_source)


def _clear_manifest(directory: Path, model_name: str) -> None:
    """Remove the manifest describing the artifact about to be replaced.

    See the module docstring: this is the first half of keeping the pair
    consistent. Removing the stale manifest before the artifact is rewritten
    means the window between the two writes holds an artifact with no manifest —
    which the loader migrates — rather than a mismatched pair, which it refuses.
    """
    path = directory / f"{model_name}{MANIFEST_SUFFIX}"
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("training: could not remove the stale manifest %s", path, exc_info=True)


def _retain_examples(
    model_name: str,
    contract: FeatureContract | None,
    vectors: Sequence[Sequence[float]],
    labels: Sequence[float],
    as_of: Sequence[int],
) -> AppendResult | None:
    """Append this run's examples to ``model_name``'s retained set (FR-009).

    Best effort by design: retention exists so a future base refresh has
    something to replay, and failing to retain must not fail the training run
    that just succeeded. Every failure is logged and swallowed.

    Returns the append result, whose ``generation`` the manifest records, or
    ``None`` when nothing was retained.
    """
    if contract is None or not contract.service_version:
        logger.info(
            "training: no contract version for %r, so its examples cannot be stamped; not retaining",
            model_name,
        )
        return None

    from kenaz_ml.modelstore.registry import Example, append_examples

    examples = [
        Example(x=tuple(float(v) for v in x), y=float(y), as_of_ms=int(a))
        for x, y, a in zip(vectors, labels, as_of, strict=True)
    ]

    try:
        result = append_examples(model_name, examples, contract)
    except Exception:
        logger.warning("training: failed to retain examples for %r", model_name, exc_info=True)
        return None

    if not result.ok:
        logger.warning("training: retention refused for %r — %s", model_name, result.reason)
        return None

    logger.info(
        "training: retained %d example(s) for %r (generation %s, %d evicted)",
        result.written,
        model_name,
        result.generation,
        result.evicted,
    )
    return result


class _ManifestUpdate:
    """A training run in progress, and the manifest it will leave behind.

    Nothing is written until :meth:`record` is called and the surrounding block
    exits without an exception, so a run that fails part way leaves no manifest
    at all rather than one describing an artifact it never produced.
    """

    def __init__(self, directory: Path | None, model_name: str, contract: FeatureContract | None) -> None:
        self.directory = directory
        self.model_name = model_name
        self.contract = contract
        self.provenance: Provenance | None = None
        self._recorded = False
        self._n_samples = 0
        self._estimator = ""
        self._retained_generation: str | None = None
        self._as_of_ms: int | None = None
        self._metrics: dict[str, Any] = {}

    def record(
        self,
        *,
        n_samples: int,
        model: Any = None,
        retained_generation: str | None = None,
        as_of_ms: int | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Declare the run a success and supply what the manifest should say."""
        self._recorded = True
        self._n_samples = n_samples
        self._estimator = type(model).__name__ if model is not None else ""
        self._retained_generation = retained_generation
        self._as_of_ms = as_of_ms
        self._metrics = dict(metrics or {})

    def write(self) -> Path | None:
        """Write the manifest over the artifact that is now on disk."""
        from kenaz_ml.modelstore.registry import (
            Manifest,
            Runtime,
            Training,
            running_sklearn_version,
            write_manifest,
        )

        if not self._recorded or self.directory is None or self.provenance is None:
            return None

        if self.contract is None:
            # No contract this run can vouch for -- see _training_contract. The
            # tempting thing is to write the manifest anyway with the contract
            # left empty, and it is exactly wrong: the loader validates against
            # the contract the *install* declares, so an empty one recorded for
            # a model that has a registered service refuses on the next load and
            # throws away the model this run just trained. Writing nothing
            # leaves the artifact in the pre-registry state, which the loader
            # migrates -- reconstructing a manifest that is true rather than
            # preserving one that is not.
            logger.warning(
                "training: no contract to stamp %r with, so no manifest written; "
                "the artifact will be migrated on the next load",
                self.model_name,
            )
            return None

        artifact = self.directory / f"{self.model_name}{ARTIFACT_SUFFIX}"
        if not artifact.is_file():
            # train() is monkeypatched, or the store wrote somewhere else. A
            # manifest with no artifact is a broken pair; write nothing.
            logger.debug("training: no artifact at %s, so no manifest written", artifact)
            return None

        # The digest is taken over the bytes actually on disk rather than over
        # whatever was serialized in memory, so the manifest describes the file
        # it will sit beside even if the store transformed it on the way out.
        try:
            payload = artifact.read_bytes()
        except OSError:
            logger.warning("training: could not read %s to record its digest", artifact, exc_info=True)
            return None

        manifest = Manifest(
            name=self.model_name,
            version=str(self.provenance.n_local_extensions),
            artifact_sha256=hashlib.sha256(payload).hexdigest(),
            created_at=int(time.time() * 1000),
            provenance=self.provenance,
            runtime=Runtime(
                estimator=self._estimator,
                sklearn_version=running_sklearn_version() or "",
                python_version=platform.python_version(),
            ),
            feature_contract=self.contract,
            training=Training(
                n_samples=self._n_samples,
                retained_generation=self._retained_generation,
                as_of_ms=self._as_of_ms,
            ),
            metrics=self._metrics,
        )

        path = self.directory / f"{self.model_name}{MANIFEST_SUFFIX}"
        try:
            write_manifest(path, manifest)
        except OSError:
            logger.warning("training: could not write the manifest %s", path, exc_info=True)
            return None

        logger.info(
            "training: wrote manifest %s (version %s, %s, %d sample(s))",
            path,
            manifest.version,
            self.provenance.training_source,
            self._n_samples,
        )
        return path


class Trainer:
    """Orchestrates training of all kenaz-ml models from local data."""

    def __init__(self, store: DataStore, model_store: ModelStore | None = None) -> None:
        self.store = store
        self._model_store = model_store

    @contextmanager
    def _manifest_update(
        self,
        model_name: str,
        contract: FeatureContract | None,
        training_source: str,
    ) -> Iterator[_ManifestUpdate]:
        """Wrap a training run so it leaves a manifest matching what it wrote.

        Reads the provenance to carry forward, clears the stale manifest, yields
        to the caller to fit and save the model, and writes the new manifest on a
        clean exit. See the module docstring for why the order is this way.

        The whole mechanism is skipped, with a log line, when the artifact does
        not land on a filesystem this process owns — and refused outright if the
        directory resolves into the base slot (FR-015).
        """
        update = _ManifestUpdate(_artifact_dir(self._model_store), model_name, contract)

        if update.directory is None:
            logger.debug("training: %r is not stored on the local filesystem; no manifest written", model_name)
        elif _is_base_slot(update.directory):
            # Not reachable through any supported configuration; asserted here
            # so that it stays unreachable rather than becoming a silent write
            # into a signed, read-only slot.
            logger.error(
                "training: refusing to write %r into the base slot %s — the base slot is read-only (FR-015)",
                model_name,
                update.directory,
            )
            update.directory = None
        else:
            update.provenance = _next_provenance(update.directory, model_name, training_source)
            _clear_manifest(update.directory, model_name)

        yield update
        update.write()

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
        contract = _training_contract("stuck", STUCK_FEATURES)
        names = tuple(contract.names) if contract is not None else tuple(STUCK_FEATURES)

        task_ids = self.store.get_completed_task_ids()

        X_list = []
        y_list = []
        as_of_list = []
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
            # Strict lookup (FR-007), safe only because _training_contract has
            # already established that `names` is what this extractor emits. A
            # KeyError here means that validation was skipped -- a defect, not a
            # data condition, and not something to paper over by restoring
            # `.get(f, 0.0)`: a defaulted zero is indistinguishable from a real
            # one and would train the model on a column of silence.
            x = [features[f] for f in names]
            X_list.append(x)
            as_of_list.append(as_of)
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
            with self._manifest_update("stuck", contract, TRAINING_SOURCE_SYNTHETIC) as update:
                predictor = StuckPredictor(model_store=self._model_store)
                predictor.train(X, y)
                # Synthetic examples are not retained: they are not this user's
                # behaviour, so replaying them during a base refresh would
                # personalize the new base towards nobody.
                update.record(n_samples=500, model=predictor.model)
            return 500

        X = np.array(X_list)
        y = np.array(y_list)

        with self._manifest_update("stuck", contract, TRAINING_SOURCE_LOCAL) as update:
            predictor = StuckPredictor(model_store=self._model_store)
            predictor.train(X, y)
            # Retention happens only once the fit and save have succeeded, so a
            # failed run leaves the retained set exactly as it found it.
            retained = _retain_examples("stuck", contract, X_list, y_list, as_of_list)
            update.record(
                n_samples=len(X),
                model=predictor.model,
                retained_generation=retained.generation if retained else None,
                as_of_ms=max(as_of_list),
            )
        return len(X)

    def _train_duration(self) -> int:
        """Train the duration estimator from completed tasks.

        Returns:
            Number of samples used, or 0 if insufficient data.
        """
        contract = _training_contract("duration", DURATION_FEATURES)
        names = tuple(contract.names) if contract is not None else tuple(DURATION_FEATURES)

        rows = self.store.get_completed_tasks_with_timestamps()

        X_list = []
        y_list = []
        as_of_list = []
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
            # Strict lookup (FR-007) -- see the note in _train_stuck. Validation
            # ran first; a KeyError here is a defect, not a data condition.
            x = [features[f] for f in names]
            X_list.append(x)
            as_of_list.append(as_of)
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
            with self._manifest_update("duration", contract, TRAINING_SOURCE_SYNTHETIC) as update:
                estimator = DurationEstimator(model_store=self._model_store)
                estimator.train(X, y)
                update.record(n_samples=500, model=estimator.model)
            return 500

        X = np.array(X_list)
        y = np.array(y_list)

        with self._manifest_update("duration", contract, TRAINING_SOURCE_LOCAL) as update:
            estimator = DurationEstimator(model_store=self._model_store)
            estimator.train(X, y)
            retained = _retain_examples("duration", contract, X_list, y_list, as_of_list)
            update.record(
                n_samples=len(X),
                model=estimator.model,
                retained_generation=retained.generation if retained else None,
                as_of_ms=max(as_of_list),
            )
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

        from kenaz_ml.signals.pattern_detector import PatternDetector

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

        from kenaz_ml.features import extract_action_token
        from kenaz_ml.signals.next_action import NextActionPredictor

        predictor = NextActionPredictor()
        predictor.reset()  # Start fresh for full rebuild
        total_tokens = 0

        # Create classifier once before the loop
        from kenaz_ml.models.activity import ActivityClassifier

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
        from kenaz_ml.signals.file_recommender import FileRecommender

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
