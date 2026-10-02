"""Activity classifier for semantic event categorization."""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

from kenaz_ml.features import extract_activity_features
from kenaz_ml.modelstore import LocalModelStore, ModelStore
from kenaz_ml.modelstore.loader import resolve_for_serving  # not in the pinned package __all__

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kenaz_ml.modelstore.registry import FeatureContract, Resolution

logger = logging.getLogger(__name__)

#: The ordered input vector of the ML classifier -- the registered activity
#: contract (feature-vocabulary-refresh D-D6). Explicit and ordered, replacing
#: the implicit ``sorted(features.keys())`` derivation: a vocabulary change in
#: ``features.extract_activity_features`` is now a *visible* edit here (a test
#: pins the two together) and a versioned event through
#: :func:`activity_feature_contract`. Order is the vector layout; do not sort
#: it at the call site, do not compare it as a set.
ACTIVITY_FEATURE_NAMES: tuple[str, ...] = (
    "cmd_is_build",
    "cmd_is_git",
    "cmd_is_lint",
    "cmd_is_test",
    "exit_code_nonzero",
    "ext_code",
    "ext_config",
    "ext_docs",
    "has_branch",
    "has_cmd",
    "has_exit_code",
    "has_path",
    "kind_browser",
    "kind_file",
    "kind_hyprland",
    "kind_other",
    "kind_power",
    "kind_process",
    "kind_terminal",
)

#: The dtype every activity feature is declared with (registry convention).
ACTIVITY_FEATURE_DTYPE = "float64"


def activity_feature_contract() -> FeatureContract:
    """The registered, ordered contract for the activity classifier.

    Hand-authored (there is no Feast service for ``activity``), but versioned by
    the same recipe as the Feast-derived contracts --
    :func:`kenaz_ml.feature_store.materialize.versioned_contract_hash` with the
    ``activity`` service name, so the ``VOCABULARY_VERSION`` salt applies. The
    loader seam (``modelstore.loader._expected_contract``) returns this for
    ``activity`` instead of the empty unregistered contract.
    """
    from kenaz_ml.feature_store.materialize import versioned_contract_hash
    from kenaz_ml.modelstore.registry import FeatureContract

    references = [f"activity:{name}" for name in ACTIVITY_FEATURE_NAMES]
    return FeatureContract(
        service="activity",
        service_version=versioned_contract_hash("activity", references),
        names=ACTIVITY_FEATURE_NAMES,
        dtypes=(ACTIVITY_FEATURE_DTYPE,) * len(ACTIVITY_FEATURE_NAMES),
    )


CATEGORIES = [
    "editing",
    "verifying",
    "navigating",
    "researching",
    "integrating",
    "communicating",
    "idle",
]

# Full categories used after ML training splits editing into creating/refining.
CATEGORIES_FULL = [
    "creating",
    "refining",
    "verifying",
    "navigating",
    "researching",
    "integrating",
    "communicating",
    "idle",
]

# Terminal commands that indicate verifying activity.
_VERIFY_PREFIXES = (
    "go test",
    "go build",
    "go vet",
    "make",
    "cargo test",
    "cargo build",
    "npm test",
    "npm run test",
    "npm run build",
    "pytest",
    "python -m pytest",
    "python -m unittest",
    "./gradlew",
    "mvn test",
    "mvn build",
    "flake8",
    "pylint",
    "mypy",
    "ruff",
    "jest",
    "vitest",
    "mocha",
)

# Terminal commands that indicate integrating activity.
_INTEGRATE_PREFIXES = (
    "git commit",
    "git push",
    "git merge",
    "git rebase",
    "git tag",
    "gh pr",
)


class ActivityClassifier:
    """Classifies raw events into semantic activity categories.

    Starts rule-based on cold start, upgrades to ML (SGDClassifier)
    after sufficient training data (~500 events).

    Categories (cold start):
        editing     - file creation and modification (splits into creating/refining after ML)
        verifying   - test, build, lint commands
        navigating  - window/file/branch switches
        researching - AI queries, doc browsing
        integrating - commits, merges, pushes
        communicating - chat, review events (plugin-sourced)
        idle        - gaps between events
    """

    def __init__(self, model_store: ModelStore | None = None, *, registry: bool = False) -> None:
        self._store = model_store or LocalModelStore()
        self._ml_model: SGDClassifier | None = None
        self._trained = False

        self.resolution: Resolution | None = None
        self._load_activity(registry)

    def _load_activity(self, registry: bool) -> None:
        """Load the persisted activity classifier — through the registry when the store is a filesystem.

        two-client-engine-01MSK2EN WP01 (FR-001, FR-022): a filesystem-backed
        store resolves local slot -> base slot -> cold start with integrity,
        ordered-contract and runtime checks before deserialization, and a
        pre-registry artifact (no manifest) is migrated in place. The outcome is
        kept on ``self.resolution`` for ``/introspect`` and ``/health``. Any
        other store (S3, a test double) keeps the legacy byte-load path.
        Neither path raises: no usable artifact means untrained, as before.

        Opt-in (``registry=True``), set by the serving path
        (``AppState.load_models``). The trainers also construct predictors --
        inside the window where the stale manifest has been cleared and the new
        artifact not yet written -- and a constructor-time migration there would
        write a synthesized manifest for the *old* bytes, the mismatched pair the
        trainer's write ordering exists to prevent. Training therefore keeps the
        legacy load (whose result ``train()`` discards anyway).
        """
        resolution = resolve_for_serving(self._store, "activity") if registry else None
        if resolution is not None:
            self.resolution = resolution
            if resolution.served:
                if self._accept_width(resolution.model):
                    self._ml_model = resolution.model
                    self._trained = True
                    logger.info(
                        "Loaded activity classifier from %s (%s slot)", type(self._store).__name__, resolution.slot
                    )
                else:
                    # The guard refused what the registry served. Do not let /introspect keep claiming a
                    # local/base slot while the classifier answers from rules: report cold start + the refusal.
                    self.resolution = self._as_width_refused(resolution)
            return

        data = self._store.load("activity")
        if data is not None:
            try:
                model = joblib.load(io.BytesIO(data))
                if self._accept_width(model):
                    self._ml_model = model
                    self._trained = True
                    logger.info("Loaded activity classifier from %s", type(self._store).__name__)
            except Exception:
                logger.warning("Failed to load activity classifier, using rules")
                self._ml_model = None

    @staticmethod
    def _as_width_refused(resolution: Resolution) -> Resolution:
        """The resolution to report after the width guard refused a served artifact (cold start + why)."""
        from dataclasses import replace

        from kenaz_ml.modelstore.registry import SLOT_COLD_START, Refusal, SlotRefusal
        from kenaz_ml.modelstore.registry.slots import CHECK_SLOT

        width = getattr(resolution.model, "n_features_in_", None)
        refusal = SlotRefusal(
            resolution.slot,
            resolution.name,
            Refusal(
                CHECK_SLOT,
                "feature_width_mismatch",
                f"{resolution.name}: artifact was fitted on {width} input features but the current contract has "
                f"{len(ACTIVITY_FEATURE_NAMES)}; serving rules until retrained",
            ),
        )
        return replace(
            resolution,
            slot=SLOT_COLD_START,
            model=None,
            manifest=None,
            artifact=None,
            refusals=(*resolution.refusals, refusal),
        )

    @staticmethod
    def _accept_width(model: object) -> bool:
        """Refuse an artifact fitted on a different input width (rules fallback instead).

        The registry contract refuses a stale *manifested* artifact before it is
        deserialized; this catches the case a manifest cannot: a pre-registry
        artifact (no manifest) that the loader would stamp with the *current*
        contract. Such an artifact was fitted on the pre-refresh vector, and
        would otherwise fail on every ``predict``/``partial_fit``.
        """
        width = getattr(model, "n_features_in_", None)
        if width is None or width == len(ACTIVITY_FEATURE_NAMES):
            return True
        logger.warning(
            "activity classifier refused: it was fitted on %s input features but the current activity "
            "contract has %d; using rules until it is retrained",
            width,
            len(ACTIVITY_FEATURE_NAMES),
        )
        return False

    @classmethod
    def from_trained_model(cls, model: SGDClassifier, store: ModelStore | None = None) -> ActivityClassifier:
        """Create an instance from an already-trained sklearn model.

        Use this instead of ``__new__`` to avoid bypassing ``__init__``.
        """
        instance = object.__new__(cls)
        instance._store = store or LocalModelStore()
        instance._ml_model = model
        instance._trained = True
        return instance

    @property
    def is_trained(self) -> bool:
        return self._trained

    def classify(self, event: dict) -> dict:
        """Classify a single event into an activity category.

        Args:
            event: Dict with at least 'kind' key. Optionally 'payload', 'source'.

        Returns:
            {"category": str, "confidence": float, "method": "rules"|"ml"}
        """
        if self._trained and self._ml_model is not None:
            return self._classify_ml(event)
        return self._classify_rules(event)

    def classify_batch(self, events: list[dict]) -> list[dict]:
        """Classify a list of events.

        Args:
            events: List of event dicts.

        Returns:
            List of classification dicts, one per event.
        """
        return [self.classify(e) for e in events]

    def _classify_rules(self, event: dict) -> dict:
        """Deterministic rule-based classification from event kind + payload."""
        kind = event.get("kind", "")
        payload = event.get("payload") or {}
        if isinstance(payload, str):
            payload = {}

        # File events → editing (creating/refining split after ML training).
        if kind == "file":
            return {"category": "editing", "confidence": 0.8, "method": "rules"}

        # Terminal events → check command to distinguish verifying vs other.
        if kind == "terminal":
            cmd = str(payload.get("cmd", "")).strip().lower()

            # Check for verifying commands (test, build, lint).
            for prefix in _VERIFY_PREFIXES:
                if cmd.startswith(prefix):
                    return {"category": "verifying", "confidence": 0.9, "method": "rules"}

            # Check for integrating commands (git commit, push, merge).
            for prefix in _INTEGRATE_PREFIXES:
                if cmd.startswith(prefix):
                    return {"category": "integrating", "confidence": 0.9, "method": "rules"}

            # Other terminal commands → editing (likely running code).
            return {"category": "editing", "confidence": 0.6, "method": "rules"}

        # Git events → integrating.
        if kind == "git":
            return {"category": "integrating", "confidence": 0.8, "method": "rules"}

        # AI interaction events → researching.
        if kind == "ai":
            return {"category": "researching", "confidence": 0.85, "method": "rules"}

        # Window focus / compositor events → navigating.
        if kind == "hyprland":
            return {"category": "navigating", "confidence": 0.8, "method": "rules"}

        # Process events → navigating (app switching).
        if kind == "process":
            return {"category": "navigating", "confidence": 0.6, "method": "rules"}

        # Plugin-sourced events.
        source = event.get("source", "")
        if source in ("github", "jira", "slack"):
            return {"category": "communicating", "confidence": 0.7, "method": "rules"}

        # Unknown → idle.
        return {"category": "idle", "confidence": 0.5, "method": "rules"}

    def _classify_ml(self, event: dict) -> dict:
        """ML-based classification using trained SGDClassifier."""
        try:
            features = extract_activity_features(event)
            # Positional and strict: the registered contract, not sorted(keys).
            x = np.array([[features[f] for f in ACTIVITY_FEATURE_NAMES]])
            model = self._ml_model
            if model is None:
                raise RuntimeError("no trained activity model is loaded")
            category = model.predict(x)[0]
            proba = model.predict_proba(x)[0]
            confidence = float(max(proba))
        except Exception:
            logger.debug("ML classification failed, falling back to rules", exc_info=True)
            return self._classify_rules(event)

        return {"category": category, "confidence": round(confidence, 4), "method": "ml"}

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train or incrementally update the ML classifier.

        Uses SGDClassifier with partial_fit for incremental learning.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Category labels (strings from CATEGORIES or CATEGORIES_FULL).
        """
        if self._ml_model is None:
            self._ml_model = SGDClassifier(loss="log_loss", random_state=42)

        classes = np.array(CATEGORIES_FULL)
        self._ml_model.partial_fit(X, y, classes=classes)
        self._trained = True

        buf = io.BytesIO()
        joblib.dump(self._ml_model, buf)
        self._store.save("activity", buf.getvalue())
        logger.info("Saved activity classifier via %s", type(self._store).__name__)
