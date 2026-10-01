"""Model loading interface for pluggable storage backends.

Defines the ModelLoader protocol that storage backends must implement, and the
filesystem implementation that serving uses.

Loading goes through the registry (T018)
----------------------------------------
`FilesystemModelLoader.load()` no longer calls `joblib.load` itself. It resolves
a slot directory — tenant-specific first, shared second, exactly as before — and
hands that to `resolve_model()`, which runs integrity, then the *ordered* feature
contract, then runtime compatibility, and only then deserializes (FR-005,
FR-006, FR-008). There is no route from this module to a model object that skips
those checks.

**What did not change is the contract this module publishes**: `load()` returns
`None` when no usable model exists and never raises. A refused artifact is
indistinguishable, from the caller's side, from an absent one — the difference
is a log line, not an exception. Callers (`ModelCache`, the predictors) branch
on `None` and fall back to their heuristics; a raise here would turn a corrupt
file on disk into a failed prediction request, which is what FR-017 forbids.
Every path out of `load()` is therefore wrapped, including the paths through
code this module does not own.

Pre-registry artifacts are migrated, not discarded (T022)
---------------------------------------------------------
Before this mission an artifact was whatever `.joblib` happened to be on disk;
there were no manifests. Applying the new rules unchanged would refuse every
artifact that every existing install has already trained — and because the
retraining scheduler fires on accumulated *new* data rather than on a clock, the
gap would not reliably close. Existing installs would silently lose their
personalized models, possibly indefinitely.

So when the **local** slot holds an artifact with no manifest beside it, this
module treats it as pre-registry and synthesizes one: the digest computed from
the bytes actually on disk, `training_source="local"`, `n_local_extensions=0`,
no base recorded, runtime from the current environment, and the feature contract
read from the current Feast state.

Why that is not a hole in the integrity guarantee: the guarantee exists to catch
a *shipped base* artifact that was tampered with in transit or at rest, which is
why a base artifact without a manifest is still refused. A local artifact was
written by this install, into a directory this install owns and writes to on
every training run. Synthesizing a manifest for it grants an attacker who can
write there nothing they did not already have — they could equally have written
a matching manifest themselves. **The migration is deliberately confined to the
local slot; `resolve_model` handles the base slot and refuses it unchanged.**

One thing a synthesized manifest cannot recover is the runtime that *serialized*
the artifact — a pre-registry artifact records nothing, and reading it out of the
pickle would mean deserializing before verifying, which is the one ordering this
mission exists to prevent. So `runtime.sklearn_version` records the environment
doing the migration. That is self-consistent — the manifest validates in the
environment that wrote it — but it means that after a later scikit-learn upgrade
a migrated artifact is refused as `integrity/undeserializable` rather than with
the cleaner `runtime/sklearn_version_incompatible` diagnostic. Same refusal,
same fallback, worse message; the next training run replaces the manifest with a
truthful one.

Models with no registered feature service
-----------------------------------------
`quality`, `workflow`, `activity` and the `fleet_*` family declare no Feast
feature service, so there is no contract for them to record or be compared
against. `validate_feature_contract` fails closed when it is given nothing to
compare — correct as a default, but here it would refuse artifacts that have no
contract to disagree about and that load fine today.

This module therefore states the comparison explicitly: the contract expected of
a model with no registered service is :data:`UNREGISTERED_CONTRACT`, the empty
one, and a synthesized manifest records the same. They match, so those artifacts
keep serving. A manifest that *does* name a service while the install registers
none still refuses, on `service_mismatch` — the loosening covers "neither side
declares a contract", not "the two sides disagree".
"""

from __future__ import annotations

import hashlib
import logging
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kenaz_ml.modelstore.registry import FeatureContract, Manifest, Refusal
    from kenaz_ml.modelstore.registry.slots import Resolution

logger = logging.getLogger(__name__)

#: The sidecar pair's file extensions. Restated rather than imported from
#: `kenaz_ml.modelstore.registry`, because every import in this module is
#: deliberately deferred into the function that needs it: `kenaz_ml.modelstore`
#: is imported by `kenaz_ml.app` on every start, and the registry is only
#: reached on the far rarer path where a model is actually loaded.
ARTIFACT_SUFFIX = ".joblib"
MANIFEST_SUFFIX = ".json"

#: `Manifest.version` recorded for a migrated pre-registry artifact. Local
#: versions increment per extension and a pre-registry artifact has none
#: recorded, so it starts the count at zero rather than claiming an extension it
#: cannot evidence. The first training run after migration writes version "1".
PRE_REGISTRY_VERSION = "0"

#: Marks a synthesized manifest in `Manifest.extra`, so `/introspect` and the
#: refresh path can tell "this artifact's provenance was reconstructed" apart
#: from "this artifact was trained under the registry". Preserved across
#: read/write cycles by the manifest's unknown-key handling.
MIGRATION_MARKER = "synthesized_by"
MIGRATION_REASON = "pre_registry_migration"

#: Cache for :func:`_unregistered_contract`. Built lazily so importing this
#: module still costs nothing but the standard library (NFR-003).
_UNREGISTERED: FeatureContract | None = None


def _unregistered_contract() -> FeatureContract:
    """The contract validated against for a model with no Feast feature service.

    Empty on both sides: it matches a manifest that also records no contract,
    and refuses one that names a service. See the module docstring.
    """
    global _UNREGISTERED
    if _UNREGISTERED is None:
        from kenaz_ml.modelstore.registry import FeatureContract

        _UNREGISTERED = FeatureContract()
    return _UNREGISTERED


@runtime_checkable
class ModelLoader(Protocol):
    """Protocol for loading model objects from a storage backend.

    Implementations must:
    - Handle tenant-specific model resolution.
    - Return None when no model exists (not raise exceptions).
    - Be thread-safe (may be called concurrently).
    """

    def load(self, tenant_id: str, model_name: str) -> Any | None:
        """Load a model for the given tenant and model name.

        Args:
            tenant_id: Tenant identifier.
            model_name: One of "stuck", "suggest", "workflow",
                        "duration", "activity", "quality".

        Returns:
            The loaded model object, or None if not found.
        """
        ...


# ---------------------------------------------------------------------------
# Contract resolution
# ---------------------------------------------------------------------------


def _expected_contract(model_name: str) -> FeatureContract | None:
    """Return the contract ``model_name``'s artifacts are validated against.

    Four outcomes, and the difference between the last two matters:

    * A registered Feast feature service — its ordered contract. Identical to
      what `validate_feature_contract` would source itself.
    * ``activity`` — its hand-registered ordered contract
      (``kenaz_ml.models.activity.activity_feature_contract``).
    * No registered service — :func:`_unregistered_contract`, so an artifact that
      also records no contract is served rather than refused. See the module
      docstring.
    * Feast could not answer at all — ``None``. Not the same as "no service":
      the install cannot establish what the contract *is*, so nothing may be
      loosened and nothing may be migrated. Validation then refuses, which is
      the fail-closed direction.
    """
    from kenaz_ml.modelstore.registry import local_feature_contract

    if model_name == "activity":
        # Registered by hand, not Feast-derived (feature-vocabulary-refresh
        # D-D6): the ordered contract the classifier's vector is built from.
        from kenaz_ml.models.activity import activity_feature_contract

        return activity_feature_contract()

    try:
        contract = local_feature_contract(model_name)
    except Exception:
        logger.warning(
            "loader: cannot establish the local feature contract for %r; "
            "artifacts will be validated fail-closed until the feature store answers",
            model_name,
            exc_info=True,
        )
        return None
    return contract if contract is not None else _unregistered_contract()


# ---------------------------------------------------------------------------
# T022 — pre-registry migration
# ---------------------------------------------------------------------------


def _synthesize_manifest(artifact: Path, model_name: str, contract: FeatureContract) -> Manifest:
    """Build the manifest a pre-registry local artifact would have had.

    The digest is computed over the bytes on disk, so the manifest describes the
    artifact it will sit beside rather than asserting anything about it. The
    provenance is deliberately thin — an artifact from before the registry has
    no recorded base and no extension count, and inventing either would put a
    fabricated lineage into a file whose whole job is to be trustworthy.
    """
    from kenaz_ml.modelstore.registry import (
        Manifest,
        Provenance,
        Runtime,
        running_sklearn_version,
    )

    payload = artifact.read_bytes()
    try:
        created_at: int | None = int(artifact.stat().st_mtime * 1000)
    except OSError:
        created_at = None

    return Manifest(
        name=model_name,
        version=PRE_REGISTRY_VERSION,
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        created_at=created_at,
        provenance=Provenance(
            base_version=None,
            base_sha256=None,
            n_local_extensions=0,
            training_source="local",
        ),
        runtime=Runtime(
            estimator="",
            sklearn_version=running_sklearn_version() or "",
            python_version=platform.python_version(),
        ),
        feature_contract=contract,
        extra={MIGRATION_MARKER: MIGRATION_REASON},
    )


def _migrate_pre_registry(slot_dir: Path, model_name: str, contract: FeatureContract | None) -> Manifest | None:
    """Give a pre-registry local artifact a manifest, in place.

    Does nothing when there is nothing to migrate: no artifact, a manifest
    already there, something that is not a regular file at the artifact path, or
    no contract this install can vouch for.

    Returns:
        ``None`` in the ordinary case — including a successful migration, whose
        result is then read back off disk by :func:`resolve_model` like any
        other manifest. Returns the synthesized manifest **only** when it could
        not be persisted, so the caller can validate against it in memory rather
        than let an unwritable model directory un-train the install.
    """
    # KNOWN LIMITATION (dated 2026-09-30, Amendment A4 / two-client-engine
    # security review; no code change ruled): the migration stamps the artifact
    # with the *current* contract, because a manifest-less artifact records none.
    # In the contrived sequence "trainer clears the manifest -> process crashes
    # before the new manifest is written -> the model's feature contract changes
    # in an upgrade -> next start", an artifact trained under the OLD contract is
    # stamped with the NEW one and served with a mismatched vector layout until
    # the next training run rewrites a truthful manifest. Reaching it needs a
    # crash inside the trainer's write window and a contract change before the
    # next retrain; the refresh/registry path cannot tell the two apart without
    # deserializing first, which FR-005 forbids.
    from kenaz_ml.modelstore.registry import write_manifest

    artifact = slot_dir / f"{model_name}{ARTIFACT_SUFFIX}"
    manifest_file = slot_dir / f"{model_name}{MANIFEST_SUFFIX}"

    if contract is None or manifest_file.exists() or not artifact.is_file():
        return None

    try:
        manifest = _synthesize_manifest(artifact, model_name, contract)
    except OSError:
        # Unreadable artifact. Not a migration failure worth reporting here --
        # resolution reaches the same file and reports it as an integrity
        # refusal with the operating-system detail attached.
        logger.debug("loader: cannot read %s to migrate it", artifact, exc_info=True)
        return None

    try:
        write_manifest(manifest_file, manifest)
    except OSError:
        logger.warning(
            "loader: %r has no manifest and one could not be written to %s; "
            "validating against a synthesized manifest held in memory instead",
            model_name,
            manifest_file,
            exc_info=True,
        )
        return manifest

    logger.info(
        "loader: migrated pre-registry artifact %s — wrote synthesized manifest %s",
        artifact,
        manifest_file,
    )
    return None


def _validated_load(
    manifest: Manifest,
    artifact: Path,
    contract: FeatureContract | None,
) -> tuple[Any | None, Refusal | None]:
    """Validate then deserialize, against a manifest that is not on disk.

    The same three checks `resolve_model` runs, in the same order, reached
    through the same functions — integrity before deserialization, then the
    ordered contract, then the runtime. The only difference is where the
    manifest came from. Used solely for the unpersisted-migration case above.
    """
    from kenaz_ml.modelstore.registry import deserialize_verified, validate_artifact

    outcome = validate_artifact(manifest, artifact, expected_contract=contract)
    if not outcome.ok or outcome.verified is None:
        return None, outcome.refusal
    loaded = deserialize_verified(outcome.verified)
    if not loaded.ok:
        return None, loaded.refusal
    return loaded.model, None


class FilesystemModelLoader:
    """Loads model weights from the local filesystem, through registry validation.

    Directory layout (the local slot; the base slot is not tenant-scoped):
        {models_dir}/{tenant_id}/{model_name}.joblib  (tenant-specific)
        {models_dir}/{tenant_id}/{model_name}.json    (its manifest)
        {models_dir}/{model_name}.joblib              (shared fallback)
        {models_dir}/{model_name}.json                (its manifest)

    Tenant resolution is decided by *presence*, before validation, and is not
    revisited afterwards: a tenant that has its own artifact gets that artifact
    or nothing. Falling back to the shared model when a tenant's own one fails
    to validate would hand one tenant another's model, which is a worse outcome
    than serving no model at all.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize with the base directory for model weights.

        Args:
            base_dir: Root of the *local* slot. Defaults to config.models_dir().
                The base slot is resolved separately, by the registry, and is
                never written to from here.
        """
        if base_dir is None:
            from kenaz_ml import config

            base_dir = config.models_dir()
        self._base_dir = base_dir

    def load(self, tenant_id: str, model_name: str) -> Any | None:
        """Load a validated model from the filesystem.

        Tries the tenant-specific slot directory first, then the shared one,
        then — via the registry — the read-only base slot.

        Returns:
            The model object, or ``None`` when no slot holds a usable artifact.
            Never raises: a refused artifact is reported in the log and returned
            as an absence, because that is what the caller can act on (FR-017).
        """
        try:
            return self._load(tenant_id, model_name)
        except Exception:
            # The documented protocol is None-on-miss, and nothing reached from
            # here -- not validation, not the feature store, not the filesystem
            # -- is allowed to turn a bad artifact into a failed prediction.
            logger.warning(
                "loader: failed to load %s/%s",
                tenant_id,
                model_name,
                exc_info=True,
            )
            return None

    def _slot_dir(self, tenant_id: str, model_name: str) -> Path:
        """Return the local slot directory to resolve in: tenant's, else shared.

        Decided by presence of either half of the pair, so a tenant directory
        holding a manifest with no artifact is reported as the broken pair it is
        rather than quietly resolving to somebody else's model.
        """
        tenant_dir = self._base_dir / tenant_id
        if self._occupied(tenant_dir, model_name):
            return tenant_dir

        if self._occupied(self._base_dir, model_name):
            logger.info("loader: using shared model for %s/%s", tenant_id, model_name)
        return self._base_dir

    @staticmethod
    def _occupied(slot_dir: Path, model_name: str) -> bool:
        """True when ``slot_dir`` holds either half of ``model_name``'s pair."""
        return (slot_dir / f"{model_name}{ARTIFACT_SUFFIX}").exists() or (
            slot_dir / f"{model_name}{MANIFEST_SUFFIX}"
        ).exists()

    def _load(self, tenant_id: str, model_name: str) -> Any | None:
        resolution = self._resolve_in(self._slot_dir(tenant_id, model_name), tenant_id, model_name)
        return resolution.model if resolution.served else None

    def resolve_in(self, slot_dir: Path, model_name: str, *, label: str = "local") -> Resolution:
        """Resolve ``model_name`` in ``slot_dir`` and return the whole outcome.

        Added by two-client-engine-01MSK2EN WP01 (T004): local serving needs the
        :class:`~kenaz_ml.modelstore.registry.Resolution` — the slot that
        answered, its manifest, and every refusal collected on the way — so
        ``/introspect`` and ``/health`` can report provenance and refusal text.
        :meth:`load` discards it; this is the same path with the outcome kept.

        Same migration, same ``UNREGISTERED_CONTRACT`` handling, same base-slot
        fallback as :meth:`load`, and the same promise: never raises. An
        unexpected failure is reported as a cold-start resolution carrying a
        ``load_error`` refusal, never as an exception.
        """
        try:
            return self._resolve_in(slot_dir, label, model_name)
        except Exception as exc:
            logger.warning("loader: failed to load %s/%s", label, model_name, exc_info=True)
            return _error_resolution(model_name, exc)

    def _resolve_in(self, slot_dir: Path, tenant_id: str, model_name: str) -> Resolution:
        from kenaz_ml.modelstore.registry import REASON_SLOT_EMPTY, SLOT_COLD_START, SLOT_LOCAL, resolve_model
        from kenaz_ml.modelstore.registry.slots import Resolution, SlotRefusal

        contract = _expected_contract(model_name)

        unpersisted = _migrate_pre_registry(slot_dir, model_name, contract)
        if unpersisted is not None:
            artifact = slot_dir / f"{model_name}{ARTIFACT_SUFFIX}"
            model, refusal = _validated_load(unpersisted, artifact, contract)
            if refusal is None:
                logger.info("loader: loaded %s/%s from %s", tenant_id, model_name, artifact)
                return Resolution(
                    name=model_name, slot=SLOT_LOCAL, model=model, manifest=unpersisted, artifact=artifact
                )
            logger.warning(
                "loader: failed to load %s/%s from %s — %s",
                tenant_id,
                model_name,
                artifact,
                refusal,
            )
            return Resolution(
                name=model_name,
                slot=SLOT_COLD_START,
                refusals=(SlotRefusal(slot=SLOT_LOCAL, model_name=model_name, refusal=refusal),),
            )

        resolution = resolve_model(model_name, local_dir=slot_dir, expected_contract=contract)
        if resolution.served:
            logger.info(
                "loader: loaded %s/%s from %s (%s slot)",
                tenant_id,
                model_name,
                resolution.artifact,
                resolution.slot,
            )
            return resolution

        # Cold start. An empty slot is the ordinary state and is not worth a
        # warning; anything else is an artifact that exists and was refused, and
        # the operator needs the reason to be able to act on it.
        refused = [r for r in resolution.refusals if r.reason != REASON_SLOT_EMPTY]
        if refused:
            logger.warning(
                "loader: failed to load %s/%s — %s",
                tenant_id,
                model_name,
                "; ".join(str(r) for r in refused),
            )
        else:
            logger.debug("loader: no model found for %s/%s", tenant_id, model_name)
        return resolution


#: :attr:`Refusal.reason` for an unexpected exception on the load path. The
#: loader's contract turns every such exception into an absence; this code is
#: what that absence reports as when the caller asked for the outcome.
REASON_LOAD_ERROR = "load_error"


def _error_resolution(model_name: str, exc: BaseException) -> Resolution:
    """A cold-start :class:`Resolution` recording an unexpected load failure."""
    from kenaz_ml.modelstore.registry import SLOT_COLD_START, SLOT_LOCAL, Refusal
    from kenaz_ml.modelstore.registry.slots import Resolution, SlotRefusal

    refusal = Refusal("slot", REASON_LOAD_ERROR, f"{type(exc).__name__}: {exc}")
    return Resolution(
        name=model_name,
        slot=SLOT_COLD_START,
        refusals=(SlotRefusal(slot=SLOT_LOCAL, model_name=model_name, refusal=refusal),),
    )


# ---------------------------------------------------------------------------
# two-client-engine-01MSK2EN WP01 — the local serving seam
# ---------------------------------------------------------------------------


def filesystem_slot_dir(model_store: Any) -> Path | None:
    """Return the directory ``model_store`` persists artifacts in, if it is a filesystem.

    Mirrors ``training.trainer._artifact_dir``: a :class:`LocalModelStore`, or a
    :class:`CachedModelStore` fronting one, is filesystem-backed and so can hold
    sidecar manifests (D-004). Anything else — an ``S3ModelStore``, a test
    double — returns ``None`` and keeps the legacy byte-load path, because a
    manifest is a filesystem concept.
    """
    from kenaz_ml.modelstore.stores import CachedModelStore, LocalModelStore

    store: Any = model_store
    while isinstance(store, CachedModelStore):
        store = getattr(store, "_inner", None)
    if isinstance(store, LocalModelStore):
        base = getattr(store, "_base_dir", None)
        if base is None:
            from kenaz_ml import config

            base = config.models_dir()
        return Path(base)
    return None


def resolve_for_serving(model_store: Any, model_name: str) -> Resolution | None:
    """Resolve ``model_name`` for a local predictor through the registry seam.

    Returns ``None`` when ``model_store`` is not filesystem-backed — the caller
    then keeps its legacy ``store.load`` + ``joblib.load`` path. Otherwise the
    :class:`~kenaz_ml.modelstore.registry.Resolution` from the store's own
    directory (the local slot), then the base slot, then cold start — with the
    pre-registry migration and ``UNREGISTERED_CONTRACT`` handling applied
    exactly as :class:`FilesystemModelLoader` applies them. Never raises.
    """
    slot_dir = filesystem_slot_dir(model_store)
    if slot_dir is None:
        return None
    return FilesystemModelLoader(base_dir=slot_dir).resolve_in(slot_dir, model_name)
