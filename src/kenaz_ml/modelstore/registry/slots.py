"""Two-slot model resolution: local, then base, then cold start (FR-003, FR-004).

The two slots are not siblings and are not interchangeable (D-001):

* The **local** slot is ``config.models_dir()`` — ``~/.local/share/sigild/ml-models``,
  user-writable because local training writes there. It keeps its existing flat
  ``{name}.joblib`` layout so pre-registry artifacts are still *found*; whether
  they are *used* is decided by the resolution rules below, and an artifact with
  no manifest beside it is not.
* The **base** slot is ``config.base_models_dir()`` — read in place inside the
  distribution, read-only, and covered by the same signing as the binary.

This module reads from both and writes to neither. That is not a convention it
maintains but something the module can be checked for: it contains no ``mkdir``,
no ``open`` for writing, and no call into ``write_manifest`` (FR-015, C-004).
``tests/test_registry_slots.py`` asserts that structurally, because a write path
added "just for tests" is still a write path in the shipped build.

Resolution order, per model, per data-model.md §Resolution::

    local artifact + manifest → validate → serve
    base  artifact + manifest → validate → serve
    otherwise                 → cold start, i.e. today's behavior

Every unusable condition falls through to the next step; none raises (FR-017).
Validation itself is WP01's — :func:`~kenaz_ml.modelstore.registry.manifest.validate_artifact`
runs integrity, then contract, then runtime — and this module adds no checks of
its own beyond "are both files actually there".

Two things about the logging are deliberate:

**A base-slot refusal is louder than a local-slot refusal.** A bad local model
is repaired by the next training run without anyone doing anything; a bad base
model is a broken release that the install has no way to fix, so it is
``ERROR`` where local is ``WARNING``.

**An empty slot is not a refusal worth reporting.** No base models have shipped
yet, so *every* install today resolves with an empty base slot. Logging that at
warning level would put a permanent, unactionable warning in front of every
user and train them to ignore the level that a real broken release needs. An
empty slot is recorded in :attr:`Resolution.refusals` for diagnostics and logged
at ``DEBUG``. A *half*-populated slot — artifact without manifest, or manifest
without artifact — is a different thing entirely and gets the full level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kenaz_ml import config
from kenaz_ml.modelstore.registry.manifest import (
    FeatureContract,
    Manifest,
    Refusal,
    deserialize_verified,
    read_manifest,
    validate_artifact,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ARTIFACT_SUFFIX",
    "CHECK_SLOT",
    "MANIFEST_SUFFIX",
    "REASON_ARTIFACT_NOT_FOUND",
    "REASON_SLOT_EMPTY",
    "SLOT_BASE",
    "SLOT_COLD_START",
    "SLOT_LOCAL",
    "Resolution",
    "SlotRefusal",
    "artifact_path",
    "base_slot_dir",
    "local_slot_dir",
    "manifest_path",
    "resolve_model",
    "slot_is_empty",
]

#: The slot that answered, as reported by :attr:`Resolution.slot`. ``cold_start``
#: is not a location — it means no slot answered and the caller should do what it
#: did before the registry existed.
SLOT_LOCAL = "local"
SLOT_BASE = "base"
SLOT_COLD_START = "cold_start"

ARTIFACT_SUFFIX = ".joblib"
MANIFEST_SUFFIX = ".json"

#: :attr:`Refusal.check` for the two conditions this module decides itself,
#: before any of WP01's checks are reached. Distinct from WP01's check names so
#: a caller can tell "the slot did not hold a usable pair" apart from "the pair
#: was there and failed validation".
CHECK_SLOT = "slot"

#: Neither artifact nor manifest present. The ordinary state of the base slot
#: today, and of the local slot before the first training run.
REASON_SLOT_EMPTY = "slot_empty"

#: Manifest present, artifact gone. Not an empty slot — something removed half
#: of a pair, and the remaining half names what is missing.
REASON_ARTIFACT_NOT_FOUND = "artifact_not_found"

#: Log level per slot for a genuine refusal. See the module docstring.
_REFUSAL_LEVEL = {SLOT_LOCAL: logging.WARNING, SLOT_BASE: logging.ERROR}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotRefusal:
    """One slot's reason for not answering, carried so callers can report it.

    The log line is emitted as the refusal happens, but resolution also returns
    every refusal it collected: ``/introspect`` and the diagnostics in later
    work packages need the reasons as data, not as text someone has to grep the
    log for.
    """

    slot: str
    model_name: str
    refusal: Refusal

    @property
    def check(self) -> str:
        """Which check refused — WP01's name, or :data:`CHECK_SLOT`."""
        return self.refusal.check

    @property
    def reason(self) -> str:
        """Stable machine-readable code. Safe to branch on."""
        return self.refusal.reason

    @property
    def detail(self) -> str:
        """Operator-actionable text naming the specific disagreement."""
        return self.refusal.detail

    def __str__(self) -> str:
        return f"{self.model_name} [{self.slot}] {self.refusal}"


@dataclass(frozen=True)
class Resolution:
    """Which slot answered for one model, and why the earlier ones did not.

    ``slot`` is always set. On cold start it is :data:`SLOT_COLD_START` and
    ``model`` is ``None`` — the caller then does exactly what it did before the
    registry existed, which for the predictors is to stay untrained and return
    their heuristic defaults.

    Branch on :attr:`served`, never on ``model``: a served model may be a falsy
    object.
    """

    name: str
    slot: str
    model: Any | None = None
    manifest: Manifest | None = None
    artifact: Path | None = None
    refusals: tuple[SlotRefusal, ...] = ()

    @property
    def served(self) -> bool:
        """True when a slot answered with a validated model."""
        return self.slot != SLOT_COLD_START

    @property
    def cold_start(self) -> bool:
        """True when no slot answered and pre-registry behavior applies."""
        return self.slot == SLOT_COLD_START

    @property
    def reasons(self) -> tuple[str, ...]:
        """Refusal reason codes in the order they were encountered."""
        return tuple(r.reason for r in self.refusals)


# ---------------------------------------------------------------------------
# T006 — slot locations
# ---------------------------------------------------------------------------


def base_slot_dir() -> Path:
    """Return the read-only base slot. Never created; may not exist."""
    return config.base_models_dir()


def local_slot_dir(tenant_id: str | None = None) -> Path:
    """Return the writable local slot, optionally scoped to a tenant.

    The base slot has no tenant equivalent: it ships with the product, so there
    is nothing tenant-specific to ship.
    """
    root = config.models_dir()
    return root / tenant_id if tenant_id else root


def artifact_path(slot_dir: Path | str, model_name: str) -> Path:
    """Return the artifact path for ``model_name`` within a slot."""
    return Path(slot_dir) / f"{model_name}{ARTIFACT_SUFFIX}"


def manifest_path(slot_dir: Path | str, model_name: str) -> Path:
    """Return the sidecar manifest path for ``model_name`` within a slot (D-004)."""
    return Path(slot_dir) / f"{model_name}{MANIFEST_SUFFIX}"


def slot_is_empty(slot_dir: Path | str, model_name: str) -> bool:
    """True when the slot holds neither half of the pair for ``model_name``.

    The ordinary, expected state — not an error, and deliberately distinguished
    from a slot holding one half of a pair.
    """
    return not artifact_path(slot_dir, model_name).exists() and not manifest_path(slot_dir, model_name).exists()


# ---------------------------------------------------------------------------
# T008 — refusal reporting
# ---------------------------------------------------------------------------


def _refusal_of(refusal: Refusal | None, fallback_detail: str) -> Refusal:
    """Return ``refusal``, or a stand-in if a WP01 result somehow carried none.

    WP01's result types guarantee a refusal whenever ``ok`` is false, so the
    fallback is unreachable. It exists because the alternative narrowing —
    ``assert`` — is stripped under ``python -O`` and would turn a broken
    invariant into an ``AttributeError`` on the prediction path, which is
    exactly the exception FR-017 forbids.
    """
    if refusal is not None:
        return refusal
    return Refusal(CHECK_SLOT, "unusable", fallback_detail)


def _record(slot: str, model_name: str, refusal: Refusal) -> SlotRefusal:
    """Log a refusal at the level its slot deserves and return it as data.

    Every line names the model, the slot, the check, the stable reason code and
    WP01's detail — enough to diagnose a broken release from the log alone,
    which is the bar T008 sets. A fallthrough that logged nothing would be as
    bad as the corruption it fell through on.
    """
    recorded = SlotRefusal(slot=slot, model_name=model_name, refusal=refusal)

    if refusal.reason == REASON_SLOT_EMPTY:
        level = logging.DEBUG
    else:
        level = _REFUSAL_LEVEL.get(slot, logging.WARNING)

    logger.log(
        level,
        "registry: %s slot cannot serve %r — %s/%s: %s",
        slot,
        model_name,
        refusal.check,
        refusal.reason,
        refusal.detail,
    )
    return recorded


# ---------------------------------------------------------------------------
# T007 — resolution
# ---------------------------------------------------------------------------


def _try_slot(
    model_name: str,
    slot: str,
    slot_dir: Path,
    *,
    expected_contract: FeatureContract | None,
    running_version: str | None,
) -> tuple[Resolution | None, SlotRefusal | None]:
    """Attempt one slot. Returns ``(resolution, refusal)``, exactly one of them set.

    Nothing here raises. Every path out that is not a validated, deserialized
    model is a refusal the caller falls through on.
    """
    artifact = artifact_path(slot_dir, model_name)
    manifest_file = manifest_path(slot_dir, model_name)

    # Nothing at all: the default state, reported quietly.
    if not artifact.exists() and not manifest_file.exists():
        return None, _record(
            slot,
            model_name,
            Refusal(CHECK_SLOT, REASON_SLOT_EMPTY, f"{slot_dir}: no {model_name} artifact or manifest present"),
        )

    # Manifest without artifact. Checked before reading the manifest so the
    # diagnostic names the missing half rather than whatever the manifest says.
    if not artifact.exists():
        return None, _record(
            slot,
            model_name,
            Refusal(
                CHECK_SLOT,
                REASON_ARTIFACT_NOT_FOUND,
                f"{manifest_file} describes an artifact that is not there: {artifact} is missing",
            ),
        )

    # Artifact without a usable manifest — missing, unreadable, corrupt or
    # structurally wrong. All four arrive here as one refusal from WP01. An
    # artifact predating the registry lands here too, and is refused for the
    # same reason: without a manifest there is no digest to verify against, and
    # FR-005 puts verification before deserialization unconditionally.
    read = read_manifest(manifest_file)
    if not read.ok or read.manifest is None:
        return None, _record(slot, model_name, _refusal_of(read.refusal, f"{manifest_file}: unusable manifest"))

    # WP01's three checks, in order, short-circuiting: integrity before any
    # deserialization, then the ordered feature contract, then the runtime.
    outcome = validate_artifact(
        read.manifest,
        artifact,
        expected_contract=expected_contract,
        running_version=running_version,
    )
    if not outcome.ok or outcome.verified is None:
        return None, _record(slot, model_name, _refusal_of(outcome.refusal, f"{artifact}: failed validation"))

    # The only route to a model object, and it takes the verification handle.
    loaded = deserialize_verified(outcome.verified)
    if not loaded.ok:
        return None, _record(slot, model_name, _refusal_of(loaded.refusal, f"{artifact}: could not be deserialized"))

    logger.info("registry: serving %r from the %s slot (%s)", model_name, slot, artifact)
    resolution = Resolution(name=model_name, slot=slot, model=loaded.model, manifest=read.manifest, artifact=artifact)
    return resolution, None


def resolve_model(
    model_name: str,
    *,
    local_dir: Path | str | None = None,
    base_dir: Path | str | None = None,
    expected_contract: FeatureContract | None = None,
    running_version: str | None = None,
) -> Resolution:
    """Resolve one model: local slot, then base slot, then cold start (FR-004).

    Args:
        model_name: The model's registry name — ``stuck``, ``duration``, and so
            on. Both the artifact and the manifest are named after it.
        local_dir: Overrides the local slot. Defaults to
            :func:`local_slot_dir`. Callers doing tenant resolution pass the
            tenant directory here and fall back to the shared one themselves.
        base_dir: Overrides the base slot, for tests. Defaults to
            :func:`base_slot_dir`. This is a *read* location in both cases —
            overriding it cannot turn the base slot into somewhere this module
            writes, because this module writes nowhere.
        expected_contract: The contract to validate against. ``None`` — the
            normal case — lets WP01 source it from the model's Feast feature
            service. Models with no registered service (``quality``,
            ``workflow``, ``activity``, the ``fleet_*`` family) therefore refuse
            with ``contract_unregistered`` and fall through to cold start, which
            is what they already do in practice: they have no artifact on disk
            to load either.
        running_version: The scikit-learn version to check against. ``None``
            uses the running one.

    Returns:
        A :class:`Resolution` naming the slot that answered, carrying every
        refusal collected on the way. Never raises: an unusable artifact is a
        fallthrough, not an exception on the prediction path (FR-017).
    """
    local_root = Path(local_dir) if local_dir is not None else local_slot_dir()
    base_root = Path(base_dir) if base_dir is not None else base_slot_dir()

    refusals: list[SlotRefusal] = []
    for slot, slot_dir in ((SLOT_LOCAL, local_root), (SLOT_BASE, base_root)):
        resolved, refusal = _try_slot(
            model_name,
            slot,
            slot_dir,
            expected_contract=expected_contract,
            running_version=running_version,
        )
        if resolved is not None:
            return Resolution(
                name=resolved.name,
                slot=resolved.slot,
                model=resolved.model,
                manifest=resolved.manifest,
                artifact=resolved.artifact,
                refusals=tuple(refusals),
            )
        if refusal is not None:
            refusals.append(refusal)

    logger.debug("registry: no usable artifact for %r in either slot; cold start", model_name)
    return Resolution(name=model_name, slot=SLOT_COLD_START, refusals=tuple(refusals))
