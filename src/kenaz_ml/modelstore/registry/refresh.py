"""Base refresh — replay local adaptation onto a newly shipped base model.

A user who has been running for months has a local model shaped by how *they*
work. An upgrade ships a new base. The policy this module implements is
**re-extend**: the training examples behind that adaptation were retained
(WP03), so they are replayed onto the new base rather than thrown away.

Three outcomes, decided in this order:

======================  ===========================================================
``rebuild``             The retained set's ``contract_version`` matches the shipped
                        base's. Full retraining on the retained examples produces
                        a new local artifact descended from the new base
                        (FR-013, D-003).
``reset``               The contract versions differ. The retained vectors were
                        computed under a contract the new base does not speak, so
                        they are discarded, the generation is incremented, and the
                        shipped base is served unextended with
                        ``provenance.reset_reason = "contract_version_changed"``
                        (FR-014).
``adopt_base``          Contract matches but nothing is retained. The new base is
                        served directly, which is a success and not an error
                        (User Story 3, scenario 3).
======================  ===========================================================

Detection carries no timestamps
-------------------------------
FR-012 requires the change be detectable *from manifest provenance alone*.
:func:`detect_base_change` compares two pairs of strings and nothing else:

* the base manifest's ``version`` against the local manifest's
  ``provenance.base_version``, and
* the base manifest's ``artifact_sha256`` against the local manifest's
  ``provenance.base_sha256``.

The digest comparison is not redundant. A version number is a human-assigned
label and can be reused by mistake — a rebuilt "v3" that is not the v3 already
descended from is exactly the case a version comparison cannot see, and exactly
the case that silently serves a model whose provenance is a lie. The digest can
not be reused by mistake.

No mtime is read anywhere in this module, and ``ModelStore.last_modified()`` is
not called. Its own docstring concedes it is a proxy, and D-007 rejected it:
file times are rewritten by package installs, archive extraction, and file
copies, so an mtime-triggered refresh fires on upgrades that changed no model
and stays silent on ones that did. Version-and-digest comparison is exact.
``tests/test_registry_refresh.py`` asserts the absence structurally, by walking
this module's AST, because a timestamp added later would still pass every
behavioural test written today.

Rebuild is full retraining
--------------------------
:func:`_full_retrain` clones the new base estimator — hyperparameters only, no
fitted state — and fits it on the retained examples. There is deliberately no
``warm_start`` and no ``partial_fit``: C-006 defers warm-start extension to a
later mission, several estimators on the roster support neither, and full
retraining at these data volumes takes seconds. A later mission can optimize
this path without changing its inputs, its outputs, or the manifest it writes.

Atomicity is the point of the write path (FR-019)
-------------------------------------------------
An artifact and its manifest that disagree are worse than no refresh at all:
the manifest's ``artifact_sha256`` would not match the artifact beside it, WP01
would refuse the pair on the next load, and the install would silently drop to
the base model having lost its personalization. So both files are written to
temporary paths in the destination directory first, and moved into place only
once both are complete. If the second move fails, the previous pair is restored
byte-for-byte before this function returns.

Nothing here raises to its caller. A refresh that cannot proceed returns a
:class:`~kenaz_ml.modelstore.registry.manifest.Refusal` as data and leaves the
previously-served model exactly where it was (FR-017, FR-019).
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kenaz_ml.modelstore.registry.manifest import (
    SCHEMA_VERSION,
    FeatureContract,
    Manifest,
    Provenance,
    Refusal,
    Runtime,
    Training,
    deserialize_verified,
    manifest_to_dict,
    read_manifest,
    running_sklearn_version,
    validate_artifact,
    write_manifest,
)
from kenaz_ml.modelstore.registry.retained import (
    RetainedSet,
    read_retained,
    reset_retained,
)
from kenaz_ml.modelstore.registry.slots import (
    artifact_path,
    base_slot_dir,
    local_slot_dir,
    manifest_path,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ACTION_ADOPT_BASE",
    "ACTION_NONE",
    "ACTION_REBUILD",
    "ACTION_RESET",
    "CHECK_REFRESH",
    "REASON_BASE_DIGEST_CHANGED",
    "REASON_BASE_MANIFEST_UNUSABLE",
    "REASON_BASE_VERSION_CHANGED",
    "REASON_LOCAL_MANIFEST_UNUSABLE",
    "REASON_NO_BASE",
    "REASON_NO_LOCAL",
    "REASON_UP_TO_DATE",
    "RESET_REASON_CONTRACT_CHANGED",
    "TRAINING_SOURCE_BASE",
    "TRAINING_SOURCE_LOCAL",
    "BaseChange",
    "RefreshResult",
    "TrainFn",
    "describe",
    "detect_base_change",
    "refresh_all",
    "refresh_model",
]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: What a refresh did, as reported by :attr:`RefreshResult.action`.
ACTION_NONE = "none"
ACTION_REBUILD = "rebuild"
ACTION_RESET = "reset"
ACTION_ADOPT_BASE = "adopt_base"

#: Why detection decided as it did — stable codes, safe to branch on.
REASON_UP_TO_DATE = "up_to_date"
REASON_NO_BASE = "no_base"
REASON_NO_LOCAL = "no_local"
REASON_BASE_MANIFEST_UNUSABLE = "base_manifest_unusable"
REASON_LOCAL_MANIFEST_UNUSABLE = "local_manifest_unusable"
REASON_BASE_VERSION_CHANGED = "base_version_changed"
REASON_BASE_DIGEST_CHANGED = "base_digest_changed"

#: :attr:`Refusal.check` for the conditions this module decides itself.
CHECK_REFRESH = "refresh"

#: The value written to ``provenance.reset_reason`` by the reset path. This
#: string is the discoverable record of a lost personalization (SC-004) — it is
#: what an operator greps for, so it is a constant and not a formatted message.
RESET_REASON_CONTRACT_CHANGED = "contract_version_changed"

#: ``provenance.training_source`` values, per data-model.md §Manifest.
TRAINING_SOURCE_LOCAL = "local"
TRAINING_SOURCE_BASE = "base"

#: Suffix for the staging copies written before either file is moved into
#: place. Leading dot so a half-finished refresh is never mistaken for a slot
#: entry — :func:`~kenaz_ml.modelstore.registry.slots.resolve_model` looks for
#: exactly ``{name}.joblib``/``{name}.json`` and will not see these.
_STAGING_SUFFIX = ".refresh-tmp"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseChange:
    """Whether the shipped base differs from the one the local model descended from.

    Pure comparison of manifest fields. Reading this answers FR-012 on its own:
    every field it branched on is here, and not one of them is a time.
    """

    name: str
    due: bool
    reason: str
    detail: str
    local_base_version: str | None = None
    base_version: str | None = None
    local_base_sha256: str | None = None
    base_sha256: str | None = None
    local_manifest: Manifest | None = None
    base_manifest: Manifest | None = None

    def __str__(self) -> str:
        return f"{self.name}: {'refresh due' if self.due else 'no refresh'} ({self.reason}) — {self.detail}"


@dataclass(frozen=True)
class RefreshResult:
    """What a refresh did, or why it did nothing.

    ``ok`` is false only for a genuine failure. A refresh that was not due, and
    a refresh that adopted the new base because nothing was retained, are both
    successes — branch on :attr:`changed` to tell "something was written" from
    "nothing needed writing".
    """

    name: str
    action: str
    change: BaseChange
    ok: bool = True
    refusal: Refusal | None = None
    manifest: Manifest | None = None
    artifact: Path | None = None
    n_samples: int = 0
    retained_generation: str | None = None
    previous_generation: str | None = None
    reset_reason: str | None = None

    @property
    def changed(self) -> bool:
        """True when this refresh wrote a new local artifact and manifest."""
        return self.ok and self.action != ACTION_NONE

    def __str__(self) -> str:
        if not self.ok and self.refusal is not None:
            return f"{self.name}: refresh refused — {self.refusal}"
        return f"{self.name}: {self.action}"


# ---------------------------------------------------------------------------
# T014 — detection. Versions and digests only; no timestamp, ever.
# ---------------------------------------------------------------------------


def detect_base_change(
    name: str,
    *,
    local_dir: Path | str | None = None,
    base_dir: Path | str | None = None,
) -> BaseChange:
    """Decide whether the shipped base differs from the local model's ancestor.

    Evaluated at startup and on demand, never on a schedule: nothing about the
    base slot changes between upgrades, and an upgrade implies a restart (D-007).

    The three absence cases are all "nothing to do", and are distinguished only
    so the reason code says which one happened:

    * **No base present.** The state of every install in the field today — no
      base models have ever been built. Reported ``no_base``.
    * **No local present.** The base serves directly through ordinary
      resolution; there is no local model to refresh. Reported ``no_local``.
    * **Neither present.** Falls out of the first check as ``no_base``.

    Args:
        name: The model's registry name.
        local_dir: Overrides the local slot; defaults to
            :func:`~kenaz_ml.modelstore.registry.slots.local_slot_dir`.
        base_dir: Overrides the base slot; defaults to
            :func:`~kenaz_ml.modelstore.registry.slots.base_slot_dir`.

    Returns:
        A :class:`BaseChange`. Never raises: an unreadable manifest on either
        side is a reason not to refresh, not an exception on the startup path.
    """
    local_root = Path(local_dir) if local_dir is not None else local_slot_dir()
    base_root = Path(base_dir) if base_dir is not None else base_slot_dir()

    base_manifest_file = manifest_path(base_root, name)
    base_artifact = artifact_path(base_root, name)

    if not base_manifest_file.exists() or not base_artifact.exists():
        return BaseChange(
            name=name,
            due=False,
            reason=REASON_NO_BASE,
            detail=(
                f"{base_root}: no shipped base pair for {name!r}; nothing to refresh from "
                "(the ordinary state of an install with no base models)"
            ),
        )

    base_read = read_manifest(base_manifest_file)
    if not base_read.ok or base_read.manifest is None:
        detail = base_read.refusal.detail if base_read.refusal else f"{base_manifest_file}: unusable manifest"
        return BaseChange(
            name=name,
            due=False,
            reason=REASON_BASE_MANIFEST_UNUSABLE,
            detail=f"{name}: the shipped base manifest cannot be read, so no refresh can be decided — {detail}",
        )
    base_manifest = base_read.manifest

    local_manifest_file = manifest_path(local_root, name)
    local_artifact = artifact_path(local_root, name)
    if not local_manifest_file.exists() or not local_artifact.exists():
        return BaseChange(
            name=name,
            due=False,
            reason=REASON_NO_LOCAL,
            detail=(
                f"{local_root}: no local pair for {name!r}; the base slot serves it directly "
                "and there is no local model to refresh"
            ),
            base_version=base_manifest.version,
            base_sha256=base_manifest.artifact_sha256,
            base_manifest=base_manifest,
        )

    local_read = read_manifest(local_manifest_file)
    if not local_read.ok or local_read.manifest is None:
        detail = local_read.refusal.detail if local_read.refusal else f"{local_manifest_file}: unusable manifest"
        return BaseChange(
            name=name,
            due=False,
            reason=REASON_LOCAL_MANIFEST_UNUSABLE,
            detail=(
                f"{name}: the local manifest cannot be read, so its ancestry is unknown and no refresh "
                f"can be decided; resolution will fall through to the base slot — {detail}"
            ),
            base_version=base_manifest.version,
            base_sha256=base_manifest.artifact_sha256,
            base_manifest=base_manifest,
        )
    local_manifest = local_read.manifest

    recorded_version = local_manifest.provenance.base_version
    recorded_digest = (local_manifest.provenance.base_sha256 or "").strip().lower() or None
    shipped_version = base_manifest.version
    shipped_digest = (base_manifest.artifact_sha256 or "").strip().lower() or None

    common = {
        "name": name,
        "local_base_version": recorded_version,
        "base_version": shipped_version,
        "local_base_sha256": recorded_digest,
        "base_sha256": shipped_digest,
        "local_manifest": local_manifest,
        "base_manifest": base_manifest,
    }

    if recorded_version != shipped_version:
        return BaseChange(
            due=True,
            reason=REASON_BASE_VERSION_CHANGED,
            detail=(
                f"{name}: the local model descends from base version {recorded_version!r} but the shipped base "
                f"is version {shipped_version!r}"
            ),
            **common,
        )

    # Same version number. The digest is what distinguishes "the same base" from
    # "a different base that reused the number" — the failure a version
    # comparison alone cannot see.
    if recorded_digest != shipped_digest:
        return BaseChange(
            due=True,
            reason=REASON_BASE_DIGEST_CHANGED,
            detail=(
                f"{name}: the shipped base is still labelled version {shipped_version!r} but its artifact digest "
                f"changed — the local model descends from {recorded_digest}, the shipped base is {shipped_digest}; "
                "a reused version number does not make it the same base"
            ),
            **common,
        )

    return BaseChange(
        due=False,
        reason=REASON_UP_TO_DATE,
        detail=(
            f"{name}: the local model already descends from base version {shipped_version!r} "
            f"(digest {shipped_digest}); no refresh due"
        ),
        **common,
    )


# ---------------------------------------------------------------------------
# T015 — full retraining on the retained set
# ---------------------------------------------------------------------------


def _full_retrain(base_model: Any, x: Any, y: Any) -> Any:
    """Fit a fresh copy of the base estimator on the retained examples (D-003).

    ``clone`` copies hyperparameters and *discards fitted state*, so what comes
    back is the new base's configuration trained entirely on this install's own
    examples. That is the whole of the rebuild: no ``warm_start``, no
    ``partial_fit``, no incremental extension. C-006 defers those, and reaching
    for one here would change what this package can be verified to do — a
    warm-started model's predictions are a blend nobody has specified, whereas a
    full retrain's are a function of the retained set and the base's
    hyperparameters alone.
    """
    from sklearn.base import clone

    fresh = clone(base_model)
    fresh.fit(x, y)
    return fresh


#: The training callable's shape: ``(base_model, X, y) -> fitted model``.
TrainFn = Callable[[Any, Any, Any], Any]


def _matrix(retained: RetainedSet, width: int) -> tuple[Any, Any] | Refusal:
    """Build ``(X, y)`` from the retained examples, or refuse if they do not fit.

    Every vector must be the width the contract declares. A short or long row
    means the file and the contract disagree about the world despite agreeing on
    the contract version, which is a condition to refuse on rather than pad —
    padding is exactly the silent-zero-default behaviour FR-007 removes.
    """
    import numpy as np

    rows: list[tuple[float, ...]] = []
    labels: list[float] = []
    for index, example in enumerate(retained.examples):
        if len(example.x) != width:
            return Refusal(
                CHECK_REFRESH,
                "retained_width_mismatch",
                (
                    f"retained example {index} in {retained.path} has {len(example.x)} value(s) but the base "
                    f"contract declares {width} feature(s); refusing to rebuild from vectors that do not fit"
                ),
            )
        rows.append(example.x)
        labels.append(example.y)

    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=float)


# ---------------------------------------------------------------------------
# T017 — the atomic write
# ---------------------------------------------------------------------------


def _read_bytes_or_none(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _restore(path: Path, previous: bytes | None) -> None:
    """Put ``path`` back exactly as it was, or remove it if it was not there.

    Deliberately does **not** go through ``os.replace``. This runs only after a
    move has already failed, and the most likely reason a move failed is that
    ``os.replace`` itself is unavailable — a recovery path that depends on the
    thing that just broke is not a recovery path. A direct rewrite is not
    atomic, but the alternative on this branch is leaving an artifact and a
    manifest that disagree, which is the state FR-019 exists to prevent.
    """
    try:
        if previous is None:
            if path.exists():
                path.unlink()
            return
        path.write_bytes(previous)
    except OSError as exc:
        logger.error(
            "registry: refresh failed and %s could not be restored (%s); the local slot may be inconsistent",
            path,
            exc,
        )


def _place_atomically(
    artifact: Path,
    manifest_file: Path,
    payload: bytes,
    manifest: Manifest,
) -> Refusal | None:
    """Write both files to staging paths, then move both into place.

    Order of operations, and why:

    1. Snapshot the current artifact and manifest bytes.
    2. Write *both* staging files completely. A failure here has moved nothing.
    3. Move the artifact, then the manifest. If the manifest move fails, the
       artifact is restored from the snapshot before returning, so the pair on
       disk is the one that was there before — byte for byte.

    Returns ``None`` on success, or a :class:`Refusal` naming what failed. Never
    raises: the caller is a startup path, and a refresh that could not be
    written is a reason to keep serving the previous model (FR-019).
    """
    previous_artifact = _read_bytes_or_none(artifact)
    previous_manifest = _read_bytes_or_none(manifest_file)

    staged_artifact = artifact.with_name(f".{artifact.name}{_STAGING_SUFFIX}")
    staged_manifest = manifest_file.with_name(f".{manifest_file.name}{_STAGING_SUFFIX}")

    try:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        staged_artifact.write_bytes(payload)
        write_manifest(staged_manifest, manifest)
    except (OSError, ValueError, TypeError) as exc:
        _discard_staging(artifact.parent)
        return Refusal(
            CHECK_REFRESH,
            "staging_failed",
            (
                f"{manifest.name}: the rebuilt artifact and manifest could not be staged ({type(exc).__name__}: "
                f"{exc}); nothing was moved and the previous model is unchanged"
            ),
        )

    try:
        os.replace(staged_artifact, artifact)
    except OSError as exc:
        _discard_staging(artifact.parent)
        return Refusal(
            CHECK_REFRESH,
            "artifact_move_failed",
            (
                f"{manifest.name}: the rebuilt artifact could not be moved into place ({exc}); "
                "neither file was replaced and the previous model is unchanged"
            ),
        )

    try:
        os.replace(staged_manifest, manifest_file)
    except OSError as exc:
        # The dangerous state: a new artifact beside an old manifest, whose
        # recorded digest would no longer describe it. Undo the artifact move.
        _restore(artifact, previous_artifact)
        _restore(manifest_file, previous_manifest)
        _discard_staging(artifact.parent)
        return Refusal(
            CHECK_REFRESH,
            "manifest_move_failed",
            (
                f"{manifest.name}: the rebuilt manifest could not be moved into place ({exc}); the artifact move "
                "was undone so the previous artifact and manifest remain the pair on disk"
            ),
        )

    return None


def _discard_staging(directory: Path) -> None:
    """Sweep every staging file out of ``directory``.

    A glob rather than the two known paths, because :func:`write_manifest` does
    its own temp-then-replace *into* the staging path and leaves its inner
    temporary behind when that replace is the thing that failed. Sweeping by
    the staging suffix cleans up after it without this module having to know its
    internals, and stops a failed refresh from leaving debris the next one has
    to step over. Failure to remove a file here is not worth reporting.
    """
    try:
        leftovers = list(directory.glob(f"*{_STAGING_SUFFIX}*"))
    except OSError:
        return
    for path in leftovers:
        try:
            path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def _next_local_version(previous: Manifest | None) -> str:
    """The local artifact's own version counter, independent of the base's.

    Local versions count local writes (data-model.md §Manifest shows a local
    ``version`` of ``"4"`` descending from ``base_version`` ``"1"``). An
    unparseable predecessor restarts the count rather than propagating garbage.
    """
    if previous is None:
        return "1"
    try:
        return str(int(str(previous.version)) + 1)
    except (TypeError, ValueError):
        return "1"


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _base_copy_manifest(
    *,
    name: str,
    base_manifest: Manifest,
    previous_local: Manifest | None,
    generation: str | None,
    reset_reason: str | None,
    now_ms: int,
) -> Manifest:
    """A local manifest for an artifact that *is* the shipped base, byte for byte.

    Used by both the reset path and the empty-retained-set path. The artifact
    written beside it is the base's own bytes, so ``artifact_sha256``,
    ``runtime`` and ``metrics`` are the base's — SC-005 requires the shipped
    artifact stay byte-identical however many times it is adopted, and copying
    its digest rather than recomputing a "local" one keeps that checkable.

    ``n_local_extensions`` is 0 and ``training_source`` is ``base``: nothing
    local has been applied to these bytes. The manifest is written into the
    *local* slot rather than the model simply being left to resolve from the
    base slot, because the base slot is read-only (C-004, FR-015) and
    ``reset_reason`` has to be recorded somewhere an operator can find it.
    """
    return Manifest(
        name=name,
        version=_next_local_version(previous_local),
        artifact_sha256=base_manifest.artifact_sha256,
        schema_version=SCHEMA_VERSION,
        created_at=now_ms,
        provenance=Provenance(
            base_version=base_manifest.version,
            base_sha256=base_manifest.artifact_sha256,
            n_local_extensions=0,
            training_source=TRAINING_SOURCE_BASE,
            reset_reason=reset_reason,
        ),
        runtime=base_manifest.runtime,
        feature_contract=base_manifest.feature_contract,
        training=Training(n_samples=0, retained_generation=generation, as_of_ms=now_ms),
        metrics=dict(base_manifest.metrics),
    )


def _rebuilt_manifest(
    *,
    name: str,
    model: Any,
    payload: bytes,
    base_manifest: Manifest,
    previous_local: Manifest | None,
    retained: RetainedSet,
    n_samples: int,
    now_ms: int,
) -> Manifest:
    """A local manifest for a model retrained on the retained set.

    ``n_local_extensions`` is 1, not ``previous + 1``: the rebuild started from
    the shipped base and applied one local extension to it. The count describes
    the distance from the base named right beside it, and after a rebuild that
    distance is one however many times this install has refreshed before.
    """
    return Manifest(
        name=name,
        version=_next_local_version(previous_local),
        artifact_sha256=_digest(payload),
        schema_version=SCHEMA_VERSION,
        created_at=now_ms,
        provenance=Provenance(
            base_version=base_manifest.version,
            base_sha256=base_manifest.artifact_sha256,
            n_local_extensions=1,
            training_source=TRAINING_SOURCE_LOCAL,
            reset_reason=None,
        ),
        runtime=Runtime(
            estimator=type(model).__name__,
            sklearn_version=running_sklearn_version() or base_manifest.runtime.sklearn_version,
            python_version=_python_version(),
        ),
        feature_contract=base_manifest.feature_contract,
        training=Training(
            n_samples=n_samples,
            retained_generation=retained.generation,
            as_of_ms=now_ms,
        ),
    )


# ---------------------------------------------------------------------------
# The refresh itself
# ---------------------------------------------------------------------------


def _refused(name: str, change: BaseChange, refusal: Refusal) -> RefreshResult:
    logger.warning(
        "registry: refresh of %r did not proceed — %s/%s: %s", name, refusal.check, refusal.reason, refusal.detail
    )
    return RefreshResult(name=name, action=ACTION_NONE, change=change, ok=False, refusal=refusal)


def refresh_model(
    name: str,
    *,
    local_dir: Path | str | None = None,
    base_dir: Path | str | None = None,
    retained_dir: Path | str | None = None,
    expected_contract: FeatureContract | None = None,
    running_version: str | None = None,
    train: TrainFn | None = None,
    now_ms: int | None = None,
) -> RefreshResult:
    """Refresh one model against the shipped base, if one has changed.

    Does nothing at all unless :func:`detect_base_change` says a refresh is due.
    When it is, the retained set decides which of the three paths runs; see the
    module docstring for the table.

    Args:
        name: The model's registry name.
        local_dir: Overrides the writable local slot.
        base_dir: Overrides the read-only base slot. This module reads it and
            never writes to it (C-004, FR-015).
        retained_dir: Overrides the retained-set directory.
        expected_contract: The contract to validate the base artifact against
            before deserializing it. ``None`` lets WP01 source it from Feast.
        running_version: Override for the running scikit-learn version.
        train: The training callable, ``(base_model, X, y) -> model``. Defaults
            to :func:`_full_retrain`. Present so a caller can substitute a
            trainer; substituting one that warm-starts would violate C-006.
        now_ms: Override for the manifest timestamps. Note these are *recorded*
            values, not inputs to any decision — nothing in refresh reads a time
            to decide anything (FR-012, D-007).

    Returns:
        A :class:`RefreshResult`. Never raises. Any failure leaves the previous
        local artifact and manifest exactly as they were (FR-019).
    """
    change = detect_base_change(name, local_dir=local_dir, base_dir=base_dir)
    if not change.due:
        logger.debug("registry: %s", change)
        return RefreshResult(name=name, action=ACTION_NONE, change=change)

    base_manifest = change.base_manifest
    if base_manifest is None:  # unreachable: due implies both manifests were read
        return _refused(
            name,
            change,
            Refusal(CHECK_REFRESH, "no_base_manifest", f"{name}: refresh is due but no base manifest was read"),
        )

    base_root = Path(base_dir) if base_dir is not None else base_slot_dir()
    local_root = Path(local_dir) if local_dir is not None else local_slot_dir()

    logger.info(
        "registry: base refresh due for %r — %s/%s: %s",
        name,
        CHECK_REFRESH,
        change.reason,
        change.detail,
    )

    # The base is about to be either deserialized or copied into the local slot.
    # Both require it to be what its manifest says it is, so it goes through
    # WP01's three checks first — integrity, then the ordered contract, then the
    # runtime. That call is also the only producer of a `VerifiedArtifact` and
    # therefore the only route to `deserialize_verified`: there is no path from
    # here to `joblib.load` that skipped the digest.
    verified_outcome = validate_artifact(
        base_manifest,
        artifact_path(base_root, name),
        expected_contract=expected_contract,
        running_version=running_version,
    )
    if not verified_outcome.ok or verified_outcome.verified is None:
        detail = verified_outcome.refusal.detail if verified_outcome.refusal else "base artifact failed verification"
        return _refused(
            name,
            change,
            Refusal(
                CHECK_REFRESH,
                "base_artifact_unusable",
                f"{name}: the shipped base cannot be trusted, so no refresh is attempted — {detail}",
            ),
        )
    verified = verified_outcome.verified

    retained = read_retained(name, directory=retained_dir)
    base_contract = base_manifest.feature_contract
    base_contract_version = base_contract.service_version

    artifact = artifact_path(local_root, name)
    manifest_file = manifest_path(local_root, name)
    stamp = now_ms if now_ms is not None else _now_ms()

    # --- T016: the contract moved. The retained vectors are not replayable. ---
    if retained.ok and retained.contract_version != base_contract_version:
        return _reset(
            name=name,
            change=change,
            base_manifest=base_manifest,
            verified_payload=verified.payload,
            retained=retained,
            retained_dir=retained_dir,
            artifact=artifact,
            manifest_file=manifest_file,
            now_ms=stamp,
        )

    # --- User Story 3 scenario 3: nothing retained. Serve the new base. ---
    if not retained.examples:
        return _adopt_base(
            name=name,
            change=change,
            base_manifest=base_manifest,
            verified_payload=verified.payload,
            retained=retained,
            artifact=artifact,
            manifest_file=manifest_file,
            now_ms=stamp,
        )

    # --- T015: the contract held. Replay the retained set onto the new base. ---
    return _rebuild(
        name=name,
        change=change,
        base_manifest=base_manifest,
        verified=verified,
        retained=retained,
        artifact=artifact,
        manifest_file=manifest_file,
        train=train or _full_retrain,
        now_ms=stamp,
    )


def _rebuild(
    *,
    name: str,
    change: BaseChange,
    base_manifest: Manifest,
    verified: Any,
    retained: RetainedSet,
    artifact: Path,
    manifest_file: Path,
    train: TrainFn,
    now_ms: int,
) -> RefreshResult:
    """FR-013 — full retraining on the retained examples, onto the new base."""
    width = len(base_manifest.feature_contract.names) or len(retained.names)
    built = _matrix(retained, width)
    if isinstance(built, Refusal):
        return _refused(name, change, built)
    x, y = built

    loaded = deserialize_verified(verified)
    if not loaded.ok or loaded.model is None:
        detail = loaded.refusal.detail if loaded.refusal else "base artifact could not be deserialized"
        return _refused(
            name,
            change,
            Refusal(CHECK_REFRESH, "base_undeserializable", f"{name}: {detail}"),
        )

    try:
        model = train(loaded.model, x, y)
        payload = _dump(model)
    except Exception as exc:  # noqa: BLE001 — a failed rebuild must not take the process down
        return _refused(
            name,
            change,
            Refusal(
                CHECK_REFRESH,
                "retraining_failed",
                (
                    f"{name}: retraining on {len(retained.examples)} retained example(s) failed "
                    f"({type(exc).__name__}: {exc}); the previous local model is unchanged"
                ),
            ),
        )

    manifest = _rebuilt_manifest(
        name=name,
        model=model,
        payload=payload,
        base_manifest=base_manifest,
        previous_local=change.local_manifest,
        retained=retained,
        n_samples=len(retained.examples),
        now_ms=now_ms,
    )

    refusal = _place_atomically(artifact, manifest_file, payload, manifest)
    if refusal is not None:
        return _refused(name, change, refusal)

    logger.info(
        "registry: rebuilt %r from %d retained example(s) (generation %s); base %s -> %s",
        name,
        len(retained.examples),
        retained.generation or "-",
        change.local_base_version or "-",
        base_manifest.version,
    )
    return RefreshResult(
        name=name,
        action=ACTION_REBUILD,
        change=change,
        manifest=manifest,
        artifact=artifact,
        n_samples=len(retained.examples),
        retained_generation=retained.generation,
    )


def _adopt_base(
    *,
    name: str,
    change: BaseChange,
    base_manifest: Manifest,
    verified_payload: bytes,
    retained: RetainedSet,
    artifact: Path,
    manifest_file: Path,
    now_ms: int,
) -> RefreshResult:
    """User Story 3 scenario 3 — nothing retained, so the new base is served as is.

    Not an error and not a reset: no personalization was lost because none had
    been accumulated. ``reset_reason`` stays unset precisely so the two are
    distinguishable by an operator reading the manifest.
    """
    manifest = _base_copy_manifest(
        name=name,
        base_manifest=base_manifest,
        previous_local=change.local_manifest,
        generation=retained.generation,
        reset_reason=None,
        now_ms=now_ms,
    )

    refusal = _place_atomically(artifact, manifest_file, verified_payload, manifest)
    if refusal is not None:
        return _refused(name, change, refusal)

    logger.info(
        "registry: no retained examples for %r; serving shipped base %s unextended (was %s)",
        name,
        base_manifest.version,
        change.local_base_version or "-",
    )
    return RefreshResult(
        name=name,
        action=ACTION_ADOPT_BASE,
        change=change,
        manifest=manifest,
        artifact=artifact,
        retained_generation=retained.generation,
    )


def _reset(
    *,
    name: str,
    change: BaseChange,
    base_manifest: Manifest,
    verified_payload: bytes,
    retained: RetainedSet,
    retained_dir: Path | str | None,
    artifact: Path,
    manifest_file: Path,
    now_ms: int,
) -> RefreshResult:
    """FR-014 — the contract moved, so the retained set is discarded honestly.

    The order matters. The local pair is written *first* and the retained set is
    reset only once that succeeded: a reset that discarded the examples and then
    failed to write would have destroyed the personalization without delivering
    the new base, which is the worst of both outcomes.
    """
    following = _peek_next_generation(retained.generation)

    manifest = _base_copy_manifest(
        name=name,
        base_manifest=base_manifest,
        previous_local=change.local_manifest,
        generation=following,
        reset_reason=RESET_REASON_CONTRACT_CHANGED,
        now_ms=now_ms,
    )

    refusal = _place_atomically(artifact, manifest_file, verified_payload, manifest)
    if refusal is not None:
        return _refused(name, change, refusal)

    result = reset_retained(
        name,
        contract=base_manifest.feature_contract,
        directory=retained_dir,
        created_at=now_ms,
    )

    # WARNING, not DEBUG. This install just lost every locally-adapted example
    # it had accumulated — potentially months of them. It is recoverable only by
    # accumulating again, and the user is entitled to find out from the log
    # rather than by noticing the predictions changed.
    logger.warning(
        "registry: personalization reset for %r — the shipped base %s declares feature contract %s but the "
        "%d retained example(s) were recorded under %s, so they cannot be replayed. They have been discarded "
        "(generation %s -> %s) and base %s is now served unextended; reset_reason=%s is recorded in %s.",
        name,
        base_manifest.version,
        base_manifest.feature_contract.service_version,
        len(retained.examples),
        retained.contract_version,
        result.previous_generation or "-",
        result.next_generation,
        base_manifest.version,
        RESET_REASON_CONTRACT_CHANGED,
        manifest_file,
    )

    return RefreshResult(
        name=name,
        action=ACTION_RESET,
        change=change,
        manifest=manifest,
        artifact=artifact,
        retained_generation=result.next_generation,
        previous_generation=result.previous_generation,
        reset_reason=RESET_REASON_CONTRACT_CHANGED,
    )


def _peek_next_generation(current: str | None) -> str:
    """The generation the reset will land on, computed before the reset runs.

    The manifest is written before the retained set is reset, so the generation
    it records has to be known in advance. WP03's :func:`next_generation` is the
    single definition of the increment; this only calls it early.
    """
    from kenaz_ml.modelstore.registry.retained import next_generation

    return next_generation(current)


def _dump(model: Any) -> bytes:
    import io

    import joblib

    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()


def refresh_all(
    names: Sequence[str],
    *,
    local_dir: Path | str | None = None,
    base_dir: Path | str | None = None,
    retained_dir: Path | str | None = None,
    train: TrainFn | None = None,
    now_ms: int | None = None,
) -> tuple[RefreshResult, ...]:
    """Refresh a roster of models, one independently of the next.

    One model's refusal never stops another's refresh: a broken artifact for
    ``stuck`` is no reason to leave ``duration`` descended from a base that is
    no longer shipped.
    """
    return tuple(
        refresh_model(
            name,
            local_dir=local_dir,
            base_dir=base_dir,
            retained_dir=retained_dir,
            train=train,
            now_ms=now_ms,
        )
        for name in names
    )


def describe(result: RefreshResult) -> dict[str, Any]:
    """Render a refresh result as plain data, for a diagnostics surface.

    Kept here rather than in a route so the shape is defined next to the values
    it describes; the discovery surface that consumes it is a later mission.
    """
    return {
        "name": result.name,
        "action": result.action,
        "ok": result.ok,
        "reason": result.change.reason,
        "detail": result.change.detail,
        "local_base_version": result.change.local_base_version,
        "base_version": result.change.base_version,
        "n_samples": result.n_samples,
        "retained_generation": result.retained_generation,
        "previous_generation": result.previous_generation,
        "reset_reason": result.reset_reason,
        "refusal": str(result.refusal) if result.refusal else None,
        "manifest": manifest_to_dict(result.manifest) if result.manifest else None,
    }
