"""Model registry: manifests, validation, and the slot layout above `ModelStore`.

The registry is a layer *over* `ModelStore`, which keeps its bytes-in/bytes-out
protocol untouched (D-005). It lives under `modelstore/` because everything it
governs is a model-artifact concern; a sibling top-level package would scatter
that back apart.

What it adds is a sidecar JSON manifest per artifact (D-004) and the three
checks that stand between an artifact on disk and a model object in memory:
integrity before deserialization (FR-005), an *ordered* feature-contract
comparison (FR-006), and runtime compatibility (FR-008). None of them raises;
each returns a `Refusal` naming what disagreed, so resolution falls through
rather than crashing (FR-017).

The shortest safe path from a file to a model::

    read = read_manifest(models_dir / "stuck.json")
    if read.ok:
        outcome = validate_artifact(read.manifest, models_dir / "stuck.joblib")
        if outcome.ok:
            loaded = deserialize_verified(outcome.verified)

`deserialize_verified` accepts only a `VerifiedArtifact`, and only
`verify_artifact_bytes`/`verify_artifact_file` produce one — so there is no
shorter path, and in particular none that reaches `joblib.load` without a
matching digest.

Above that sit slot resolution — local slot, then base slot, then cold start
(FR-004) — the retained training set the local examples are kept in, and the
refresh policy that decides what a newly shipped base means for them::

    resolution = resolve_model("stuck")
    if resolution.served:
        use(resolution.model)          # validated; the slot that answered is
                                       # named by resolution.slot

Import from this package, not from `manifest`, `slots`, `retained` or `refresh`
directly. The submodule split is an implementation detail; this `__all__` is the
supported surface, and it is the surface the tests should reach for too.
"""

from sigil_ml.modelstore.registry.manifest import (
    CHECK_CONTRACT,
    CHECK_INTEGRITY,
    CHECK_MANIFEST,
    CHECK_RUNTIME,
    SCHEMA_VERSION,
    FeatureContract,
    LoadOutcome,
    Manifest,
    ManifestRead,
    Provenance,
    Refusal,
    Runtime,
    Training,
    ValidationOutcome,
    VerifiedArtifact,
    deserialize_verified,
    local_feature_contract,
    manifest_from_dict,
    manifest_to_dict,
    read_manifest,
    read_manifest_text,
    running_sklearn_version,
    validate_artifact,
    validate_feature_contract,
    validate_runtime,
    verify_artifact_bytes,
    verify_artifact_file,
    write_manifest,
)
from sigil_ml.modelstore.registry.refresh import (
    ACTION_ADOPT_BASE,
    ACTION_NONE,
    ACTION_REBUILD,
    ACTION_RESET,
    CHECK_REFRESH,
    REASON_BASE_DIGEST_CHANGED,
    REASON_BASE_MANIFEST_UNUSABLE,
    REASON_BASE_VERSION_CHANGED,
    REASON_LOCAL_MANIFEST_UNUSABLE,
    REASON_NO_BASE,
    REASON_NO_LOCAL,
    REASON_UP_TO_DATE,
    RESET_REASON_CONTRACT_CHANGED,
    TRAINING_SOURCE_BASE,
    TRAINING_SOURCE_LOCAL,
    BaseChange,
    RefreshResult,
    TrainFn,
    describe,
    detect_base_change,
    refresh_all,
    refresh_model,
)
from sigil_ml.modelstore.registry.retained import (
    DEFAULT_MAX_BYTES,
    INITIAL_GENERATION,
    RECORD_EXAMPLE,
    RECORD_HEADER,
    AppendResult,
    Example,
    ResetResult,
    RetainedHeader,
    RetainedSet,
    RetainedSummary,
    append_examples,
    delete_retained,
    next_generation,
    read_retained,
    reset_retained,
    retained_path,
    summarize_all,
    summarize_retained,
)
from sigil_ml.modelstore.registry.slots import (
    ARTIFACT_SUFFIX,
    CHECK_SLOT,
    MANIFEST_SUFFIX,
    REASON_ARTIFACT_NOT_FOUND,
    REASON_SLOT_EMPTY,
    SLOT_BASE,
    SLOT_COLD_START,
    SLOT_LOCAL,
    Resolution,
    SlotRefusal,
    artifact_path,
    base_slot_dir,
    local_slot_dir,
    manifest_path,
    resolve_model,
    slot_is_empty,
)

__all__ = [
    "ACTION_ADOPT_BASE",
    "ACTION_NONE",
    "ACTION_REBUILD",
    "ACTION_RESET",
    "ARTIFACT_SUFFIX",
    "CHECK_CONTRACT",
    "CHECK_INTEGRITY",
    "CHECK_MANIFEST",
    "CHECK_REFRESH",
    "CHECK_RUNTIME",
    "CHECK_SLOT",
    "DEFAULT_MAX_BYTES",
    "INITIAL_GENERATION",
    "MANIFEST_SUFFIX",
    "REASON_ARTIFACT_NOT_FOUND",
    "REASON_BASE_DIGEST_CHANGED",
    "REASON_BASE_MANIFEST_UNUSABLE",
    "REASON_BASE_VERSION_CHANGED",
    "REASON_LOCAL_MANIFEST_UNUSABLE",
    "REASON_NO_BASE",
    "REASON_NO_LOCAL",
    "REASON_SLOT_EMPTY",
    "REASON_UP_TO_DATE",
    "RECORD_EXAMPLE",
    "RECORD_HEADER",
    "RESET_REASON_CONTRACT_CHANGED",
    "SCHEMA_VERSION",
    "SLOT_BASE",
    "SLOT_COLD_START",
    "SLOT_LOCAL",
    "TRAINING_SOURCE_BASE",
    "TRAINING_SOURCE_LOCAL",
    "AppendResult",
    "BaseChange",
    "Example",
    "FeatureContract",
    "LoadOutcome",
    "Manifest",
    "ManifestRead",
    "Provenance",
    "RefreshResult",
    "Refusal",
    "ResetResult",
    "Resolution",
    "RetainedHeader",
    "RetainedSet",
    "RetainedSummary",
    "Runtime",
    "SlotRefusal",
    "TrainFn",
    "Training",
    "ValidationOutcome",
    "VerifiedArtifact",
    "append_examples",
    "artifact_path",
    "base_slot_dir",
    "delete_retained",
    "describe",
    "deserialize_verified",
    "detect_base_change",
    "local_feature_contract",
    "local_slot_dir",
    "manifest_from_dict",
    "manifest_path",
    "manifest_to_dict",
    "next_generation",
    "read_manifest",
    "read_manifest_text",
    "read_retained",
    "refresh_all",
    "refresh_model",
    "reset_retained",
    "resolve_model",
    "retained_path",
    "running_sklearn_version",
    "slot_is_empty",
    "summarize_all",
    "summarize_retained",
    "validate_artifact",
    "validate_feature_contract",
    "validate_runtime",
    "verify_artifact_bytes",
    "verify_artifact_file",
    "write_manifest",
]
