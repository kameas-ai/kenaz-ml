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

Import from this package, not from `manifest` directly.
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

__all__ = [
    "CHECK_CONTRACT",
    "CHECK_INTEGRITY",
    "CHECK_MANIFEST",
    "CHECK_RUNTIME",
    "SCHEMA_VERSION",
    "FeatureContract",
    "LoadOutcome",
    "Manifest",
    "ManifestRead",
    "Provenance",
    "Refusal",
    "Runtime",
    "Training",
    "ValidationOutcome",
    "VerifiedArtifact",
    "deserialize_verified",
    "local_feature_contract",
    "manifest_from_dict",
    "manifest_to_dict",
    "read_manifest",
    "read_manifest_text",
    "running_sklearn_version",
    "validate_artifact",
    "validate_feature_contract",
    "validate_runtime",
    "verify_artifact_bytes",
    "verify_artifact_file",
    "write_manifest",
]
