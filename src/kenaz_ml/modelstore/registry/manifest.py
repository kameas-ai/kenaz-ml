"""Manifest schema and load-time validation for model artifacts.

A manifest is a sidecar JSON file named ``{model_name}.json`` sitting beside
``{model_name}.joblib`` (D-004). It records what the artifact *is* — identity,
provenance, runtime compatibility, feature contract, training summary, metrics
— and the SHA-256 of the artifact bytes. The same schema is the interface the
cloud MLflow export job writes (C-002), so nothing in it may assume a local
path: the manifest names no directory, no slot, and no filesystem root.

Three checks run before an artifact may be used, in this order (data-model.md
§Resolution):

1. **Integrity** — recomputed SHA-256 over the artifact bytes equals
   ``artifact_sha256``. Runs *before* deserialization, never after (FR-005).
2. **Contract** — the recorded feature contract equals, *in order*, the contract
   the local Feast feature service declares (FR-006). Fails closed.
3. **Runtime** — ``runtime.sklearn_version`` is compatible with the running
   scikit-learn (FR-008).

**No validation path raises to its caller** (FR-017). Every check returns a
:class:`Refusal` naming what disagreed. A corrupt manifest, a tampered
artifact, and a contract mismatch are all data conditions the resolver handles
by falling through to the next slot; none of them is an exception.

**Dependencies**: module scope imports the standard library only (C-001, NFR-001).
``joblib`` and ``sklearn`` are imported inside the two functions that need them,
and Feast inside the one that sources the contract — so reading and writing a
manifest costs no import of the heavy stack (NFR-002).

Directory artifacts (laya-serving-and-packs-01MSK2SP WP04, FR-011)
-----------------------------------------------------------------
A laya checkpoint is a *directory*, not a joblib file. The manifest therefore
carries an optional ``artifact_kind`` (``"file"`` -- the default, absent from
every existing manifest -- or ``"directory"``) and, for a directory, an optional
``artifact_members`` map of relative path -> sha256 that makes refusals
specific. For a directory, ``artifact_sha256`` is the **canonical tree digest**
(:func:`tree_digest`: ``"sha256:"`` + sha256 over sha256sum-format
``"<member hash>  <path>\\n"`` lines sorted bytewise by path) -- the same recipe,
byte for byte, as the Go clients' ``tree.go`` (design doc A5 addendum (b)),
deterministic and independent of filesystem listing order. The loader hands back
the **verified directory path** (:class:`VerifiedDirectory`), never a
deserialized object: laya loads its own checkpoint, and is given the per-member
digests to re-verify as it parses (so the window between this check and laya's
read is closed by laya, not assumed away). Joblib pairs are untouched.

Ordering is the whole point
---------------------------
Two properties of this module are load-bearing and easy to break:

* **Integrity precedes deserialization structurally, not by convention.**
  :func:`verify_artifact_file` and :func:`verify_artifact_bytes` are the only
  producers of a :class:`VerifiedArtifact`, and :func:`deserialize_verified` is
  the only consumer — it accepts a ``VerifiedArtifact`` and nothing else, and
  ``VerifiedArtifact`` cannot be constructed from outside this module. There is
  therefore no call sequence that reaches ``joblib.load`` without a matching
  digest. This matters because ``joblib.load`` on a shipped artifact is
  arbitrary code execution: a checksum verified *after* deserialization
  provides no protection whatsoever.

* **Contract comparison is ordered, element by element.** Both trainers build
  vectors positionally (``[features.get(f, 0.0) for f in FEATURE_NAMES]``), so a
  permutation of the same names feeds every value into the wrong slot with no
  error raised anywhere. ``set(a) == set(b)`` accepts exactly that permutation
  and must never appear here (D-006).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CHECK_CONTRACT",
    "CHECK_INTEGRITY",
    "CHECK_MANIFEST",
    "CHECK_RUNTIME",
    "SCHEMA_VERSION",
    "ARTIFACT_KINDS",
    "ARTIFACT_KIND_DIRECTORY",
    "ARTIFACT_KIND_FILE",
    "KNOWN_TRAINING_SOURCES",
    "TRAINING_SOURCE_ORG",
    "DirectoryDigest",
    "VerifiedDirectory",
    "TREE_DIGEST_PREFIX",
    "directory_digest",
    "tree_digest",
    "validate_artifact_directory",
    "verify_artifact_directory",
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

#: ``Manifest.artifact_kind`` values (FR-011). ``file`` is the default and is never written.
ARTIFACT_KIND_FILE = "file"
ARTIFACT_KIND_DIRECTORY = "directory"
ARTIFACT_KINDS: tuple[str, ...] = (ARTIFACT_KIND_FILE, ARTIFACT_KIND_DIRECTORY)

#: ``Provenance.training_source`` values. ``org`` (WP04, R5) stamps an artifact that
#: arrived as an org-distributed pack; nothing in this repository produces one
#: (the org fine-tune producer is a named future mission). The field stays free-form
#: on read -- an unknown value is preserved, never refused -- so this is the
#: vocabulary, not a gate.
TRAINING_SOURCE_ORG = "org"
KNOWN_TRAINING_SOURCES: tuple[str, ...] = ("base", "local", "synthetic", TRAINING_SOURCE_ORG)

#: Manifest format version. Distinct from the feature-contract version, which
#: tracks the feature set rather than the shape of this file.
SCHEMA_VERSION = "1"

#: Names of the three checks, used as :attr:`Refusal.check`.
CHECK_MANIFEST = "manifest"
CHECK_INTEGRITY = "integrity"
CHECK_CONTRACT = "contract"
CHECK_RUNTIME = "runtime"

#: Chunk size for streaming digests. Large enough that the loop overhead is
#: irrelevant against NFR-003 (under 200ms for a 50MB artifact), small enough
#: that the hash is genuinely incremental rather than one buffer-wide call.
_DIGEST_CHUNK_BYTES = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Structured results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Refusal:
    """Why an artifact was not used. Returned, never raised (FR-017).

    Attributes:
        check: Which check refused — ``manifest``, ``integrity``, ``contract``
            or ``runtime``. Lets a caller log the stage without parsing text.
        reason: Stable machine-readable code, e.g. ``digest_mismatch``. Safe to
            branch on; the wording of ``detail`` is not.
        detail: Operator-actionable text naming the specific disagreement —
            both digests, the diverging feature index, both versions. "Contract
            mismatch" alone is not something anyone can act on, so no refusal
            here is allowed to say only that.
    """

    check: str
    reason: str
    detail: str

    def __str__(self) -> str:
        return f"{self.check}/{self.reason}: {self.detail}"


class VerifiedArtifact:
    """Artifact bytes whose SHA-256 matched the manifest.

    Holding one of these *is* the proof that integrity verification ran: the
    constructor is guarded by a module-private token, so the only way to obtain
    an instance is through :func:`verify_artifact_bytes` or
    :func:`verify_artifact_file`, and the only function that deserializes is
    :func:`deserialize_verified`, which takes this type and nothing else.

    The payload is carried in memory rather than re-read from disk at
    deserialization time. That is deliberate: re-reading would reopen the
    window between "these bytes hashed correctly" and "these bytes were
    executed", which is the window the check exists to close.
    """

    __slots__ = ("_payload", "digest", "manifest")

    def __init__(self, token: object, manifest: Manifest, payload: bytes, digest: str) -> None:
        if token is not _VERIFIED_TOKEN:
            # Not a validation path: reaching here means code tried to forge a
            # verification result, which is a defect and not a data condition.
            raise TypeError(
                "VerifiedArtifact cannot be constructed directly; obtain one from "
                "verify_artifact_bytes() or verify_artifact_file() so that the digest is checked"
            )
        self.manifest = manifest
        self.digest = digest
        self._payload = payload

    @property
    def payload(self) -> bytes:
        """The verified bytes. Safe to hand anywhere: their digest matched."""
        return self._payload

    def __len__(self) -> int:
        return len(self._payload)

    def __repr__(self) -> str:
        return f"VerifiedArtifact(name={self.manifest.name!r}, digest={self.digest[:12]}…, bytes={len(self._payload)})"


_VERIFIED_TOKEN = object()


@dataclass(frozen=True)
class ValidationOutcome:
    """Result of a check. Exactly one of ``refusal`` / a value is meaningful.

    ``ok`` is the only thing callers should branch on. A check that passes but
    has nothing to hand back (contract, runtime) leaves ``verified`` unset.
    """

    refusal: Refusal | None = None
    verified: VerifiedArtifact | None = None

    @property
    def ok(self) -> bool:
        return self.refusal is None


@dataclass(frozen=True)
class LoadOutcome:
    """Result of deserializing a verified artifact.

    ``model`` is unset when ``refusal`` is set. ``model`` may legitimately be a
    falsy object, so branch on :attr:`ok`, never on ``model`` itself.
    """

    model: Any | None = None
    refusal: Refusal | None = None

    @property
    def ok(self) -> bool:
        return self.refusal is None


@dataclass(frozen=True)
class ManifestRead:
    """Result of reading a manifest from disk.

    A missing, unreadable, unparseable or structurally wrong manifest yields a
    refusal — the "unusable" outcome of FR-017 — never an exception.
    """

    manifest: Manifest | None = None
    refusal: Refusal | None = None

    @property
    def ok(self) -> bool:
        return self.manifest is not None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
#
# Every section carries an ``extra`` mapping holding fields this version does
# not know about. Unknown fields are *preserved*, not dropped: a newer cloud
# exporter (C-002) may add fields, and a local read/write cycle must not
# silently strip them. They are re-emitted after the known fields, key-sorted,
# so output stays deterministic.


def _extra() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class Provenance:
    """Where this artifact came from and what has been done to it.

    A *base* manifest omits ``base_version``/``base_sha256`` and carries
    ``training_source="base"``. A local manifest names the base it descended
    from and how many times it has been extended.
    """

    base_version: str | None = None
    base_sha256: str | None = None
    n_local_extensions: int = 0
    training_source: str = "base"
    reset_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=_extra)

    @property
    def is_base(self) -> bool:
        """True when this manifest describes a pristine shipped base artifact."""
        return self.base_version is None


@dataclass(frozen=True)
class Runtime:
    """The runtime that serialized the artifact.

    ``sklearn_version`` is checked at load (FR-008); the other two fields are
    recorded for support and are not gates.
    """

    estimator: str = ""
    sklearn_version: str = ""
    python_version: str = ""
    extra: dict[str, Any] = field(default_factory=_extra)


@dataclass(frozen=True)
class FeatureContract:
    """The feature set this artifact was trained against.

    ``service`` and ``service_version`` name a Feast feature service and its
    content hash (see :func:`local_feature_contract`); together they are the
    exact identity of the contract. ``names`` and ``dtypes`` are the *ordered*
    expansion of that service, carried in the manifest so a mismatch can be
    diagnosed positionally rather than reported as an opaque hash difference.

    ``names`` is ordered and the order is the vector layout. Do not sort it,
    do not compare it as a set.
    """

    service: str | None = None
    service_version: str | None = None
    names: tuple[str, ...] = ()
    dtypes: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=_extra)


@dataclass(frozen=True)
class Training:
    """Summary of the run that produced the artifact. Entirely optional."""

    n_samples: int | None = None
    retained_generation: str | None = None
    as_of_ms: int | None = None
    extra: dict[str, Any] = field(default_factory=_extra)


@dataclass(frozen=True)
class Manifest:
    """One artifact's record. See the module docstring for the JSON shape."""

    name: str
    version: str
    artifact_sha256: str
    schema_version: str = SCHEMA_VERSION
    created_at: int | None = None
    provenance: Provenance = field(default_factory=Provenance)
    runtime: Runtime = field(default_factory=Runtime)
    feature_contract: FeatureContract = field(default_factory=FeatureContract)
    training: Training = field(default_factory=Training)
    metrics: dict[str, Any] = field(default_factory=_extra)
    extra: dict[str, Any] = field(default_factory=_extra)
    #: ``file`` (default; a joblib pair) or ``directory`` (a laya checkpoint). FR-011.
    artifact_kind: str = ARTIFACT_KIND_FILE
    #: Directory artifacts only: relative posix path -> sha256 of that member. Optional;
    #: when present it makes a refusal name the member, and is handed to laya for its own re-verification.
    artifact_members: dict[str, str] = field(default_factory=_extra)


# ---------------------------------------------------------------------------
# T001 — read and write
# ---------------------------------------------------------------------------

_PROVENANCE_KEYS = ("base_version", "base_sha256", "n_local_extensions", "training_source", "reset_reason")
_RUNTIME_KEYS = ("estimator", "sklearn_version", "python_version")
_CONTRACT_KEYS = ("service", "service_version", "names", "dtypes")
_TRAINING_KEYS = ("n_samples", "retained_generation", "as_of_ms")
_MANIFEST_KEYS = (
    "schema_version",
    "name",
    "version",
    "created_at",
    "provenance",
    "runtime",
    "feature_contract",
    "training",
    "metrics",
    "artifact_sha256",
    "artifact_kind",
    "artifact_members",
)


def _unknown(source: dict[str, Any], known: tuple[str, ...]) -> dict[str, Any]:
    return {k: v for k, v in source.items() if k not in known}


def _section(raw: Any) -> dict[str, Any]:
    """Coerce a manifest section to a mapping, tolerating a missing one."""
    return raw if isinstance(raw, dict) else {}


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _str_tuple(value: Any) -> tuple[str, ...]:
    """Order-preserving coercion of a JSON array to a tuple of strings.

    Order is preserved because order is the contract (D-006). Anything that is
    not a list yields an empty tuple, which validation then refuses rather than
    silently accepting a scalar as a one-feature contract.
    """
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def manifest_from_dict(raw: dict[str, Any], *, source: str = "<dict>") -> ManifestRead:
    """Build a :class:`Manifest` from a parsed JSON object.

    Tolerant by design (T001): unknown fields are preserved into ``extra``,
    missing optional fields fall back to their defaults, and a base-shaped
    manifest — no ``provenance.base_version``, no ``base_sha256`` — parses
    normally. The only hard requirements are ``name``, ``version`` and
    ``artifact_sha256``; without those there is nothing to identify or verify,
    and the manifest is unusable.
    """
    if not isinstance(raw, dict):
        return ManifestRead(
            refusal=Refusal(
                CHECK_MANIFEST,
                "not_an_object",
                f"{source}: manifest must be a JSON object, found {type(raw).__name__}",
            )
        )

    missing = [k for k in ("name", "version", "artifact_sha256") if not raw.get(k)]
    if missing:
        return ManifestRead(
            refusal=Refusal(
                CHECK_MANIFEST,
                "missing_required_field",
                f"{source}: manifest is missing required field(s) {', '.join(missing)}",
            )
        )

    kind = str(raw.get("artifact_kind", ARTIFACT_KIND_FILE))
    if kind not in ARTIFACT_KINDS:
        return ManifestRead(
            refusal=Refusal(
                CHECK_MANIFEST,
                "unknown_artifact_kind",
                f"{source}: artifact_kind {kind!r} is not one of {list(ARTIFACT_KINDS)}",
            )
        )
    members_raw = raw.get("artifact_members")
    members = {str(k): str(v).lower() for k, v in members_raw.items()} if isinstance(members_raw, dict) else {}

    prov_raw = _section(raw.get("provenance"))
    runtime_raw = _section(raw.get("runtime"))
    contract_raw = _section(raw.get("feature_contract"))
    training_raw = _section(raw.get("training"))

    n_extensions = _int_or_none(prov_raw.get("n_local_extensions"))
    manifest = Manifest(
        name=str(raw["name"]),
        version=str(raw["version"]),
        artifact_sha256=str(raw["artifact_sha256"]).lower(),
        schema_version=str(raw.get("schema_version", SCHEMA_VERSION)),
        created_at=_int_or_none(raw.get("created_at")),
        provenance=Provenance(
            base_version=_str_or_none(prov_raw.get("base_version")),
            base_sha256=_str_or_none(prov_raw.get("base_sha256")),
            n_local_extensions=n_extensions if n_extensions is not None else 0,
            training_source=str(prov_raw.get("training_source", "base")),
            reset_reason=_str_or_none(prov_raw.get("reset_reason")),
            extra=_unknown(prov_raw, _PROVENANCE_KEYS),
        ),
        runtime=Runtime(
            estimator=str(runtime_raw.get("estimator", "")),
            sklearn_version=str(runtime_raw.get("sklearn_version", "")),
            python_version=str(runtime_raw.get("python_version", "")),
            extra=_unknown(runtime_raw, _RUNTIME_KEYS),
        ),
        feature_contract=FeatureContract(
            service=_str_or_none(contract_raw.get("service")),
            service_version=_str_or_none(contract_raw.get("service_version")),
            names=_str_tuple(contract_raw.get("names")),
            dtypes=_str_tuple(contract_raw.get("dtypes")),
            extra=_unknown(contract_raw, _CONTRACT_KEYS),
        ),
        training=Training(
            n_samples=_int_or_none(training_raw.get("n_samples")),
            retained_generation=_str_or_none(training_raw.get("retained_generation")),
            as_of_ms=_int_or_none(training_raw.get("as_of_ms")),
            extra=_unknown(training_raw, _TRAINING_KEYS),
        ),
        metrics=dict(_section(raw.get("metrics"))),
        extra=_unknown(raw, _MANIFEST_KEYS),
        artifact_kind=kind,
        artifact_members=members,
    )
    return ManifestRead(manifest=manifest)


def read_manifest_text(text: str, *, source: str = "<text>") -> ManifestRead:
    """Parse manifest JSON from a string. Never raises (FR-017)."""
    try:
        raw = json.loads(text)
    except (ValueError, UnicodeDecodeError) as exc:
        return ManifestRead(refusal=Refusal(CHECK_MANIFEST, "unparseable", f"{source}: not valid JSON — {exc}"))
    return manifest_from_dict(raw, source=source)


def read_manifest(path: Path | str) -> ManifestRead:
    """Read and parse a manifest file.

    Missing file, unreadable file, corrupt JSON and a structurally wrong
    document all produce a refusal — the "unusable" outcome the resolver falls
    through on (FR-017). Nothing here raises.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ManifestRead(refusal=Refusal(CHECK_MANIFEST, "not_found", f"{path}: no manifest file"))
    except OSError as exc:
        return ManifestRead(refusal=Refusal(CHECK_MANIFEST, "unreadable", f"{path}: cannot be read — {exc}"))
    except UnicodeDecodeError as exc:
        return ManifestRead(refusal=Refusal(CHECK_MANIFEST, "unparseable", f"{path}: not valid UTF-8 — {exc}"))
    return read_manifest_text(text, source=str(path))


def _section_to_dict(values: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Known fields in declaration order, then unknown fields key-sorted."""
    out = dict(values)
    for key in sorted(extra):
        if key not in out:
            out[key] = extra[key]
    return out


def manifest_to_dict(manifest: Manifest) -> dict[str, Any]:
    """Render a manifest as a JSON-ready dict in a stable key order.

    Known keys come out in the declaration order of `data-model.md`; unknown
    keys preserved from a newer writer follow, sorted. The result is a pure
    function of the manifest, so two writes of equal manifests produce
    byte-identical files and manifests diff cleanly.
    """
    prov = manifest.provenance
    runtime = manifest.runtime
    contract = manifest.feature_contract
    training = manifest.training

    body: dict[str, Any] = {
        "schema_version": manifest.schema_version,
        "name": manifest.name,
        "version": manifest.version,
        "created_at": manifest.created_at,
        "provenance": _section_to_dict(
            {
                "base_version": prov.base_version,
                "base_sha256": prov.base_sha256,
                "n_local_extensions": prov.n_local_extensions,
                "training_source": prov.training_source,
                "reset_reason": prov.reset_reason,
            },
            prov.extra,
        ),
        "runtime": _section_to_dict(
            {
                "estimator": runtime.estimator,
                "sklearn_version": runtime.sklearn_version,
                "python_version": runtime.python_version,
            },
            runtime.extra,
        ),
        "feature_contract": _section_to_dict(
            {
                "service": contract.service,
                "service_version": contract.service_version,
                # list(), not sorted() — order is the vector layout.
                "names": list(contract.names),
                "dtypes": list(contract.dtypes),
            },
            contract.extra,
        ),
        "training": _section_to_dict(
            {
                "n_samples": training.n_samples,
                "retained_generation": training.retained_generation,
                "as_of_ms": training.as_of_ms,
            },
            training.extra,
        ),
        "metrics": dict(manifest.metrics),
        "artifact_sha256": manifest.artifact_sha256,
    }
    # FR-011: emitted only for a directory artifact, so every existing (file)
    # manifest serializes byte-for-byte as before.
    if manifest.artifact_kind != ARTIFACT_KIND_FILE:
        body["artifact_kind"] = manifest.artifact_kind
        if manifest.artifact_members:
            body["artifact_members"] = {k: manifest.artifact_members[k] for k in sorted(manifest.artifact_members)}
    for key in sorted(manifest.extra):
        if key not in body:
            body[key] = manifest.extra[key]
    return body


def write_manifest(path: Path | str, manifest: Manifest) -> Path:
    """Write a manifest deterministically, replacing any existing file.

    Written via a temporary file in the same directory and an atomic
    ``os.replace``, so an interrupted write cannot leave a half-written
    manifest that the next read would report as corrupt.

    Note this function *does* raise on an OS-level failure. It is a writer, not
    a validation path: FR-017 governs what happens when an artifact is
    unusable, and a caller who cannot write to its own model directory has a
    problem no fallback slot can answer.
    """
    path = Path(path)
    payload = json.dumps(manifest_to_dict(manifest), indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# T002 — integrity, before deserialization
# ---------------------------------------------------------------------------


def _digest_bytes(payload: bytes) -> str:
    """SHA-256 over ``payload``, fed in chunks rather than one buffer-wide call.

    Chunks are taken through a ``memoryview``, so the incremental feed costs no
    copy of the artifact — the bytes are read once and hashed in place.
    """
    digest = hashlib.sha256()
    view = memoryview(payload)
    for start in range(0, len(view), _DIGEST_CHUNK_BYTES):
        digest.update(view[start : start + _DIGEST_CHUNK_BYTES])
    view.release()
    return digest.hexdigest()


def verify_artifact_bytes(payload: bytes, manifest: Manifest) -> ValidationOutcome:
    """Verify raw artifact bytes against the manifest digest (FR-005).

    Takes bytes and returns those same bytes wrapped in a
    :class:`VerifiedArtifact`, or a refusal. It never takes an already-loaded
    model, because a checksum verified after ``joblib.load`` protects nothing —
    the arbitrary code in the pickle has already run by then.

    A mismatch names both digests in the detail so an operator can tell a
    truncated download from a substituted file.
    """
    expected = (manifest.artifact_sha256 or "").strip().lower()
    if not expected:
        return ValidationOutcome(
            refusal=Refusal(
                CHECK_INTEGRITY,
                "no_expected_digest",
                f"{manifest.name}: manifest records no artifact_sha256, so the artifact cannot be verified",
            )
        )

    actual = _digest_bytes(payload)
    if actual != expected:
        return ValidationOutcome(
            refusal=Refusal(
                CHECK_INTEGRITY,
                "digest_mismatch",
                (
                    f"{manifest.name}: artifact bytes do not match the manifest — "
                    f"expected sha256 {expected}, computed {actual} over {len(payload)} bytes; "
                    "the artifact was not deserialized"
                ),
            )
        )
    return ValidationOutcome(verified=VerifiedArtifact(_VERIFIED_TOKEN, manifest, payload, actual))


def verify_artifact_file(path: Path | str, manifest: Manifest) -> ValidationOutcome:
    """Read an artifact once and verify it (FR-005).

    The file is read a single time and the digest is computed over that buffer,
    which is then carried into the :class:`VerifiedArtifact`. Reading twice —
    once to hash, once to load — would be both slower and unsound, since the
    file could change between the two reads.
    """
    path = Path(path)
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        return ValidationOutcome(refusal=Refusal(CHECK_INTEGRITY, "not_found", f"{path}: artifact file is missing"))
    except OSError as exc:
        return ValidationOutcome(
            refusal=Refusal(CHECK_INTEGRITY, "unreadable", f"{path}: artifact cannot be read — {exc}")
        )
    return verify_artifact_bytes(payload, manifest)


def deserialize_verified(verified: VerifiedArtifact) -> LoadOutcome:
    """Deserialize an artifact that has already passed integrity verification.

    This is the only deserializing function in the registry, and its only
    parameter is a :class:`VerifiedArtifact` — a type nothing outside this
    module can construct. Verification is therefore not a convention a caller
    may forget; it is the only route to a value of the type this function
    accepts. There is no overload taking a path, and none taking bytes.

    Returns a :class:`LoadOutcome` carrying the loaded object, or a refusal if
    the (verified) bytes still fail to deserialize — an artifact written by an
    incompatible pickle protocol, say.
    """
    if not isinstance(verified, VerifiedArtifact):
        # A defect, not a data condition: the type system is the guarantee here.
        raise TypeError(
            "deserialize_verified() requires a VerifiedArtifact from verify_artifact_bytes()/"
            f"verify_artifact_file(), not {type(verified).__name__}"
        )

    import io

    import joblib

    try:
        return LoadOutcome(model=joblib.load(io.BytesIO(verified.payload)))
    except Exception as exc:
        return LoadOutcome(
            refusal=Refusal(
                CHECK_INTEGRITY,
                "undeserializable",
                (
                    f"{verified.manifest.name}: artifact digest {verified.digest} matched the manifest but the "
                    f"bytes could not be deserialized — {type(exc).__name__}: {exc}"
                ),
            )
        )


# ---------------------------------------------------------------------------
# T003 — ordered feature-contract validation
# ---------------------------------------------------------------------------


def local_feature_contract(model_name: str) -> FeatureContract | None:
    """Return the contract the local install declares for ``model_name``.

    Sourced from the Feast feature service registered in
    ``kenaz_ml.feature_store.definitions``, whose schema is itself built by
    iterating the ``FEATURE_NAMES`` constant that lives beside the model. That
    chain is what keeps validation and vector construction from disagreeing:
    the names checked here and the names the trainer indexes are the same list
    object, not two copies of it.

    ``service_version`` is
    :func:`kenaz_ml.feature_store.materialize.feature_service_version` — a hash
    over the service name and its ordered feature references, so it changes
    exactly when the contract does.

    Order comes from ``FeatureViewProjection.features``. **Never**
    ``FeatureView.schema``, which is implemented as ``list(set(...))`` and
    returns fields in arbitrary order; an ordering assertion against it passes
    or fails by luck.

    Returns ``None`` when no feature service is registered for the model. Most
    of the model roster is in that state today — ``workflow``, ``activity``,
    ``quality`` and the ``fleet_*`` families declare no ``FEATURE_NAMES``
    constant, so there is no contract to derive and nothing to validate
    against. That is a legitimate state, not an error, and it is reported as
    ``None`` rather than a raised ``KeyError``.
    """
    from kenaz_ml.feature_store import definitions
    from kenaz_ml.feature_store.materialize import feature_service_version

    service = definitions.FEATURE_SERVICES.get(model_name)
    if service is None:
        return None

    names: list[str] = []
    dtypes: list[str] = []
    for projection in service.feature_view_projections:
        for feature in projection.features:
            names.append(feature.name)
            dtypes.append(str(feature.dtype).lower())

    return FeatureContract(
        service=service.name,
        service_version=feature_service_version(service),
        names=tuple(names),
        dtypes=tuple(dtypes),
    )


def _ordered_name_diagnostic(expected: tuple[str, ...], actual: tuple[str, ...]) -> str:
    """Describe how two ordered name lists disagree, positionally.

    Names the missing names, the unexpected ones, and — the case a set
    comparison cannot see — the first index at which the order diverges.
    """
    parts: list[str] = []
    expected_set = set(expected)
    actual_set = set(actual)

    missing = [n for n in expected if n not in actual_set]
    if missing:
        parts.append(f"missing {missing}")
    unexpected = [n for n in actual if n not in expected_set]
    if unexpected:
        parts.append(f"unexpected {unexpected}")

    for index, (want, have) in enumerate(zip(expected, actual)):
        if want != have:
            parts.append(f"order diverges at index {index}: expected {want!r}, manifest has {have!r}")
            break
    else:
        if len(expected) != len(actual):
            parts.append(f"length differs: expected {len(expected)} features, manifest has {len(actual)}")

    if not missing and not unexpected and expected_set == actual_set:
        parts.append(
            "the same names appear in both, in a different order — the order is the vector layout, "
            "so this would feed every value into the wrong position"
        )

    parts.append(f"expected order {list(expected)}, manifest order {list(actual)}")
    return "; ".join(parts)


def validate_feature_contract(manifest: Manifest, expected: FeatureContract | None = None) -> ValidationOutcome:
    """Validate the manifest's feature contract against the local one (FR-006).

    ``service`` and ``service_version`` are the exact identity of the contract
    and are the primary check: the version is a hash over the service's ordered
    feature references, so it moves whenever a feature is added, removed,
    renamed or **reordered**. ``names`` and ``dtypes`` are compared too — they
    are what makes a refusal actionable, and they also catch a hand-edited
    manifest whose recorded version no longer describes its own name list.

    The name comparison is element-by-element, in order (D-006). A set
    comparison would accept a permutation of the same names, and both trainers
    build their vectors positionally, so a permutation silently feeds every
    value into the wrong slot with no error raised anywhere.

    Fails closed. Any difference is a refusal, and so is the absence of a local
    contract to compare against: a contract that cannot be validated has not
    been validated, and FR-006 requires validation *before* the model is used.
    Resolution then falls through to the next slot (FR-004), which for a model
    with no registered feature service means today's behavior, unchanged.

    Args:
        manifest: The manifest under validation.
        expected: The local contract. Defaults to
            :func:`local_feature_contract` for ``manifest.name``.
    """
    if expected is None:
        expected = local_feature_contract(manifest.name)

    if expected is None:
        return ValidationOutcome(
            refusal=Refusal(
                CHECK_CONTRACT,
                "contract_unregistered",
                (
                    f"{manifest.name}: no feature service is registered locally for this model, so its recorded "
                    "feature contract cannot be validated; a registry-managed artifact is only servable once "
                    "kenaz_ml.feature_store.definitions.FEATURE_SERVICES declares its contract"
                ),
            )
        )

    recorded = manifest.feature_contract

    if recorded.service != expected.service:
        return ValidationOutcome(
            refusal=Refusal(
                CHECK_CONTRACT,
                "service_mismatch",
                (
                    f"{manifest.name}: manifest names feature service {recorded.service!r}, "
                    f"local install declares {expected.service!r}"
                ),
            )
        )

    names_agree = tuple(recorded.names) == tuple(expected.names)

    if recorded.service_version != expected.service_version:
        return ValidationOutcome(
            refusal=Refusal(
                CHECK_CONTRACT,
                "service_version_mismatch",
                (
                    f"{manifest.name}: feature service {expected.service!r} version differs — manifest records "
                    f"{recorded.service_version!r}, local install computes {expected.service_version!r}"
                    + ("" if names_agree else "; " + _ordered_name_diagnostic(expected.names, tuple(recorded.names)))
                ),
            )
        )

    if not names_agree:
        return ValidationOutcome(
            refusal=Refusal(
                CHECK_CONTRACT,
                "feature_names_mismatch",
                (
                    f"{manifest.name}: recorded feature names disagree with the local contract — "
                    + _ordered_name_diagnostic(expected.names, tuple(recorded.names))
                ),
            )
        )

    if tuple(recorded.dtypes) != tuple(expected.dtypes):
        for index, (want, have) in enumerate(zip(expected.dtypes, tuple(recorded.dtypes))):
            if want != have:
                where = f"at index {index} ({expected.names[index]!r}): expected {want!r}, manifest has {have!r}"
                break
        else:
            where = f"expected {len(expected.dtypes)} dtypes, manifest has {len(recorded.dtypes)}"
        return ValidationOutcome(
            refusal=Refusal(
                CHECK_CONTRACT,
                "feature_dtypes_mismatch",
                f"{manifest.name}: recorded feature dtypes disagree with the local contract — {where}",
            )
        )

    return ValidationOutcome()


# ---------------------------------------------------------------------------
# T004 — runtime compatibility
# ---------------------------------------------------------------------------


def _major_minor(version: str) -> tuple[str, str] | None:
    parts = (version or "").strip().split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return None
    return parts[0], parts[1]


def running_sklearn_version() -> str | None:
    """The scikit-learn version in this process, or ``None`` if unimportable."""
    try:
        import sklearn
    except Exception:  # noqa: BLE001 - a missing or broken sklearn is a data condition here
        return None
    return str(getattr(sklearn, "__version__", "") or "") or None


def validate_runtime(manifest: Manifest, running_version: str | None = None) -> ValidationOutcome:
    """Check the manifest's serialization runtime against this one (FR-008).

    **The compatibility rule**: identical ``major.minor`` passes; any different
    ``major.minor`` refuses. The patch component is ignored — scikit-learn's
    own pickle-compatibility warning is keyed to major.minor, and patch releases
    do not change estimator attribute layouts. A version that cannot be parsed
    into ``major.minor``, or a manifest that records no runtime version at all,
    refuses: an unknown runtime is not a compatible one, and this check exists
    precisely to turn an unpickling traceback into a diagnostic.

    **Why this matters asymmetrically.** The frozen PyInstaller bundle pins the
    exact scikit-learn it was built with, so a base artifact built against that
    pin will always match there and this check will never fire. A source
    install resolves whatever satisfies ``scikit-learn>=1.4``, which today may
    be any version from 1.4 to whatever ships next — so the same shipped base
    artifact meets a runtime nobody chose. That is where the check earns its
    keep, and it is why the rule is stated on the manifest side (what
    serialized it) rather than assumed from the bundle.

    Args:
        manifest: The manifest under validation.
        running_version: Override for the running scikit-learn version. Present
            for tests; production callers leave it unset.
    """
    recorded = (manifest.runtime.sklearn_version or "").strip()
    if not recorded:
        return ValidationOutcome(
            refusal=Refusal(
                CHECK_RUNTIME,
                "no_recorded_version",
                (
                    f"{manifest.name}: manifest records no runtime.sklearn_version, so serialization "
                    "compatibility cannot be established"
                ),
            )
        )

    if running_version is None:
        running_version = running_sklearn_version()
    if not running_version:
        return ValidationOutcome(
            refusal=Refusal(
                CHECK_RUNTIME,
                "runtime_unavailable",
                (
                    f"{manifest.name}: scikit-learn is not importable in this process, so the artifact "
                    f"serialized by scikit-learn {recorded} cannot be checked for compatibility"
                ),
            )
        )

    recorded_mm = _major_minor(recorded)
    running_mm = _major_minor(running_version)
    if recorded_mm is None or running_mm is None:
        unparseable = recorded if recorded_mm is None else running_version
        return ValidationOutcome(
            refusal=Refusal(
                CHECK_RUNTIME,
                "unparseable_version",
                (
                    f"{manifest.name}: cannot compare scikit-learn versions — {unparseable!r} is not "
                    f"major.minor; manifest records {recorded!r}, this process runs {running_version!r}"
                ),
            )
        )

    if recorded_mm != running_mm:
        return ValidationOutcome(
            refusal=Refusal(
                CHECK_RUNTIME,
                "sklearn_version_incompatible",
                (
                    f"{manifest.name}: artifact was serialized by scikit-learn {recorded}, this process runs "
                    f"{running_version}; compatibility requires an identical major.minor "
                    f"({'.'.join(recorded_mm)} vs {'.'.join(running_mm)})"
                ),
            )
        )

    return ValidationOutcome()


# ---------------------------------------------------------------------------
# Directory artifacts (FR-011)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectoryDigest:
    """The digest of a directory and the per-member hashes it was built from."""

    digest: str
    members: dict[str, str]


class DirectoryWalkError(Exception):
    """Internal: a directory could not be digested (carries a :class:`Refusal`)."""

    def __init__(self, refusal: Refusal) -> None:
        super().__init__(str(refusal))
        self.refusal = refusal


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_DIGEST_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


#: Prefix of a tree digest. Part of the canonical recipe, not decoration: the
#: Go clients write and compare the prefixed string.
TREE_DIGEST_PREFIX = "sha256:"


def _path_bytes(rel: str) -> bytes:
    # surrogateescape round-trips a non-UTF-8 POSIX name to its original bytes,
    # so the sort below is the Go side's bytewise sort and encoding never raises.
    return rel.encode("utf-8", "surrogateescape")


def tree_digest(members: dict[str, str]) -> str:
    """The canonical tree digest of ``{relative posix path: member hash}``.

    **Byte-for-byte the cross-repo recipe** ruled canonical in design-doc
    Amendment A5 addendum (b) -- Kenaz ``internal/ml/tree.go`` and the harness's
    ``core/mlsidecar/tree.go`` compute the same value, and
    ``tests/test_org_slot_shape.py`` pins their shared vector::

        manifest = concat over members sorted BYTEWISE by path of
                   "<member hash>  <path>\\n"      (sha256sum format, two spaces)
        digest   = "sha256:" + hex(sha256(manifest))

    One recipe, three implementations: a pack digest the client's Go updater
    records must be the value this registry recomputes, or every real pack is
    refused as a digest mismatch.
    """
    ordered = sorted(members, key=_path_bytes)
    text = b"".join(members[rel].encode("ascii") + b"  " + _path_bytes(rel) + b"\n" for rel in ordered)
    return TREE_DIGEST_PREFIX + hashlib.sha256(text).hexdigest()


def directory_digest(root: Path | str, *, allow_symlinks: bool = False) -> DirectoryDigest:
    """The canonical tree digest of a directory (:func:`tree_digest`) and its per-member hashes.

    Members are every non-directory entry, addressed by POSIX path relative to
    ``root``; directories themselves (so empty ones) do not participate. A
    regular file's member hash is ``hex(sha256(bytes))``; anything that is
    neither a file, a directory nor a symlink is refused.

    **Symlinks.** The canonical recipe hashes a symlink member as
    ``hex(sha256("symlink:" + link target))`` -- the engine onedir the Go
    clients digest legitimately contains them. A **directory artifact** (a laya
    checkpoint pack) is held to a stricter *policy* on top of the same recipe:
    with ``allow_symlinks=False`` -- the default, and the only mode the registry
    uses -- **any symlink member is refused** (``symlink_member``), because the
    registry hands laya a path, and following a link is how a verified
    directory stops being the thing that gets read. For a symlink-free tree the
    two modes produce the identical digest, so the policy never forks the recipe.

    Raises :class:`DirectoryWalkError`; callers wrapping it
    (:func:`verify_artifact_directory`) turn that into a :class:`Refusal`.
    """
    root = Path(root)
    members: dict[str, str] = {}
    try:
        for path in root.rglob("*"):
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                if not allow_symlinks:
                    raise DirectoryWalkError(
                        Refusal(
                            CHECK_INTEGRITY,
                            "symlink_member",
                            f"{path}: symlinks are not allowed inside a directory artifact",
                        )
                    )
                target = os.readlink(path).encode("utf-8", "surrogateescape")
                members[rel] = hashlib.sha256(b"symlink:" + target).hexdigest()
                continue
            if path.is_dir():
                continue
            if not path.is_file():
                raise DirectoryWalkError(
                    Refusal(CHECK_INTEGRITY, "not_a_regular_file", f"{path}: not a regular file or directory")
                )
            members[rel] = _hash_file(path)
    except OSError as exc:
        raise DirectoryWalkError(
            Refusal(CHECK_INTEGRITY, "unreadable", f"{root}: directory artifact cannot be read — {exc}")
        ) from exc
    return DirectoryDigest(tree_digest(members), members)


class VerifiedDirectory:
    """A directory whose :func:`directory_digest` matched the manifest.

    Like :class:`VerifiedArtifact`, only the verification functions can build
    one. It carries the **path** -- the registry never deserializes a directory
    artifact; laya loads its own checkpoint, and is handed :attr:`members` to
    re-verify each file as it parses it.
    """

    __slots__ = ("digest", "manifest", "members", "path")

    def __init__(self, token: object, manifest: Manifest, path: Path, digest: str, members: dict[str, str]) -> None:
        if token is not _VERIFIED_TOKEN:
            raise TypeError("VerifiedDirectory cannot be constructed directly; use verify_artifact_directory()")
        self.manifest = manifest
        self.path = path
        self.digest = digest
        self.members = members

    def __repr__(self) -> str:
        return f"VerifiedDirectory(name={self.manifest.name!r}, path={str(self.path)!r}, digest={self.digest[:12]}…)"


def _member_diagnostic(declared: dict[str, str], actual: dict[str, str]) -> str | None:
    """Name the member that disagrees with ``artifact_members``, or ``None`` if they agree."""
    missing = sorted(set(declared) - set(actual))
    extra = sorted(set(actual) - set(declared))
    changed = sorted(k for k in set(declared) & set(actual) if declared[k] != actual[k])
    parts = []
    if missing:
        parts.append(f"missing member(s) {missing}")
    if extra:
        parts.append(f"unexpected extra member(s) {extra}")
    if changed:
        parts.append(f"member(s) whose sha256 differs {changed}")
    return "; ".join(parts) or None


def verify_artifact_directory(path: Path | str, manifest: Manifest) -> tuple[VerifiedDirectory | None, Refusal | None]:
    """Verify a directory artifact against the manifest; exactly one of the pair is set. Never raises.

    Integrity only (the contract is :func:`validate_artifact_directory`'s next
    step). A refusal names the failing member when the manifest declares
    ``artifact_members``; otherwise both whole-directory digests.
    """
    path = Path(path)
    expected = (manifest.artifact_sha256 or "").strip().lower()
    if not expected:
        return None, Refusal(
            CHECK_INTEGRITY,
            "no_expected_digest",
            f"{manifest.name}: manifest records no artifact_sha256, so the directory cannot be verified",
        )
    if path.is_symlink():
        return None, Refusal(CHECK_INTEGRITY, "symlink_root", f"{path}: the artifact directory itself is a symlink")
    if not path.is_dir():
        return None, Refusal(CHECK_INTEGRITY, "not_found", f"{path}: artifact directory is missing")
    try:
        computed = directory_digest(path)
    except DirectoryWalkError as exc:
        return None, exc.refusal
    if manifest.artifact_members:
        diagnostic = _member_diagnostic(manifest.artifact_members, computed.members)
        if diagnostic is not None:
            return None, Refusal(
                CHECK_INTEGRITY,
                "member_mismatch",
                f"{manifest.name}: directory does not match the manifest's members — {diagnostic}; "
                "the checkpoint was not handed to laya",
            )
    # Compared the way the Go clients compare (``digestsEqual``): case-insensitive,
    # the "sha256:" prefix optional on the recorded side.
    if computed.digest.removeprefix(TREE_DIGEST_PREFIX) != expected.removeprefix(TREE_DIGEST_PREFIX):
        return None, Refusal(
            CHECK_INTEGRITY,
            "digest_mismatch",
            f"{manifest.name}: directory does not match the manifest — expected sha256 {expected}, "
            f"computed {computed.digest} over {len(computed.members)} member(s); "
            "the checkpoint was not handed to laya",
        )
    return VerifiedDirectory(_VERIFIED_TOKEN, manifest, path, computed.digest, computed.members), None


def validate_artifact_directory(
    manifest: Manifest,
    artifact_dir: Path | str,
    *,
    expected_contract: FeatureContract | None = None,
) -> tuple[VerifiedDirectory | None, Refusal | None]:
    """Integrity, then the ordered feature contract, for a directory artifact.

    No runtime check: :func:`validate_runtime` pins a scikit-learn version, and a
    laya checkpoint is not a scikit-learn artifact. Never raises.
    """
    verified, refusal = verify_artifact_directory(artifact_dir, manifest)
    if refusal is not None or verified is None:
        return None, refusal
    contract = validate_feature_contract(manifest, expected_contract)
    if not contract.ok:
        return None, contract.refusal
    return verified, None


# ---------------------------------------------------------------------------
# The composed check
# ---------------------------------------------------------------------------


def validate_artifact(
    manifest: Manifest,
    artifact_path: Path | str,
    *,
    expected_contract: FeatureContract | None = None,
    running_version: str | None = None,
) -> ValidationOutcome:
    """Run the three checks in order, short-circuiting (data-model.md §Resolution).

    Integrity first, then contract, then runtime. On success the outcome
    carries the :class:`VerifiedArtifact` — so a caller that wants a model
    object gets it by passing that handle to :func:`deserialize_verified`, and
    has no path to a model object that did not go through the integrity check.

    Never raises. A refusal names the check that failed and what disagreed
    (FR-006, FR-017).
    """
    integrity = verify_artifact_file(artifact_path, manifest)
    if not integrity.ok:
        return integrity

    contract = validate_feature_contract(manifest, expected_contract)
    if not contract.ok:
        return contract

    runtime = validate_runtime(manifest, running_version)
    if not runtime.ok:
        return runtime

    return integrity
