"""Tests for the model-artifact manifest and its three load-time checks.

Four behaviors are proved here, and two of them are the reason the module
exists:

* Integrity verification **precedes** deserialization structurally, not by
  convention (FR-005). It is not enough to assert that a tampered artifact is
  refused — an implementation that hashed *after* ``joblib.load`` would also
  pass that assertion, having already executed the attacker's pickle. So the
  tests below also assert that ``joblib.load`` is never reached, that the type
  ``deserialize_verified`` requires cannot be forged, and that the module
  contains exactly one call to ``joblib.load``.

* Contract comparison is **ordered** (FR-006, D-006). The permutation test is
  the most important one in this file: it feeds a manifest whose feature names
  are the same *set* in a different order, asserts the refusal, and asserts in
  the same test that ``set(a) == set(b)`` is true for those inputs — so a
  set-based implementation would have accepted exactly this case, and the test
  would fail.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path

import joblib
import pytest

from kenaz_ml.models.duration import FEATURE_NAMES as DURATION_FEATURE_NAMES
from kenaz_ml.models.stuck import FEATURE_NAMES as STUCK_FEATURE_NAMES
from kenaz_ml.modelstore.registry import (
    CHECK_CONTRACT,
    CHECK_INTEGRITY,
    CHECK_MANIFEST,
    CHECK_RUNTIME,
    SCHEMA_VERSION,
    FeatureContract,
    Manifest,
    Provenance,
    Refusal,
    Runtime,
    Training,
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
from kenaz_ml.modelstore.registry import manifest as manifest_module

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

STUCK_CONTRACT = FeatureContract(
    service="stuck",
    service_version="deadbeefdeadbeef",
    names=tuple(STUCK_FEATURE_NAMES),
    dtypes=tuple("float64" for _ in STUCK_FEATURE_NAMES),
)


def _artifact(tmp_path: Path, payload: object = None) -> tuple[Path, str]:
    """Write a real joblib artifact and return its path and SHA-256."""
    path = tmp_path / "stuck.joblib"
    joblib.dump(payload if payload is not None else {"weights": [1.0, 2.0, 3.0]}, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(
    digest: str,
    *,
    contract: FeatureContract | None = None,
    sklearn_version: str | None = None,
) -> Manifest:
    if sklearn_version is None:
        sklearn_version = running_sklearn_version() or "1.5.2"
    return Manifest(
        name="stuck",
        version="4",
        artifact_sha256=digest,
        created_at=1753900000000,
        provenance=Provenance(
            base_version="1",
            base_sha256="9f2c",
            n_local_extensions=3,
            training_source="local",
        ),
        runtime=Runtime(
            estimator="GradientBoostingClassifier",
            sklearn_version=sklearn_version,
            python_version="3.12",
        ),
        feature_contract=contract if contract is not None else STUCK_CONTRACT,
        training=Training(n_samples=847, retained_generation="2", as_of_ms=1753900000000),
        metrics={"accuracy": 0.81, "n_holdout": 120},
    )


@pytest.fixture
def exploding_joblib(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Make any ``joblib.load`` call fail the test loudly, and record attempts.

    This is the assertion that distinguishes "verified before deserializing"
    from "refused, eventually". If any code path reaches deserialization while
    a check is meant to be refusing, the test fails here rather than passing on
    the refusal alone.
    """
    calls: list[object] = []

    def _boom(*args: object, **kwargs: object) -> object:
        calls.append(args)
        raise AssertionError("joblib.load was reached — deserialization happened before/despite verification")

    monkeypatch.setattr(joblib, "load", _boom)
    return calls


# ---------------------------------------------------------------------------
# T001 — schema, read/write
# ---------------------------------------------------------------------------


class TestManifestRoundTrip:
    def test_round_trips_without_loss(self, tmp_path: Path) -> None:
        original = _manifest("a" * 64)
        path = write_manifest(tmp_path / "stuck.json", original)

        read = read_manifest(path)
        assert read.ok
        assert read.manifest == original

    def test_written_json_matches_the_documented_shape(self, tmp_path: Path) -> None:
        path = write_manifest(tmp_path / "stuck.json", _manifest("a" * 64))
        body = json.loads(path.read_text())

        assert body["schema_version"] == SCHEMA_VERSION
        assert body["name"] == "stuck"
        assert body["provenance"]["training_source"] == "local"
        assert body["runtime"]["sklearn_version"]
        assert body["feature_contract"]["names"] == list(STUCK_FEATURE_NAMES)
        assert body["feature_contract"]["service"] == "stuck"
        assert body["artifact_sha256"] == "a" * 64

    def test_writes_are_deterministic(self, tmp_path: Path) -> None:
        """Stable key order, so manifests diff cleanly."""
        manifest = _manifest("a" * 64)
        first = write_manifest(tmp_path / "one.json", manifest).read_bytes()
        second = write_manifest(tmp_path / "two.json", manifest).read_bytes()
        assert first == second

        keys = list(json.loads(first.decode()))
        assert keys == [
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
        ]

    def test_write_leaves_no_temporary_file(self, tmp_path: Path) -> None:
        write_manifest(tmp_path / "stuck.json", _manifest("a" * 64))
        assert [p.name for p in tmp_path.iterdir()] == ["stuck.json"]

    def test_feature_name_order_survives_the_round_trip(self, tmp_path: Path) -> None:
        """Order is the vector layout; a writer that sorted would corrupt it."""
        reversed_names = tuple(reversed(STUCK_FEATURE_NAMES))
        contract = FeatureContract(
            service="stuck",
            service_version="v",
            names=reversed_names,
            dtypes=tuple("float64" for _ in reversed_names),
        )
        path = write_manifest(tmp_path / "stuck.json", _manifest("a" * 64, contract=contract))

        assert json.loads(path.read_text())["feature_contract"]["names"] == list(reversed_names)
        assert read_manifest(path).manifest.feature_contract.names == reversed_names


class TestManifestTolerance:
    def test_unknown_fields_are_preserved(self, tmp_path: Path) -> None:
        """Forward compatibility with a newer cloud exporter (C-002)."""
        raw = manifest_to_dict(_manifest("a" * 64))
        raw["signature"] = {"alg": "ed25519", "sig": "…"}
        raw["provenance"]["mlflow_run_id"] = "abc123"
        raw["feature_contract"]["transform_version"] = "7"
        path = tmp_path / "stuck.json"
        path.write_text(json.dumps(raw))

        read = read_manifest(path)
        assert read.ok
        assert read.manifest.extra["signature"] == {"alg": "ed25519", "sig": "…"}
        assert read.manifest.provenance.extra["mlflow_run_id"] == "abc123"

        rewritten = json.loads(write_manifest(tmp_path / "again.json", read.manifest).read_text())
        assert rewritten["signature"] == {"alg": "ed25519", "sig": "…"}
        assert rewritten["provenance"]["mlflow_run_id"] == "abc123"
        assert rewritten["feature_contract"]["transform_version"] == "7"

    def test_missing_optional_fields_parse(self) -> None:
        read = manifest_from_dict(
            {
                "schema_version": "1",
                "name": "stuck",
                "version": "1",
                "artifact_sha256": "a" * 64,
                "provenance": {"training_source": "base"},
                "runtime": {"sklearn_version": "1.5.2"},
                "feature_contract": {"service": "stuck", "names": [], "dtypes": []},
            }
        )
        assert read.ok
        assert read.manifest.provenance.reset_reason is None
        assert read.manifest.training.retained_generation is None
        assert read.manifest.metrics == {}
        assert read.manifest.created_at is None

    def test_base_shaped_manifest_parses(self) -> None:
        """A base manifest omits base_version/base_sha256 entirely."""
        read = manifest_from_dict(
            {
                "name": "stuck",
                "version": "1",
                "artifact_sha256": "a" * 64,
                "provenance": {"n_local_extensions": 0, "training_source": "base"},
            }
        )
        assert read.ok
        assert read.manifest.provenance.base_version is None
        assert read.manifest.provenance.base_sha256 is None
        assert read.manifest.provenance.is_base

    def test_corrupt_json_returns_unusable_rather_than_raising(self, tmp_path: Path) -> None:
        path = tmp_path / "stuck.json"
        path.write_text('{"name": "stuck", "version":')

        read = read_manifest(path)
        assert not read.ok
        assert read.manifest is None
        assert read.refusal.check == CHECK_MANIFEST
        assert read.refusal.reason == "unparseable"
        assert "stuck.json" in read.refusal.detail

    def test_missing_manifest_returns_unusable(self, tmp_path: Path) -> None:
        read = read_manifest(tmp_path / "absent.json")
        assert not read.ok
        assert read.refusal.reason == "not_found"

    def test_non_object_json_returns_unusable(self) -> None:
        read = read_manifest_text("[1, 2, 3]")
        assert not read.ok
        assert read.refusal.reason == "not_an_object"

    @pytest.mark.parametrize("dropped", ["name", "version", "artifact_sha256"])
    def test_missing_required_field_returns_unusable(self, dropped: str) -> None:
        raw = {"name": "stuck", "version": "1", "artifact_sha256": "a" * 64}
        del raw[dropped]
        read = manifest_from_dict(raw)
        assert not read.ok
        assert read.refusal.reason == "missing_required_field"
        assert dropped in read.refusal.detail

    def test_manifest_module_imports_only_the_standard_library(self) -> None:
        """C-001: no new dependency creeps in for schema handling."""
        tree = inspect.getsource(manifest_module)
        module_level = [
            line
            for line in tree.splitlines()
            if (line.startswith("import ") or line.startswith("from ")) and "kenaz_ml" not in line
        ]
        assert module_level == [
            "from __future__ import annotations",
            "import hashlib",
            "import json",
            "import os",
            "from dataclasses import dataclass, field",
            "from pathlib import Path",
            "from typing import Any",
        ]


# ---------------------------------------------------------------------------
# T002 — integrity before deserialization
# ---------------------------------------------------------------------------


class TestIntegrity:
    def test_matching_digest_yields_a_verified_artifact(self, tmp_path: Path) -> None:
        path, digest = _artifact(tmp_path)
        outcome = verify_artifact_file(path, _manifest(digest))

        assert outcome.ok
        assert outcome.refusal is None
        assert outcome.verified.digest == digest
        assert len(outcome.verified) == path.stat().st_size

    def test_one_tampered_byte_is_refused_and_nothing_is_deserialized(
        self, tmp_path: Path, exploding_joblib: list[object]
    ) -> None:
        path, digest = _artifact(tmp_path)
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 0x01
        path.write_bytes(bytes(raw))

        outcome = validate_artifact(_manifest(digest), path)

        assert not outcome.ok
        assert outcome.verified is None
        assert outcome.refusal.check == CHECK_INTEGRITY
        assert outcome.refusal.reason == "digest_mismatch"
        # Both digests, so an operator can tell truncation from substitution.
        assert digest in outcome.refusal.detail
        assert hashlib.sha256(bytes(raw)).hexdigest() in outcome.refusal.detail
        # The refusal alone would not prove ordering. This does.
        assert exploding_joblib == []

    def test_missing_artifact_is_refused(self, tmp_path: Path) -> None:
        outcome = verify_artifact_file(tmp_path / "absent.joblib", _manifest("a" * 64))
        assert not outcome.ok
        assert outcome.refusal.reason == "not_found"

    def test_manifest_without_a_digest_is_refused(self, tmp_path: Path) -> None:
        path, digest = _artifact(tmp_path)
        manifest = _manifest(digest)
        object.__setattr__(manifest, "artifact_sha256", "")
        outcome = verify_artifact_file(path, manifest)
        assert not outcome.ok
        assert outcome.refusal.reason == "no_expected_digest"

    def test_bytes_and_file_verification_agree(self, tmp_path: Path) -> None:
        path, digest = _artifact(tmp_path)
        manifest = _manifest(digest)
        assert verify_artifact_bytes(path.read_bytes(), manifest).ok
        assert verify_artifact_file(path, manifest).ok

    def test_verified_bytes_deserialize_to_the_original_object(self, tmp_path: Path) -> None:
        path, digest = _artifact(tmp_path, {"weights": [4.0, 5.0]})
        outcome = verify_artifact_file(path, _manifest(digest))
        loaded = deserialize_verified(outcome.verified)
        assert loaded.ok
        assert loaded.model == {"weights": [4.0, 5.0]}

    def test_deserialization_failure_is_a_refusal_not_a_traceback(self, tmp_path: Path) -> None:
        path = tmp_path / "stuck.joblib"
        path.write_bytes(b"this is not a pickle")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        outcome = verify_artifact_file(path, _manifest(digest))
        assert outcome.ok  # the bytes are exactly what the manifest recorded
        loaded = deserialize_verified(outcome.verified)
        assert not loaded.ok
        assert loaded.refusal.reason == "undeserializable"


class TestIntegrityIsUnskippable:
    """Reviewer guidance #1: try to construct a call that skips verification."""

    def test_verified_artifact_cannot_be_constructed_directly(self, tmp_path: Path) -> None:
        path, digest = _artifact(tmp_path)
        with pytest.raises(TypeError, match="cannot be constructed directly"):
            VerifiedArtifact(object(), _manifest(digest), path.read_bytes(), digest)

    @pytest.mark.parametrize("forgery", ["path", "bytes", "none"])
    def test_deserialize_refuses_anything_that_is_not_verified(
        self, tmp_path: Path, forgery: str, exploding_joblib: list[object]
    ) -> None:
        path, _ = _artifact(tmp_path)
        candidate: object = {"path": path, "bytes": path.read_bytes(), "none": None}[forgery]
        with pytest.raises(TypeError, match="requires a VerifiedArtifact"):
            deserialize_verified(candidate)  # type: ignore[arg-type]
        assert exploding_joblib == []

    def test_the_module_deserializes_in_exactly_one_place(self) -> None:
        """Only ``deserialize_verified`` may call ``joblib.load``.

        A second call site would be a second route to arbitrary code execution,
        and the type guarantee would stop being a guarantee.
        """
        source = inspect.getsource(manifest_module)
        assert source.count("joblib.load(") == 1
        assert "joblib.load(" in inspect.getsource(manifest_module.deserialize_verified)

    def test_no_validation_entry_point_accepts_a_loaded_model(self) -> None:
        """Verification takes bytes or a path — never an already-loaded object."""
        for fn in (verify_artifact_bytes, verify_artifact_file, validate_artifact):
            annotations = [p.annotation for p in inspect.signature(fn).parameters.values()]
            assert not any("Any" in str(a) for a in annotations), fn.__name__


class TestIntegrityPerformance:
    def test_fifty_megabytes_verifies_well_under_the_budget(self, tmp_path: Path) -> None:
        """NFR-003: under 200ms for a 50MB artifact.

        The budget is generous against SHA-256 throughput on any machine this
        ships to; the assertion exists to catch an implementation that reads or
        hashes the artifact more than once, which is the failure this NFR is
        really about.
        """
        path = tmp_path / "big.joblib"
        path.write_bytes(b"\xab" * (50 * 1024 * 1024))
        manifest = _manifest(hashlib.sha256(path.read_bytes()).hexdigest())

        start = time.perf_counter()
        outcome = verify_artifact_file(path, manifest)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert outcome.ok
        assert elapsed_ms < 200, f"integrity verification took {elapsed_ms:.1f}ms"


# ---------------------------------------------------------------------------
# T003 — ordered feature-contract validation
# ---------------------------------------------------------------------------


class TestContractSourcedFromFeast:
    """The expected contract is derived from Feast, never retyped (T003 amended)."""

    def test_stuck_contract_comes_from_the_registered_feature_service(self) -> None:
        contract = local_feature_contract("stuck")
        assert contract is not None
        assert contract.service == "stuck"
        # Ordered equality against the constant the trainer indexes positionally.
        assert contract.names == tuple(STUCK_FEATURE_NAMES)
        assert contract.dtypes == tuple("float64" for _ in STUCK_FEATURE_NAMES)

    def test_duration_contract_comes_from_the_registered_feature_service(self) -> None:
        contract = local_feature_contract("duration")
        assert contract is not None
        assert contract.names == tuple(DURATION_FEATURE_NAMES)

    def test_service_version_is_the_feast_content_hash(self) -> None:
        from kenaz_ml.feature_store import definitions
        from kenaz_ml.feature_store.materialize import feature_service_version

        contract = local_feature_contract("stuck")
        assert contract.service_version == feature_service_version(definitions.FEATURE_SERVICES["stuck"])

    def test_models_without_a_feature_service_report_absence_rather_than_raising(self) -> None:
        """`quality`, `workflow`, `activity` and the fleet models are unregistered."""
        for name in ("quality", "workflow", "activity", "fleet_focus", "not_a_model_at_all"):
            assert local_feature_contract(name) is None


class TestContractValidation:
    def _local(self) -> FeatureContract:
        contract = local_feature_contract("stuck")
        assert contract is not None
        return contract

    def test_identical_ordered_contract_passes(self) -> None:
        local = self._local()
        outcome = validate_feature_contract(_manifest("a" * 64, contract=local), local)
        assert outcome.ok
        assert outcome.refusal is None

    def test_the_manifest_default_is_the_live_feast_contract(self) -> None:
        """No `expected` argument: the local contract is looked up for the model."""
        outcome = validate_feature_contract(_manifest("a" * 64, contract=self._local()))
        assert outcome.ok

    def test_reordered_but_identical_set_is_refused(self) -> None:
        """THE test. A set comparison passes here; an ordered one must not.

        Both trainers build vectors as ``[features.get(f, 0.0) for f in names]``,
        so accepting this permutation would feed `edit_velocity` into the
        `time_in_phase_sec` slot for every prediction, silently, forever.
        """
        local = self._local()
        permuted = (local.names[1], local.names[0], *local.names[2:])

        # The premise: these two lists are equal as sets. A set-based
        # implementation would accept the manifest below.
        assert set(permuted) == set(local.names)
        assert permuted != local.names
        assert sorted(permuted) == sorted(local.names)

        recorded = FeatureContract(
            service=local.service,
            service_version=local.service_version,
            names=permuted,
            dtypes=local.dtypes,
        )
        outcome = validate_feature_contract(_manifest("a" * 64, contract=recorded), local)

        assert not outcome.ok
        assert outcome.refusal.check == CHECK_CONTRACT
        assert outcome.refusal.reason == "feature_names_mismatch"
        # The diagnostic names the position, not just "mismatch".
        assert "order diverges at index 0" in outcome.refusal.detail
        assert local.names[0] in outcome.refusal.detail
        assert local.names[1] in outcome.refusal.detail
        assert "different order" in outcome.refusal.detail

    def test_missing_name_is_refused_and_named(self) -> None:
        local = self._local()
        recorded = FeatureContract(
            service=local.service,
            service_version=local.service_version,
            names=local.names[:-1],
            dtypes=local.dtypes[:-1],
        )
        outcome = validate_feature_contract(_manifest("a" * 64, contract=recorded), local)

        assert not outcome.ok
        assert outcome.refusal.reason == "feature_names_mismatch"
        assert f"missing ['{local.names[-1]}']" in outcome.refusal.detail

    def test_extra_name_is_refused_and_named(self) -> None:
        local = self._local()
        recorded = FeatureContract(
            service=local.service,
            service_version=local.service_version,
            names=(*local.names, "phase_of_the_moon"),
            dtypes=(*local.dtypes, "float64"),
        )
        outcome = validate_feature_contract(_manifest("a" * 64, contract=recorded), local)

        assert not outcome.ok
        assert outcome.refusal.reason == "feature_names_mismatch"
        assert "unexpected ['phase_of_the_moon']" in outcome.refusal.detail

    def test_renamed_feature_names_both_sides(self) -> None:
        local = self._local()
        recorded = FeatureContract(
            service=local.service,
            service_version=local.service_version,
            names=("renamed_first", *local.names[1:]),
            dtypes=local.dtypes,
        )
        outcome = validate_feature_contract(_manifest("a" * 64, contract=recorded), local)

        assert not outcome.ok
        assert f"missing ['{local.names[0]}']" in outcome.refusal.detail
        assert "unexpected ['renamed_first']" in outcome.refusal.detail
        assert "order diverges at index 0" in outcome.refusal.detail

    def test_dtype_mismatch_is_refused(self) -> None:
        local = self._local()
        recorded = FeatureContract(
            service=local.service,
            service_version=local.service_version,
            names=local.names,
            dtypes=("int64", *local.dtypes[1:]),
        )
        outcome = validate_feature_contract(_manifest("a" * 64, contract=recorded), local)

        assert not outcome.ok
        assert outcome.refusal.reason == "feature_dtypes_mismatch"
        assert "at index 0" in outcome.refusal.detail
        assert "int64" in outcome.refusal.detail
        assert "float64" in outcome.refusal.detail

    def test_stale_service_version_is_refused(self) -> None:
        """The primary, exact check: the contract hash moved."""
        local = self._local()
        recorded = FeatureContract(
            service=local.service,
            service_version="0000000000000000",
            names=local.names,
            dtypes=local.dtypes,
        )
        outcome = validate_feature_contract(_manifest("a" * 64, contract=recorded), local)

        assert not outcome.ok
        assert outcome.refusal.reason == "service_version_mismatch"
        assert "0000000000000000" in outcome.refusal.detail
        assert local.service_version in outcome.refusal.detail

    def test_wrong_service_is_refused(self) -> None:
        local = self._local()
        recorded = FeatureContract(
            service="duration",
            service_version=local.service_version,
            names=local.names,
            dtypes=local.dtypes,
        )
        outcome = validate_feature_contract(_manifest("a" * 64, contract=recorded), local)

        assert not outcome.ok
        assert outcome.refusal.reason == "service_mismatch"
        assert "duration" in outcome.refusal.detail

    def test_unregistered_model_fails_closed(self) -> None:
        """No local service means the contract cannot be validated — so it isn't."""
        manifest = Manifest(name="quality", version="1", artifact_sha256="a" * 64)
        outcome = validate_feature_contract(manifest)

        assert not outcome.ok
        assert outcome.refusal.check == CHECK_CONTRACT
        assert outcome.refusal.reason == "contract_unregistered"
        assert "quality" in outcome.refusal.detail
        assert "FEATURE_SERVICES" in outcome.refusal.detail

    def test_empty_recorded_contract_is_refused(self) -> None:
        local = self._local()
        outcome = validate_feature_contract(_manifest("a" * 64, contract=FeatureContract()), local)
        assert not outcome.ok
        assert outcome.refusal.reason == "service_mismatch"


# ---------------------------------------------------------------------------
# T004 — runtime compatibility
# ---------------------------------------------------------------------------


class TestRuntimeCompatibility:
    def test_identical_version_passes(self) -> None:
        outcome = validate_runtime(_manifest("a" * 64, sklearn_version="1.5.2"), running_version="1.5.2")
        assert outcome.ok

    def test_differing_patch_passes(self) -> None:
        """Documented rule: major.minor is the unit of compatibility."""
        outcome = validate_runtime(_manifest("a" * 64, sklearn_version="1.5.2"), running_version="1.5.9")
        assert outcome.ok

    @pytest.mark.parametrize("running", ["1.4.2", "1.6.0", "2.0.0"])
    def test_differing_major_minor_is_refused(self, running: str) -> None:
        outcome = validate_runtime(_manifest("a" * 64, sklearn_version="1.5.2"), running_version=running)

        assert not outcome.ok
        assert outcome.refusal.check == CHECK_RUNTIME
        assert outcome.refusal.reason == "sklearn_version_incompatible"
        # Both versions, per T004.
        assert "1.5.2" in outcome.refusal.detail
        assert running in outcome.refusal.detail

    def test_refusal_replaces_a_deserialization_traceback(self, tmp_path: Path, exploding_joblib: list[object]) -> None:
        path, digest = _artifact(tmp_path)
        outcome = validate_artifact(
            _manifest(digest, contract=local_feature_contract("stuck"), sklearn_version="0.24.1"),
            path,
            running_version="1.5.2",
        )
        assert not outcome.ok
        assert outcome.refusal.check == CHECK_RUNTIME
        assert exploding_joblib == []

    def test_manifest_without_a_recorded_version_is_refused(self) -> None:
        outcome = validate_runtime(_manifest("a" * 64, sklearn_version=""), running_version="1.5.2")
        assert not outcome.ok
        assert outcome.refusal.reason == "no_recorded_version"

    @pytest.mark.parametrize(
        ("recorded", "running"),
        [("nightly", "1.5.2"), ("1.5.2", "dev"), ("1", "1.5.2")],
    )
    def test_unparseable_versions_are_refused(self, recorded: str, running: str) -> None:
        outcome = validate_runtime(_manifest("a" * 64, sklearn_version=recorded), running_version=running)
        assert not outcome.ok
        assert outcome.refusal.reason == "unparseable_version"

    def test_running_version_defaults_to_the_installed_sklearn(self) -> None:
        installed = running_sklearn_version()
        assert installed is not None
        assert validate_runtime(_manifest("a" * 64, sklearn_version=installed)).ok


# ---------------------------------------------------------------------------
# The composed check
# ---------------------------------------------------------------------------


class TestComposedValidation:
    def test_a_good_artifact_passes_all_three_and_yields_the_bytes(self, tmp_path: Path) -> None:
        path, digest = _artifact(tmp_path)
        outcome = validate_artifact(_manifest(digest, contract=local_feature_contract("stuck")), path)

        assert outcome.ok
        assert outcome.verified is not None
        loaded = deserialize_verified(outcome.verified)
        assert loaded.ok
        assert loaded.model == {"weights": [1.0, 2.0, 3.0]}

    def test_checks_short_circuit_in_order(self, tmp_path: Path) -> None:
        """Integrity, then contract, then runtime — a bad artifact reports integrity."""
        path, _ = _artifact(tmp_path)
        broken_everything = _manifest(
            "b" * 64,
            contract=FeatureContract(service="nope", service_version="x", names=("z",), dtypes=("float64",)),
            sklearn_version="0.24.1",
        )
        outcome = validate_artifact(broken_everything, path, running_version="1.5.2")
        assert outcome.refusal.check == CHECK_INTEGRITY

    def test_contract_is_reported_before_runtime(self, tmp_path: Path) -> None:
        path, digest = _artifact(tmp_path)
        outcome = validate_artifact(
            _manifest(
                digest,
                contract=FeatureContract(service="nope", service_version="x", names=("z",), dtypes=("float64",)),
                sklearn_version="0.24.1",
            ),
            path,
            running_version="1.5.2",
        )
        assert outcome.refusal.check == CHECK_CONTRACT

    def test_no_check_raises_to_the_caller(self, tmp_path: Path) -> None:
        """FR-017: every failure mode is a structured result."""
        cases = [
            (tmp_path / "absent.joblib", _manifest("a" * 64)),
            (tmp_path / "absent.joblib", Manifest(name="quality", version="1", artifact_sha256="")),
        ]
        for artifact_path, manifest in cases:
            outcome = validate_artifact(manifest, artifact_path)
            assert isinstance(outcome.refusal, Refusal)

    def test_manifest_read_and_validate_is_not_a_perceptible_cost(self, tmp_path: Path) -> None:
        """NFR-002: under 50ms per model, excluding the artifact digest."""
        path, digest = _artifact(tmp_path)
        manifest_path = write_manifest(
            tmp_path / "stuck.json", _manifest(digest, contract=local_feature_contract("stuck"))
        )
        local_feature_contract("stuck")  # warm the Feast import

        start = time.perf_counter()
        read = read_manifest(manifest_path)
        assert validate_artifact(read.manifest, path).ok
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"read + validate took {elapsed_ms:.1f}ms"


class TestRefusalsAreActionable:
    """Every refusal in this package carries a reason and a usable detail."""

    def _every_refusal(self, tmp_path: Path) -> list[Refusal]:
        path, digest = _artifact(tmp_path)
        local = local_feature_contract("stuck")
        refusals = [
            read_manifest(tmp_path / "absent.json").refusal,
            read_manifest_text("{oops").refusal,
            read_manifest_text("[]").refusal,
            manifest_from_dict({"name": "stuck"}).refusal,
            verify_artifact_file(tmp_path / "absent.joblib", _manifest(digest)).refusal,
            verify_artifact_file(path, _manifest("b" * 64)).refusal,
            validate_feature_contract(Manifest(name="quality", version="1", artifact_sha256="a")).refusal,
            validate_feature_contract(_manifest(digest, contract=FeatureContract(service="duration")), local).refusal,
            validate_feature_contract(
                _manifest(
                    digest,
                    contract=FeatureContract(
                        service="stuck",
                        service_version=local.service_version,
                        names=tuple(reversed(local.names)),
                        dtypes=local.dtypes,
                    ),
                ),
                local,
            ).refusal,
            validate_runtime(_manifest(digest, sklearn_version="0.24.1"), running_version="1.5.2").refusal,
            validate_runtime(_manifest(digest, sklearn_version=""), running_version="1.5.2").refusal,
        ]
        assert all(r is not None for r in refusals)
        return refusals

    def test_every_refusal_names_a_check_a_reason_and_a_detail(self, tmp_path: Path) -> None:
        for refusal in self._every_refusal(tmp_path):
            assert refusal.check in {CHECK_MANIFEST, CHECK_INTEGRITY, CHECK_CONTRACT, CHECK_RUNTIME}
            assert refusal.reason and " " not in refusal.reason
            # Long enough to say what disagreed, not just that something did.
            assert len(refusal.detail) > 40, refusal
            assert str(refusal).startswith(f"{refusal.check}/{refusal.reason}")

    def test_reason_codes_are_distinct_within_a_check(self, tmp_path: Path) -> None:
        """A caller branching on (check, reason) gets one meaning per pair."""
        pairs = [(r.check, r.reason) for r in self._every_refusal(tmp_path)]
        assert len(set(pairs)) == len(pairs)
