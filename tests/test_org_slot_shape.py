"""laya-serving-and-packs-01MSK2SP WP04 -- the ``local -> org -> base -> cold start`` ladder and directory artifacts.

The existing registry suite is the regression guard for ``org_dir=None``; this
file adds the org slot's ordering, the ``org`` provenance value, and the
directory artifact type (FR-011) that lets a laya checkpoint occupy a slot.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pytest

from kenaz_ml import config
from kenaz_ml.advice.contracts import build_contract, contract_for
from kenaz_ml.advice.dispatch import BACKEND_LAYA, LayaBackend, RecommendRequest, Refused, build_table, dispatch
from kenaz_ml.laya import agent as agent_mod
from kenaz_ml.laya import eligibility as el
from kenaz_ml.laya.agent import LayaRuntime, find_checkpoint
from kenaz_ml.modelstore.registry import (
    ARTIFACT_KIND_DIRECTORY,
    KNOWN_TRAINING_SOURCES,
    SLOT_BASE,
    SLOT_COLD_START,
    SLOT_LOCAL,
    SLOT_ORG,
    TRAINING_SOURCE_ORG,
    Manifest,
    Provenance,
    Runtime,
    directory_digest,
    manifest_to_dict,
    read_manifest,
    resolve_model,
    running_sklearn_version,
    write_manifest,
)
from kenaz_ml.modelstore.registry import slots as slots_mod

NAME = "ladder_model"
CONTRACT = build_contract(NAME, ("a", "b"))


@pytest.fixture
def dirs(tmp_path: Path) -> dict[str, Path]:
    out = {k: tmp_path / k for k in ("local", "org", "base")}
    for d in out.values():
        d.mkdir()
    return out


def _pair(slot: Path, who: str, *, name: str = NAME, tamper: bool = False) -> None:
    import io

    buf = io.BytesIO()
    joblib.dump({"who": who}, buf)
    payload = buf.getvalue()
    (slot / f"{name}.joblib").write_bytes(b"tampered" if tamper else payload)
    write_manifest(
        slot / f"{name}.json",
        Manifest(
            name=name,
            version="1",
            artifact_sha256=hashlib.sha256(payload).hexdigest(),
            provenance=Provenance(training_source={"local": "local", "org": "org", "base": "base"}[who]),
            runtime=Runtime(sklearn_version=running_sklearn_version() or ""),
            feature_contract=CONTRACT,
        ),
    )


def _resolve(dirs: dict[str, Path], **kw: Any) -> Any:
    return resolve_model(NAME, local_dir=dirs["local"], base_dir=dirs["base"], expected_contract=CONTRACT, **kw)


# ---------------------------------------------------------------------------
# org_dir=None is exactly today's behaviour
# ---------------------------------------------------------------------------


def test_public_surface_exports_slot_org() -> None:
    assert SLOT_ORG == "org"
    assert TRAINING_SOURCE_ORG == "org" and "org" in KNOWN_TRAINING_SOURCES
    assert {"base", "local", "synthetic"} <= set(KNOWN_TRAINING_SOURCES)  # existing values unchanged


def test_org_dir_none_is_indistinguishable_from_the_call_without_it(
    dirs: dict[str, Path], caplog: pytest.LogCaptureFixture
) -> None:
    _pair(dirs["base"], "base")
    with caplog.at_level(logging.DEBUG, logger=slots_mod.logger.name):
        without = _resolve(dirs)
        log_without = [(r.levelno, r.getMessage()) for r in caplog.records]
        caplog.clear()
        with_none = _resolve(dirs, org_dir=None)
        log_with_none = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert (without.slot, without.model, without.reasons) == (with_none.slot, with_none.model, with_none.reasons)
    assert log_without == log_with_none and log_without


def test_the_default_path_never_touches_an_org_location(dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    _pair(dirs["base"], "base")
    attempted: list[str] = []
    real = slots_mod._try_slot
    monkeypatch.setattr(
        slots_mod, "_try_slot", lambda name, slot, *a, **k: (attempted.append(slot), real(name, slot, *a, **k))[1]
    )
    _resolve(dirs)
    assert attempted == [SLOT_LOCAL, SLOT_BASE]  # the org slot is not attempted, not logged, not read
    # ...and there is no function that would invent an org location.
    assert not [n for n in dir(slots_mod) if "org" in n.lower() and n != "SLOT_ORG" and "ORG" not in n]
    assert not hasattr(slots_mod, "org_slot_dir")
    before = sorted(p.name for p in dirs["local"].parent.iterdir())
    _resolve(dirs)
    assert sorted(p.name for p in dirs["local"].parent.iterdir()) == before


# ---------------------------------------------------------------------------
# Ordering: local, org, base, cold start
# ---------------------------------------------------------------------------


def test_local_beats_org_beats_base_then_cold_start(dirs: dict[str, Path]) -> None:
    _pair(dirs["local"], "local")
    _pair(dirs["org"], "org")
    _pair(dirs["base"], "base")
    assert _resolve(dirs, org_dir=dirs["org"]).slot == SLOT_LOCAL
    (dirs["local"] / f"{NAME}.joblib").unlink()
    (dirs["local"] / f"{NAME}.json").unlink()
    org = _resolve(dirs, org_dir=dirs["org"])
    assert org.slot == SLOT_ORG == "org" and org.model == {"who": "org"}
    assert org.manifest is not None and org.manifest.provenance.training_source == "org"
    (dirs["org"] / f"{NAME}.joblib").unlink()
    (dirs["org"] / f"{NAME}.json").unlink()
    assert _resolve(dirs, org_dir=dirs["org"]).slot == SLOT_BASE
    (dirs["base"] / f"{NAME}.joblib").unlink()
    (dirs["base"] / f"{NAME}.json").unlink()
    cold = _resolve(dirs, org_dir=dirs["org"])
    assert cold.slot == SLOT_COLD_START and not cold.served


def test_empty_org_dir_is_the_quiet_slot_empty_case(dirs: dict[str, Path], caplog: pytest.LogCaptureFixture) -> None:
    _pair(dirs["base"], "base")
    with caplog.at_level(logging.WARNING, logger=slots_mod.logger.name):
        resolution = _resolve(dirs, org_dir=dirs["org"])
    assert resolution.slot == SLOT_BASE
    assert caplog.records == []  # nothing at warning level or above
    assert any(r.slot == SLOT_ORG and r.reason == "slot_empty" for r in resolution.refusals)


def test_a_tampered_org_artifact_is_refused_loudly_and_falls_through(
    dirs: dict[str, Path], caplog: pytest.LogCaptureFixture
) -> None:
    _pair(dirs["org"], "org", tamper=True)
    _pair(dirs["base"], "base")
    with caplog.at_level(logging.DEBUG, logger=slots_mod.logger.name):
        resolution = _resolve(dirs, org_dir=dirs["org"])
    assert resolution.slot == SLOT_BASE
    refusal = next(r for r in resolution.refusals if r.slot == SLOT_ORG)
    assert refusal.reason == "digest_mismatch"
    org_records = [r for r in caplog.records if "org slot" in r.getMessage()]
    assert org_records and all(r.levelno == logging.ERROR for r in org_records)  # like base: a broken delivery


def test_local_refusal_is_a_warning_org_and_base_are_errors(
    dirs: dict[str, Path], caplog: pytest.LogCaptureFixture
) -> None:
    _pair(dirs["local"], "local", tamper=True)
    with caplog.at_level(logging.DEBUG, logger=slots_mod.logger.name):
        _resolve(dirs, org_dir=dirs["org"])
    assert next(r for r in caplog.records if "local slot" in r.getMessage()).levelno == logging.WARNING
    assert slots_mod._REFUSAL_LEVEL[SLOT_ORG] == logging.ERROR == slots_mod._REFUSAL_LEVEL[SLOT_BASE]


def test_a_contract_mismatched_org_artifact_is_refused_like_any_slot(dirs: dict[str, Path]) -> None:
    _pair(dirs["org"], "org")
    other = build_contract(NAME, ("b", "a"))  # same names, different order
    resolution = resolve_model(
        NAME, local_dir=dirs["local"], base_dir=dirs["base"], org_dir=dirs["org"], expected_contract=other
    )
    assert not resolution.served
    assert next(r for r in resolution.refusals if r.slot == SLOT_ORG).check == "contract"


# ---------------------------------------------------------------------------
# The org provenance stamp
# ---------------------------------------------------------------------------


def test_an_org_stamped_manifest_round_trips(tmp_path: Path) -> None:
    manifest = Manifest(name="m", version="1", artifact_sha256="a" * 64, provenance=Provenance(training_source="org"))
    path = write_manifest(tmp_path / "m.json", manifest)
    read = read_manifest(path)
    assert read.ok and read.manifest is not None and read.manifest.provenance.training_source == "org"
    assert json.loads(path.read_text())["provenance"]["training_source"] == "org"


# ---------------------------------------------------------------------------
# FR-011 -- the directory artifact type
# ---------------------------------------------------------------------------

MEMBERS = {
    "laya.onnx": b"\x08\x07fake-onnx",
    "rl_agent_config.json": b'{"temperature": [1.0, 1.0, 1.0]}',
    "tokenizer/vocab.txt": b"[PAD]\n[CLS]\n",
    "tokenizer/tokenizer_config.json": b"{}",
}


def _checkpoint(slot: Path, name: str = "compact_now", *, provenance: str = "base", version: str = "5") -> Path:
    root = slot / f"{name}.ckpt"
    for rel in reversed(list(MEMBERS)):  # created in reverse order: the digest must not care
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(MEMBERS[rel])
    computed = directory_digest(root)
    write_manifest(
        slot / f"{name}.json",
        Manifest(
            name=name,
            version=version,
            artifact_sha256=computed.digest,
            provenance=Provenance(training_source=provenance),
            feature_contract=contract_for(name),
            metrics={"serving_backend": "laya"},
            artifact_kind=ARTIFACT_KIND_DIRECTORY,
            artifact_members=computed.members,
        ),
    )
    return root


def _resolve_kind(dirs: dict[str, Path], name: str = "compact_now", **kw: Any) -> Any:
    return resolve_model(
        name, local_dir=dirs["local"], base_dir=dirs["base"], expected_contract=contract_for(name), **kw
    )


def test_directory_digest_is_a_sorted_manifest_of_paths_and_hashes(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    for root, order in ((a, list(MEMBERS)), (b, list(reversed(MEMBERS)))):
        for rel in order:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(MEMBERS[rel])
    da, db = directory_digest(a), directory_digest(b)
    assert da.digest == db.digest and da.members == db.members
    # The canonical cross-repo recipe (design doc A5 addendum (b)): sha256sum-format lines, bytewise path order.
    expected = "".join(f"{hashlib.sha256(MEMBERS[rel]).hexdigest()}  {rel}\n" for rel in sorted(MEMBERS))
    assert da.digest == "sha256:" + hashlib.sha256(expected.encode()).hexdigest()
    (b / "extra.txt").write_bytes(b"x")
    assert directory_digest(b).digest != da.digest
    # A rename changes the digest: paths are part of the identity, not just contents.
    (b / "extra.txt").unlink()
    (b / "laya.onnx").rename(b / "laya2.onnx")
    assert directory_digest(b).digest != da.digest


def _write_tree(root: Path, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)


def test_directory_digest_reproduces_the_go_clients_canonical_vectors(tmp_path: Path) -> None:
    """Review fix: ONE recipe across kenaz-ml, the harness and Kenaz, pinned to vectors the Go code computes.

    ``sha256:71c475...`` is the harness's ``TestTreeDigest_CanonicalVector`` (core/mlsidecar/tree_test.go),
    itself cross-checked against Kenaz's internal/ml/tree.go. The checkpoint-shaped vector was computed by
    running that same Go ``TreeDigest`` over the tree below (2026-09-30 review). A drift here means the
    client's recorded pack digest and this registry's recomputation disagree, and every real pack is refused.
    """
    from tests.test_rebrand import ARTIFACT_NAME  # the frozen launcher's name: a cross-repo interface

    onedir = tmp_path / ARTIFACT_NAME
    _write_tree(
        onedir,
        {
            ARTIFACT_NAME: b"launcher bytes",
            "_internal/lib/python.dylib": b"interpreter",
            "_internal/base_library.zip": b"zip",
            "a b.txt": b"space name",
        },
    )
    (onedir / "_internal" / "Python").symlink_to("lib/python.dylib")
    canonical = "sha256:71c475c52113b8077e24afbefa21d0c4cb0938e75fcefc0a089130f7843622ec"
    assert directory_digest(onedir, allow_symlinks=True).digest == canonical
    # The checkpoint POLICY is stricter than the recipe: a symlink member is refused, never hashed.
    with pytest.raises(Exception, match="symlink"):
        directory_digest(onedir)

    ckpt = tmp_path / "ckpt"
    _write_tree(
        ckpt,
        {
            "laya.onnx": b"x",
            "rl_agent_config.json": b"{}",
            "tokenizer/vocab.txt": b"v",
            "B.txt": b"B",  # uppercase sorts before lowercase bytewise
            "a-b.txt": b"a",
        },
    )
    go_vector = "sha256:92566b9174e903c101fdca6f7fa60db29cf0751b9c4965ee9f8f317af144030d"
    assert directory_digest(ckpt).digest == go_vector
    assert directory_digest(ckpt, allow_symlinks=True).digest == go_vector  # symlink-free: the modes agree


def test_a_recorded_digest_compares_like_the_go_clients(dirs: dict[str, Path]) -> None:
    """Case-insensitive, "sha256:" prefix optional on the recorded side (the harness's ``digestsEqual``)."""
    root = _checkpoint(dirs["local"])
    manifest = read_manifest(dirs["local"] / "compact_now.json").manifest
    assert manifest.artifact_sha256.startswith("sha256:")
    bare = manifest.artifact_sha256.removeprefix("sha256:").upper()
    write_manifest(
        dirs["local"] / "compact_now.json",
        Manifest(**{**manifest.__dict__, "artifact_members": {}, "artifact_sha256": bare}),
    )
    assert _resolve_kind(dirs).served
    (root / "laya.onnx").write_bytes(b"evil")
    assert _resolve_kind(dirs).refusals[0].reason == "digest_mismatch"


def test_a_directory_pair_verifies_and_resolves_to_its_path(dirs: dict[str, Path]) -> None:
    root = _checkpoint(dirs["local"])
    resolution = _resolve_kind(dirs)
    assert resolution.served and resolution.slot == SLOT_LOCAL
    assert resolution.model == root and resolution.artifact == root  # the verified PATH, not a deserialized object
    assert isinstance(resolution.model, Path)
    assert resolution.manifest is not None and resolution.manifest.artifact_kind == "directory"


def test_a_directory_artifact_occupies_the_org_slot(dirs: dict[str, Path]) -> None:
    _checkpoint(dirs["org"], provenance="org")
    resolution = _resolve_kind(dirs, org_dir=dirs["org"])
    assert resolution.slot == SLOT_ORG and resolution.manifest.provenance.training_source == "org"
    ref = find_checkpoint(
        "compact_now",
        local_dir=dirs["local"],
        base_dir=dirs["base"],
        org_dir=dirs["org"],
        expected_contract=contract_for("compact_now"),
    )
    assert ref is not None and ref.provenance == "org" and ref.path == dirs["org"] / "compact_now.ckpt"
    assert ref.expected_sha256 == directory_digest(ref.path).members  # handed to laya for its own re-verification


@pytest.mark.parametrize("damage", ["tamper", "extra", "missing", "symlink", "rename"])
def test_a_damaged_directory_is_refused_and_names_the_member(dirs: dict[str, Path], damage: str) -> None:
    root = _checkpoint(dirs["local"])
    if damage == "tamper":
        (root / "laya.onnx").write_bytes(b"evil")
    elif damage == "extra":
        (root / "payload.py").write_bytes(b"import os")
    elif damage == "missing":
        (root / "tokenizer" / "vocab.txt").unlink()
    elif damage == "symlink":
        (root / "link").symlink_to(dirs["local"])
    elif damage == "rename":
        (root / "laya.onnx").rename(root / "other.onnx")
    resolution = _resolve_kind(dirs)
    assert not resolution.served
    refusal = resolution.refusals[0]
    assert refusal.check == "integrity"
    expected = {
        "tamper": ("member_mismatch", "laya.onnx"),
        "extra": ("member_mismatch", "payload.py"),
        "missing": ("member_mismatch", "tokenizer/vocab.txt"),
        "symlink": ("symlink_member", "link"),
        "rename": ("member_mismatch", "other.onnx"),
    }[damage]
    assert refusal.reason == expected[0] and expected[1] in refusal.detail


def test_without_declared_members_a_damaged_directory_is_a_digest_mismatch(dirs: dict[str, Path]) -> None:
    root = _checkpoint(dirs["local"])
    manifest = read_manifest(dirs["local"] / "compact_now.json").manifest
    write_manifest(dirs["local"] / "compact_now.json", Manifest(**{**manifest.__dict__, "artifact_members": {}}))
    (root / "laya.onnx").write_bytes(b"evil")
    refusal = _resolve_kind(dirs).refusals[0]
    assert refusal.reason == "digest_mismatch" and manifest.artifact_sha256 in refusal.detail


def test_directory_missing_symlinked_root_and_contract_mismatch(dirs: dict[str, Path], tmp_path: Path) -> None:
    root = _checkpoint(dirs["local"])
    wrong = build_contract("compact_now", ("x", "y"))
    mismatch = resolve_model("compact_now", local_dir=dirs["local"], base_dir=dirs["base"], expected_contract=wrong)
    assert not mismatch.served and mismatch.refusals[0].check == "contract"
    moved = tmp_path / "moved"
    root.rename(moved)
    missing = _resolve_kind(dirs)
    assert not missing.served and missing.refusals[0].reason == "artifact_not_found"
    root.symlink_to(moved, target_is_directory=True)
    assert _resolve_kind(dirs).refusals[0].reason == "symlink_root"


def test_unknown_artifact_kind_is_a_manifest_refusal(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"name": "m", "version": "1", "artifact_sha256": "a" * 64, "artifact_kind": "tarball"}))
    read = read_manifest(path)
    assert not read.ok and read.refusal is not None and read.refusal.reason == "unknown_artifact_kind"


def test_file_manifests_are_unchanged_and_directory_manifests_round_trip(tmp_path: Path) -> None:
    plain = Manifest(name="m", version="1", artifact_sha256="a" * 64)
    assert "artifact_kind" not in manifest_to_dict(plain) and "artifact_members" not in manifest_to_dict(plain)
    old = read_manifest_text_compat({"name": "m", "version": "1", "artifact_sha256": "a" * 64})
    assert old.artifact_kind == "file" and old.artifact_members == {}  # an old manifest reads as a file artifact
    directory = Manifest(
        name="m",
        version="1",
        artifact_sha256="b" * 64,
        artifact_kind="directory",
        artifact_members={"z": "1" * 64, "a": "2" * 64},
    )
    path = write_manifest(tmp_path / "d.json", directory)
    back = read_manifest(path).manifest
    assert (
        back is not None and back.artifact_kind == "directory" and back.artifact_members == directory.artifact_members
    )
    assert list(json.loads(path.read_text())["artifact_members"]) == ["a", "z"]  # deterministic order


def read_manifest_text_compat(raw: dict[str, Any]) -> Manifest:
    from kenaz_ml.modelstore.registry import manifest_from_dict

    read = manifest_from_dict(raw)
    assert read.manifest is not None
    return read.manifest


# ---------------------------------------------------------------------------
# End to end: a verified laya checkpoint directory becomes a served kind
# ---------------------------------------------------------------------------


class _Agent:
    def system_one(self, state: Any, questions: dict[str, Any], **kw: Any) -> dict[str, Any]:
        return {"answers": {"decision": {"noul": 0.8, "confidence": 0.3, "answer_confidence": 0.75}}}

    def predict_batch(self, states: list[Any], questions: dict[str, Any], **kw: Any) -> list[dict[str, Any]]:
        return [self.system_one(s, questions) for s in states]


@pytest.fixture
def slots_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    base = tmp_path / "base_models"
    base.mkdir()
    monkeypatch.setattr(config, "base_models_dir", lambda: base)
    el.reset()
    agent_mod.set_runtime(None)
    yield {"local": config.models_dir(), "base": base}
    el.reset()
    agent_mod.set_runtime(None)


def _request(name: str = "compact_now") -> RecommendRequest:
    contract = contract_for(name)
    return RecommendRequest(
        features={n: 0.5 for n in contract.names}, feature_contract_version=contract.service_version
    )


def test_a_verified_checkpoint_directory_serves_through_dispatch(
    slots_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _checkpoint(slots_env["base"], provenance="base", version="9")
    monkeypatch.setattr(el, "free_memory_bytes", lambda: (8 * 1024**3, "test"))
    loaded: list[Any] = []
    agent_mod.set_runtime(LayaRuntime(lambda ref: (loaded.append(ref), _Agent())[1]))
    table = build_table(slots_env["local"], slots_env["base"])
    entry = table.snapshot()["compact_now"]
    assert entry.available and entry.backend == BACKEND_LAYA and isinstance(entry.server, LayaBackend)
    assert el.checkpoint_set() == [f"compact_now:{entry.server.checkpoint.sha8}@base"]
    response = dispatch(table.snapshot(), "compact_now", _request())
    assert response.backend == "laya" and response.confidence == 75
    assert response.checkpoint_provenance == "base" and response.generation == "9"
    assert response.model_id_sha8 == directory_digest(root).digest.removeprefix("sha256:")[:8]
    assert loaded[0].path == root and loaded[0].expected_sha256 == directory_digest(root).members


def test_a_tampered_checkpoint_is_refused_not_reported_as_not_installed(slots_env: dict[str, Path]) -> None:
    root = _checkpoint(slots_env["local"], provenance="local")
    (root / "laya.onnx").write_bytes(b"evil")
    entry = build_table(slots_env["local"], slots_env["base"]).snapshot()["compact_now"]
    assert not entry.available and entry.reason == "checkpoint_refused"
    assert "member_mismatch" in (entry.detail or "") and "laya.onnx" in (entry.detail or "")
    assert el.checkpoint_set() == []
    with pytest.raises(Refused) as refused:
        dispatch({"compact_now": entry}, "compact_now", _request())
    assert refused.value.reason == "checkpoint_refused"


def test_a_verified_checkpoint_without_the_laya_runtime_is_not_advertised_as_available(
    slots_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review fix: the shipped state (a checkpoint placed, no laya runtime) must not read as servable.

    Before the fix ``/v1/contracts`` said ``available: true, backend: laya`` for a kind every
    request refused ``laya_backend_not_installed`` -- before and after the first refusal.
    """
    from kenaz_ml.advice.dispatch import contracts_payload

    _checkpoint(slots_env["local"])
    monkeypatch.setattr(el, "free_memory_bytes", lambda: (8 * 1024**3, "test"))
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a, **k: None if name == "laya" else object())
    table = build_table(slots_env["local"], slots_env["base"])
    for _ in range(2):  # before any dispatch, and after one
        kind = contracts_payload(table.snapshot()).kinds["compact_now"]
        assert not kind.available and kind.reason == "laya_backend_not_installed"
        with pytest.raises(Refused) as refused:
            dispatch(table.snapshot(), "compact_now", _request())
        assert refused.value.reason == "laya_backend_not_installed"
    assert el.cached_verdict() is not None  # contracts never measured; dispatch did, once
    # ...and with a runtime able to build agents, the same entry is advertised truthfully as available.
    agent_mod.set_runtime(LayaRuntime(lambda ref: _Agent()))
    assert contracts_payload(table.snapshot()).kinds["compact_now"].available


def test_a_directory_artifact_naming_a_non_laya_backend_is_refused(slots_env: dict[str, Path]) -> None:
    _checkpoint(slots_env["local"])
    manifest = read_manifest(slots_env["local"] / "compact_now.json").manifest
    write_manifest(
        slots_env["local"] / "compact_now.json",
        Manifest(**{**manifest.__dict__, "metrics": {"serving_backend": "classic"}}),
    )
    entry = build_table(slots_env["local"], slots_env["base"]).snapshot()["compact_now"]
    assert not entry.available and entry.reason == "backend_unknown"


def test_nothing_installed_serves_nothing_and_finds_no_checkpoint(slots_env: dict[str, Path]) -> None:
    """The day-one state through the real table: nothing names laya, nothing is found, nothing crashes.

    (A *laya-configured* kind with nothing installed refuses ``laya_backend_not_installed``:
    see ``test_laya_eligibility.test_eligible_host_with_no_checkpoint_still_refuses_end_to_end``.)
    """
    entry = build_table(slots_env["local"], slots_env["base"]).snapshot()["compact_now"]
    assert not entry.available and entry.reason == "kind_not_served"  # nothing names laya yet
    assert find_checkpoint("compact_now", local_dir=slots_env["local"], base_dir=slots_env["base"]) is None
