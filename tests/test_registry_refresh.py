"""Tests for base refresh: detection, rebuild, reset, and atomicity.

Two of these carry the package, and the rest support them.

**The inequality** (``TestPersonalizationSurvives``). A refresh that merely
completes proves nothing — serving the new base unchanged would satisfy every
"did it run" assertion while silently discarding months of adaptation. So the
test that matters measures: it builds a local model from a known retained set,
ships a new base with an unchanged contract, refreshes, and asserts the rebuilt
model's predictions on a *holdout* are closer to the pre-refresh local model's
than the bare new base's are. That inequality is the only thing that can tell
"the retained data was replayed" from "the new base was served".

**Interruption, not inspection** (``TestAtomicity``). Reading the source and
seeing a temp-then-move is not evidence. Each atomicity test injects a failure
at a different point — staging, the artifact move, the manifest move, the
retraining itself — and then asserts the previous artifact and manifest are
*byte-identical* to what they were and still resolve and serve.

Detection is asserted to be timestamp-free structurally as well as behaviourally
(``TestDetectionUsesNoTimestamps``): the behavioural test can only show that
today's code ignores mtime, whereas walking the module's AST also catches a
timestamp added tomorrow.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from kenaz_ml.modelstore.registry import (
    FeatureContract,
    Manifest,
    Provenance,
    Runtime,
    Training,
    read_manifest,
    running_sklearn_version,
    write_manifest,
)
from kenaz_ml.modelstore.registry import refresh as refresh_module
from kenaz_ml.modelstore.registry.refresh import (
    ACTION_ADOPT_BASE,
    ACTION_NONE,
    ACTION_REBUILD,
    ACTION_RESET,
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
    describe,
    detect_base_change,
    refresh_all,
    refresh_model,
)
from kenaz_ml.modelstore.registry.retained import (
    Example,
    append_examples,
    read_retained,
)
from kenaz_ml.modelstore.registry.slots import (
    SLOT_LOCAL,
    artifact_path,
    manifest_path,
    resolve_model,
)

REFRESH_LOGGER = "kenaz_ml.modelstore.registry.refresh"

MODEL = "stuck"

#: A six-name contract shaped like the real ``stuck`` one. Passed explicitly
#: everywhere so these tests never depend on the Feast registry — what is under
#: test is the refresh policy, not contract derivation.
NAMES = (
    "test_failure_count",
    "time_in_phase_sec",
    "edit_velocity",
    "file_switch_rate",
    "session_length_sec",
    "time_since_last_commit_sec",
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _identifiers(module: Any) -> set[str]:
    """Every name, attribute and keyword-argument identifier in a module's *code*.

    Docstrings are excluded by construction: this walks the parsed tree rather
    than the text, so a docstring that names ``warm_start`` in order to explain
    why it is absent does not count as a use of it. Only an actual reference to
    the identifier does.
    """
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, (ast.keyword, ast.arg)) and node.arg:
            found.add(node.arg)
    return found


def _contract(version: str = "contract-v1", names: tuple[str, ...] = NAMES) -> FeatureContract:
    return FeatureContract(
        service="stuck",
        service_version=version,
        names=names,
        dtypes=tuple("float64" for _ in names),
    )


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _population(n: int, *, seed: int, shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A separable two-class problem. ``shift`` moves the decision region.

    The base and the local user are deliberately given *different* shifts: that
    is what "this install works differently from the population the base was
    trained on" means, and without it a rebuild and the bare base would agree
    and the inequality would be untestable.
    """
    rng = _rng(seed)
    x = rng.normal(size=(n, len(NAMES))) * 2.0
    score = x[:, 0] + 0.5 * x[:, 1] - x[:, 2] + shift
    y = (score > 0).astype(float)
    return x, y


def _fit(x: np.ndarray, y: np.ndarray, *, seed: int = 42) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(n_estimators=40, max_depth=3, learning_rate=0.1, random_state=seed)
    model.fit(x, y)
    return model


def _dump(model: Any) -> bytes:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _install(
    slot_dir: Path,
    *,
    name: str = MODEL,
    payload: bytes,
    version: str,
    contract: FeatureContract,
    provenance: Provenance | None = None,
    training: Training | None = None,
    digest: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> Manifest:
    """Place a consistent artifact/manifest pair in a slot and return the manifest."""
    slot_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_path(slot_dir, name)
    artifact.write_bytes(payload)
    manifest = Manifest(
        name=name,
        version=version,
        artifact_sha256=digest if digest is not None else _digest(payload),
        created_at=1_753_900_000_000,
        provenance=provenance if provenance is not None else Provenance(training_source=TRAINING_SOURCE_BASE),
        runtime=Runtime(
            estimator="GradientBoostingClassifier",
            sklearn_version=running_sklearn_version() or "1.5.2",
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        ),
        feature_contract=contract,
        training=training if training is not None else Training(),
        metrics=metrics or {},
    )
    write_manifest(manifest_path(slot_dir, name), manifest)
    return manifest


@pytest.fixture
def slots(tmp_path: Path) -> dict[str, Path]:
    local = tmp_path / "ml-models"
    base = tmp_path / "ml-base"
    retained = tmp_path / "ml-models" / "retained"
    local.mkdir(parents=True, exist_ok=True)
    base.mkdir(parents=True, exist_ok=True)
    retained.mkdir(parents=True, exist_ok=True)
    return {"local": local, "base": base, "retained": retained}


def _retain(directory: Path, x: np.ndarray, y: np.ndarray, contract: FeatureContract, generation: str = "1") -> None:
    examples = [Example(x=tuple(float(v) for v in row), y=float(label)) for row, label in zip(x, y)]
    result = append_examples(MODEL, examples, contract, directory=directory, generation=generation)
    assert result.ok, result.reason


class _World:
    """A populated install: a base, a local model descended from it, retained data."""

    def __init__(self, slots: dict[str, Path]) -> None:
        self.local = slots["local"]
        self.base = slots["base"]
        self.retained = slots["retained"]
        self.contract = _contract()
        self.base_v1: Manifest | None = None
        self.local_manifest: Manifest | None = None
        self.local_model: Any = None

    def ship_base(
        self,
        version: str,
        *,
        seed: int,
        shift: float = 0.0,
        contract: FeatureContract | None = None,
        n: int = 400,
    ) -> Manifest:
        x, y = _population(n, seed=seed, shift=shift)
        model = _fit(x, y, seed=seed)
        return _install(
            self.base,
            payload=_dump(model),
            version=version,
            contract=contract if contract is not None else self.contract,
            metrics={"accuracy": 0.8},
        )

    def build_local(self, base: Manifest, x: np.ndarray, y: np.ndarray) -> Manifest:
        """Train a local model on ``x``/``y`` and record it as descended from ``base``."""
        self.local_model = _fit(x, y, seed=7)
        self.local_manifest = _install(
            self.local,
            payload=_dump(self.local_model),
            version="1",
            contract=self.contract,
            provenance=Provenance(
                base_version=base.version,
                base_sha256=base.artifact_sha256,
                n_local_extensions=1,
                training_source=TRAINING_SOURCE_LOCAL,
            ),
            training=Training(n_samples=len(x), retained_generation="1"),
        )
        return self.local_manifest


@pytest.fixture
def world(slots: dict[str, Path]) -> _World:
    return _World(slots)


def _load_local_model(local_dir: Path, name: str = MODEL) -> Any:
    return joblib.load(io.BytesIO(artifact_path(local_dir, name).read_bytes()))


def _snapshot(local_dir: Path, name: str = MODEL) -> tuple[bytes, bytes]:
    return (
        artifact_path(local_dir, name).read_bytes(),
        manifest_path(local_dir, name).read_bytes(),
    )


# ---------------------------------------------------------------------------
# T014 — detection
# ---------------------------------------------------------------------------


class TestDetection:
    """FR-012 — a changed shipped base is detectable from manifest provenance alone."""

    def test_same_version_and_digest_is_not_due(self, world: _World) -> None:
        base = world.ship_base("1", seed=1)
        world.build_local(base, *_population(120, seed=11, shift=2.0))

        change = detect_base_change(MODEL, local_dir=world.local, base_dir=world.base)

        assert change.due is False
        assert change.reason == REASON_UP_TO_DATE
        assert change.base_version == "1"
        assert change.local_base_version == "1"

    def test_a_different_version_is_due(self, world: _World) -> None:
        base_v1 = world.ship_base("1", seed=1)
        world.build_local(base_v1, *_population(120, seed=11, shift=2.0))
        world.ship_base("2", seed=2)

        change = detect_base_change(MODEL, local_dir=world.local, base_dir=world.base)

        assert change.due is True
        assert change.reason == REASON_BASE_VERSION_CHANGED
        assert change.local_base_version == "1"
        assert change.base_version == "2"

    def test_the_same_version_with_a_different_digest_is_due(self, world: _World) -> None:
        """A reused version number is the case a version comparison cannot see."""
        base_v1 = world.ship_base("1", seed=1)
        world.build_local(base_v1, *_population(120, seed=11, shift=2.0))

        # Ship a genuinely different base still labelled version 1.
        reshipped = world.ship_base("1", seed=99, shift=3.0)
        assert reshipped.version == base_v1.version
        assert reshipped.artifact_sha256 != base_v1.artifact_sha256

        change = detect_base_change(MODEL, local_dir=world.local, base_dir=world.base)

        assert change.due is True
        assert change.reason == REASON_BASE_DIGEST_CHANGED
        assert change.local_base_sha256 == base_v1.artifact_sha256
        assert change.base_sha256 == reshipped.artifact_sha256
        assert base_v1.artifact_sha256[:12] in change.detail
        assert reshipped.artifact_sha256[:12] in change.detail

    def test_no_base_present_is_nothing_to_do(self, world: _World) -> None:
        """The state of every install in the field today."""
        world.build_local(
            Manifest(name=MODEL, version="1", artifact_sha256="deadbeef"),
            *_population(120, seed=11, shift=2.0),
        )

        change = detect_base_change(MODEL, local_dir=world.local, base_dir=world.base)

        assert change.due is False
        assert change.reason == REASON_NO_BASE

    def test_no_local_present_means_the_base_serves_directly(self, world: _World) -> None:
        world.ship_base("1", seed=1)

        change = detect_base_change(MODEL, local_dir=world.local, base_dir=world.base)

        assert change.due is False
        assert change.reason == REASON_NO_LOCAL
        assert change.base_version == "1"

    def test_neither_present_is_nothing_to_do(self, world: _World) -> None:
        change = detect_base_change(MODEL, local_dir=world.local, base_dir=world.base)

        assert change.due is False
        assert change.reason == REASON_NO_BASE

    def test_half_a_base_pair_is_not_a_base(self, world: _World) -> None:
        """A manifest with no artifact beside it cannot be refreshed from."""
        base = world.ship_base("2", seed=2)
        artifact_path(world.base, MODEL).unlink()
        world.build_local(base, *_population(120, seed=11, shift=2.0))

        change = detect_base_change(MODEL, local_dir=world.local, base_dir=world.base)

        assert change.due is False
        assert change.reason == REASON_NO_BASE

    def test_an_unreadable_base_manifest_does_not_trigger_a_refresh(self, world: _World) -> None:
        base = world.ship_base("2", seed=2)
        world.build_local(base, *_population(120, seed=11, shift=2.0))
        manifest_path(world.base, MODEL).write_text("{ not json", encoding="utf-8")

        change = detect_base_change(MODEL, local_dir=world.local, base_dir=world.base)

        assert change.due is False
        assert change.reason == REASON_BASE_MANIFEST_UNUSABLE

    def test_an_unreadable_local_manifest_does_not_trigger_a_refresh(self, world: _World) -> None:
        base = world.ship_base("2", seed=2)
        world.build_local(base, *_population(120, seed=11, shift=2.0))
        manifest_path(world.local, MODEL).write_text("{ not json", encoding="utf-8")

        change = detect_base_change(MODEL, local_dir=world.local, base_dir=world.base)

        assert change.due is False
        assert change.reason == REASON_LOCAL_MANIFEST_UNUSABLE

    def test_a_local_model_with_no_recorded_ancestor_is_due(self, world: _World) -> None:
        """A pre-registry local model, once a base finally ships."""
        world.ship_base("1", seed=1)
        _install(
            world.local,
            payload=_dump(_fit(*_population(120, seed=11, shift=2.0))),
            version="1",
            contract=world.contract,
            provenance=Provenance(training_source=TRAINING_SOURCE_LOCAL),
        )

        change = detect_base_change(MODEL, local_dir=world.local, base_dir=world.base)

        assert change.due is True
        assert change.reason == REASON_BASE_VERSION_CHANGED
        assert change.local_base_version is None

    def test_detection_never_raises(self, world: _World, tmp_path: Path) -> None:
        """FR-017 applies here too: detection is on the startup path."""
        for local, base in (
            (tmp_path / "absent-local", tmp_path / "absent-base"),
            (world.local, tmp_path / "absent-base"),
            (tmp_path / "absent-local", world.base),
        ):
            change = detect_base_change(MODEL, local_dir=local, base_dir=base)
            assert change.due is False


class TestDetectionUsesNoTimestamps:
    """D-007 — version and digest comparison only, never an mtime."""

    def test_the_module_reads_no_mtime_and_calls_no_last_modified(self) -> None:
        """Structural, because a behavioural test cannot catch a regression added later.

        Walks the module's code for every identifier a timestamp-based trigger
        would have to go through — ``st_mtime``, ``getmtime``, ``last_modified``,
        and ``stat`` itself, since ``Path.stat()`` is the only route to an mtime
        that does not name one. Docstrings are excluded, so the module may
        explain why it avoids ``last_modified`` without tripping this.
        """
        forbidden = {
            "st_mtime",
            "st_mtime_ns",
            "st_ctime",
            "st_atime",
            "getmtime",
            "getctime",
            "getatime",
            "last_modified",
            "stat",
            "lstat",
            "utime",
        }
        found = _identifiers(refresh_module) & forbidden
        assert found == set(), f"refresh.py reaches for a timestamp: {sorted(found)}"

    def test_the_only_clock_read_feeds_a_recorded_field_not_a_decision(self) -> None:
        """``_now_ms`` stamps manifests. It must not be reachable from detection."""
        tree = ast.parse(Path(refresh_module.__file__).read_text(encoding="utf-8"))
        detection = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "detect_base_change"
        )
        clocks = {
            node.func.id
            for node in ast.walk(detection)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "_now_ms" not in clocks
        assert "time" not in {node.id for node in ast.walk(detection) if isinstance(node, ast.Name)}

    def test_detection_ignores_mtimes_entirely(self, world: _World) -> None:
        """A base far *older* than the local model is still detected as changed."""
        base_v1 = world.ship_base("1", seed=1)
        world.build_local(base_v1, *_population(120, seed=11, shift=2.0))
        world.ship_base("2", seed=2)

        # Backdate the whole base slot by a decade. An mtime-driven trigger
        # would conclude nothing had changed; a version comparison does not care.
        ancient = 1_000_000_000
        for path in (artifact_path(world.base, MODEL), manifest_path(world.base, MODEL)):
            os.utime(path, (ancient, ancient))

        change = detect_base_change(MODEL, local_dir=world.local, base_dir=world.base)

        assert change.due is True
        assert change.reason == REASON_BASE_VERSION_CHANGED

    def test_touching_the_base_without_changing_it_triggers_nothing(self, world: _World) -> None:
        """The false positive an mtime trigger fires on at every package install."""
        base = world.ship_base("1", seed=1)
        world.build_local(base, *_population(120, seed=11, shift=2.0))

        future = 2_000_000_000
        for path in (artifact_path(world.base, MODEL), manifest_path(world.base, MODEL)):
            os.utime(path, (future, future))

        change = detect_base_change(MODEL, local_dir=world.local, base_dir=world.base)

        assert change.due is False
        assert change.reason == REASON_UP_TO_DATE


# ---------------------------------------------------------------------------
# T015 — the inequality. This is the package.
# ---------------------------------------------------------------------------


class TestPersonalizationSurvives:
    """SC-003 / User Story 3 — the retained data measurably shaped the result."""

    @staticmethod
    def _scenario(world: _World) -> dict[str, Any]:
        """Base v1, a local model adapted to a *different* regime, then base v2.

        The local user's data is shifted away from the population the bases were
        trained on. That shift is the personalization: if it survives the
        refresh, the rebuilt model tracks the local model; if it does not, the
        rebuilt model is just base v2.
        """
        base_v1 = world.ship_base("1", seed=1, shift=0.0)

        local_x, local_y = _population(300, seed=11, shift=3.0)
        world.build_local(base_v1, local_x, local_y)
        _retain(world.retained, local_x, local_y, world.contract)

        base_v2_manifest = world.ship_base("2", seed=2, shift=0.0)
        base_v2 = joblib.load(io.BytesIO(artifact_path(world.base, MODEL).read_bytes()))

        holdout_x, _ = _population(400, seed=77, shift=3.0)
        return {
            "base_v1": base_v1,
            "base_v2_manifest": base_v2_manifest,
            "base_v2": base_v2,
            "pre_refresh_local": world.local_model,
            "holdout": holdout_x,
        }

    def test_the_rebuilt_model_is_closer_to_the_pre_refresh_local_than_the_bare_new_base_is(
        self, world: _World
    ) -> None:
        scenario = self._scenario(world)
        holdout = scenario["holdout"]

        before = scenario["pre_refresh_local"].predict_proba(holdout)[:, 1]
        bare_new_base = scenario["base_v2"].predict_proba(holdout)[:, 1]

        result = refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=world.contract,
        )
        assert result.ok, result.refusal
        assert result.action == ACTION_REBUILD

        rebuilt = _load_local_model(world.local).predict_proba(holdout)[:, 1]

        rebuilt_distance = float(np.mean(np.abs(rebuilt - before)))
        base_distance = float(np.mean(np.abs(bare_new_base - before)))

        # The inequality. Without it, "refresh completed" would pass while the
        # new base had simply replaced the user's model.
        assert rebuilt_distance < base_distance, (
            f"personalization did not survive the refresh: the rebuilt model is {rebuilt_distance:.4f} "
            f"from the pre-refresh local model while the bare new base is {base_distance:.4f} — "
            "the retained data made no difference"
        )
        # And by a margin, not by a rounding error.
        assert rebuilt_distance < base_distance / 2

    def test_the_rebuilt_model_is_not_simply_the_new_base(self, world: _World) -> None:
        """The complementary direction: the artifact actually changed."""
        scenario = self._scenario(world)
        base_bytes = artifact_path(world.base, MODEL).read_bytes()

        result = refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=world.contract,
        )
        assert result.action == ACTION_REBUILD
        assert artifact_path(world.local, MODEL).read_bytes() != base_bytes

        holdout = scenario["holdout"]
        rebuilt = _load_local_model(world.local).predict_proba(holdout)[:, 1]
        bare = scenario["base_v2"].predict_proba(holdout)[:, 1]
        assert float(np.mean(np.abs(rebuilt - bare))) > 0.05


class TestSameContractRebuild:
    """FR-013 — the mechanics around the inequality."""

    @staticmethod
    def _due(world: _World, *, n: int = 60) -> Manifest:
        base_v1 = world.ship_base("1", seed=1)
        x, y = _population(n, seed=11, shift=3.0)
        world.build_local(base_v1, x, y)
        _retain(world.retained, x, y, world.contract)
        return world.ship_base("2", seed=2)

    def test_provenance_records_the_new_base_and_one_extension(self, world: _World) -> None:
        base_v2 = self._due(world)

        result = refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=world.contract,
        )

        assert result.action == ACTION_REBUILD
        written = read_manifest(manifest_path(world.local, MODEL)).manifest
        assert written is not None
        assert written.provenance.base_version == base_v2.version == "2"
        assert written.provenance.base_sha256 == base_v2.artifact_sha256
        assert written.provenance.n_local_extensions == 1
        assert written.provenance.training_source == TRAINING_SOURCE_LOCAL
        assert written.provenance.reset_reason is None

    def test_training_records_the_generation_and_sample_count(self, world: _World) -> None:
        self._due(world, n=60)

        result = refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=world.contract,
        )

        written = read_manifest(manifest_path(world.local, MODEL)).manifest
        assert written is not None
        assert written.training.n_samples == 60
        assert written.training.retained_generation == "1"
        assert result.n_samples == 60
        assert result.retained_generation == "1"

    def test_the_written_manifest_describes_the_written_artifact(self, world: _World) -> None:
        """The invariant a torn write would break: the digest must match."""
        self._due(world)

        refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=world.contract,
        )

        written = read_manifest(manifest_path(world.local, MODEL)).manifest
        assert written is not None
        assert written.artifact_sha256 == _digest(artifact_path(world.local, MODEL).read_bytes())

    def test_the_rebuilt_model_resolves_and_serves(self, world: _World) -> None:
        self._due(world)

        refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=world.contract,
        )

        resolution = resolve_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            expected_contract=world.contract,
        )
        assert resolution.slot == SLOT_LOCAL
        assert resolution.model is not None

    def test_the_rebuild_is_logged_with_counts_and_versions(
        self, world: _World, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._due(world, n=60)

        with caplog.at_level(logging.INFO, logger=REFRESH_LOGGER):
            refresh_model(
                MODEL,
                local_dir=world.local,
                base_dir=world.base,
                retained_dir=world.retained,
                expected_contract=world.contract,
            )

        rebuilt = [r for r in caplog.records if "rebuilt" in r.getMessage()]
        assert rebuilt, caplog.text
        message = rebuilt[0].getMessage()
        assert "60 retained example" in message
        assert "base 1 -> 2" in message

    def test_no_warm_start_or_partial_fit_anywhere_in_the_module(self) -> None:
        """C-006 — warm-start extension is a later mission (Reviewer Guidance 4)."""
        used = _identifiers(refresh_module)
        assert "warm_start" not in used
        assert "partial_fit" not in used

    def test_the_rebuild_discards_the_base_fitted_state(self, world: _World) -> None:
        """Full retraining, not extension: the tree count is the base's, not base+local."""
        self._due(world, n=60)

        refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=world.contract,
        )

        base_model = joblib.load(io.BytesIO(artifact_path(world.base, MODEL).read_bytes()))
        rebuilt = _load_local_model(world.local)
        assert rebuilt.n_estimators == base_model.n_estimators
        assert len(rebuilt.estimators_) == len(base_model.estimators_)

    def test_a_retained_vector_of_the_wrong_width_is_refused_not_padded(self, world: _World) -> None:
        base_v1 = world.ship_base("1", seed=1)
        x, y = _population(40, seed=11, shift=3.0)
        world.build_local(base_v1, x, y)
        _retain(world.retained, x, y, world.contract)
        world.ship_base("2", seed=2)

        # Append a short row directly, bypassing the writer's own checks.
        path = world.retained / f"{MODEL}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"record": "example", "x": [1.0, 2.0], "y": 1.0, "as_of_ms": None}) + "\n")

        before = _snapshot(world.local)
        result = refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=world.contract,
        )

        assert result.ok is False
        assert result.refusal is not None
        assert result.refusal.reason == "retained_width_mismatch"
        assert _snapshot(world.local) == before

    def test_a_refresh_that_is_not_due_does_nothing(self, world: _World) -> None:
        base = world.ship_base("1", seed=1)
        x, y = _population(40, seed=11, shift=3.0)
        world.build_local(base, x, y)
        _retain(world.retained, x, y, world.contract)

        before = _snapshot(world.local)
        result = refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=world.contract,
        )

        assert result.action == ACTION_NONE
        assert result.changed is False
        assert _snapshot(world.local) == before


class TestEmptyRetainedSet:
    """User Story 3 scenario 3 — nothing retained, so the new base is served."""

    @staticmethod
    def _due(world: _World) -> Manifest:
        base_v1 = world.ship_base("1", seed=1)
        world.build_local(base_v1, *_population(60, seed=11, shift=3.0))
        return world.ship_base("2", seed=2)

    def test_no_retention_file_at_all_serves_the_new_base_without_error(self, world: _World) -> None:
        base_v2 = self._due(world)

        result = refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=world.contract,
        )

        assert result.ok is True
        assert result.action == ACTION_ADOPT_BASE
        assert artifact_path(world.local, MODEL).read_bytes() == artifact_path(world.base, MODEL).read_bytes()

        written = read_manifest(manifest_path(world.local, MODEL)).manifest
        assert written is not None
        assert written.provenance.n_local_extensions == 0
        assert written.provenance.training_source == TRAINING_SOURCE_BASE
        assert written.provenance.base_version == base_v2.version
        assert written.provenance.reset_reason is None, "adopting a base is not a reset"

    def test_a_header_only_retention_file_serves_the_new_base(self, world: _World) -> None:
        self._due(world)
        _retain(world.retained, np.empty((0, len(NAMES))), np.empty(0), world.contract)

        result = refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=world.contract,
        )

        assert result.ok is True
        assert result.action == ACTION_ADOPT_BASE

    def test_the_adopted_base_resolves_and_serves(self, world: _World) -> None:
        self._due(world)

        refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=world.contract,
        )

        resolution = resolve_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            expected_contract=world.contract,
        )
        assert resolution.slot == SLOT_LOCAL
        assert resolution.manifest is not None
        assert resolution.manifest.provenance.n_local_extensions == 0

    def test_the_shipped_base_artifact_is_untouched(self, world: _World) -> None:
        """SC-005 and C-004 — the base slot is read-only."""
        self._due(world)
        before = (
            artifact_path(world.base, MODEL).read_bytes(),
            manifest_path(world.base, MODEL).read_bytes(),
        )

        refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=world.contract,
        )

        assert (
            artifact_path(world.base, MODEL).read_bytes(),
            manifest_path(world.base, MODEL).read_bytes(),
        ) == before


# ---------------------------------------------------------------------------
# T016 — the reset
# ---------------------------------------------------------------------------


class TestChangedContractReset:
    """FR-014 / User Story 4 — the honest failure."""

    @staticmethod
    def _due(world: _World, *, n: int = 50) -> tuple[Manifest, FeatureContract]:
        base_v1 = world.ship_base("1", seed=1)
        x, y = _population(n, seed=11, shift=3.0)
        world.build_local(base_v1, x, y)
        _retain(world.retained, x, y, world.contract)

        # The new base speaks a different contract — a feature was added.
        new_names = (*NAMES, "review_comment_count")
        new_contract = _contract(version="contract-v2", names=new_names)
        base_v2 = world.ship_base("2", seed=2, contract=new_contract)
        return base_v2, new_contract

    def _run(self, world: _World, new_contract: FeatureContract) -> Any:
        return refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=new_contract,
        )

    def test_the_retained_set_is_discarded(self, world: _World) -> None:
        _, new_contract = self._due(world, n=50)
        assert len(read_retained(MODEL, directory=world.retained).examples) == 50

        result = self._run(world, new_contract)

        assert result.action == ACTION_RESET
        after = read_retained(MODEL, directory=world.retained)
        assert after.examples == ()

    def test_the_generation_is_incremented(self, world: _World) -> None:
        _, new_contract = self._due(world)

        result = self._run(world, new_contract)

        assert result.previous_generation == "1"
        assert result.retained_generation == "2"
        after = read_retained(MODEL, directory=world.retained)
        assert after.generation == "2"

    def test_accumulation_restarts_under_the_new_contract(self, world: _World) -> None:
        """User Story 4 scenario 3."""
        _, new_contract = self._due(world)
        self._run(world, new_contract)

        after = read_retained(MODEL, directory=world.retained)
        assert after.contract_version == "contract-v2"
        assert after.names == new_contract.names

        appended = append_examples(
            MODEL,
            [Example(x=tuple(0.5 for _ in new_contract.names), y=1.0)],
            new_contract,
            directory=world.retained,
        )
        assert appended.ok, appended.reason
        assert appended.generation == "2"

    def test_the_new_base_is_served_unextended(self, world: _World) -> None:
        base_v2, new_contract = self._due(world)

        self._run(world, new_contract)

        assert artifact_path(world.local, MODEL).read_bytes() == artifact_path(world.base, MODEL).read_bytes()
        written = read_manifest(manifest_path(world.local, MODEL)).manifest
        assert written is not None
        assert written.provenance.n_local_extensions == 0
        assert written.provenance.training_source == TRAINING_SOURCE_BASE
        assert written.provenance.base_version == base_v2.version
        assert written.feature_contract.names == new_contract.names

        resolution = resolve_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            expected_contract=new_contract,
        )
        assert resolution.slot == SLOT_LOCAL
        assert resolution.model is not None

    def test_the_reset_reason_is_recorded_in_the_manifest(self, world: _World) -> None:
        """SC-004 — this is what makes the loss discoverable rather than silent."""
        _, new_contract = self._due(world)

        result = self._run(world, new_contract)

        assert result.reset_reason == RESET_REASON_CONTRACT_CHANGED
        written = read_manifest(manifest_path(world.local, MODEL)).manifest
        assert written is not None
        assert written.provenance.reset_reason == RESET_REASON_CONTRACT_CHANGED

        # And it survives a round trip through the file an operator would read.
        raw = json.loads(manifest_path(world.local, MODEL).read_text(encoding="utf-8"))
        assert raw["provenance"]["reset_reason"] == "contract_version_changed"

    def test_the_reset_is_logged_where_an_operator_will_see_it(
        self, world: _World, caplog: pytest.LogCaptureFixture
    ) -> None:
        _, new_contract = self._due(world, n=50)

        with caplog.at_level(logging.DEBUG, logger=REFRESH_LOGGER):
            self._run(world, new_contract)

        resets = [r for r in caplog.records if "personalization reset" in r.getMessage()]
        assert resets, caplog.text
        assert resets[0].levelno >= logging.WARNING, "a user losing months of adaptation deserves more than DEBUG"
        message = resets[0].getMessage()
        assert "50 retained example" in message
        assert "contract-v1" in message
        assert "contract-v2" in message
        assert RESET_REASON_CONTRACT_CHANGED in message

    def test_a_reset_does_not_retrain(self, world: _World) -> None:
        """The vectors are not replayable, so they must not reach a fit()."""
        _, new_contract = self._due(world)

        def forbidden(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("the reset path must not train on stale-contract vectors")

        result = refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=new_contract,
            train=forbidden,
        )
        assert result.action == ACTION_RESET


# ---------------------------------------------------------------------------
# T017 — atomicity, proved by interruption
# ---------------------------------------------------------------------------


class TestAtomicity:
    """FR-019 — a failed refresh leaves the previous model intact and servable.

    Each test injects a failure at a different point and then asserts, on the
    bytes, that nothing moved. Asserting only "no exception escaped" would pass
    against a half-written pair.
    """

    @staticmethod
    def _due(world: _World, *, n: int = 60) -> None:
        base_v1 = world.ship_base("1", seed=1)
        x, y = _population(n, seed=11, shift=3.0)
        world.build_local(base_v1, x, y)
        _retain(world.retained, x, y, world.contract)
        world.ship_base("2", seed=2)

    def _refresh(self, world: _World, **kwargs: Any) -> Any:
        return refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=world.contract,
            **kwargs,
        )

    def _assert_unchanged_and_servable(self, world: _World, before: tuple[bytes, bytes]) -> None:
        assert _snapshot(world.local) == before, "the previous artifact/manifest pair was not left byte-identical"
        resolution = resolve_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            expected_contract=world.contract,
        )
        assert resolution.slot == SLOT_LOCAL, "the previous local model is no longer served"
        assert resolution.model is not None
        # No staging debris left behind for the next run to trip over.
        leftovers = [p.name for p in world.local.iterdir() if p.name.startswith(".")]
        assert leftovers == [], leftovers

    def test_a_failure_during_retraining_leaves_the_previous_pair_intact(self, world: _World) -> None:
        self._due(world)
        before = _snapshot(world.local)

        def explode(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("power cut during fit()")

        result = self._refresh(world, train=explode)

        assert result.ok is False
        assert result.refusal is not None
        assert result.refusal.reason == "retraining_failed"
        self._assert_unchanged_and_servable(world, before)

    def test_a_failure_moving_the_artifact_leaves_the_previous_pair_intact(
        self, world: _World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first move fails, so nothing was replaced at all."""
        self._due(world)
        before = _snapshot(world.local)
        target = artifact_path(world.local, MODEL)
        real_replace = os.replace

        def interrupted(src: Any, dst: Any, **kwargs: Any) -> None:
            if Path(dst) == target:
                raise OSError("power cut between write and replace")
            real_replace(src, dst, **kwargs)

        monkeypatch.setattr(refresh_module.os, "replace", interrupted)

        result = self._refresh(world)

        assert result.ok is False
        assert result.refusal is not None
        assert result.refusal.reason == "artifact_move_failed"
        self._assert_unchanged_and_servable(world, before)

    def test_a_failure_moving_the_manifest_undoes_the_artifact_move(
        self, world: _World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The dangerous case: a new artifact would otherwise sit beside an old manifest."""
        self._due(world)
        before = _snapshot(world.local)
        target = manifest_path(world.local, MODEL)
        real_replace = os.replace

        def interrupted(src: Any, dst: Any, **kwargs: Any) -> None:
            if Path(dst) == target:
                raise OSError("power cut between the artifact move and the manifest move")
            real_replace(src, dst, **kwargs)

        monkeypatch.setattr(refresh_module.os, "replace", interrupted)

        result = self._refresh(world)

        assert result.ok is False
        assert result.refusal is not None
        assert result.refusal.reason == "manifest_move_failed"
        self._assert_unchanged_and_servable(world, before)

    def test_a_totally_unavailable_replace_leaves_the_previous_pair_intact(
        self, world: _World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every move fails, including the ones the recovery path would want."""
        self._due(world)
        before = _snapshot(world.local)

        def never(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("the filesystem went away")

        monkeypatch.setattr(refresh_module.os, "replace", never)

        result = self._refresh(world)

        assert result.ok is False
        self._assert_unchanged_and_servable(world, before)

    def test_a_failure_writing_the_staging_artifact_leaves_the_previous_pair_intact(
        self, world: _World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._due(world)
        before = _snapshot(world.local)
        real_write_bytes = Path.write_bytes

        def refuse(self: Path, data: bytes) -> int:
            if self.name.endswith(".refresh-tmp"):
                raise OSError("no space left on device")
            return real_write_bytes(self, data)

        monkeypatch.setattr(Path, "write_bytes", refuse)

        result = self._refresh(world)

        assert result.ok is False
        assert result.refusal is not None
        assert result.refusal.reason == "staging_failed"
        assert _snapshot(world.local) == before

    def test_the_retained_set_survives_a_failed_reset(self, world: _World, monkeypatch: pytest.MonkeyPatch) -> None:
        """A reset that could not deliver the new base must not have discarded the examples."""
        base_v1 = world.ship_base("1", seed=1)
        x, y = _population(50, seed=11, shift=3.0)
        world.build_local(base_v1, x, y)
        _retain(world.retained, x, y, world.contract)
        new_contract = _contract(version="contract-v2", names=(*NAMES, "review_comment_count"))
        world.ship_base("2", seed=2, contract=new_contract)

        before = _snapshot(world.local)
        target = artifact_path(world.local, MODEL)
        real_replace = os.replace

        def interrupted(src: Any, dst: Any, **kwargs: Any) -> None:
            if Path(dst) == target:
                raise OSError("power cut")
            real_replace(src, dst, **kwargs)

        monkeypatch.setattr(refresh_module.os, "replace", interrupted)

        result = refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=new_contract,
        )

        assert result.ok is False
        assert _snapshot(world.local) == before
        surviving = read_retained(MODEL, directory=world.retained)
        assert len(surviving.examples) == 50, "the reset discarded the examples without delivering the new base"
        assert surviving.generation == "1"

    def test_a_corrupt_shipped_base_is_refused_before_anything_is_written(self, world: _World) -> None:
        self._due(world)
        before = _snapshot(world.local)

        # Tamper with the base artifact so its digest no longer matches.
        artifact_path(world.base, MODEL).write_bytes(b"not a model")

        result = self._refresh(world)

        assert result.ok is False
        assert result.refusal is not None
        assert result.refusal.reason == "base_artifact_unusable"
        self._assert_unchanged_and_servable(world, before)

    def test_the_refresh_never_raises(self, world: _World, monkeypatch: pytest.MonkeyPatch) -> None:
        """FR-017 on the startup path: every failure above is a returned value."""
        self._due(world)

        def never(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("the filesystem went away")

        monkeypatch.setattr(refresh_module.os, "replace", never)
        assert self._refresh(world).ok is False

        monkeypatch.undo()

        def explode(*_args: Any, **_kwargs: Any) -> Any:
            raise ValueError("y contains a single class")

        assert self._refresh(world, train=explode).ok is False


# ---------------------------------------------------------------------------
# Roster behaviour and the reporting surface
# ---------------------------------------------------------------------------


class TestRosterAndReporting:
    def test_refresh_all_is_independent_per_model(self, world: _World, slots: dict[str, Path]) -> None:
        """A broken artifact for one model is no reason to strand another."""
        base_v1 = world.ship_base("1", seed=1)
        x, y = _population(40, seed=11, shift=3.0)
        world.build_local(base_v1, x, y)
        _retain(world.retained, x, y, world.contract)
        world.ship_base("2", seed=2)

        results = refresh_all(
            (MODEL, "duration"),
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
        )

        by_name = {r.name: r for r in results}
        assert set(by_name) == {MODEL, "duration"}
        assert by_name["duration"].action == ACTION_NONE
        assert by_name["duration"].change.reason == REASON_NO_BASE

    def test_describe_renders_the_result_as_plain_data(self, world: _World) -> None:
        base_v1 = world.ship_base("1", seed=1)
        x, y = _population(40, seed=11, shift=3.0)
        world.build_local(base_v1, x, y)
        _retain(world.retained, x, y, world.contract)
        world.ship_base("2", seed=2)

        result = refresh_model(
            MODEL,
            local_dir=world.local,
            base_dir=world.base,
            retained_dir=world.retained,
            expected_contract=world.contract,
        )
        rendered = describe(result)

        assert json.loads(json.dumps(rendered))["action"] == ACTION_REBUILD
        assert rendered["base_version"] == "2"
        assert rendered["local_base_version"] == "1"
        assert rendered["n_samples"] == 40
        assert rendered["reset_reason"] is None

    def test_the_default_state_of_every_install_today_is_a_no_op(self, tmp_path: Path) -> None:
        """No base models have ever been built. Nothing may happen, and nothing may raise."""
        local = tmp_path / "ml-models"
        base = tmp_path / "ml-base"
        local.mkdir()

        results = refresh_all(("stuck", "duration", "suggest", "quality", "profile"), local_dir=local, base_dir=base)

        assert all(r.action == ACTION_NONE for r in results)
        assert all(r.ok for r in results)
        assert all(r.change.reason == REASON_NO_BASE for r in results)
        assert list(local.iterdir()) == []
