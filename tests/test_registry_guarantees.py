"""WP06 — the three guarantees no single package owns.

Each earlier work package tests its own surface. These three properties emerge
only from their combination, and each is a claim the *product* makes rather than
a function any module implements — which is exactly the kind that goes
unverified.

T022 — the no-base default state (SC-008)
-----------------------------------------
**No base model has ever been built.** Every install in the field runs with an
empty base slot and will until the first one ships, so this is not an edge case:
it is the only case. If it regresses, every install regresses and nothing else
in this mission matters.

"It did not raise" is not evidence that behaviour is unchanged. So the
pre-mission behaviour was *captured*, not assumed. :data:`PRE_MISSION` holds
values produced by the code at commit ``827058c`` — the last commit before WP01,
and therefore the behaviour every install has today. Not one of them was chosen
to make this file pass.

They were obtained by running the operations this module performs — the same
fixture, the same features, the same call order — against a worktree at
``827058c`` and against this tree, dumping both to JSON and diffing. The two
dumps were byte-identical, and :data:`PRE_MISSION` is that dump transcribed.

To reproduce, or to re-baseline after a behaviour change that is *intended*::

    git worktree add /tmp/premission 827058c

Then, against each tree in turn, with ``XDG_DATA_HOME`` pointed at a scratch
directory and ``config.base_models_dir`` pointed at an absent one, record:

    * ``StuckPredictor``/``DurationEstimator``/``QualityEstimator`` `.predict()`
      and `.is_trained` on an empty local slot — the `cold_start_predictions` key;
    * ``FilesystemModelLoader.load()`` for each of :data:`ROSTER` — `loader_*`;
    * ``Trainer(_training_store()).train_all()`` and the artifacts it leaves,
      then the same predictions again — `train_all` / `trained_predictions`;
    * each call in :data:`ENDPOINT_CALLS` against ``create_app()`` under
      ``TestClient`` — `endpoints_cold_start`.

Diff the two dumps. A non-empty diff is the regression this section exists to
catch; only update :data:`PRE_MISSION` once that diff is understood and wanted.

T023 — no egress (SC-006, FR-011, C-003)
----------------------------------------
This mission is the first to persist *user-derived training data* on disk, in a
product whose central promise is that nothing leaves the machine. The assertion
is therefore structural, not an allow-list: :func:`no_network` — reused from
``tests/test_no_egress.py`` rather than reimplemented, so the two cannot drift —
installs a :func:`sys.addaudithook` hook and refuses **every** socket the
interpreter creates, from any path.

The audit hook, not a monkeypatch, is the primary mechanism. Audit events are
raised from CPython's C implementation of the socket type, so the hook sees
sockets built through ``_socket`` directly, through a reference captured before
the test started, and from any C extension in Feast's dependency tree. A
module-level monkeypatch of ``socket`` misses all three, and
:class:`TestTheGuardIsNotVacuous` proves the guard fires by injecting real egress
into the retention path and watching it get caught.

T024 — end-to-end provenance (SC-007)
-------------------------------------
One realistic lifecycle, five stages, and at every stage the three SC-007
questions must be answerable: *which base, how many extensions, which contract
version*. Provenance is checked against the served artifact **by digest** rather
than read out of the manifest and believed — a manifest that disagreed with the
bytes beside it would be a lie that reads perfectly.

Isolation
---------
Every test redirects ``XDG_DATA_HOME`` at ``tmp_path`` (so ``models_dir()`` and
``retained_data_dir()`` are per-test) and monkeypatches ``config.base_models_dir``
at a ``tmp_path`` subdirectory. The real ``~/.local/share/sigild/`` tree and the
real shipped base slot are never read or written.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import GradientBoostingClassifier

from kenaz_ml import config
from kenaz_ml.models.duration import DurationEstimator
from kenaz_ml.models.quality import QualityEstimator
from kenaz_ml.models.stuck import FEATURE_NAMES as STUCK_FEATURE_NAMES
from kenaz_ml.models.stuck import StuckPredictor
from kenaz_ml.modelstore import FilesystemModelLoader, LocalModelStore
from kenaz_ml.modelstore.registry import (
    ACTION_ADOPT_BASE,
    ACTION_NONE,
    ACTION_REBUILD,
    ACTION_RESET,
    REASON_NO_BASE,
    RESET_REASON_CONTRACT_CHANGED,
    SLOT_BASE,
    SLOT_COLD_START,
    SLOT_LOCAL,
    TRAINING_SOURCE_BASE,
    TRAINING_SOURCE_LOCAL,
    Example,
    FeatureContract,
    Manifest,
    Provenance,
    Resolution,
    Runtime,
    append_examples,
    delete_retained,
    detect_base_change,
    local_feature_contract,
    read_retained,
    refresh_model,
    reset_retained,
    resolve_model,
    retained_path,
    running_sklearn_version,
    summarize_retained,
    write_manifest,
)
from kenaz_ml.training.trainer import Trainer

# The egress guard is *imported*, not copied. It is the same hook the
# feature-store mission proved catches raw `_socket` construction; reproducing it
# here would create a second copy free to rot into something weaker.
from tests.test_no_egress import EgressAttempted, no_network

HOUR_MS = 3_600_000
BASE_MS = 1_760_000_000_000
TENANT = "default"
MODEL = "stuck"

#: Every model the local app instantiates. `stuck` and `duration` have a
#: registered Feast feature service; the other three do not.
ROSTER = ("stuck", "duration", "activity", "workflow", "quality")


# ---------------------------------------------------------------------------
# The captured pre-mission behaviour. See the module docstring.
# ---------------------------------------------------------------------------

STUCK_FEATURES = {
    "test_failure_count": 5.0,
    "time_in_phase_sec": 1800.0,
    "edit_velocity": 2.5,
    "file_switch_rate": 0.4,
    "session_length_sec": 3600.0,
    "time_since_last_commit_sec": 900.0,
}
DURATION_FEATURES = {
    "task_complexity": 3.0,
    "file_count": 7.0,
    "historical_avg_min": 45.0,
    "time_of_day": 14.0,
    "day_of_week": 2.0,
}
QUALITY_FEATURES = {
    "edit_count": 12.0,
    "file_count": 4.0,
    "test_run_count": 2.0,
    "browser_event_count": 1.0,
    "terminal_event_count": 6.0,
    "avg_gap_sec": 30.0,
    "session_length_sec": 1800.0,
    "unique_dirs": 2.0,
}

#: Transcribed from the pre-mission dump. Not one of these values was chosen;
#: every one was produced by the code at 827058c and reproduced identically here.
PRE_MISSION: dict[str, Any] = {
    "local_slot_files_before": [],
    "cold_start_predictions": {
        "stuck": {"is_trained": False, "predict": {"probability": 0.5, "confidence": "weak"}},
        "duration": {
            "is_trained": False,
            "predict": {"estimated_minutes": 60.0, "confidence_interval": [30.0, 90.0]},
        },
        "quality": {
            "is_trained": False,
            "predict": {
                "score": 56,
                "status": "normal",
                "threshold_high": 70,
                "threshold_low": 40,
                "components": {
                    "commit_frequency": 0.0,
                    "edit_focus": 0.5,
                    "no_revert_penalty": 1.0,
                    "test_pass_rate": 0.7,
                    "velocity_vs_baseline": 0.5,
                },
            },
        },
    },
    "loader_cold_start": dict.fromkeys(ROSTER, "None"),
    "train_all": {"trained": ["duration", "next_action", "stuck"], "samples": 147},
    "artifacts_after_training": ["duration.joblib", "next_action.joblib", "stuck.joblib"],
    "trained_predictions": {
        "stuck": {"is_trained": True, "predict": {"probability": 1.0, "confidence": "strong"}},
        "duration": {
            "is_trained": True,
            "predict": {"estimated_minutes": 120.0, "confidence_interval": [120.0, 120.0]},
        },
    },
    "loader_after_training": {
        "stuck": "MODEL",
        "duration": "MODEL",
        "activity": "None",
        "workflow": "None",
        "quality": "None",
    },
    "endpoints_cold_start": {
        "GET /health": {
            "status": 200,
            "json": {
                "status": "ok",
                "mode": "local",
                "models": {
                    "stuck": "untrained",
                    "activity": "untrained",
                    "workflow": "untrained",
                    "duration": "untrained",
                    "quality": "ready",
                },
            },
        },
        "POST /predict/stuck features": {"status": 200, "json": {"probability": 0.5, "confidence": "weak"}},
        "POST /predict/stuck empty": {"status": 200, "json": {"probability": 0.5, "confidence": "weak"}},
        "POST /predict/suggest empty": {
            "status": 200,
            "json": {
                "flow_state": {
                    "deep_work": 0.2,
                    "shallow_work": 0.5999,
                    "exploring": 0.0667,
                    "blocked": 0.0667,
                    "winding_down": 0.0667,
                },
                "dominant_state": "shallow_work",
                "momentum": 0.0,
                "focus_score": 1.0,
                "method": "rules",
                "confidence": 0.5,
                "session_elapsed_min": 0.0,
                "activity_distribution": {"idle": 1.0},
                "dominant_activity": "idle",
            },
        },
        "POST /predict/suggest editing": {
            "status": 200,
            "json": {
                "flow_state": {
                    "deep_work": 0.7001,
                    "shallow_work": 0.2,
                    "exploring": 0.0333,
                    "blocked": 0.0333,
                    "winding_down": 0.0333,
                },
                "dominant_state": "deep_work",
                "momentum": 0.0,
                "focus_score": 1.0,
                "method": "rules",
                "confidence": 0.5,
                "session_elapsed_min": 0.0,
                "activity_distribution": {"editing": 1.0},
                "dominant_activity": "editing",
            },
        },
        "POST /predict/duration features": {
            "status": 200,
            "json": {"estimated_minutes": 60.0, "confidence_interval": [30.0, 90.0]},
        },
        "POST /predict/duration empty": {
            "status": 200,
            "json": {"estimated_minutes": 60.0, "confidence_interval": [30.0, 90.0]},
        },
        "POST /predict/quality features": {
            "status": 200,
            "json": {
                "score": 56,
                "status": "normal",
                "components": {
                    "commit_frequency": 0.0,
                    "edit_focus": 0.5,
                    "no_revert_penalty": 1.0,
                    "test_pass_rate": 0.7,
                    "velocity_vs_baseline": 0.5,
                },
            },
        },
    },
}

ENDPOINT_CALLS: list[tuple[str, str, str, dict | None]] = [
    ("GET /health", "GET", "/health", None),
    ("POST /predict/stuck features", "POST", "/predict/stuck", {"features": STUCK_FEATURES}),
    ("POST /predict/stuck empty", "POST", "/predict/stuck", {}),
    ("POST /predict/suggest empty", "POST", "/predict/suggest", {}),
    (
        "POST /predict/suggest editing",
        "POST",
        "/predict/suggest",
        {"classified_events": [{"kind": "file", "_category": "editing", "ts": 1000}]},
    ),
    (
        "POST /predict/duration features",
        "POST",
        "/predict/duration",
        {"features": {"file_count": 10, "total_edits": 80, "time_of_day_hour": 14, "branch_name_length": 25}},
    ),
    ("POST /predict/duration empty", "POST", "/predict/duration", {}),
    ("POST /predict/quality features", "POST", "/predict/quality", {"features": QUALITY_FEATURES}),
]

#: Fields whose value is a clock reading or a counter and therefore cannot be
#: compared across two runs. Nothing behavioural lives here.
_VOLATILE = {"uptime_sec", "last_trained", "last_trained_ms", "timestamp", "created_at"}


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_volatile(v) for k, v in value.items() if k not in _VOLATILE}
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    return value


def _rounded(value: Any) -> Any:
    """Round floats to the precision the capture used, so the two are comparable."""
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    if isinstance(value, np.integer):
        return int(value)
    return value


# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------


@pytest.fixture
def slots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Both slots isolated at `tmp_path`, with the base slot *absent*.

    The base slot is deliberately not created: `config.base_models_dir()` must
    never create it (D-001), and a missing directory is the state of every
    install today. A test that quietly mkdir'd it would be testing a state that
    does not exist in the field.

    The retained directory is named but *not* created either, for the same
    reason in miniature: an install that has never trained has no retained set,
    and calling `retained_data_dir()` here would fabricate one and make the
    "empty local slot" assertions describe a directory this fixture created.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv("KENAZ_MODE", raising=False)
    monkeypatch.delenv("KENAZ_ML_MODE", raising=False)
    base = tmp_path / "base_models"
    monkeypatch.setattr(config, "base_models_dir", lambda: base)
    local = config.models_dir()
    return {"root": tmp_path, "local": local, "base": base, "retained": local / config.RETAINED_DIRNAME}


def _events_for(task_id: str, index: int, reference_ms: int) -> list[dict]:
    """Events for one task, measured against its own reference time.

    Even-indexed tasks come out labelled stuck, odd-indexed ones do not. Both
    classes have to be present or `GradientBoostingClassifier.fit` refuses the
    matrix, which would make every assertion downstream about scikit-learn.
    """
    phase_age_ms = 700_000 if index % 2 == 0 else 60_000
    events: list[dict] = [
        {"kind": "phase_change", "source": "sigild", "payload": {"phase": "coding"}, "ts": reference_ms - phase_age_ms},
    ]
    for n in range(index + 1):
        events.append(
            {"kind": "edit", "source": "editor", "payload": {"file": f"{task_id}-{n}.py"}, "ts": reference_ms - 1_000}
        )
    return events


class FakeStore:
    """Exactly the DataStore surface the local trainer touches, and nothing more."""

    def __init__(self, tasks: list[dict], events: dict[str, list[dict]]) -> None:
        self._tasks = {t["id"]: t for t in tasks}
        self._events = events

    def get_completed_task_ids(self) -> list[str]:
        return list(self._tasks)

    def get_task_by_id(self, task_id: str) -> dict | None:
        return self._tasks.get(task_id)

    def get_events_for_task(self, task_id: str, since: int | None = None) -> list[dict]:
        return list(self._events.get(task_id, []))

    def get_completed_tasks_with_timestamps(self) -> list[dict]:
        return [
            {"id": t["id"], "started_at": t["started_at"], "completed_at": t["completed_at"]}
            for t in self._tasks.values()
        ]

    def get_signal_feedback(self, since_ms: int = 0) -> list[dict]:
        return []


def _training_store(n: int = 14) -> FakeStore:
    """`n` completed tasks — enough to clear the trainer's 10-example threshold."""
    tasks: list[dict] = []
    events: dict[str, list[dict]] = {}
    for i in range(n):
        completed_at = BASE_MS + i * HOUR_MS
        task_id = f"task-{i}"
        tasks.append(
            {
                "id": task_id,
                "repo_root": "/tmp/repo",
                "branch": "main",
                "phase": "done",
                "files": json.dumps({"main.py": 5}),
                "started_at": completed_at - 2 * HOUR_MS,
                "last_active": completed_at,
                "completed_at": completed_at,
                "test_fails": 5 if i % 2 == 0 else 0,
            }
        )
        events[task_id] = _events_for(task_id, i, completed_at)
    return FakeStore(tasks, events)


def _train_locally(local: Path) -> dict:
    """One full local training run against the isolated slot."""
    return Trainer(_training_store(), model_store=LocalModelStore(base_dir=local)).train_all()


def _fit_base_model(seed: int) -> GradientBoostingClassifier:
    """A genuinely fitted estimator, so `clone().fit()` works on the rebuild path."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(24, len(STUCK_FEATURE_NAMES)))
    y = np.array([i % 2 for i in range(24)])
    model = GradientBoostingClassifier(n_estimators=4, max_depth=2, random_state=seed)
    model.fit(x, y)
    return model


def _dump(model: Any) -> bytes:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def ship_base(base_dir: Path, name: str, version: str, *, seed: int, contract: FeatureContract) -> Manifest:
    """Write a base artifact and its manifest into the read-only slot.

    This is what an install or upgrade does — the only writer the base slot has
    (C-004). `training_source` is `base` and no `base_version` is recorded,
    because a shipped base descends from nothing.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    payload = _dump(_fit_base_model(seed))
    (base_dir / f"{name}.joblib").write_bytes(payload)
    manifest = Manifest(
        name=name,
        version=version,
        artifact_sha256=_sha256(payload),
        created_at=BASE_MS,
        provenance=Provenance(training_source=TRAINING_SOURCE_BASE, n_local_extensions=0),
        runtime=Runtime(
            estimator="GradientBoostingClassifier",
            sklearn_version=running_sklearn_version() or "",
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        ),
        feature_contract=contract,
    )
    write_manifest(base_dir / f"{name}.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# Provenance, as the three questions SC-007 asks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceReport:
    """The answers SC-007 requires, read off whatever is actually being served.

    Built from the served resolution rather than from a manifest picked off
    disk, so it can never describe a model other than the one in use.
    """

    served: bool
    slot: str
    #: Which base. For a pristine base artifact that is its own `version`; for a
    #: local artifact it is `provenance.base_version`.
    base_version: str | None = None
    base_sha256: str | None = None
    #: How many extensions.
    n_local_extensions: int | None = None
    #: Which contract version.
    contract_version: str | None = None
    training_source: str | None = None
    reset_reason: str | None = None
    artifact_sha256: str | None = None
    retained_generation: str | None = None

    @property
    def answers_sc007(self) -> bool:
        """True when all three questions have an answer."""
        return (
            self.base_version is not None and self.n_local_extensions is not None and self.contract_version is not None
        )


def provenance_of(name: str, *, local: Path, base: Path) -> tuple[ProvenanceReport, Resolution]:
    """Resolve `name` and describe the provenance of what actually answered."""
    resolution = resolve_model(name, local_dir=local, base_dir=base)
    if not resolution.served or resolution.manifest is None:
        return ProvenanceReport(served=False, slot=resolution.slot), resolution

    manifest = resolution.manifest
    prov = manifest.provenance
    return (
        ProvenanceReport(
            served=True,
            slot=resolution.slot,
            base_version=manifest.version if prov.is_base else prov.base_version,
            base_sha256=manifest.artifact_sha256 if prov.is_base else prov.base_sha256,
            n_local_extensions=prov.n_local_extensions,
            contract_version=manifest.feature_contract.service_version,
            training_source=prov.training_source,
            reset_reason=prov.reset_reason,
            artifact_sha256=manifest.artifact_sha256,
            retained_generation=manifest.training.retained_generation,
        ),
        resolution,
    )


def assert_provenance_matches_the_served_artifact(resolution: Resolution) -> None:
    """The manifest describes the bytes actually serving — verified, not believed.

    Two independent checks, because either alone can be satisfied by a lie:

    1. The digest recomputed over the file on disk equals the one recorded. A
       manifest that had drifted from its artifact fails here.
    2. The model object in memory behaves identically to one deserialized from
       those same bytes. A digest that matched a file nobody loaded would pass
       (1) and fail this.
    """
    assert resolution.artifact is not None
    assert resolution.manifest is not None
    payload = resolution.artifact.read_bytes()
    assert _sha256(payload) == resolution.manifest.artifact_sha256, (
        f"{resolution.artifact} does not hash to the digest its manifest records; "
        "provenance and the served artifact disagree"
    )

    from_disk = joblib.load(io.BytesIO(payload))
    probe = np.zeros((1, len(STUCK_FEATURE_NAMES)))
    np.testing.assert_allclose(resolution.model.predict_proba(probe), from_disk.predict_proba(probe))


# ---------------------------------------------------------------------------
# A file-activity recorder, for "writes stay inside retained_data_dir()"
# ---------------------------------------------------------------------------


@dataclass
class FileActivity:
    """Every path this process opened for writing, renamed, created or removed."""

    writes: list[str] = field(default_factory=list)

    def matching(self, suffix: str) -> list[str]:
        return sorted({p for p in self.writes if p.endswith(suffix)})


_activity: FileActivity | None = None
_file_hook_installed = False

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_TRUNC", 0)


def _decode(value: Any) -> str | None:
    if isinstance(value, (str, bytes)):
        return os.fsdecode(value)
    if isinstance(value, Path):
        return str(value)
    return None


def _file_audit(event: str, args: tuple[Any, ...]) -> None:
    """Record write intent at the interpreter level. Never raises, never blocks."""
    activity = _activity
    if activity is None:
        return
    if event == "open":
        path, mode, flags = args
        decoded = _decode(path)
        if decoded is None:
            return
        writing = any(ch in mode for ch in "wxa+") if isinstance(mode, str) else bool(flags & _WRITE_FLAGS)
        if writing:
            activity.writes.append(decoded)
    elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.truncate"}:
        decoded = _decode(args[0])
        if decoded is not None:
            activity.writes.append(decoded)
    elif event == "os.rename":  # also raised by os.replace
        for value in args[:2]:
            decoded = _decode(value)
            if decoded is not None:
                activity.writes.append(decoded)


@contextmanager
def record_file_writes() -> Iterator[FileActivity]:
    """Record every write the interpreter performs inside the block."""
    global _activity, _file_hook_installed
    if not _file_hook_installed:
        sys.addaudithook(_file_audit)
        _file_hook_installed = True
    previous = _activity
    recorder = FileActivity()
    _activity = recorder
    try:
        yield recorder
    finally:
        _activity = previous


# ---------------------------------------------------------------------------
# The full local flow the no-egress guarantee covers
# ---------------------------------------------------------------------------


def run_full_local_flow(slots: dict[str, Path]) -> dict[str, Any]:
    """Training, retention, eviction, manifest write, migration and refresh.

    Every write path this mission introduced, in one call, so the egress
    assertion covers the flow rather than a sample of it.
    """
    local, base, retained_dir = slots["local"], slots["base"], slots["retained"]
    contract = local_feature_contract(MODEL)
    assert contract is not None

    out: dict[str, Any] = {}

    # 1. Local training — writes artifacts, retains examples, writes manifests.
    out["train"] = _train_locally(local)
    out["retained_after_training"] = len(read_retained(MODEL, directory=retained_dir).examples)

    # 2. Eviction — an append past the cap, forcing the rewrite path. Labels
    #    alternate so the surviving suffix still carries both classes and the
    #    rebuild below exercises retraining rather than a scikit-learn refusal.
    extra = [
        Example(x=tuple(float(i + n) for i in range(len(contract.names))), y=float(n % 2), as_of_ms=BASE_MS)
        for n in range(40)
    ]
    out["evicting_append"] = append_examples(MODEL, extra, contract, directory=retained_dir, max_bytes=2048)
    out["evicted"] = out["evicting_append"].evicted

    # 3. The pre-registry migration — an artifact with its manifest removed.
    (local / f"{MODEL}.json").unlink()
    out["migrated"] = FilesystemModelLoader(base_dir=local).load(TENANT, MODEL) is not None

    # 4. Refresh against a newly shipped base, same contract — the rebuild path.
    ship_base(base, MODEL, "2", seed=7, contract=contract)
    out["refresh"] = refresh_model(MODEL, local_dir=local, base_dir=base, retained_dir=retained_dir)

    # 5. Inspection and deletion (FR-018).
    out["summary"] = summarize_retained(MODEL, directory=retained_dir)
    out["deleted"] = delete_retained(MODEL, directory=retained_dir)
    return out


# ===========================================================================
# T022 — the no-base default state (SC-008)
# ===========================================================================


class TestNoBaseDefaultStateIsUnchanged:
    """Empty base slot, empty local slot: today's state, and it must not move.

    Every assertion here compares against :data:`PRE_MISSION` — values captured
    by running the same operations against commit ``827058c``. See the module
    docstring.
    """

    def test_the_base_slot_is_absent_and_nothing_creates_it(self, slots: dict[str, Path]) -> None:
        """D-001: `base_models_dir()` returns a path; it does not make one."""
        assert not slots["base"].exists()
        assert config.base_models_dir() == slots["base"]
        assert not slots["base"].exists()

        # Neither does resolving through it.
        resolve_model(MODEL, local_dir=slots["local"], base_dir=slots["base"])
        assert not slots["base"].exists()

    def test_the_real_base_models_dir_creates_nothing_in_either_distribution_form(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The genuine `config.base_models_dir()`, with no stub in the way.

        Every other test here substitutes that function, which means none of
        them would notice a `mkdir` appearing inside it — and inside a signed
        PyInstaller bundle that `mkdir` fails rather than helps (D-001). So this
        one calls the real implementation, down both of its branches.
        """
        # Source install: resolves beside the package, where nothing is shipped
        # yet, and calling it must leave that true.
        monkeypatch.delattr(sys, "frozen", raising=False)
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)
        packaged = config.base_models_dir()
        assert packaged == Path(config.__file__).resolve().parent / config.BASE_MODELS_DIRNAME
        assert not packaged.exists(), (
            f"{packaged} exists — no base model has shipped, so either one was committed "
            "or base_models_dir() created it"
        )

        # Frozen bundle: resolves under _MEIPASS, and must not create that either.
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
        frozen = config.base_models_dir()
        assert frozen == tmp_path / "bundle" / "kenaz_ml" / config.BASE_MODELS_DIRNAME
        assert not frozen.exists()
        assert not (tmp_path / "bundle").exists()

    def test_both_slots_empty_reaches_cold_start(self, slots: dict[str, Path]) -> None:
        """FR-004's third step, which is the only step reachable today."""
        assert sorted(p.name for p in slots["local"].iterdir()) == PRE_MISSION["local_slot_files_before"]

        for name in ROSTER:
            resolution = resolve_model(name, local_dir=slots["local"], base_dir=slots["base"])
            assert resolution.cold_start, f"{name} did not reach cold start"
            assert resolution.slot == SLOT_COLD_START
            assert resolution.model is None
            # Both slots reported, both as "empty" rather than as a failure.
            assert resolution.reasons == ("slot_empty", "slot_empty"), name

    def test_cold_start_predictions_are_what_they_were_before_the_mission(self, slots: dict[str, Path]) -> None:
        """Not "it returned something" — it returned *these* values."""
        store = LocalModelStore(base_dir=slots["local"])
        expected = PRE_MISSION["cold_start_predictions"]

        stuck = StuckPredictor(model_store=store)
        assert stuck.is_trained is expected["stuck"]["is_trained"]
        assert _rounded(stuck.predict(STUCK_FEATURES)) == expected["stuck"]["predict"]

        duration = DurationEstimator(model_store=store)
        assert duration.is_trained is expected["duration"]["is_trained"]
        assert _rounded(duration.predict(DURATION_FEATURES)) == expected["duration"]["predict"]

        quality = QualityEstimator(model_store=store)
        assert _rounded(quality.predict(QUALITY_FEATURES)) == expected["quality"]["predict"]

    def test_the_loader_reports_absence_for_every_model(self, slots: dict[str, Path]) -> None:
        """`load()` returns None, for all five, and raises for none (FR-017)."""
        loader = FilesystemModelLoader(base_dir=slots["local"])
        observed = {name: ("None" if loader.load(TENANT, name) is None else "MODEL") for name in ROSTER}
        assert observed == PRE_MISSION["loader_cold_start"]

    def test_training_produces_exactly_the_models_it_produced_before(self, slots: dict[str, Path]) -> None:
        """Same models, same sample count, same artifacts on disk."""
        summary = _train_locally(slots["local"])
        assert sorted(summary["trained"]) == PRE_MISSION["train_all"]["trained"]
        assert summary["samples"] == PRE_MISSION["train_all"]["samples"]
        assert sorted(p.name for p in slots["local"].glob("*.joblib")) == PRE_MISSION["artifacts_after_training"]

    def test_predictions_after_training_are_unchanged(self, slots: dict[str, Path]) -> None:
        """The trained model's actual output, not merely that one exists."""
        _train_locally(slots["local"])
        store = LocalModelStore(base_dir=slots["local"])
        expected = PRE_MISSION["trained_predictions"]

        stuck = StuckPredictor(model_store=store)
        assert stuck.is_trained is True
        assert _rounded(stuck.predict(STUCK_FEATURES)) == expected["stuck"]["predict"]

        duration = DurationEstimator(model_store=store)
        assert duration.is_trained is True
        assert _rounded(duration.predict(DURATION_FEATURES)) == expected["duration"]["predict"]

    def test_the_loader_serves_the_models_training_wrote(self, slots: dict[str, Path]) -> None:
        """Registry validation must not refuse what this install just trained."""
        _train_locally(slots["local"])
        loader = FilesystemModelLoader(base_dir=slots["local"])
        observed = {name: ("None" if loader.load(TENANT, name) is None else "MODEL") for name in ROSTER}
        assert observed == PRE_MISSION["loader_after_training"]

    @pytest.mark.parametrize("label", [call[0] for call in ENDPOINT_CALLS])
    def test_prediction_endpoints_return_what_they_returned_before(self, slots: dict[str, Path], label: str) -> None:
        """Driven over HTTP through the real app, in the default no-base state."""
        from kenaz_ml.app import create_app

        _, method, path, body = next(call for call in ENDPOINT_CALLS if call[0] == label)
        with TestClient(create_app()) as client:
            response = client.request(method, path) if body is None else client.request(method, path, json=body)

        expected = PRE_MISSION["endpoints_cold_start"][label]
        assert response.status_code == expected["status"]
        assert _strip_volatile(_rounded(response.json())) == expected["json"]

    def test_the_default_state_logs_nothing_that_would_alarm_a_user(
        self, slots: dict[str, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        """An empty slot is universal, so it is DEBUG — not WARNING (WP02).

        Every install in the world is in this state. A warning here would be a
        permanent, unactionable line in front of every user, and would train
        them to ignore the level a genuinely broken release needs.
        """
        with caplog.at_level(logging.DEBUG, logger="kenaz_ml"):
            for name in ROSTER:
                resolve_model(name, local_dir=slots["local"], base_dir=slots["base"])
            FilesystemModelLoader(base_dir=slots["local"]).load(TENANT, MODEL)
            _train_locally(slots["local"])

        alarming = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert alarming == [], "the default no-base state logged at WARNING or above:\n  " + "\n  ".join(
            f"{r.levelname} {r.name}: {r.getMessage()}" for r in alarming
        )

        # Vacuity check: the empty slot really was reported, just quietly.
        empty = [r for r in caplog.records if "slot_empty" in r.getMessage()]
        assert empty, "no empty-slot refusal was recorded at all"
        assert {r.levelno for r in empty} == {logging.DEBUG}


class TestEmptyBaseWithPopulatedLocal:
    """An install that has been training for a while, still with no base shipped.

    The second half of the default state, and the one a long-lived install is
    actually in. The local model must serve and refresh must do nothing at all.
    """

    @pytest.fixture
    def trained(self, slots: dict[str, Path]) -> dict[str, Path]:
        _train_locally(slots["local"])
        return slots

    def test_the_local_slot_serves(self, trained: dict[str, Path]) -> None:
        resolution = resolve_model(MODEL, local_dir=trained["local"], base_dir=trained["base"])
        assert resolution.served
        assert resolution.slot == SLOT_LOCAL
        assert resolution.model is not None
        # The base slot was never even reached, so it produced no refusal.
        assert resolution.reasons == ()
        assert_provenance_matches_the_served_artifact(resolution)

    def test_detection_reports_no_base_rather_than_a_change(self, trained: dict[str, Path]) -> None:
        change = detect_base_change(MODEL, local_dir=trained["local"], base_dir=trained["base"])
        assert change.due is False
        assert change.reason == REASON_NO_BASE
        assert change.base_version is None

    def test_refresh_is_a_no_op_and_writes_nothing(self, trained: dict[str, Path]) -> None:
        """Byte-for-byte: a no-op refresh may not so much as rewrite a manifest."""
        artifact = trained["local"] / f"{MODEL}.joblib"
        manifest = trained["local"] / f"{MODEL}.json"
        before = (artifact.read_bytes(), manifest.read_bytes())
        retained_before = retained_path(MODEL, directory=trained["retained"]).read_bytes()

        result = refresh_model(
            MODEL, local_dir=trained["local"], base_dir=trained["base"], retained_dir=trained["retained"]
        )

        assert result.ok is True
        assert result.action == ACTION_NONE
        assert result.changed is False
        assert (artifact.read_bytes(), manifest.read_bytes()) == before
        assert retained_path(MODEL, directory=trained["retained"]).read_bytes() == retained_before
        assert not trained["base"].exists(), "a no-op refresh created the base slot"

    def test_the_populated_install_still_predicts(self, trained: dict[str, Path]) -> None:
        store = LocalModelStore(base_dir=trained["local"])
        assert (
            _rounded(StuckPredictor(model_store=store).predict(STUCK_FEATURES))
            == (PRE_MISSION["trained_predictions"]["stuck"]["predict"])
        )


# ===========================================================================
# T023 — no egress (SC-006, FR-011, C-003)
# ===========================================================================


class TestTheGuardIsNotVacuous:
    """A guard that cannot fail proves nothing. These make it fail on purpose."""

    def test_a_socket_opened_inside_the_retention_path_is_caught(
        self, slots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Egress injected into the code the guarantee is *about*.

        `retained._now_ms` is called while stamping a retention header, so this
        puts a socket on the exact path FR-011 covers rather than somewhere
        convenient.
        """
        import socket

        from kenaz_ml.modelstore.registry import retained as retained_mod

        def leaking_now_ms() -> int:
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            return BASE_MS

        monkeypatch.setattr(retained_mod, "_now_ms", leaking_now_ms)
        contract = local_feature_contract(MODEL)
        assert contract is not None

        with no_network() as recorder, pytest.raises(EgressAttempted):
            append_examples(
                MODEL,
                [Example(x=(0.0,) * len(contract.names), y=0.0, as_of_ms=BASE_MS)],
                contract,
                directory=slots["retained"],
            )
        assert recorder.socket_attempts, "the guard let a socket through the retention path"

    def test_egress_inside_the_full_flow_is_recorded_even_when_swallowed(
        self, slots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The strongest vacuity proof available: break the real flow.

        `Trainer._retain_examples` wraps retention in `except Exception` — by
        design, since failing to retain must not fail a training run that
        succeeded. So an exception raised by the guard is *swallowed by
        production code* and never reaches the test. If the assertion in
        :meth:`TestNoEgress.test_the_full_local_flow_opens_no_socket` depended on
        that exception propagating, it would pass while the flow leaked. It does
        not: the recorder is checked, and the recorder sees it regardless.
        """
        import socket

        from kenaz_ml.modelstore.registry import retained as retained_mod

        real_now_ms = retained_mod._now_ms

        def leaking_now_ms() -> int:
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            return real_now_ms()

        monkeypatch.setattr(retained_mod, "_now_ms", leaking_now_ms)

        with no_network() as recorder:
            try:
                run_full_local_flow(slots)
            except EgressAttempted:
                pass  # whether it escapes is the point; either way it is recorded

        assert recorder.socket_attempts, (
            "egress injected into the retention path of the full local flow was not caught — "
            "the no-egress assertion is vacuous"
        )

    def test_a_socket_built_below_the_socket_module_is_caught(self, slots: dict[str, Path]) -> None:
        """`_socket.socket()` bypasses the `socket` module entirely.

        A monkeypatch of `socket.socket` would miss this, and so would miss a C
        extension in Feast's tree doing the same thing. The audit hook does not.
        """
        import _socket

        with no_network() as recorder, pytest.raises(EgressAttempted):
            _socket.socket()
        assert recorder.socket_attempts[0].startswith("socket.__new__")

    def test_a_pre_captured_socket_reference_is_caught(self, slots: dict[str, Path]) -> None:
        """A reference grabbed before the guard armed is still caught."""
        from socket import socket as captured_before_the_guard

        with no_network() as recorder, pytest.raises(EgressAttempted):
            captured_before_the_guard()
        assert recorder.socket_attempts

    def test_name_resolution_alone_is_caught(self) -> None:
        """Egress does not require a connection; a DNS lookup already leaks."""
        import socket

        with no_network() as recorder, pytest.raises(EgressAttempted):
            socket.getaddrinfo("exfil.invalid", 443)
        assert any("getaddrinfo" in attempt for attempt in recorder.socket_attempts)

    def test_the_write_recorder_sees_the_retention_file(self, slots: dict[str, Path]) -> None:
        """The other guard used below must not be vacuous either."""
        contract = local_feature_contract(MODEL)
        assert contract is not None
        with record_file_writes() as activity:
            append_examples(
                MODEL,
                [Example(x=(0.0,) * len(contract.names), y=0.0, as_of_ms=BASE_MS)],
                contract,
                directory=slots["retained"],
            )
        assert activity.matching(".jsonl") or activity.matching(".jsonl.tmp"), "the write recorder saw nothing"


class TestNoEgress:
    """SC-006. Not "no known uploader ran" — no socket existed, by any path."""

    def test_the_full_local_flow_opens_no_socket(self, slots: dict[str, Path]) -> None:
        """Training, retention, eviction, manifest write, migration, refresh."""
        with no_network() as recorder:
            out = run_full_local_flow(slots)

        assert recorder.socket_attempts == [], "the local registry flow attempted network access:\n  " + "\n  ".join(
            recorder.socket_attempts
        )
        # Vacuity: the flow really did all of it, rather than failing early and
        # being quiet because nothing happened.
        assert out["train"]["samples"] > 0
        assert out["retained_after_training"] >= 10
        assert out["evicted"] > 0, "the eviction path did not run"
        assert out["migrated"] is True, "the pre-registry migration did not run"
        assert out["refresh"].action == ACTION_REBUILD, out["refresh"].refusal
        assert out["deleted"] is True

    def test_the_flow_succeeds_with_the_network_genuinely_unavailable(self, slots: dict[str, Path]) -> None:
        """The guard *raises* on a socket, so the flow ran with egress impossible.

        This distinguishes "did not need the network" from "happened to be
        offline": every path that would have connected failed here instead of
        succeeding quietly on a developer machine that is online.
        """
        with no_network():
            out = run_full_local_flow(slots)
        assert out["refresh"].ok is True
        assert out["summary"].readable is True

    def test_retained_writes_stay_inside_the_retained_data_dir(self, slots: dict[str, Path]) -> None:
        """FR-011/C-003: nothing carrying retained data is written elsewhere."""
        retained_dir = config.retained_data_dir().resolve()
        assert retained_dir == slots["retained"].resolve()
        assert retained_path(MODEL, directory=None).parent.resolve() == retained_dir
        assert retained_dir.is_relative_to(config.models_dir().resolve())

        with record_file_writes() as activity, no_network():
            run_full_local_flow(slots)

        jsonl_writes = [Path(p).resolve() for p in activity.writes if ".jsonl" in Path(p).name]
        assert jsonl_writes, "no retention write was recorded at all"
        escaped = [p for p in jsonl_writes if p.parent != retained_dir]
        assert escaped == [], "retained training data was written outside retained_data_dir():\n  " + "\n  ".join(
            str(p) for p in escaped
        )

    def test_no_retained_data_lands_anywhere_else_on_disk(self, slots: dict[str, Path]) -> None:
        """Checked against the filesystem, not only against recorded intent."""
        with no_network():
            _train_locally(slots["local"])

        found = sorted(p for p in slots["root"].rglob("*.jsonl") if p.is_file())
        assert found, "training retained nothing, so this assertion would be empty"
        for path in found:
            assert path.parent.resolve() == slots["retained"].resolve(), f"{path} is outside retained_data_dir()"

    def test_the_registry_imports_nothing_that_can_open_a_socket(self) -> None:
        """Structural, so a future edit fails here rather than at runtime.

        Reads every import in the registry package — module scope *and* inside
        functions, since the registry defers most of its imports — and asserts
        none of them names a module that can reach the network. This is what
        stops someone adding `import urllib.request` to the retention writer and
        having it pass every behavioural test because no test happened to reach
        the new branch.

        Deliberately scoped to the registry package plus `kenaz_ml.config`,
        which is all of it that runs on the retention path. The wider tree —
        Feast, and `modelstore.stores`, whose `S3ModelStore` imports `boto3`
        inside its constructor for the *cloud* deployment — is covered by
        :meth:`test_the_full_local_flow_opens_no_socket` instead, which asserts
        the stronger thing: that none of it opens a socket when actually run.
        """
        from kenaz_ml.modelstore import registry

        forbidden = {
            "_socket",
            "asyncio",
            "boto3",
            "botocore",
            "fastapi",
            "ftplib",
            "http",
            "httpx",
            "requests",
            "socket",
            "socketserver",
            "ssl",
            "telnetlib",
            "urllib.error",
            "urllib.request",
            "uvicorn",
            "webbrowser",
            "xmlrpc",
        }
        modules = [f"{registry.__name__}.{sub}" for sub in ("manifest", "slots", "retained", "refresh")]
        modules += [registry.__name__, "kenaz_ml.config"]
        offences: list[str] = []
        inspected = 0

        for module_name in modules:
            module = sys.modules.get(module_name)
            assert module is not None and getattr(module, "__file__", None), f"{module_name} was not importable"
            inspected += 1
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    if name in forbidden or name.split(".")[0] in forbidden:
                        offences.append(f"{module_name} imports {name}")

        assert inspected == len(modules)
        assert offences == [], "the registry reaches network-capable code:\n  " + "\n  ".join(offences)


class TestNoConfigurationEnablesTransmission:
    """C-003: no configuration of the *local* deployment can transmit retained data."""

    #: Every environment variable `kenaz_ml.config` reads. Extracted from the
    #: source rather than listed by hand, so a new one cannot be added without
    #: appearing here.
    #:
    #: Two call shapes count, because FR-014 introduced a second one. Most reads
    #: now go through `config.env("KENAZ_X")`, the deprecation shim that falls
    #: back to the pre-rebrand `SIGIL_X`; a few variables this product does not
    #: own (`SIGILD_PLUGIN_URL`, `XDG_DATA_HOME`) are still read directly via
    #: `os.environ.get`. Both are collected, or the shim would have quietly
    #: emptied this guard's view of the configuration surface.
    #:
    #: The shim's own `os.environ.get(name)` reads inside `env()` are invisible
    #: here by construction: their argument is a variable, not a constant.
    @staticmethod
    def _config_env_vars() -> list[str]:
        tree = ast.parse(Path(config.__file__).read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "env":
                if node.args and isinstance(node.args[0], ast.Constant):
                    names.add(str(node.args[0].value))
                continue
            if isinstance(func, ast.Attribute) and func.attr in {"get", "environ"}:
                owner = func.value
                is_environ = (isinstance(owner, ast.Attribute) and owner.attr == "environ") or (
                    isinstance(owner, ast.Name) and owner.id == "environ"
                )
                if is_environ and node.args and isinstance(node.args[0], ast.Constant):
                    names.add(str(node.args[0].value))
        return sorted(names)

    def test_the_extractor_sees_both_call_shapes(self) -> None:
        """Guard the guard: if `env()` reads stopped being collected, the set
        above would shrink silently and every knob would look reviewed."""
        found = set(self._config_env_vars())
        assert "KENAZ_POSTGRES_URL" in found, "reads via the FR-014 shim must be collected"
        assert "SIGILD_PLUGIN_URL" in found, "direct os.environ.get reads must still be collected"

    def test_the_environment_surface_is_the_one_we_think_it_is(self) -> None:
        """A new configuration knob must be considered, not silently inherited."""
        assert set(self._config_env_vars()) == {
            "AWS_REGION",
            "SIGILD_PLUGIN_URL",
            "KENAZ_MODEL_CACHE_TTL",
            "KENAZ_MODE",
            "KENAZ_ML_MODE",
            "KENAZ_POSTGRES_URL",
            "KENAZ_S3_BUCKET",
            "KENAZ_S3_ENDPOINT_URL",
            "KENAZ_TENANT",
            "XDG_DATA_HOME",
        }

    @pytest.mark.parametrize(
        "variable",
        [
            "AWS_REGION",
            "SIGILD_PLUGIN_URL",
            "KENAZ_MODEL_CACHE_TTL",
            "KENAZ_POSTGRES_URL",
            "KENAZ_S3_BUCKET",
            "KENAZ_S3_ENDPOINT_URL",
            "KENAZ_TENANT",
        ],
    )
    def test_no_setting_redirects_retained_data_off_the_filesystem(
        self, slots: dict[str, Path], monkeypatch: pytest.MonkeyPatch, variable: str
    ) -> None:
        """Point each knob at a remote host; retention must stay on this disk.

        `KENAZ_MODE` and `KENAZ_ML_MODE` are excluded deliberately: they select
        the *deployment*, and the cloud deployment is the documented, gated
        exception. Every other knob is one the open-source local install can
        carry, and none of them may move a byte of retained data.
        """
        monkeypatch.setenv(variable, "https://exfil.invalid:443/collect")
        contract = local_feature_contract(MODEL)
        assert contract is not None

        with no_network() as recorder:
            result = append_examples(
                MODEL,
                [Example(x=(1.0,) * len(contract.names), y=1.0, as_of_ms=BASE_MS)],
                contract,
                directory=None,
            )

        assert recorder.socket_attempts == []
        assert result.ok is True
        assert result.path.parent.resolve() == config.retained_data_dir().resolve()
        assert result.path.resolve().is_relative_to(slots["root"].resolve())

    def test_the_retention_writer_takes_no_destination_but_a_directory(self) -> None:
        """There is no URL, host or endpoint parameter to point anywhere.

        A configuration key can only cause transmission if some function accepts
        a destination. The retained-set API accepts a directory and nothing else.
        """
        import inspect

        from kenaz_ml.modelstore.registry import retained as retained_mod

        forbidden = {"url", "host", "endpoint", "bucket", "uri", "remote", "upload", "sink"}
        for name, function in inspect.getmembers(retained_mod, inspect.isfunction):
            if function.__module__ != retained_mod.__name__:
                continue
            for parameter in inspect.signature(function).parameters:
                assert parameter.lower() not in forbidden, (
                    f"retained.{name} accepts a destination parameter {parameter}"
                )


# ===========================================================================
# T024 — end-to-end provenance (SC-007)
# ===========================================================================


class TestProvenanceLifecycle:
    """One install, five stages, and provenance answerable at every one.

    The stages run in sequence inside a single test rather than as five
    independent ones on purpose: provenance is a property of *history*, and a
    stage asserted against a hand-built fixture would not prove the history
    carried forward. Each stage asserts, then hands the state to the next.
    """

    def test_the_whole_lifecycle(self, slots: dict[str, Path]) -> None:
        local, base, retained_dir = slots["local"], slots["base"], slots["retained"]
        contract = local_feature_contract(MODEL)
        assert contract is not None
        contract_version = contract.service_version

        # -- Stage 1: fresh install, no models anywhere. -------------------
        report, resolution = provenance_of(MODEL, local=local, base=base)
        assert report.served is False
        assert report.slot == SLOT_COLD_START
        assert report.base_version is None
        assert report.n_local_extensions is None
        assert report.contract_version is None
        assert resolution.model is None

        # -- Stage 2: a base is shipped and never extended. ----------------
        base_v1 = ship_base(base, MODEL, "1", seed=11, contract=contract)
        report, resolution = provenance_of(MODEL, local=local, base=base)
        assert report.served is True
        assert report.slot == SLOT_BASE
        assert report.answers_sc007
        assert report.base_version == "1"
        assert report.base_sha256 == base_v1.artifact_sha256
        assert report.n_local_extensions == 0
        assert report.contract_version == contract_version
        assert report.training_source == TRAINING_SOURCE_BASE
        assert report.reset_reason is None
        assert_provenance_matches_the_served_artifact(resolution)

        # -- Stage 3: extended locally, three times. -----------------------
        for expected_extensions in (1, 2, 3):
            _train_locally(local)
            report, resolution = provenance_of(MODEL, local=local, base=base)
            assert report.served is True
            assert report.slot == SLOT_LOCAL, "a local model must win over the base slot (FR-004)"
            assert report.answers_sc007
            assert report.base_version == "1", "the local model lost track of which base it descends from"
            assert report.base_sha256 == base_v1.artifact_sha256
            assert report.n_local_extensions == expected_extensions
            assert report.contract_version == contract_version
            assert report.training_source == TRAINING_SOURCE_LOCAL
            assert report.reset_reason is None
            assert_provenance_matches_the_served_artifact(resolution)

        retained = read_retained(MODEL, directory=retained_dir)
        assert retained.contract_version == contract_version
        assert len(retained.examples) >= 3 * 10
        generation_before_refresh = retained.generation

        # -- Stage 4: a new base, same contract, refreshed. -----------------
        base_v2 = ship_base(base, MODEL, "2", seed=23, contract=contract)
        change = detect_base_change(MODEL, local_dir=local, base_dir=base)
        assert change.due is True
        assert change.local_base_version == "1"
        assert change.base_version == "2"

        result = refresh_model(MODEL, local_dir=local, base_dir=base, retained_dir=retained_dir)
        assert result.ok is True, result.refusal
        assert result.action == ACTION_REBUILD
        assert result.n_samples == len(retained.examples)
        assert result.retained_generation == generation_before_refresh

        report, resolution = provenance_of(MODEL, local=local, base=base)
        assert report.served is True
        assert report.slot == SLOT_LOCAL
        assert report.answers_sc007
        assert report.base_version == "2", "the rebuilt model does not descend from the new base"
        assert report.base_sha256 == base_v2.artifact_sha256
        assert report.n_local_extensions == 1, "a rebuild is one extension from the base beside it"
        assert report.contract_version == contract_version
        assert report.training_source == TRAINING_SOURCE_LOCAL
        assert report.reset_reason is None
        assert report.retained_generation == generation_before_refresh
        assert_provenance_matches_the_served_artifact(resolution)

        # -- Stage 5: a new base whose contract moved. ----------------------
        # The retained vectors were computed under the previous contract, so
        # they cannot be replayed. Stamping them that way is the realistic
        # shape of an upgrade that moved the feature set.
        stale = FeatureContract(
            service=contract.service,
            service_version=f"{contract_version}-previous",
            names=contract.names,
            dtypes=contract.dtypes,
        )
        reset_retained(MODEL, contract=stale, directory=retained_dir)
        append_examples(
            MODEL,
            [Example(x=(1.0,) * len(contract.names), y=1.0, as_of_ms=BASE_MS) for _ in range(12)],
            stale,
            directory=retained_dir,
        )
        stale_generation = read_retained(MODEL, directory=retained_dir).generation

        base_v3 = ship_base(base, MODEL, "3", seed=37, contract=contract)
        result = refresh_model(MODEL, local_dir=local, base_dir=base, retained_dir=retained_dir)
        assert result.ok is True, result.refusal
        assert result.action == ACTION_RESET
        assert result.reset_reason == RESET_REASON_CONTRACT_CHANGED
        assert result.previous_generation == stale_generation
        assert result.retained_generation != stale_generation

        report, resolution = provenance_of(MODEL, local=local, base=base)
        assert report.served is True
        assert report.slot == SLOT_LOCAL
        assert report.answers_sc007
        assert report.base_version == "3"
        assert report.base_sha256 == base_v3.artifact_sha256
        assert report.n_local_extensions == 0, "a reset serves the base unextended"
        assert report.contract_version == contract_version
        assert report.training_source == TRAINING_SOURCE_BASE
        assert report.reset_reason == RESET_REASON_CONTRACT_CHANGED
        assert_provenance_matches_the_served_artifact(resolution)

        # The reset is honest about what it cost: the served bytes are the
        # shipped base's, byte for byte (SC-005), and the retained set restarted.
        assert report.artifact_sha256 == base_v3.artifact_sha256
        assert (local / f"{MODEL}.joblib").read_bytes() == (base / f"{MODEL}.joblib").read_bytes()
        assert read_retained(MODEL, directory=retained_dir).examples == ()

    def test_a_base_with_nothing_retained_is_adopted_not_reset(self, slots: dict[str, Path]) -> None:
        """User Story 3 scenario 3 — adoption is a success, and says so.

        `adopt_base` and `reset` both serve the base unextended, so the only
        thing telling an operator whether personalization was *lost* is whether
        `reset_reason` is recorded. It must not be, here.
        """
        local, base, retained_dir = slots["local"], slots["base"], slots["retained"]
        contract = local_feature_contract(MODEL)
        assert contract is not None

        ship_base(base, MODEL, "1", seed=5, contract=contract)
        _train_locally(local)
        delete_retained(MODEL, directory=retained_dir)

        ship_base(base, MODEL, "2", seed=6, contract=contract)
        result = refresh_model(MODEL, local_dir=local, base_dir=base, retained_dir=retained_dir)
        assert result.ok is True, result.refusal
        assert result.action == ACTION_ADOPT_BASE

        report, resolution = provenance_of(MODEL, local=local, base=base)
        assert report.base_version == "2"
        assert report.n_local_extensions == 0
        assert report.training_source == TRAINING_SOURCE_BASE
        assert report.reset_reason is None, "adoption was recorded as a lost personalization"
        assert_provenance_matches_the_served_artifact(resolution)

    def test_provenance_refuses_an_artifact_its_manifest_does_not_describe(self, slots: dict[str, Path]) -> None:
        """The digest check is load-bearing, not decorative.

        If provenance could disagree with the served artifact, every assertion
        above would be worthless. Corrupt the artifact behind a good manifest
        and nothing may be served.
        """
        local, base = slots["local"], slots["base"]
        contract = local_feature_contract(MODEL)
        assert contract is not None
        ship_base(base, MODEL, "1", seed=3, contract=contract)

        report, resolution = provenance_of(MODEL, local=local, base=base)
        assert report.served is True
        assert_provenance_matches_the_served_artifact(resolution)

        artifact = base / f"{MODEL}.joblib"
        artifact.write_bytes(artifact.read_bytes() + b"tampered")

        report, resolution = provenance_of(MODEL, local=local, base=base)
        assert report.served is False, "an artifact that does not match its manifest was served anyway"
        assert resolution.cold_start
        assert "digest_mismatch" in resolution.reasons
