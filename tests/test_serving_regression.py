"""Serving non-regression suite (WP04 — FR-009, SC-003, NFR-002).

WP01 added a keyword-only ``as_of_ms`` reference time to the feature
extractors. The serving path (``poller.py``, ``routes.py``) deliberately never
passes it, so serving must behave exactly as it did before WP01 landed.

"Unchanged by construction" is a claim; this file is the evidence. It does
three things:

1. **Baseline equality.** The ``BASELINE_*`` dicts near the top are *recorded
   observations* of the pre-WP01 implementation — see the provenance block
   above them. Every serving-shaped call is asserted equal to them, key
   ordering included.
2. **End-to-end.** The prediction routes are exercised through a real
   ``TestClient`` against a real ``SqliteStore``, so the assertion covers the
   whole HTTP -> route -> extractor -> store path rather than the extractor
   alone.
3. **Signature guards.** ``as_of_ms`` must stay keyword-only with a ``None``
   default on every extractor that has it, so no future change can silently
   make it required and break the serving call sites (which pass it
   positionally-free and by omission).

Note on style: the serving call sites intentionally omit ``as_of_ms``. Tests
here do the same — passing ``as_of_ms=None`` explicitly would not exercise the
same code shape the poller and routes use.
"""

from __future__ import annotations

import inspect
import json
import os
import sqlite3
import time as _real_time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from kenaz_ml import features
from kenaz_ml.datastore.sqlite import SqliteStore
from kenaz_ml.features import (
    extract_activity_features,
    extract_duration_features,
    extract_duration_features_from_data,
    extract_features_from_buffer,
    extract_stuck_features,
    extract_stuck_features_from_data,
    extract_workflow_features,
)

# ---------------------------------------------------------------------------
# Pinned clock
# ---------------------------------------------------------------------------

# 2026-01-15T12:34:56Z. Any instant would do; this one is fixed forever so the
# recorded baselines below stay meaningful.
NOW_MS = 1768480496000
NOW_SEC = NOW_MS / 1000.0


class _FrozenTime:
    """Stand-in for the ``time`` module as seen from ``kenaz_ml.features``.

    ``features`` does ``import time`` and calls ``time.time()`` /
    ``time.localtime()``. Swapping the module attribute is narrower than
    patching the stdlib: nothing outside ``features`` observes a frozen clock.
    """

    def __init__(self, now_ms: int) -> None:
        self._now_sec = now_ms / 1000.0

    def time(self) -> float:
        return self._now_sec

    def localtime(self, secs: float | None = None) -> _real_time.struct_time:
        return _real_time.localtime(self._now_sec if secs is None else secs)

    def __getattr__(self, name: str) -> Any:
        return getattr(_real_time, name)


@contextmanager
def frozen_clock(now_ms: int = NOW_MS) -> Iterator[None]:
    """Pin the wall clock that ``kenaz_ml.features`` reads."""
    previous = features.time
    features.time = _FrozenTime(now_ms)  # type: ignore[assignment]
    try:
        yield
    finally:
        features.time = previous  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _pinned_timezone() -> Iterator[None]:
    """Pin TZ to UTC.

    ``time_of_day_hour`` is derived with ``time.localtime()``, so the duration
    baselines are only reproducible under a fixed timezone. UTC is what the
    baselines were captured under.
    """
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    if hasattr(_real_time, "tzset"):
        _real_time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        if hasattr(_real_time, "tzset"):
            _real_time.tzset()


# ---------------------------------------------------------------------------
# Fixtures — fixed data, all timestamps strictly in the past relative to NOW_MS
# ---------------------------------------------------------------------------

TASK_A_ID = "task-alpha"

TASK_A: dict[str, Any] = {
    "id": TASK_A_ID,
    "repo_root": "/tmp/repo",
    "branch": "feature/point-in-time",  # 21 characters
    "phase": "implement",
    "files": json.dumps({"src/a.py": 3, "src/b.py": 1}),
    "started_at": NOW_MS - 3_600_000,  # 60 min before NOW
    "last_active": NOW_MS - 600_000,  # 10 min before NOW
    "completed_at": None,
    "commit_count": 2,
    "test_runs": 12,
    "test_fails": 5,
}

# Task-window event vocabulary: phase_change / edit / file_edit / save / commit.
EVENTS_A: list[dict[str, Any]] = [
    {"id": 1, "kind": "commit", "source": "git", "payload": {"hash": "aaa"}, "ts": NOW_MS - 3_000_000},
    {"id": 2, "kind": "edit", "source": "editor", "payload": {"file": "src/a.py"}, "ts": NOW_MS - 2_400_000},
    {"id": 3, "kind": "phase_change", "source": "sigild", "payload": {"phase": "implement"}, "ts": NOW_MS - 1_800_000},
    {"id": 4, "kind": "file_edit", "source": "editor", "payload": {"file": "src/b.py"}, "ts": NOW_MS - 1_500_000},
    {"id": 5, "kind": "save", "source": "editor", "payload": {"file": "src/a.py"}, "ts": NOW_MS - 1_200_000},
    {"id": 6, "kind": "commit", "source": "git", "payload": {"hash": "bbb"}, "ts": NOW_MS - 900_000},
    {
        "id": 7,
        "kind": "terminal",
        "source": "shell",
        "payload": {"cmd": "pytest", "exit_code": 1},
        "ts": NOW_MS - 700_000,
    },
]

# Raw daemon buffer vocabulary: file / git / terminal. This is what the poller
# holds when there is no active task.
BUFFER_A: list[dict[str, Any]] = [
    {"id": 101, "kind": "file", "source": "fs", "payload": {"path": "src/a.py"}, "ts": NOW_MS - 1_200_000},
    {
        "id": 102,
        "kind": "terminal",
        "source": "shell",
        "payload": {"cmd": "pytest", "exit_code": 1},
        "ts": NOW_MS - 1_100_000,
    },
    {"id": 103, "kind": "file", "source": "fs", "payload": {"path": "src/b.py"}, "ts": NOW_MS - 1_000_000},
    {"id": 104, "kind": "git", "source": "git", "payload": {"branch": "main"}, "ts": NOW_MS - 900_000},
    {
        "id": 105,
        "kind": "terminal",
        "source": "shell",
        "payload": {"cmd": "pytest", "exit_code": 0},
        "ts": NOW_MS - 800_000,
    },
    {"id": 106, "kind": "file", "source": "fs", "payload": {"path": "src/a.py"}, "ts": NOW_MS - 600_000},
]


# ---------------------------------------------------------------------------
# Recorded pre-WP01 baselines
# ---------------------------------------------------------------------------
#
# PROVENANCE — read this before changing any number below.
#
# These are not expected values derived from the spec. They are *observations*
# recorded by running the pre-WP01 feature extractor against the fixtures above.
#
#   source module : `git show 5db84f7:src/sigil_ml/features.py`
#                   (5db84f7 is the merge-base of this mission's branch — the
#                   last commit before WP01's point-in-time change landed as
#                   dc64ffe. The pre-change module has no `as_of_ms` parameter
#                   at all, so these were produced by the only call shape it
#                   supported.)
#   clock         : time.time() pinned to NOW_MS (1768480496000)
#   timezone      : TZ=UTC
#   fixtures      : TASK_A / EVENTS_A / BUFFER_A above, verbatim
#
# To re-derive: extract the module at 5db84f7, import it under any name, pin
# the clock and TZ as above, and call the extractors with the fixtures. Every
# number below must come back identical, key order included.
#
# A change to any of these literals is a change to serving behavior. It is
# never a test-maintenance edit.

# extract_stuck_features_from_data(TASK_A, EVENTS_A)
# and extract_stuck_features(store, "task-alpha") — the two agree by design.
BASELINE_STUCK_TASK_A: dict[str, float] = {
    "test_failure_count": 5.0,
    "time_in_phase_sec": 1800.0,
    "edit_velocity": 0.06,
    "file_switch_rate": 0.6666666666666666,
    "session_length_sec": 3000.0,
    "time_since_last_commit_sec": 900.0,
}

# extract_stuck_features(store, "<unknown id>") — task lookup miss.
BASELINE_STUCK_MISSING_TASK: dict[str, float] = {
    "test_failure_count": 0.0,
    "time_in_phase_sec": 0.0,
    "edit_velocity": 0.0,
    "file_switch_rate": 0.0,
    "session_length_sec": 0.0,
    "time_since_last_commit_sec": 0.0,
}

# extract_duration_features_from_data(TASK_A, EVENTS_A)
# and extract_duration_features(store, "task-alpha").
BASELINE_DURATION_TASK_A: dict[str, float] = {
    "file_count": 2.0,
    "total_edits": 3.0,
    "time_of_day_hour": 11.0,
    "branch_name_length": 21.0,
}

# extract_duration_features(store, "<unknown id>") — task lookup miss.
# time_of_day_hour falls back to the *current* local hour, which under the
# pinned clock and TZ=UTC is the hour of NOW_MS.
BASELINE_DURATION_MISSING_TASK: dict[str, float] = {
    "file_count": 0.0,
    "total_edits": 0.0,
    "time_of_day_hour": 12.0,
    "branch_name_length": 0.0,
}

# extract_features_from_buffer(BUFFER_A) — the poller's no-active-task path.
BASELINE_BUFFER_A: dict[str, float] = {
    "test_failure_count": 1.0,
    "time_in_phase_sec": 600.0,
    "edit_velocity": 0.3,
    "file_switch_rate": 0.6666666666666666,
    "session_length_sec": 600.0,
    "time_since_last_commit_sec": 900.0,
}

# extract_features_from_buffer([]) — poller startup, nothing buffered yet.
BASELINE_BUFFER_EMPTY: dict[str, float] = {
    "test_failure_count": 0.0,
    "time_in_phase_sec": 0.0,
    "edit_velocity": 0.0,
    "file_switch_rate": 0.0,
    "session_length_sec": 0.0,
    "time_since_last_commit_sec": 0.0,
}


# ---------------------------------------------------------------------------
# Store fixtures — a real SqliteStore over the sigild schema
# ---------------------------------------------------------------------------


def _write_fixture_db(db_path: Path) -> None:
    """Create the Go-owned tables kenaz-ml reads and load TASK_A / EVENTS_A."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            repo_root TEXT,
            branch TEXT,
            phase TEXT,
            files TEXT,
            started_at INTEGER,
            last_active INTEGER,
            completed_at INTEGER,
            commit_count INTEGER,
            test_runs INTEGER,
            test_fails INTEGER
        );

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            kind TEXT,
            source TEXT,
            payload TEXT,
            ts INTEGER
        );
    """)
    conn.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            TASK_A["id"],
            TASK_A["repo_root"],
            TASK_A["branch"],
            TASK_A["phase"],
            TASK_A["files"],
            TASK_A["started_at"],
            TASK_A["last_active"],
            TASK_A["completed_at"],
            TASK_A["commit_count"],
            TASK_A["test_runs"],
            TASK_A["test_fails"],
        ),
    )
    conn.executemany(
        "INSERT INTO events (id, kind, source, payload, ts) VALUES (?, ?, ?, ?, ?)",
        [(e["id"], e["kind"], e["source"], json.dumps(e["payload"]), e["ts"]) for e in EVENTS_A],
    )
    conn.commit()
    conn.close()


@pytest.fixture
def fixture_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "sigild" / "data.db"
    _write_fixture_db(db_path)
    return db_path


@pytest.fixture
def store(fixture_db: Path) -> Iterator[SqliteStore]:
    s = SqliteStore(fixture_db)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# T024 — baseline equality with no as_of_ms passed
# ---------------------------------------------------------------------------


class TestStuckBaseline:
    """The stuck family, called the way the serving path calls it."""

    def test_from_data_matches_pre_change_baseline(self) -> None:
        with frozen_clock():
            assert extract_stuck_features_from_data(TASK_A, EVENTS_A) == BASELINE_STUCK_TASK_A

    def test_from_store_matches_pre_change_baseline(self, store: SqliteStore) -> None:
        """The poller/routes entry point: DataStore + task_id, no as_of_ms."""
        with frozen_clock():
            assert extract_stuck_features(store, TASK_A_ID) == BASELINE_STUCK_TASK_A

    def test_missing_task_matches_pre_change_baseline(self, store: SqliteStore) -> None:
        with frozen_clock():
            assert extract_stuck_features(store, "no-such-task") == BASELINE_STUCK_MISSING_TASK

    def test_key_order_matches_pre_change_baseline(self) -> None:
        """Callers build vectors positionally; key order is part of the contract."""
        with frozen_clock():
            result = extract_stuck_features_from_data(TASK_A, EVENTS_A)
        assert list(result) == list(BASELINE_STUCK_TASK_A)


class TestDurationBaseline:
    """The duration family, called the way the serving path calls it."""

    def test_from_data_matches_pre_change_baseline(self) -> None:
        with frozen_clock():
            assert extract_duration_features_from_data(TASK_A, EVENTS_A) == BASELINE_DURATION_TASK_A

    def test_from_store_matches_pre_change_baseline(self, store: SqliteStore) -> None:
        with frozen_clock():
            assert extract_duration_features(store, TASK_A_ID) == BASELINE_DURATION_TASK_A

    def test_missing_task_matches_pre_change_baseline(self, store: SqliteStore) -> None:
        with frozen_clock():
            assert extract_duration_features(store, "no-such-task") == BASELINE_DURATION_MISSING_TASK

    def test_key_order_matches_pre_change_baseline(self) -> None:
        with frozen_clock():
            result = extract_duration_features_from_data(TASK_A, EVENTS_A)
        assert list(result) == list(BASELINE_DURATION_TASK_A)


class TestBufferBaseline:
    """``extract_features_from_buffer`` — the poller's no-active-task path.

    ``poller.py`` reaches this branch whenever ``store.get_active_task()``
    returns None, so it carries as much serving traffic as the task path.
    """

    def test_buffer_matches_pre_change_baseline(self) -> None:
        with frozen_clock():
            assert extract_features_from_buffer(BUFFER_A) == BASELINE_BUFFER_A

    def test_empty_buffer_matches_pre_change_baseline(self) -> None:
        with frozen_clock():
            assert extract_features_from_buffer([]) == BASELINE_BUFFER_EMPTY

    def test_key_order_matches_pre_change_baseline(self) -> None:
        with frozen_clock():
            result = extract_features_from_buffer(BUFFER_A)
        assert list(result) == list(BASELINE_BUFFER_A)

    def test_as_of_zero_returns_empty_vector_rather_than_raising(self) -> None:
        """WP01 moved the no-lookahead filter *ahead* of the empty guard.

        With ``as_of_ms=0`` every event is filtered out, so the function hits
        the same guard an empty buffer hits and returns the documented all-zero
        vector. It must not raise on the ``events[0]`` access. There is no
        pre-WP01 baseline for this call — the parameter did not exist — so the
        assertion is against the documented empty vector, which is itself a
        recorded pre-WP01 observation (BASELINE_BUFFER_EMPTY).
        """
        with frozen_clock():
            assert extract_features_from_buffer(BUFFER_A, as_of_ms=0) == BASELINE_BUFFER_EMPTY


class TestServingOmitsAsOfMs:
    """Omitting ``as_of_ms`` and passing ``None`` must stay indistinguishable."""

    def test_stuck_omitted_equals_explicit_none(self) -> None:
        with frozen_clock():
            assert extract_stuck_features_from_data(TASK_A, EVENTS_A) == extract_stuck_features_from_data(
                TASK_A, EVENTS_A, as_of_ms=None
            )

    def test_duration_omitted_equals_explicit_none(self) -> None:
        with frozen_clock():
            assert extract_duration_features_from_data(TASK_A, EVENTS_A) == extract_duration_features_from_data(
                TASK_A, EVENTS_A, as_of_ms=None
            )

    def test_buffer_omitted_equals_explicit_none(self) -> None:
        with frozen_clock():
            assert extract_features_from_buffer(BUFFER_A) == extract_features_from_buffer(BUFFER_A, as_of_ms=None)


# ---------------------------------------------------------------------------
# T024 — end-to-end through the real prediction routes
# ---------------------------------------------------------------------------


@pytest.fixture
def serving_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A TestClient whose app resolves to the fixture DB.

    ``config.db_path()`` is ``$XDG_DATA_HOME/sigild/data.db``, so redirecting
    XDG_DATA_HOME points the app's real ``SqliteStore`` at our fixture data —
    no store is stubbed. Model weights land in the same clean temp dir, so the
    predictors are untrained, which is what a fresh install serves.
    """
    _write_fixture_db(tmp_path / "sigild" / "data.db")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    from kenaz_ml.app import create_app

    with TestClient(create_app()) as client:
        yield client


class TestPredictionRoutesEndToEnd:
    """HTTP -> route -> extractor -> SqliteStore, with the clock pinned."""

    def test_stuck_route_extracts_baseline_features(
        self, serving_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The vector the route hands the model is the pre-WP01 vector.

        The model itself is untrained, so its output would not vary with the
        input; capturing the argument is what proves the extractor ran and
        produced unchanged values.
        """
        from kenaz_ml.models.stuck import StuckPredictor

        captured: list[dict[str, float]] = []
        original_predict = StuckPredictor.predict

        def spy(self: StuckPredictor, feats: dict[str, float]) -> dict[str, Any]:
            captured.append(feats)
            return original_predict(self, feats)

        monkeypatch.setattr(StuckPredictor, "predict", spy)

        with frozen_clock():
            resp = serving_client.post("/predict/stuck", json={"task_id": TASK_A_ID})

        assert resp.status_code == 200
        assert captured, "route did not reach the stuck predictor"
        assert captured[-1] == BASELINE_STUCK_TASK_A

    def test_stuck_route_response_shape_unchanged(self, serving_client: TestClient) -> None:
        """`sigilctl` reads this body; its shape must not move (CLAUDE.md)."""
        with frozen_clock():
            resp = serving_client.post("/predict/stuck", json={"task_id": TASK_A_ID})

        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {"probability", "confidence"}
        assert body["confidence"] in ("weak", "moderate", "strong")
        # Untrained model -> documented fallback, unchanged by WP01.
        assert body == {"probability": 0.5, "confidence": "weak"}

    def test_duration_route_extracts_baseline_features(
        self, serving_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kenaz_ml.models.duration import DurationEstimator

        captured: list[dict[str, float]] = []
        original_predict = DurationEstimator.predict

        def spy(self: DurationEstimator, feats: dict[str, float]) -> dict[str, Any]:
            captured.append(feats)
            return original_predict(self, feats)

        monkeypatch.setattr(DurationEstimator, "predict", spy)

        with frozen_clock():
            resp = serving_client.post("/predict/duration", json={"task_id": TASK_A_ID})

        assert resp.status_code == 200
        assert captured, "route did not reach the duration estimator"
        assert captured[-1] == BASELINE_DURATION_TASK_A

    def test_duration_route_response_shape_unchanged(self, serving_client: TestClient) -> None:
        with frozen_clock():
            resp = serving_client.post("/predict/duration", json={"task_id": TASK_A_ID})

        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {"estimated_minutes", "confidence_interval"}
        assert body == {"estimated_minutes": 60.0, "confidence_interval": [30.0, 90.0]}

    def test_as_of_ms_does_not_appear_in_the_http_surface(self, serving_client: TestClient) -> None:
        """``as_of_ms`` is internal and must never reach the public schema.

        Checked against the app's live OpenAPI document rather than a single
        route, so a leak anywhere — path, query, body, or response — fails
        here. ``docs/openapi.json`` is generated from this same document
        (``scripts/gen_openapi.py``), which is what keeps the checked-in spec
        honest.
        """
        resp = serving_client.get("/openapi.json")
        assert resp.status_code == 200
        assert "as_of_ms" not in json.dumps(resp.json())

    def test_prediction_request_models_expose_no_reference_time(self) -> None:
        """The request bodies `sigilctl` sends are exactly what they were."""
        from kenaz_ml.routes import DurationRequest, DurationResponse, StuckRequest, StuckResponse

        assert set(StuckRequest.model_fields) == {"task_id", "features"}
        assert set(StuckResponse.model_fields) == {"probability", "confidence"}
        assert set(DurationRequest.model_fields) == {"task_id", "features"}
        assert set(DurationResponse.model_fields) == {"estimated_minutes", "confidence_interval"}

    @pytest.mark.parametrize("path", ["/predict/stuck", "/predict/duration"])
    def test_unknown_as_of_ms_body_field_cannot_steer_the_reference_time(
        self, serving_client: TestClient, path: str
    ) -> None:
        """A client that sends ``as_of_ms`` anyway must be ignored, not obeyed.

        The models do not declare the field, so pydantic drops it and the
        response is identical to the request without it. If this ever starts
        differing, the parameter has become reachable from the network.
        """
        with frozen_clock():
            plain = serving_client.post(path, json={"task_id": TASK_A_ID})
            with_extra = serving_client.post(path, json={"task_id": TASK_A_ID, "as_of_ms": 1})

        assert plain.status_code == 200
        assert with_extra.status_code == 200
        assert with_extra.json() == plain.json()


# ---------------------------------------------------------------------------
# T024 — signature guards
# ---------------------------------------------------------------------------

# Every public extractor that carries a reference time. The serving path calls
# all of these with the parameter omitted.
POINT_IN_TIME_EXTRACTORS = [
    extract_stuck_features,
    extract_duration_features,
    extract_features_from_buffer,
    extract_stuck_features_from_data,
    extract_duration_features_from_data,
]

# Extractors that describe an event or a pre-windowed set rather than an
# instant. They have no reference time and must not grow one silently.
TIME_REFERENCE_FREE_EXTRACTORS = [
    extract_activity_features,
    extract_workflow_features,
]


@pytest.mark.parametrize("fn", POINT_IN_TIME_EXTRACTORS, ids=lambda f: f.__name__)
def test_as_of_ms_is_optional_and_keyword_only(fn: Any) -> None:
    """Guards the whole serving path in one assertion.

    Making ``as_of_ms`` required, or positional, would break every serving call
    site at once — silently in the positional case, since a store or an event
    list would be bound to it.
    """
    param = inspect.signature(fn).parameters["as_of_ms"]
    assert param.default is None, f"{fn.__name__}: as_of_ms must default to None (current wall clock)"
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, f"{fn.__name__}: as_of_ms must be keyword-only"


@pytest.mark.parametrize("fn", POINT_IN_TIME_EXTRACTORS, ids=lambda f: f.__name__)
def test_extractors_are_callable_without_as_of_ms(fn: Any) -> None:
    """A binding-level check: the serving call shape must stay valid."""
    sig = inspect.signature(fn)
    required = [
        name
        for name, p in sig.parameters.items()
        if p.default is inspect.Parameter.empty and p.kind is not inspect.Parameter.VAR_KEYWORD
    ]
    assert "as_of_ms" not in required


@pytest.mark.parametrize("fn", TIME_REFERENCE_FREE_EXTRACTORS, ids=lambda f: f.__name__)
def test_time_reference_free_extractors_have_no_as_of_ms(fn: Any) -> None:
    assert "as_of_ms" not in inspect.signature(fn).parameters
