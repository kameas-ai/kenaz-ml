"""Local serving proves it computes live rather than reading the store (D-003).

The headline test in this module is
:meth:`TestLiveComputeBeatsAStaleStore.test_seeded_wrong_value_never_reaches_a_prediction`.
It seeds the online store with a deliberately wrong vector for the active task,
generates live activity that computes to something else, and asserts the wrong
value never reaches the model. If that passes *with the store seeded wrong*,
live compute genuinely won — no amount of reading the code can give the same
assurance, because the conventional Feast pattern (read the online store at
serving time) is exactly the shape a reviewer's eye is trained to accept.

The rest of the module holds the surrounding guarantees in place: the push
happens (FR-014), a failing push cannot fail a prediction (FR-017), the vector
is byte-identical to the extractor's own output including key order (C-005), an
empty store on first run serves fine, and the serving path stays inside
NFR-002's latency budget — measured here, not merely cited.
"""

from __future__ import annotations

import ast
import logging
import statistics
import time
from pathlib import Path
from typing import Any

import pytest
from feast import FeatureStore

from kenaz_ml.config import ServingMode
from kenaz_ml.feature_store import config as fsc
from kenaz_ml.feature_store import resolve
from kenaz_ml.feature_store.definitions import (
    REGISTERED_FEATURE_NAMES,
    duration_features,
    stuck_features,
    task,
)
from kenaz_ml.features import (
    extract_duration_features,
    extract_features_from_buffer,
    extract_stuck_features,
)

RESOLVE_LOGGER = "kenaz_ml.feature_store.resolve"

#: The value seeded into the online store. Nothing the extractors can compute
#: from the fixtures below comes anywhere near it, so its appearance in a
#: prediction is unambiguous evidence that the store was read.
WRONG = 999999.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeDataStore:
    """The slice of ``DataStore`` the serving path touches, with live events."""

    def __init__(self, *, task_id: str = "task-live", events: list[dict] | None = None) -> None:
        self.task_id = task_id
        now_ms = int(time.time() * 1000)
        self.started_at = now_ms - 600_000
        self._events = events if events is not None else _live_events(now_ms)
        self.predictions: list[tuple[str, dict]] = []

    def get_active_task(self) -> str | None:
        return self.task_id

    def get_task_by_id(self, task_id: str) -> dict[str, Any] | None:
        if task_id != self.task_id:
            return None
        return {
            "id": task_id,
            "started_at": self.started_at,
            "last_active": self.started_at + 600_000,
            "completed_at": None,
            "test_fails": 3,
            "phase": "implementing",
            "branch": "feat/live-serving",
            "files": '{"alpha.py": 4, "beta.py": 2, "gamma.py": 1}',
        }

    def get_events_for_task(self, task_id: str, since: int | None = None) -> list[dict]:
        return list(self._events)

    def get_session_info(self, task_id: str) -> dict[str, Any] | None:
        return {"started_at": self.started_at, "phase": "implementing", "test_fails": 3}

    def get_quality_task_stats(self) -> dict[str, Any] | None:
        return {"test_runs": 10, "test_fails": 2, "commit_count": 4}

    def insert_prediction(self, model: str, result: dict, confidence: float, ttl_sec: int | None) -> None:
        self.predictions.append((model, result))

    def insert_ml_event(self, kind: str, endpoint: str, routing: str, latency_ms: int) -> None:
        return None

    def commit(self) -> None:
        return None


def _live_events(now_ms: int, count: int = 60) -> list[dict]:
    """Events across the task window, of the kinds the extractors actually read."""
    events: list[dict] = []
    for index in range(count):
        events.append(
            {
                "id": index,
                "ts": now_ms - (count - index) * 5_000,
                "kind": "edit" if index % 2 else "terminal",
                "payload": {"file": f"alpha{index % 3}.py", "exit_code": index % 5},
            }
        )
    return events


class FrozenClock:
    """``time`` with a ``time()`` that stops on first use.

    Three stuck features are elapsed durations measured against the current wall
    clock, which is correct for an active task (C-006) and means two consecutive
    extractions of the same task differ by however long the first one took. That
    is real behaviour, not a defect, but it makes "identical to the extractor's
    own output" unassertable to the millisecond. Freezing the clock at its first
    reading removes the drift without changing which reference time is used —
    ``as_of_ms`` is still ``None``, still meaning now.
    """

    def __init__(self, real: Any) -> None:
        self._real = real
        self._at: float | None = None

    def time(self) -> float:
        if self._at is None:
            self._at = self._real.time()
        return self._at

    def localtime(self, secs: float | None = None) -> Any:
        return self._real.localtime(self.time() if secs is None else secs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


@pytest.fixture
def freeze_clock() -> bool:
    """Whether to freeze the extractors' clock. Overridden by the latency test."""
    return True


@pytest.fixture(autouse=True)
def _frozen_clock(freeze_clock: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze the extractors' notion of now. Only ``kenaz_ml.features`` is patched.

    The resolver's own ``time.time`` is left alone, so the timestamp stamped on
    a pushed row is still the real one.
    """
    if not freeze_clock:
        return
    import kenaz_ml.features as features_module

    monkeypatch.setattr(features_module, "time", FrozenClock(time))


@pytest.fixture(autouse=True)
def _clean_resolver() -> Any:
    """Leave no worker state, cached store or counter behind between tests."""
    resolve.use_feature_store(None)
    resolve.reset_push_state()
    yield
    resolve.flush_pushes()
    resolve.use_feature_store(None)
    resolve.reset_push_state()


@pytest.fixture
def local_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A read-only-in-spirit bundle directory and a writable data directory."""
    monkeypatch.delenv("KENAZ_MODE", raising=False)
    bundle = tmp_path / "bundle"
    user_data = tmp_path / "data"
    bundle.mkdir()
    user_data.mkdir()
    return bundle, user_data


@pytest.fixture
def online_store(local_dirs: tuple[Path, Path]) -> FeatureStore:
    """An applied local registry with an empty SQLite online store attached.

    Applied through :func:`kenaz_ml.feature_store.resolve.local_feature_views`,
    the same binding the build-time apply uses, so the registry these tests push
    into is the registry the local deployment ships.
    """
    bundle, user_data = local_dirs
    FeatureStore(config=fsc.load_repo_config(bundle=bundle, user_data=user_data)).apply(
        [task, *resolve.local_feature_views()]
    )
    # Reopened from a second read of the configuration, so the registry is read
    # back from disk rather than served by the object that wrote it.
    store = FeatureStore(config=fsc.load_repo_config(bundle=bundle, user_data=user_data))
    resolve.use_feature_store(store)
    return store


def _seed(store: FeatureStore, family: resolve.FeatureFamily, entity_id: str, value: float) -> None:
    """Write ``value`` into every feature of ``family`` for ``entity_id``."""
    import pandas as pd
    from feast.data_source import PushMode

    names = REGISTERED_FEATURE_NAMES[family.name]
    frame = pd.DataFrame(
        {
            family.join_key: [entity_id],
            "event_timestamp": [pd.Timestamp.now(tz="UTC")],
            **{name: [value] for name in names},
        }
    )
    store.push(family.push_source_name, frame, to=PushMode.ONLINE)


class RecordingModel:
    """A trained-looking model that records the vector it was asked to predict on."""

    def __init__(self, result: dict) -> None:
        self.is_trained = True
        self.result = result
        self.seen: list[dict[str, float]] = []

    def predict(self, features: dict[str, float]) -> dict:
        self.seen.append(dict(features))
        return dict(self.result)


class PassthroughActivity:
    is_trained = True

    def classify(self, event: dict) -> dict:
        return {"category": "editing", "confidence": 0.9}


class StaticWorkflow:
    is_trained = True

    def predict(self, events: list[dict], session_info: dict) -> dict:
        return {"confidence": 0.5}


class StaticQuality:
    def predict(self, features: dict) -> dict:
        return {"score": 70}


def _poller(data_store: FakeDataStore) -> tuple[Any, RecordingModel, RecordingModel]:
    """Build an ``EventPoller`` whose stuck and duration models record their input."""
    from kenaz_ml.poller import EventPoller

    stuck = RecordingModel({"probability": 0.6, "confidence": "moderate"})
    duration = RecordingModel({"estimated_minutes": 42.0, "confidence_interval": [30.0, 60.0]})
    poller = EventPoller(
        data_store,  # type: ignore[arg-type]
        {
            "stuck": stuck,
            "activity": PassthroughActivity(),
            "workflow": StaticWorkflow(),
            "duration": duration,
            "quality": StaticQuality(),
        },
    )
    return poller, stuck, duration


def _routes_state(data_store: FakeDataStore) -> tuple[Any, RecordingModel, RecordingModel]:
    """Build an ``AppState`` wired into a real FastAPI app with the real routes."""
    from fastapi import FastAPI

    from kenaz_ml.app import AppState
    from kenaz_ml.routes import register_routes

    state = AppState(mode=ServingMode.LOCAL)
    state.store = data_store  # type: ignore[assignment]
    stuck = RecordingModel({"probability": 0.61, "confidence": "moderate"})
    duration = RecordingModel({"estimated_minutes": 43.0, "confidence_interval": [30.0, 60.0]})
    state.stuck = stuck  # type: ignore[assignment]
    state.duration = duration  # type: ignore[assignment]
    app = FastAPI()
    register_routes(app, state)
    return app, stuck, duration


# ---------------------------------------------------------------------------
# The headline: live compute beats a stale store
# ---------------------------------------------------------------------------


class TestLiveComputeBeatsAStaleStore:
    """US5 scenario 3, FR-014. The test this whole work package exists to pass."""

    def test_seeded_wrong_value_never_reaches_a_prediction(self, online_store: FeatureStore) -> None:
        data_store = FakeDataStore()
        _seed(online_store, resolve.STUCK, data_store.task_id, WRONG)
        _seed(online_store, resolve.DURATION, data_store.task_id, WRONG)

        # Vacuity guard: the seed really is in the store, so a serving path that
        # read the store would in fact find it. Without this the assertions
        # below would pass against an empty store for the wrong reason.
        stored = resolve.read_online_features(resolve.STUCK, data_store.task_id)
        assert stored is not None
        assert set(stored.values()) == {WRONG}

        poller, stuck, duration = _poller(data_store)
        poller._predict_and_write()

        assert stuck.seen, "the stuck model was never asked to predict"
        assert duration.seen, "the duration model was never asked to predict"
        for vector in stuck.seen + duration.seen:
            assert WRONG not in vector.values(), (
                f"a stored value reached the model: {vector}. Active-task predictions must "
                "compute live and never read the online store (D-003, FR-014)."
            )

        assert stuck.seen[0] == extract_stuck_features(data_store, data_store.task_id)
        assert duration.seen[0] == extract_duration_features(data_store, data_store.task_id)

    def test_seeded_wrong_value_never_reaches_the_http_endpoints(self, online_store: FeatureStore) -> None:
        from fastapi.testclient import TestClient

        data_store = FakeDataStore()
        _seed(online_store, resolve.STUCK, data_store.task_id, WRONG)
        _seed(online_store, resolve.DURATION, data_store.task_id, WRONG)

        app, stuck, duration = _routes_state(data_store)
        with TestClient(app) as client:
            assert client.post("/predict/stuck", json={"task_id": data_store.task_id}).status_code == 200
            assert client.post("/predict/duration", json={"task_id": data_store.task_id}).status_code == 200

        assert stuck.seen and duration.seen
        for vector in stuck.seen + duration.seen:
            assert WRONG not in vector.values(), f"a stored value reached an endpoint: {vector}"

    def test_a_second_prediction_still_computes_live(self, online_store: FeatureStore) -> None:
        """The push must not turn into a cache on the next cycle.

        The first resolution populates the store; if anything read it afterwards
        the second resolution would return the *first* vector rather than a
        recomputed one. Advancing the task's events between the two makes the
        difference observable.
        """
        data_store = FakeDataStore()
        first = resolve.resolve_stuck_features(data_store, data_store.task_id)
        assert resolve.flush_pushes(timeout=30.0)

        data_store._events = data_store._events + [
            {"id": 900, "ts": data_store.started_at + 1_000, "kind": "edit", "payload": {"file": "delta.py"}}
        ]
        second = resolve.resolve_stuck_features(data_store, data_store.task_id)

        assert second == extract_stuck_features(data_store, data_store.task_id)
        assert second["edit_velocity"] != first["edit_velocity"]


# ---------------------------------------------------------------------------
# The structural guarantee
# ---------------------------------------------------------------------------


#: Every API by which a stored vector could enter the serving path.
FORBIDDEN_READ_CALLS = frozenset(
    {
        "get_online_features",
        "get_historical_features",
        "retrieve_online_documents",
        "read_online_features",
    }
)

ACTIVE_ENTRY_POINTS = (
    "resolve_stuck_features",
    "resolve_duration_features",
    "resolve_stuck_features_from_buffer",
)


def _module_ast() -> ast.Module:
    return ast.parse(Path(resolve.__file__).read_text(encoding="utf-8"))


def _referenced_names(node: ast.AST) -> set[str]:
    """Every name a function calls, or merely mentions.

    Bare mentions count, not only calls: ``threading.Thread(target=_push_worker)``
    reaches ``_push_worker`` without ever writing ``_push_worker(...)``, and a
    reachability argument that missed that would be worth nothing.
    """
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
    return names


class TestNoStoreReadOnTheActivePath:
    """T009's central claim, checked three independent ways."""

    def test_no_reachable_call_from_an_active_resolver_reads_the_store(self) -> None:
        module = _module_ast()
        functions = {n.name: n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)}

        for entry in ACTIVE_ENTRY_POINTS:
            assert entry in functions, f"{entry} is not defined in resolve.py"
            seen: set[str] = set()
            frontier = [entry]
            while frontier:
                name = frontier.pop()
                if name in seen:
                    continue
                seen.add(name)
                referenced = _referenced_names(functions[name])
                offending = referenced & FORBIDDEN_READ_CALLS
                assert not offending, (
                    f"{entry} reaches {sorted(offending)} via {name}. No code path may return a "
                    "stored value for an active task (D-003, FR-014)."
                )
                frontier.extend(sorted(referenced & set(functions) - seen))

    def test_active_resolvers_take_no_parameter_that_could_select_a_stored_value(self) -> None:
        """No ``as_of_ms``, no ``allow_cached``: one behaviour, no switch."""
        module = _module_ast()
        functions = {n.name: n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)}
        for entry in ACTIVE_ENTRY_POINTS:
            args = functions[entry].args
            names = [a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]]
            assert "as_of_ms" not in names, (
                f"{entry} accepts as_of_ms. Serving passes nothing, meaning now (C-006), and the "
                "absence of the parameter is what stops this resolver being reused for a replay."
            )
            assert not any("cache" in n or "stored" in n or "online" in n for n in names), names

    def test_reading_the_store_inside_an_active_resolution_raises(self, online_store: FeatureStore) -> None:
        """The runtime half: a future edit that adds a read fails loudly.

        Simulated by making the extractor itself attempt the read, which is the
        shape the mistake would actually take — a helper several frames down
        reaching for the store, not a literal call in the resolver body.
        """
        data_store = FakeDataStore()
        _seed(online_store, resolve.STUCK, data_store.task_id, WRONG)

        def peeking_extractor(store: Any, task_id: str, **kwargs: Any) -> dict[str, float]:
            stored = resolve.read_online_features(resolve.STUCK, task_id)
            return stored or {}

        original = resolve.extract_stuck_features
        resolve.extract_stuck_features = peeking_extractor  # type: ignore[assignment]
        try:
            with pytest.raises(resolve.ActiveEntityStoreReadError) as excinfo:
                resolve.resolve_stuck_features(data_store, data_store.task_id)
        finally:
            resolve.extract_stuck_features = original  # type: ignore[assignment]

        assert "compute live" in str(excinfo.value)

    def test_the_guard_is_released_afterwards(self, online_store: FeatureStore) -> None:
        """Otherwise the non-active read path would be dead for the process."""
        data_store = FakeDataStore()
        resolve.resolve_stuck_features(data_store, data_store.task_id)
        assert resolve.read_online_features(resolve.STUCK, "somebody-else") is not None

    def test_the_non_active_path_may_read_the_store(self, online_store: FeatureStore) -> None:
        _seed(online_store, resolve.DURATION, "finished-task", 7.0)
        stored = resolve.read_online_features(resolve.DURATION, "finished-task")
        assert stored == dict.fromkeys(REGISTERED_FEATURE_NAMES["duration"], 7.0)

    def test_an_unavailable_store_fails_the_read_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FR-017: a read never degrades to a sentinel a caller could feed a model.

        The opposite of the push path, on purpose — see
        :func:`kenaz_ml.feature_store.resolve.read_online_features`.
        """

        def unavailable() -> Any:
            raise RuntimeError("online store is unavailable")

        monkeypatch.setattr(resolve, "_feature_store", unavailable)
        with pytest.raises(RuntimeError, match="unavailable"):
            resolve.read_online_features(resolve.STUCK, "any-task")


# ---------------------------------------------------------------------------
# The push happened
# ---------------------------------------------------------------------------


class TestPushHappens:
    def test_the_store_holds_the_freshly_computed_vector(self, online_store: FeatureStore) -> None:
        data_store = FakeDataStore()
        computed = resolve.resolve_stuck_features(data_store, data_store.task_id)
        assert resolve.flush_pushes(timeout=30.0), "the push queue did not drain"

        stored = resolve.read_online_features(resolve.STUCK, data_store.task_id)
        assert stored == pytest.approx(computed)
        assert resolve.push_stats()["succeeded"] == 1
        assert resolve.push_stats()["failed"] == 0

    def test_duration_pushes_under_its_own_view(self, online_store: FeatureStore) -> None:
        data_store = FakeDataStore()
        computed = resolve.resolve_duration_features(data_store, data_store.task_id)
        assert resolve.flush_pushes(timeout=30.0)
        assert resolve.read_online_features(resolve.DURATION, data_store.task_id) == pytest.approx(computed)

    def test_the_pushed_row_is_stamped_with_the_moment_it_describes(self, online_store: FeatureStore) -> None:
        """FR-009's local counterpart: the event time is when the vector was computed."""
        pushes: list[Any] = []
        real_push = online_store.push

        def recording_push(name: str, frame: Any, **kwargs: Any) -> Any:
            pushes.append(frame)
            return real_push(name, frame, **kwargs)

        online_store.push = recording_push  # type: ignore[assignment]
        before = time.time()
        resolve.resolve_stuck_features(FakeDataStore(), "task-live")
        assert resolve.flush_pushes(timeout=30.0)
        after = time.time()

        assert len(pushes) == 1
        stamped = pushes[0]["event_timestamp"].iloc[0].timestamp()
        assert before - 1.0 <= stamped <= after + 1.0

    def test_the_buffer_path_computes_live_and_pushes_nothing(self) -> None:
        """No active task means no entity key, so there is nothing to key a row by."""
        recording = BrokenFeatureStore()
        resolve.use_feature_store(recording)
        events = _live_events(int(time.time() * 1000))

        resolved = resolve.resolve_stuck_features_from_buffer(events)
        assert resolve.flush_pushes(timeout=30.0)

        assert resolved == extract_features_from_buffer(events)
        assert recording.calls == []
        assert resolve.push_stats() == {
            "dropped": 0,
            "succeeded": 0,
            "failed": 0,
            "suppressed_warnings": 0,
        }


# ---------------------------------------------------------------------------
# Best effort — FR-017
# ---------------------------------------------------------------------------


class BrokenFeatureStore:
    """A store whose push always fails, counting how often it was asked."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def push(self, push_source_name: str, frame: Any, **kwargs: Any) -> Any:
        self.calls.append(push_source_name)
        raise RuntimeError("online store is unavailable")


class TestPushIsBestEffort:
    def test_a_failing_push_leaves_the_prediction_intact(self, caplog: pytest.LogCaptureFixture) -> None:
        resolve.use_feature_store(BrokenFeatureStore())
        data_store = FakeDataStore()

        with caplog.at_level(logging.WARNING, logger=RESOLVE_LOGGER):
            resolved = resolve.resolve_stuck_features(data_store, data_store.task_id)
            assert resolve.flush_pushes(timeout=30.0)

        assert resolved == extract_stuck_features(data_store, data_store.task_id)
        assert resolve.push_stats()["failed"] == 1
        assert resolve.push_stats()["succeeded"] == 0
        assert any("push failed" in r.message for r in caplog.records)

    def test_a_failing_push_leaves_the_whole_prediction_cycle_intact(self) -> None:
        resolve.use_feature_store(BrokenFeatureStore())
        data_store = FakeDataStore()
        poller, stuck, duration = _poller(data_store)

        poller._predict_and_write()
        assert resolve.flush_pushes(timeout=30.0)

        assert stuck.seen[0] == extract_stuck_features(data_store, data_store.task_id)
        assert duration.seen[0] == extract_duration_features(data_store, data_store.task_id)
        assert [model for model, _ in data_store.predictions] == [
            "stuck",
            "activity",
            "suggest",
            "duration",
            "quality",
        ]

    def test_the_push_is_not_retried_inline(self) -> None:
        broken = BrokenFeatureStore()
        resolve.use_feature_store(broken)
        data_store = FakeDataStore()

        for _ in range(4):
            resolve.resolve_stuck_features(data_store, data_store.task_id)
        assert resolve.flush_pushes(timeout=30.0)

        assert len(broken.calls) == 4, (
            f"{len(broken.calls)} pushes for 4 resolutions — a failed push must not be retried "
            "inline; the next cycle pushes a fresher vector instead."
        )

    def test_failure_logging_is_rate_limited(self, caplog: pytest.LogCaptureFixture) -> None:
        resolve.use_feature_store(BrokenFeatureStore())
        data_store = FakeDataStore()

        with caplog.at_level(logging.WARNING, logger=RESOLVE_LOGGER):
            for _ in range(25):
                resolve.resolve_stuck_features(data_store, data_store.task_id)
            assert resolve.flush_pushes(timeout=60.0)

        warnings = [r for r in caplog.records if r.name == RESOLVE_LOGGER and r.levelno == logging.WARNING]
        assert len(warnings) == 1, f"{len(warnings)} warnings for 25 failures — the log is not rate-limited"
        stats = resolve.push_stats()
        assert stats["failed"] == 25
        assert stats["suppressed_warnings"] == 24

    def test_the_suppression_count_is_reported_when_the_window_reopens(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A store failing forever must keep saying so, not go quiet after the first."""
        monkeypatch.setattr(resolve, "PUSH_LOG_INTERVAL_SEC", 0.05)
        resolve.use_feature_store(BrokenFeatureStore())
        data_store = FakeDataStore()

        with caplog.at_level(logging.WARNING, logger=RESOLVE_LOGGER):
            for _ in range(3):
                resolve.resolve_stuck_features(data_store, data_store.task_id)
            assert resolve.flush_pushes(timeout=30.0)
            time.sleep(0.1)
            resolve.resolve_stuck_features(data_store, data_store.task_id)
            assert resolve.flush_pushes(timeout=30.0)

        warnings = [r for r in caplog.records if r.name == RESOLVE_LOGGER and r.levelno == logging.WARNING]
        assert len(warnings) == 2
        assert "suppressed" in warnings[1].message

    def test_an_unopenable_store_does_not_fail_the_prediction(
        self, local_dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No registry has been applied — the shape of a broken or partial install."""
        bundle, user_data = local_dirs
        monkeypatch.setattr(
            resolve,
            "_feature_store",
            lambda: FeatureStore(config=fsc.load_repo_config(bundle=bundle, user_data=user_data)),
        )
        data_store = FakeDataStore()

        with caplog.at_level(logging.WARNING, logger=RESOLVE_LOGGER):
            resolved = resolve.resolve_stuck_features(data_store, data_store.task_id)
            assert resolve.flush_pushes(timeout=30.0)

        assert resolved == extract_stuck_features(data_store, data_store.task_id)
        assert resolve.push_stats()["failed"] == 1

    def test_push_outcome_never_alters_the_returned_vector(self, online_store: FeatureStore) -> None:
        """The same input resolves to the same vector whether the push works or not."""
        data_store = FakeDataStore()
        with_working_store = resolve.resolve_stuck_features(data_store, data_store.task_id)
        assert resolve.flush_pushes(timeout=30.0)

        resolve.use_feature_store(BrokenFeatureStore())
        with_broken_store = resolve.resolve_stuck_features(data_store, data_store.task_id)
        assert resolve.flush_pushes(timeout=30.0)

        assert with_working_store == with_broken_store


# ---------------------------------------------------------------------------
# Values unchanged — C-005
# ---------------------------------------------------------------------------


class TestVectorIdentity:
    def test_stuck_vector_is_identical_to_the_extractor_including_order(self, online_store: FeatureStore) -> None:
        data_store = FakeDataStore()
        direct = extract_stuck_features(data_store, data_store.task_id)
        resolved = resolve.resolve_stuck_features(data_store, data_store.task_id)

        assert resolved == direct
        assert list(resolved) == list(direct), "key order changed; the trainers index positionally"

    def test_duration_vector_is_identical_to_the_extractor_including_order(self, online_store: FeatureStore) -> None:
        data_store = FakeDataStore()
        direct = extract_duration_features(data_store, data_store.task_id)
        resolved = resolve.resolve_duration_features(data_store, data_store.task_id)

        assert resolved == direct
        assert list(resolved) == list(direct)

    def test_resolved_keys_match_the_registered_feature_names_in_order(self, online_store: FeatureStore) -> None:
        data_store = FakeDataStore()
        assert (
            list(resolve.resolve_stuck_features(data_store, data_store.task_id)) == (REGISTERED_FEATURE_NAMES["stuck"])
        )
        assert (
            list(resolve.resolve_duration_features(data_store, data_store.task_id))
            == (REGISTERED_FEATURE_NAMES["duration"])
        )

    def test_an_unknown_task_still_yields_the_extractor_s_documented_empty_vector(
        self, online_store: FeatureStore
    ) -> None:
        data_store = FakeDataStore()
        assert resolve.resolve_stuck_features(data_store, "no-such-task") == extract_stuck_features(
            data_store, "no-such-task"
        )


# ---------------------------------------------------------------------------
# First run
# ---------------------------------------------------------------------------


class TestEmptyStoreOnFirstRun:
    def test_serving_works_before_anything_has_been_pushed(self, online_store: FeatureStore) -> None:
        """Spec edge case: an empty online store must serve without error."""
        data_store = FakeDataStore()
        assert resolve.read_online_features(resolve.STUCK, data_store.task_id) == dict.fromkeys(
            REGISTERED_FEATURE_NAMES["stuck"], None
        )

        poller, stuck, _duration = _poller(data_store)
        poller._predict_and_write()
        assert stuck.seen[0] == extract_stuck_features(data_store, data_store.task_id)

    def test_the_first_push_populates_an_empty_store(self, online_store: FeatureStore) -> None:
        data_store = FakeDataStore()
        computed = resolve.resolve_stuck_features(data_store, data_store.task_id)
        assert resolve.flush_pushes(timeout=30.0)
        assert resolve.read_online_features(resolve.STUCK, data_store.task_id) == pytest.approx(computed)


# ---------------------------------------------------------------------------
# The local source binding
# ---------------------------------------------------------------------------


class TestLocalSourceBinding:
    def test_family_names_agree_with_the_shipped_definitions(self) -> None:
        """The constants in resolve.py are spelled out to keep Feast off the import path."""
        assert resolve.STUCK.view_name == stuck_features.name
        assert resolve.DURATION.view_name == duration_features.name
        assert resolve.STUCK.join_key == task.join_key
        assert resolve.DURATION.join_key == task.join_key
        assert set(resolve.FAMILIES) == set(REGISTERED_FEATURE_NAMES)

    def test_local_views_carry_the_push_sources_the_resolver_pushes_into(self) -> None:
        views = {view.name: view for view in resolve.local_feature_views()}
        assert views[resolve.STUCK.view_name].stream_source.name == resolve.STUCK.push_source_name
        assert views[resolve.DURATION.view_name].stream_source.name == resolve.DURATION.push_source_name

    def test_the_batch_source_is_named_and_pathed_as_unused(self) -> None:
        """Feast requires a batch source for a push source; this one is never read.

        Asserted rather than merely commented so that a later edit pointing it at
        a plausible parquet path — which is precisely the April 2026 mistake —
        fails here.
        """
        for view in resolve.local_feature_views():
            batch = view.stream_source.batch_source
            assert "unused" in batch.name
            assert batch.path == resolve.UNUSED_BATCH_SOURCE_PATH
            assert not Path(batch.path).exists()

    def test_local_views_preserve_declaration_order(self) -> None:
        """``FeatureView.schema`` is ``list(set(...))``; only ``.features`` is ordered."""
        views = {view.name: view for view in resolve.local_feature_views()}
        for family_name, names in REGISTERED_FEATURE_NAMES.items():
            view = views[resolve.FAMILIES[family_name].view_name]
            assert [f.name for f in view.features] == names


# ---------------------------------------------------------------------------
# NFR-002 — measured, not cited
# ---------------------------------------------------------------------------

#: Measured on the implementing machine (Apple Silicon, CPython 3.12, feast
#: 0.65.0, SQLite online store), three rounds of 1000 resolutions per task size,
#: real clock, push worker running and the store genuinely being written. The
#: recorded figures are printed by the test below on every run; see the final
#: report for the numbers from the implementing run.
#:
#: The resolver's overhead is a *constant* ~1 µs per resolution — a context
#: variable set/reset, a six-key dict copy and a queue enqueue — so the ratio is
#: worst on the smallest task, which is why the small size is measured too
#: rather than only the realistic one.
#:
#: What is emphatically *not* in that ~1 µs is the push itself: a synchronous
#: ``FeatureStore.push`` measures a median of 1.05 ms against a 5-12 µs
#: extraction, an 85x-200x serving regression. That measurement is why the push
#: runs on a background worker rather than in the caller.
NFR_002_BUDGET = 1.20
NFR_002_ROUNDS = 3
NFR_002_RESOLUTIONS = 1000

#: 60 events is a task a few minutes old; 500 is a long-running one. A real
#: sigild database carries ~3.5k events across all tasks (CLAUDE.md).
NFR_002_TASK_SIZES = (60, 500)


class TestNfr002ServingLatency:
    @pytest.fixture
    def freeze_clock(self) -> bool:
        """Measure against the real clock.

        The frozen clock the other tests use removes a ``time.time()`` call from
        the extractor while the resolver still makes one, which would deflate the
        baseline and inflate the ratio. A latency budget measured against a
        doctored baseline is worth nothing.
        """
        return False

    @pytest.mark.parametrize("event_count", NFR_002_TASK_SIZES)
    def test_resolution_stays_within_twenty_percent_of_the_direct_extractor(
        self, online_store: FeatureStore, capsys: pytest.CaptureFixture[str], event_count: int
    ) -> None:
        """NFR-002. The pre-migration serving path is the direct extractor call.

        Measured with the push worker genuinely running and the online store
        genuinely being written, so the number includes whatever the background
        push costs the serving thread in contention — the pessimistic reading,
        not a measurement of the resolver with its consequences removed.
        """
        now_ms = int(time.time() * 1000)
        data_store = FakeDataStore(events=_live_events(now_ms, count=event_count))

        # Warm up: the worker's first push imports pandas and Feast's SQLite
        # store, which would otherwise land inside the measured window.
        resolve.resolve_stuck_features(data_store, data_store.task_id)
        assert resolve.flush_pushes(timeout=60.0)

        ratios: list[float] = []
        report: list[str] = []
        for round_index in range(NFR_002_ROUNDS):
            baseline = _median_seconds(lambda: extract_stuck_features(data_store, data_store.task_id))
            resolved = _median_seconds(lambda: resolve.resolve_stuck_features(data_store, data_store.task_id))
            resolve.flush_pushes(timeout=60.0)
            ratios.append(resolved / baseline)
            report.append(
                f"  {event_count:>4} events, round {round_index + 1}: "
                f"baseline {baseline * 1e6:7.2f} us  resolver {resolved * 1e6:7.2f} us  "
                f"ratio {resolved / baseline:.3f}"
            )

        measured = statistics.median(ratios)
        with capsys.disabled():
            print(
                f"\nNFR-002 over {NFR_002_ROUNDS}x{NFR_002_RESOLUTIONS} resolutions of a "
                f"{event_count}-event task (median ratio {measured:.3f}, budget "
                f"{NFR_002_BUDGET:.2f}):\n" + "\n".join(report)
            )

        assert measured <= NFR_002_BUDGET, (
            f"serving-path resolution on a {event_count}-event task is {measured:.3f}x the direct "
            f"extractor, over NFR-002's {NFR_002_BUDGET:.2f}x budget. "
            f"Rounds: {[round(r, 3) for r in ratios]}"
        )


def _median_seconds(call: Any) -> float:
    samples = []
    for _ in range(NFR_002_RESOLUTIONS):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)
