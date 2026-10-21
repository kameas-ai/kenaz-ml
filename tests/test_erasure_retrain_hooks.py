"""Engine hooks a cloud host uses for a member-erasure retrain.

After a member's data is erased, the host must replace the models that may
have been trained on it and remove the versions they superseded. The engine
supplies the generic pieces, tested here:

* ``TrainingScheduler.force_retrain`` / ``StreamWorker.force_retrain``: a run
  now, reading only what the store and the retained mirror hold now;
* ``Trainer.train_all()["synthetic"]``: which models fell below their minimum;
* ``S3ModelStore.list_models`` / ``delete_model`` / ``purge_versions_before``:
  version removal in a versioned bucket that never touches the current version.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from kenaz_ml.datastore.sqlite import SqliteStore
from kenaz_ml.modelstore import LocalModelStore, S3ModelStore
from kenaz_ml.modelstore.registry import Example, FeatureContract, append_examples, read_retained
from kenaz_ml.modelstore.s3_retained import S3RetainedMirror
from kenaz_ml.training import scheduler as scheduler_mod
from kenaz_ml.training.scheduler import TrainingScheduler
from kenaz_ml.training.trainer import Trainer
from kenaz_ml.worker import StreamWorker
from tests.test_worker import _stream_db

T0 = datetime(2026, 1, 1, tzinfo=UTC)
TENANT = "tenant_a"
MODELS = f"{TENANT}/models/"


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class NoSuchKey(Exception):
    pass


class ClockedBucket:
    """A versioned bucket whose every write is stamped from a settable clock."""

    class exceptions:  # noqa: N801 - boto3's spelling
        NoSuchKey = NoSuchKey

    def __init__(self) -> None:
        self.now = T0
        self.versions: list[dict[str, Any]] = []
        self._n = 0

    def tick(self, minutes: int = 1) -> datetime:
        self.now = self.now + timedelta(minutes=minutes)
        return self.now

    def _add(self, key: str, body: bytes, marker: bool) -> None:
        self._n += 1
        self.versions.append(
            {"Key": key, "VersionId": f"v{self._n}", "Body": body, "marker": marker, "LastModified": self.now}
        )

    def _latest(self, key: str) -> dict[str, Any] | None:
        for v in reversed(self.versions):
            if v["Key"] == key:
                return v
        return None

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:  # noqa: N803
        self._add(Key, Body, False)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        v = self._latest(Key)
        if v is None or v["marker"]:
            raise NoSuchKey(Key)
        return {"Body": _Body(v["Body"])}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        return self.get_object(Bucket=Bucket, Key=Key)

    def list_objects_v2(self, **_: Any) -> dict[str, Any]:
        return {}

    def delete_objects(self, *, Bucket: str, Delete: dict[str, Any]) -> dict[str, Any]:  # noqa: N803
        for obj in Delete["Objects"]:
            if "VersionId" in obj:
                self.versions = [
                    v for v in self.versions if not (v["Key"] == obj["Key"] and v["VersionId"] == obj["VersionId"])
                ]
            else:
                self._add(obj["Key"], b"", True)
        return {}

    def get_paginator(self, op: str) -> Any:
        bucket = self

        class _P:
            def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:  # noqa: N803
                mine = [v for v in bucket.versions if v["Key"].startswith(Prefix)]
                if op == "list_objects_v2":
                    keys = sorted({v["Key"] for v in mine})
                    live = [k for k in keys if not bucket._latest(k)["marker"]]  # type: ignore[index]
                    return [{"Contents": [{"Key": k} for k in live]}]
                latest = {bucket._latest(v["Key"])["VersionId"] for v in mine}  # type: ignore[index]

                def row(v: dict[str, Any]) -> dict[str, Any]:
                    return {
                        "Key": v["Key"],
                        "VersionId": v["VersionId"],
                        "IsLatest": v["VersionId"] in latest,
                        "LastModified": v["LastModified"],
                    }

                return [
                    {
                        "Versions": [row(v) for v in mine if not v["marker"]],
                        "DeleteMarkers": [row(v) for v in mine if v["marker"]],
                    }
                ]

        return _P()

    def bodies(self, prefix: str) -> list[tuple[str, bytes]]:
        return [(v["Key"], v["Body"]) for v in self.versions if v["Key"].startswith(prefix)]


def _save(store: S3ModelStore, bucket: ClockedBucket, name: str, body: bytes) -> str:
    """Write one version in S3ModelStore's layout, named from the bucket's clock.

    (``S3ModelStore.save`` names versions by wall-clock second; writing the
    same layout directly keeps these tests from sleeping between saves.)
    """
    version = bucket.tick().strftime("%Y%m%dT%H%M%SZ")
    bucket.put_object(Bucket="b", Key=f"{MODELS}{name}/{version}/model.joblib", Body=body)
    bucket.put_object(Bucket="b", Key=f"{MODELS}{name}/latest", Body=version.encode())
    assert store.current_version(name) == version
    return version


@pytest.fixture
def bucket() -> ClockedBucket:
    return ClockedBucket()


@pytest.fixture
def store(bucket: ClockedBucket) -> S3ModelStore:
    return S3ModelStore("b", tenant_id=TENANT, client=bucket)


# --- S3ModelStore: listing, deleting, purging superseded versions -------------


def test_save_and_load_round_trip_through_the_pointer(store: S3ModelStore, bucket: ClockedBucket) -> None:
    store.save("stuck", b"weights")
    assert store.load("stuck") == b"weights"
    assert store.list_models() == ["stuck"]


def test_list_models_names_only_prefixes_with_a_pointer(store: S3ModelStore, bucket: ClockedBucket) -> None:
    bucket.put_object(Bucket="b", Key=f"{MODELS}stuck/20260101T000000Z/model.joblib", Body=b"m")
    bucket.put_object(Bucket="b", Key=f"{MODELS}stuck/latest", Body=b"20260101T000000Z")
    bucket.put_object(Bucket="b", Key=f"{MODELS}duration/latest", Body=b"x")
    bucket.put_object(Bucket="b", Key=f"{MODELS}retained/user-1/stuck.jsonl", Body=b"{}")
    bucket.put_object(Bucket="b", Key=f"{MODELS}_records/case.json", Body=b"{}")
    bucket.put_object(Bucket="b", Key="tenant_b/models/other/latest", Body=b"x")
    assert store.list_models() == ["duration", "stuck"]


def test_purge_removes_versions_before_the_cutoff_and_keeps_the_new_one(
    store: S3ModelStore, bucket: ClockedBucket
) -> None:
    _save(store, bucket, "stuck", b"trained-with-erased-member")
    _save(store, bucket, "stuck", b"also-before")
    cutoff = bucket.tick()
    new = _save(store, bucket, "stuck", b"after-retrain")

    assert store.purge_versions_before("stuck", cutoff) == 4  # two artifacts + two pointer versions
    left = bucket.bodies(f"{MODELS}stuck/")
    assert sorted(left) == sorted(
        [(f"{MODELS}stuck/{new}/model.joblib", b"after-retrain"), (f"{MODELS}stuck/latest", new.encode())]
    )
    assert store.load("stuck") == b"after-retrain"
    assert store.purge_versions_before("stuck", cutoff) == 0, "idempotent"


def test_the_current_version_is_never_purged_whatever_its_timestamp(store: S3ModelStore, bucket: ClockedBucket) -> None:
    _save(store, bucket, "stuck", b"old")
    current = _save(store, bucket, "stuck", b"current")
    # A cutoff after every write (e.g. clock skew between host and S3): the
    # live pointer and the artifact it names survive regardless.
    store.purge_versions_before("stuck", bucket.tick(60))
    assert sorted(k for k, _ in bucket.bodies(f"{MODELS}stuck/")) == sorted(
        [f"{MODELS}stuck/{current}/model.joblib", f"{MODELS}stuck/latest"]
    )
    assert store.load("stuck") == b"current"


def test_purge_removes_delete_markers_and_leaves_other_models(store: S3ModelStore, bucket: ClockedBucket) -> None:
    _save(store, bucket, "stuck", b"a")
    _save(store, bucket, "stuck_v2", b"sibling")
    bucket.delete_objects(Bucket="b", Delete={"Objects": [{"Key": f"{MODELS}stuck/stray"}]})
    cutoff = bucket.tick()
    _save(store, bucket, "stuck", b"b")
    store.purge_versions_before("stuck", cutoff)
    assert not [k for k, _ in bucket.bodies(f"{MODELS}stuck/stray")], "the delete marker is gone"
    assert store.load("stuck_v2") == b"sibling", "a model sharing the name's leading characters is untouched"


def test_purge_refuses_a_model_with_no_current_version_and_a_naive_cutoff(store: S3ModelStore) -> None:
    with pytest.raises(LookupError):
        store.purge_versions_before("stuck", T0)
    with pytest.raises(ValueError, match="timezone-aware"):
        store.purge_versions_before("stuck", T0.replace(tzinfo=None))


@pytest.mark.parametrize("name", ["", "a/b", "..", "."])
def test_bad_model_names_are_refused(store: S3ModelStore, name: str) -> None:
    with pytest.raises(ValueError, match="refusing model name"):
        store.delete_model(name)


def test_delete_model_removes_every_version_and_only_that_model(store: S3ModelStore, bucket: ClockedBucket) -> None:
    _save(store, bucket, "pattern_detector", b"1")
    _save(store, bucket, "pattern_detector", b"2")
    _save(store, bucket, "pattern_detector_x", b"keep")
    bucket.put_object(Bucket="b", Key=f"{MODELS}retained/u/stuck.jsonl", Body=b"r")
    assert store.delete_model("pattern_detector") == 4
    assert bucket.bodies(f"{MODELS}pattern_detector/") == []
    assert store.load("pattern_detector") is None
    assert store.list_models() == ["pattern_detector_x"]
    assert bucket.bodies(f"{MODELS}retained/") != []
    assert store.delete_model("pattern_detector") == 0, "idempotent"


# --- Trainer: which models fell below their minimum ---------------------------


def test_train_all_names_the_models_fitted_on_synthetic_data(tmp_path: Path) -> None:
    store = SqliteStore(_stream_db(tmp_path / "s.db"))
    try:
        result = Trainer(store, model_store=LocalModelStore(base_dir=tmp_path / "m")).train_all()
    finally:
        store.close()
    assert set(result["synthetic"]) == {"stuck", "duration"}, "no completed tasks: both fall back to synthetic"
    assert set(result["synthetic"]) <= set(result["trained"])
    assert "pattern_detector" not in result["trained"], "below its minimum and no fallback: nothing written"


# --- TrainingScheduler.force_retrain ---------------------------------------------


CONTRACT = FeatureContract(names=("a",), service_version="c1")
PREFIX = f"{MODELS}retained/user-1/"


class _RecordingTrainer:
    """Stands in for Trainer: records the retained examples it could read."""

    seen: list[int] = []
    gate: threading.Event | None = None
    fail = False

    def __init__(self, store: Any, model_store: Any = None, *, retained_dir: Path | None = None) -> None:
        self._dir = retained_dir

    def train_all(self) -> dict:
        if _RecordingTrainer.gate is not None:
            _RecordingTrainer.gate.wait(5)
        if _RecordingTrainer.fail:
            raise RuntimeError("fit failed")
        assert self._dir is not None
        _RecordingTrainer.seen.append(len(read_retained("stuck", directory=self._dir)))
        return {"trained": ["stuck"], "synthetic": [], "samples": 1, "duration_sec": 0.0}


class _Store:
    def count_completed_tasks(self) -> int:
        return 0

    def insert_ml_event(self, *a: Any) -> None:
        pass

    def commit(self) -> None:
        pass


@pytest.fixture
def recording(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingTrainer]:
    _RecordingTrainer.seen = []
    _RecordingTrainer.gate = None
    _RecordingTrainer.fail = False
    monkeypatch.setattr(scheduler_mod, "Trainer", _RecordingTrainer)
    return _RecordingTrainer


def test_force_retrain_reads_only_what_the_mirror_holds_now(
    tmp_path: Path, bucket: ClockedBucket, recording: type[_RecordingTrainer]
) -> None:
    work = tmp_path / "work"
    # A stale scratch copy holding examples the mirror no longer has (they
    # were erased from S3 after an earlier run left this copy behind).
    now = int(time.time() * 1000)
    append_examples("stuck", [Example(x=(1.0,), y=1.0, as_of_ms=now)] * 3, CONTRACT, directory=work)
    seed = tmp_path / "seed"
    append_examples("stuck", [Example(x=(2.0,), y=0.0, as_of_ms=now)], CONTRACT, directory=seed)
    S3RetainedMirror(bucket, "b", PREFIX).push(seed)

    sched = TrainingScheduler(_Store(), retained_dir=work, retained_mirror=S3RetainedMirror(bucket, "b", PREFIX))  # type: ignore[arg-type]
    reloads: list[int] = []
    sched._reload = lambda: reloads.append(1)
    result = sched.force_retrain()

    assert recording.seen == [1], "the stale local examples were replaced by the mirror's before training"
    assert result["trained"] == ["stuck"]
    assert reloads == [1]
    assert not work.exists()


def test_force_retrain_ignores_the_threshold_and_the_interval(recording: type[_RecordingTrainer]) -> None:
    sched = TrainingScheduler(_Store(), retained_dir=None)  # type: ignore[arg-type]
    sched._last_retrain = time.time()  # inside the minimum interval
    sched.check_and_retrain()
    assert recording.seen == [], "the scheduled check declines"
    sched._retained_dir = Path("/nonexistent-dir-for-test")
    sched.force_retrain()
    assert recording.seen == [0]


def test_force_retrain_raises_on_failure(recording: type[_RecordingTrainer]) -> None:
    recording.fail = True
    with pytest.raises(RuntimeError, match="fit failed"):
        TrainingScheduler(_Store()).force_retrain()  # type: ignore[arg-type]


def test_busy_while_training_and_the_scheduled_check_steps_aside(recording: type[_RecordingTrainer]) -> None:
    recording.gate = threading.Event()
    sched = TrainingScheduler(_Store(), retained_dir=Path("/nonexistent-dir-for-test"))  # type: ignore[arg-type]
    t = threading.Thread(target=sched.force_retrain)
    t.start()
    deadline = time.time() + 5
    while not sched.busy and time.time() < deadline:
        time.sleep(0.01)
    assert sched.busy
    started = time.time()
    sched.check_and_retrain()
    sched.sweep_retention()
    assert time.time() - started < 1, "neither waits for the run in progress"
    recording.gate.set()
    t.join(5)
    assert not sched.busy


def test_a_stream_worker_exposes_readiness_and_forced_retrain(tmp_path: Path) -> None:
    worker = StreamWorker(
        name="tenant_a/user-1",
        store=SqliteStore(_stream_db(tmp_path / "s.db")),
        model_store=LocalModelStore(base_dir=tmp_path / "m"),
    )
    assert not worker.ready and not worker.training_in_progress
    assert worker.completed_task_count() == 0
    with pytest.raises(RuntimeError, match="not set up"):
        worker.force_retrain()
    worker.setup()
    try:
        assert worker.ready
        result = worker.force_retrain()
        assert "stuck" in result["synthetic"]
        assert not worker.training_in_progress
    finally:
        worker.store.close()


# --- review regressions ----------------------------------------------------------


class _FlakyGet(ClockedBucket):
    """get_object fails once with a non-404 error (throttling, a timeout)."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_next_get = False

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if self.fail_next_get:
            self.fail_next_get = False
            err = Exception("SlowDown")
            err.response = {"Error": {"Code": "SlowDown"}}  # type: ignore[attr-defined]
            raise err
        return super().get_object(Bucket=Bucket, Key=Key)


def test_a_transient_read_error_is_raised_not_mistaken_for_a_missing_model() -> None:
    bucket = _FlakyGet()
    store = S3ModelStore("b", tenant_id=TENANT, client=bucket)
    _save(store, bucket, "stuck", b"old")
    cutoff = bucket.tick()
    new = _save(store, bucket, "stuck", b"new")
    before = list(bucket.versions)

    bucket.fail_next_get = True
    with pytest.raises(Exception, match="SlowDown") as info:
        store.purge_versions_before("stuck", cutoff)
    assert not isinstance(info.value, LookupError), "a failed read is not 'no current version'"
    assert bucket.versions == before, "nothing deleted"

    bucket.fail_next_get = True
    with pytest.raises(Exception, match="SlowDown"):
        store.current_version("stuck")
    assert store.current_version("stuck") == new


def test_only_a_404_means_no_current_version(store: S3ModelStore, bucket: ClockedBucket) -> None:
    assert store.current_version("absent") is None  # NoSuchKey

    class ClientError(Exception):
        response = {"Error": {"Code": "404"}}

    def raise_404(**_: Any) -> dict[str, Any]:
        raise ClientError()

    bucket.get_object = raise_404  # type: ignore[method-assign]
    assert store.current_version("stuck") is None
    with pytest.raises(LookupError):
        store.purge_versions_before("stuck", T0)


def test_a_noncurrent_version_s3_stamps_after_the_cutoff_is_kept(store: S3ModelStore, bucket: ClockedBucket) -> None:
    """Clock skew only moves the boundary for noncurrent versions; the live one is kept by key."""
    _save(store, bucket, "stuck", b"before")
    cutoff = bucket.tick()
    _save(store, bucket, "stuck", b"stamped-after-cutoff")  # e.g. S3's clock ahead of the host's
    live = _save(store, bucket, "stuck", b"live")
    store.purge_versions_before("stuck", cutoff)
    bodies = [b for k, b in bucket.bodies(f"{MODELS}stuck/") if k.endswith("model.joblib")]
    assert bodies == [b"stamped-after-cutoff", b"live"]
    assert store.current_version("stuck") == live


def test_a_queued_forced_retrain_counts_as_busy(recording: type[_RecordingTrainer]) -> None:
    recording.gate = threading.Event()
    sched = TrainingScheduler(_Store(), retained_dir=Path("/nonexistent-dir-for-test"))  # type: ignore[arg-type]
    sched._lock.acquire()  # another run holds the stream
    t = threading.Thread(target=sched.force_retrain)
    t.start()
    deadline = time.time() + 5
    while sched._waiting == 0 and time.time() < deadline:
        time.sleep(0.01)
    assert sched._waiting == 1 and sched.busy
    sched._lock.release()  # the first run ends; the queued one has not yet taken the lock
    assert sched.busy, "never idle between the two runs"
    recording.gate.set()
    t.join(5)
    assert not sched.busy and recording.seen == [0]
