"""Retained sets in a versioned S3 bucket, and their age bound.

The bucket keeps every version, so a rewrite or a delete that left old
versions behind would keep expired or erased examples readable. These tests
run the mirror against a fake versioned bucket and assert what is left.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from kenaz_ml.modelstore.registry import Example, FeatureContract, append_examples, expire_examples, read_retained
from kenaz_ml.modelstore.s3_retained import S3RetainedMirror, purge_prefix
from kenaz_ml.training.scheduler import TrainingScheduler

DAY_MS = 86_400_000


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class VersionedBucket:
    """Just enough of a versioned S3 bucket: every put and delete adds a version."""

    def __init__(self) -> None:
        self.versions: list[dict[str, Any]] = []  # oldest first
        self._next = 0

    def _new_id(self) -> str:
        self._next += 1
        return f"v{self._next}"

    def _latest(self, key: str) -> dict[str, Any] | None:
        for v in reversed(self.versions):
            if v["Key"] == key:
                return v
        return None

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.versions.append({"Key": Key, "VersionId": self._new_id(), "Body": Body, "marker": False})

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        v = self._latest(Key)
        assert v is not None and not v["marker"]
        return {"Body": _Body(v["Body"])}

    def delete_objects(self, *, Bucket: str, Delete: dict[str, Any]) -> dict[str, Any]:
        for obj in Delete["Objects"]:
            if "VersionId" in obj:
                self.versions = [
                    v for v in self.versions if not (v["Key"] == obj["Key"] and v["VersionId"] == obj["VersionId"])
                ]
            else:
                self.versions.append({"Key": obj["Key"], "VersionId": self._new_id(), "Body": b"", "marker": True})
        return {}

    def get_paginator(self, op: str) -> Any:
        bucket = self

        class _P:
            def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:
                if op == "list_objects_v2":
                    keys = sorted({v["Key"] for v in bucket.versions if v["Key"].startswith(Prefix)})
                    live = [k for k in keys if not bucket._latest(k)["marker"]]  # type: ignore[index]
                    return [{"Contents": [{"Key": k} for k in live]}]
                latest_ids = {bucket._latest(v["Key"])["VersionId"] for v in bucket.versions}  # type: ignore[index]
                mine = [v for v in bucket.versions if v["Key"].startswith(Prefix)]
                return [
                    {
                        "Versions": [
                            {"Key": v["Key"], "VersionId": v["VersionId"], "IsLatest": v["VersionId"] in latest_ids}
                            for v in mine
                            if not v["marker"]
                        ],
                        "DeleteMarkers": [
                            {"Key": v["Key"], "VersionId": v["VersionId"], "IsLatest": v["VersionId"] in latest_ids}
                            for v in mine
                            if v["marker"]
                        ],
                    }
                ]

        return _P()

    def keys(self, prefix: str = "") -> list[str]:
        return [v["Key"] for v in self.versions if v["Key"].startswith(prefix)]


CONTRACT = FeatureContract(names=("a",), service_version="c1")
PREFIX = "tenant_a/models/retained/user-1/"


def _append(directory: Path, *ages_days: float) -> None:
    now = int(time.time() * 1000)
    examples = [Example(x=(1.0,), y=1.0, as_of_ms=now - int(a * DAY_MS)) for a in ages_days]
    assert append_examples("stuck", examples, CONTRACT, directory=directory).ok


def test_expire_drops_only_examples_older_than_the_bound(tmp_path: Path) -> None:
    _append(tmp_path, 40, 10, 1)
    cutoff = int(time.time() * 1000) - 30 * DAY_MS
    assert expire_examples("stuck", before_ms=cutoff, directory=tmp_path) == 1
    kept = read_retained("stuck", directory=tmp_path)
    assert kept.ok and len(kept) == 2 and kept.generation is not None


def test_a_push_leaves_exactly_one_version_of_each_file(tmp_path: Path) -> None:
    bucket = VersionedBucket()
    mirror = S3RetainedMirror(bucket, "b", PREFIX)
    _append(tmp_path, 1)
    mirror.push(tmp_path)
    _append(tmp_path, 0)
    mirror.push(tmp_path)
    assert bucket.keys() == [PREFIX + "stuck.jsonl"], "no noncurrent version may survive a rewrite"


def test_a_file_removed_locally_is_gone_from_every_version(tmp_path: Path) -> None:
    bucket = VersionedBucket()
    mirror = S3RetainedMirror(bucket, "b", PREFIX)
    _append(tmp_path, 1)
    mirror.push(tmp_path)
    (tmp_path / "stuck.jsonl").unlink()
    mirror.push(tmp_path)
    assert bucket.keys() == []


def test_pull_round_trips_and_replaces_stale_local_files(tmp_path: Path) -> None:
    bucket = VersionedBucket()
    src, dst = tmp_path / "src", tmp_path / "dst"
    _append(src, 1, 2)
    S3RetainedMirror(bucket, "b", PREFIX).push(src)
    dst.mkdir()
    (dst / "duration.jsonl").write_text("stale")
    assert S3RetainedMirror(bucket, "b", PREFIX).pull(dst) == 1
    assert sorted(p.name for p in dst.iterdir()) == ["stuck.jsonl"]
    assert len(read_retained("stuck", directory=dst)) == 2


def test_purge_removes_every_version_and_marker_and_nothing_else() -> None:
    bucket = VersionedBucket()
    bucket.put_object(Bucket="b", Key=PREFIX + "stuck.jsonl", Body=b"1")
    bucket.put_object(Bucket="b", Key=PREFIX + "stuck.jsonl", Body=b"2")
    bucket.delete_objects(Bucket="b", Delete={"Objects": [{"Key": PREFIX + "stuck.jsonl"}]})
    bucket.put_object(Bucket="b", Key="tenant_a/models/retained/user-10/stuck.jsonl", Body=b"x")
    assert purge_prefix(bucket, "b", PREFIX) == 3
    assert bucket.keys() == ["tenant_a/models/retained/user-10/stuck.jsonl"], "a sibling prefix is untouched"


@pytest.mark.parametrize("prefix", ["tenant_a/models", "", "/"])
def test_purge_refuses_a_prefix_that_is_not_a_directory(prefix: str) -> None:
    with pytest.raises(ValueError, match="refusing to purge"):
        purge_prefix(VersionedBucket(), "b", prefix)


def test_a_training_run_expires_pushes_and_leaves_no_local_copy(tmp_path: Path) -> None:
    bucket = VersionedBucket()
    seed = tmp_path / "seed"
    _append(seed, 100, 1)
    S3RetainedMirror(bucket, "b", PREFIX).push(seed)

    work = tmp_path / "work"
    scheduler = TrainingScheduler(
        store=None,  # type: ignore[arg-type]
        retained_dir=work,
        retained_mirror=S3RetainedMirror(bucket, "b", PREFIX),
        retention_days=lambda: 30,
    )
    seen: list[int] = []
    scheduler._with_retained(lambda: seen.append(len(read_retained("stuck", directory=work))))

    assert seen == [1], "training sees the set already expired"
    assert not work.exists(), "the scratch copy is always cleared"
    pulled = tmp_path / "check"
    S3RetainedMirror(bucket, "b", PREFIX).pull(pulled)
    assert len(read_retained("stuck", directory=pulled)) == 1
    assert len(bucket.keys()) == 1


def test_the_scratch_copy_is_cleared_even_when_training_fails(tmp_path: Path) -> None:
    bucket = VersionedBucket()
    work = tmp_path / "work"
    scheduler = TrainingScheduler(
        store=None,  # type: ignore[arg-type]
        retained_dir=work,
        retained_mirror=S3RetainedMirror(bucket, "b", PREFIX),
    )

    def boom() -> None:
        _append(work, 1)
        raise RuntimeError("fit failed")

    with pytest.raises(RuntimeError):
        scheduler._with_retained(boom)
    assert not work.exists()


def test_a_mirror_without_a_directory_is_refused() -> None:
    with pytest.raises(ValueError, match="retained_dir"):
        TrainingScheduler(store=None, retained_mirror=S3RetainedMirror(VersionedBucket(), "b", PREFIX))  # type: ignore[arg-type]
