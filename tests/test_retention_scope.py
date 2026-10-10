"""Retained sets are scoped to one stream in the cloud worker.

The default retention directory is one per install: right for the local
deployment's single person, wrong for a worker process that trains many
organizations' streams. These tests pin that a Trainer over a non-filesystem
model store retains only into an explicit per-stream directory, and that the
worker hands each stream its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kenaz_ml import config
from kenaz_ml.modelstore import LocalModelStore
from kenaz_ml.training import trainer as trainer_mod
from kenaz_ml.training.trainer import Trainer
from kenaz_ml.worker import stream_retained_dir


class _RemoteModelStore:
    """A model store that is not filesystem-backed, as S3ModelStore is not."""

    def load(self, model_name: str) -> bytes | None:
        return None

    def save(self, model_name: str, data: bytes) -> None:
        pass

    def exists(self, model_name: str) -> bool:
        return False


@pytest.fixture
def appended(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record the directory every retention append would write to."""
    calls: list[Any] = []

    def fake(model_name, contract, vectors, labels, as_of, *, directory=None):  # noqa: ANN001
        calls.append(directory)
        return None

    monkeypatch.setattr(trainer_mod, "_retain_examples", fake)
    return calls


def _retain(trainer: Trainer) -> None:
    trainer._retain("stuck", None, [[1.0]], [1.0], [1])


def test_a_non_filesystem_store_without_a_stream_directory_retains_nothing(appended: list[Any]) -> None:
    _retain(Trainer(store=None, model_store=_RemoteModelStore()))  # type: ignore[arg-type]
    assert appended == [], "a cloud trainer must never fall back to the shared install directory"


def test_a_stream_directory_is_where_a_cloud_trainer_retains(appended: list[Any], tmp_path: Path) -> None:
    target = tmp_path / "tenant_a" / "user-1"
    _retain(Trainer(store=None, model_store=_RemoteModelStore(), retained_dir=target))  # type: ignore[arg-type]
    assert appended == [target]


def test_the_local_deployment_still_retains_to_its_default_directory(appended: list[Any], tmp_path: Path) -> None:
    _retain(Trainer(store=None, model_store=LocalModelStore(base_dir=tmp_path)))  # type: ignore[arg-type]
    assert appended == [None], "local behaviour is unchanged: the default directory"


def test_two_organizations_never_share_a_retained_directory() -> None:
    a = stream_retained_dir("tenant_a", "user-1")
    b = stream_retained_dir("tenant_b", "user-1")
    c = stream_retained_dir("tenant_a", "user-2")
    assert len({a, b, c}) == 3
    for d in (a, b, c):
        assert config.retained_data_dir() in d.parents
    assert a.parent != b.parent


@pytest.mark.parametrize("user", ["..", ".", "a/b", "a\\b", "has space", ""])
def test_a_stream_id_that_could_escape_its_directory_is_refused(user: str) -> None:
    with pytest.raises(ValueError, match="retained-set directory"):
        stream_retained_dir("tenant_a", user)


def test_a_bad_tenant_id_is_refused() -> None:
    with pytest.raises(ValueError, match="retained-set directory"):
        stream_retained_dir("../tenant_a", "user-1")


def test_a_stream_worker_hands_its_retained_directory_to_training(tmp_path: Path) -> None:
    from kenaz_ml.datastore.sqlite import SqliteStore
    from kenaz_ml.worker import StreamWorker
    from tests.test_worker import _stream_db

    target = tmp_path / "retained" / "tenant_a" / "user-1"
    worker = StreamWorker(
        name="tenant_a/user-1",
        store=SqliteStore(str(_stream_db(tmp_path / "s.db"))),
        model_store=LocalModelStore(base_dir=tmp_path / "models"),
        retained_dir=target,
    )
    worker.setup()
    assert worker._scheduler._retained_dir == target
