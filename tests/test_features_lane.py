"""two-client-engine-01MSK2EN WP09 — the stateless /v1/features push lane (FR-023)."""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kenaz_ml import features_push
from kenaz_ml.features_push import FeaturePushStore


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from kenaz_ml.app import create_app

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("KENAZ_ML_INSTALL_ROOT", str(tmp_path / "install"))
    with TestClient(create_app()) as c:
        yield c


def commit(ts: int, session: str | None = "s1", **extra) -> dict:
    event = {"event_class": "commit", "ts_ms": ts, "fields": {}}
    if session is not None:
        event["session_id"] = session
    event.update(extra)
    return event


def push(client: TestClient, events: list[dict]):
    return client.post("/v1/features", json={"client": "kenaz", "events": events})


def test_commit_is_accepted_and_readable_through_the_accessor(client: TestClient) -> None:
    resp = push(client, [commit(1_700_000_000_000)])
    assert resp.status_code == 200
    assert resp.json() == {"accepted": 1, "refused": 0, "refusals": []}
    assert features_push.commit_ts_ms("s1") == 1_700_000_000_000
    assert features_push.commit_ts_ms("never-pushed") is None
    assert features_push.latest("commit", "s1")["client"] == "kenaz"


def test_global_commit_without_session(client: TestClient) -> None:
    push(client, [commit(5, session=None)])
    assert features_push.commit_ts_ms() == 5
    assert features_push.commit_ts_ms("s1") is None


def test_refusals_are_typed_and_distinct_and_never_block_good_events(client: TestClient) -> None:
    events = [
        {"event_class": "teleport", "ts_ms": 1},
        {"event_class": "egress_deny", "ts_ms": 1, "fields": {"host": "x"}},
        commit(0),
        commit(-5),
        commit("123"),  # type: ignore[arg-type]
        commit(1.5),  # type: ignore[arg-type]
        commit(10, fields={"sha": "abc"}),
        commit(20),
    ]
    body = push(client, events).json()
    assert body["accepted"] == 1
    reasons = {r["index"]: r["reason"] for r in body["refusals"]}
    assert reasons == {
        0: "unknown_class",
        1: "class_schema_not_defined",
        2: "invalid_ts_ms",
        3: "invalid_ts_ms",
        4: "invalid_ts_ms",
        5: "invalid_ts_ms",
        6: "invalid_fields",
    }
    assert features_push.commit_ts_ms("s1") == 20


@pytest.mark.parametrize("cls", sorted(features_push.DEFERRED_CLASSES))
def test_every_deferred_class_is_refused_not_accepted(client: TestClient, cls: str) -> None:
    body = push(client, [{"event_class": cls, "ts_ms": 1}]).json()
    assert body["refusals"][0]["reason"] == "class_schema_not_defined"


def test_oversized_batch_is_refused(client: TestClient) -> None:
    resp = push(client, [commit(i + 1) for i in range(features_push.MAX_BATCH + 1)])
    assert resp.status_code == 422
    assert resp.json()["refusal"]["reason"] == "batch_too_large"
    assert features_push.commit_ts_ms("s1") is None


def test_replayed_older_commit_does_not_move_the_answer_backwards(client: TestClient) -> None:
    push(client, [commit(200), commit(100)])
    assert features_push.commit_ts_ms("s1") == 200


def test_store_is_bounded_oldest_first() -> None:
    store = FeaturePushStore(per_key_limit=3, total_limit=5, clock=lambda: 1.0)
    for i in range(1, 11):
        store.push("c", [commit(i, session="a")])
    assert [e["ts_ms"] for e in store.events("commit", "a")] == [8, 9, 10]
    for i in range(1, 4):
        store.push("c", [commit(100 + i, session=f"k{i}")])
    assert len(store) == 5
    # Global cap evicted the oldest session-a events first.
    assert [e["ts_ms"] for e in store.events("commit", "a")] == [9, 10]
    assert store.commit_ts_ms("k1") == 101


def test_order_bookkeeping_stays_bounded() -> None:
    store = FeaturePushStore(per_key_limit=2, total_limit=10)
    store.push("c", [commit(1, session="long-lived")])
    for i in range(1000):
        store.push("c", [commit(i + 2, session="hot")])
    assert len(store._order) <= 2 * 10 + 1
    assert store.commit_ts_ms("long-lived") == 1


def test_concurrent_pushes_do_not_corrupt_the_store() -> None:
    store = FeaturePushStore(per_key_limit=50, total_limit=500)

    def worker(n: int) -> None:
        for i in range(200):
            store.push(f"c{n}", [commit(i + 1, session=f"s{n % 4}")])

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store) == sum(len(store.events("commit", f"s{k}")) for k in range(4))
    assert len(store) <= 200


def test_two_apps_do_not_share_buffered_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kenaz_ml.app import create_app

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    with TestClient(create_app()) as first:
        push(first, [commit(42)])
        first_store = features_push.current_store()
        with TestClient(create_app()) as second:
            assert features_push.current_store() is not first_store
            assert features_push.commit_ts_ms("s1") is None
            push(second, [commit(7)])
        assert first_store is not None and first_store.commit_ts_ms("s1") == 42


def test_route_is_in_the_openapi_document(client: TestClient) -> None:
    assert "/v1/features" in client.get("/openapi.json").json()["paths"]


def test_cloud_mode_reports_the_lane_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kenaz_ml.app import create_app
    from kenaz_ml.config import ServingMode

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    with TestClient(create_app(ServingMode.CLOUD)) as c:
        resp = push(c, [commit(1)])
        assert resp.status_code == 422
        assert resp.json()["refusal"]["reason"] == "lane_unavailable"


def test_module_imports_no_http_client_no_sqlite_and_opens_no_file() -> None:
    tree = ast.parse(Path(features_push.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open":
            pytest.fail("features_push opens a file")
    forbidden = {"sqlite3", "socket", "http", "urllib", "requests", "httpx", "aiohttp", "psycopg2"}
    assert not imported & forbidden
