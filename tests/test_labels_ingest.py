"""two-client-engine-01MSK2EN WP04 — /v1/labels/{kind} under the frozen contract (A3.3)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kenaz_ml import config
from kenaz_ml.advice import label_log
from kenaz_ml.advice.contracts import contract_for
from kenaz_ml.advice.dispatch import build_table
from kenaz_ml.modelstore.registry import Example, append_examples, read_retained

KIND = "compact_now"
CONTRACT = contract_for(KIND)
assert CONTRACT is not None
NAMES = CONTRACT.names


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from kenaz_ml.app import AppState
    from kenaz_ml.routes import register_routes

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    base = tmp_path / "base"
    monkeypatch.setattr(config, "base_models_dir", lambda: base)
    state = AppState()
    state.dispatch_table = build_table(config.models_dir(), base)
    app = FastAPI()
    register_routes(app, state)
    return {"client": TestClient(app), "retained": config.retained_data_dir()}


def row(ts: int, *, revision: int = 1, action: str = "accepted", complete: bool = True, **extra: Any) -> dict:
    features = {name: float(i + ts % 7) for i, name in enumerate(NAMES)}
    if not complete:
        features.pop(NAMES[-1])
    base = {
        "client": "harness",
        "kind": KIND,
        "features_hash": f"h{ts}",
        "ts": ts,
        "revision": revision,
        "feature_contract_version": CONTRACT.service_version,
        "features": features,
        "features_complete": complete,
        "shown": True,
        "confidence": 81,
        "model_id": "heuristic/fill-threshold",
        "rung": "heuristic",
        "prompt_version": "p1",
        "user_action": action,
        "recommendation": {"decision": True},
        "latency_ms": 3.5,
        "session_id": "s1",
    }
    base.update(extra)
    return base


def push(client: TestClient, rows: list[dict], kind: str = KIND) -> Any:
    return client.post(f"/v1/labels/{kind}", json={"client": "harness", "rows": rows})


def labels_on_disk(retained: Path) -> list[dict]:
    path = retained / f"{KIND}.labels.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def retained_rows(retained: Path) -> tuple[Example, ...]:
    return read_retained(KIND, directory=retained).examples


# ---------------------------------------------------------------------------
# Scenario 1 / SC-005 — idempotency
# ---------------------------------------------------------------------------


def test_same_batch_twice_is_stored_once(env: dict) -> None:
    batch = [row(1000), row(2000, action="dismissed")]
    first = push(env["client"], batch).json()
    second = push(env["client"], batch).json()

    assert first["accepted"] == 2
    assert second["accepted"] == 0 and second["duplicate"] == 2
    assert len(labels_on_disk(env["retained"])) == 2
    assert len(retained_rows(env["retained"])) == 2


def test_internal_duplicates_in_one_batch(env: dict) -> None:
    body = push(env["client"], [row(1000), row(1000), row(1000)]).json()
    assert body["accepted"] == 1 and body["duplicate"] == 2
    assert len(labels_on_disk(env["retained"])) == 1
    assert len(retained_rows(env["retained"])) == 1


def test_cursor_zero_repush_adds_nothing(env: dict) -> None:
    history = [row(t) for t in range(1000, 1010)]
    for chunk in (history[:4], history[4:]):
        push(env["client"], chunk)
    replay = push(env["client"], history).json()
    assert replay["accepted"] == 0 and replay["duplicate"] == 10
    assert len(labels_on_disk(env["retained"])) == 10
    assert len(retained_rows(env["retained"])) == 10


def test_index_survives_process_restart(env: dict) -> None:
    push(env["client"], [row(1000)])
    label_log._INDEXES.clear()  # simulate a fresh process: index rebuilt from disk
    assert push(env["client"], [row(1000)]).json()["duplicate"] == 1


# ---------------------------------------------------------------------------
# Scenario 3 — propensity fields survive in the label log; Example unchanged
# ---------------------------------------------------------------------------


def test_shown_false_round_trips_through_the_label_log(env: dict) -> None:
    push(env["client"], [row(1000, shown=False, confidence=42, action="ignored")])
    (record,) = labels_on_disk(env["retained"])  # read off disk, not out of the response
    assert record["shown"] is False
    assert record["confidence"] == 42
    for name in ("features_complete", "model_id", "user_action", "rung", "prompt_version", "features_hash", "ts"):
        assert name in record
    assert record["record"] == "label"
    assert [f.name for f in dataclasses.fields(Example)] == ["x", "y", "as_of_ms"]


# ---------------------------------------------------------------------------
# Completeness — features_complete=false never reaches retained
# ---------------------------------------------------------------------------


def test_incomplete_row_is_logged_flagged_and_never_retained(env: dict) -> None:
    body = push(env["client"], [row(1000, complete=False)]).json()
    assert body["accepted"] == 1
    (record,) = labels_on_disk(env["retained"])
    assert record["features_complete"] is False
    assert retained_rows(env["retained"]) == ()


# ---------------------------------------------------------------------------
# Scenario 2 — contract mismatch is loud and writes nothing
# ---------------------------------------------------------------------------


def test_contract_version_mismatch_is_a_409_and_writes_nothing(env: dict) -> None:
    resp = push(env["client"], [row(1000), row(2000, feature_contract_version="deadbeefdeadbeef")])
    assert resp.status_code == 409
    refusal = resp.json()["refusal"]
    assert refusal["reason"] == "contract_mismatch"
    assert "deadbeefdeadbeef" in refusal["detail"] and CONTRACT.service_version in refusal["detail"]
    assert not (env["retained"] / f"{KIND}.labels.jsonl").exists()
    assert not (env["retained"] / f"{KIND}.jsonl").exists()


def test_names_mismatch_is_a_409(env: dict) -> None:
    bad = row(1000)
    bad["features"]["surplus"] = 1.0
    resp = push(env["client"], [bad])
    assert resp.status_code == 409
    assert resp.json()["refusal"]["reason"] == "names_mismatch"
    assert "surplus" in resp.json()["refusal"]["detail"]


def test_retained_header_mismatch_is_visible(env: dict) -> None:
    other = dataclasses.replace(CONTRACT, service_version="0123456789abcdef")
    assert append_examples(KIND, [], other, directory=env["retained"]).ok
    resp = push(env["client"], [row(1000)])
    assert resp.status_code == 409
    refusal = resp.json()["refusal"]
    assert refusal["reason"] == "retained_refused"
    assert "contract-version-mismatch" in refusal["detail"]
    assert not (env["retained"] / f"{KIND}.labels.jsonl").exists()


def test_unknown_kind_is_refused_with_a_diagnostic(env: dict) -> None:
    resp = push(env["client"], [row(1000)], kind="never_registered")
    assert resp.status_code == 404
    assert resp.json()["refusal"]["reason"] == "unknown_kind"
    assert "never_registered" in resp.json()["refusal"]["detail"]


# ---------------------------------------------------------------------------
# Scenario 4 — the existing cap and eviction apply unchanged
# ---------------------------------------------------------------------------


def test_retained_cap_uses_the_existing_contiguous_eviction(env: dict) -> None:
    rows = [row(t) for t in range(1000, 1200)]
    result = label_log.ingest(KIND, "harness", rows, CONTRACT, retained_max_bytes=4096)
    assert result.refusal is None
    kept = retained_rows(env["retained"])
    assert 0 < len(kept) < 200
    # Contiguous oldest-first: what survives is the newest suffix, in order.
    assert [e.as_of_ms for e in kept] == list(range(1200 - len(kept), 1200))
    assert len(labels_on_disk(env["retained"])) == 200


# ---------------------------------------------------------------------------
# Scenario 5 — revision upsert
# ---------------------------------------------------------------------------


def test_higher_revision_replaces_and_lower_is_ignored(env: dict) -> None:
    push(env["client"], [row(1000, revision=1, action="ignored")])
    up = push(env["client"], [row(1000, revision=2, action="dismissed")]).json()
    assert up["superseded"] == 1
    (record,) = labels_on_disk(env["retained"])
    assert record["revision"] == 2 and record["user_action"] == "dismissed"
    # ignored -> dismissed is an ordinary append
    assert [e.y for e in retained_rows(env["retained"])] == [0.0]

    stale = push(env["client"], [row(1000, revision=2, action="accepted"), row(1000, revision=1)]).json()
    assert stale["duplicate"] == 2
    (record,) = labels_on_disk(env["retained"])
    assert record["user_action"] == "dismissed"


def test_label_changing_revision_rebuilds_retained_consistently(env: dict) -> None:
    push(env["client"], [row(1000, action="accepted"), row(2000, action="accepted")])
    before = read_retained(KIND, directory=env["retained"]).generation
    body = push(env["client"], [row(1000, revision=2, action="dismissed")]).json()
    assert body["retained_rebuilt"] is True
    assert body["retained_generation"] != before

    labels = {r["ts"]: r for r in labels_on_disk(env["retained"])}
    assert labels[1000]["user_action"] == "dismissed"
    got = sorted((e.as_of_ms, e.y) for e in retained_rows(env["retained"]))
    assert got == [(1000, 0.0), (2000, 1.0)]


def test_revision_to_untrainable_drops_it_from_retained(env: dict) -> None:
    push(env["client"], [row(1000, action="accepted")])
    push(env["client"], [row(1000, revision=5, action="ignored")])
    assert retained_rows(env["retained"]) == ()


# ---------------------------------------------------------------------------
# Scenario 6 — training mapping
# ---------------------------------------------------------------------------


def test_training_mapping(env: dict) -> None:
    push(
        env["client"],
        [
            row(1000, action="accepted"),
            row(2000, action="auto_acted"),
            row(3000, action="dismissed"),
            row(4000, action="ignored"),
        ],
    )
    assert sorted((e.as_of_ms, e.y) for e in retained_rows(env["retained"])) == [
        (1000, 1.0),
        (2000, 1.0),
        (3000, 0.0),
    ]
    assert len(labels_on_disk(env["retained"])) == 4


def test_as_of_ms_is_carried_not_recomputed(env: dict) -> None:
    push(env["client"], [row(5000, as_of_ms=4321)])
    assert [e.as_of_ms for e in retained_rows(env["retained"])] == [4321]


# ---------------------------------------------------------------------------
# Ack cursor over (ts, revision)
# ---------------------------------------------------------------------------


def test_ack_covers_only_the_contiguous_prefix(env: dict) -> None:
    rows = [row(3000), row(1000), row(2000, action="bogus"), row(1000, revision=2)]
    body = push(env["client"], rows).json()
    # sorted: (1000,1) accepted, (1000,2) superseded, (2000,1) refused, (3000,1) accepted
    assert body["ack"] == {"ts": 1000, "revision": 2}
    assert body["refused"] == 1
    assert body["refusals"][0]["reason"] == "unknown_user_action"
    assert body["refusals"][0]["ts"] == 2000


def test_first_row_refused_means_no_ack(env: dict) -> None:
    body = push(env["client"], [row(1000, client="someone-else"), row(2000)]).json()
    assert body["ack"] is None
    assert body["refusals"][0]["reason"] == "client_mismatch"


def test_full_success_acks_the_last_row(env: dict) -> None:
    body = push(env["client"], [row(1000), row(2000, revision=3)]).json()
    assert body["ack"] == {"ts": 2000, "revision": 3}


# ---------------------------------------------------------------------------
# Reader tolerance and the shared writer
# ---------------------------------------------------------------------------


def test_reader_skips_malformed_and_truncated_lines(env: dict) -> None:
    push(env["client"], [row(1000)])
    path = env["retained"] / f"{KIND}.labels.jsonl"
    with path.open("a") as handle:
        handle.write("not json\n")
        handle.write('{"record": "label", "trunc')
    read = label_log.read_label_log(KIND, directory=env["retained"])
    assert read.skipped_lines == 1
    assert read.truncated_final_line is True
    assert len(read.labels()) == 1
    # And a later push still lands on its own line.
    push(env["client"], [row(2000)])
    assert len(label_log.read_label_log(KIND, directory=env["retained"]).labels()) == 2


def test_other_record_types_survive_an_upsert_rewrite(env: dict) -> None:
    push(env["client"], [row(1000)])
    label_log.append_records(KIND, [{"record": "shadow", "features_hash": "h1000", "ts": 1000}])
    push(env["client"], [row(1000, revision=2, action="dismissed")])
    kinds = [r["record"] for r in labels_on_disk(env["retained"])]
    assert sorted(kinds) == ["label", "shadow"]


def test_label_log_is_bounded(tmp_path: Path) -> None:
    rows = [row(t) for t in range(1000, 1100)]
    label_log.ingest(KIND, "harness", rows, CONTRACT, directory=tmp_path, max_bytes=4096)
    path = tmp_path / f"{KIND}.labels.jsonl"
    assert path.stat().st_size <= 4096
    surviving = [json.loads(line)["ts"] for line in path.read_text().splitlines()]
    assert surviving == list(range(1100 - len(surviving), 1100))
