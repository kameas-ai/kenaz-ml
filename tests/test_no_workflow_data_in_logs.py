"""No workflow data in log lines (legal package doc 11, item C19; DPA §10.6).

The hosted service's logs may carry an organization id, an event kind, a
status and a timing, and nothing a developer typed or touched. This test
seeds a stream with unmistakable marker values in the places workflow data
lives -- file paths, command lines, repository and branch names, browser
URLs -- runs the engine's polling, feature extraction and prediction paths
with every logger at DEBUG, and fails if any marker reaches any log record.
It is the automated check the contract promises; it does not scrub, it
proves there is nothing to scrub.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

import pytest

from kenaz_ml.app import AppState
from kenaz_ml.config import ServingMode
from kenaz_ml.datastore.sqlite import SqliteStore
from kenaz_ml.feature_store.resolve import resolve_duration_features, resolve_stuck_features
from kenaz_ml.features import extract_activity_features, extract_stuck_features_from_data
from kenaz_ml.modelstore import LocalModelStore
from kenaz_ml.poller import EventPoller

#: Values that must never appear in a log line. Each is distinctive enough
#: that a false positive is impossible and a substring match is enough.
MARKERS = {
    "path": "/Users/dev/secret-project-ZK9Q/payroll_export.py",
    "command": "aws s3 cp --token AKIAZK9QSECRETTOKEN ./dump.sql s3://bucket",
    "repo": "/Users/dev/secret-project-ZK9Q",
    "branch": "feature/acquire-ZK9Q-target",
    "url": "https://intranet.example.com/ZK9Q/merger-plan",
    "window": "Board deck ZK9Q - confidential.key",
}

SCHEMA = """
CREATE TABLE tasks (
    id TEXT PRIMARY KEY, repo_root TEXT NOT NULL, branch TEXT NOT NULL DEFAULT '',
    phase TEXT NOT NULL DEFAULT 'idle', files TEXT NOT NULL DEFAULT '{}',
    started_at INTEGER NOT NULL, last_active INTEGER NOT NULL, completed_at INTEGER,
    commit_count INTEGER NOT NULL DEFAULT 0, test_runs INTEGER NOT NULL DEFAULT 0,
    test_fails INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, source TEXT NOT NULL,
    payload TEXT NOT NULL, ts INTEGER NOT NULL
);
CREATE TABLE ml_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT NOT NULL, result TEXT NOT NULL,
    confidence REAL NOT NULL, created_at INTEGER NOT NULL, expires_at INTEGER
);
CREATE TABLE ml_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, endpoint TEXT NOT NULL,
    routing TEXT NOT NULL, latency_ms INTEGER NOT NULL, ts INTEGER NOT NULL
);
"""


def _seed(db: Path) -> list[dict]:
    """A stream whose every free-text field is a marker."""
    conn = sqlite3.connect(str(db))
    conn.executescript(SCHEMA)
    now = int(time.time() * 1000)
    conn.execute(
        "INSERT INTO tasks (id, repo_root, branch, phase, files, started_at, last_active) VALUES (?,?,?,?,?,?,?)",
        (
            "task-1",
            MARKERS["repo"],
            MARKERS["branch"],
            "coding",
            json.dumps({MARKERS["path"]: 4}),
            now - 900_000,
            now - 1_000,
        ),
    )
    events = [
        ("file", "fs", {"path": MARKERS["path"], "op": "write"}),
        ("terminal", "shell", {"cmd": MARKERS["command"], "cwd": MARKERS["repo"], "exit": 0}),
        ("process", "proc", {"name": "python", "args": [MARKERS["path"]]}),
        ("browser", "chrome", {"url": MARKERS["url"], "title": MARKERS["window"]}),
        ("hyprland", "wm", {"window": MARKERS["window"], "class": "keynote"}),
        ("file", "fs", {"path": MARKERS["path"], "op": "write"}),
        ("terminal", "shell", {"cmd": "pytest tests/ -k ZK9Q", "cwd": MARKERS["repo"], "exit": 1}),
    ]
    rows = []
    for i, (kind, source, payload) in enumerate(events):
        ts = now - 600_000 + i * 60_000
        conn.execute(
            "INSERT INTO events (kind, source, payload, ts) VALUES (?,?,?,?)", (kind, source, json.dumps(payload), ts)
        )
        rows.append({"id": i + 1, "kind": kind, "source": source, "payload": payload, "ts": ts})
    conn.commit()
    conn.close()
    return rows


@pytest.fixture
def every_logger_at_debug(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.DEBUG)
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("kenaz_ml"):
            logging.getLogger(name).setLevel(logging.DEBUG)
    return caplog


def _offending_records(caplog: pytest.LogCaptureFixture) -> list[str]:
    hits = []
    for record in caplog.records:
        text = record.getMessage()
        if record.exc_text:
            text += " " + record.exc_text
        for marker in MARKERS.values():
            if marker in text or "ZK9Q" in text:
                hits.append(f"{record.name}:{record.levelname}: {text[:200]}")
                break
    return hits


def test_polling_prediction_and_feature_paths_log_no_workflow_data(
    tmp_path: Path, every_logger_at_debug: pytest.LogCaptureFixture
) -> None:
    db = tmp_path / "stream.db"
    rows = _seed(db)
    store = SqliteStore(db)
    store.ensure_tables()
    model_store = LocalModelStore(base_dir=tmp_path / "models")

    state = AppState(ServingMode.LOCAL)
    state.store = store
    state.model_store = model_store
    state.refresh_registry(model_store)
    state.load_models(model_store)
    poller = EventPoller(
        store=store,
        models={
            "stuck": state.stuck,
            "activity": state.activity,
            "workflow": state.workflow,
            "duration": state.duration,
            "quality": state.quality,
        },
    )
    for _ in range(3):
        poller._poll_once()

    # The feature paths the serving routes and the trainer use.
    task = store.get_task_by_id("task-1")
    assert task is not None
    extract_stuck_features_from_data(task, rows)
    for row in rows:
        extract_activity_features(row)
    resolve_stuck_features(store, "task-1")
    resolve_duration_features(store, "task-1")

    predictions = sqlite3.connect(str(db)).execute("SELECT COUNT(*) FROM ml_predictions").fetchone()[0]
    assert predictions > 0, "the loop ran: predictions were written"
    assert every_logger_at_debug.records, "logging was captured"
    offending = _offending_records(every_logger_at_debug)
    assert not offending, "workflow data reached a log line:\n" + "\n".join(offending)
