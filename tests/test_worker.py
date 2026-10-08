"""The cloud worker: the engine's loop per stream, outside the HTTP app.

Exercised against SQLite with the daemon's schema, which is the same
``DataStore`` protocol the Postgres store implements; what is under test is
the worker's lifecycle (setup, poll, retrain checks, stop) and its isolation
of one stream's failure from the others, not the stores.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

from kenaz_ml.datastore.sqlite import SqliteStore
from kenaz_ml.modelstore import LocalModelStore
from kenaz_ml.worker import StreamWorker, run_workers

DAEMON_SCHEMA = """
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


def _stream_db(path: Path, *, events: int = 6) -> Path:
    """One active task with a few file events: enough for the poller to predict."""
    conn = sqlite3.connect(str(path))
    conn.executescript(DAEMON_SCHEMA)
    now = int(time.time() * 1000)
    conn.execute(
        "INSERT INTO tasks (id, repo_root, branch, phase, files, started_at, last_active) VALUES (?,?,?,?,?,?,?)",
        ("task-1", "/repo", "main", "coding", '{"a.py": 3}', now - 600_000, now - 1_000),
    )
    for i in range(events):
        conn.execute(
            "INSERT INTO events (kind, source, payload, ts) VALUES (?,?,?,?)",
            ("file", "fs", '{"path": "a.py", "op": "write"}', now - 500_000 + i * 60_000),
        )
    conn.commit()
    conn.close()
    return path


def _run(workers: list[StreamWorker], *, for_sec: float) -> None:
    async def main() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(run_workers(workers, stop))
        await asyncio.sleep(for_sec)
        stop.set()
        await asyncio.wait_for(task, timeout=10)

    asyncio.run(main())


def test_a_stream_worker_polls_predicts_and_stops_cleanly(tmp_path: Path) -> None:
    db = _stream_db(tmp_path / "stream.db")
    worker = StreamWorker(
        name="org-a/user-1",
        store=SqliteStore(db),
        model_store=LocalModelStore(base_dir=tmp_path / "models"),
        poll_interval_sec=0.05,
        retrain_check_sec=0.1,
    )
    _run([worker], for_sec=1.0)

    assert worker.state.stuck is not None, "models were loaded"
    assert worker.state.poller is not None
    assert worker.state.poller._running is False, "the poller was stopped"
    rows = sqlite3.connect(str(db)).execute("SELECT COUNT(*) FROM ml_predictions").fetchone()[0]
    assert rows > 0, "the loop ran and wrote predictions for the stream"


def test_one_streams_failure_does_not_stop_the_others(tmp_path: Path) -> None:
    class BrokenStore:
        """A store whose schema cannot be reached: setup must fail loudly."""

        def ensure_tables(self) -> None:
            raise RuntimeError('relation "events" does not exist')

        def close(self) -> None:
            pass

    good_db = _stream_db(tmp_path / "good.db")
    good = StreamWorker(
        name="org-a/user-good",
        store=SqliteStore(good_db),
        model_store=LocalModelStore(base_dir=tmp_path / "models"),
        poll_interval_sec=0.05,
        retrain_check_sec=0.1,
    )
    bad = StreamWorker(
        name="org-a/user-bad",
        store=BrokenStore(),  # type: ignore[arg-type]
        model_store=LocalModelStore(base_dir=tmp_path / "models"),
    )
    _run([bad, good], for_sec=1.0)

    assert good.state.poller is not None and good.state.poller._running is False
    rows = sqlite3.connect(str(good_db)).execute("SELECT COUNT(*) FROM ml_predictions").fetchone()[0]
    assert rows > 0
    assert bad.state.poller is None, "the broken stream never started polling"
