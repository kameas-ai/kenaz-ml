"""Activity-DB path selection, the locality guard, and local retention.

- The path comes only from ``KENAZ_ML_ACTIVITY_DB`` or ``--activity-db`` (the
  flag wins), must be absolute, and is never derived.
- An activity DB with cloud mode or an S3 model store refuses to start.
- ``KENAZ_ML_ACTIVITY_RETENTION_DAYS`` follows an exact contract: 1..3650 prunes,
  ``hold`` skips, anything else skips with one warning; never clamped, no default.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from kenaz_ml import config
from kenaz_ml.datastore.sqlite import SqliteStore

DAY = 86_400_000


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        config.ACTIVITY_DB_ENV,
        config.ACTIVITY_RETENTION_ENV,
        "KENAZ_MODE",
        "SIGIL_MODE",
        "KENAZ_ML_MODE",
        "SIGIL_ML_MODE",
        "KENAZ_S3_BUCKET",
        "SIGIL_S3_BUCKET",
        "SIGIL_ML_ACTIVITY_DB",
    ):
        # set-then-delete so monkeypatch records the original state and restores
        # it, including after code under test writes os.environ directly.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


# ---------------------------------------------------------------------------
# Path selection
# ---------------------------------------------------------------------------


def test_no_activity_db_means_the_sigild_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert config.activity_db_path() is None
    assert config.local_db_path() == tmp_path / "sigild" / "data.db"


def test_the_env_var_selects_the_activity_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "ml" / "activity.db"
    monkeypatch.setenv(config.ACTIVITY_DB_ENV, str(target))
    assert config.activity_db_path() == target
    assert config.local_db_path() == target


def test_the_flag_wins_over_the_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(config.ACTIVITY_DB_ENV, str(tmp_path / "env.db"))
    assert config.activity_db_path(str(tmp_path / "flag.db")) == tmp_path / "flag.db"


def test_an_empty_value_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.ACTIVITY_DB_ENV, "  ")
    assert config.activity_db_path() is None
    assert config.activity_db_path("") is None


@pytest.mark.parametrize("value", ["activity.db", "./ml/activity.db", "~/activity.db"])
def test_a_relative_path_is_refused(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(config.ACTIVITY_DB_ENV, value)
    with pytest.raises(config.ActivityDbConfigError, match="absolute"):
        config.activity_db_path()
    with pytest.raises(config.ActivityDbConfigError):
        config.activity_db_path(value)


def test_the_legacy_env_spelling_is_not_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SIGIL_ML_ACTIVITY_DB", str(tmp_path / "legacy.db"))
    assert config.activity_db_path() is None


def test_create_store_opens_the_activity_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from kenaz_ml.datastore import create_store

    target = tmp_path / "activity.db"
    monkeypatch.setenv(config.ACTIVITY_DB_ENV, str(target))
    store = create_store("local")
    assert isinstance(store, SqliteStore) and store.db_path == target


def test_cli_flag_is_bridged_into_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from kenaz_ml import cli

    monkeypatch.setenv(config.ACTIVITY_DB_ENV, str(tmp_path / "env.db"))
    cli._apply_activity_db(str(tmp_path / "flag.db"), serving_mode="local")
    assert config.activity_db_path() == tmp_path / "flag.db"


def test_cli_refuses_a_relative_flag(capsys: pytest.CaptureFixture[str]) -> None:
    from kenaz_ml import cli

    with pytest.raises(SystemExit) as exc:
        cli._apply_activity_db("activity.db", serving_mode="local")
    assert exc.value.code == 2
    assert "absolute" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Locality guard
# ---------------------------------------------------------------------------


def test_no_activity_db_never_trips_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KENAZ_MODE", "cloud")
    config.check_activity_db_locality(serving_mode="cloud")


def test_local_mode_with_an_activity_db_is_allowed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from kenaz_ml.modelstore import LocalModelStore

    monkeypatch.setenv(config.ACTIVITY_DB_ENV, str(tmp_path / "a.db"))
    config.check_activity_db_locality(serving_mode="local", model_store=LocalModelStore(tmp_path / "m"))


def test_cloud_serving_mode_with_an_activity_db_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(config.ACTIVITY_DB_ENV, str(tmp_path / "a.db"))
    with pytest.raises(config.LocalityError, match="serving mode is cloud"):
        config.check_activity_db_locality(serving_mode="cloud")


def test_cloud_operating_mode_with_an_activity_db_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(config.ACTIVITY_DB_ENV, str(tmp_path / "a.db"))
    monkeypatch.setenv("KENAZ_MODE", "cloud")
    with pytest.raises(config.LocalityError, match="KENAZ_MODE is cloud"):
        config.check_activity_db_locality(serving_mode="local")


def test_an_s3_model_store_with_an_activity_db_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from kenaz_ml.modelstore import CachedModelStore, S3ModelStore

    monkeypatch.setenv(config.ACTIVITY_DB_ENV, str(tmp_path / "a.db"))
    s3 = S3ModelStore.__new__(S3ModelStore)  # no client needed: the type is the question
    with pytest.raises(config.LocalityError, match="model store is S3"):
        config.check_activity_db_locality(serving_mode="local", model_store=s3)
    with pytest.raises(config.LocalityError, match="model store is S3"):
        config.check_activity_db_locality(serving_mode="local", model_store=CachedModelStore(s3, ttl_seconds=1))


def test_create_store_refuses_cloud_with_an_activity_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from kenaz_ml.datastore import create_store

    monkeypatch.setenv(config.ACTIVITY_DB_ENV, str(tmp_path / "a.db"))
    with pytest.raises(config.LocalityError):
        create_store("cloud")


@pytest.mark.parametrize("argv", [["serve", "--mode", "cloud"], ["train", "--mode", "cloud", "--tenant", "t"]])
def test_cli_refuses_cloud_with_an_activity_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    from kenaz_ml import cli

    monkeypatch.setattr(cli, "setup_logging", lambda: None)
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **k: pytest.fail("must not start"))
    monkeypatch.setattr(sys, "argv", ["kenaz-ml", *argv, "--activity-db", str(tmp_path / "a.db")])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert config.ACTIVITY_DB_ENV in capsys.readouterr().err


def test_the_worker_refuses_with_an_activity_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from kenaz_ml import cli

    monkeypatch.setattr(cli, "setup_logging", lambda: None)
    monkeypatch.setenv(config.ACTIVITY_DB_ENV, str(tmp_path / "a.db"))
    monkeypatch.setattr(sys, "argv", ["kenaz-ml", "worker", "--tenants", "t"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_the_app_refuses_to_start_in_cloud_mode_with_an_activity_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    from kenaz_ml.app import create_app
    from kenaz_ml.config import ServingMode

    monkeypatch.setenv(config.ACTIVITY_DB_ENV, str(tmp_path / "a.db"))
    app = create_app(ServingMode.CLOUD)
    with pytest.raises(config.LocalityError), TestClient(app):
        pass


# ---------------------------------------------------------------------------
# Retention contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "days", "reason"),
    [
        ("30", 30, "valid"),
        (" 30 ", 30, "valid"),
        ("1", 1, "valid"),
        ("3650", 3650, "valid"),
        ("hold", None, "hold"),
        ("HOLD", None, "hold"),
        (" Hold ", None, "hold"),
        (None, None, "missing"),
        ("", None, "empty"),
        ("   ", None, "empty"),
        ("0", None, "invalid"),
        ("-5", None, "invalid"),
        ("3651", None, "invalid"),
        ("abc", None, "invalid"),
        ("30.0", None, "invalid"),
        ("+30", None, "invalid"),
        ("30d", None, "invalid"),
    ],
)
def test_retention_contract(raw: str | None, days: int | None, reason: str) -> None:
    setting = config.parse_activity_retention(raw)
    assert (setting.days, setting.reason) == (days, reason)


@pytest.mark.parametrize("activity_db", [True, False], ids=["activity-db", "no-activity-db"])
@pytest.mark.parametrize(
    ("raw", "warns_with_db", "warns_without_db"),
    [
        # With no activity DB retention is off whatever the value: never a warning.
        ("30", False, False),
        ("hold", False, False),
        (None, True, False),
        ("", True, False),
        ("0", True, False),
        ("3651", True, False),
        ("abc", True, False),
    ],
)
def test_retention_warns_once_exactly_when_unusable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    raw: str | None,
    warns_with_db: bool,
    warns_without_db: bool,
    activity_db: bool,
) -> None:
    warns = warns_with_db if activity_db else warns_without_db
    if activity_db:
        monkeypatch.setenv(config.ACTIVITY_DB_ENV, str(tmp_path / "activity.db"))
    if raw is not None:
        monkeypatch.setenv(config.ACTIVITY_RETENTION_ENV, raw)
    with caplog.at_level(logging.DEBUG, logger="kenaz_ml.config"):
        setting = config.activity_retention()
    warnings = [
        r for r in caplog.records if r.levelno >= logging.WARNING and config.ACTIVITY_RETENTION_ENV in r.getMessage()
    ]
    assert len(warnings) == (1 if warns else 0)
    if not activity_db:
        assert (setting.days, setting.reason) == (None, "no_activity_db"), "no activity DB: never prunes"
        assert any(r.levelno == logging.DEBUG for r in caplog.records), "the silent skip still logs at debug"
    elif raw is None:
        assert setting.days is None, "unset never prunes"


def _activity_db(path: Path, now_ms: int) -> None:
    ddl = (Path(__file__).parent / "fixtures" / "harness_activity_db.sql").read_text()
    conn = sqlite3.connect(str(path))
    conn.executescript(ddl)
    for age_days in (1, 29, 31, 400):
        ts = now_ms - age_days * DAY
        conn.execute(
            "INSERT INTO ml_predictions (model, result, confidence, created_at) VALUES ('stuck', '{}', 0.5, ?)", (ts,)
        )
        conn.execute(
            "INSERT INTO ml_events (kind, endpoint, routing, latency_ms, ts) VALUES ('p', 'poller', 'local', 1, ?)",
            (ts,),
        )
        conn.execute(
            "INSERT INTO events (kind, source, payload, ts, user_id, node_id, device_event_id, producer) "
            "VALUES ('file', 'h', '{}', ?, '', '', ?, 'harness')",
            (ts, age_days),
        )
        conn.execute(
            "INSERT INTO tasks (user_id, node_id, id, repo_root, started_at, last_active, completed_at, producer) "
            "VALUES ('', '', ?, '/r', ?, ?, ?, 'harness')",
            (f"t{age_days}", ts, ts, ts),
        )
    conn.commit()
    conn.close()


def test_prune_deletes_only_old_engine_rows(tmp_path: Path) -> None:
    now = int(time.time() * 1000)
    path = tmp_path / "activity.db"
    _activity_db(path, now)
    deleted = SqliteStore(path).prune_ml_rows(30, now_ms=now)
    assert deleted == {"ml_predictions": 2, "ml_events": 2}
    conn = sqlite3.connect(str(path))
    try:
        assert conn.execute("SELECT COUNT(*) FROM ml_predictions").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM ml_events").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 4, "events are never pruned"
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 4, "tasks are never pruned"
    finally:
        conn.close()


def test_prune_runs_in_begin_immediate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = int(time.time() * 1000)
    path = tmp_path / "activity.db"
    _activity_db(path, now)
    store = SqliteStore(path)
    statements: list[str] = []
    real_connect = store._connect

    def traced() -> sqlite3.Connection:
        conn = real_connect()
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(store, "_connect", traced)
    store.prune_ml_rows(30, now_ms=now)
    assert [s for s in statements if s.upper().startswith("BEGIN")] == ["BEGIN IMMEDIATE"]
    assert statements[-1].upper() == "COMMIT"


def test_prune_tolerates_missing_tables(tmp_path: Path) -> None:
    assert SqliteStore(tmp_path / "empty.db").prune_ml_rows(30) == {}


@pytest.mark.parametrize("bad", [0, -1, True])
def test_prune_rejects_a_non_positive_retention(tmp_path: Path, bad: int) -> None:
    with pytest.raises(ValueError):
        SqliteStore(tmp_path / "x.db").prune_ml_rows(bad)


def test_prune_once_logs_and_swallows_failures(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from kenaz_ml.app import prune_once

    class Broken(SqliteStore):
        def prune_ml_rows(self, retention_days: int, *, now_ms: int | None = None) -> dict[str, int]:
            raise sqlite3.OperationalError("database is locked")

    with caplog.at_level(logging.WARNING):
        assert prune_once(Broken(tmp_path / "x.db"), 30) == {}
    assert any("retention" in r.getMessage() for r in caplog.records)


def _local_lifespan_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv(config.INSTALL_ROOT_ENV, str(tmp_path / "install"))


@pytest.mark.parametrize(
    ("raw", "expected_days"),
    [("30", 30), ("hold", None), (None, None), ("", None), ("0", None), ("3651", None), ("abc", None)],
)
def test_the_app_schedules_pruning_only_for_a_valid_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: str | None, expected_days: int | None
) -> None:
    from fastapi.testclient import TestClient

    from kenaz_ml import app as app_mod
    from kenaz_ml.config import ServingMode

    _local_lifespan_env(monkeypatch, tmp_path)
    monkeypatch.setenv(config.ACTIVITY_DB_ENV, str(tmp_path / "activity.db"))
    if raw is not None:
        monkeypatch.setenv(config.ACTIVITY_RETENTION_ENV, raw)
    calls: list[int] = []

    async def fake_loop(store: SqliteStore, days: int) -> None:
        assert store.db_path == tmp_path / "activity.db"
        calls.append(days)

    monkeypatch.setattr(app_mod, "activity_retention_loop", fake_loop)
    with TestClient(app_mod.create_app(ServingMode.LOCAL)):
        pass
    assert calls == ([] if expected_days is None else [expected_days])


def test_a_valid_value_without_an_activity_db_prunes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """sigild's data.db ml_* tables are Go-owned: Python only inserts, never prunes them."""
    from fastapi.testclient import TestClient

    from kenaz_ml import app as app_mod
    from kenaz_ml.config import ServingMode

    _local_lifespan_env(monkeypatch, tmp_path)
    monkeypatch.setenv(config.ACTIVITY_RETENTION_ENV, "1")
    data_db = tmp_path / "xdg" / "sigild" / "data.db"
    data_db.parent.mkdir(parents=True)
    old = int(time.time() * 1000) - 400 * DAY
    conn = sqlite3.connect(str(data_db))
    conn.executescript(
        "CREATE TABLE ml_predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT NOT NULL, result TEXT NOT NULL,"
        " confidence REAL NOT NULL, created_at INTEGER NOT NULL, expires_at INTEGER);"
        "CREATE TABLE ml_events (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, endpoint TEXT NOT NULL,"
        " routing TEXT NOT NULL, latency_ms INTEGER NOT NULL, ts INTEGER NOT NULL);"
    )
    conn.execute(
        "INSERT INTO ml_predictions (model, result, confidence, created_at) VALUES ('stuck', '{}', 0.5, ?)", (old,)
    )
    conn.execute("INSERT INTO ml_events (kind, endpoint, routing, latency_ms, ts) VALUES ('p', 'x', 'l', 1, ?)", (old,))
    conn.commit()
    conn.close()

    calls: list[int] = []

    async def spy_loop(store: SqliteStore, days: int) -> None:
        calls.append(days)

    monkeypatch.setattr(app_mod, "activity_retention_loop", spy_loop)
    pruned: list[int] = []
    monkeypatch.setattr(SqliteStore, "prune_ml_rows", lambda self, days, **k: pruned.append(days) or {})
    with TestClient(app_mod.create_app(ServingMode.LOCAL)):
        pass
    assert calls == [] and pruned == []
    conn = sqlite3.connect(str(data_db))
    try:
        assert conn.execute("SELECT COUNT(*) FROM ml_predictions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM ml_events").fetchone()[0] == 1
    finally:
        conn.close()


def test_train_refuses_db_and_activity_db_together(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from kenaz_ml import cli

    monkeypatch.setattr(cli, "setup_logging", lambda: None)
    monkeypatch.setattr(cli, "Trainer", lambda *a, **k: pytest.fail("must not train"))
    monkeypatch.setattr(
        sys, "argv", ["kenaz-ml", "train", "--db", str(tmp_path / "d.db"), "--activity-db", str(tmp_path / "a.db")]
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--db" in err and "--activity-db" in err
