"""Cloud offline store and point-in-time training retrieval (WP04).

What is genuinely exercised here, and what is not
------------------------------------------------
Everything about the **write** side is real: real tasks, real extractors from
``sigil_ml.features``, the real reference-time resolver, the real row objects
that get persisted.

The **join** side is real too, but not over PostgreSQL. No PostgreSQL server is
available in this environment (only libpq's client tools), so these tests run
Feast's actual ``get_historical_features`` against its Dask offline store, over
parquet written by :func:`~sigil_ml.feature_store.materialize.rows_to_source_frame`
-- the same projection the production SQL performs, sharing its column list. The
feature views, the TTLs, the feature services, the entity frame and the as-of
join are the production objects and the production code path; only the physical
storage differs.

What that leaves unexercised is exactly one thing: that PostgreSQL executes
``feature_view_source_query`` as written. That is asserted structurally below
(table, predicate, timestamp field, projected columns) and covered end-to-end by
``test_live_postgres_point_in_time``, which runs the identical scenario through
``PostgreSQLSource`` when ``SIGIL_ML_TEST_POSTGRES_URL`` is set and skips
otherwise. It is skipped here.

None of these tests mock retrieval. A mocked as-of join would assert nothing
about the property the whole work package exists to establish.
"""

from __future__ import annotations

import inspect
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from sigil_ml.feature_store import definitions, materialize
from sigil_ml.feature_store.materialize import (
    DURATION_OFFLINE_VIEW,
    ML_FEATURES_DDL,
    ML_FEATURES_INSERT,
    ML_FEATURES_TABLE,
    OFFLINE_FEATURE_VIEWS,
    STUCK_OFFLINE_VIEW,
    MaterializationResult,
    OfflineFeatureRow,
    OfflineRetrievalError,
    OfflineStoreUnavailableError,
    PartialRetrievalError,
    build_offline_rows,
    cloud_feature_services,
    cloud_feature_views,
    feature_service_version,
    feature_view_source_query,
    materialize_tasks,
    rows_to_source_frame,
    source_query_columns,
)
from sigil_ml.features import (
    extract_duration_features_from_data,
    extract_stuck_features_from_data,
)
from sigil_ml.models.duration import FEATURE_NAMES as DURATION_FEATURE_NAMES
from sigil_ml.models.stuck import FEATURE_NAMES as STUCK_FEATURE_NAMES
from sigil_ml.training import cloud_trainer as cloud_trainer_module
from sigil_ml.training.cloud_trainer import CloudTrainer
from sigil_ml.training.models import CloudTrainingConfig
from sigil_ml.training.trainer import _reference_time_for

pd = pytest.importorskip("pandas")

# ---------------------------------------------------------------------------
# Clock fixtures
# ---------------------------------------------------------------------------
# The tasks live in January 2026; materialization is frozen in July 2026. The
# gap is six months rather than six seconds on purpose: if a writer ever stamped
# `event_timestamp` with the write time, the failure is a timestamp half a year
# away from the task it describes, not a few milliseconds that could be waved
# through as clock skew.

T0 = datetime(2026, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
T0_MS = int(T0.timestamp() * 1000)
HOUR_MS = 3_600_000

FROZEN_NOW = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)
FROZEN_NOW_MS = int(FROZEN_NOW.timestamp() * 1000)


@pytest.fixture(autouse=True)
def _isolate_models(tmp_path, monkeypatch):
    """Keep model weights and any data dir out of the developer's real config."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))


# ---------------------------------------------------------------------------
# Fixtures: tasks, events, stores
# ---------------------------------------------------------------------------


def _task(
    task_id: str,
    *,
    completed_at: int | None = None,
    last_active: int | None = None,
    started_at: int | None = None,
    test_fails: int = 0,
    branch: str = "feat/offline-store",
    files: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A completed-task record shaped like the rows the DataStore returns."""
    task: dict[str, Any] = {
        "id": task_id,
        "started_at": started_at if started_at is not None else T0_MS - HOUR_MS,
        "test_fails": test_fails,
        "branch": branch,
        "files": files if files is not None else {"a.py": 1, "b.py": 1},
    }
    if completed_at is not None:
        task["completed_at"] = completed_at
    if last_active is not None:
        task["last_active"] = last_active
    return task


def _edit(ts_ms: int, path: str = "a.py") -> dict[str, Any]:
    return {"kind": "edit", "ts": ts_ms, "payload": {"file": path}}


class FakeDataStore:
    """The slice of the DataStore protocol materialization and training touch.

    Deliberately narrow: materialization reads completed tasks and their events
    and writes nothing, so a store that offered more would be describing a
    capability this path must not have.
    """

    def __init__(
        self,
        tasks_per_tenant: dict[str, list[dict]] | None = None,
        events_per_task: dict[str, list[dict]] | None = None,
    ) -> None:
        self._tasks = tasks_per_tenant or {}
        self._events = events_per_task or {}
        self.ml_events: list[dict] = []
        self.commits = 0

    def get_completed_tasks_for_tenant(self, tenant_id: str) -> list[dict]:
        return [dict(task) for task in self._tasks.get(tenant_id, [])]

    def get_events_for_task_id(self, task_id: str) -> list[dict]:
        return [dict(event) for event in self._events.get(task_id, [])]

    def get_last_training_ts(self, tenant_id: str) -> float | None:
        return None

    def insert_ml_event(self, kind: str, endpoint: str, routing: str, latency_ms: int) -> None:
        self.ml_events.append({"kind": kind, "endpoint": endpoint, "routing": routing, "latency_ms": latency_ms})

    def commit(self) -> None:
        self.commits += 1


class FakeModelStore:
    def __init__(self) -> None:
        self.saved: dict[str, bytes] = {}

    def load(self, model_name: str) -> bytes | None:
        return self.saved.get(model_name)

    def save(self, model_name: str, data: bytes) -> None:
        self.saved[model_name] = data

    def exists(self, model_name: str) -> bool:
        return model_name in self.saved


class RecordingSink:
    """An offline sink that keeps rows in memory instead of persisting them."""

    def __init__(self) -> None:
        self.rows: list[OfflineFeatureRow] = []

    def write(self, rows) -> int:
        self.rows.extend(rows)
        return len(rows)


# ---------------------------------------------------------------------------
# A real Feast store over the materialized rows
# ---------------------------------------------------------------------------


def _parquet_source_factory(directory: Path):
    """Bind each feature view to parquet holding that view's projected rows.

    The parquet files genuinely exist and genuinely contain the materializer's
    output -- this is a substitute *backend*, not a substitute for data. The
    April 2026 attempt's failure was pointing shipped definitions at paths that
    were never written; nothing here ships, and everything here is written.
    """
    from feast import FileSource

    def factory(view) -> Any:
        return FileSource(
            name=f"{view.name}__{ML_FEATURES_TABLE}",
            path=str(directory / f"{view.name}.parquet"),
            timestamp_field="event_timestamp",
            created_timestamp_column="created_at",
        )

    return factory


def build_offline_feature_store(tmp_path: Path, rows: list[OfflineFeatureRow]) -> Any:
    """Materialize ``rows`` into a real, queryable Feast store.

    Uses the production ``cloud_feature_views`` / ``cloud_feature_services``
    wiring, the production row projection, and Feast's own point-in-time join.
    """
    from feast import FeatureStore
    from feast.repo_config import RepoConfig

    repo = tmp_path / "feast_repo"
    data = repo / "data"
    data.mkdir(parents=True, exist_ok=True)

    for view in OFFLINE_FEATURE_VIEWS.values():
        frame = rows_to_source_frame(rows, view)
        frame.to_parquet(data / f"{view.name}.parquet", index=False)

    config = RepoConfig(
        project="sigil_ml",
        provider="local",
        repo_path=str(repo),
        registry={"registry_type": "file", "path": str(repo / "registry.db")},
        online_store={"type": "sqlite", "path": str(repo / "online.db")},
        offline_store={"type": "dask"},
        entity_key_serialization_version=3,
    )
    store = FeatureStore(config=config)
    views = cloud_feature_views(source_factory=_parquet_source_factory(data))
    materialize.apply_cloud_definitions(store, views)
    return store


def _entity_frame(pairs: list[tuple[str, datetime]]) -> Any:
    return pd.DataFrame(
        {
            "task_id": [task_id for task_id, _ in pairs],
            "event_timestamp": [moment for _, moment in pairs],
        }
    )


def _row(task_id: str, moment: datetime, values: dict[str, float], *, created: datetime | None = None):
    """A stuck_features row written at ``created`` describing ``moment``."""
    full = {name: 0.0 for name in STUCK_FEATURE_NAMES}
    full.update(values)
    return OfflineFeatureRow(
        entity_type="task",
        entity_id=task_id,
        feature_view=STUCK_OFFLINE_VIEW.name,
        event_timestamp_ms=int(moment.timestamp() * 1000),
        created_at_ms=int((created or FROZEN_NOW).timestamp() * 1000),
        feature_values=full,
    )


# ===========================================================================
# T013 -- the ml_features table
# ===========================================================================


def _ddl_text() -> str:
    return "\n".join(ML_FEATURES_DDL)


def test_ddl_matches_the_data_model():
    ddl = _ddl_text()
    assert f"CREATE TABLE IF NOT EXISTS {ML_FEATURES_TABLE}" in ddl
    for column, type_name in (
        ("entity_type", "TEXT"),
        ("entity_id", "TEXT"),
        ("feature_view", "TEXT"),
        ("event_timestamp", "TIMESTAMPTZ"),
        ("created_at", "TIMESTAMPTZ"),
        ("feature_values", "JSONB"),
    ):
        assert f"{column}      " in ddl or f"{column} " in ddl
        assert type_name in ddl, f"{column} should be declared {type_name}"


def test_event_timestamp_is_not_null():
    """A row without an event timestamp cannot answer a lookup, so it must not exist."""
    create = ML_FEATURES_DDL[0]
    line = next(ln for ln in create.splitlines() if "event_timestamp" in ln)
    assert "NOT NULL" in line
    assert "TIMESTAMPTZ" in line


def test_primary_key_is_the_point_in_time():
    create = ML_FEATURES_DDL[0]
    assert "PRIMARY KEY (entity_type, entity_id, feature_view, event_timestamp)" in create


def test_point_in_time_index_present_and_descending():
    index = ML_FEATURES_DDL[1]
    assert "ml_features_pit" in index
    assert "(entity_type, entity_id, feature_view, event_timestamp DESC)" in index


def test_created_at_audit_only_role_is_documented_in_the_schema():
    """The prohibition ships with the table, not only with this module."""
    ddl = _ddl_text()
    comment = next(s for s in ML_FEATURES_DDL if "COMMENT ON COLUMN" in s and "created_at" in s)
    assert "Audit only" in comment
    assert "NEVER be used for retrieval ordering" in comment
    assert "feature-extraction-correctness" in comment
    # And the module says so too, for anyone reading the code rather than psql.
    assert "audit" in materialize.__doc__.lower()
    assert "ml_features.event_timestamp" not in ddl.replace(f"{ML_FEATURES_TABLE}.event_timestamp", "", 1)


def test_no_go_owned_table_is_touched():
    """C-004: the offline store is additive and Python-owned."""
    statements = [*ML_FEATURES_DDL, ML_FEATURES_INSERT]
    go_owned = ("events", "tasks", "patterns", "suggestions", "ml_predictions", "ml_events")
    for statement in statements:
        for verb in ("ALTER TABLE", "DROP TABLE", "DELETE FROM", "UPDATE "):
            assert verb not in statement.upper() or ML_FEATURES_TABLE in statement
        for table in go_owned:
            assert f" {table} " not in statement.replace("\n", " "), (
                f"{table} is Go-owned; the offline store must not name it"
            )


def test_insert_is_an_upsert_keyed_on_the_point_in_time():
    assert "ON CONFLICT (entity_type, entity_id, feature_view, event_timestamp)" in ML_FEATURES_INSERT
    assert "created_at = EXCLUDED.created_at" in ML_FEATURES_INSERT
    # Re-materializing must not invent a new moment.
    assert "event_timestamp = EXCLUDED" not in ML_FEATURES_INSERT


def test_postgres_sink_creates_schema_and_writes(monkeypatch):
    """The sink issues the DDL and the upsert through a DB-API connection."""

    class FakeCursor:
        def __init__(self, log):
            self.log = log

        def execute(self, sql, params=None):
            self.log.append(("execute", sql))

        def executemany(self, sql, seq):
            self.log.append(("executemany", sql, list(seq)))

        def close(self):
            pass

    class FakeConnection:
        def __init__(self):
            self.log: list[tuple] = []
            self.commits = 0

        def cursor(self):
            return FakeCursor(self.log)

        def commit(self):
            self.commits += 1

    connection = FakeConnection()
    sink = materialize.PostgresOfflineSink(connection)
    sink.create_schema()
    assert [entry[1] for entry in connection.log] == list(ML_FEATURES_DDL)

    row = _row("t1", T0, {"edit_velocity": 1.0})
    assert sink.write([row]) == 1
    kind, sql, params = connection.log[-1]
    assert kind == "executemany"
    assert sql == ML_FEATURES_INSERT
    assert params[0][3] == row.event_timestamp
    assert params[0][4] == row.created_at
    assert '"edit_velocity": 1.0' in params[0][5]
    assert sink.write([]) == 0


# ===========================================================================
# T014 / T016 -- materialization writes the described moment
# ===========================================================================


def test_event_timestamp_is_the_reference_time_not_the_write_time():
    """FR-009, the property the whole package rests on."""
    completed = T0_MS
    task = _task("t1", completed_at=completed)
    result = build_offline_rows([task], lambda _: [_edit(T0_MS - 1000)], now_ms=FROZEN_NOW_MS)

    assert result.rows, "expected rows for a task with a resolvable reference time"
    for row in result.rows:
        assert row.event_timestamp_ms == completed
        assert row.event_timestamp_ms == _reference_time_for(task)
        assert row.created_at_ms == FROZEN_NOW_MS


def test_event_timestamp_differs_from_created_at_for_a_historical_task():
    """If the two ever coincide, nothing proves the right value was stored."""
    task = _task("t1", completed_at=T0_MS)
    result = build_offline_rows([task], lambda _: [], now_ms=FROZEN_NOW_MS)

    for row in result.rows:
        assert row.event_timestamp_ms != row.created_at_ms
        gap_days = (row.created_at - row.event_timestamp) / timedelta(days=1)
        assert gap_days > 180, (
            "the frozen write clock is six months past the fixtures; a smaller gap means "
            "event_timestamp drifted toward the write time"
        )
        assert row.event_timestamp < row.created_at


def test_last_active_is_the_fallback_reference_time():
    """The resolver's second branch, exercised through the writer."""
    task = _task("t2", last_active=T0_MS + HOUR_MS)
    result = build_offline_rows([task], lambda _: [], now_ms=FROZEN_NOW_MS)
    assert {row.event_timestamp_ms for row in result.rows} == {T0_MS + HOUR_MS}


def test_completed_at_wins_over_last_active():
    task = _task("t3", completed_at=T0_MS, last_active=T0_MS + HOUR_MS)
    result = build_offline_rows([task], lambda _: [], now_ms=FROZEN_NOW_MS)
    assert {row.event_timestamp_ms for row in result.rows} == {T0_MS}


def test_unresolvable_tasks_are_skipped_and_counted_once(caplog):
    """No wall-clock fallback, and no log line per row."""
    resolvable = _task("good", completed_at=T0_MS)
    unresolvable = [_task("bad1"), _task("bad2"), _task("bad3")]
    for task in unresolvable:
        task.pop("completed_at", None)
        task.pop("last_active", None)

    with caplog.at_level(logging.INFO, logger="sigil_ml.feature_store.materialize"):
        result = build_offline_rows([resolvable, *unresolvable], lambda _: [], now_ms=FROZEN_NOW_MS)

    assert result.skipped == 3
    assert result.materialized_tasks == 1
    assert {row.entity_id for row in result.rows} == {"good"}

    skip_lines = [r for r in caplog.records if "no resolvable reference time" in r.getMessage()]
    assert len(skip_lines) == 1, "the skip count is logged once, not per row"
    assert "3" in skip_lines[0].getMessage()


def test_zero_timestamps_are_treated_as_absent():
    """`0` is not a real epoch in this data; the resolver already says so."""
    task = _task("t4", completed_at=0, last_active=0)
    result = build_offline_rows([task], lambda _: [], now_ms=FROZEN_NOW_MS)
    assert result.rows == []
    assert result.skipped == 1


def test_materialization_stores_exactly_what_the_extractor_produces():
    """Materialization must not transform. Same input, same numbers."""
    task = _task("t1", completed_at=T0_MS, test_fails=4)
    events = [_edit(T0_MS - 60_000, "a.py"), _edit(T0_MS - 30_000, "b.py")]
    result = build_offline_rows([task], lambda _: events, now_ms=FROZEN_NOW_MS)

    by_view = {row.feature_view: row for row in result.rows}
    assert by_view[STUCK_OFFLINE_VIEW.name].feature_values == pytest.approx(
        extract_stuck_features_from_data(task, events, as_of_ms=T0_MS)
    )
    assert by_view[DURATION_OFFLINE_VIEW.name].feature_values == pytest.approx(
        extract_duration_features_from_data(task, events, as_of_ms=T0_MS)
    )


def test_events_after_the_reference_time_are_excluded():
    """The window is the reference time's, not the writer's."""
    task = _task("t1", completed_at=T0_MS)
    events = [_edit(T0_MS - 1000), _edit(T0_MS + HOUR_MS), _edit(FROZEN_NOW_MS)]
    result = build_offline_rows([task], lambda _: events, now_ms=FROZEN_NOW_MS)

    duration_row = next(r for r in result.rows if r.feature_view == DURATION_OFFLINE_VIEW.name)
    assert duration_row.feature_values["total_edits"] == 1.0


def test_materialize_tasks_is_tenant_scoped():
    tasks = {
        "acme": [_task("acme-1", completed_at=T0_MS)],
        "globex": [_task("globex-1", completed_at=T0_MS), _task("globex-2", completed_at=T0_MS)],
    }
    store = FakeDataStore(tasks_per_tenant=tasks)
    sink = RecordingSink()

    result = materialize_tasks(store, sink, tenant_id="acme", now_ms=FROZEN_NOW_MS)

    assert {row.entity_id for row in sink.rows} == {"acme-1"}
    assert result.tenant_id == "acme"
    assert result.written == len(result.rows) == len(OFFLINE_FEATURE_VIEWS)


def test_materialize_tasks_defaults_the_write_clock_to_now():
    store = FakeDataStore(tasks_per_tenant={"acme": [_task("t1", completed_at=T0_MS)]})
    sink = RecordingSink()
    before_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    materialize_tasks(store, sink, tenant_id="acme")

    after_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    for row in sink.rows:
        assert row.event_timestamp_ms == T0_MS
        assert before_ms <= row.created_at_ms <= after_ms


def test_counts_by_view_reports_both_views():
    result = build_offline_rows(
        [_task("t1", completed_at=T0_MS), _task("t2", completed_at=T0_MS)],
        lambda _: [],
        now_ms=FROZEN_NOW_MS,
    )
    assert result.counts_by_view() == {
        STUCK_OFFLINE_VIEW.name: 2,
        DURATION_OFFLINE_VIEW.name: 2,
    }


def test_materialization_is_not_reachable_from_a_training_run():
    """T014.4: a training run retrieves; it does not materialize."""
    source = inspect.getsource(cloud_trainer_module)
    for symbol in ("materialize_tasks", "build_offline_rows", "PostgresOfflineSink"):
        assert symbol not in source, f"cloud_trainer references {symbol}; materialization must stay a scheduled job"


# ===========================================================================
# T015 -- the PostgreSQLSource over ml_features
# ===========================================================================


def test_offline_views_mirror_the_model_feature_names_in_order():
    """Assert against .features -- .schema is list(set(...)) and loses order."""
    assert list(STUCK_OFFLINE_VIEW.feature_names) == STUCK_FEATURE_NAMES
    assert list(DURATION_OFFLINE_VIEW.feature_names) == DURATION_FEATURE_NAMES

    views = {view.name: view for view in cloud_feature_views(source_factory=lambda v: None)}
    assert [f.name for f in views["stuck_features"].features] == STUCK_FEATURE_NAMES
    assert [f.name for f in views["duration_features"].features] == DURATION_FEATURE_NAMES


def test_source_query_projects_ml_features_with_the_right_timestamp():
    query = feature_view_source_query(STUCK_OFFLINE_VIEW)
    assert f"FROM {ML_FEATURES_TABLE}" in query
    assert "entity_id AS task_id" in query
    assert "event_timestamp" in query
    assert "WHERE entity_type = 'task'" in query
    assert "AND feature_view = 'stuck_features'" in query
    for name in STUCK_FEATURE_NAMES:
        assert f"(feature_values ->> '{name}')::double precision AS {name}" in query
    # Selecting the right row for a moment is the as-of join's job.
    assert "ORDER BY" not in query.upper()
    assert "created_at <" not in query and "created_at >" not in query


def test_sql_projection_and_python_projection_share_one_column_list():
    for view in OFFLINE_FEATURE_VIEWS.values():
        columns = source_query_columns(view)
        frame = rows_to_source_frame([], view)
        assert list(frame.columns) == list(columns)
        query = feature_view_source_query(view)
        for column in columns:
            assert column in query


def test_postgres_source_uses_event_timestamp():
    pytest.importorskip("psycopg")
    source = materialize.postgres_source(STUCK_OFFLINE_VIEW)
    assert source.timestamp_field == "event_timestamp"
    assert source.created_timestamp_column == "created_at"
    assert ML_FEATURES_TABLE in source.get_table_query_string()


def _executable_source(module) -> str:
    """Return a module's source with comments and string literals removed.

    These assertions are about what the code *does*, not what its prose is
    allowed to discuss. Scanning raw source would forbid the docstrings from
    naming the very failure modes they exist to warn about.
    """
    import io
    import tokenize

    kept: list[str] = []
    readline = io.StringIO(inspect.getsource(module)).readline
    for token in tokenize.generate_tokens(readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept)


def test_no_placeholder_or_nonexistent_source_paths():
    """The failure the April 2026 branch shipped: sources pointing at nothing."""
    code = _executable_source(materialize)
    for smell in ("parquet", "FileSource", "placeholder", "TODO"):
        assert smell not in code, f"{smell!r} suggests a stand-in source"

    pytest.importorskip("psycopg")
    from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
        PostgreSQLSource,
    )

    for view in cloud_feature_views():
        assert isinstance(view.batch_source, PostgreSQLSource)
        assert ML_FEATURES_TABLE in view.batch_source.get_table_query_string()


def test_shipped_definitions_stay_deployment_neutral():
    """FR-001/FR-005: the open-source registry names no table and no host."""
    code = _executable_source(definitions)
    for smell in (ML_FEATURES_TABLE, "postgres", "PostgreSQL", "SELECT", "psycopg"):
        assert smell not in code

    # Binding a cloud source must not mutate the shipped objects.
    cloud_feature_views(source_factory=lambda v: None)
    assert definitions.stuck_features.batch_source is None
    assert definitions.duration_features.batch_source is None


def test_materialize_holds_no_connection_details():
    """Connection config is reused from feature_store.config, never duplicated."""
    code = _executable_source(materialize)
    for smell in ("SIGIL_POSTGRES_URL", "environ", "getenv", "5432", "password", "psycopg"):
        assert smell not in code
    # The connection is injected; this module never opens one of its own.
    assert "connect (" not in code
    assert "load_cloud_repo_config" in code


def test_cloud_feature_services_mirror_the_shipped_contract():
    views = cloud_feature_views(source_factory=lambda v: None)
    services = cloud_feature_services(views)
    assert {s.name for s in services} == set(definitions.FEATURE_SERVICES)
    for service in services:
        shipped = definitions.FEATURE_SERVICES[service.name]
        assert [p.name for p in service.feature_view_projections] == [p.name for p in shipped.feature_view_projections]


def test_feature_service_version_tracks_the_contract():
    """FR-010: the version moves when, and only when, the contract moves."""
    from feast import FeatureService

    views = {v.name: v for v in cloud_feature_views(source_factory=lambda v: None)}
    services = {s.name: s for s in cloud_feature_services(list(views.values()))}

    baseline = feature_service_version(services["stuck"])
    assert baseline == feature_service_version(services["stuck"]), "must be deterministic"
    assert baseline != feature_service_version(services["duration"])

    renamed = FeatureService(name="stuck-v2", features=[views["stuck_features"]])
    assert feature_service_version(renamed) != baseline


# ===========================================================================
# T018 -- point-in-time and leakage
# ===========================================================================


def test_leakage_a_later_value_does_not_reach_an_earlier_example(tmp_path):
    """The reason this work package exists (FR-008).

    Two values for one entity, an hour apart. Retrieved as of one minute after
    the first, the second must be invisible.
    """
    rows = [
        _row("t1", T0, {"edit_velocity": 1.0}),
        _row("t1", T0 + timedelta(hours=1), {"edit_velocity": 9.0}),
    ]
    store = build_offline_feature_store(tmp_path, rows)

    frame = store.get_historical_features(
        entity_df=_entity_frame([("t1", T0 + timedelta(minutes=1))]),
        features=store.get_feature_service("stuck"),
    ).to_df()

    assert len(frame) == 1
    assert frame.iloc[0]["edit_velocity"] == 1.0
    assert frame.iloc[0]["edit_velocity"] != 9.0


def test_rows_with_different_timestamps_receive_different_values(tmp_path):
    """A single shared value across rows would mean no as-of join is happening."""
    rows = [
        _row("t1", T0, {"edit_velocity": 1.0}),
        _row("t1", T0 + timedelta(hours=1), {"edit_velocity": 9.0}),
    ]
    store = build_offline_feature_store(tmp_path, rows)

    frame = store.get_historical_features(
        entity_df=_entity_frame([("t1", T0 + timedelta(minutes=1)), ("t1", T0 + timedelta(hours=2))]),
        features=store.get_feature_service("stuck"),
    ).to_df()

    values = list(frame.sort_values("event_timestamp")["edit_velocity"])
    assert values == [1.0, 9.0]
    assert values[0] != values[1]


def test_a_value_older_than_the_ttl_is_not_returned(tmp_path):
    """TTL bounds how far back the as-of window reaches."""
    assert timedelta(days=7) == definitions.STUCK_TTL
    rows = [_row("t1", T0, {"edit_velocity": 4.0})]
    store = build_offline_feature_store(tmp_path, rows)
    service = store.get_feature_service("stuck")

    within = T0 + definitions.STUCK_TTL - timedelta(hours=1)
    beyond = T0 + definitions.STUCK_TTL + timedelta(hours=1)

    inside = store.get_historical_features(entity_df=_entity_frame([("t1", within)]), features=service).to_df()
    assert len(inside) == 1
    assert inside.iloc[0]["edit_velocity"] == 4.0

    # Past the TTL the example is dropped from the result entirely, rather than
    # returned carrying a stale value. Either shape would satisfy "not served";
    # this one is stricter, and the trainer's row-count check turns it into the
    # loud failure of FR-017 rather than a quietly shorter training set.
    outside = store.get_historical_features(entity_df=_entity_frame([("t1", beyond)]), features=service).to_df()
    assert len(outside) == 0, "a value past its TTL must not be served"

    combined = store.get_historical_features(
        entity_df=_entity_frame([("t1", within), ("t1", beyond)]), features=service
    ).to_df()
    assert list(combined["event_timestamp"]) == [pd.Timestamp(within)]


def test_a_ttl_expired_example_fails_the_training_run(tmp_path):
    """A row the TTL dropped must abort the run, not shorten it."""
    task = _task("t1", completed_at=T0_MS)
    rows = build_offline_rows([task], lambda _: [], now_ms=FROZEN_NOW_MS).rows
    store = build_offline_feature_store(tmp_path, rows)

    stale = dict(task)
    stale["completed_at"] = T0_MS + int(definitions.DURATION_TTL.total_seconds() * 1000) + HOUR_MS
    trainer = CloudTrainer(
        data_store=FakeDataStore(),
        model_store=FakeModelStore(),
        feature_store=store,
    )
    with pytest.raises(PartialRetrievalError):
        trainer._retrieve_offline_features([stale], "acme")


def test_retrieved_values_equal_direct_extraction(tmp_path):
    """Closes the loop: retrieval and replay agree number for number."""
    task = _task("t1", completed_at=T0_MS, test_fails=3)
    events = [_edit(T0_MS - 120_000, "a.py"), _edit(T0_MS - 60_000, "b.py")]
    result = build_offline_rows([task], lambda _: events, now_ms=FROZEN_NOW_MS)
    store = build_offline_feature_store(tmp_path, result.rows)

    frame = store.get_historical_features(
        entity_df=_entity_frame([("t1", T0)]),
        features=store.get_feature_service("stuck"),
    ).to_df()

    direct = extract_stuck_features_from_data(task, events, as_of_ms=T0_MS)
    retrieved = frame.iloc[0]
    for name in STUCK_FEATURE_NAMES:
        assert retrieved[name] == pytest.approx(direct[name]), name


def test_materialized_history_does_not_leak_into_an_earlier_example(tmp_path):
    """The same property, driven end to end by the real materializer."""
    early = _task("t1", last_active=T0_MS, started_at=T0_MS - 60_000)
    late = _task("t1", last_active=T0_MS + HOUR_MS, started_at=T0_MS - 60_000)
    early_events = [_edit(T0_MS - 30_000)]
    late_events = [_edit(T0_MS - 30_000)] + [_edit(T0_MS + i * 1000) for i in range(20)]

    rows = build_offline_rows([early], lambda _: early_events, now_ms=FROZEN_NOW_MS).rows
    rows += build_offline_rows([late], lambda _: late_events, now_ms=FROZEN_NOW_MS).rows
    store = build_offline_feature_store(tmp_path, rows)

    frame = store.get_historical_features(
        entity_df=_entity_frame([("t1", T0 + timedelta(minutes=1))]),
        features=store.get_feature_service("stuck"),
    ).to_df()

    expected = extract_stuck_features_from_data(early, early_events, as_of_ms=T0_MS)
    later = extract_stuck_features_from_data(late, late_events, as_of_ms=T0_MS + HOUR_MS)
    assert frame.iloc[0]["edit_velocity"] == pytest.approx(expected["edit_velocity"])
    assert expected["edit_velocity"] != later["edit_velocity"], "fixture must actually differ"
    assert frame.iloc[0]["edit_velocity"] != pytest.approx(later["edit_velocity"])


def test_empty_offline_store_yields_nothing(tmp_path):
    """An empty store must look empty, not like a store of zeroes."""
    store = build_offline_feature_store(tmp_path, [])
    frame = store.get_historical_features(
        entity_df=_entity_frame([("t1", T0)]),
        features=store.get_feature_service("stuck"),
    ).to_df()
    assert len(frame) == 1
    assert pd.isna(frame.iloc[0]["edit_velocity"])


# ===========================================================================
# T017 -- cloud training via point-in-time retrieval
# ===========================================================================


def _training_fixture(n: int = 12) -> tuple[FakeDataStore, list[dict], dict[str, list[dict]]]:
    """A tenant with enough completed tasks to clear the real-data threshold."""
    tasks: list[dict] = []
    events: dict[str, list[dict]] = {}
    for i in range(n):
        completed = T0_MS + i * HOUR_MS
        task = _task(
            f"task-{i}",
            completed_at=completed,
            started_at=completed - 3_600_000,
            # Half the tasks carry the stuck label so both classes are present.
            test_fails=5 if i % 2 == 0 else 0,
            branch=f"feat/branch-{i}",
        )
        tasks.append(task)
        events[task["id"]] = [_edit(completed - 60_000 * (j + 1), f"f{j}.py") for j in range(i % 4 + 1)]
    store = FakeDataStore(tasks_per_tenant={"acme": tasks}, events_per_task=events)
    return store, tasks, events


def _materialized_store(tmp_path, tasks, events) -> Any:
    rows = build_offline_rows(tasks, lambda tid: events[tid], now_ms=FROZEN_NOW_MS).rows
    return build_offline_feature_store(tmp_path, rows)


def test_training_retrieves_from_the_offline_store(tmp_path):
    data_store, tasks, events = _training_fixture()
    trainer = CloudTrainer(
        data_store=data_store,
        model_store=FakeModelStore(),
        config=CloudTrainingConfig(min_interval_sec=0),
        feature_store=_materialized_store(tmp_path, tasks, events),
    )

    run = trainer.train_tenant("acme")

    assert run.status == "trained", run.error
    assert "stuck" in run.models_trained
    assert "duration" in run.models_trained


def test_training_records_the_feature_service_it_consumed(tmp_path):
    """FR-010."""
    data_store, tasks, events = _training_fixture()
    trainer = CloudTrainer(
        data_store=data_store,
        model_store=FakeModelStore(),
        config=CloudTrainingConfig(min_interval_sec=0),
        feature_store=_materialized_store(tmp_path, tasks, events),
    )

    trainer.train_tenant("acme")

    by_model = {record.model: record for record in trainer.last_feature_sets}
    assert set(by_model) == {"stuck", "duration"}
    assert by_model["stuck"].feature_service == "stuck"
    assert by_model["stuck"].feature_names == tuple(STUCK_FEATURE_NAMES)
    assert by_model["stuck"].rows == len(tasks)
    assert len(by_model["stuck"].version) == 16
    assert by_model["stuck"].version != by_model["duration"].version

    audits = [e for e in data_store.ml_events if e["kind"] == "feature_retrieval"]
    assert {e["endpoint"] for e in audits} == {
        f"feature_service:stuck@{by_model['stuck'].version}",
        f"feature_service:duration@{by_model['duration'].version}",
    }
    assert {e["routing"] for e in audits} == {"acme"}


def test_retrieval_and_replay_produce_identical_training_sets(tmp_path, monkeypatch):
    """This package changes where features come from, not what is learned."""
    captured: dict[str, tuple] = {}

    def _capture(name):
        def train(self, X, y, **kwargs):
            captured[name] = (X.tolist(), y.tolist())

        return train

    from sigil_ml.models.duration import DurationEstimator
    from sigil_ml.models.stuck import StuckPredictor

    data_store, tasks, events = _training_fixture()
    feature_store = _materialized_store(tmp_path, tasks, events)

    results: dict[str, dict[str, tuple]] = {}
    for label, injected in (("replay", None), ("retrieval", feature_store)):
        captured = {}
        monkeypatch.setattr(StuckPredictor, "train", _capture("stuck"))
        monkeypatch.setattr(DurationEstimator, "train", _capture("duration"))
        trainer = CloudTrainer(
            data_store=FakeDataStore(tasks_per_tenant={"acme": tasks}, events_per_task=events),
            model_store=FakeModelStore(),
            config=CloudTrainingConfig(min_interval_sec=0),
            feature_store=injected,
        )
        trainer.train_tenant("acme")
        results[label] = dict(captured)

    assert set(results["replay"]) == {"stuck", "duration"}
    for model in ("stuck", "duration"):
        replay_X, replay_y = results["replay"][model]
        retrieved_X, retrieved_y = results["retrieval"][model]
        assert len(retrieved_X) == len(replay_X), f"{model} example count differs"
        for position, (retrieved_row, replay_row) in enumerate(zip(retrieved_X, replay_X)):
            assert retrieved_row == pytest.approx(replay_row), f"{model} row {position} differs"
        assert retrieved_y == replay_y, f"{model} labels differ -- the label expression changed"


def test_training_fails_loudly_when_the_offline_store_is_unavailable(tmp_path):
    """FR-017: never a partial training set, never a quiet success."""

    class BrokenStore:
        def get_feature_service(self, name):
            raise ConnectionError("offline store is down")

    data_store, tasks, _ = _training_fixture()
    model_store = FakeModelStore()
    trainer = CloudTrainer(
        data_store=data_store,
        model_store=model_store,
        config=CloudTrainingConfig(min_interval_sec=0),
        feature_store=BrokenStore(),
    )

    run = trainer.train_tenant("acme")

    assert run.status == "failed"
    assert "offline store is down" in (run.error or "")
    assert run.models_trained == []
    assert model_store.saved == {}, "nothing may be trained from an unavailable store"


def test_training_refuses_a_short_retrieval(tmp_path):
    """A training set shorter than the entity frame is an error, not a result."""

    class ShortStore:
        def __init__(self, real):
            self._real = real

        def get_feature_service(self, name):
            return self._real.get_feature_service(name)

        def get_historical_features(self, *, entity_df, features):
            job = self._real.get_historical_features(entity_df=entity_df, features=features)

            class Truncated:
                def to_df(self_inner):
                    return job.to_df().iloc[:-1]

            return Truncated()

    data_store, tasks, events = _training_fixture()
    model_store = FakeModelStore()
    trainer = CloudTrainer(
        data_store=data_store,
        model_store=model_store,
        config=CloudTrainingConfig(min_interval_sec=0),
        feature_store=ShortStore(_materialized_store(tmp_path, tasks, events)),
    )

    run = trainer.train_tenant("acme")

    assert run.status == "failed"
    assert "row(s) for" in (run.error or "")
    assert model_store.saved == {}


def test_training_refuses_an_empty_offline_store(tmp_path):
    """The empty store trains nothing rather than a model of zeroes."""
    data_store, tasks, _ = _training_fixture()
    model_store = FakeModelStore()
    trainer = CloudTrainer(
        data_store=data_store,
        model_store=model_store,
        config=CloudTrainingConfig(min_interval_sec=0),
        feature_store=build_offline_feature_store(tmp_path, []),
    )

    run = trainer.train_tenant("acme")

    assert run.status == "failed"
    assert run.models_trained == []
    assert model_store.saved == {}


def test_entity_frame_carries_one_row_per_example_with_its_own_moment():
    data_store, tasks, events = _training_fixture(4)
    trainer = CloudTrainer(
        data_store=data_store,
        model_store=FakeModelStore(),
        feature_store=object(),
    )

    frame = trainer._build_entity_frame(tasks, "acme")

    assert list(frame.columns) == ["task_id", "event_timestamp"]
    assert len(frame) == len(tasks)
    assert list(frame["task_id"]) == [t["id"] for t in tasks]
    for task, moment in zip(tasks, frame["event_timestamp"]):
        assert int(moment.timestamp() * 1000) == _reference_time_for(task)
    assert frame["event_timestamp"].nunique() == len(tasks), "each example must carry its own moment, not a shared one"


def test_entity_frame_skips_unresolvable_tasks_and_logs_once(caplog):
    data_store, tasks, _ = _training_fixture(4)
    for task in tasks[:2]:
        task.pop("completed_at")
        task.pop("last_active", None)

    trainer = CloudTrainer(data_store=data_store, model_store=FakeModelStore(), feature_store=object())
    with caplog.at_level(logging.INFO, logger="sigil_ml.training.cloud_trainer"):
        frame = trainer._build_entity_frame(tasks, "acme")

    assert len(frame) == 2
    skips = [r for r in caplog.records if "no resolvable reference time" in r.getMessage()]
    assert len(skips) == 1


def test_entity_frame_refuses_to_be_empty():
    tasks = [_task("t1"), _task("t2")]
    for task in tasks:
        task.pop("completed_at", None)
        task.pop("last_active", None)

    trainer = CloudTrainer(data_store=FakeDataStore(), model_store=FakeModelStore(), feature_store=object())
    with pytest.raises(OfflineStoreUnavailableError):
        trainer._build_entity_frame(tasks, "acme")


def test_tenant_scoping_survives_retrieval(tmp_path):
    """A tenant's run retrieves that tenant's task ids and no others."""
    _, acme_tasks, acme_events = _training_fixture()
    other = _task("globex-1", completed_at=T0_MS)
    data_store = FakeDataStore(
        tasks_per_tenant={"acme": acme_tasks, "globex": [other]},
        events_per_task={**acme_events, "globex-1": []},
    )

    seen: list[list[str]] = []
    real = _materialized_store(tmp_path, acme_tasks, acme_events)

    class Spy:
        def get_feature_service(self, name):
            return real.get_feature_service(name)

        def get_historical_features(self, *, entity_df, features):
            seen.append(list(entity_df["task_id"]))
            return real.get_historical_features(entity_df=entity_df, features=features)

    trainer = CloudTrainer(
        data_store=data_store,
        model_store=FakeModelStore(),
        config=CloudTrainingConfig(min_interval_sec=0),
        feature_store=Spy(),
    )
    trainer.train_tenant("acme")

    assert seen, "retrieval never happened"
    for requested in seen:
        assert requested == [t["id"] for t in acme_tasks]
        assert "globex-1" not in requested


def test_without_a_feature_store_behaviour_is_unchanged(tmp_path):
    """The replay path is untouched for callers that inject nothing."""
    data_store, _, _ = _training_fixture()
    trainer = CloudTrainer(
        data_store=data_store,
        model_store=FakeModelStore(),
        config=CloudTrainingConfig(min_interval_sec=0),
    )

    run = trainer.train_tenant("acme")

    assert run.status == "trained"
    assert trainer.last_feature_sets == []
    assert [e for e in data_store.ml_events if e["kind"] == "feature_retrieval"] == []


def test_retrieval_errors_are_one_family():
    assert issubclass(OfflineStoreUnavailableError, OfflineRetrievalError)
    assert issubclass(PartialRetrievalError, OfflineRetrievalError)


def test_materialization_result_defaults():
    empty = MaterializationResult()
    assert empty.rows == [] and empty.skipped == 0 and empty.written == 0


# ===========================================================================
# The same scenario over a real PostgreSQL, when one is available
# ===========================================================================

_LIVE_PG = os.environ.get("SIGIL_ML_TEST_POSTGRES_URL")


@pytest.mark.skipif(not _LIVE_PG, reason="SIGIL_ML_TEST_POSTGRES_URL is not set")
def test_live_postgres_point_in_time(tmp_path):
    """The leakage scenario through PostgreSQLSource against a live server.

    Skipped unless a server is provided. Everything above proves the write side
    and Feast's as-of join; this proves that PostgreSQL executes
    ``feature_view_source_query`` as written.
    """
    import psycopg2
    from feast import FeatureStore
    from feast.repo_config import RepoConfig

    from sigil_ml.feature_store.config import _postgres_connection

    connection = psycopg2.connect(_LIVE_PG)
    sink = materialize.PostgresOfflineSink(connection)
    sink.create_schema()
    sink.write(
        [
            _row("pit-1", T0, {"edit_velocity": 1.0}),
            _row("pit-1", T0 + timedelta(hours=1), {"edit_velocity": 9.0}),
        ]
    )

    settings = _postgres_connection()
    repo = tmp_path / "pg_repo"
    repo.mkdir(parents=True, exist_ok=True)
    store = FeatureStore(
        config=RepoConfig(
            project="sigil_ml",
            provider="local",
            repo_path=str(repo),
            registry={"registry_type": "file", "path": str(repo / "registry.db")},
            online_store={"type": "postgres", **settings},
            offline_store={"type": "postgres", **settings},
            entity_key_serialization_version=3,
        )
    )
    materialize.apply_cloud_definitions(store)

    frame = store.get_historical_features(
        entity_df=_entity_frame([("pit-1", T0 + timedelta(minutes=1))]),
        features=store.get_feature_service("stuck"),
    ).to_df()

    assert frame.iloc[0]["edit_velocity"] == 1.0
