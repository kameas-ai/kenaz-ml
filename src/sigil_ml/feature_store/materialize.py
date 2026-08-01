"""The cloud offline store: ``ml_features``, its writer, and its Feast source.

This module is the half the April 2026 attempt skipped. That branch wired a
push source into the online store and pointed its offline views at parquet
paths that did not exist, so nothing it claimed about point-in-time retrieval
could be checked. Everything here is executable: the table is real, the rows
are written by real extractors, and the Feast source reads the real table.

The load-bearing column
-----------------------
``ml_features.event_timestamp`` is **the moment the feature values describe**,
not the moment the row was written. For a completed task that is its
``completed_at`` (falling back to ``last_active``), resolved by the *same*
:func:`~sigil_ml.training.trainer._reference_time_for` the training path uses —
imported, never reimplemented, because two copies of that rule would drift and
the drift would be invisible.

``created_at`` exists for auditing only. Ordering or filtering a retrieval by it
reproduces exactly the defect the ``feature-extraction-correctness`` mission
removed: a historical example measured against the wall clock describes the time
of the *write*, not the behaviour. The prohibition is stated in the module
docstring, in a ``COMMENT ON COLUMN`` that ships with the table itself, and
asserted in ``tests/test_feature_store_cloud.py``.

Shape of the table versus shape of the source
---------------------------------------------
The table stores one row per ``(entity, feature view, event timestamp)`` with the
values in a single JSONB document, which keeps a new feature from requiring a
migration. Feast, however, expects one typed column per feature. The two shapes
are bridged by :func:`feature_view_source_query`, which projects the JSONB
document into columns — and both the SQL and its in-Python mirror
(:func:`rows_to_source_frame`) take their column list from
:func:`source_query_columns`, so the projection has exactly one definition.

Deployment neutrality
---------------------
``feature_store/definitions.py`` stays sourceless (FR-001/FR-002): the shipped
registry that open-source users receive must not name a table, a host, or a
query. The cloud source is bound *here*, by calling the ``*_feature_view()``
factories with a :class:`PostgreSQLSource`. Nothing in this module is imported by
the local serving path.

Materialization is a job
------------------------
:func:`materialize_tasks` is invoked by a scheduled or triggered run, never
synchronously from a training run. A training run that materialized its own
features would be computing them, not retrieving them, and every point-in-time
guarantee below would reduce to "the extractor was called with the right
argument" — which is the guarantee this mission exists to stop relying on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from sigil_ml.feature_store import definitions
from sigil_ml.features import (
    extract_duration_features_from_data,
    extract_stuck_features_from_data,
)
from sigil_ml.models.duration import FEATURE_NAMES as DURATION_FEATURE_NAMES
from sigil_ml.models.stuck import FEATURE_NAMES as STUCK_FEATURE_NAMES

# Imported, not reimplemented. The offline store and the training path must
# resolve an example's reference time by one rule; a second copy would drift and
# the drift would only show up as a quietly worse model.
from sigil_ml.training.trainer import _reference_time_for

logger = logging.getLogger(__name__)

__all__ = [
    "DURATION_OFFLINE_VIEW",
    "ENTITY_TYPE_TASK",
    "ML_FEATURES_DDL",
    "ML_FEATURES_INSERT",
    "ML_FEATURES_TABLE",
    "OFFLINE_FEATURE_VIEWS",
    "STUCK_OFFLINE_VIEW",
    "MaterializationResult",
    "OfflineFeatureRow",
    "OfflineFeatureSink",
    "OfflineFeatureView",
    "OfflineRetrievalError",
    "OfflineStoreUnavailableError",
    "PartialRetrievalError",
    "PostgresOfflineSink",
    "apply_cloud_definitions",
    "build_offline_rows",
    "cloud_feature_services",
    "cloud_feature_views",
    "feature_service_version",
    "feature_store_for_cloud",
    "feature_view_source_query",
    "materialize_tasks",
    "postgres_source",
    "rows_to_source_frame",
    "source_query_columns",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OfflineRetrievalError(RuntimeError):
    """Base class for every way offline retrieval can refuse to produce a set.

    FR-017: feature resolution fails loudly rather than returning partial
    results. A short training set that trains without complaint is worse than an
    error, because the model it produces looks exactly like a good one.
    """


class OfflineStoreUnavailableError(OfflineRetrievalError):
    """The offline store could not be reached, or held nothing to retrieve."""


class PartialRetrievalError(OfflineRetrievalError):
    """Retrieval returned fewer rows, or emptier rows, than were requested.

    Raised on a row-count mismatch, a missing feature column, or a null value in
    a feature column. Any of the three means at least one training example would
    silently carry a default instead of its measured value.
    """


# ---------------------------------------------------------------------------
# The offline view registry
# ---------------------------------------------------------------------------

ML_FEATURES_TABLE = "ml_features"

#: The only entity type materialized today. `node` features are computed from
#: fleet rows that declare no `FEATURE_NAMES` constant, so there is no view to
#: materialize them into yet (data-model.md §Feature views).
ENTITY_TYPE_TASK = "task"

#: Extractor signature: ``(task, events, *, as_of_ms) -> {feature name: value}``.
Extractor = Callable[..., dict[str, float]]


@dataclass(frozen=True)
class OfflineFeatureView:
    """What a single feature view needs in order to be materialized.

    ``feature_names`` mirrors the model module's ``FEATURE_NAMES`` constant by
    reference rather than by retyping, so it cannot drift from the vector the
    trainer builds positionally. A test asserts it equals the corresponding
    Feast ``FeatureView.features`` — ``.features``, never ``.schema``, because
    ``FeatureView.schema`` is built as ``list(set(...))`` and returns fields in
    arbitrary order, which would make an ordering assertion pass by luck.
    """

    name: str
    entity_type: str
    join_key: str
    feature_names: tuple[str, ...]
    extractor: Extractor
    ttl: timedelta


STUCK_OFFLINE_VIEW = OfflineFeatureView(
    name="stuck_features",
    entity_type=ENTITY_TYPE_TASK,
    join_key=definitions.task.join_key,
    feature_names=tuple(STUCK_FEATURE_NAMES),
    extractor=extract_stuck_features_from_data,
    ttl=definitions.STUCK_TTL,
)

DURATION_OFFLINE_VIEW = OfflineFeatureView(
    name="duration_features",
    entity_type=ENTITY_TYPE_TASK,
    join_key=definitions.task.join_key,
    feature_names=tuple(DURATION_FEATURE_NAMES),
    extractor=extract_duration_features_from_data,
    ttl=definitions.DURATION_TTL,
)

#: Keyed by feature-view name, which is also the name in the shipped registry.
OFFLINE_FEATURE_VIEWS: dict[str, OfflineFeatureView] = {
    STUCK_OFFLINE_VIEW.name: STUCK_OFFLINE_VIEW,
    DURATION_OFFLINE_VIEW.name: DURATION_OFFLINE_VIEW,
}


# ---------------------------------------------------------------------------
# T013 — the table
# ---------------------------------------------------------------------------

#: PostgreSQL DDL for the offline store, matching data-model.md §Offline store.
#:
#: Python-owned and purely additive (C-004): no statement here touches ``events``,
#: ``tasks``, ``patterns``, ``suggestions``, ``ml_predictions`` or ``ml_events``,
#: all of which belong to Go.
#:
#: ``event_timestamp`` is ``NOT NULL`` deliberately. A row without one cannot
#: answer a point-in-time lookup at all, so making it unwritable is cheaper than
#: discovering it during a training run.
ML_FEATURES_DDL: tuple[str, ...] = (
    f"""
    CREATE TABLE IF NOT EXISTS {ML_FEATURES_TABLE} (
        entity_type      TEXT        NOT NULL,
        entity_id        TEXT        NOT NULL,
        feature_view     TEXT        NOT NULL,
        event_timestamp  TIMESTAMPTZ NOT NULL,
        created_at       TIMESTAMPTZ NOT NULL,
        feature_values   JSONB       NOT NULL,
        PRIMARY KEY (entity_type, entity_id, feature_view, event_timestamp)
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS ml_features_pit
        ON {ML_FEATURES_TABLE} (entity_type, entity_id, feature_view, event_timestamp DESC)
    """,
    # The rule travels with the table. Anyone reading the schema in psql sees it
    # without having to find this module.
    f"""
    COMMENT ON COLUMN {ML_FEATURES_TABLE}.event_timestamp IS
        'The moment the feature values describe (the training example''s reference time: '
        'completed_at, else last_active). Never the write time. Point-in-time retrieval '
        'orders by this column and only this column.'
    """,
    f"""
    COMMENT ON COLUMN {ML_FEATURES_TABLE}.created_at IS
        'Audit only: when this row was written. MUST NEVER be used for retrieval ordering '
        'or filtering. Doing so measures rows against the wall clock instead of the moment '
        'they describe, which is the exact defect the feature-extraction-correctness '
        'mission removed.'
    """,
)

#: Idempotent upsert. Re-materializing a task rewrites its values for the same
#: event timestamp rather than accumulating duplicates, and never invents a new
#: point in time — the primary key is the point in time.
ML_FEATURES_INSERT = f"""
    INSERT INTO {ML_FEATURES_TABLE}
        (entity_type, entity_id, feature_view, event_timestamp, created_at, feature_values)
    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
    ON CONFLICT (entity_type, entity_id, feature_view, event_timestamp)
    DO UPDATE SET created_at = EXCLUDED.created_at,
                  feature_values = EXCLUDED.feature_values
"""


def _to_datetime(epoch_ms: int) -> datetime:
    """Convert epoch milliseconds to a timezone-aware UTC datetime."""
    return datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)


@dataclass(frozen=True)
class OfflineFeatureRow:
    """One materialized row of ``ml_features``.

    Kept as an ordinary object rather than a tuple of SQL parameters so that the
    two timestamps can be asserted independently. T016 turns on exactly that:
    for a historical task ``event_timestamp`` must differ from ``created_at``,
    and if the writer ever collapses the two, this is where it shows.
    """

    entity_type: str
    entity_id: str
    feature_view: str
    #: Epoch ms of the moment the values describe.
    event_timestamp_ms: int
    #: Epoch ms of the write. Audit only.
    created_at_ms: int
    feature_values: dict[str, float]

    @property
    def event_timestamp(self) -> datetime:
        return _to_datetime(self.event_timestamp_ms)

    @property
    def created_at(self) -> datetime:
        return _to_datetime(self.created_at_ms)

    def as_insert_params(self) -> tuple[Any, ...]:
        """Return the positional parameters for :data:`ML_FEATURES_INSERT`."""
        return (
            self.entity_type,
            self.entity_id,
            self.feature_view,
            self.event_timestamp,
            self.created_at,
            json.dumps(self.feature_values, sort_keys=True),
        )


class OfflineFeatureSink(Protocol):
    """Where materialized rows are persisted.

    A protocol rather than a concrete class because the rows themselves are the
    thing worth asserting about; persistence is the thin, dialect-specific part.
    """

    def write(self, rows: Sequence[OfflineFeatureRow]) -> int:
        """Persist ``rows`` and return the number written."""
        ...


class PostgresOfflineSink:
    """Writes ``ml_features`` rows through a DB-API connection.

    The connection is injected rather than opened here: ``psycopg2`` is a cloud
    extra, connection settings already live in
    :mod:`sigil_ml.feature_store.config`, and a module that opens its own socket
    is a module the no-egress test cannot constrain.
    """

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def create_schema(self) -> None:
        """Create ``ml_features``, its point-in-time index, and its comments."""
        cursor = self._connection.cursor()
        try:
            for statement in ML_FEATURES_DDL:
                cursor.execute(statement)
        finally:
            cursor.close()
        self._connection.commit()

    def write(self, rows: Sequence[OfflineFeatureRow]) -> int:
        if not rows:
            return 0
        cursor = self._connection.cursor()
        try:
            cursor.executemany(ML_FEATURES_INSERT, [row.as_insert_params() for row in rows])
        finally:
            cursor.close()
        self._connection.commit()
        return len(rows)


# ---------------------------------------------------------------------------
# T014 — materialization
# ---------------------------------------------------------------------------


@dataclass
class MaterializationResult:
    """Outcome of one materialization pass."""

    rows: list[OfflineFeatureRow] = field(default_factory=list)
    #: Tasks excluded because they yielded no honest reference time.
    skipped: int = 0
    #: Tasks that contributed at least one row.
    materialized_tasks: int = 0
    tenant_id: str | None = None
    written: int = 0

    def counts_by_view(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.feature_view] = counts.get(row.feature_view, 0) + 1
        return counts


def build_offline_rows(
    tasks: Iterable[dict[str, Any]],
    events_for_task: Callable[[str], list[dict[str, Any]]],
    *,
    now_ms: int,
    views: Sequence[OfflineFeatureView] | None = None,
    tenant_id: str | None = None,
) -> MaterializationResult:
    """Compute the offline rows for ``tasks`` without persisting them.

    Args:
        tasks: Task records, already scoped to whatever tenant the caller means.
        events_for_task: Returns the events belonging to a task id.
        now_ms: The write time, in epoch ms. Injected rather than read from the
            clock so a test can freeze it far past the fixtures — a writer that
            ever stamped ``event_timestamp`` with the write time then shows up as
            a timestamp years away from the task, not as a few milliseconds of
            drift that could be mistaken for rounding.
        views: Which feature views to materialize. Defaults to all of them.
        tenant_id: Recorded on the result for logging. Tenancy is expressed by
            the caller's choice of ``tasks``; there is no tenant column, because
            adding one would let a retrieval silently span tenants whenever it
            was omitted from a predicate.

    Returns:
        A :class:`MaterializationResult` whose ``rows`` carry
        ``event_timestamp`` = the task's reference time and ``created_at`` =
        ``now_ms``.
    """
    selected = list(views) if views is not None else list(OFFLINE_FEATURE_VIEWS.values())
    result = MaterializationResult(tenant_id=tenant_id)

    for task in tasks:
        as_of_ms = _reference_time_for(task)
        if as_of_ms is None:
            # Same rule as training: an example with no honest reference time is
            # not an example. There is deliberately no wall-clock fallback.
            result.skipped += 1
            continue

        task_id = task["id"]
        events = events_for_task(task_id)
        for view in selected:
            values = view.extractor(task, events, as_of_ms=as_of_ms)
            result.rows.append(
                OfflineFeatureRow(
                    entity_type=view.entity_type,
                    entity_id=task_id,
                    feature_view=view.name,
                    event_timestamp_ms=as_of_ms,
                    created_at_ms=now_ms,
                    # Materialization stores; it does not transform. The values
                    # are whatever `sigil_ml.features` returned, coerced only to
                    # float so JSON round-trips them unchanged.
                    feature_values={name: float(values.get(name, 0.0)) for name in view.feature_names},
                )
            )
        result.materialized_tasks += 1

    if result.skipped:
        # Once, not per row: a per-row line turns a normal backfill into a log
        # flood and hides the count that actually matters.
        logger.info(
            "materialization: skipped %d task(s) with no resolvable reference time (tenant %s)",
            result.skipped,
            tenant_id,
        )

    return result


def materialize_tasks(
    data_store: Any,
    sink: OfflineFeatureSink,
    *,
    tenant_id: str,
    now_ms: int | None = None,
    views: Sequence[OfflineFeatureView] | None = None,
) -> MaterializationResult:
    """Materialize one tenant's completed tasks into the offline store.

    This is the scheduled/triggered job entry point. It is never called from a
    training run: retrieval and computation must stay separable, or the
    point-in-time guarantee degenerates into "we passed the right argument".

    Args:
        data_store: A :class:`~sigil_ml.store.DataStore`. Reads only.
        sink: Where rows are persisted.
        tenant_id: Scopes the task query. Rows for one tenant are computed from
            that tenant's tasks and no others.
        now_ms: Write time in epoch ms; defaults to the current clock.
        views: Which feature views to materialize. Defaults to all of them.
    """
    write_time = int(time.time() * 1000) if now_ms is None else now_ms
    tasks = data_store.get_completed_tasks_for_tenant(tenant_id)
    result = build_offline_rows(
        tasks,
        data_store.get_events_for_task_id,
        now_ms=write_time,
        views=views,
        tenant_id=tenant_id,
    )
    result.written = sink.write(result.rows)
    logger.info(
        "materialized %d row(s) for %d task(s) into %s (tenant %s, skipped %d)",
        result.written,
        result.materialized_tasks,
        ML_FEATURES_TABLE,
        tenant_id,
        result.skipped,
    )
    return result


# ---------------------------------------------------------------------------
# T015 — the Feast source
# ---------------------------------------------------------------------------


def source_query_columns(view: OfflineFeatureView) -> tuple[str, ...]:
    """Return the column list the offline source exposes, in order.

    One definition, consumed by both :func:`feature_view_source_query` (the SQL
    the cloud actually runs) and :func:`rows_to_source_frame` (its in-Python
    mirror). Two hand-maintained column lists would be a second source of truth
    of exactly the kind FR-001 exists to remove.
    """
    return (view.join_key, "event_timestamp", "created_at", *view.feature_names)


def feature_view_source_query(view: OfflineFeatureView) -> str:
    """Return the SELECT that presents ``ml_features`` in the shape Feast wants.

    The table stores values as one JSONB document per row; a Feast source needs
    one typed column per feature. The projection is generated from
    ``view.feature_names`` — the model module's own constant — so a renamed or
    reordered feature cannot leave a stale literal behind here.

    Note what the predicate does *not* contain: no ordering, no ``created_at``
    filter, no date bound. Selecting the right row for a given moment is the
    as-of join's job, and doing any part of it here would quietly change what
    "as of" means.
    """
    projections = ",\n        ".join(
        f"(feature_values ->> '{name}')::double precision AS {name}" for name in view.feature_names
    )
    return (
        f"SELECT\n"
        f"        entity_id AS {view.join_key},\n"
        f"        event_timestamp,\n"
        f"        created_at,\n"
        f"        {projections}\n"
        f"    FROM {ML_FEATURES_TABLE}\n"
        f"    WHERE entity_type = '{view.entity_type}'\n"
        f"      AND feature_view = '{view.name}'"
    )


def postgres_source(view: OfflineFeatureView) -> Any:
    """Return the :class:`PostgreSQLSource` over ``ml_features`` for ``view``.

    ``timestamp_field`` is ``event_timestamp`` — the moment the values describe.
    ``created_timestamp_column`` is ``created_at``, which Feast uses only to
    break ties between two rows sharing an event timestamp; it never widens or
    reorders the as-of window.

    Connection details are absent on purpose. They come from
    :func:`sigil_ml.feature_store.config.load_cloud_repo_config`, which derives
    them from the same ``SIGIL_POSTGRES_URL`` ``PostgresStore`` already uses, so
    the feature store and the data store cannot be pointed at different
    databases. A source is not a placeholder: this one names a table that the
    DDL above creates and the writer above fills.

    The import is deferred because the PostgreSQL offline store lives in Feast's
    ``contrib`` tree and needs the ``cloud`` extra; importing it eagerly would
    make this module unimportable in a local-only install.
    """
    from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
        PostgreSQLSource,
    )

    return PostgreSQLSource(
        name=f"{view.name}__{ML_FEATURES_TABLE}",
        query=feature_view_source_query(view),
        timestamp_field="event_timestamp",
        created_timestamp_column="created_at",
        description=(
            f"Point-in-time source for {view.name}, projected from {ML_FEATURES_TABLE}. "
            "Ordered by event_timestamp only; created_at is audit metadata."
        ),
    )


def cloud_feature_views(
    source_factory: Callable[[OfflineFeatureView], Any] | None = None,
) -> list[Any]:
    """Return the shipped feature views bound to their cloud offline sources.

    The shipped definitions stay sourceless so the open-source registry names no
    table and no host; binding happens here, in the cloud path.

    Args:
        source_factory: Builds the source for a view. Defaults to
            :func:`postgres_source`. Overridable so a conformance test can bind a
            source it is able to execute — never so that a placeholder can be
            substituted in production, which is why the default is the real one.
    """
    factory = source_factory or postgres_source
    return [
        definitions.stuck_feature_view(source=factory(STUCK_OFFLINE_VIEW)),
        definitions.duration_feature_view(source=factory(DURATION_OFFLINE_VIEW)),
    ]


def cloud_feature_services(views: Sequence[Any]) -> list[Any]:
    """Rebuild the shipped feature services over source-bound views.

    Names and membership are read from :data:`definitions.FEATURE_SERVICES`
    rather than retyped: a service name is the contract a training run records
    (FR-010), and a divergent literal here would silently change the meaning of
    every record that referenced it.
    """
    from feast import FeatureService

    by_name = {view.name: view for view in views}
    services = []
    for service in definitions.FEATURE_SERVICES.values():
        projected = [by_name[projection.name] for projection in service.feature_view_projections]
        services.append(
            FeatureService(
                name=service.name,
                features=projected,
                description=service.description,
            )
        )
    return services


def feature_service_version(service: Any) -> str:
    """Return a stable content version for a feature service (FR-010).

    Derived from the service name and its ordered feature references, so the
    version changes exactly when the contract does — a feature added, removed,
    renamed, or reordered — and does not change when unrelated code does. A
    training run records this alongside the service name so the feature set it
    consumed is identifiable after the fact.
    """
    references = [
        f"{projection.name}:{feature.name}"
        for projection in service.feature_view_projections
        for feature in projection.features
    ]
    payload = "|".join([service.name, *references])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def apply_cloud_definitions(store: Any, views: Sequence[Any] | None = None) -> list[Any]:
    """Register entities, source-bound views and services with a Feast store.

    Returns the applied objects so a caller can hold onto the exact services it
    registered rather than looking them up by name and hoping.
    """
    bound = list(views) if views is not None else cloud_feature_views()
    services = cloud_feature_services(bound)
    store.apply([*definitions.ENTITIES, *bound, *services])
    return [*bound, *services]


def feature_store_for_cloud() -> Any:
    """Build a Feast ``FeatureStore`` for the cloud deployment.

    Raises:
        OfflineStoreUnavailableError: If the cloud configuration cannot be built
            — no Postgres URL, wrong operating mode, missing extra. Raised as a
            retrieval error rather than propagated raw so that the caller's
            "the store is unavailable, do not train" branch catches it (FR-017).
    """
    from feast import FeatureStore

    from sigil_ml.feature_store.config import load_cloud_repo_config

    try:
        return FeatureStore(config=load_cloud_repo_config())
    except Exception as exc:  # noqa: BLE001 - re-raised with the contract's type
        raise OfflineStoreUnavailableError(
            f"The cloud feature store could not be opened: {exc}. Training must not fall back "
            "to computing features itself — that would produce a model that looks trained but "
            "was never retrieved point-in-time (FR-017)."
        ) from exc


# ---------------------------------------------------------------------------
# Verification projection
# ---------------------------------------------------------------------------


def rows_to_source_frame(rows: Sequence[OfflineFeatureRow], view: OfflineFeatureView) -> Any:
    """Project materialized rows into the frame :func:`feature_view_source_query` yields.

    This is the in-Python mirror of the SQL projection, sharing its column list
    via :func:`source_query_columns`. It exists so the point-in-time contract of
    ``ml_features`` can be executed — and therefore asserted — against a Feast
    offline store on a machine with no PostgreSQL server: the rows are the real
    writer's output and the join is Feast's real as-of join, with only the
    physical storage differing.

    Rows for other feature views are dropped, exactly as the SQL's
    ``feature_view = ...`` predicate drops them.
    """
    import pandas as pd

    columns = source_query_columns(view)
    records = [
        {
            view.join_key: row.entity_id,
            "event_timestamp": row.event_timestamp,
            "created_at": row.created_at,
            **{name: row.feature_values.get(name) for name in view.feature_names},
        }
        for row in rows
        if row.feature_view == view.name and row.entity_type == view.entity_type
    ]
    frame = pd.DataFrame.from_records(records, columns=list(columns))
    # Dtypes are pinned rather than inferred so an empty result keeps the same
    # schema as a populated one. An empty offline store must look like an empty
    # store, not like a table with different column types.
    frame[view.join_key] = frame[view.join_key].astype("object")
    for column in ("event_timestamp", "created_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    for name in view.feature_names:
        frame[name] = frame[name].astype("float64")
    return frame
