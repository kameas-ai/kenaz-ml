"""Feast entities, feature views, and feature services for ``kenaz_ml``.

This module is the single declaration of *what features exist* consulted by both
the local and the cloud deployment. It is deliberately inert: it declares shape
(entity, field names, ordering, type, TTL) and nothing else.

**No feature arithmetic lives here** (plan.md D-002). ``kenaz_ml.features``
remains the sole authority for computing feature values; Feast only registers,
stores, materializes, and retrieves them. If a calculation ever appears in this
file it is in the wrong module.

**Field lists are derived, never retyped** (FR-001, FR-002). Every feature view's
schema is built by iterating the ``FEATURE_NAMES`` constant that already lives
beside its model. Both trainers build vectors positionally as
``[features[f] for f in FEATURE_NAMES]``, so a name list that merely *looks*
right but is ordered differently would silently permute every model input with
no error raised anywhere. Retyping the names here would create exactly the
second source of truth this migration exists to remove.

**Sources are not bound here.** The shipped definitions carry no data source, so
nothing cloud-only (host, URL, port, credential) can leak into the registry that
open-source users receive. Each deployment binds its own source by calling the
``*_feature_view()`` factories with the source it wants (local: push; cloud: the
``ml_features`` PostgreSQL table) — wired up in WP02/WP04.

A note for anyone writing assertions against these objects: use
``FeatureView.features``, which preserves declaration order. ``FeatureView.schema``
is implemented as ``list(set(...))`` and returns fields in arbitrary order.
"""

from __future__ import annotations

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field
from feast.data_source import DataSource
from feast.types import Float64, String
from feast.value_type import ValueType

from kenaz_ml.models.duration import FEATURE_NAMES as DURATION_FEATURE_NAMES
from kenaz_ml.models.stuck import FEATURE_NAMES as STUCK_FEATURE_NAMES

# --------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------
# Join keys match the identifiers the existing tables and store methods already
# use (`tasks`/`events` are keyed by task id; the fleet layer is keyed by node;
# the cloud store partitions by tenant). No new names are invented here.

task = Entity(
    name="task",
    join_keys=["task_id"],
    value_type=ValueType.STRING,
    description="A unit of developer work. Primary entity for the stuck and duration models.",
)

node = Entity(
    name="node",
    join_keys=["node_id"],
    value_type=ValueType.STRING,
    description="A single machine reporting into the fleet layer.",
)

# Cloud-only. The local deployment is single-user and has no tenant concept, so
# `tenant` must never become a join key of a feature view the local registry has
# to resolve — that would make a locally-served prediction depend on an
# identifier the local daemon cannot supply. It is declared here so the cloud
# deployment does not have to fork this file; see LOCAL_FEATURE_VIEWS below,
# whose test asserts the separation holds.
tenant = Entity(
    name="tenant",
    join_keys=["tenant_id"],
    value_type=ValueType.STRING,
    description="Cloud-only isolation boundary. Not used by any locally-resolvable feature view.",
)

ENTITIES: list[Entity] = [task, node, tenant]

# --------------------------------------------------------------------------
# TTLs
# --------------------------------------------------------------------------
# A TTL bounds how far back a point-in-time lookup may reach: a stored row with
# event timestamp T can answer a lookup at time E only while T <= E < T + ttl.
# It therefore trades two failure modes against each other — too long serves a
# semantically stale vector, too short silently yields empty features for
# training rows whose materialization run lagged.

# Stuck features describe an in-flight coding session (`time_in_phase_sec`,
# `edit_velocity`, `file_switch_rate`); they decay fast and a vector from a
# previous sprint says nothing useful about the current one. Seven days is the
# shortest window that still spans a full work week plus a weekend gap between a
# task's last activity and a scheduled materialization run.
STUCK_TTL = timedelta(days=7)

# Duration features are per-task structural counts (`file_count`, `total_edits`,
# `branch_name_length`) that are fixed once the task is done and do not decay the
# way session-volatility signals do. Thirty days keeps historical training rows
# resolvable across a monthly retraining cycle without any staleness risk that a
# shorter window would actually remove.
DURATION_TTL = timedelta(days=30)


def _schema(entity: Entity, feature_names: list[str]) -> list[Field]:
    """Build a Feast schema from an entity and an ordered feature-name list.

    The join-key column is declared first so Feast recognises it as an entity
    column rather than a feature; it is stripped out of ``FeatureView.features``,
    which consequently equals ``feature_names`` exactly, in order.

    Every feature is ``Float64`` — the extractors in ``kenaz_ml.features`` return
    Python floats uniformly, and both estimators feed a float ndarray to
    scikit-learn.
    """
    return [Field(name=entity.join_key, dtype=String)] + [Field(name=name, dtype=Float64) for name in feature_names]


# --------------------------------------------------------------------------
# Feature views
# --------------------------------------------------------------------------


def stuck_feature_view(source: DataSource | None = None) -> FeatureView:
    """Features consumed by the ``stuck`` model, in ``models/stuck.py`` order."""
    return FeatureView(
        name="stuck_features",
        entities=[task],
        ttl=STUCK_TTL,
        schema=_schema(task, STUCK_FEATURE_NAMES),
        source=source,
        online=True,
        description="Session-volatility signals for the stuck predictor.",
    )


def duration_feature_view(source: DataSource | None = None) -> FeatureView:
    """Features consumed by the ``duration`` model, in ``models/duration.py`` order."""
    return FeatureView(
        name="duration_features",
        entities=[task],
        ttl=DURATION_TTL,
        schema=_schema(task, DURATION_FEATURE_NAMES),
        source=source,
        online=True,
        description="Per-task structural counts for the duration estimator.",
    )


stuck_features = stuck_feature_view()
duration_features = duration_feature_view()

FEATURE_VIEWS: list[FeatureView] = [stuck_features, duration_features]

# Every view above is resolvable by the local deployment; none of them is keyed
# by `tenant`. A view that needs the tenant entity belongs outside this list.
LOCAL_FEATURE_VIEWS: list[FeatureView] = [stuck_features, duration_features]

# --------------------------------------------------------------------------
# Registration coverage
# --------------------------------------------------------------------------
# Maps the model module that declares a `FEATURE_NAMES` constant to the feature
# view registering it. A model whose feature list is declared as a constant but
# absent from this mapping would ship unregistered; the conformance test walks
# `kenaz_ml.models` and fails if that ever happens.
#
# The other model families (`workflow`, `activity`, `quality`, and the `fleet_*`
# models) declare no `FEATURE_NAMES` constant: workflow and activity derive their
# vector at predict time from `sorted(features.keys())`, quality is a rules-based
# scorer reading keys ad hoc, and the fleet models take fixed positional rows.
# There is no constant to derive a schema from, so registering them would mean
# retyping names — the duplication FR-001 forbids. They are registered once a
# `FEATURE_NAMES` constant exists to derive from, and the coverage test below
# fires the moment one appears.
REGISTERED_FEATURE_NAMES: dict[str, list[str]] = {
    "stuck": STUCK_FEATURE_NAMES,
    "duration": DURATION_FEATURE_NAMES,
}

# --------------------------------------------------------------------------
# Feature services
# --------------------------------------------------------------------------
# One service per model: the versioned contract naming exactly what that model
# consumes, and the thing a training run records (FR-010). Service names are the
# model names Go already queries in `ml_predictions.model` — fixed by CLAUDE.md
# and not renameable here. Keep them stable: changing a name silently changes the
# meaning of every training record that referenced the old one.

stuck_service = FeatureService(
    name="stuck",
    features=[stuck_features],
    description="Feature contract for the stuck predictor.",
)

duration_service = FeatureService(
    name="duration",
    features=[duration_features],
    description="Feature contract for the duration estimator.",
)

FEATURE_SERVICES: dict[str, FeatureService] = {
    "stuck": stuck_service,
    "duration": duration_service,
}

__all__ = [
    "DURATION_TTL",
    "ENTITIES",
    "FEATURE_SERVICES",
    "FEATURE_VIEWS",
    "LOCAL_FEATURE_VIEWS",
    "REGISTERED_FEATURE_NAMES",
    "STUCK_TTL",
    "duration_feature_view",
    "duration_features",
    "duration_service",
    "node",
    "stuck_feature_view",
    "stuck_features",
    "stuck_service",
    "task",
    "tenant",
]
