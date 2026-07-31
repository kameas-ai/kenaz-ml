"""Conformance tests: Feast definitions vs. the ``FEATURE_NAMES`` constants.

The claim under test is that ``feature_store/definitions.py`` is not a second
source of truth. Both trainers build vectors positionally as
``[features[f] for f in FEATURE_NAMES]``, so a feature view whose fields carry
the right names in the wrong order permutes every model input and raises
nothing — no exception, no warning, just quietly wrong predictions. These tests
are the only mechanism that makes that failure loud.

Two things matter about how they are written:

1. **Comparisons are ordered** (``list == list``). A set comparison would pass
   on precisely the reordering being guarded against, making the whole file
   decorative.
2. **They read ``FeatureView.features``, not ``FeatureView.schema``.** Feast
   implements ``schema`` as ``list(set(entity_columns + features))``, so it
   returns fields in arbitrary, run-to-run-unstable order. Asserting ordering
   against ``schema`` would produce a test that fails at random and, worse,
   could pass while the real ordering was wrong.
"""

import importlib
import inspect
import pkgutil
from datetime import timedelta

import pytest
from feast import FeatureView
from feast.infra.offline_stores.file_source import FileSource
from feast.types import Float64, String

import sigil_ml.models
from sigil_ml.feature_store import definitions as defs
from sigil_ml.models.duration import FEATURE_NAMES as DURATION_FEATURE_NAMES
from sigil_ml.models.stuck import FEATURE_NAMES as STUCK_FEATURE_NAMES

# (feature view, the constant it must mirror, expected field count)
VIEW_CONSTANT_PAIRS = [
    (defs.stuck_features, STUCK_FEATURE_NAMES, 6),
    (defs.duration_features, DURATION_FEATURE_NAMES, 4),
]


def field_names(view: FeatureView) -> list[str]:
    """Ordered feature names of a view, excluding entity/join-key columns."""
    return [f.name for f in view.features]


class TestEntities:
    """T001 — entities and their join keys."""

    def test_three_entities_declared(self):
        assert [e.name for e in defs.ENTITIES] == ["task", "node", "tenant"]

    @pytest.mark.parametrize(
        ("entity", "join_key"),
        [(defs.task, "task_id"), (defs.node, "node_id"), (defs.tenant, "tenant_id")],
    )
    def test_join_keys(self, entity, join_key):
        # Feast normalises the `join_keys=[...]` constructor argument down to a
        # single `join_key` attribute; there is no `join_keys` to read back.
        assert entity.join_key == join_key

    def test_tenant_not_required_by_any_locally_resolvable_view(self):
        """The local deployment is single-user and cannot supply a tenant id.

        A local view keyed by `tenant` would make a local prediction depend on
        an identifier that does not exist on that machine.
        """
        for view in defs.LOCAL_FEATURE_VIEWS:
            assert "tenant" not in view.entities, f"{view.name} requires the cloud-only tenant entity"
            assert "tenant_id" not in [f.name for f in view.entity_columns]

    def test_every_shipped_view_is_locally_resolvable(self):
        # If these ever diverge, the split is intentional and this test should be
        # updated deliberately rather than silently.
        assert defs.FEATURE_VIEWS == defs.LOCAL_FEATURE_VIEWS


class TestFieldOrdering:
    """T002 — the expensive-mistake subtask."""

    @pytest.mark.parametrize(("view", "constant", "count"), VIEW_CONSTANT_PAIRS)
    def test_fields_match_constant_in_order(self, view, constant, count):
        # Ordered comparison. Do not weaken this to set(...) == set(...).
        assert field_names(view) == list(constant)

    @pytest.mark.parametrize(("view", "constant", "count"), VIEW_CONSTANT_PAIRS)
    def test_field_count(self, view, constant, count):
        assert len(field_names(view)) == count
        assert len(constant) == count

    def test_a_reordering_would_be_caught(self):
        """Guards the guard: prove ordered comparison rejects a permutation.

        A set-based assertion passes on the reversed list below, which is why
        this file uses list equality throughout.
        """
        reordered = list(reversed(STUCK_FEATURE_NAMES))
        assert set(reordered) == set(STUCK_FEATURE_NAMES)  # a set check is blind to this
        assert reordered != field_names(defs.stuck_features)  # list equality is not

    @pytest.mark.parametrize(("view", "constant", "count"), VIEW_CONSTANT_PAIRS)
    def test_ordering_survives_the_registry_round_trip(self, view, constant, count):
        """Ordering must hold in the serialized protobuf, not just in memory.

        What ships to users is the registry file, not these Python objects.
        """
        restored = FeatureView.from_proto(view.to_proto())
        assert field_names(restored) == list(constant)


class TestFieldTypes:
    """T002 — every feature is Float64."""

    @pytest.mark.parametrize(("view", "constant", "count"), VIEW_CONSTANT_PAIRS)
    def test_all_features_are_float64(self, view, constant, count):
        assert [f.dtype for f in view.features] == [Float64] * len(constant)

    @pytest.mark.parametrize(("view", "constant", "count"), VIEW_CONSTANT_PAIRS)
    def test_join_key_column_is_string(self, view, constant, count):
        assert [(f.name, f.dtype) for f in view.entity_columns] == [("task_id", String)]


class TestTTLs:
    """T002 — each view bounds how far back a lookup reaches."""

    @pytest.mark.parametrize(("view", "constant", "count"), VIEW_CONSTANT_PAIRS)
    def test_ttl_is_set_and_bounded(self, view, constant, count):
        # A ttl of 0 (Feast's default) means "lives forever", which is not a bound.
        assert view.ttl is not None
        assert view.ttl > timedelta(0), f"{view.name} has an unbounded TTL"

    def test_declared_ttl_values(self):
        assert defs.stuck_features.ttl == defs.STUCK_TTL == timedelta(days=7)
        assert defs.duration_features.ttl == defs.DURATION_TTL == timedelta(days=30)


class TestNoSourceBaked:
    """T002 — sources are bound per deployment, in WP02/WP04."""

    @pytest.mark.parametrize(("view", "constant", "count"), VIEW_CONSTANT_PAIRS)
    def test_shipped_views_carry_no_source(self, view, constant, count):
        assert view.batch_source is None
        assert view.stream_source is None

    def test_factories_accept_a_deployment_supplied_source(self):
        """The seam WP02/WP04 bind through, without changing the declared shape."""
        source = FileSource(name="test_source", path="/tmp/does-not-need-to-exist.parquet")
        bound = defs.stuck_feature_view(source=source)
        assert bound.batch_source is source
        assert field_names(bound) == list(STUCK_FEATURE_NAMES)


class TestFeatureServices:
    """T003 — one versioned contract per model."""

    def test_one_service_per_registered_model(self):
        assert sorted(defs.FEATURE_SERVICES) == sorted(defs.REGISTERED_FEATURE_NAMES)

    def test_service_names_match_their_keys(self):
        for model_name, service in defs.FEATURE_SERVICES.items():
            assert service.name == model_name

    def test_service_names_are_the_names_go_queries(self):
        """`ml_predictions.model` values are fixed by CLAUDE.md and not renameable."""
        go_model_names = {"stuck", "suggest", "duration", "quality", "profile"}
        assert set(defs.FEATURE_SERVICES).issubset(go_model_names)

    def test_each_service_references_an_existing_view(self):
        known = {v.name for v in defs.FEATURE_VIEWS}
        for service in defs.FEATURE_SERVICES.values():
            referenced = [p.name for p in service.feature_view_projections]
            assert referenced, f"service {service.name} references no view"
            assert set(referenced).issubset(known)

    @pytest.mark.parametrize("model_name", ["stuck", "duration"])
    def test_service_exposes_its_model_features_in_order(self, model_name):
        service = defs.FEATURE_SERVICES[model_name]
        (projection,) = service.feature_view_projections
        assert [f.name for f in projection.features] == list(defs.REGISTERED_FEATURE_NAMES[model_name])

    def test_no_duplicate_service_names(self):
        names = [s.name for s in defs.FEATURE_SERVICES.values()]
        assert len(names) == len(set(names))


class TestRegistrationCoverage:
    """T004 — a new FEATURE_NAMES constant must not ship unregistered."""

    @staticmethod
    def discover_feature_name_constants() -> dict[str, list[str]]:
        """Walk `sigil_ml.models` for modules declaring a `FEATURE_NAMES` constant."""
        found = {}
        for info in pkgutil.iter_modules(sigil_ml.models.__path__):
            module = importlib.import_module(f"sigil_ml.models.{info.name}")
            constant = getattr(module, "FEATURE_NAMES", None)
            if constant is not None:
                found[info.name] = constant
        return found

    def test_every_feature_names_constant_has_a_feature_view(self):
        discovered = self.discover_feature_name_constants()
        unregistered = sorted(set(discovered) - set(defs.REGISTERED_FEATURE_NAMES))
        assert not unregistered, (
            f"model module(s) {unregistered} declare FEATURE_NAMES but no feature view registers them. "
            "Add a feature view in sigil_ml/feature_store/definitions.py deriving its schema from the "
            "constant, add a FeatureService for the model, and list it in REGISTERED_FEATURE_NAMES."
        )

    def test_no_registration_points_at_a_missing_constant(self):
        discovered = self.discover_feature_name_constants()
        stale = sorted(set(defs.REGISTERED_FEATURE_NAMES) - set(discovered))
        assert not stale, f"REGISTERED_FEATURE_NAMES references model module(s) with no FEATURE_NAMES: {stale}"

    def test_registered_constants_are_the_real_objects(self):
        """Registration must hold the imported constant, not a copy of its text."""
        discovered = self.discover_feature_name_constants()
        for model_name, constant in defs.REGISTERED_FEATURE_NAMES.items():
            assert constant is discovered[model_name]

    def test_every_registered_model_has_a_view_mirroring_its_constant(self):
        by_name = {v.name: v for v in defs.FEATURE_VIEWS}
        for model_name, constant in defs.REGISTERED_FEATURE_NAMES.items():
            view = by_name[f"{model_name}_features"]
            assert field_names(view) == list(constant)


class TestNoArithmetic:
    """D-002 — this package registers features; it never computes them."""

    def test_definitions_module_imports_no_numeric_machinery(self):
        assert not hasattr(defs, "np")
        assert not hasattr(defs, "numpy")

    def test_definitions_do_not_import_the_extractors(self):
        """Importing `sigil_ml.features` here would blur the D-002 boundary."""
        source = inspect.getsource(defs)
        assert "from sigil_ml.features import" not in source
        assert "import sigil_ml.features" not in source
