"""The VOCABULARY_VERSION semantic salt (feature-vocabulary-refresh WP02, D-D4).

The name/order hash cannot see a semantics-only change; the salt can. It is
folded in only for the vocabulary-consuming services (stuck, activity); every
other service hashes exactly as it did before the salt existed.
"""

from __future__ import annotations

import hashlib

import pytest

from kenaz_ml.feature_store import definitions, materialize
from kenaz_ml.feature_store.materialize import (
    VOCABULARY_SALTED_SERVICES,
    feature_service_version,
    versioned_contract_hash,
)
from kenaz_ml.features import extract_activity_features
from kenaz_ml.models.activity import ACTIVITY_FEATURE_NAMES, activity_feature_contract
from kenaz_ml.modelstore.registry import local_feature_contract


def _legacy_version(service) -> str:
    """The pre-salt recipe, reproduced independently of the code under test."""
    refs = [f"{p.name}:{f.name}" for p in service.feature_view_projections for f in p.features]
    return hashlib.sha256("|".join([service.name, *refs]).encode()).hexdigest()[:16]


def test_the_salted_services_are_exactly_stuck_and_activity() -> None:
    assert {"stuck", "activity"} == VOCABULARY_SALTED_SERVICES


def test_stuck_version_changed_and_duration_did_not() -> None:
    stuck = definitions.FEATURE_SERVICES["stuck"]
    duration = definitions.FEATURE_SERVICES["duration"]
    assert feature_service_version(stuck) != _legacy_version(stuck), "the salt is the bump"
    assert feature_service_version(duration) == _legacy_version(duration), "duration hashes exactly as before"
    assert local_feature_contract("stuck").service_version == feature_service_version(stuck)


def test_a_vocabulary_bump_moves_stuck_and_activity_but_not_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    before = {
        "stuck": local_feature_contract("stuck").service_version,
        "activity": activity_feature_contract().service_version,
        "duration": local_feature_contract("duration").service_version,
    }
    monkeypatch.setattr(materialize, "VOCABULARY_VERSION", materialize.VOCABULARY_VERSION + 1)
    assert local_feature_contract("stuck").service_version != before["stuck"]
    assert activity_feature_contract().service_version != before["activity"]
    assert local_feature_contract("duration").service_version == before["duration"]


def test_unsalted_names_hash_without_the_salt() -> None:
    assert versioned_contract_hash("other", ["a:b"]) == hashlib.sha256(b"other|a:b").hexdigest()[:16]


def test_activity_contract_is_ordered_stable_and_matches_the_extractor() -> None:
    contract = activity_feature_contract()
    assert contract.names == ACTIVITY_FEATURE_NAMES
    assert contract == activity_feature_contract(), "stable across calls"
    assert contract.service == "activity" and contract.service_version
    assert len(contract.dtypes) == len(contract.names)
    # The explicit names must track the extractor: a vocabulary edit is a visible, versioned edit.
    assert list(ACTIVITY_FEATURE_NAMES) == sorted(extract_activity_features({"kind": "file"}))


def test_the_loader_seam_returns_the_registered_activity_contract() -> None:
    from kenaz_ml.modelstore.loader import _expected_contract

    assert _expected_contract("activity") == activity_feature_contract()
    assert local_feature_contract("activity") is None, "activity is hand-registered, not Feast-derived"
