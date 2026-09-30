"""two-client-engine-01MSK2EN WP02 — hand-authored harness-kind contracts (D-A4)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kenaz_ml.advice import contracts
from kenaz_ml.advice.contracts import (
    ERROR_KINDS,
    KIND_IDS,
    VOCABULARY_VERSION,
    build_contract,
    contract_for,
    contract_version,
)
from kenaz_ml.modelstore.registry import Example, Manifest, append_examples, validate_feature_contract

#: The design doc §4 catalog, literally (research.md §4).
CATALOG = {
    "branch_now": (
        "turns_since_session_start",
        "turns_since_last_branch",
        "prior_branch_count",
        "edit_resend_precursor",
        "heuristic_signal_count",
        "heuristic_noise_count",
        "last_user_msg_len",
        "tool_call_density_window",
    ),
    "compact_now": (
        "context_fill_fraction",
        "tokens_in_span",
        "turns_since_last_compaction",
        "tool_result_token_fraction",
        "model_context_limit",
        "historical_compression_ratio",
    ),
    "escalate_model": (
        "consecutive_tool_failures",
        "retries_in_window",
        "turn_latency_trend",
        "current_rung",
        "error_kind_auth",
        "error_kind_transient",
        "error_kind_cancelled",
        "error_kind_budget",
        "error_kind_unknown",
        "budget_remaining_fraction",
    ),
}


def test_kind_ids_are_the_trio() -> None:
    assert KIND_IDS == ("branch_now", "compact_now", "escalate_model")


@pytest.mark.parametrize("kind", list(CATALOG))
def test_names_and_dtypes_match_the_catalog_literally(kind: str) -> None:
    contract = contract_for(kind)
    assert contract is not None
    assert contract.names == CATALOG[kind]
    assert contract.dtypes == ("float64",) * len(CATALOG[kind])
    assert len(set(contract.names)) == len(contract.names), "duplicate feature name"
    assert contract.service == kind


def test_error_kind_vocabulary_is_the_harness_enum_in_order() -> None:
    assert ERROR_KINDS == ("auth", "transient", "cancelled", "budget", "unknown")


def test_unknown_kind_has_no_contract() -> None:
    assert contract_for("stuck") is None
    assert contract_for("nope") is None


def test_no_serialized_vector_ambiguity_between_kinds() -> None:
    versions = {contract_for(k).service_version for k in KIND_IDS}  # type: ignore[union-attr]
    assert len(versions) == len(KIND_IDS)


@pytest.mark.parametrize("kind", list(CATALOG))
def test_service_version_is_non_empty_and_16_hex(kind: str) -> None:
    version = contract_for(kind).service_version  # type: ignore[union-attr]
    assert version and len(version) == 16
    int(version, 16)


def test_service_version_covers_order_membership_and_salt() -> None:
    names = CATALOG["compact_now"]
    base = contract_version("compact_now", names)
    reordered = (names[1], names[0], *names[2:])
    assert contract_version("compact_now", reordered) != base
    assert contract_version("compact_now", names[:-1]) != base
    assert contract_version("compact_now", (*names, "extra")) != base
    assert contract_version("compact_now", names, VOCABULARY_VERSION + 1) != base
    assert build_contract("compact_now", names, VOCABULARY_VERSION + 1).service_version != base


def test_contracts_are_stable_across_a_process_restart() -> None:
    script = (
        "import json; from kenaz_ml.advice.contracts import all_contracts;"
        "print(json.dumps({k: [list(c.names), list(c.dtypes), c.service_version]"
        " for k, c in all_contracts().items()}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin"},
    ).stdout
    other = json.loads(out)
    here = {k: [list(c.names), list(c.dtypes), c.service_version] for k, c in contracts.all_contracts().items()}
    assert other == here


@pytest.mark.parametrize("kind", list(CATALOG))
def test_validates_through_the_registry_ordered_comparison(kind: str) -> None:
    contract = contract_for(kind)
    manifest = Manifest(name=kind, version="1", artifact_sha256="0" * 64, feature_contract=contract)
    assert validate_feature_contract(manifest, contract).ok

    names = contract.names
    swapped = build_contract(kind, (names[1], names[0], *names[2:]))
    refused = validate_feature_contract(manifest, swapped)
    assert not refused.ok
    assert refused.refusal is not None
    assert "index 0" in refused.refusal.detail


@pytest.mark.parametrize("kind", list(CATALOG))
def test_each_contract_can_stamp_a_retained_set(kind: str, tmp_path: Path) -> None:
    contract = contract_for(kind)
    example = Example(x=tuple(0.0 for _ in contract.names), y=1, as_of_ms=1)
    result = append_examples(kind, [example], contract, directory=tmp_path)
    assert result.ok, result.reason
