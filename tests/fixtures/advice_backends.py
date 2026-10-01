"""TEST-ONLY dispatch backends (two-client-engine-01MSK2EN WP03, plan D-A5).

The engine ships no heuristic backend (Amendment A3.2). These fixtures exist
only to prove the dispatch layer end to end: a served kind whose backend is a
trivial threshold on one feature, and a kind configured for the dormant ``laya``
backend. They are honest about what they are — ``backend: "heuristic"``,
``model: "heuristic/<fixture>@<version>"``, ``unbenchmarked: true`` — and port
no harness rule. They must never move under ``src/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kenaz_ml.advice.contracts import build_contract
from kenaz_ml.advice.dispatch import (
    BACKEND_HEURISTIC,
    BACKEND_LAYA,
    REASON_LAYA_NOT_INSTALLED,
    BackendAnswer,
    DispatchEntry,
    DispatchTable,
)

FIXTURE_KIND = "fixture_threshold"
FIXTURE_LAYA_KIND = "fixture_laya"
FIXTURE_FEATURES = ("signal", "noise")


@dataclass
class ThresholdFixtureBackend:
    """``decision = signal >= threshold``; ``confidence`` is whatever the test sets."""

    fixture_name: str = "threshold-fixture"
    version: str = "1"
    threshold: float = 0.5
    confidence: Any = 80
    name: str = BACKEND_HEURISTIC
    rung: str = BACKEND_HEURISTIC
    checkpoint_provenance: str = "local"
    model_id_sha8: str | None = None
    generation: str = "0"
    unbenchmarked: bool = True

    @property
    def model_label(self) -> str:
        return f"heuristic/{self.fixture_name}@{self.version}"

    def answer(self, vector: tuple[float, ...]) -> BackendAnswer:
        return BackendAnswer(decision=vector[0] >= self.threshold, confidence=self.confidence)


def fixture_contract(kind_id: str = FIXTURE_KIND) -> Any:
    return build_contract(kind_id, FIXTURE_FEATURES)


def served_entry(backend: ThresholdFixtureBackend | None = None, kind_id: str = FIXTURE_KIND) -> DispatchEntry:
    server = backend or ThresholdFixtureBackend()
    return DispatchEntry(kind_id=kind_id, contract=fixture_contract(kind_id), backend=server.name, server=server)


def laya_entry(kind_id: str = FIXTURE_LAYA_KIND) -> DispatchEntry:
    return DispatchEntry(
        kind_id=kind_id,
        contract=fixture_contract(kind_id),
        backend=BACKEND_LAYA,
        reason=REASON_LAYA_NOT_INSTALLED,
        detail="kind unavailable: laya backend not installed",
    )


def register_fixtures(table: DispatchTable, backend: ThresholdFixtureBackend | None = None) -> ThresholdFixtureBackend:
    """Install the served fixture kind and the laya-configured kind; return the served backend."""
    server = backend or ThresholdFixtureBackend()
    table.replace_entry(served_entry(server))
    table.replace_entry(laya_entry())
    return server


def fixture_features(signal: float = 0.9) -> dict[str, float]:
    return {"signal": signal, "noise": 0.1}
