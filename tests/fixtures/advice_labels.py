"""TEST-ONLY label-row builders for harness-recommendation-models-01MSK2RM.

Rows are pushed through Mission A's real ingest (``advice.label_log.ingest``),
so the retained set and label log the tests train from are exactly what the
engine would hold — never a hand-written imitation of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from kenaz_ml.advice import label_log
from kenaz_ml.advice.contracts import contract_for

DAY_MS = 24 * 3600 * 1000
T0 = 1_750_000_000_000  # a fixed epoch-ms origin; tests never read the wall clock


def features_for(kind: str, rng: np.random.Generator, signal: float) -> dict[str, float]:
    """A feature dict whose first feature carries ``signal``; the rest are noise."""
    names = contract_for(kind).names  # type: ignore[union-attr]
    values = {name: float(rng.normal()) for name in names}
    values[names[0]] = float(signal)
    return values


def label_row(
    kind: str,
    ts: int,
    features: dict[str, float],
    *,
    action: str = "accepted",
    complete: bool = True,
    shown: bool = True,
    revision: int = 1,
    model_id: str = "heuristic/rule",
    rung: str = "heuristic",
    session_id: str | None = "s1",
    confidence: int = 80,
) -> dict[str, Any]:
    contract = contract_for(kind)
    assert contract is not None
    feats = dict(features)
    if not complete:
        feats.pop(contract.names[-1], None)
    return {
        "client": "harness",
        "kind": kind,
        "features_hash": f"h{ts}",
        "ts": ts,
        "revision": revision,
        "feature_contract_version": contract.service_version,
        "features": feats,
        "features_complete": complete,
        "shown": shown,
        "confidence": confidence,
        "model_id": model_id,
        "rung": rung,
        "prompt_version": "p1",
        "user_action": action,
        "recommendation": {"decision": True},
        "latency_ms": 2.0,
        "session_id": session_id,
    }


def separable_rows(
    kind: str, n: int, *, start_ts: int = T0, seed: int = 0, step_ms: int = 60_000, **row_kwargs: Any
) -> list[dict[str, Any]]:
    """``n`` complete rows whose label follows the first feature (with some noise)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        signal = float(rng.normal())
        accepted = (signal + 0.5 * rng.normal()) > 0
        rows.append(
            label_row(
                kind,
                start_ts + i * step_ms,
                features_for(kind, rng, signal),
                action="accepted" if accepted else "dismissed",
                **row_kwargs,
            )
        )
    return rows


def push(kind: str, rows: list[dict[str, Any]], directory: Path) -> label_log.IngestResult:
    contract = contract_for(kind)
    result = label_log.ingest(kind, "harness", rows, contract, directory=directory, now_ms=T0)
    assert result.refusal is None, (result.refusal, result.refusal_detail)
    return result
