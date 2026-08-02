"""ML Signal Pipeline — event-driven behavioral signal detection.

This package contains:
  - Signal: shared dataclass for all signal models
  - SignalEngine: orchestrator that runs all models per poll cycle (WP06)
  - BehaviorProfile: incremental per-user behavioral profile (WP02)
  - PatternDetector: z-score and Isolation Forest anomaly detection (WP03)
  - NextActionPredictor: n-gram action sequence prediction (WP04)
  - FileRecommender: co-occurrence file recommendation (WP05)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from kenaz_ml.signals.profile import BehaviorProfile, RollingStat

__all__ = [
    "BehaviorProfile",
    "FileRecommender",
    "NextActionPredictor",
    "PatternDetector",
    "RollingStat",
    "Signal",
    "SignalEngine",
]


def __getattr__(name: str) -> type:  # noqa: N807
    """Lazy imports for signal model classes to avoid circular imports."""
    if name == "PatternDetector":
        from kenaz_ml.signals.pattern_detector import PatternDetector

        return PatternDetector
    if name == "NextActionPredictor":
        from kenaz_ml.signals.next_action import NextActionPredictor

        return NextActionPredictor
    if name == "FileRecommender":
        from kenaz_ml.signals.file_recommender import FileRecommender

        return FileRecommender
    if name == "SignalEngine":
        from kenaz_ml.signals.engine import SignalEngine

        return SignalEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class Signal:
    """A structured signal emitted when a model detects something noteworthy.

    Produced by signal models (PatternDetector, NextActionPredictor, FileRecommender).
    Consumed by SignalEngine, which writes it to ml_signals via DataStore.
    """

    signal_type: str
    """Model-generated type, e.g. 'velocity_deviation', 'divergence_test', 'file_cluster'."""

    confidence: float
    """Model's confidence (0.0 to 1.0) that this signal is worth surfacing."""

    evidence: dict[str, Any]
    """Structured evidence for LLM rendering. Must include 'source_model' key."""

    suggested_action: str | None = None
    """Generic action hint for LLM: 'investigate', 'test', 'commit', 'take_break'."""

    ttl_sec: int | None = None
    """Time-to-live in seconds. None = no expiry."""

    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    """Unix ms timestamp. Auto-populated on creation."""
