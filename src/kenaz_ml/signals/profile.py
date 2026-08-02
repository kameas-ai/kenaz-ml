"""Per-user behavior profile built incrementally from observed events.

The profile tracks tool frequencies, file type distributions, and
workflow rhythm metrics. It serves as the personalization backbone
for all signal models — ensuring signals only reference tools
and patterns the user actually uses.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class RollingStat:
    """Exponentially weighted rolling mean and variance.

    Updates in O(1) time per observation. Tracks count for
    minimum-observation thresholds.
    """

    mean: float = 0.0
    variance: float = 0.0
    count: int = 0
    _alpha: float = 0.995  # decay factor (~500 event half-life)

    @property
    def std(self) -> float:
        """Standard deviation (square root of variance)."""
        return self.variance**0.5

    def update(self, value: float) -> None:
        """Update the rolling statistics with a new observation."""
        self.count += 1
        if self.count == 1:
            self.mean = value
            self.variance = 0.0
            return
        diff = value - self.mean
        self.mean += (1 - self._alpha) * diff
        self.variance = self._alpha * (self.variance + (1 - self._alpha) * diff * diff)

    def to_dict(self) -> dict[str, float | int]:
        """Serialize to a JSON-compatible dict."""
        return {"mean": self.mean, "variance": self.variance, "count": self.count}

    def z_score(self, value: float) -> float | None:
        """Compute z-score of value against this stat's baseline.

        Returns None if insufficient observations (count < 2) or
        if the standard deviation is near zero (constant values).
        """
        if self.count < 2 or self.std < 1e-10:
            return None
        return (value - self.mean) / self.std

    @classmethod
    def from_dict(cls, d: dict, alpha: float = 0.995) -> RollingStat:
        """Restore from a serialized dict."""
        return cls(
            mean=d.get("mean", 0.0),
            variance=d.get("variance", 0.0),
            count=d.get("count", 0),
            _alpha=alpha,
        )


# Test command prefixes — shared with ActivityClassifier._classify_rules.
_TEST_PREFIXES = (
    "pytest",
    "go test",
    "npm test",
    "cargo test",
    "jest",
    "vitest",
    "mocha",
    "python -m pytest",
)


class BehaviorProfile:
    """Incrementally updated per-user behavioral profile.

    Tracks tool usage, file type distribution, and workflow rhythm metrics.
    All signal models consume this profile for personalization — ensuring
    signals only reference tools and patterns the user actually uses.

    Args:
        decay: Exponential decay factor for rolling statistics.
               Default 0.995 gives ~500 event half-life.
    """

    def __init__(self, decay: float = 0.995) -> None:
        self._decay = decay
        self.tool_counts: Counter[str] = Counter()
        self.file_type_counts: Counter[str] = Counter()
        self.metrics: dict[str, RollingStat] = {}
        self.active_sources: set[str] = set()
        self.total_events_processed: int = 0
        self._last_commit_ts: int = 0
        self._last_test_ts: int = 0
        self._session_start_ts: int = 0

    def update(self, events: list[dict]) -> None:
        """Incrementally update profile from a buffer of classified events.

        Call this on every poll cycle with the new events. Updates tool counts,
        file type distribution, workflow rhythm metrics, and active sources.
        """
        if not events:
            return
        for e in events:
            self._update_tool_count(e)
            self._update_file_types(e)
            self._track_sources(e)
        self._update_rhythm_stats(events)
        self.total_events_processed += len(events)
        # Periodic decay to remove stale entries
        if self.total_events_processed % 1000 < len(events):
            self.apply_decay()

    # --- Tool frequency tracking (T008) ---

    def _update_tool_count(self, event: dict) -> None:
        """Update tool frequency counter from a single event."""
        kind = event.get("kind", "")
        payload = event.get("payload") or {}
        if isinstance(payload, str):
            return

        if kind == "process":
            comm = str(payload.get("comm", "")).split("/")[-1].strip("()")
            if comm:
                self.tool_counts[comm] += 1

        elif kind == "terminal":
            cmd = str(payload.get("cmd", "")).strip()
            first_word = cmd.split()[0].split("/")[-1] if cmd else ""
            if first_word:
                self.tool_counts[first_word] += 1

        elif kind == "git":
            self.tool_counts["git"] += 1

        elif kind == "ai":
            source = event.get("source", "ai")
            self.tool_counts[source] += 1

    # --- File type distribution tracking (T009) ---

    def _update_file_types(self, event: dict) -> None:
        """Update file type distribution from file events."""
        if event.get("kind") != "file":
            return
        payload = event.get("payload") or {}
        if isinstance(payload, str):
            return
        path = str(payload.get("path", ""))
        if not path:
            return
        if "." in path:
            ext = "." + path.rsplit(".", 1)[-1].lower()
        else:
            ext = "other"
        self.file_type_counts[ext] += 1

    # --- Source tracking ---

    def _track_sources(self, event: dict) -> None:
        """Track which event sources are active."""
        kind = event.get("kind", "")
        if kind:
            self.active_sources.add(kind)

    # --- Workflow rhythm metrics (T010) ---

    def _update_rhythm_stats(self, events: list[dict]) -> None:
        """Compute and update rolling statistics for workflow rhythm metrics."""
        if not events:
            return

        now_ms = int(time.time() * 1000)

        # Track session start
        if self._session_start_ts == 0:
            self._session_start_ts = events[0].get("ts", now_ms)

        # Edit velocity: file events per minute in this buffer
        window_start = events[0].get("ts", now_ms)
        window_end = events[-1].get("ts", now_ms)
        window_min = max((window_end - window_start) / 60000.0, 1 / 60.0)
        edit_count = sum(1 for e in events if e.get("kind") == "file")
        if edit_count > 0:
            velocity = edit_count / window_min
            self._get_stat("edit_velocity").update(velocity)

        # Commit cadence: time between git events (minutes)
        for e in events:
            if e.get("kind") == "git":
                ts = e.get("ts", now_ms)
                if self._last_commit_ts > 0:
                    gap_min = (ts - self._last_commit_ts) / 60000.0
                    if gap_min > 0:
                        self._get_stat("commit_cadence").update(gap_min)
                self._last_commit_ts = ts

        # Test cadence: time between terminal test commands (minutes)
        for e in events:
            if e.get("kind") == "terminal":
                payload = e.get("payload") or {}
                if isinstance(payload, dict):
                    cmd = str(payload.get("cmd", "")).lower()
                    if any(cmd.startswith(p) for p in _TEST_PREFIXES):
                        ts = e.get("ts", now_ms)
                        if self._last_test_ts > 0:
                            gap_min = (ts - self._last_test_ts) / 60000.0
                            if gap_min > 0:
                                self._get_stat("test_cadence").update(gap_min)
                        self._last_test_ts = ts

        # Context switch rate: category transitions per buffer
        transitions = 0
        for i in range(1, len(events)):
            if events[i].get("_category") != events[i - 1].get("_category"):
                transitions += 1
        if len(events) > 1:
            switch_rate = transitions / (len(events) - 1)
            self._get_stat("context_switch_rate").update(switch_rate)

    def _get_stat(self, name: str) -> RollingStat:
        """Get or create a RollingStat for the given metric name."""
        if name not in self.metrics:
            self.metrics[name] = RollingStat(_alpha=self._decay)
        return self.metrics[name]

    # --- Exponential decay (T012) ---

    def apply_decay(self) -> None:
        """Apply exponential decay to frequency counters.

        Called periodically (every ~1000 events) to prevent stale
        tools/patterns from persisting indefinitely. Entries that
        decay to zero are removed.
        """
        decay = self._decay**100  # Apply 100 steps of decay at once

        decayed_tools: Counter[str] = Counter()
        for tool, count in self.tool_counts.items():
            new_count = int(count * decay)
            if new_count > 0:
                decayed_tools[tool] = new_count
        self.tool_counts = decayed_tools

        decayed_files: Counter[str] = Counter()
        for ext, count in self.file_type_counts.items():
            new_count = int(count * decay)
            if new_count > 0:
                decayed_files[ext] = new_count
        self.file_type_counts = decayed_files

    # --- Serialization (T011) ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile to a JSON-compatible dict for storage in ml_predictions."""
        total_files = sum(self.file_type_counts.values()) or 1
        return {
            "tool_frequency": dict(self.tool_counts.most_common(50)),
            "file_type_distribution": {
                ext: round(count / total_files, 4) for ext, count in self.file_type_counts.most_common(20)
            },
            "workflow_rhythms": {name: stat.to_dict() for name, stat in self.metrics.items()},
            "active_sources": sorted(self.active_sources),
            "total_events_processed": self.total_events_processed,
            "profile_version": 1,
            "updated_at": int(time.time() * 1000),
        }

    @classmethod
    def from_dict(cls, data: dict, decay: float = 0.995) -> BehaviorProfile:
        """Restore a profile from a serialized dict.

        This avoids replaying full event history on startup. The profile
        may be approximate for file type counts but quickly corrects
        as new events arrive.
        """
        profile = cls(decay=decay)
        profile.tool_counts = Counter(data.get("tool_frequency", {}))

        # Reconstruct file type counts from distribution (approximate)
        total_events = data.get("total_events_processed", 0)
        dist = data.get("file_type_distribution", {})
        for ext, frac in dist.items():
            profile.file_type_counts[ext] = int(frac * total_events)

        # Restore rolling stats
        rhythms = data.get("workflow_rhythms", {})
        for name, stat_dict in rhythms.items():
            profile.metrics[name] = RollingStat.from_dict(stat_dict, alpha=decay)

        profile.active_sources = set(data.get("active_sources", []))
        profile.total_events_processed = data.get("total_events_processed", 0)
        return profile

    # --- Query helpers (for signal models) ---

    def get_metric_stats(self, name: str) -> RollingStat | None:
        """Return the rolling statistics for a named metric, or None."""
        return self.metrics.get(name)

    def has_tool(self, tool: str) -> bool:
        """Return True if the user has used this tool."""
        return tool in self.tool_counts

    def top_tools(self, n: int = 10) -> list[tuple[str, int]]:
        """Return the top N tools by usage count."""
        return self.tool_counts.most_common(n)
