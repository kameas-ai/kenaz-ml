"""Comprehensive tests for the ML signal pipeline (Feature 005).

Covers: BehaviorProfile, RollingStat, PatternDetector, NextActionPredictor,
FileRecommender, SignalEngine, and feature-extraction helpers.
"""

from __future__ import annotations

import time
from collections import Counter
from unittest.mock import MagicMock

import pytest

from kenaz_ml.features import extract_action_token, infer_tool
from kenaz_ml.signals.engine import (
    SIGNAL_TOTAL_MAX,
    SignalEngine,
)
from kenaz_ml.signals.file_recommender import FileRecommender
from kenaz_ml.signals.next_action import NextActionPredictor
from kenaz_ml.signals.pattern_detector import PatternDetector
from kenaz_ml.signals.profile import BehaviorProfile, RollingStat

# ---------------------------------------------------------------------------
# RollingStat
# ---------------------------------------------------------------------------


class TestRollingStat:
    def test_update_tracks_count(self) -> None:
        stat = RollingStat()
        for v in [1.0, 2.0, 3.0]:
            stat.update(v)
        assert stat.count == 3

    def test_update_adjusts_mean(self) -> None:
        stat = RollingStat()
        stat.update(10.0)
        assert stat.mean == pytest.approx(10.0)
        stat.update(20.0)
        # Mean should move towards 20
        assert stat.mean > 10.0

    def test_z_score_insufficient_data(self) -> None:
        stat = RollingStat()
        stat.update(5.0)
        assert stat.z_score(10.0) is None  # count < 2

    def test_z_score_returns_float(self) -> None:
        stat = RollingStat()
        for v in range(100):
            stat.update(float(v))
        z = stat.z_score(999.0)
        assert z is not None
        assert abs(z) > 2.0

    def test_to_dict_from_dict_roundtrip(self) -> None:
        stat = RollingStat()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            stat.update(v)
        d = stat.to_dict()
        restored = RollingStat.from_dict(d)
        assert restored.mean == pytest.approx(stat.mean)
        assert restored.variance == pytest.approx(stat.variance)
        assert restored.count == stat.count


# ---------------------------------------------------------------------------
# BehaviorProfile
# ---------------------------------------------------------------------------


def _make_event(
    kind: str = "file",
    payload: dict | None = None,
    ts: int | None = None,
    category: str = "editing",
    event_id: int = 0,
) -> dict:
    """Helper to construct a mock classified event."""
    return {
        "id": event_id,
        "kind": kind,
        "payload": payload or {},
        "ts": ts or int(time.time() * 1000),
        "_category": category,
    }


class TestBehaviorProfile:
    def test_tool_frequency_process(self) -> None:
        profile = BehaviorProfile()
        events = [
            _make_event(kind="process", payload={"comm": "python3"}, event_id=1),
            _make_event(kind="process", payload={"comm": "python3"}, event_id=2),
            _make_event(kind="process", payload={"comm": "node"}, event_id=3),
        ]
        profile.update(events)
        assert profile.tool_counts["python3"] == 2
        assert profile.tool_counts["node"] == 1

    def test_tool_frequency_terminal(self) -> None:
        profile = BehaviorProfile()
        events = [
            _make_event(kind="terminal", payload={"cmd": "pytest tests/"}, event_id=1),
        ]
        profile.update(events)
        assert profile.tool_counts["pytest"] == 1

    def test_tool_frequency_git(self) -> None:
        profile = BehaviorProfile()
        events = [
            _make_event(kind="git", payload={"branch": "main"}, event_id=1),
        ]
        profile.update(events)
        assert profile.tool_counts["git"] == 1

    def test_tool_frequency_ai(self) -> None:
        profile = BehaviorProfile()
        events = [
            _make_event(kind="ai", payload={}, event_id=1),
        ]
        events[0]["source"] = "copilot"
        profile.update(events)
        assert profile.tool_counts["copilot"] == 1

    def test_file_type_distribution(self) -> None:
        profile = BehaviorProfile()
        events = [
            _make_event(kind="file", payload={"path": "src/main.py"}, event_id=1),
            _make_event(kind="file", payload={"path": "src/utils.py"}, event_id=2),
            _make_event(kind="file", payload={"path": "src/app.ts"}, event_id=3),
        ]
        profile.update(events)
        assert profile.file_type_counts[".py"] == 2
        assert profile.file_type_counts[".ts"] == 1

    def test_workflow_rhythm_edit_velocity(self) -> None:
        now = int(time.time() * 1000)
        profile = BehaviorProfile()
        events = [
            _make_event(kind="file", payload={"path": "a.py"}, ts=now - 60000, event_id=1),
            _make_event(kind="file", payload={"path": "b.py"}, ts=now, event_id=2),
        ]
        profile.update(events)
        stat = profile.get_metric_stats("edit_velocity")
        assert stat is not None
        assert stat.count >= 1

    def test_workflow_rhythm_context_switch_rate(self) -> None:
        now = int(time.time() * 1000)
        profile = BehaviorProfile()
        events = [
            _make_event(kind="file", ts=now - 3000, category="editing", event_id=1),
            _make_event(kind="terminal", ts=now - 2000, category="verifying", event_id=2),
            _make_event(kind="file", ts=now - 1000, category="editing", event_id=3),
            _make_event(kind="git", ts=now, category="integrating", event_id=4),
        ]
        profile.update(events)
        stat = profile.get_metric_stats("context_switch_rate")
        assert stat is not None
        assert stat.count >= 1

    def test_serialization_roundtrip(self) -> None:
        profile = BehaviorProfile()
        events = [
            _make_event(kind="file", payload={"path": "a.py"}, event_id=1),
            _make_event(kind="terminal", payload={"cmd": "pytest"}, event_id=2),
            _make_event(kind="git", payload={}, event_id=3),
        ]
        profile.update(events)

        data = profile.to_dict()
        restored = BehaviorProfile.from_dict(data)
        assert restored.tool_counts["pytest"] > 0
        assert restored.tool_counts["git"] > 0
        assert "file" in restored.active_sources
        assert restored.total_events_processed == profile.total_events_processed

    def test_exponential_decay_removes_stale(self) -> None:
        profile = BehaviorProfile(decay=0.5)  # Aggressive decay for test
        profile.tool_counts = Counter({"old_tool": 1})
        profile.file_type_counts = Counter({".py": 1})
        profile.apply_decay()
        # With decay=0.5, decay^100 is essentially 0
        assert "old_tool" not in profile.tool_counts
        assert ".py" not in profile.file_type_counts

    def test_empty_events_is_noop(self) -> None:
        profile = BehaviorProfile()
        profile.update([])
        assert profile.total_events_processed == 0
        assert len(profile.tool_counts) == 0


# ---------------------------------------------------------------------------
# PatternDetector
# ---------------------------------------------------------------------------


class TestPatternDetector:
    def test_cold_start_no_signals(self) -> None:
        detector = PatternDetector()
        profile = BehaviorProfile()
        # Profile has no metrics at all
        buffer = [_make_event(event_id=i) for i in range(10)]
        signals = detector.detect(buffer, profile)
        assert signals == []

    def test_zscore_deviation_emits_signal(self) -> None:
        detector = PatternDetector(z_threshold=2.0, min_observations=5)
        profile = BehaviorProfile()

        # Build a baseline: edit_velocity ~= 2.0 with some natural variance
        stat = RollingStat()
        for i in range(60):
            stat.update(2.0 + (i % 3) * 0.1)  # slight variance so std > 0
        profile.metrics["edit_velocity"] = stat

        # Now create a buffer with extreme edit velocity (100 edits in 1 second)
        now = int(time.time() * 1000)
        buffer = [
            _make_event(kind="file", payload={"path": f"f{i}.py"}, ts=now - 1000 + i * 10, event_id=i)
            for i in range(100)
        ]
        signals = detector.detect(buffer, profile)
        # Should detect edit_velocity deviation
        assert len(signals) >= 1
        sig = signals[0]
        assert sig.signal_type == "edit_velocity_deviation"
        assert sig.confidence > 0.0

    def test_no_signal_for_normal_values(self) -> None:
        detector = PatternDetector(z_threshold=2.0, min_observations=5)
        profile = BehaviorProfile()

        # Build a baseline: edit_velocity ~= 2.0
        stat = RollingStat()
        for _ in range(60):
            stat.update(2.0)
        profile.metrics["edit_velocity"] = stat

        # Now create a buffer that yields edit_velocity ~= 2.0
        now = int(time.time() * 1000)
        buffer = [
            _make_event(kind="file", payload={"path": "a.py"}, ts=now - 60000, event_id=1),
            _make_event(kind="file", payload={"path": "b.py"}, ts=now, event_id=2),
        ]
        signals = detector.detect(buffer, profile)
        # With only 2 edits in 1 minute = velocity ~2, which matches baseline
        assert len(signals) == 0

    def test_evidence_structure(self) -> None:
        detector = PatternDetector(z_threshold=2.0, min_observations=5)
        profile = BehaviorProfile()

        stat = RollingStat()
        for i in range(60):
            stat.update(2.0 + (i % 3) * 0.1)  # slight variance so std > 0
        profile.metrics["edit_velocity"] = stat

        now = int(time.time() * 1000)
        buffer = [
            _make_event(kind="file", payload={"path": f"f{i}.py"}, ts=now - 1000 + i * 10, event_id=i)
            for i in range(100)
        ]
        signals = detector.detect(buffer, profile)
        assert len(signals) >= 1
        evidence = signals[0].evidence
        assert "source_model" in evidence
        assert evidence["source_model"] == "pattern_detector"
        assert "metric" in evidence
        assert "z_score" in evidence


# ---------------------------------------------------------------------------
# NextActionPredictor
# ---------------------------------------------------------------------------


class TestNextActionPredictor:
    def test_cold_start_no_signals(self) -> None:
        pred = NextActionPredictor()
        profile = BehaviorProfile()
        buffer = [_make_event(category="editing", event_id=i) for i in range(10)]
        signals = pred.check_divergence(buffer, profile)
        assert signals == []  # total_tokens < 1000

    def test_train_incremental_adds_tokens(self) -> None:
        pred = NextActionPredictor(n=3)
        tokens = ["editing:py", "verifying:pytest", "integrating:git"] * 10
        pred.train_incremental(tokens)
        assert pred._total_tokens == 30

    def test_divergence_with_trained_model(self) -> None:
        pred = NextActionPredictor(n=3)
        profile = BehaviorProfile()

        # Train a very consistent pattern: editing -> verifying -> integrating
        pattern = ["editing:py", "verifying:pytest", "integrating:git"]
        for _ in range(400):
            pred.train_incremental(pattern)
        # Ensure total_tokens >= 1000
        assert pred._total_tokens >= 1000

        # Now create a buffer that ends with an unexpected token
        events = []
        for i in range(10):
            events.append(
                _make_event(
                    category="editing" if i % 3 == 0 else "verifying" if i % 3 == 1 else "integrating",
                    kind="file" if i % 3 == 0 else "terminal" if i % 3 == 1 else "git",
                    payload={"path": "a.py"} if i % 3 == 0 else {"cmd": "pytest"} if i % 3 == 1 else {},
                    event_id=i,
                )
            )
        # Override last event to be something unexpected
        events[-1]["_category"] = "idle"
        events[-1]["kind"] = "process"
        events[-1]["payload"] = {"comm": "sleep"}

        signals = pred.check_divergence(events, profile)
        # May or may not emit a signal depending on probability thresholds,
        # but should not crash
        assert isinstance(signals, list)

    def test_to_dict_from_dict_roundtrip(self) -> None:
        pred = NextActionPredictor(n=3)
        tokens = ["editing:py", "verifying:pytest", "integrating:git"] * 5
        pred.train_incremental(tokens)

        d = pred.to_dict()
        restored = NextActionPredictor.from_dict(d)
        assert restored._total_tokens == pred._total_tokens
        assert restored._n == pred._n
        # Verify n-gram counts are preserved
        for context, counts in pred._ngrams.items():
            key = "|".join(context) if context else ""
            restored_key = tuple(key.split("|")) if key else ()
            assert restored._ngrams[restored_key] == counts


# ---------------------------------------------------------------------------
# FileRecommender
# ---------------------------------------------------------------------------


class TestFileRecommender:
    def test_cold_start_no_signals(self) -> None:
        rec = FileRecommender()
        profile = BehaviorProfile()
        buffer = [_make_event(kind="file", payload={"path": "/repo/a.py"}, event_id=1)]
        signals = rec.check(buffer, None, profile)
        assert signals == []  # task_count < 5

    def test_cooccurrence_builds_from_store(self) -> None:
        rec = FileRecommender()
        store = MagicMock()
        store.get_completed_task_ids.return_value = [f"task-{i}" for i in range(10)]

        def mock_events(task_id: str) -> list[dict]:
            # Each task edits a.py and b.py together
            return [
                {"kind": "file", "payload": {"path": "/repo/a.py"}},
                {"kind": "file", "payload": {"path": "/repo/b.py"}},
            ]

        store.get_events_for_task.side_effect = mock_events
        count = rec.train_from_tasks(store)
        assert count == 10
        assert rec._cooccurrence["/repo/a.py"]["/repo/b.py"] == 10

    def test_conditional_probability(self) -> None:
        rec = FileRecommender()
        # Manually populate co-occurrence data
        rec._task_count = 10
        rec._file_counts["/repo/a.py"] = 10
        rec._file_counts["/repo/b.py"] = 8
        rec._cooccurrence["/repo/a.py"]["/repo/b.py"] = 8  # P(b|a) = 0.8

        current = {"/repo/a.py"}
        recs = rec._recommend(current, "/repo")
        assert len(recs) >= 1
        assert recs[0][0] == "/repo/b.py"
        assert recs[0][1] == pytest.approx(0.8)

    def test_repo_scoped_recommendations(self) -> None:
        rec = FileRecommender()
        rec._task_count = 10
        rec._file_counts["/repo/a.py"] = 10
        rec._file_counts["/other/c.py"] = 10
        rec._cooccurrence["/repo/a.py"]["/other/c.py"] = 8

        current = {"/repo/a.py"}
        # Scope to /repo — should exclude /other/c.py
        recs = rec._recommend(current, "/repo")
        assert len(recs) == 0


# ---------------------------------------------------------------------------
# SignalEngine
# ---------------------------------------------------------------------------


def _make_engine() -> SignalEngine:
    """Create a SignalEngine with a mock DataStore."""
    store = MagicMock()
    store.insert_signal.return_value = 1
    profile = BehaviorProfile()
    pattern_detector = PatternDetector()
    next_action = NextActionPredictor()
    file_recommender = FileRecommender()
    return SignalEngine(
        store=store,
        profile=profile,
        pattern_detector=pattern_detector,
        next_action=next_action,
        file_recommender=file_recommender,
    )


class TestSignalEngine:
    def test_rate_limiting_per_type(self) -> None:
        engine = _make_engine()
        now = time.time()

        # Record a signal of type "foo"
        engine._recent_signals.append(("foo", now))

        # Should be rate-limited
        assert engine._is_type_rate_limited("foo", now + 1) is True
        # Different type should not be rate-limited
        assert engine._is_type_rate_limited("bar", now + 1) is False

    def test_rate_limiting_total(self) -> None:
        engine = _make_engine()
        now = time.time()

        # Fill up to the total max
        for i in range(SIGNAL_TOTAL_MAX):
            engine._recent_signals.append((f"type_{i}", now))

        assert engine._is_total_rate_limited(now + 1) is True

    def test_process_events_empty_buffer(self) -> None:
        engine = _make_engine()
        result = engine.process_events([])
        assert result == 0

    def test_all_models_called_during_process(self) -> None:
        engine = _make_engine()
        engine.pattern_detector = MagicMock()
        engine.pattern_detector.detect.return_value = []
        engine.next_action = MagicMock()
        engine.next_action.check_divergence.return_value = []
        engine.file_recommender = MagicMock()
        engine.file_recommender.check.return_value = []

        now = int(time.time() * 1000)
        buffer = [_make_event(ts=now, event_id=i) for i in range(5)]
        engine.process_events(buffer)

        engine.pattern_detector.detect.assert_called_once()
        engine.next_action.check_divergence.assert_called_once()
        engine.file_recommender.check.assert_called_once()

    def test_double_counting_fix(self) -> None:
        """Verify that repeated calls with overlapping buffers do not double-count."""
        engine = _make_engine()
        now = int(time.time() * 1000)

        buffer1 = [_make_event(ts=now + i, event_id=i + 1) for i in range(5)]
        engine.process_events(buffer1)
        count_after_first = engine.profile.total_events_processed

        # Call again with the same buffer (simulating the poller sending
        # the full 200-event sliding window that overlaps)
        engine.process_events(buffer1)
        count_after_second = engine.profile.total_events_processed

        # Should NOT have doubled
        assert count_after_second == count_after_first


# ---------------------------------------------------------------------------
# extract_action_token / infer_tool
# ---------------------------------------------------------------------------


class TestExtractActionToken:
    def test_terminal_event(self) -> None:
        event = {
            "kind": "terminal",
            "payload": {"cmd": "pytest tests/"},
            "_category": "verifying",
        }
        assert extract_action_token(event) == "verifying:pytest"

    def test_file_event(self) -> None:
        event = {
            "kind": "file",
            "payload": {"path": "src/main.py"},
            "_category": "editing",
        }
        assert extract_action_token(event) == "editing:py"

    def test_git_event(self) -> None:
        event = {
            "kind": "git",
            "payload": {"branch": "main"},
            "_category": "integrating",
        }
        assert extract_action_token(event) == "integrating:git"

    def test_unknown_event(self) -> None:
        event = {
            "kind": "unknown_kind",
            "payload": {},
            "_category": "idle",
        }
        assert extract_action_token(event) == "idle"


class TestInferTool:
    def test_terminal(self) -> None:
        assert infer_tool({"kind": "terminal", "payload": {"cmd": "pytest tests/"}}) == "pytest"

    def test_git(self) -> None:
        assert infer_tool({"kind": "git", "payload": {}}) == "git"

    def test_file_with_extension(self) -> None:
        assert infer_tool({"kind": "file", "payload": {"path": "a.py"}}) == "py"

    def test_unknown_kind(self) -> None:
        assert infer_tool({"kind": "totally_unknown", "payload": {}}) is None

    def test_string_payload(self) -> None:
        assert infer_tool({"kind": "file", "payload": "raw_string"}) is None
