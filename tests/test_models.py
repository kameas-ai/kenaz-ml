"""Tests for ML model classes."""

import numpy as np
import pytest

from kenaz_ml.training.synthetic import generate_duration_data, generate_stuck_data


@pytest.fixture(autouse=True)
def _isolate_models(tmp_path, monkeypatch):
    """Redirect model weights to a temp directory so tests don't pollute real config."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))


class TestStuckPredictor:
    def test_untrained_returns_default(self) -> None:
        from kenaz_ml.models.stuck import StuckPredictor

        model = StuckPredictor()
        result = model.predict(
            {
                "test_failure_count": 5,
                "time_in_phase_sec": 1200,
                "edit_velocity": 4.0,
                "file_switch_rate": 0.7,
                "session_length_sec": 3600,
                "time_since_last_commit_sec": 1800,
            }
        )
        assert result["probability"] == 0.5
        assert result["confidence"] == "weak"

    def test_train_and_predict(self) -> None:
        from kenaz_ml.models.stuck import StuckPredictor

        X, y = generate_stuck_data(200)
        model = StuckPredictor()
        model.train(X, y)

        assert model.is_trained

        # Stuck-looking sample
        stuck_features = {
            "test_failure_count": 8,
            "time_in_phase_sec": 2400,
            "edit_velocity": 5.0,
            "file_switch_rate": 0.8,
            "session_length_sec": 5000,
            "time_since_last_commit_sec": 2500,
        }
        result = model.predict(stuck_features)
        assert result["probability"] > 0.5

        # Not-stuck sample
        ok_features = {
            "test_failure_count": 0,
            "time_in_phase_sec": 100,
            "edit_velocity": 1.0,
            "file_switch_rate": 0.2,
            "session_length_sec": 600,
            "time_since_last_commit_sec": 120,
        }
        result_ok = model.predict(ok_features)
        assert result_ok["probability"] < 0.5

    def test_confidence_levels(self) -> None:
        from kenaz_ml.models.stuck import StuckPredictor

        X, y = generate_stuck_data(200)
        model = StuckPredictor()
        model.train(X, y)

        # The model should produce varying confidence levels across different inputs
        results = []
        for _ in range(50):
            features = {
                "test_failure_count": float(np.random.randint(0, 10)),
                "time_in_phase_sec": float(np.random.uniform(30, 3600)),
                "edit_velocity": float(np.random.uniform(0.5, 8)),
                "file_switch_rate": float(np.random.uniform(0.1, 1.0)),
                "session_length_sec": float(np.random.uniform(300, 7200)),
                "time_since_last_commit_sec": float(np.random.uniform(60, 3600)),
            }
            results.append(model.predict(features)["confidence"])
        # Should see at least 2 different confidence levels
        assert len(set(results)) >= 2

    def test_weights_persist(self) -> None:
        from kenaz_ml.models.stuck import StuckPredictor

        X, y = generate_stuck_data(100)

        model1 = StuckPredictor()
        model1.train(X, y)

        features = {
            "test_failure_count": 5,
            "time_in_phase_sec": 1500,
            "edit_velocity": 4.0,
            "file_switch_rate": 0.6,
            "session_length_sec": 3000,
            "time_since_last_commit_sec": 1800,
        }
        pred1 = model1.predict(features)

        # Load from disk
        model2 = StuckPredictor()
        assert model2.is_trained
        pred2 = model2.predict(features)
        assert pred1["probability"] == pred2["probability"]


class TestActivityClassifier:
    def test_file_event_classifies_as_editing(self) -> None:
        from kenaz_ml.models.activity import ActivityClassifier

        clf = ActivityClassifier()
        result = clf.classify({"kind": "file", "payload": {"path": "/src/main.py", "action": "write"}})
        assert result["category"] == "editing"
        assert result["method"] == "rules"
        assert result["confidence"] > 0

    def test_terminal_test_command_classifies_as_verifying(self) -> None:
        from kenaz_ml.models.activity import ActivityClassifier

        clf = ActivityClassifier()
        result = clf.classify({"kind": "terminal", "payload": {"cmd": "pytest tests/ -v"}})
        assert result["category"] == "verifying"

    def test_terminal_build_command_classifies_as_verifying(self) -> None:
        from kenaz_ml.models.activity import ActivityClassifier

        clf = ActivityClassifier()
        result = clf.classify({"kind": "terminal", "payload": {"cmd": "go test ./..."}})
        assert result["category"] == "verifying"

    def test_terminal_commit_classifies_as_integrating(self) -> None:
        from kenaz_ml.models.activity import ActivityClassifier

        clf = ActivityClassifier()
        result = clf.classify({"kind": "terminal", "payload": {"cmd": "git commit -m 'fix'"}})
        assert result["category"] == "integrating"

    def test_git_event_classifies_as_integrating(self) -> None:
        from kenaz_ml.models.activity import ActivityClassifier

        clf = ActivityClassifier()
        result = clf.classify({"kind": "git", "payload": {"branch": "main"}})
        assert result["category"] == "integrating"

    def test_ai_event_classifies_as_researching(self) -> None:
        from kenaz_ml.models.activity import ActivityClassifier

        clf = ActivityClassifier()
        result = clf.classify({"kind": "ai", "payload": {}})
        assert result["category"] == "researching"

    def test_hyprland_event_classifies_as_navigating(self) -> None:
        from kenaz_ml.models.activity import ActivityClassifier

        clf = ActivityClassifier()
        result = clf.classify({"kind": "hyprland", "payload": {"window_class": "firefox"}})
        assert result["category"] == "navigating"

    def test_process_event_classifies_as_navigating(self) -> None:
        from kenaz_ml.models.activity import ActivityClassifier

        clf = ActivityClassifier()
        result = clf.classify({"kind": "process", "payload": {"exe": "nvim"}})
        assert result["category"] == "navigating"

    def test_unknown_event_classifies_as_idle(self) -> None:
        from kenaz_ml.models.activity import ActivityClassifier

        clf = ActivityClassifier()
        result = clf.classify({"kind": "unknown"})
        assert result["category"] == "idle"

    def test_plugin_source_classifies_as_communicating(self) -> None:
        from kenaz_ml.models.activity import ActivityClassifier

        clf = ActivityClassifier()
        result = clf.classify({"kind": "plugin", "source": "github", "payload": {}})
        assert result["category"] == "communicating"

    def test_classify_batch(self) -> None:
        from kenaz_ml.models.activity import ActivityClassifier

        clf = ActivityClassifier()
        events = [
            {"kind": "file", "payload": {"path": "/a.py"}},
            {"kind": "terminal", "payload": {"cmd": "pytest"}},
            {"kind": "git", "payload": {}},
        ]
        results = clf.classify_batch(events)
        assert len(results) == 3
        assert results[0]["category"] == "editing"
        assert results[1]["category"] == "verifying"
        assert results[2]["category"] == "integrating"

    def test_untrained_is_trained_false(self) -> None:
        from kenaz_ml.models.activity import ActivityClassifier

        clf = ActivityClassifier()
        assert clf.is_trained is False

    def test_train_and_classify_ml(self) -> None:
        from kenaz_ml.features import extract_activity_features
        from kenaz_ml.models.activity import CATEGORIES_FULL, ActivityClassifier

        clf = ActivityClassifier()

        # Generate training data with the correct feature shape.
        sample_event = {"kind": "file", "payload": {"path": "/a.py"}}
        sample_feats = extract_activity_features(sample_event)
        feature_names = sorted(sample_feats.keys())
        n_features = len(feature_names)

        rng = np.random.RandomState(42)
        n = 200
        X = rng.rand(n, n_features)
        y = rng.choice(CATEGORIES_FULL, size=n)

        clf.train(X, y)
        assert clf.is_trained is True

        # ML classify should return method="ml".
        result = clf._classify_ml(sample_event)
        assert result["method"] == "ml"
        assert result["category"] in CATEGORIES_FULL

    def test_weights_persist(self) -> None:
        from kenaz_ml.models.activity import CATEGORIES_FULL, ActivityClassifier

        rng = np.random.RandomState(42)
        X = rng.rand(100, 6)
        y = rng.choice(CATEGORIES_FULL, size=100)

        clf1 = ActivityClassifier()
        clf1.train(X, y)

        clf2 = ActivityClassifier()
        assert clf2.is_trained is True


class TestWorkflowStatePredictor:
    def _make_events(self, categories: list[str], ts_start: int = 1000000) -> list[dict]:
        """Helper to create classified events."""
        return [{"kind": "file", "_category": cat, "ts": ts_start + i * 1000} for i, cat in enumerate(categories)]

    def test_rules_deep_work(self) -> None:
        from kenaz_ml.models.workflow import WorkflowStatePredictor

        model = WorkflowStatePredictor()
        # 90% editing, 10% verifying — high focus, low navigating.
        events = self._make_events(["editing"] * 9 + ["verifying"])
        result = model.predict(events, {"session_elapsed_min": 30, "test_failures": 0})
        assert result["dominant_state"] == "deep_work"
        assert result["method"] == "rules"

    def test_rules_blocked(self) -> None:
        from kenaz_ml.models.workflow import WorkflowStatePredictor

        model = WorkflowStatePredictor()
        events = self._make_events(["verifying"] * 6 + ["editing"] * 4)
        result = model.predict(events, {"session_elapsed_min": 30, "test_failures": 5})
        assert result["dominant_state"] == "blocked"

    def test_rules_exploring(self) -> None:
        from kenaz_ml.models.workflow import WorkflowStatePredictor

        model = WorkflowStatePredictor()
        events = self._make_events(["navigating"] * 5 + ["researching"] * 4 + ["editing"])
        result = model.predict(events, {"session_elapsed_min": 15, "test_failures": 0})
        assert result["dominant_state"] == "exploring"

    def test_rules_default_shallow_work(self) -> None:
        from kenaz_ml.models.workflow import WorkflowStatePredictor

        model = WorkflowStatePredictor()
        # Mixed activity without any strong signal — should default to shallow_work.
        events = self._make_events(["editing"] * 4 + ["navigating"] * 2 + ["verifying"] * 2 + ["researching"] * 2)
        result = model.predict(events, {"session_elapsed_min": 10, "test_failures": 0})
        assert result["dominant_state"] == "shallow_work"

    def test_flow_state_sums_to_one(self) -> None:
        from kenaz_ml.models.workflow import WorkflowStatePredictor

        model = WorkflowStatePredictor()
        events = self._make_events(["editing"] * 5 + ["verifying"] * 5)
        result = model.predict(events, {"session_elapsed_min": 20, "test_failures": 1})
        total = sum(result["flow_state"].values())
        assert abs(total - 1.0) < 0.01

    def test_momentum_positive_when_accelerating(self) -> None:
        from kenaz_ml.models.workflow import WorkflowStatePredictor

        model = WorkflowStatePredictor()
        # Even split — momentum should be near 0 for equal halves.
        events = self._make_events(["editing"] * 10)
        result = model.predict(events, {"session_elapsed_min": 10, "test_failures": 0})
        assert -0.5 <= result["momentum"] <= 0.5

    def test_focus_score_high_for_concentrated(self) -> None:
        from kenaz_ml.models.workflow import WorkflowStatePredictor

        model = WorkflowStatePredictor()
        events = self._make_events(["editing"] * 10)
        result = model.predict(events, {"session_elapsed_min": 10, "test_failures": 0})
        assert result["focus_score"] == 1.0

    def test_focus_score_lower_for_scattered(self) -> None:
        from kenaz_ml.models.workflow import WorkflowStatePredictor

        model = WorkflowStatePredictor()
        events = self._make_events(["editing", "verifying", "navigating", "researching"] * 3)
        result = model.predict(events, {"session_elapsed_min": 10, "test_failures": 0})
        assert result["focus_score"] < 0.5

    def test_output_shape(self) -> None:
        from kenaz_ml.models.workflow import FLOW_STATES, WorkflowStatePredictor

        model = WorkflowStatePredictor()
        events = self._make_events(["editing"] * 5)
        result = model.predict(events, {"session_elapsed_min": 5, "test_failures": 0})

        assert "flow_state" in result
        assert all(s in result["flow_state"] for s in FLOW_STATES)
        assert "dominant_state" in result
        assert "momentum" in result
        assert "focus_score" in result
        assert "dominant_activity" in result
        assert "activity_distribution" in result
        assert "session_elapsed_min" in result
        assert "method" in result
        assert "confidence" in result

    def test_untrained_is_trained_false(self) -> None:
        from kenaz_ml.models.workflow import WorkflowStatePredictor

        model = WorkflowStatePredictor()
        assert model.is_trained is False

    def test_empty_events(self) -> None:
        from kenaz_ml.models.workflow import WorkflowStatePredictor

        model = WorkflowStatePredictor()
        result = model.predict([], {"session_elapsed_min": 0, "test_failures": 0})
        assert result["dominant_activity"] == "idle"
        assert result["focus_score"] == 1.0

    def test_weights_persist(self) -> None:
        from kenaz_ml.features import extract_workflow_features
        from kenaz_ml.models.workflow import FLOW_STATES, WorkflowStatePredictor

        rng = np.random.RandomState(42)
        events = [{"_category": rng.choice(["editing", "verifying"]), "ts": 1000 + i * 1000} for i in range(50)]
        session_info = {"session_elapsed_min": 10, "test_failures": 0}
        feats = extract_workflow_features(events, session_info)
        n_features = len(feats)

        X = rng.rand(100, n_features)
        y = rng.choice(FLOW_STATES, size=100)

        model1 = WorkflowStatePredictor()
        model1.train(X, y)
        assert model1.is_trained

        model2 = WorkflowStatePredictor()
        assert model2.is_trained


class TestDurationEstimator:
    def test_untrained_returns_default(self) -> None:
        from kenaz_ml.models.duration import DurationEstimator

        model = DurationEstimator()
        result = model.predict(
            {
                "file_count": 5,
                "total_edits": 50,
                "time_of_day_hour": 14,
                "branch_name_length": 20,
            }
        )
        assert result["estimated_minutes"] == 60.0
        assert len(result["confidence_interval"]) == 2

    def test_train_and_predict(self) -> None:
        from kenaz_ml.models.duration import DurationEstimator

        X, y = generate_duration_data(200)
        model = DurationEstimator()
        model.train(X, y)

        assert model.is_trained

        # Many files + edits should predict longer duration
        big_task = {
            "file_count": 25,
            "total_edits": 180,
            "time_of_day_hour": 14,
            "branch_name_length": 40,
        }
        result_big = model.predict(big_task)

        small_task = {
            "file_count": 2,
            "total_edits": 10,
            "time_of_day_hour": 14,
            "branch_name_length": 10,
        }
        result_small = model.predict(small_task)

        assert result_big["estimated_minutes"] > result_small["estimated_minutes"]

    def test_confidence_interval(self) -> None:
        from kenaz_ml.models.duration import DurationEstimator

        X, y = generate_duration_data(200)
        model = DurationEstimator()
        model.train(X, y)

        result = model.predict(
            {
                "file_count": 10,
                "total_edits": 80,
                "time_of_day_hour": 10,
                "branch_name_length": 25,
            }
        )
        low, high = result["confidence_interval"]
        assert low <= result["estimated_minutes"] <= high
        assert low >= 0

    def test_weights_persist(self) -> None:
        from kenaz_ml.models.duration import DurationEstimator

        X, y = generate_duration_data(100)

        model1 = DurationEstimator()
        model1.train(X, y)

        features = {
            "file_count": 10,
            "total_edits": 60,
            "time_of_day_hour": 12,
            "branch_name_length": 20,
        }
        pred1 = model1.predict(features)

        model2 = DurationEstimator()
        assert model2.is_trained
        pred2 = model2.predict(features)
        assert pred1["estimated_minutes"] == pred2["estimated_minutes"]
