"""Tests for fleet team-level models (focus, meeting, onboarding)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sigil_ml.models.fleet_focus import FleetFocusModel
from sigil_ml.models.fleet_meeting import FleetMeetingModel
from sigil_ml.models.fleet_onboarding import FleetOnboardingModel
from sigil_ml.storage.model_store import LocalModelStore


@pytest.fixture
def tmp_store(tmp_path: Path) -> LocalModelStore:
    """Create a temporary LocalModelStore."""
    return LocalModelStore(base_dir=tmp_path)


# ---------- FleetFocusModel ----------


class TestFleetFocusModel:
    def test_untrained_predict_returns_empty(self, tmp_store: LocalModelStore) -> None:
        model = FleetFocusModel(team_id=1, model_store=tmp_store)
        assert not model.is_trained
        result = model.predict()
        assert result["predictions"] == {}
        assert result["optimal_windows"] == []

    def test_train_and_predict(self, tmp_store: LocalModelStore) -> None:
        model = FleetFocusModel(team_id=1, model_store=tmp_store)
        data = [
            {"hour": h, "day_of_week": d, "meeting_minutes": 20, "context_switches": 5, "focus_score": 60 + h}
            for h in range(8, 18)
            for d in range(5)
        ]
        result = model.train(data)
        assert result["model"] == "fleet_focus"
        assert result["team_id"] == 1
        assert result["samples"] == 50
        assert "rmse" in result["metrics"]
        assert "r2" in result["metrics"]
        assert model.is_trained

        preds = model.predict()
        assert "mon" in preds["predictions"]
        assert len(preds["predictions"]["mon"]) == 24

    def test_train_too_few_samples(self, tmp_store: LocalModelStore) -> None:
        model = FleetFocusModel(team_id=1, model_store=tmp_store)
        with pytest.raises(ValueError, match="at least 5"):
            model.train([{"hour": 1, "focus_score": 50}])

    def test_persistence_roundtrip(self, tmp_store: LocalModelStore) -> None:
        model = FleetFocusModel(team_id=42, model_store=tmp_store)
        data = [
            {"hour": h, "day_of_week": 0, "meeting_minutes": 10, "context_switches": 3, "focus_score": 50 + h}
            for h in range(10)
        ]
        model.train(data)

        # Load a fresh instance — should pick up the saved model
        loaded = FleetFocusModel(team_id=42, model_store=tmp_store)
        assert loaded.is_trained
        preds = loaded.predict()
        assert len(preds["predictions"]) == 7


# ---------- FleetMeetingModel ----------


class TestFleetMeetingModel:
    def test_untrained_predict_returns_empty(self, tmp_store: LocalModelStore) -> None:
        model = FleetMeetingModel(team_id=1, model_store=tmp_store)
        assert not model.is_trained
        result = model.predict()
        assert result["scenarios"] == []

    def test_train_and_predict(self, tmp_store: LocalModelStore) -> None:
        model = FleetMeetingModel(team_id=1, model_store=tmp_store)
        data = [
            {"meeting_duration": d, "time_of_day": 10, "focus_before": 80, "focus_after": 80 - d // 2}
            for d in range(10, 100, 5)
        ]
        result = model.train(data)
        assert result["model"] == "fleet_meeting"
        assert result["samples"] == len(data)
        assert "accuracy" in result["metrics"]

        preds = model.predict()
        assert len(preds["scenarios"]) > 0
        for s in preds["scenarios"]:
            assert s["disruption"] in ("low", "medium", "high")

    def test_train_too_few_samples(self, tmp_store: LocalModelStore) -> None:
        model = FleetMeetingModel(team_id=1, model_store=tmp_store)
        with pytest.raises(ValueError, match="at least 5"):
            model.train([{"meeting_duration": 30}])

    def test_persistence_roundtrip(self, tmp_store: LocalModelStore) -> None:
        model = FleetMeetingModel(team_id=7, model_store=tmp_store)
        data = [
            {"meeting_duration": d, "time_of_day": 10, "focus_before": 80, "focus_after": 60} for d in range(10, 60, 5)
        ]
        model.train(data)

        loaded = FleetMeetingModel(team_id=7, model_store=tmp_store)
        assert loaded.is_trained


# ---------- FleetOnboardingModel ----------


class TestFleetOnboardingModel:
    def test_untrained_predict_returns_empty(self, tmp_store: LocalModelStore) -> None:
        model = FleetOnboardingModel(team_id=1, model_store=tmp_store)
        assert not model.is_trained
        result = model.predict()
        assert result["trajectory"] == []
        assert result["predicted_ramp_up_days"] is None

    def test_train_and_predict(self, tmp_store: LocalModelStore) -> None:
        model = FleetOnboardingModel(team_id=1, model_store=tmp_store)
        # Linear ramp: day 1 → 20%, day 90 → 100%
        data = [{"day_number": d, "performance_pct": 20 + (80 * d / 90)} for d in range(1, 91)]
        result = model.train(data)
        assert result["model"] == "fleet_onboarding"
        assert result["samples"] == 90
        assert "r2" in result["metrics"]
        assert result["metrics"]["r2"] > 0.9  # linear data should fit well

        preds = model.predict()
        assert len(preds["trajectory"]) == 90
        # Ramp-up should be predicted around day 67-68 (when 20 + 80*d/90 >= 80)
        assert preds["predicted_ramp_up_days"] is not None
        assert 60 <= preds["predicted_ramp_up_days"] <= 75

    def test_train_too_few_samples(self, tmp_store: LocalModelStore) -> None:
        model = FleetOnboardingModel(team_id=1, model_store=tmp_store)
        with pytest.raises(ValueError, match="at least 5"):
            model.train([{"day_number": 1, "performance_pct": 20}])

    def test_persistence_roundtrip(self, tmp_store: LocalModelStore) -> None:
        model = FleetOnboardingModel(team_id=99, model_store=tmp_store)
        data = [{"day_number": d, "performance_pct": d} for d in range(1, 20)]
        model.train(data)

        loaded = FleetOnboardingModel(team_id=99, model_store=tmp_store)
        assert loaded.is_trained
        preds = loaded.predict()
        assert len(preds["trajectory"]) == 90
