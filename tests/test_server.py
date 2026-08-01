"""Tests for the FastAPI server endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolate_models(tmp_path, monkeypatch):
    """Redirect model weights to a temp directory."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))


@pytest.fixture
def client():
    """Create a test client for the FastAPI app.

    Uses raise_server_exceptions=False to avoid leaking startup errors.
    The TestClient context manager triggers startup/shutdown events.
    """
    from sigil_ml.app import create_app

    application = create_app()
    with TestClient(application) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models" in data
        assert "uptime_sec" in data

    def test_health_reports_model_status(self, client: TestClient) -> None:
        resp = client.get("/health")
        data = resp.json()
        models = data["models"]
        assert "stuck" in models
        assert "activity" in models
        assert "workflow" in models
        assert "duration" in models
        # Models should be untrained since we use a clean temp dir
        assert models["stuck"] == "untrained"
        assert models["duration"] == "untrained"


class TestStuckEndpoint:
    def test_predict_with_features(self, client: TestClient) -> None:
        resp = client.post(
            "/predict/stuck",
            json={
                "features": {
                    "test_failure_count": 5,
                    "time_in_phase_sec": 1200,
                    "edit_velocity": 4.0,
                    "file_switch_rate": 0.7,
                    "session_length_sec": 3600,
                    "time_since_last_commit_sec": 1800,
                }
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "probability" in data
        assert "confidence" in data
        assert data["confidence"] in ("weak", "moderate", "strong")

    def test_predict_no_input(self, client: TestClient) -> None:
        resp = client.post("/predict/stuck", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["probability"] == 0.5
        assert data["confidence"] == "weak"


class TestSuggestEndpoint:
    def test_predict_returns_workflow_state(self, client: TestClient) -> None:
        resp = client.post("/predict/suggest", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "flow_state" in data
        assert "dominant_state" in data
        assert "momentum" in data
        assert "focus_score" in data
        assert "method" in data
        assert "confidence" in data
        # Flow state should have all 5 states.
        from sigil_ml.models.workflow import FLOW_STATES

        for state in FLOW_STATES:
            assert state in data["flow_state"]

    def test_predict_with_classified_events(self, client: TestClient) -> None:
        from sigil_ml.models.workflow import FLOW_STATES

        events = [
            {"kind": "file", "_category": "editing", "ts": 1000},
            {"kind": "terminal", "_category": "verifying", "ts": 2000},
        ]
        resp = client.post("/predict/suggest", json={"classified_events": events})
        assert resp.status_code == 200
        data = resp.json()
        assert data["dominant_state"] in FLOW_STATES


class TestDurationEndpoint:
    def test_predict_with_features(self, client: TestClient) -> None:
        resp = client.post(
            "/predict/duration",
            json={
                "features": {
                    "file_count": 10,
                    "total_edits": 80,
                    "time_of_day_hour": 14,
                    "branch_name_length": 25,
                }
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "estimated_minutes" in data
        assert "confidence_interval" in data
        assert len(data["confidence_interval"]) == 2

    def test_predict_no_input(self, client: TestClient) -> None:
        resp = client.post("/predict/duration", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["estimated_minutes"] == 60.0


class TestTrainEndpoint:
    def test_train_returns_started(self, client: TestClient) -> None:
        resp = client.post("/train", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"

    def test_train_with_custom_db(self, client: TestClient, tmp_path) -> None:
        # db field is deprecated but still accepted for backward compat
        db = str(tmp_path / "custom.db")
        resp = client.post("/train", json={"db": db})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"


class TestIntrospectEndpoint:
    """GET /introspect — sidecar self-description for kenaz spec 060."""

    def test_introspect_returns_service_identity(self, client: TestClient) -> None:
        resp = client.get("/introspect")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "kameas-ml"
        assert data["mode"] == "local"
        assert data["uptime_sec"] >= 0

        from sigil_ml import __version__

        assert data["version"] == __version__

    def test_introspect_lists_all_local_models(self, client: TestClient) -> None:
        resp = client.get("/introspect")
        data = resp.json()
        by_name = {m["name"]: m for m in data["models"]}
        assert set(by_name) == {"stuck", "activity", "workflow", "duration", "quality"}

        # Clean temp model dir → sklearn models are loaded but untrained.
        stuck = by_name["stuck"]
        assert stuck["display_name"] == "Stuck Predictor"
        assert stuck["prediction_model"] == "stuck"
        assert stuck["status"] == "untrained"
        assert stuck["trained"] is False
        assert stuck["enabled"] is True

        # workflow predictions land in ml_predictions under "suggest".
        assert by_name["workflow"]["prediction_model"] == "suggest"

    def test_introspect_returns_honest_nulls(self, client: TestClient) -> None:
        """Untracked metadata must be null/zero, never invented (spec 060)."""
        resp = client.get("/introspect")
        data = resp.json()
        for model in data["models"]:
            # Per-model training sample counts are not tracked today.
            assert model["sample_count"] is None
            # Nothing has been trained in this clean temp dir.
            assert model["last_trained"] is None
            # No predictions have been written in this test environment.
            assert model["recent_predictions"] == 0

    def test_introspect_capabilities(self, client: TestClient) -> None:
        resp = client.get("/introspect")
        data = resp.json()
        # Local mode supports retrain (POST /train); no per-predictor toggle
        # exists — kenaz hides the control on discovery (FR-008).
        assert data["capabilities"] == {"retrain": True, "toggle": False}

    def test_introspect_reports_last_trained_after_weights_write(self, client: TestClient) -> None:
        """A persisted weights file surfaces as an ISO last_trained stamp."""
        import io

        import joblib
        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier

        from sigil_ml.modelstore import LocalModelStore

        clf = GradientBoostingClassifier(n_estimators=2)
        clf.fit(np.zeros((4, 6)), [0, 1, 0, 1])
        buf = io.BytesIO()
        joblib.dump(clf, buf)
        LocalModelStore().save("stuck", buf.getvalue())

        resp = client.get("/introspect")
        by_name = {m["name"]: m for m in resp.json()["models"]}
        assert by_name["stuck"]["last_trained"] is not None
        # ISO-8601 UTC stamp — parseable and tz-aware.
        from datetime import datetime

        parsed = datetime.fromisoformat(by_name["stuck"]["last_trained"])
        assert parsed.tzinfo is not None
