"""Event poller — continuous push-to-db prediction loop.

Polls events -> classifies activity -> runs models -> writes to ml_predictions.
Runs as an asyncio background task inside the FastAPI process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kenaz_ml.signals.engine import SignalEngine

from kenaz_ml.datastore import DataStore
from kenaz_ml.feature_store.resolve import (
    resolve_duration_features,
    resolve_stuck_features,
    resolve_stuck_features_from_buffer,
)

logger = logging.getLogger(__name__)

POLL_INTERVAL_SEC = 0.5
PREDICT_EVERY_N_EVENTS = 3  # minimum events before predicting
PREDICT_MIN_INTERVAL_SEC = 60  # minimum seconds between prediction cycles
QUALITY_WINDOW_SEC = 1800  # 30-minute rolling window for quality features
PREDICTION_TTL_SEC = 90  # 90-second expiry for stuck/activity/workflow
QUALITY_TTL_SEC = 120  # 2-minute expiry for quality


class EventPoller:
    """Polls sigild's events table and writes predictions to ml_predictions."""

    def __init__(self, store: DataStore, models: dict[str, Any], signal_engine: SignalEngine | None = None) -> None:
        self.store = store
        self.stuck = models["stuck"]
        self.activity = models["activity"]
        self.workflow = models["workflow"]
        self.duration = models["duration"]
        self.quality = models["quality"]
        self.signal_engine = signal_engine  # Optional: None when not configured
        self._buffer: list[dict] = []
        self._since_last_predict = 0
        self._last_predict_time = 0.0
        self._running = False

    async def run(self) -> None:
        """Main loop — call as an asyncio task."""
        self._running = True
        logger.info("poller: started with %s", type(self.store).__name__)
        while self._running:
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._poll_once)
            except Exception as e:
                # Database may not exist yet or be locked — retry silently.
                logger.debug("poller: store error (will retry): %s", e)
            await asyncio.sleep(POLL_INTERVAL_SEC)

    def stop(self) -> None:
        self._running = False

    def _poll_once(self) -> None:
        since = self.store.get_cursor()

        rows = self.store.get_events_since(since, limit=100)

        if not rows:
            return

        events = []
        for e in rows:
            if isinstance(e.get("payload"), str):
                try:
                    e["payload"] = json.loads(e["payload"])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Classify each event as it enters the buffer.
            classification = self.activity.classify(e)
            e["_category"] = classification["category"]
            e["_category_confidence"] = classification["confidence"]

            events.append(e)

        self._buffer.extend(events)
        self._buffer = self._buffer[-200:]  # keep last 200
        self._since_last_predict += len(events)

        max_id = max(e["id"] for e in events)
        self.store.update_cursor(max_id)

        elapsed = time.time() - self._last_predict_time
        if self._since_last_predict >= PREDICT_EVERY_N_EVENTS and elapsed >= PREDICT_MIN_INTERVAL_SEC:
            self._predict_and_write()
            self._since_last_predict = 0
            self._last_predict_time = time.time()

        self.store.commit()

        # --- Signal detection (additive, does not affect predictions) ---
        if self.signal_engine is not None:
            try:
                task_id = self.store.get_active_task()
                task_context = {"task_id": task_id} if task_id else None
                self.signal_engine.process_events(self._buffer, task_context)
            except Exception:
                logger.debug("poller: signal engine error (non-fatal)", exc_info=True)

    # Fallback predictions for untrained models.
    _FALLBACK_STUCK = {"probability": 0.5, "confidence": "weak"}
    _FALLBACK_WORKFLOW = {
        "flow_state": {
            "deep_work": 0.0,
            "shallow_work": 1.0,
            "exploring": 0.0,
            "blocked": 0.0,
            "winding_down": 0.0,
        },
        "dominant_state": "shallow_work",
        "momentum": 0.0,
        "focus_score": 0.5,
        "dominant_activity": "idle",
        "activity_distribution": {},
        "session_elapsed_min": 0.0,
        "method": "rules",
        "confidence": 0.5,
    }
    _FALLBACK_DURATION = {"estimated_minutes": 60.0, "confidence_interval": [30.0, 90.0]}

    def _predict_and_write(self) -> None:
        start = time.time()
        task_id = self.store.get_active_task()

        # Stuck prediction — check is_trained before calling predict.
        if self.stuck.is_trained:
            if task_id:
                feats = resolve_stuck_features(self.store, task_id)
            else:
                feats = resolve_stuck_features_from_buffer(self._buffer)
            result = self.stuck.predict(feats)
        else:
            result = self._FALLBACK_STUCK
        self.store.insert_prediction("stuck", result, result.get("probability", 0.5), PREDICTION_TTL_SEC)

        # Activity summary — classify and summarize the buffer.
        activity_result = self._activity_summary()
        self.store.insert_prediction(
            "activity", activity_result, activity_result.get("confidence", 0.5), PREDICTION_TTL_SEC
        )

        # Workflow state prediction — replaces old suggestion policy.
        session_info = self._session_info(task_id)
        result = self.workflow.predict(self._buffer, session_info)
        self.store.insert_prediction("suggest", result, result.get("confidence", 0.5), PREDICTION_TTL_SEC)

        # Duration — only when active task AND model is trained.
        if task_id and self.duration.is_trained:
            try:
                feats = resolve_duration_features(self.store, task_id)
                result = self.duration.predict(feats)
                ci = result.get("confidence_interval", [30, 90])
                est = result.get("estimated_minutes", 60)
                rel_width = (ci[1] - ci[0]) / max(est, 1.0)
                conf = max(0.0, min(1.0, 1.0 - rel_width / 2.0))
                self.store.insert_prediction("duration", result, conf, None)
            except Exception:
                logger.debug("poller: duration prediction skipped", exc_info=True)
        elif task_id:
            self.store.insert_prediction("duration", self._FALLBACK_DURATION, 0.5, None)

        # Quality score — always callable (rule-based, no training required).
        qfeats = self._quality_features()
        result = self.quality.predict(qfeats)
        self.store.insert_prediction("quality", result, result.get("score", 50) / 100.0, QUALITY_TTL_SEC)

        # Write behavior profile (signal pipeline)
        if self.signal_engine is not None:
            try:
                profile_data = self.signal_engine.profile.to_dict()
                self.store.insert_prediction("profile", profile_data, 1.0, None)
                # Refresh dismissed signal types while we're here
                self.signal_engine.refresh_dismissed_types()
            except Exception:
                logger.debug("poller: profile write failed (non-fatal)", exc_info=True)

        # Audit log
        latency_ms = int((time.time() - start) * 1000)
        self.store.insert_ml_event("prediction", "poller", "local", latency_ms)

    def _activity_summary(self) -> dict:
        """Build activity summary from classified buffer events."""
        window_summary: dict[str, int] = {}
        for e in self._buffer:
            cat = e.get("_category", "idle")
            window_summary[cat] = window_summary.get(cat, 0) + 1

        recent = [
            {"ts": e.get("ts", 0), "kind": e.get("kind", ""), "category": e.get("_category", "idle")}
            for e in self._buffer[-10:]
        ]

        dominant = max(window_summary, key=lambda k: window_summary[k]) if window_summary else "idle"

        # Average confidence of classifications.
        confidences = [e.get("_category_confidence", 0.5) for e in self._buffer]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.5

        return {
            "window_summary": window_summary,
            "recent": recent,
            "dominant": dominant,
            "method": "rules",
            "confidence": round(avg_conf, 4),
        }

    def _session_info(self, task_id: str | None) -> dict:
        """Build session info for WorkflowStatePredictor."""
        session_elapsed_min = 0.0
        task_phase = None
        test_failures = 0

        if task_id:
            info = self.store.get_session_info(task_id)
            if info:
                started_at = info["started_at"] or 0
                session_elapsed_min = (time.time() * 1000 - started_at) / 60000.0
                task_phase = info["phase"]
                test_failures = info["test_fails"] or 0

        return {
            "session_elapsed_min": max(session_elapsed_min, 0.0),
            "task_phase": task_phase,
            "test_failures": test_failures,
        }

    def _quality_features(self) -> dict:
        now_ms = int(time.time() * 1000)
        window_start = now_ms - QUALITY_WINDOW_SEC * 1000
        window = [e for e in self._buffer if e.get("ts", 0) >= window_start]

        edit_events = [e for e in window if e.get("kind") == "file"]
        edits = len(edit_events)
        files: set[str] = set()
        for e in edit_events:
            p = e.get("payload") or {}
            if isinstance(p, dict) and "path" in p:
                files.add(p["path"])
        edit_focus = 1.0 - (len(files) / max(edits, 1))

        commit_events = [e for e in window if e.get("kind") == "git"]
        terminal_events = [e for e in window if e.get("kind") == "terminal"]

        stats = self.store.get_quality_task_stats()

        if stats and stats["test_runs"]:
            test_pass_rate = 1.0 - (stats["test_fails"] / max(stats["test_runs"], 1))
            baseline_commits = max(stats["commit_count"], 1)
        else:
            test_pass_rate = 0.7
            baseline_commits = 1

        return {
            "test_pass_rate": max(0.0, min(1.0, test_pass_rate)),
            "test_total": stats["test_runs"] if stats else 0,
            "edit_focus": max(0.0, min(1.0, edit_focus)),
            "velocity_ratio": min(edits / max(len(terminal_events), 1), 2.0),
            "commits_in_window": len(commit_events),
            "expected_commits": baseline_commits / 8.0,
            "revert_count": 0,
            "edits_in_window": edits,
        }
