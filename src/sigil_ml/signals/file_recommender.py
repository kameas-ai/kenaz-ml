"""File recommendation based on co-occurrence patterns within task sessions.

Builds a co-occurrence matrix from completed tasks. When a user edits
file A, recommends files B, C that are commonly edited alongside A.
Scoped to the current repository.
"""

from __future__ import annotations

import logging
import os
from collections import Counter, defaultdict
from typing import Any

from sigil_ml.modelstore import ModelStore
from sigil_ml.signals import Signal
from sigil_ml.signals.profile import BehaviorProfile

logger = logging.getLogger(__name__)

# Minimum completed tasks before recommendations are made.
MIN_TASKS_FOR_RECOMMENDATION = 5
# Minimum co-occurrence probability to trigger a recommendation.
MIN_COOCCURRENCE_PROB = 0.3
# Maximum number of files to recommend per signal.
MAX_RECOMMENDATIONS = 3


class FileRecommender:
    """Recommends related files based on co-occurrence in task sessions.

    Maintains a sparse co-occurrence matrix built from completed tasks.
    """

    def __init__(self) -> None:
        self._cooccurrence: dict[str, Counter[str]] = defaultdict(Counter)
        self._file_counts: Counter[str] = Counter()
        self._task_count: int = 0

    def check(
        self,
        buffer: list[dict],
        task_context: dict | None,
        profile: BehaviorProfile,
    ) -> list[Signal]:
        """Check for file recommendations based on current editing context.

        Args:
            buffer: Recent classified events from the poller.
            task_context: Dict with active task info (may include repo_root).
            profile: Current user behavior profile.

        Returns:
            List of Signal objects with file recommendations.
        """
        if self._task_count < MIN_TASKS_FOR_RECOMMENDATION:
            return []

        current_files = self._extract_current_files(buffer)
        if not current_files:
            return []

        repo_root = self._infer_repo_root(current_files)
        recommendations = self._recommend(current_files, repo_root)

        if not recommendations:
            return []

        return [self._build_signal(current_files, recommendations, repo_root)]

    # --- Co-occurrence matrix building ---

    def train_from_tasks(self, store: Any) -> int:
        """Build co-occurrence matrix from completed tasks.

        Queries the DataStore for all completed tasks and their events.
        Extracts file pairs edited within the same task.

        Args:
            store: DataStore instance.

        Returns:
            Number of tasks processed.
        """
        self._cooccurrence = defaultdict(Counter)
        self._file_counts = Counter()
        self._task_count = 0

        task_ids = store.get_completed_task_ids()
        for task_id in task_ids:
            events = store.get_events_for_task(task_id)
            files = self._extract_files_from_events(events)
            if len(files) < 2:
                continue

            self._task_count += 1
            for f in files:
                self._file_counts[f] += 1
                for g in files:
                    if f != g:
                        self._cooccurrence[f][g] += 1

        logger.info(
            "FileRecommender: built matrix from %d tasks, %d files",
            self._task_count,
            len(self._file_counts),
        )
        return self._task_count

    def _extract_files_from_events(self, events: list[dict]) -> set[str]:
        """Extract unique file paths from a list of events."""
        files: set[str] = set()
        for e in events:
            if e.get("kind") != "file":
                continue
            payload = e.get("payload") or {}
            if isinstance(payload, dict) and "path" in payload:
                path = str(payload["path"])
                if path:
                    files.add(path)
        return files

    # --- Conditional probability and recommendation ---

    def _recommend(self, current_files: set[str], repo_root: str | None) -> list[tuple[str, float]]:
        """Generate file recommendations based on conditional probabilities.

        Args:
            current_files: Set of files currently being edited.
            repo_root: Repository root path for scoping, or None.

        Returns:
            List of (file_path, score) tuples, sorted by score descending.
        """
        candidates: Counter = Counter()

        for f in current_files:
            co = self._cooccurrence.get(f)
            if co is None:
                continue
            f_count = self._file_counts.get(f, 1)

            for other, count in co.items():
                # Skip files already being edited
                if other in current_files:
                    continue
                # Scope to current repository
                if repo_root and not other.startswith(repo_root):
                    continue
                # Conditional probability: P(other | f)
                prob = count / f_count
                if prob >= MIN_COOCCURRENCE_PROB:
                    candidates[other] = max(candidates[other], prob)

        return candidates.most_common(MAX_RECOMMENDATIONS)

    # --- Helper methods ---

    def _extract_current_files(self, buffer: list[dict]) -> set[str]:
        """Extract files currently being edited from the event buffer."""
        files: set[str] = set()
        for e in buffer:
            if e.get("kind") != "file":
                continue
            payload = e.get("payload") or {}
            if isinstance(payload, dict) and "path" in payload:
                path = str(payload["path"])
                if path:
                    files.add(path)
        return files

    def _infer_repo_root(self, files: set[str]) -> str | None:
        """Infer the repository root from file paths.

        Looks for common path prefix and checks for .git directory markers.
        Falls back to longest common prefix of all files.
        """
        if not files:
            return None

        paths = sorted(files)
        # Use os.path.commonpath for the common directory prefix
        try:
            common = os.path.commonpath(paths)
            # Walk up to find a likely repo root (directory with common project markers)
            # For now, use the common path as repo root
            return common
        except ValueError:
            # Paths on different drives (Windows) or no common prefix
            return None

    def _build_signal(
        self,
        current_files: set[str],
        recommendations: list[tuple[str, float]],
        repo_root: str | None,
    ) -> Signal:
        """Build a file recommendation signal."""
        # Use the alphabetically last file as the trigger
        trigger_file = sorted(current_files)[-1] if current_files else "unknown"

        return Signal(
            signal_type="file_recommendation",
            confidence=round(recommendations[0][1], 4) if recommendations else 0.0,
            evidence={
                "source_model": "file_recommender",
                "current_file": trigger_file,
                "recommended_files": [
                    {"path": path, "co_occurrence": round(score, 4)} for path, score in recommendations
                ],
                "context": {
                    "repo": repo_root,
                    "files_being_edited": len(current_files),
                },
            },
            suggested_action="investigate",
        )

    # --- Model persistence ---

    def save(self, model_store: ModelStore) -> None:
        """Persist co-occurrence matrix via ModelStore."""
        import io

        import joblib

        data = {
            "cooccurrence": dict(self._cooccurrence),
            "file_counts": dict(self._file_counts),
            "task_count": self._task_count,
        }
        buf = io.BytesIO()
        joblib.dump(data, buf)
        model_store.save("file_recommender", buf.getvalue())
        logger.info(
            "FileRecommender: saved matrix with %d files, %d tasks",
            len(self._file_counts),
            self._task_count,
        )

    def load(self, model_store: ModelStore) -> bool:
        """Load co-occurrence matrix from ModelStore. Returns True if loaded."""
        raw = model_store.load("file_recommender")
        if raw is None:
            return False
        import io

        import joblib

        try:
            data = joblib.load(io.BytesIO(raw))
            loaded_co = data.get("cooccurrence", {})
            self._cooccurrence = defaultdict(Counter)
            for f, counts in loaded_co.items():
                self._cooccurrence[f] = Counter(counts)
            self._file_counts = Counter(data.get("file_counts", {}))
            self._task_count = data.get("task_count", 0)
            logger.info(
                "FileRecommender: loaded matrix with %d files, %d tasks",
                len(self._file_counts),
                self._task_count,
            )
            return True
        except Exception:
            logger.warning("FileRecommender: failed to load model", exc_info=True)
            return False

    # --- Serialization helpers (dict-based, no ModelStore) ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize co-occurrence state to a JSON-compatible dict."""
        return {
            "cooccurrence": {f: dict(counts) for f, counts in self._cooccurrence.items()},
            "file_counts": dict(self._file_counts),
            "task_count": self._task_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FileRecommender:
        """Restore from a serialized dict."""
        rec = cls()
        rec._task_count = data.get("task_count", 0)
        rec._file_counts = Counter(data.get("file_counts", {}))
        loaded_co = data.get("cooccurrence", {})
        for f, counts in loaded_co.items():
            rec._cooccurrence[f] = Counter(counts)
        return rec
