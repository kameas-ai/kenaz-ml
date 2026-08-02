"""Data models for the cloud training pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Valid TrainingRun statuses
STATUS_TRAINED = "trained"
STATUS_SKIPPED = "skipped"
STATUS_SKIPPED_LOCKED = "skipped_locked"
STATUS_FAILED = "failed"


@dataclass
class CloudTrainingConfig:
    """Configuration for cloud training runs."""

    min_interval_sec: int = 3600
    min_tasks: int = 10
    max_tasks_per_tenant: int = 1000
    aggregate_min_tenants: int = 3


@dataclass
class TrainingRun:
    """Result of training models for a single tenant or aggregate pool."""

    tenant_id: str
    status: str  # "trained", "failed", "skipped", "skipped_locked"
    models_trained: list[str] = field(default_factory=list)
    sample_count: int = 0
    duration_ms: int = 0
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    data_freshness_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        d: dict[str, Any] = {
            "tenant_id": self.tenant_id,
            "status": self.status,
            "models_trained": self.models_trained,
            "sample_count": self.sample_count,
            "duration_ms": self.duration_ms,
        }
        if self.error is not None:
            d["error"] = self.error
        if self.started_at is not None:
            d["started_at"] = self.started_at.isoformat()
        if self.completed_at is not None:
            d["completed_at"] = self.completed_at.isoformat()
        if self.data_freshness_sec is not None:
            d["data_freshness_sec"] = self.data_freshness_sec
        return d

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class TrainingBatch:
    """Aggregated result of batch training across multiple tenants."""

    runs: list[TrainingRun] = field(default_factory=list)
    total_duration_ms: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def trained(self) -> int:
        """Count of successfully trained runs."""
        return sum(1 for r in self.runs if r.status == STATUS_TRAINED)

    @property
    def skipped(self) -> int:
        """Count of skipped runs (including skipped_locked)."""
        return sum(1 for r in self.runs if r.status.startswith("skipped"))

    @property
    def failed(self) -> int:
        """Count of failed runs."""
        return sum(1 for r in self.runs if r.status == STATUS_FAILED)

    @property
    def total(self) -> int:
        """Total number of runs."""
        return len(self.runs)

    @property
    def status_breakdown(self) -> dict[str, int]:
        """Count runs by status for monitoring."""
        counts: dict[str, int] = {}
        for run in self.runs:
            counts[run.status] = counts.get(run.status, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        d: dict[str, Any] = {
            "total": self.total,
            "trained": self.trained,
            "skipped": self.skipped,
            "failed": self.failed,
            "status_breakdown": self.status_breakdown,
            "total_duration_ms": self.total_duration_ms,
            "runs": [r.to_dict() for r in self.runs],
        }
        if self.started_at is not None:
            d["started_at"] = self.started_at.isoformat()
        if self.completed_at is not None:
            d["completed_at"] = self.completed_at.isoformat()
        return d

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class TrainingSummary:
    """Human-readable summary for CLI output."""

    mode: str  # "single", "batch", "aggregate"
    total_tenants: int = 0
    trained: int = 0
    skipped: int = 0
    failed: int = 0
    total_samples: int = 0
    total_duration_ms: int = 0
    per_tenant: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "mode": self.mode,
            "total_tenants": self.total_tenants,
            "trained": self.trained,
            "skipped": self.skipped,
            "failed": self.failed,
            "total_samples": self.total_samples,
            "total_duration_ms": self.total_duration_ms,
            "per_tenant": self.per_tenant,
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
