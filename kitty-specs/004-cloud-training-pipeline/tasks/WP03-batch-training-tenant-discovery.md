---
work_package_id: "WP03"
title: "Batch Training & Tenant Discovery"
lane: "planned"
dependencies: ["WP02"]
subtasks:
  - "T013"
  - "T014"
  - "T015"
  - "T016"
  - "T017"
phase: "Phase 2 - Story Delivery"
assignee: ""
agent: ""
shell_pid: ""
review_status: ""
reviewed_by: ""
requirement_refs:
  - "FR-002"
  - "FR-007"
  - "FR-008"
history:
  - timestamp: "2026-03-30T01:45:09Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt regenerated via /spec-kitty.tasks"
---

# Work Package Prompt: WP03 -- Batch Training & Tenant Discovery

## IMPORTANT: Review Feedback Status

**Read this first if you are implementing this task!**

- **Has review feedback?**: Check the `review_status` field above. If it says `has_feedback`, scroll to the **Review Feedback** section immediately.
- **You must address all feedback** before your work is complete.
- **Mark as acknowledged**: When you understand the feedback and begin addressing it, update `review_status: acknowledged` in the frontmatter.

---

## Review Feedback

> **Populated by `/spec-kitty.review`** -- Reviewers add detailed feedback here when work needs changes.

*[This section is empty initially.]*

---

## Markdown Formatting
Wrap HTML/XML tags in backticks: `` `<div>` ``, `` `<script>` ``
Use language identifiers in code blocks: ````python`, ````bash`

---

## Implementation Command

```bash
spec-kitty implement WP03 --base WP02
```

Depends on WP02 (per-tenant training logic -- `train_tenant()` must be fully functional).

---

## Objectives & Success Criteria

1. `CloudTrainer.train_all_tenants()` discovers all tenants with synced data and trains each sequentially.
2. Per-tenant failures are caught and logged without interrupting the batch (FR-007).
3. A `TrainingBatch` with accurate trained/skipped/failed counts and per-tenant details is returned.
4. The "nothing to train" case (zero tenants or all skipped) completes successfully with a clean summary -- no error raised.
5. CLI `--mode cloud --all-tenants` invokes batch training and prints structured JSON summary (FR-008).
6. Tenant discovery is implemented in a dedicated `tenant_discovery.py` module.

## Context & Constraints

- **Spec**: User Story 2 (All-Tenant Batch Training), FR-002, FR-007, FR-008
- **WP01/WP02 artifacts**: CloudTrainer with working `train_tenant()`, TrainingRun/TrainingBatch dataclasses
- **DataStore dependency**: Requires a `list_tenants()` method. If the DataStore protocol from feature 002 does not include this, extend the Protocol stub and document the extension.
- **Sequential execution**: Batch training processes tenants one at a time. No parallelism (future optimization). This ensures predictable resource usage and simplifies fault isolation.
- **Exit codes**: 0 = all tenants succeeded or were skipped; 1 = at least one tenant failed. This allows CronJob monitoring to detect partial failures.

---

## Subtasks & Detailed Guidance

### Subtask T013 -- Tenant Discovery from DataStore

- **Purpose**: Discover all tenant IDs that have synced data and are candidates for training. Implemented as a dedicated module for testability and reuse by WP05 (aggregate training).
- **Steps**:
  1. Create `src/sigil_ml/training/tenant_discovery.py`:
     ```python
     """Tenant discovery for cloud training pipeline."""

     from __future__ import annotations

     import logging
     from typing import TYPE_CHECKING

     if TYPE_CHECKING:
         from sigil_ml.training.cloud_trainer import DataStore

     logger = logging.getLogger(__name__)


     def discover_eligible_tenants(data_store: "DataStore") -> list[str]:
         """Discover all tenant IDs with synced data.

         Returns all tenant IDs from the DataStore. Eligibility filtering
         (threshold, interval) is handled by train_tenant() for each tenant.

         Returns:
             List of tenant ID strings.
         """
         tenants = data_store.list_tenants()
         logger.info("Discovered %d tenants with synced data", len(tenants))
         return tenants
     ```
  2. The simpler approach is to return ALL tenants and let `train_tenant()` handle per-tenant eligibility checks (threshold/interval). This keeps discovery simple and avoids duplicating filtering logic.
  3. Ensure the DataStore Protocol stub includes:
     ```python
     def list_tenants(self) -> list[str]:
         """Return all tenant IDs that have synced data."""
         ...
     ```
  4. In `CloudTrainer`, add a private method that delegates to the discovery function:
     ```python
     def _discover_tenants(self) -> list[str]:
         from sigil_ml.training.tenant_discovery import discover_eligible_tenants
         return discover_eligible_tenants(self.data_store)
     ```
- **Files**: `src/sigil_ml/training/tenant_discovery.py` (new, ~30 lines)
- **Parallel?**: No -- feeds into T014.
- **Validation**:
  - [ ] Returns a list of tenant ID strings
  - [ ] Empty list is handled (no tenants found, no crash)
  - [ ] Logs the count of discovered tenants
  - [ ] DataStore.list_tenants() is called exactly once

### Subtask T014 -- Implement train_all_tenants()

- **Purpose**: The core batch training method. Iterates all discovered tenants, calls `train_tenant()` for each, collects results into a `TrainingBatch`.
- **Steps**:
  1. Implement in `CloudTrainer`:
     ```python
     def train_all_tenants(self) -> TrainingBatch:
         """Discover and train all eligible tenants in batch.

         Each tenant is trained independently. Failures for individual
         tenants do not interrupt the batch (FR-007).

         Returns:
             TrainingBatch with per-tenant TrainingRun results.
         """
         start = time.time()
         tenants = self._discover_tenants()

         batch = TrainingBatch(
             started_at=datetime.now(timezone.utc),
         )

         for i, tenant_id in enumerate(tenants, 1):
             logger.info("Processing tenant %d/%d: %s", i, len(tenants), tenant_id)
             run = self._train_tenant_safe(tenant_id)  # T015
             batch.runs.append(run)

         batch.total_duration_ms = int((time.time() - start) * 1000)
         batch.completed_at = datetime.now(timezone.utc)

         logger.info(
             "Batch complete: %d trained, %d skipped, %d failed (of %d total) in %dms",
             batch.trained, batch.skipped, batch.failed, batch.total, batch.total_duration_ms,
         )
         return batch
     ```
  2. Edge cases:
     - **Zero tenants**: Returns a `TrainingBatch` with empty `runs`, all counts at 0 -- no crash.
     - **All tenants skipped** (recently trained): Returns batch with `trained=0, skipped=N` -- exit code 0.
     - **Mix of outcomes**: Accurate counts via `TrainingBatch` computed properties.
  3. Log progress per tenant (`"Processing tenant 3/10: abc-123"`) so operators can monitor long-running batches and tune CronJob timeouts.
- **Files**: `src/sigil_ml/training/cloud_trainer.py` (modify, ~30 lines)
- **Parallel?**: No -- core implementation.
- **Validation**:
  - [ ] 10 tenants, 8 eligible: batch.trained == 8, batch.skipped == 2
  - [ ] 0 tenants: batch.total == 0, all counts 0, no crash
  - [ ] All skipped: batch.trained == 0, batch.skipped == N, exit code 0
  - [ ] Progress is logged per tenant

### Subtask T015 -- Fault Isolation Per Tenant

- **Purpose**: Ensure one tenant's training failure does not prevent training of remaining tenants (FR-007). This is the critical resilience layer.
- **Steps**:
  1. Create `_train_tenant_safe()` wrapper in `CloudTrainer`:
     ```python
     def _train_tenant_safe(self, tenant_id: str) -> TrainingRun:
         """Train a tenant with full error isolation.

         Catches all exceptions, logs them, and returns a failed TrainingRun
         instead of propagating the error.
         """
         try:
             return self.train_tenant(tenant_id)
         except Exception as e:
             logger.error(
                 "Training failed for tenant %s: %s",
                 tenant_id, str(e),
                 exc_info=True,
             )
             # Truncate very long error messages
             error_msg = str(e)[:500] if len(str(e)) > 500 else str(e)
             return TrainingRun(
                 tenant_id=tenant_id,
                 status="failed",
                 error=error_msg,
             )
     ```
  2. Key requirements:
     - Catches `Exception` (not `BaseException` -- let `KeyboardInterrupt` and `SystemExit` propagate).
     - Logs the full traceback (`exc_info=True`) for debugging.
     - Returns a `TrainingRun` with `status="failed"` and the error message (truncated to 500 chars).
     - Never allows one tenant's failure to prevent processing of the next.
  3. The spec explicitly requires: "The failure is logged, that tenant is skipped, and remaining tenants continue training."
- **Files**: `src/sigil_ml/training/cloud_trainer.py` (modify, ~20 lines)
- **Parallel?**: No -- wraps `train_tenant()`.
- **Validation**:
  - [ ] If `train_tenant()` raises `ValueError` for tenant A, tenant B still trains
  - [ ] If `train_tenant()` raises `ConnectionError` for tenant A, tenant B still trains
  - [ ] Failed tenant appears in batch with `status="failed"` and error message
  - [ ] Full traceback appears in log output (stderr), not in structured output (stdout)
  - [ ] Very long error messages are truncated to 500 chars

### Subtask T016 -- Wire CLI --all-tenants to Batch Training

- **Purpose**: Connect `--mode cloud --all-tenants` to `CloudTrainer.train_all_tenants()` and display the structured summary.
- **Steps**:
  1. In `cli.py`, replace the WP01 placeholder:
     ```python
     elif args.all_tenants:
         batch = trainer.train_all_tenants()
         print(batch.to_json())
         # Exit code: 0 if no failures, 1 if any tenant failed
         sys.exit(0 if batch.failed == 0 else 1)
     ```
  2. Exit code strategy:
     - `0`: All tenants trained or skipped (no failures)
     - `1`: At least one tenant failed (CronJob monitoring can detect this)
  3. Consider adding a `--max-tenants` flag for partial batches:
     ```python
     train_parser.add_argument(
         "--max-tenants", type=int, default=None,
         help="Limit batch to first N tenants (for testing/gradual rollout)"
     )
     ```
     Then in the handler:
     ```python
     if args.max_tenants:
         tenants = tenants[:args.max_tenants]
     ```
- **Files**: `src/sigil_ml/cli.py` (modify, ~10 lines)
- **Parallel?**: No -- depends on T014.
- **Validation**:
  - [ ] `kenaz-ml train --mode cloud --all-tenants` prints JSON to stdout
  - [ ] Exit code 0 when all succeed or skip
  - [ ] Exit code 1 when any tenant fails
  - [ ] `--max-tenants 5` limits batch to 5 tenants

### Subtask T017 -- Extend TrainingBatch Dataclass

- **Purpose**: Ensure `TrainingBatch` from WP01 covers all batch-specific needs. Add any missing computed properties or serialization methods.
- **Steps**:
  1. Review `TrainingBatch` from `training/models.py` (WP01).
  2. Verify it has:
     - `runs: list[TrainingRun]` -- per-tenant results
     - `total_duration_ms: int` -- overall batch timing
     - `started_at` / `completed_at` timestamps
     - `trained`, `skipped`, `failed` computed properties
     - `to_dict()` and `to_json()` serialization
  3. If additional fields are needed, add them:
     ```python
     @property
     def status_breakdown(self) -> dict[str, int]:
         """Count runs by status for monitoring."""
         counts: dict[str, int] = {}
         for run in self.runs:
             counts[run.status] = counts.get(run.status, 0) + 1
         return counts
     ```
  4. Update `to_dict()` to include `status_breakdown`:
     ```python
     def to_dict(self) -> dict[str, Any]:
         return {
             "total": self.total,
             "trained": self.trained,
             "skipped": self.skipped,
             "failed": self.failed,
             "status_breakdown": self.status_breakdown,
             "total_duration_ms": self.total_duration_ms,
             "runs": [r.to_dict() for r in self.runs],
         }
     ```
- **Files**: `src/sigil_ml/training/models.py` (modify, ~15 lines)
- **Parallel?**: Yes -- dataclass changes only, can proceed alongside T013/T014.
- **Validation**:
  - [ ] `status_breakdown` correctly counts each status type
  - [ ] Serializes to JSON with all batch-specific fields
  - [ ] Compatible with `train_all_tenants()` output

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| DataStore.list_tenants() not available | Medium | Medium | Add to Protocol stub; document expected return value |
| Batch takes longer than CronJob timeout | Low | Medium | Log per-tenant progress; add --max-tenants for partial batches |
| One tenant corrupts shared state | Low | High | Each train_tenant() call is fully isolated; no shared mutable state |
| Memory pressure from many tenants | Low | Low | Sequential processing; model objects GC'd between tenants |

## Review Guidance

- **Fault isolation is critical**: Mentally simulate DataStore.query_completed_tasks() raising for one tenant. Verify the batch survives.
- **Zero-tenant edge case**: Verify no crash and clean summary output.
- **Exit codes**: Verify 0 for all-skip, 1 for any failure.
- **Progress logging**: Verify per-tenant progress is logged (operators need this for long batches).
- **Tenant discovery module**: Verify it's testable independently (accepts DataStore, returns list).
- **No over-filtering**: Discovery returns all tenants; eligibility filtering happens in train_tenant().

---

## Activity Log

- 2026-03-30T01:45:09Z -- system -- lane=planned -- Prompt regenerated.
