---
work_package_id: WP06
title: Training Observability & Structured Output
lane: planned
dependencies:
- WP01
- WP03
- WP02
subtasks:
- T029
- T030
- T031
- T032
- T033
phase: Phase 3 - Polish
assignee: ''
agent: ''
shell_pid: ''
review_status: ''
reviewed_by: ''
history:
- timestamp: '2026-03-30T01:45:09Z'
  lane: planned
  agent: system
  shell_pid: ''
  action: Prompt regenerated via /spec-kitty.tasks
requirement_refs:
- FR-008
- FR-009
---

# Work Package Prompt: WP06 -- Training Observability & Structured Output

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
spec-kitty implement WP06 --base WP03
```

Depends on WP01 (TrainingRun/TrainingSummary dataclasses) and WP03 (batch training produces TrainingBatch to format).

---

## Objectives & Success Criteria

1. Training output is structured JSON, parseable by monitoring systems (FR-008).
2. Per-tenant status includes: tenant_id, status, sample_count, models_trained, duration_ms, error (nullable).
3. Batch training produces a JSON summary with trained/skipped/failed breakdowns and per-tenant details.
4. Aggregate training produces a JSON summary with opt-in tenant count and total samples.
5. All training runs record audit events to `ml_training_runs` table via DataStore (FR-009).
6. CLI detects TTY vs pipe for output formatting: pretty-print for terminal, compact JSON for pipes.
7. A `--json` flag forces compact JSON regardless of TTY detection.

## Context & Constraints

- **Spec**: User Story 5 (Training Observability), FR-008, FR-009
- **WP01 artifacts**: `TrainingRun` and `TrainingBatch` dataclasses with `to_dict()` and `to_json()` methods.
- **WP03 artifacts**: `train_all_tenants()` returns a `TrainingBatch`.
- **Existing pattern**: `TrainingScheduler._log_retrain()` in `training/scheduler.py` records audit events to the `ml_events` table. Cloud training should follow a compatible pattern.
- **Output strategy**: Structured JSON goes to stdout; log/diagnostic messages go to stderr. This allows piping: `kameas-ml train ... | jq .` while still seeing operational logs.
- **JSON Lines**: For K8s log collection, cloud training should emit JSON Lines format (one JSON object per line) for real-time streaming events.

---

## Subtasks & Detailed Guidance

### Subtask T029 -- Enhance TrainingRun and TrainingBatch Dataclasses

- **Purpose**: Ensure the dataclasses contain all fields needed for comprehensive observability output per FR-008.
- **Steps**:
  1. Review and enhance `TrainingRun` in `training/models.py`:
     ```python
     @dataclass
     class TrainingRun:
         tenant_id: str
         status: str
         models_trained: list[str] = field(default_factory=list)
         sample_count: int = 0
         duration_ms: int = 0
         error: str | None = None
         started_at: datetime | None = None
         completed_at: datetime | None = None
         # Observability additions:
         data_freshness_sec: float | None = None  # seconds since newest training event
     ```
  2. Enhance `TrainingBatch`:
     ```python
     @dataclass
     class TrainingBatch:
         runs: list[TrainingRun] = field(default_factory=list)
         total_duration_ms: int = 0
         started_at: datetime | None = None
         completed_at: datetime | None = None

         @property
         def status_breakdown(self) -> dict[str, int]:
             """Count runs by status for monitoring dashboards."""
             counts: dict[str, int] = {}
             for run in self.runs:
                 counts[run.status] = counts.get(run.status, 0) + 1
             return counts
     ```
  3. Update `to_dict()` methods to include new fields:
     ```python
     # TrainingRun.to_dict():
     if self.data_freshness_sec is not None:
         d["data_freshness_sec"] = self.data_freshness_sec

     # TrainingBatch.to_dict():
     d["status_breakdown"] = self.status_breakdown
     ```
  4. Ensure ISO 8601 timestamps in JSON output for `started_at`/`completed_at`.
- **Files**: `src/sigil_ml/training/models.py` (modify, ~20 lines)
- **Parallel?**: Yes -- dataclass changes only, can proceed alongside T030-T032.
- **Validation**:
  - [ ] All fields serialize to JSON correctly
  - [ ] `status_breakdown` accurately reflects run statuses
  - [ ] ISO 8601 timestamps present in serialized output
  - [ ] `data_freshness_sec` appears only when set (not null in JSON unless present)

### Subtask T030 -- JSON Lines Event Emitter for Single-Tenant Training

- **Purpose**: When `--mode cloud --tenant <id>` is used, print structured JSON to stdout showing the complete training result. Support TTY detection and `--json` flag.
- **Steps**:
  1. Add `--json` flag to the train subparser (if not already added in WP01):
     ```python
     train_parser.add_argument(
         "--json", action="store_true", default=False,
         help="Force compact JSON output (default for non-TTY)"
     )
     ```
  2. In the CLI handler for `--tenant`:
     ```python
     if args.tenant:
         result = trainer.train_tenant(args.tenant)

         if sys.stdout.isatty() and not getattr(args, 'json', False):
             # Pretty-print for terminal
             print(json.dumps(result.to_dict(), indent=2))
         else:
             # Compact JSON for pipes / --json flag
             print(json.dumps(result.to_dict()))

         sys.exit(0 if result.status != "failed" else 1)
     ```
  3. The output must include ALL FR-008 fields: tenant_id, status, sample_count, models_trained, duration_ms, error_message (when applicable).
- **Files**: `src/sigil_ml/cli.py` (modify, ~15 lines)
- **Parallel?**: No -- modifies CLI output path.
- **Validation**:
  - [ ] Output is valid JSON (parseable by `jq`)
  - [ ] Contains all FR-008 fields
  - [ ] Pretty-printed when running in terminal
  - [ ] Compact when piped: `kameas-ml train ... | jq .`
  - [ ] `--json` flag forces compact JSON even in terminal

### Subtask T031 -- Structured JSON Summary for Batch Training

- **Purpose**: When `--mode cloud --all-tenants` is used, print a structured JSON summary with per-tenant details and aggregate counts.
- **Steps**:
  1. Enhance the CLI output in the `--all-tenants` handler:
     ```python
     elif args.all_tenants:
         batch = trainer.train_all_tenants()

         if sys.stdout.isatty() and not getattr(args, 'json', False):
             # Human-readable header for terminal
             print(f"\n=== Batch Training Summary ===")
             print(f"Total tenants: {batch.total}")
             print(f"  Trained: {batch.trained}")
             print(f"  Skipped: {batch.skipped}")
             print(f"  Failed:  {batch.failed}")
             print(f"Duration: {batch.total_duration_ms}ms")
             if batch.failed > 0:
                 print(f"\nFailed tenants:")
                 for run in batch.runs:
                     if run.status == "failed":
                         print(f"  - {run.tenant_id}: {run.error}")
             print(f"\nFull JSON:")
             print(json.dumps(batch.to_dict(), indent=2))
         else:
             print(json.dumps(batch.to_dict()))

         sys.exit(0 if batch.failed == 0 else 1)
     ```
  2. The JSON output must include:
     - `total`, `trained`, `skipped`, `failed` counts
     - `status_breakdown` with per-status counts
     - `total_duration_ms`
     - `runs` array with full per-tenant `TrainingRun` details
  3. Human-readable header provides a quick overview; full JSON follows for copy-paste.
- **Files**: `src/sigil_ml/cli.py` (modify, ~25 lines)
- **Parallel?**: No -- modifies CLI output path.
- **Validation**:
  - [ ] JSON output includes `runs` array with all per-tenant results
  - [ ] `status_breakdown` shows counts per status type
  - [ ] Human-readable header in terminal mode includes failed tenant details
  - [ ] Compact JSON when piped
  - [ ] Exit code follows strategy: 0 = no failures, 1 = any failure

### Subtask T032 -- Structured JSON Output for Aggregate Training

- **Purpose**: When `--mode cloud --aggregate` is used, print a structured summary showing aggregate training results.
- **Steps**:
  1. Enhance the CLI output in the `--aggregate` handler:
     ```python
     elif args.aggregate:
         result = trainer.train_aggregate()

         if sys.stdout.isatty() and not getattr(args, 'json', False):
             print(f"\n=== Aggregate Training Summary ===")
             print(f"Status: {result.status}")
             print(f"Samples: {result.sample_count}")
             print(f"Models trained: {', '.join(result.models_trained) or 'none'}")
             print(f"Duration: {result.duration_ms}ms")
             if result.error:
                 print(f"Note: {result.error}")
             print(f"\nFull JSON:")
             print(json.dumps(result.to_dict(), indent=2))
         else:
             print(json.dumps(result.to_dict()))

         sys.exit(0 if result.status != "failed" else 1)
     ```
  2. The output should clearly show:
     - Whether aggregate training succeeded or was skipped
     - How many samples were used (after sampling)
     - Which models were trained
     - Any warnings (insufficient tenants)
- **Files**: `src/sigil_ml/cli.py` (modify, ~15 lines)
- **Parallel?**: Yes -- independent from T030/T031.
- **Validation**:
  - [ ] JSON output includes `sample_count`, `models_trained`, `status`
  - [ ] Warning appears when few tenants opted in (in `error` field)
  - [ ] Human-readable header in terminal mode
  - [ ] Compact JSON when piped

### Subtask T033 -- Audit Event Recording for All Training Modes

- **Purpose**: Ensure all training runs record structured audit events via DataStore (FR-009). Verify consistency across single-tenant, batch, and aggregate modes.
- **Steps**:
  1. **Single-tenant**: Already handled by T012 (WP02). Verify `record_training_event()` is called with:
     ```python
     {"kind": "training", "status": ..., "sample_count": ..., "models_trained": [...],
      "duration_ms": ..., "ts": ...}
     ```
  2. **Batch-level**: Add a batch summary audit event at the end of `train_all_tenants()`:
     ```python
     # At end of train_all_tenants():
     try:
         self.data_store.record_training_event("__batch__", {
             "kind": "batch_training",
             "total_tenants": batch.total,
             "trained": batch.trained,
             "skipped": batch.skipped,
             "failed": batch.failed,
             "duration_ms": batch.total_duration_ms,
             "ts": int(time.time() * 1000),
         })
     except Exception:
         logger.warning("Failed to record batch audit event", exc_info=True)
     ```
  3. **Aggregate**: Already handled by T026 (WP05). Verify `record_training_event()` is called with:
     ```python
     {"kind": "aggregate_training", "tenants_pooled": ..., "sample_count": ...,
      "models_trained": [...], "duration_ms": ..., "ts": ...}
     ```
  4. All audit recording wrapped in try/except -- audit failures must NEVER crash the training run.
  5. Event schema must be compatible with the existing `ml_events` table pattern from `TrainingScheduler._log_retrain()`:
     - `kind`: identifies the event type
     - `endpoint`: "cloud_trainer"
     - `routing`: tenant_id or "__batch__" or "__aggregate__"
     - `latency_ms`: duration
     - `ts`: unix epoch milliseconds
- **Files**: `src/sigil_ml/training/cloud_trainer.py` (modify, ~15 lines)
- **Parallel?**: Yes -- independent from output formatting.
- **Validation**:
  - [ ] Single-tenant training records an audit event (from WP02, verify still works)
  - [ ] Batch training records per-tenant events AND a batch-level summary event
  - [ ] Aggregate training records an aggregate-level event with `tenants_pooled`
  - [ ] Failed training still records an audit event with failure status
  - [ ] Audit recording failure does not crash the training run

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Output schema changes breaking monitoring | Medium | Medium | Document JSON schema clearly; version if needed |
| Mixing structured and log output | Low | Medium | stdout for structured data, stderr for logs (use logging module) |
| Large batch summaries overwhelming terminals | Low | Low | Human-readable header provides summary; full JSON follows |
| Audit event write failures | Low | Low | Catch and log; never crash the training run |

## Review Guidance

- **Valid JSON**: Run `kameas-ml train --mode cloud --all-tenants | jq .` and verify parseable output.
- **TTY detection**: Verify pretty-print in terminal, compact for pipes.
- **`--json` flag**: Verify it forces compact JSON even in terminal.
- **FR-008 completeness**: Verify per-tenant details include ALL required fields: tenant_id, status, sample_count, models_trained, duration_ms.
- **Audit events**: Verify recorded for all three modes (single, batch, aggregate).
- **Error resilience**: Verify audit recording failures are caught and logged, not propagated.
- **Exit codes**: 0 = success/skip, 1 = any failure. Consistent across all modes.

---

## Activity Log

- 2026-03-30T01:45:09Z -- system -- lane=planned -- Prompt regenerated.
