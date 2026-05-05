---
work_package_id: "WP05"
title: "Aggregate Model Training"
lane: "planned"
dependencies: ["WP02"]
subtasks:
  - "T023"
  - "T024"
  - "T025"
  - "T026"
  - "T027"
  - "T028"
phase: "Phase 3 - Advanced Features"
assignee: ""
agent: ""
shell_pid: ""
review_status: ""
reviewed_by: ""
requirement_refs:
  - "FR-003"
  - "FR-010"
history:
  - timestamp: "2026-03-30T01:45:09Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt regenerated via /spec-kitty.tasks"
---

# Work Package Prompt: WP05 -- Aggregate Model Training

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
spec-kitty implement WP05 --base WP02
```

Depends on WP02 (per-tenant training logic -- reuses model training and feature extraction functions). Does NOT depend on WP03 or WP04.

---

## Objectives & Success Criteria

1. `CloudTrainer.train_aggregate()` pools events from all opted-in tenants and trains shared aggregate models.
2. Only data from explicitly opted-in tenants is included (FR-010) -- opt-in checked at query time, never cached.
3. A sampling/weighting strategy caps each tenant's contribution at `max_tasks_per_tenant` (default: 1000) to prevent large tenants from dominating.
4. Aggregate weights are saved via ModelStore to a shared `__aggregate__` tenant prefix.
5. A warning is logged (but training proceeds) if fewer than `aggregate_min_tenants` (default: 3) tenants are opted in.
6. CLI `--mode cloud --aggregate` invokes aggregate training and prints a structured JSON summary.
7. Zero opted-in tenants returns a clean result (not an error crash).

## Context & Constraints

- **Spec**: User Story 3 (Aggregate Model Training), FR-003, FR-010
- **WP02 artifacts**: Feature extraction functions (`extract_stuck_features_from_data`, `extract_duration_features_from_data`), model training flow, `_save_model_to_store()` helper.
- **Privacy**: The opt-in flag MUST be checked at query time. Only tenants who have explicitly opted in to data pooling contribute. This is the key Team-tier privacy boundary.
- **Data volume**: Pooled data could be large (thousands of tasks across many tenants). Sampling per tenant (T025) controls memory usage.
- **Model reuse**: Same 5 model types, same training logic -- only the data source differs (pooled vs single-tenant).
- **Storage prefix**: Aggregate models use `tenant_id="__aggregate__"` as the ModelStore namespace, separate from per-tenant prefixes.

### Key Architecture Decision

Aggregate training produces model weights that the prediction API can load alongside per-user weights for blending. The blending logic is NOT part of this WP -- it belongs to the prediction serving layer. This WP only produces the aggregate model weights in the correct storage location.

---

## Subtasks & Detailed Guidance

### Subtask T023 -- Opt-In Tenant Discovery

- **Purpose**: Query the DataStore for tenants who have explicitly opted in to aggregate data pooling.
- **Steps**:
  1. Add to `src/sigil_ml/training/tenant_discovery.py`:
     ```python
     def discover_opted_in_tenants(data_store: "DataStore") -> list[str]:
         """Discover tenants that have opted in to aggregate data pooling.

         Returns only tenant IDs where data_pooling_opted_in flag is True.
         The query is always fresh (not cached) to respect opt-out changes.

         Returns:
             List of opted-in tenant ID strings.
         """
         tenants = data_store.list_opted_in_tenants()
         logger.info("Found %d opted-in tenants for aggregate training", len(tenants))
         return tenants
     ```
  2. Ensure the DataStore Protocol stub includes:
     ```python
     def list_opted_in_tenants(self) -> list[str]:
         """Return tenant IDs that have opted in to aggregate data pooling."""
         ...
     ```
  3. In `CloudTrainer`, add a private discovery method:
     ```python
     def _discover_opted_in_tenants(self) -> list[str]:
         from sigil_ml.training.tenant_discovery import discover_opted_in_tenants
         return discover_opted_in_tenants(self.data_store)
     ```
  4. The opt-in status comes from the tenant's configuration in the database. The Go side (sigild) manages the flag. Python only reads it.
  5. The query MUST be fresh (not cached) to respect real-time opt-out changes.
- **Files**: `src/sigil_ml/training/tenant_discovery.py` (extend, ~15 lines)
- **Parallel?**: No -- feeds into T024.
- **Validation**:
  - [ ] Only opted-in tenants are returned
  - [ ] Tenants who opt out between runs are excluded
  - [ ] Empty list handled gracefully (no crash)

### Subtask T024 -- Data Pooling Across Tenants

- **Purpose**: Fetch completed tasks and events from all opted-in tenants and combine into a single training dataset.
- **Steps**:
  1. In `CloudTrainer`, add a data pooling method:
     ```python
     def _pool_training_data(
         self, tenant_ids: list[str]
     ) -> tuple[list[dict], dict[str, list[dict]], dict[str, int]]:
         """Pool training data from multiple opted-in tenants.

         Returns:
             (all_tasks, task_events, tenant_task_counts)
             - all_tasks: list of task dicts, each tagged with _tenant_id
             - task_events: dict mapping task_id -> list of event dicts
             - tenant_task_counts: dict mapping tenant_id -> task count before sampling
         """
         all_tasks: list[dict] = []
         task_events: dict[str, list[dict]] = {}
         tenant_counts: dict[str, int] = {}

         for tenant_id in tenant_ids:
             tasks = self.data_store.query_completed_tasks(tenant_id)
             tenant_counts[tenant_id] = len(tasks)

             for task in tasks:
                 task["_tenant_id"] = tenant_id  # Tag with source tenant
                 events = self.data_store.query_events_for_task(tenant_id, task["id"])
                 all_tasks.append(task)
                 task_events[task["id"]] = events

         logger.info(
             "Pooled %d total tasks from %d tenants (before sampling)",
             len(all_tasks), len(tenant_ids),
         )
         return all_tasks, task_events, tenant_counts
     ```
  2. Each task is tagged with `_tenant_id` for sampling attribution (T025) and debugging.
  3. Memory consideration: loading all tasks into memory is acceptable when combined with the per-tenant sampling cap (T025). For the initial implementation, this is sufficient.
- **Files**: `src/sigil_ml/training/cloud_trainer.py`
- **Parallel?**: No -- sequential data loading per tenant.
- **Validation**:
  - [ ] Data from all opted-in tenants is included
  - [ ] Each task is tagged with `_tenant_id`
  - [ ] `tenant_counts` accurately reflects per-tenant contribution before sampling
  - [ ] Total task count is logged

### Subtask T025 -- Sampling/Weighting Strategy

- **Purpose**: Prevent large tenants from dominating the aggregate model by capping each tenant's contribution at a configurable maximum.
- **Steps**:
  1. In `CloudTrainer`, add a sampling method:
     ```python
     def _sample_pooled_data(
         self, all_tasks: list[dict], tenant_counts: dict[str, int],
     ) -> list[dict]:
         """Apply per-tenant sampling caps to pooled data.

         Each tenant contributes at most max_tasks_per_tenant tasks.
         If a tenant has fewer, all their tasks are included.
         Sampling is deterministic (seeded RNG) for reproducibility.

         Returns:
             Sampled task list.
         """
         import random
         max_per = self.config.max_tasks_per_tenant
         rng = random.Random(42)  # deterministic for reproducibility

         sampled: list[dict] = []
         for tenant_id, count in tenant_counts.items():
             tenant_tasks = [t for t in all_tasks if t.get("_tenant_id") == tenant_id]

             if len(tenant_tasks) > max_per:
                 logger.info(
                     "Sampling %d/%d tasks from tenant %s (cap: %d)",
                     max_per, len(tenant_tasks), tenant_id, max_per,
                 )
                 tenant_tasks = rng.sample(tenant_tasks, max_per)

             sampled.extend(tenant_tasks)

         logger.info(
             "Aggregate dataset after sampling: %d tasks from %d tenants",
             len(sampled), len(tenant_counts),
         )
         return sampled
     ```
  2. `max_tasks_per_tenant` is configurable via `CloudTrainingConfig` (default: 1000, overridable via `SIGIL_ML_TRAIN_MAX_TASKS_PER_TENANT` env var or `--max-tasks-per-tenant` CLI arg).
  3. Random seed (42) ensures deterministic sampling for debugging and reproducibility.
- **Files**: `src/sigil_ml/training/cloud_trainer.py`
- **Parallel?**: Independent design decision, can be developed alongside T023.
- **Edge Cases**:
  - Tenant with 5000 tasks: capped at 1000 (sampled randomly)
  - Tenant with 50 tasks: all 50 included (below cap)
  - All tenants below cap: no sampling needed, all data used
- **Validation**:
  - [ ] Tenant with 5000 tasks contributes exactly `max_tasks_per_tenant`
  - [ ] Tenant with 50 tasks contributes all 50
  - [ ] Sampling is deterministic (same results on re-run with same data)
  - [ ] Total sample size is logged

### Subtask T026 -- Implement train_aggregate()

- **Purpose**: The main aggregate training orchestrator: discover opted-in tenants, pool data, apply sampling, train all models, save weights to shared prefix.
- **Steps**:
  1. Implement in `CloudTrainer`:
     ```python
     AGGREGATE_TENANT_ID = "__aggregate__"

     def train_aggregate(self) -> TrainingRun:
         """Train aggregate models from pooled opted-in tenant data.

         Returns a TrainingRun with tenant_id="__aggregate__".
         """
         start = time.time()

         # 1. Discover opted-in tenants
         tenant_ids = self._discover_opted_in_tenants()

         # 2. Threshold check (T028)
         if len(tenant_ids) < self.config.aggregate_min_tenants:
             logger.warning(
                 "Only %d opted-in tenants (recommended minimum: %d). "
                 "Aggregate model may be unreliable.",
                 len(tenant_ids), self.config.aggregate_min_tenants,
             )

         if not tenant_ids:
             return TrainingRun(
                 tenant_id=AGGREGATE_TENANT_ID,
                 status="skipped",
                 error="No opted-in tenants found",
                 duration_ms=int((time.time() - start) * 1000),
             )

         # 3. Pool data from all opted-in tenants
         all_tasks, task_events, tenant_counts = self._pool_training_data(tenant_ids)

         # 4. Apply sampling strategy
         sampled_tasks = self._sample_pooled_data(all_tasks, tenant_counts)
         total_samples = len(sampled_tasks)

         # 5. Extract features and train all models
         models_trained = self._train_models_from_tasks(
             sampled_tasks, task_events, tenant_id=AGGREGATE_TENANT_ID,
         )

         # 6. Record audit event
         elapsed_ms = int((time.time() - start) * 1000)
         self.data_store.record_training_event(AGGREGATE_TENANT_ID, {
             "kind": "aggregate_training",
             "tenants_pooled": len(tenant_ids),
             "sample_count": total_samples,
             "models_trained": models_trained,
             "duration_ms": elapsed_ms,
             "ts": int(time.time() * 1000),
         })

         return TrainingRun(
             tenant_id=AGGREGATE_TENANT_ID,
             status="trained",
             sample_count=total_samples,
             models_trained=models_trained,
             duration_ms=elapsed_ms,
         )
     ```
  2. Factor out shared training logic into a reusable method:
     ```python
     def _train_models_from_tasks(
         self,
         tasks: list[dict],
         task_events: dict[str, list[dict]],
         tenant_id: str,
     ) -> list[str]:
         """Train all model types from provided tasks and events.

         Shared by train_tenant() (single-tenant) and train_aggregate() (pooled).

         Returns:
             List of model names that were successfully trained.
         """
         # Feature extraction, X/y construction, training, saving
         # Same logic as in train_tenant() but factored out for reuse
     ```
  3. This refactoring means WP02's training logic in `train_tenant()` should be extractable into `_train_models_from_tasks()` so both per-tenant and aggregate paths use the same code.
  4. Save aggregate weights with `tenant_id=AGGREGATE_TENANT_ID` (`"__aggregate__"`).
- **Files**: `src/sigil_ml/training/cloud_trainer.py` (extend, ~60 lines)
- **Parallel?**: No -- core implementation.
- **Validation**:
  - [ ] Aggregate training uses data from ALL opted-in tenants
  - [ ] Weights saved with `tenant_id="__aggregate__"` via ModelStore
  - [ ] `TrainingRun.sample_count` reflects total after sampling
  - [ ] `TrainingRun.models_trained` lists all successfully trained models
  - [ ] Audit event records `tenants_pooled` count
  - [ ] Zero opted-in tenants returns `status="skipped"` (not crash)

### Subtask T027 -- Wire CLI --aggregate to Aggregate Training

- **Purpose**: Connect `--mode cloud --aggregate` to `CloudTrainer.train_aggregate()` and display the result.
- **Steps**:
  1. In `cli.py`, replace the WP01 placeholder:
     ```python
     elif args.aggregate:
         result = trainer.train_aggregate()
         print(result.to_json())
         sys.exit(0 if result.status != "failed" else 1)
     ```
  2. Exit code: 0 on success or skip, 1 on failure.
- **Files**: `src/sigil_ml/cli.py` (modify, ~5 lines)
- **Parallel?**: No -- depends on T026.
- **Validation**:
  - [ ] `kameas-ml train --mode cloud --aggregate` prints JSON result
  - [ ] Exit code 0 on successful aggregate training
  - [ ] Exit code 0 on skip (no opted-in tenants)
  - [ ] Exit code 1 on failure

### Subtask T028 -- Minimum Opt-In Threshold Check

- **Purpose**: Warn operators when too few tenants have opted in for meaningful aggregate models. Training still proceeds -- the warning is informational.
- **Steps**:
  1. Already integrated into T026's flow. The check uses `CloudTrainingConfig.aggregate_min_tenants` (default: 3).
  2. The warning is logged AND included in the `TrainingRun.error` field so it appears in structured output:
     ```python
     warning_msg = None
     if len(tenant_ids) < self.config.aggregate_min_tenants:
         warning_msg = (
             f"Only {len(tenant_ids)} opted-in tenants "
             f"(recommended minimum: {self.config.aggregate_min_tenants}). "
             f"Aggregate model may be unreliable."
         )
         logger.warning(warning_msg)

     # ... training proceeds ...

     run = TrainingRun(
         tenant_id=AGGREGATE_TENANT_ID,
         status="trained",
         error=warning_msg,  # Non-fatal warning included in output
         # ... other fields ...
     )
     ```
  3. The spec acceptance scenario: "Given only 2 tenants have opted in, the job completes but logs a warning that the dataset may be insufficient."
  4. Training proceeds normally even with the warning -- it does NOT fail or skip.
- **Files**: `src/sigil_ml/training/cloud_trainer.py`
- **Parallel?**: Yes -- can be developed alongside T026.
- **Validation**:
  - [ ] 2 opted-in tenants: warning logged, training proceeds, warning in JSON output
  - [ ] 5 opted-in tenants: no warning
  - [ ] 1 opted-in tenant: warning logged, training proceeds
  - [ ] 0 opted-in tenants: returns `status="skipped"` (handled in T026)

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Privacy violation (non-opted-in data included) | Low | Critical | Opt-in checked at query time via DataStore, never cached |
| Large tenant dominates aggregate model | Medium | Medium | Per-tenant sampling cap (T025) limits contribution |
| Memory pressure from pooled data | Low | Medium | Sampling reduces total size; cap per tenant |
| Aggregate model quality with few tenants | Medium | Low | Warning at threshold (T028); training still proceeds |
| Data leakage across tenants in model | N/A | Accepted | Aggregate models inherently blend cross-user patterns. By design for Team tier. |

## Review Guidance

- **Privacy is critical**: Verify at the code level that only opted-in tenant data is queried. Trace the flow: `discover_opted_in_tenants()` -> `DataStore.list_opted_in_tenants()` -> filtered query. No caching.
- **Sampling correctness**: Verify no tenant contributes more than `max_tasks_per_tenant`.
- **Storage prefix**: Verify aggregate weights use `__aggregate__` prefix, not a regular tenant prefix.
- **Code reuse**: Verify `_train_models_from_tasks()` is shared between `train_tenant()` and `train_aggregate()`.
- **Zero-tenant edge case**: Verify clean skip, not a crash.
- **Warning vs error**: Verify low-tenant-count warning does not prevent training.

---

## Activity Log

- 2026-03-30T01:45:09Z -- system -- lane=planned -- Prompt regenerated.
