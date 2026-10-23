# Changelog

## [Unreleased]

### Added

- **Member-erasure retrain hooks (cloud).** `TrainingScheduler.force_retrain()` / `StreamWorker.force_retrain()` retrain a stream now, ignoring the new-task threshold, reading only what the store and the retained mirror hold at call time; `busy` / `training_in_progress` / `ready` let a host know when a stopped stream's last write has landed. `Trainer.train_all()` reports `"synthetic"`: the models fitted on synthetic data because the stream was below the model's minimum. `S3ModelStore.list_models()`, `delete_model()` and `purge_versions_before()` remove model versions for good in a versioned bucket, never touching the current version.

### Fixed

- **Models train on a fresh install, and training progress survives restarts.** The training scheduler trains at its first check when `stuck` or `duration` has no weights in the local or base slot (bootstrapping from synthetic data below the minimum), instead of waiting for 10 new completed tasks. The new-task baseline is rebuilt at startup from the last `ml_events` retrain row (tasks completed at or before it), so a restart no longer resets it. `POST /train` now runs through the scheduler, so a manual retrain logs that row and resets the baseline too.
- **`/introspect` reports training progress and honest counts.** New `training` block (`completed_tasks`, `baseline`, `threshold`, `last_result`, `last_retrain_at`, `next_check_at`, `in_progress`; null in cloud mode). `recent_predictions` counts rows created in the last 24 h whether or not they have expired (the poller's 90 s TTL hid almost all of them). `sample_count` is the serving manifest's `training.n_samples`, null when absent.
- **Poller failures are visible.** A failing poll logs a WARN (exception type only) at the start of a streak and every 5 minutes while it lasts, and an INFO on recovery; it was DEBUG.

### Changed

- **2026-05-16 — Workbench host-rendered pivot.** The workbench programme pivoted from in-VM rendered UI to host-rendered. Per the ADR §"Affected Specs / kenaz-ml", kenaz-ml moves host-side: per-workbench inference runs on the host against the host-primary `sigild` ledger; no in-VM ML sidecar. Canonical day-to-day plan: [`PIVOT_PLAN.md`](../PIVOT_PLAN.md) in the workspace repo. Architectural rationale: [ADR-workbench-host-rendered-pivot](../.specify/decisions/ADR-workbench-host-rendered-pivot.md). Pre-pivot state preserved at tag `v1.0.0-pivot-baseline` for rollback.
