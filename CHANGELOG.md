# Changelog

## [Unreleased]

### Added

- **Member-erasure retrain hooks (cloud).** `TrainingScheduler.force_retrain()` / `StreamWorker.force_retrain()` retrain a stream now, ignoring the new-task threshold, reading only what the store and the retained mirror hold at call time; `busy` / `training_in_progress` / `ready` let a host know when a stopped stream's last write has landed. `Trainer.train_all()` reports `"synthetic"`: the models fitted on synthetic data because the stream was below the model's minimum. `S3ModelStore.list_models()`, `delete_model()` and `purge_versions_before()` remove model versions for good in a versioned bucket, never touching the current version.

### Changed

- **2026-05-16 — Workbench host-rendered pivot.** The workbench programme pivoted from in-VM rendered UI to host-rendered. Per the ADR §"Affected Specs / kenaz-ml", kenaz-ml moves host-side: per-workbench inference runs on the host against the host-primary `sigild` ledger; no in-VM ML sidecar. Canonical day-to-day plan: [`PIVOT_PLAN.md`](../PIVOT_PLAN.md) in the workspace repo. Architectural rationale: [ADR-workbench-host-rendered-pivot](../.specify/decisions/ADR-workbench-host-rendered-pivot.md). Pre-pivot state preserved at tag `v1.0.0-pivot-baseline` for rollback.
