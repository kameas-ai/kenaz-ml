# Changelog

## [Unreleased]

### Changed

- **2026-05-16 — Workbench host-rendered pivot.** The workbench programme pivoted from in-VM rendered UI to host-rendered. Per the ADR §"Affected Specs / kenaz-ml", kenaz-ml moves host-side: per-workbench inference runs on the host against the host-primary `sigild` ledger; no in-VM ML sidecar. Canonical day-to-day plan: [`PIVOT_PLAN.md`](../PIVOT_PLAN.md) in the workspace repo. Architectural rationale: [ADR-workbench-host-rendered-pivot](../.specify/decisions/ADR-workbench-host-rendered-pivot.md). Pre-pivot state preserved at tag `v1.0.0-pivot-baseline` for rollback.
