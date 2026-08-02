# Tasks: kenaz-ml Rebrand

**Mission**: `kenaz-ml-rebrand-01KZ1AMQ`
**Branch**: `main` (planning base and merge target) · **Change mode**: `bulk_edit`
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Research**: [research.md](./research.md)
**Generated**: 2026-08-02

## Overview

11 subtasks across **2** work packages, strictly sequential.

Two packages rather than three, deliberately. The storage-layer reorganization was planned as four and ownership validation rejected it: a bulk edit rewrites `tests/**`, so a separate verification package would need to own files the rename package must also touch. The rename and its verification are one unit of work. That mission's lesson is applied here rather than rediscovered.

**The occurrence map comes before any file changes** (C-005, D-006). It is the guardrail and the review artifact, and this mission's map has one job no previous map had: separating **this product** from the **Sigil daemon** it integrates with.

## Subtask Index

| ID | Description | WP |
|---|---|---|
| T001 | Enumerate occurrences across all 8 bulk-edit categories | WP01 | [D] |
| T002 | Classify every `sigil` hit: this product vs the Sigil daemon | WP01 | [D] |
| T003 | Enumerate string-resolved and grep-invisible references | WP01 | [D] |
| T004 | Re-probe serialized artifacts; reconcile counts | WP01 | [D] |
| T005 | `git mv` the package directory | WP02 |
| T006 | Rewrite imports and intra-package references | WP02 |
| T007 | `pyproject.toml` — distribution, packages, entry point, URLs | WP02 |
| T008 | String-resolved references: uvicorn target, PyInstaller, freeze spec filename | WP02 |
| T009 | Log prefixes, docs, `CLAUDE.md`, `README`, `CHANGELOG`, CI, `Makefile` | WP02 |
| T010 | Preserve the frozen-history alias key; verify the Sigil surface is untouched | WP02 |
| T011 | Verification tests and the map cross-check | WP02 |

---

## WP01 — Occurrence Map

**Prompt**: [tasks/WP01-occurrence-map.md](./tasks/WP01-occurrence-map.md)
**Priority**: P1 · **Dependencies**: none · **Estimated prompt size**: ~220 lines

**Goal**: An exhaustive, reviewed inventory of all 1,738 `sigil_ml` and 420 `kameas-ml` occurrences — and, critically, a classification of every `sigil` hit as either this product or the Sigil daemon.

**Independent test**: Every one of the 138 Sigil-daemon references is classified `do_not_change` with a reason, and the map's `rename` count reconciles against the measured totals.

**Included subtasks**:

- [x] T001 Enumerate occurrences across all 8 bulk-edit categories (WP01)
- [x] T002 Classify every `sigil` hit: this product vs the Sigil daemon (WP01)
- [x] T003 Enumerate string-resolved and grep-invisible references (WP01)
- [x] T004 Re-probe serialized artifacts; reconcile counts (WP01)

**Risks**: T002 is the whole mission. A map that cannot tell `sigil_ml` (rename) from `sigild` (never rename) would authorise a change that points every install at a nonexistent database — and the code would still import cleanly, so it would look like it worked.

**Requirements**: NFR-003, C-005, FR-007

---

## WP02 — The Rename

**Prompt**: [tasks/WP02-the-rename.md](./tasks/WP02-the-rename.md)
**Priority**: P1 · **Dependencies**: WP01 · **Estimated prompt size**: ~330 lines

**Goal**: Execute the rename across every site the map lists, leave every site it marks `do_not_change` alone, and prove both.

**Independent test**: `import kenaz_ml` succeeds and `import sigil_ml` raises; the 138 Sigil-daemon references are unchanged by count; the suite passes with no regression; the frozen binary builds and serves a real prediction.

**Included subtasks**:

- [ ] T005 `git mv` the package directory (WP02)
- [ ] T006 Rewrite imports and intra-package references (WP02)
- [ ] T007 `pyproject.toml` — distribution, packages, entry point, URLs (WP02)
- [ ] T008 String-resolved references: uvicorn target, PyInstaller, freeze spec filename (WP02)
- [ ] T009 Log prefixes, docs, `CLAUDE.md`, `README`, `CHANGELOG`, CI, `Makefile` (WP02)
- [ ] T010 Preserve the frozen-history alias key; verify the Sigil surface is untouched (WP02)
- [ ] T011 Verification tests and the map cross-check (WP02)

**Risks**: A blanket `sigil` → `kenaz` substitution is the failure mode — fast, plausible, and catastrophic. T010 exists to make the Sigil surface an asserted invariant rather than a hope. The string-resolved references (T008) fail at runtime rather than at import, and the PyInstaller ones fail only in the packaged binary on first use.

**Requirements**: FR-001 … FR-013, NFR-001, NFR-002, NFR-004

---

## Dependencies

```
WP01 (occurrence map) → WP02 (the rename, and its verification)
```

Strictly sequential. WP01's output is WP02's checklist.

## MVP Scope

None. A partial rename leaves the codebase in a state neither name explains — worse than not starting. It lands whole or reverts whole.

## Out of Scope

- Renaming anything belonging to the Sigil daemon (C-002, FR-007)
- Renaming the GitHub organisation (C-003, D-007)
- Rewriting merged missions' specs (C-006, D-003, FR-013)
- Moving or restructuring modules — the storage reorganization settled the layout
- Changing model names in `ml_predictions.model`, which Go queries by exact string (C-004)
