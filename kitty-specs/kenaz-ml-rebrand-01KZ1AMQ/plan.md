# Implementation Plan: kenaz-ml Rebrand

**Branch**: `main` (planning base and merge target) | **Date**: 2026-08-02 | **Spec**: [spec.md](./spec.md)
**Change mode**: `bulk_edit` — an occurrence map is required before implementation (C-005)

## Summary

Rename the package `sigil_ml` → `kenaz_ml` and the distribution and CLI `kameas-ml` → `kenaz-ml`, plus URLs, log prefixes and documentation. Leave the Sigil daemon integration entirely alone.

1,738 occurrences across 165 files. No behaviour change.

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**: unchanged — no additions (NFR-002)
**Storage**: unchanged. Critically, `~/.local/share/sigild/` belongs to the *other* product and is not touched.
**Testing**: `pytest`; baseline **959 passed, 9 skipped**
**Target Platform**: frozen PyInstaller `onedir`, notarized (local); container (cloud)
**Project Type**: single
**Performance Goals**: import time within 10% (NFR-004)
**Constraints**: behaviour-preserving (C-001); Sigil daemon out of bounds (C-002); historical records not rewritten (C-006)
**Scale/Scope**: 1,738 `sigil_ml` occurrences / 165 files; 420 `kameas-ml` / 74 files; one package directory rename

## Planning Decisions

| ID | Decision | Rationale |
|---|---|---|
| D-001 | **Distinguish this product from the Sigil daemon before touching anything.** `sigil_ml` renames; `sigild`, `sigil`, `~/.local/share/sigild/`, `SIGILD_PLUGIN_URL` and prose about "the Sigil daemon" do not. | This is the only way the mission can cause real damage. `kenaz-ml` is a sidecar for a separate Go product; renaming its data path would point every install at a database that does not exist. 138 references are in that category. A blanket `sigil` → `kenaz` substitution would be catastrophic **and would look like it worked**. |
| D-002 | **`git mv` the package directory**, then rewrite imports. | Makes the diff read as a rename plus edits rather than 40 deletions and 40 additions, which is what keeps a 165-file change reviewable. |
| D-003 | **Historical records under `kitty-specs/` for merged missions are not rewritten** (C-006, FR-013). | Those specs describe what was true when they were written. Rewriting them makes the record lie — a reader would find a mission claiming to move `kenaz_ml/store.py`, which never existed under that name. |
| D-004 | **The `sys.modules` alias in `test_migration_regression.py` keeps its old key.** | It aliases `sigil_ml.store`, matching text frozen in git object `ef67e05`. The key must match history; only the value follows the rename. Changing the key breaks the test, and changing the blob is impossible — its SHA256 is asserted. This is the subtlest edit in the mission. |
| D-005 | **String-resolved references are enumerated explicitly**, not left to the import rewrite. | `"sigil_ml.app:app"` (uvicorn), `collect_submodules("sigil_ml")` and `"sigil_ml.app"` (PyInstaller), `kameas-ml = "sigil_ml.cli:main"` (entry point) are resolved by name at runtime. A statement-shaped grep cannot see them, and each fails at runtime rather than at import. |
| D-006 | **Re-verify the serialized artifacts rather than trusting the pre-planning probe.** | `registry.db` and the `.joblib` files were checked and do not embed the module path — the estimators are plain sklearn types, so no custom module path is pickled. That is the expected result, but a mission that assumes it and is wrong ships a binary that dies on first load. |
| D-007 | **The GitHub organisation stays `kameas-ai`.** | Confirmed from the remote: `git@github.com:kameas-ai/kenaz-ml.git`. Only the repository name inside the org was stale in `pyproject.toml`. |

## Charter Check

**Skipped — no charter exists.**

`CLAUDE.md` constraints hold: no dependency change (NFR-002), no table-ownership change, model names unchanged (C-004 — Go queries `ml_predictions.model` by exact string, and those names are `stuck`/`suggest`/`duration`/`quality`/`profile`, unaffected by this rename), `:7774` endpoints unchanged.

`CLAUDE.md` itself names `sigil_ml` throughout and is part of the rename surface.

## Project Structure

### Source Code — the shape of the change

```
BEFORE                          AFTER
src/sigil_ml/               →   src/kenaz_ml/          (git mv, contents follow)
  app.py cli.py config.py         same modules, imports rewritten
  datastore/ modelstore/          same subpackages
  feature_store/ models/
  signals/ training/

pyproject.toml    name = "kameas-ml"                  → "kenaz-ml"
                  packages = ["src/sigil_ml"]         → ["src/kenaz_ml"]
                  kameas-ml = "sigil_ml.cli:main"     → kenaz-ml = "kenaz_ml.cli:main"
                  URLs kameas-ai/kameas-ml            → kameas-ai/kenaz-ml

freeze/kameas-ml.spec → freeze/kenaz-ml.spec, plus collect_submodules + hidden import
.github/workflows/    CLI invocations
Makefile, docs/, *.md references

UNTOUCHED — the other product
  sigild, ~/.local/share/sigild/, SIGILD_PLUGIN_URL, "the Sigil daemon"   (138 refs)
  kitty-specs/ for merged missions                                        (historical record)
```

**Structure Decision**: A pure rename. No module moves, no package restructuring — the storage reorganization already settled the layout, and mixing a rename with a move would make both unreviewable.

## Phase 0 — Research

Complete. See [research.md](./research.md).

## Phase 1 — Design

The design is the structure above plus the occurrence map, produced as the first work package before any file changes (C-005). See [quickstart.md](./quickstart.md) for verification.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
