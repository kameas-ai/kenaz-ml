# Phase 0 Research: kenaz-ml Rebrand

**Date**: 2026-08-02 | **Plan**: [plan.md](./plan.md)

---

## Measured before planning

| | Value |
|---|---|
| `sigil_ml` | **1,738 occurrences across 165 files** |
| `kameas-ml` | **420 across 74 files**, 17 of them string literals in code |
| `sigild` and Sigil-daemon references | **138 — none may change** |
| Suite baseline | 959 passed, 9 skipped |

## The categories that hide from a working-tree grep

This repository has already produced three of them. Two prior bulk edits were each declared clean and each was wrong, so the categories below were probed empirically before planning rather than reasoned about.

**1. References inside git objects — CONFIRMED PRESENT.**
`tests/test_migration_regression.py` runs `git show ef67e05:src/sigil_ml/features.py` and executes the result. That blob contains `from sigil_ml.store import DataStore`. It is frozen history: the test asserts the blob's SHA256, so the text cannot be changed, and no working-tree grep can see it. The storage reorganization hit exactly this and it was found by running the suite, not by searching. Handled by D-004 — the `sys.modules` alias key keeps the old name.

**2. Serialized artifacts — PROBED, CLEAN.**
Feast's shipped `registry.db` does not contain the byte string `sigil_ml`. Neither do the five `.joblib` artifacts in `~/.local/share/sigild/ml-models/`. Expected, because the pickled objects are plain sklearn estimators rather than project classes, so no project module path is recorded. D-006 re-verifies rather than trusting this, because if it were wrong the failure would be a binary that builds cleanly and dies on first model load.

**3. References in a sibling branch's tree — NOT APPLICABLE NOW.**
The storage reorganization was endangered by a stale worktree holding the pre-move layout. All worktrees are currently cleaned up and all four prior missions are merged, so there is no sibling tree to resurrect old names. Worth re-checking at implementation time.

**4. String-resolved module references — CONFIRMED PRESENT (D-005).**
Four known sites resolve a module by name at runtime, each failing at runtime rather than at import:
- `"sigil_ml.app:app"` — uvicorn target, `cli.py`
- `collect_submodules("sigil_ml")` and `"sigil_ml.app"` — PyInstaller, `freeze/kameas-ml.spec`
- `kameas-ml = "sigil_ml.cli:main"` — the `[project.scripts]` entry point
- the editable install's `.pth`, written from the distribution name

---

## D-001: This product versus the Sigil daemon

**Decision**: Rename `sigil_ml` only. Leave `sigild`, `sigil`, `~/.local/share/sigild/`, `SIGILD_PLUGIN_URL` and prose about the Sigil daemon untouched.

**Rationale**: `kenaz-ml` is the ML sidecar for [`sigil`](https://github.com/wambozi/sigil), a separate Go daemon. The 138 references in that category are the integration surface. Renaming the data path would point every install at a database that does not exist — and it would *look* like the rename worked, because the code would import fine and only fail at runtime against a missing file. This is the single way this mission could cause real damage, which is why it is D-001 rather than a footnote.

**Alternatives considered**:
- *Blanket `sigil` → `kenaz` substitution* — fastest, and catastrophic. It would rewrite the data path, the plugin URL variable, and every piece of documentation describing what this product integrates with.
- *Rename the data path too, for consistency* — would require a coordinated change in a repository we do not control, and would strand every existing install's database.

---

## D-002: `git mv` the package directory

**Decision**: Move `src/sigil_ml/` → `src/kenaz_ml/` with `git mv`, then rewrite imports.

**Rationale**: A 165-file change is only reviewable if the diff shows what actually happened. `git mv` renders the move as renames; copying and deleting renders it as ~40 deletions and ~40 additions, and the real edits become invisible among them.

---

## D-003: Historical records stay historical

**Decision**: `kitty-specs/` entries for already-merged missions keep the names that were current when they were written.

**Rationale**: Those documents record decisions made at a point in time. Rewriting them produces a record that lies — a reader would find the storage-reorganization mission describing a move of `kenaz_ml/store.py`, a path that never existed under that name, and would lose the ability to match the spec against the commits it produced. Sixty-odd of the 1,738 occurrences fall here.

**Alternatives considered**:
- *Rewrite everything for consistency* — makes `git log` and the specs disagree, and destroys the audit trail the missions exist to provide.

---

## D-004: The git-object alias keeps its old key

**Decision**: In `tests/test_migration_regression.py`, `sys.modules["sigil_ml.store"]` keeps that exact key; only the module it points at follows the rename.

**Rationale**: The key exists to satisfy an import statement inside frozen historical source. The blob's SHA256 is asserted immediately before it executes, so the text cannot change; the alias must match the text. This is the one place in the mission where leaving the old name is *correct*, and a reviewer scanning for leftovers will flag it — which is why it is a recorded decision rather than an oversight to be discovered.

---

## D-005: String-resolved references enumerated explicitly

**Decision**: Treat the four known runtime-resolved sites as their own occurrence-map category rather than trusting the import rewrite to catch them.

**Rationale**: An import statement fails loudly at collection when missed. A string reference fails at runtime — the PyInstaller ones fail only in the packaged binary, on first use, which is the failure shape this codebase has already produced twice.

---

## D-006: Re-verify the serialized artifacts

**Decision**: Re-probe `registry.db` and the `.joblib` artifacts during implementation.

**Rationale**: The pre-planning probe says both are clean and the reasoning supports it. But the cost of being wrong is asymmetric: a shipped binary that dies on first model load, discovered by a user. Re-probing costs seconds.

---

## Prior art consulted

- `kitty-specs/storage-layer-reorganization-01KYXE54/occurrence_map.yaml` — the format, and the two blind spots it did not catch.
- The storage reorganization's WP03 report, which generalised the lesson: an occurrence map anchored on import-statement syntax cannot see the same module named in a different lexical form, living in a git object, or living in a sibling branch's tree.
- `src/sigil_ml/feature_store/config.py` — `bundle_dir()`, the frozen-versus-source path resolution the freeze spec depends on.
