# Feature Specification: kenaz-ml Rebrand

**Feature Branch**: `kenaz-ml-rebrand-01KZ1AMQ`
**Created**: 2026-08-02
**Status**: Draft
**Change mode**: `bulk_edit`
**Input**: Name and brand the product consistently as `kenaz-ml`. The Python package, the distribution, the CLI command, the URLs, the log prefixes and the documentation currently carry three different names between them.

## Context

Two renames landed incompletely — `sigil-tech` → `kameas-ai`, then the repository to `kenaz-ml` — each stopping at the repository boundary. The result is three names for one product:

| | Current | Should be |
|---|---|---|
| Repository | `kenaz-ml` | unchanged |
| Python package | `sigil_ml` | `kenaz_ml` |
| Distribution (`pyproject name`) | `kameas-ml` | `kenaz-ml` |
| CLI command | `kameas-ml` | `kenaz-ml` |
| Homepage / Repository URLs | `github.com/kameas-ai/kameas-ml` | `github.com/kameas-ai/kenaz-ml` |
| Log prefixes | `kameas-ml: …` | `kenaz-ml: …` |

Measured surface: **1,738 occurrences of `sigil_ml` across 165 files**, **420 of `kameas-ml` across 74 files**, 17 of which are string literals in code.

## The distinction that governs this mission

**`sigil` is not always this product.** `sigild` is the ledger `kenaz-ml` reads — it owns `~/.local/share/sigild/data.db` and the `events`/`tasks` tables.

Confirmed 2026-08-02 with the product owner, and corroborated by two post-pivot `CHANGELOG.md` entries dated `2026-05-16` in this repo and in `kenaz-fleet`: the workbench programme pivoted to host-rendered and `kenaz-ml` moved host-side, but **the ledger kept the name `sigild`**. kenaz-ml continues to interact with it. Its name is therefore a live runtime contract, independent of how the product is branded.

- **`sigil_ml`** — this package. Renames to `kenaz_ml`.
- **`sigild`, `sigil`, `~/.local/share/sigild/`, `SIGILD_PLUGIN_URL`, "the Sigil daemon"** — the other product. **242 in-scope occurrences across 46 files, 91 of them `sigild` (WP01 rev 2, measured excluding `kitty-specs/` and `.worktrees/`). None may be renamed.** Renaming any of them would point this product at a path that does not exist and break the integration it exists to serve.

The GitHub organisation `kameas-ai` also stays; only the repository name within it was wrong.

A rename that cannot tell these apart is worse than no rename.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - One name, everywhere (Priority: P1)

Someone encountering the project — reading the tree, installing it, running the CLI, or reading a log line — sees `kenaz-ml` consistently. No surface still says `kameas-ml` or `sigil_ml`.

**Why this priority**: This is the request. Three names for one product is a documentation defect that misleads every reader and makes the URLs in package metadata point at a repository that no longer exists under that name.

**Independent Test**: Install the distribution, run the CLI, read a startup log line, and open the package metadata. Every name is `kenaz-ml`.

**Acceptance Scenarios**:

1. **Given** the source tree, **When** the package is imported, **Then** it is `kenaz_ml`.
2. **Given** the built distribution, **When** its metadata is read, **Then** the name is `kenaz-ml` and its URLs resolve to the real repository.
3. **Given** an installed environment, **When** the CLI is invoked, **Then** the command is `kenaz-ml`.
4. **Given** a running service, **When** it logs, **Then** prefixes read `kenaz-ml`.

---

### User Story 2 - The Sigil integration is untouched (Priority: P1)

The daemon this product talks to keeps its name, its data path, and its configuration. Nothing about the integration changes.

**Why this priority**: `sigild` is a different product. Renaming `~/.local/share/sigild/` would point this service at a database that does not exist, silently breaking every install. This is the single way this mission could cause real damage.

**Independent Test**: After the rename, the service reads the same database path, honours the same environment variables, and all 242 in-scope `sigild`/`sigil` daemon references are unchanged.

**Acceptance Scenarios**:

1. **Given** the rename is complete, **When** the data path is resolved, **Then** it is still `~/.local/share/sigild/`.
2. **Given** the rename is complete, **When** daemon-facing configuration is read, **Then** variable names are unchanged.
3. **Given** documentation describing the Sigil daemon, **When** it is read, **Then** it still names Sigil, not kenaz.

---

### User Story 3 - Nothing behaves differently (Priority: P1)

Predictions, endpoints, model artifacts, and the frozen binary work exactly as before. This is a rename, not a change.

**Why this priority**: 1,738 occurrences is a large blast radius for a change whose entire value is cosmetic. Any behavioural difference is a defect, and the value of the rename depends on it being safe.

**Independent Test**: The full suite passes with no count regression, and the frozen binary builds and serves a real prediction.

**Acceptance Scenarios**:

1. **Given** the rename, **When** the suite runs, **Then** the same tests pass with no count regression.
2. **Given** the frozen binary, **When** it is built, **Then** it builds, notarizes as before, and serves a real prediction.
3. **Given** model artifacts written before the rename, **When** they are loaded after, **Then** they load unchanged.
4. **Given** a prediction request, **When** it is served, **Then** the response is identical to pre-rename.

---

### Edge Cases

- **Historical source loaded from a git object.** `tests/test_migration_regression.py` runs `git show ef67e05:src/sigil_ml/features.py` and executes it. That blob is frozen history and contains `from sigil_ml.store import DataStore`. The existing `sys.modules` alias key **must keep the old name** — it matches historical text, not current code — while its value follows the rename. The recorded SHA256 of that blob must not change.
- **Prior missions' specs** under `kitty-specs/` are historical records of what was true then. Rewriting them falsifies the record.
- **The editable install** writes a `.pth` naming the distribution; a stale one silently breaks imports.
- **Frozen-binary collection** uses `collect_submodules("sigil_ml")` and an explicit `"sigil_ml.app"` hidden import.
- **The uvicorn target string** `"sigil_ml.app:app"` is resolved by name at runtime, not imported statically.
- **Feast's shipped registry** and pickled `.joblib` artifacts were checked and do **not** embed the module path — but re-verify rather than assume.
- Log-message prefixes are user-visible strings, not identifiers.

## Requirements *(mandatory)*

### Functional Requirements

| ID | Requirement | Status |
|---|---|---|
| FR-001 | The Python package MUST be importable as `kenaz_ml` and MUST NOT be importable as `sigil_ml`. | Draft |
| FR-002 | The distribution name MUST be `kenaz-ml`. | Draft |
| FR-003 | The CLI command MUST be `kenaz-ml`. | Draft |
| FR-004 | Package metadata URLs MUST resolve to the real repository. | Draft |
| FR-005 | User-visible log prefixes and messages MUST say `kenaz-ml`. | Draft |
| FR-006 | Documentation MUST refer to this product as `kenaz-ml` throughout. | Draft |
| FR-007 | References to the Sigil daemon, `sigild`, its data path, and its configuration variables MUST be unchanged. | Draft |
| FR-008 | Runtime behaviour MUST be unchanged — same predictions, endpoints, artifact formats, and on-disk layout. | Draft |
| FR-009 | The frozen binary MUST build and serve after the rename. | Draft |
| FR-010 | Model artifacts written before the rename MUST load unchanged. | Draft |
| FR-011 | Module references resolved by string rather than by import statement MUST be updated. | Draft |
| FR-012 | Historical source loaded from git objects MUST continue to execute; aliases matching historical text MUST keep the old name. | Draft |
| FR-013 | Prior missions' specification records MUST NOT be rewritten. | Draft |

### Non-Functional Requirements

| ID | Requirement | Threshold | Status |
|---|---|---|---|
| NFR-001 | No test count regression. | Pass count no lower than pre-change (959 passed, 9 skipped) | Draft |
| NFR-002 | No new dependencies. | Zero additions to `pyproject.toml` | Draft |
| NFR-003 | Every occurrence is accounted for. | An occurrence map covering all 8 bulk-edit categories, reviewed before implementation | Draft |
| NFR-004 | Import time does not regress. | Within 10% of pre-change `import kenaz_ml.app` | Draft |

### Constraints

| ID | Constraint | Status |
|---|---|---|
| C-001 | Behaviour-preserving. No logic, signature, or default changes. | Draft |
| C-002 | The Sigil daemon's name, data path, and configuration are out of bounds (FR-007). | Draft |
| C-003 | The GitHub organisation `kameas-ai` is unchanged; only the repository name within it was wrong. | Draft |
| C-004 | Table ownership, model names, and the `:7774` endpoints are unchanged. | Draft |
| C-005 | This is a `bulk_edit`; an occurrence map must be produced and approved before implementation begins. | Draft |
| C-006 | Historical records under `kitty-specs/` for already-merged missions are not rewritten (FR-013). | Draft |

## Key Entities

- **This product**: the ML sidecar. Package `sigil_ml` → `kenaz_ml`; distribution and CLI `kameas-ml` → `kenaz-ml`.
- **The Sigil daemon**: a separate Go product (`sigild`) this integrates with. Untouchable.
- **Occurrence map**: the reviewed inventory of every site, across all eight categories, distinguishing the two above.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `import kenaz_ml` succeeds; `import sigil_ml` raises `ModuleNotFoundError`.
- **SC-002**: Distribution metadata reads `kenaz-ml` with URLs resolving to the real repository.
- **SC-003**: The CLI is invocable as `kenaz-ml`.
- **SC-004**: The in-scope daemon surface — 242 occurrences of `sigil` not followed by `_ml`, 91 of them `sigild` — is unchanged, verified by count.
- **SC-005**: The full suite passes with no count regression.
- **SC-006**: The frozen binary builds and serves a real prediction.
- **SC-007**: A pre-rename model artifact loads unchanged.
- **SC-008**: No occurrence of `sigil_ml` or `kameas-ml` remains outside historical records and frozen git-object aliases.

## Assumptions

- Feast's shipped `registry.db` and the pickled `.joblib` artifacts were probed and do not embed the module path, so neither needs regenerating. This is re-verified during implementation rather than trusted.
- The daemon references (242 in-scope, measured in WP01 rev 2) are the integration surface and are correct as they stand.
- Prior missions' `kitty-specs/` entries are historical records; leaving them naming `sigil_ml` is correct, since that is what was true when they were written.

## Out of Scope

- Renaming the GitHub organisation.
- Any change to the Sigil daemon integration.
- Restructuring packages or modules — this is a rename, not a reorganization.
- Changing model names in `ml_predictions.model`, which Go queries by exact string.
- The local-serving registry gap recorded in the model-registry mission.
