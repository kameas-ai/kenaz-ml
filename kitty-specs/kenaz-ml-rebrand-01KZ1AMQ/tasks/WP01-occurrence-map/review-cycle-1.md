---
affected_files: []
cycle_number: 1
mission_slug: kenaz-ml-rebrand-01KZ1AMQ
reproduction_command:
reviewed_at: '2026-08-02T14:44:46Z'
reviewer_agent: unknown
verdict: rejected
wp_id: WP01
---

# WP01 Review — REJECTED

Reviewer: claude (adversarial re-derivation). Every count below was re-measured
independently with `git grep -oP` against the tree the map was written on
(commit `9621773`, i.e. repo-wide counts with `occurrence_map.yaml` itself
excluded via a pathspec). Nothing was accepted because the map asserted it.

**Good news first, so it does not get lost:** the partition itself is correct.
I sampled both sides. There is no instance anywhere in the map where
`~/.local/share/sigild/`, `SIGILD_*`, `sigilctl`, or bare `sigild` is classified
as `rename`. The failure this mission exists to prevent is not present. The
artifact re-probe is also genuinely clean (re-run, not trusted). The rejection
is about the numbers, the schema, and the scope violation — not about the
data path.

---

## BLOCKER 1 — The artifact does not pass spec-kitty's own Bulk Edit Gate

`spec-kitty agent action review WP01 --agent claude --mission kenaz-ml-rebrand-01KZ1AMQ`
refuses to run. I could not formally claim this review. The gate reports:

- Missing required `target` section
- `filesystem_paths` action `mixed` — invalid
- `serialized_keys` action `none_required` — invalid
- `tests_fixtures` action `rename_except_one` — invalid

Permitted actions are exactly `do_not_change | manual_review | rename |
rename_if_user_visible`. Three of the eight categories use invented action
values, so T001's "eight categories with explicit actions" is not satisfied in
the terms the gate defines, and the mission cannot advance past this WP.

Fixes: add `target`; `filesystem_paths` → `manual_review` (the rename/
do_not_change split stays as sub-keys); `serialized_keys` → `do_not_change`
(the evidence block is right, only the action word is invalid);
`tests_fixtures` → `manual_review` (or `rename` with the exception recorded as
a sub-key).

---

## BLOCKER 2 — The 524 figure is wrong, and it is already committed downstream

This is the highest-stakes number in the mission. WP02's T010 is instructed to
hardcode it in an invariant test.

The map states its own partition pattern as `sigil(?!_ml)`. Run exactly as
written — case-sensitive, `git grep -oP`, repo-wide:

| | map claims | I measure |
|---|---|---|
| `sigil(?!_ml)` case-SENSITIVE, repo-wide | 524 / 97 | **286 / 79** |
| `sigil(?!_ml)` case-INSENSITIVE, repo-wide | — | 527 / 97 |
| `sigild` repo-wide | 194 / 68 | **181 / 68** |
| `SIGILD_` | 16 occ / 13 files | 16 occ / **10 files** |
| bare `\bSigil\b` prose | 83 | **80** |
| `share/sigild` | 70 / 38 | 70 / 38 ✓ |
| `sigilctl` | 15 / 11 | 15 / 11 ✓ |
| `wambozi/sigil` | 7 / 5 | 7 / 5 ✓ |

Three separate defects here:

1. **The number does not match the pattern the map states.** 524/97 is only
   reachable case-INSENSITIVELY (527/97 — file count matches exactly, occurrence
   count still off by 3). An implementer writing WP02's invariant test from the
   map's stated pattern gets 286 and the test fails on the first run. The
   documented failure mode then follows exactly as WP02 warns: someone
   "fixes" it by loosening the assertion.

2. **524 is a repo-wide count that includes `kitty-specs/`** — the historical
   record D-003/FR-013 forbid touching — and includes the occurrence map
   itself (45 of the hits are in the map). The number is self-referential: it
   changes whenever anyone edits a mission document. It cannot function as a
   committed invariant. The figure WP02 actually operates on is the **in-scope**
   one: **150 case-sensitive / 242 case-insensitive**, and that is what the map
   should have reported.

3. **`sigild` = 194 is wrong** (181), and it is the second number WP02 is told
   to assert. `SIGILD_` file count and the bare-`Sigil` prose count are also
   off. The component figures do not sum to 524 under any reading
   (194+70+16+15+7+83 = 385, and they overlap — `share_sigild` is a subset of
   `sigild`).

**This has already propagated.** Commit `e2fd389` wrote 524/97/194 into
`spec.md:31` and into `WP02-the-rename.md` lines 79 and 190. Both need
correcting, not just the map.

---

## BLOCKER 3 — The reconciliation's central claim is a self-measurement artifact

The map states, for both patterns, that the pre-planning grep *undercounted*:

> `sigil_ml`: "+72 vs the 1738 pre-planning figure. The earlier grep UNDERCOUNTED."
> `kameas_ml`: "+38 vs the 420 pre-planning figure. Same cause."

This mission's own spec directory — `spec.md`, `plan.md`, `research.md`,
`tasks.md`, `quickstart.md`, `tasks/WP01-occurrence-map.md`,
`tasks/WP02-the-rename.md`, all written *after* the pre-planning measurement —
contains **75 `sigil_ml` and 40 `kameas-ml` occurrences** (map excluded).

The claimed deltas are +72 and +38.

The earlier grep did not undercount. The map counted the mission's own
paperwork and then declared the plan wrong. T004 asked for the direction of the
discrepancy to be stated explicitly; the stated direction is backwards and the
cause is misattributed.

The figure that actually matters is untouched by any of this and the map gets
it right: in-scope `sigil_ml` = **647 / 84** (exact match, verified), in-scope
`kameas-ml` = **258** (map says 259 — see the pyproject.toml off-by-one below).

---

## BLOCKER 4 — WP01 modified files it does not own; its DoD forbids it

`owned_files` for WP01 is exactly one path:
`kitty-specs/kenaz-ml-rebrand-01KZ1AMQ/occurrence_map.yaml`.
The DoD reads: "No source file modified — this WP produces one planning artifact."

Git history says otherwise:

- `42ccf73` modified `spec.md`
- `e2fd389` modified `spec.md` and `tasks/WP02-the-rename.md` (+33 lines,
  including an entirely new `T008b` subtask)

Two consequences beyond the process violation:

1. The unverified 524/194 is now committed into two downstream artifacts
   (Blocker 2).
2. **The map's own `files_outside_wp02_ownership` finding is now false.** It
   says "owned_files must be extended to cover these before implementation" —
   but the same agent already extended them in `e2fd389`. WP02's `owned_files`
   currently includes `.pyre/**`, `.pyre_configuration`, `.gitignore`,
   `prd.json`, `tasks/**`, and `.kittify/memory/constitution.md`. All seven
   files are owned. A reviewer reading the map gets a "high severity" finding
   that does not describe the repo.

---

## Overstatement to correct — the `.pysa` claim

The map (and now WP02 T008b) asserts a stale taint-model path means
"no error, no warning, a green build, and the security analysis ... is simply
gone."

That overstates it. Pysa validates models against the callables they name and
reports `InvalidModelError` for models referencing undefined functions; `pyre
analyze` surfaces these as model-verification errors and fails by default —
suppressing requires an explicit opt-out. The *impact* claim is directionally
right and `.pysa` is worth renaming first, but "silent, green build" is not
accurate and should not be written into WP02 as a security assertion. Soften to:
"pyre will report model-verification errors for the stale paths; if verification
is suppressed, taint coverage over `/predict` is lost."

---

## Also wrong — the stale-worktree justification

The finding is real and correctly identified: `.worktrees/storage-layer-
reorganization-01KYXE54-lane-a` exists at `956e13a`, branch merged into main.
research.md's "all worktrees are cleaned up" is indeed false. Good catch.

But the justification is wrong. The map says "its `src/` is empty, so it is an
inert husk." It is not empty — it contains `src/sigil_ml/`, plus its own copies
of `.pyre/taint_models/sources_sinks.pysa`, `freeze/kameas-ml.spec`,
`freeze/entrypoint.py`, and `tasks/prd-classify-predict-pipeline.md`.

It is still not a correctness hazard (`.worktrees/` is gitignored, so `git grep`
never sees it), but anyone verifying WP02 with a plain `grep -r` will get a
tree full of false leftovers. Reword the reason, and say plainly that
verification must use `git grep`, not `grep -r`.

---

## What I verified as CORRECT

Recorded so a re-implementation does not redo this work.

1. **The partition (T002) — clean.** No `~/.local/share/sigild/`, `SIGILD_*`,
   `sigilctl`, or `sigild` instance is classified as rename anywhere in the map.
   The mixed-prose resolutions (pyproject description, the two `tasks/` files)
   are sound, including "do not rename the `companion-sigil-daemon-changes.md`
   filename."

2. **Artifact re-probe (T004) — VERIFIED, independently re-run.**
   `src/sigil_ml/feature_store/registry.db` (55 bytes): 0 hits for `sigil_ml`,
   0 for `sigil` any case, 0 for `kameas`. All five `.joblib` in
   `~/.local/share/sigild/ml-models/` (activity, duration, quality, stuck,
   workflow): 0 hits each, all three patterns. The "clean" claim is true.
   No risk of a binary dying on first model load. FR-010 should hold.

3. **Git-object entry (T003) — VERIFIED.**
   `git show ef67e05:src/sigil_ml/features.py` line 13 is
   `from sigil_ml.store import DataStore`.
   `tests/test_migration_regression.py` carries
   `PRE_MIGRATION_SHA = "ef67e0539feaa914dbd0c39b92500474fdd92b78"`,
   `PRE_MIGRATION_FEATURES_SHA256 = "2b7fd419...c556bec5"`, the `hashlib.sha256`
   assertion at ~L885, and `sys.modules["sigil_ml.store"]` at L909–917.
   Everything asserted exists. Keeping the key is correct.
   Nit: `ef67e05` is a **commit**, not a blob (`git cat-file -t` → `commit`).
   The map calls it "blob ef67e05". Cosmetic, but fix the wording.

4. **Three file findings verified byte-exact, with correct line numbers.**
   - `.pyre/taint_models/sources_sinks.pysa:4` is exactly
     `def sigil_ml.routes.predict_stuck(req: TaintSource[UserInput]): ...`
     (five route entries, L4–L8).
   - `.pyre_configuration:2` is exactly
     `"source_directories": [{"root": "src", "subdirectory": "sigil_ml"}]`.
   - `.gitignore:64` is exactly `src/sigil_ml/feature_store/registry.db`.
   The other four files exist and carry hits (prd.json 7/17,
   constitution.md 0/3, prd-classify-predict-pipeline.md 4/23,
   companion-sigil-daemon-changes.md 0/5).

5. **`wp02_checklist` — all 17 rows verified, 16 exact.** The `sigil_ml` column
   sums to exactly **647**, matching in-scope. Coverage is complete: no
   in-scope top-level path carrying a hit is missing from the checklist, and
   `CHANGELOG.md`'s absence is correct (0/0). This is the strongest part of the
   artifact.
   One error: `pyproject.toml` `kameas_ml: 8` — actual is **7**. That single
   off-by-one is the entire source of the `259` vs actual `258` in-scope
   `kameas-ml` total.

---

## GAPS — what is missing

1. **`import_paths` carries no count.** It is the only category without one, so
   WP02 cannot check itself off against it. Same for `logs_telemetry`
   ("covered by user_facing_strings").

2. **`tests_fixtures` count 343 is `sigil_ml` only.** The 11 `kameas-ml`
   occurrences in `tests/` are in the checklist but absent from the category.

3. **No in-scope figure for the do-not-change surface.** Every daemon-surface
   number in `totals` is repo-wide. WP02 needs the in-scope figure (150 cs /
   242 ci) because that is the only surface it can affect. This is the fix that
   makes the invariant test viable.

4. **The GitHub repo rename is enumerated as a `renames` entry but no category
   covers it.** There are 12 `kameas-ai/kameas-ml` URLs across 6 files
   (`README.md`, `CONTRIBUTING.md`, `pyproject.toml` in-scope). Worth noting
   the map does not record that **`git remote -v` is already
   `git@github.com:kameas-ai/kenaz-ml.git`** — the GitHub-side rename has
   already happened, so those 12 in-repo URLs are currently stale/redirecting.
   That is useful context WP02 does not have.

5. **Reference forms nobody grepped for.** The map's dynamic-reference sweep
   (T003) records `importlib.import_module`, `__import__`, `getattr`, entry
   points, quoted dotted paths. Not covered anywhere:
   - `pkg_resources` / `importlib.metadata` lookups by **distribution** name
     (`kameas-ml`), which is a different string from the import package and
     resolves at runtime.
   - `python -m sigil_ml...` invocations in `Makefile` / `.github/workflows/`
     (the `-m` form is a dotted path in argv, not an import statement).
   - Coverage/tooling config keyed on the package name: `--cov=sigil_ml`,
     `[tool.coverage] source`, `[tool.pytest] testpaths`, `mypy` /
     `ruff per-file-ignores` path globs. These fail as *silently* as the
     `known-first-party` case the map did correctly catch.
   - `sigil-ml` / `sigil ml` hyphen-and-space spelling variants, and
     `SIGIL_ML_*` uppercase env-var form.
   - Sphinx/mkdocs `automodule::` / `:: sigil_ml` directives under `docs/`.
   Each should get an explicit "searched, N found" line — an unsearched form is
   indistinguishable from a clean one, which is the same argument the WP prompt
   makes for empty categories.

---

## Required to re-submit

- [ ] Fix the four gate violations; `spec-kitty agent action review WP01` must run.
- [ ] Re-derive the do-not-change surface, state the pattern and case-sensitivity
      used, and report **in-scope** and repo-wide separately. Correct 524→(286 cs
      / 527 ci repo-wide; 150 cs / 242 ci in-scope) and 194→181.
- [ ] Correct the reconciliation: the pre-planning figures did not undercount;
      the delta is this mission's own 75 + 40 spec-directory occurrences.
- [ ] Fix `pyproject.toml kameas_ml` 8→7 and the in-scope total 259→258.
- [ ] Revert or correct the out-of-ownership edits to `spec.md` and
      `WP02-the-rename.md`, then re-request them through the proper channel with
      the corrected numbers. Drop the now-false "owned_files must be extended"
      finding — they already are.
- [ ] Soften the `.pysa` "silent, green build" claim.
- [ ] Correct the worktree "src/ is empty" claim; add "verify with `git grep`,
      not `grep -r`".
- [ ] Add counts to `import_paths` / `logs_telemetry`; add the missing
      reference-form sweeps from GAPS §5 with explicit found-counts.
