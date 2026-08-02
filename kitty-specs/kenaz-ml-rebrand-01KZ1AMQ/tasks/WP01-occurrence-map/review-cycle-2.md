---
affected_files: []
cycle_number: 2
mission_slug: kenaz-ml-rebrand-01KZ1AMQ
reproduction_command:
reviewed_at: '2026-08-02T15:02:54Z'
reviewer_agent: unknown
verdict: rejected
wp_id: WP01
---

# WP01 rev 2 — Review (cycle 2): REJECT

Rev 2 is a large, real improvement over rev 1. The schema is valid, the headline
242/46 is exactly reproducible, the partition contains no data-path hazard, the
downstream damage is repaired, and the revision history is honest. **No instance of
`sigild`, `SIGILD_*`, `sigilctl`, or `~/.local/share/sigild/` is classified as
rename.** The failure this mission exists to prevent is not present.

It is rejected for one substantive defect plus three smaller ones. All are cheap to fix.

---

## Independently re-derived (all commands run at repo root, main, clean tree)

| Claim | Map | Re-derived | |
|---|---|---|---|
| schema validation | valid | `valid=True, errors=[]` | ✅ |
| 8 categories, legal actions | yes | code_symbols=rename, import_paths=rename, filesystem_paths=manual_review, serialized_keys=manual_review, cli_commands=rename, user_facing_strings=rename_if_user_visible, tests_fixtures=rename, logs_telemetry=rename | ✅ |
| do-not-change in-scope | 242 / 46 | **242 / 46** (`git grep -oPi 'sigil(?!_ml)' -- ':!kitty-specs' ':!.worktrees'`) | ✅ exact |
| ├ sigild | 91 / 32 | **91 / 32** (`git grep -oP 'sigild'`, case-sensitive) | ✅ exact |
| ├ sigilctl | 11 / 7 | **11 / 7** | ✅ exact |
| ├ SIGILD_env | 3 / 2 | 3 (`SIGILD_PLUGIN_URL`) | ✅ |
| ├ bare Sigil prose | 23 / 11 | ~18–23 (definitional) | ✅ ok |
| └ daemon URLs | 9 / 6 | ~8–9 (definitional) | ✅ ok |
| sigil_ml in-scope | 647 / 84 | **647 / 84** | ✅ exact |
| tests_fixtures | 343 / 24 | **343 / 24** | ✅ exact |
| SIGIL_ML_* env vars | 21 / 8 | **21 / 8** | ✅ exact |
| GitHub URL refs | 9 / 3 files | **9 / 3** (README 5, pyproject 3, CONTRIBUTING 1) | ✅ exact |
| hyphenated `sigil-ml` | 3 | 3 lowercase — **plus 2 `Sigil-ML`** (see F3) | ⚠️ |
| registry.db | 55 B, 0 hits | 55 B, 0 hits | ✅ |
| .joblib | 5 probed, 0 dirty | 5/5 clean, no `sigil` bytes | ✅ |
| `ef67e05` kind | commit (rev 1 said "blob") | `git cat-file -t ef67e05` → **commit**; `:src/sigil_ml/features.py:13` = `from sigil_ml.store import DataStore` | ✅ |
| **import_paths** | **229 / 40** | **380 / 66** (src 180/42, tests 198/23, scripts 2/1) | ❌ **F2** |
| **logs_telemetry** | **17** | logging_config.py=6; the 5 named files=19; all of src/=27 | ❌ **F2** |
| **`~/.local/share/sigild/`** | **70** | 15 `.local/share/sigild`; 16 `sigild/`; 91 total `sigild` | ❌ **F2** |
| **breakdown sums to total** | implied | **91+23+11+9+3 = 137 of 242 — 105 unaccounted** | ❌ **F1** |
| kameas-ml in-scope | *(absent)* | **258 / 34** — settles 258 vs 259: pyproject has **7** (L5,14,66,73,74,75,79) | ⚠️ **F4** |

Every string-resolved site was opened and matches byte-for-byte: `cli.py:98`,
`freeze/kameas-ml.spec:265,267`, `pyproject.toml:5-6,73-75,79,106`, `.gitignore:64`,
`.pyre_configuration:2`, `app.py:215`, `manifest.py:816`, `resolve.py:540`.
Both env-var traps verified in source: `cli.py:96` `os.environ["SIGIL_ML_MODE"] = mode.value`
(a write), and `locking.py:15` `STALE_LOCK_TIMEOUT_SEC = int(os.environ.get(...))` at
module import time. Both correctly recorded.

## The six rev-1 errors

1. Schema violation — **fixed**, validates clean with `target` present and all actions legal.
2. 524/97 → 242/46 — **fixed and independently confirmed exact**, with the three
   reasons it was wrong stated accurately.
3. "Plan undercounted" blame — **retracted correctly**; the deltas are attributed to
   the mission's own spec directory. (The specific 75/40 is no longer reproducible —
   the directory has grown to 127/64 across two revisions — but the reasoning is sound.)
4. `.pysa` "silent" — **fixed**; now states Pysa raises `InvalidModelError` and
   `pyre analyze` fails by default. Accurate.
5. Worktree `src/` "empty" — **fixed**.
6. Missed forms — SIGIL_ML_* (21 ✅ exact), hyphenated sigil-ml (3, **incomplete**, see F3),
   GitHub URLs (9 ✅ exact). **Still missing: the `SIGIL_*` family — see F1.**

## Downstream — verified repaired

- `spec.md`: no `524`, no `194`. Carries 242 at SC-004 (L153) and L163; **FR-014**
  (L118) and **SC-009** (L157) both present.
- `WP02-the-rename.md`: no `524`/`194`. **T009c** present in subtasks and as a section
  (L173); T010 asserts 242; DoD line 249 carries 242.
- WP02 `owned_files` includes `.pyre/**`, `tasks/**`, `prd.json`,
  `.kittify/memory/constitution.md` — the seven scope_additions are genuinely covered.

---

## Findings

### F1 — BLOCKING. 43% of the 242 is unenumerated, and the largest missing bucket is this product's own env vars

The breakdown accounts for 137 of 242. The 105 remaining are not miscellany. 65 of
them across 15 files are a coherent, named family that the map never mentions —
`SIGIL_*` **without** `_ML`, read by **this product's own `src/sigil_ml/config.py`**:

```
17  SIGIL_POSTGRES_URL      config.py:136, cli.py:163,167, datastore/protocol.py:185,
                            feature_store/config.py:362,368, materialize.py:536, tests
17  SIGIL_MODE              config.py:124  ← os.environ.get("SIGIL_MODE", "local")
 6  SIGIL_S3_BUCKET         config.py:155, modelstore/stores.py:261
 5  SIGIL_MODEL_CACHE_TTL   config.py:170
 4  SIGIL_TENANT            config.py:145
 3  SIGIL_TENANT_HEADER     tenant.py:40
 3  SIGIL_S3_ENDPOINT_URL   config.py:160
 3  SIGIL_MODEL_BUCKET / 2 SIGIL_MODEL_REGION / 2 SIGIL_MODEL_ENDPOINT / 1 SIGIL_TENANT_ID
```
plus `sigil_config` (7 — the local alias for `sigil_ml.config` in
`feature_store/config.py`) and `sigil_features` (2 — a Postgres DB name in a test DSN).

Why this blocks:

1. **The map asserts this family does not exist.** `serialized_keys.env_vars` says the
   `SIGIL_ML_*` vars are "THIS product's own environment variables — distinct from
   `SIGILD_*` which belong to the ledger." That is a false dichotomy. There is a third
   family, 3× larger than `SIGIL_ML_*`, equally product-owned, equally a contract with
   people — the exact argument the map uses to justify escalating Q1.

2. **Q1 / FR-014 do not cover it.** FR-014 is scoped to `SIGIL_ML_*`. `SIGIL_MODE`,
   `SIGIL_POSTGRES_URL`, `SIGIL_S3_*`, `SIGIL_MODEL_*`, `SIGIL_TENANT_*` get no
   decision at all — they are defaulted into do-not-change by silence. T002 step 2 is
   explicit: *"Ambiguous cases must be resolved explicitly, not defaulted."*

3. **It produces a concrete WP02 defect in a single file.** The map lists
   `config.py:43 (SIGIL_ML_MODE)` as a rename site. `config.py:124` reads
   `SIGIL_MODE` for the same concept and is invisible to the map. An implementer
   working from this map renames one and leaves the other, twelve lines apart, with
   no note saying why.

4. **It mislabels the committed invariant.** WP02 T010 will commit a test asserting
   242. `spec.md` SC-004 calls that "the in-scope daemon surface." ~27% of it is not
   daemon surface — it is kenaz-ml's own config keys. The arithmetic is right; the
   semantics the test encodes are wrong, and a future reader will trust the label.

**Fix**: enumerate the `SIGIL_*` (non-ML, non-D) family as its own breakdown bucket;
make the breakdown sum to 242 with a residual line; record an explicit decision for
the family (extend Q1/FR-014, or do_not_change **with a stated reason**); and if the
242 stays a mixed population, relabel it in the map and in spec.md SC-004 so the
committed test does not claim to be measuring daemon surface.

### F2 — Three category counts are not reproducible

- `import_paths: 229 / 40`. Actual import statements: **380 / 66**. Note 229 is
  exactly `git grep -oP 'sigil_ml' -- src` — i.e. all `sigil_ml` occurrences in
  `src/`, over 46 files, not 40. The figure appears to be a different measurement
  wearing this label.
- `logs_telemetry: 17`. Not reproducible from any grep: logging_config.py=6, the five
  named files=19, all of src/=27.
- `filesystem_paths.do_not_change[~/.local/share/sigild/]: occurrences: 70`. Actual
  `.local/share/sigild` = 15; `sigild/` = 16; total `sigild` = 91.

These are the numbers T004 asks to be "usable as WP02's checklist." They aren't.

### F3 — `Sigil-ML` and `.pyre/taint_models/taint.config`

`sigil-ml` is recorded as 3 occurrences — correct for lowercase. Two more exist as
`Sigil-ML`: `.pyre/taint_models/sources_sinks.pysa:1` and
`.pyre/taint_models/taint.config:2`. `taint.config` is **not** in `scope_additions`
(only `sources_sinks.pysa` is). WP02's `.pyre/**` ownership covers the file, so this
is enumeration only, not a scope hole — but the rev-1 lesson was that unenumerated
means unrenamed.

### F4 — Two accounting gaps in `status`

- **No `kameas-ml` count anywhere in the map.** It is a declared secondary target;
  `sigil-ml` and the GitHub URLs both carry counts, this one carries none. For the
  record: **258 occurrences / 34 files** (+1 `kameas_ml`, the editable `.pth`).
  pyproject has 7, not 8 — so 258, not 259.
- **The 138 / 1,738 / 420 reconciliation is absent.** The strings `138`, `1738`,
  `1,738` and `420` appear nowhere in the artifact. T002 step 3 and T004 step 3 both
  require reconciling against them, and DoD says "the 138 confirmed." Rev 2 removed
  the reconciliation rather than correcting it. State 242 vs 138 and why they differ.
- `revision_history.rev1_confirmed_correct` claims *"wp02_checklist sums to exactly
  647 with complete coverage."* **There is no `wp02_checklist` key in the artifact.**
  Given F2, the per-category numbers do not sum to 647 either. Remove the claim or
  add the checklist.

---

## What to change

1. Enumerate `SIGIL_*` (non-ML) as its own bucket; make the breakdown sum to 242;
   record an explicit decision for it; fix the "distinct from SIGILD_*" framing.
2. Re-derive `import_paths`, `logs_telemetry`, and the `~/.local/share/sigild/` 70,
   or drop the numbers.
3. Add `.pyre/taint_models/taint.config` and the two `Sigil-ML` occurrences.
4. Add the kameas-ml count (258/34), the 138 reconciliation, and either add
   `wp02_checklist` or delete the claim that it exists.

Nothing here touches the partition, the 242/91 figures, or the data path — those
are correct and confirmed. This is an enumeration-completeness rejection, not a
correctness one.
