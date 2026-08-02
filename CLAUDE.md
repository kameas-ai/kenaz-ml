# CLAUDE.md — kenaz-ml

## What kenaz-ml Is

`kenaz-ml` is the ML sidecar for [`sigil`](https://github.com/wambozi/sigil) — a background daemon that observes developer workflow signals and surfaces productivity suggestions.

It ships in **two deployments from one codebase**:

| | Local (open source) | Cloud (enterprise) |
|---|---|---|
| Packaging | Frozen PyInstaller `onedir`, notarized | Container |
| Data store | SQLite (`~/.local/share/sigild/data.db`, WAL) | PostgreSQL |
| `DataStore` impl | `SqliteStore` | `PostgresStore` |
| Tenancy | Single user | Multi-tenant |
| Model artifacts | Filesystem | S3 / MinIO |
| Extra deps | none | `kenaz-ml[cloud]` |

Mode is selected by `config.operating_mode()`.

**Core principle for the local deployment: security-first, local-only.** No data leaves the machine — this is a product guarantee, not a default. Anything that could transmit user-derived data must be gated behind cloud mode and must not be reachable from the open-source local path.

Architecture beyond this file — feature layer, model lifecycle, registry, base-model strategy — lives in [`docs/ML_ARCHITECTURE.md`](docs/ML_ARCHITECTURE.md).

## The Shared Database

`sigild` and `kenaz-ml` communicate **exclusively through SQLite** at `~/.local/share/sigild/data.db` in WAL mode.

### Table Ownership

| Table | Owner | Python access |
|---|---|---|
| `events` | Go | `SELECT` only |
| `tasks` | Go | `SELECT` only |
| `patterns` | Go | `SELECT` only |
| `suggestions` | Go | `SELECT` only |
| `ml_predictions` | Go | `INSERT` — Python writes predictions here |
| `ml_events` | Go | `INSERT` — Python writes audit rows |
| `ml_cursor` | **Python** | Python creates, owns, and manages |

**Python never writes to `events`, `tasks`, `patterns`, or `suggestions`.**

### Invariants

1. Every SQLite connection Python opens must set `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000`
2. Model names in `ml_predictions.model` must exactly match Go's queries: `"stuck"`, `"suggest"`, `"duration"`, `"quality"`, `"profile"`
3. The HTTP endpoints on `:7774` must remain functional — `sigilctl` uses them
4. Local runtime dependencies are `scikit-learn`, `numpy`, `fastapi`, `uvicorn`, `joblib`, **and `feast`** — nothing else without an explicit, recorded decision
5. **Never import `sqlite3` or `psycopg2` directly.** All data access goes through the `DataStore` protocol in `kenaz_ml.datastore` (`src/kenaz_ml/datastore/`), so both deployments stay behaviourally identical

### Storage packages — data vs. model artifacts

Two unrelated things used to share the word "store". They are now separate packages, and the split is the rule for where new code goes:

| Concern | Package | Import from |
|---|---|---|
| Observed data — `events`, `tasks`, the cursor | `src/kenaz_ml/datastore/` (`protocol.py`, `sqlite.py`, `postgres.py`) | `from kenaz_ml.datastore import DataStore, create_store` |
| Model artifacts — `.joblib` bytes, loading, serving cache | `src/kenaz_ml/modelstore/` (`stores.py`, `loader.py`, `cache.py`) | `from kenaz_ml.modelstore import ModelStore, LocalModelStore, S3ModelStore, CachedModelStore, model_store_factory, ModelLoader, FilesystemModelLoader, ModelCache, create_model_cache` |

**Import from the package, not the submodule.** The submodule split inside each package is an implementation detail; the package `__all__` is the supported surface. Anything concerning a model artifact — the registry, base-model refresh, retention — belongs in `kenaz_ml.modelstore`, not next to the data layer. Note the model-store factory is `model_store_factory`, not `create_model_store`.

### The dependency ceiling, and the one exception

The local build is a PyInstaller `onedir` bundle that gets **notarized**, so every native library inside it must be signed and stapled. A dependency pulling `pyarrow`, `protobuf`, or `grpcio` adds hundreds of megabytes of signable surface to every release. That is why the ceiling exists and why it still applies to anything new.

**Feast is the deliberate exception.** The product owner chose to migrate to Feast in *both* deployments with the frozen binary retained, after the cost was measured: **+309 MB site-packages, +205 MB in the built bundle, +225 Mach-O files to sign.** `feast==0.65.0` is an unconditional local dependency, pinned exactly because the registry is a version-coupled protobuf. See `docs/ML_ARCHITECTURE.md` §3.5 and the `feast-feature-store-migration` mission.

Note what this does *not* license: the exception covers Feast's tree only. Adding another heavyweight dependency still needs its own measurement and its own decision.

## Feature Extraction

`src/kenaz_ml/features.py` is the **single authority** for feature computation, shared by both deployments and by base-model training.

- The `*_from_data(task, events, *, as_of_ms=None)` functions are the definition. The store-backed `extract_*(store, task_id, ...)` variants fetch rows and delegate — they must contain no feature arithmetic.
- **`as_of_ms` is the reference time the vector describes.** `None` means current wall clock, which is correct only when the subject is an *active* task (serving). Any path replaying history must pass the example's own reference time.
- Training resolves that reference time as `completed_at` → `last_active` → skip the example. There is deliberately **no wall-clock fallback**: computing elapsed features against `time.time()` while replaying completed tasks makes them measure task age instead of behaviour.
- Events later than the reference time are filtered out before aggregation. The boundary is inclusive.
- **Feature names and ordering are a contract.** Both trainers build vectors positionally against `FEATURE_NAMES`, so a reordering silently permutes every model input.

## Known Data Gotchas

Observed against a real `sigild` database (~3.5k events, single install — treat as strong evidence, not proof):

- Event kinds present in practice: `file`, `process`, `hyprland`, `browser`, `terminal`, `power`.
- **`kind="commit"` and `kind="git"` do not appear.** The stuck extractors key commit detection off `"commit"` and the buffer extractor off `"git"`, so in practice `time_since_last_commit_sec` always falls back to `session_length_sec` — two of six stuck features carry the same value.
- `_EVENT_KINDS` one-hot encodes `git` and `ai` (never observed) while omitting `browser` and `power` (observed).

Verify against live data before treating any event-kind branch as exercised.

## Build & Test

```bash
pip install -e ".[dev]"          # local development
pip install -e ".[dev,cloud]"    # includes psycopg2, boto3 — needed for cloud-path tests
kenaz-ml serve                   # start server with poller
pytest tests/                    # run tests
```

The repo carries **no checked-in virtualenv**, and `pytest` is not on the system interpreter. Create one before running tests (`uv venv` works).

## Spec-Driven Workflow

Feature work is organised as [spec-kitty](https://github.com/) missions under `kitty-specs/<mission-slug>/`, each with `spec.md`, `plan.md`, `research.md`, `tasks.md`, and per-work-package prompts in `tasks/`.

- Every spec-kitty CLI call needs `--mission <handle>` — the handle is the `mission_id` (ULID), its 8-char `mid8` prefix, or the full `mission_slug`.
- The `NNN-` directory prefix is display-only and assigned at **merge** time; `mission_number` is `null` before then. Never use it as a selector.
- Work packages declare `owned_files`. An agent implementing a WP must not modify anything outside that list.
- Lanes collapse on any dependency edge, so a mission where every WP depends on a common foundation gets **one** lane and one worktree — expect sequential execution, not parallel worktrees.
- When multiple agents share a lane worktree, commit with `git commit --only <path>`. Plain `git add <path>` is not sufficient isolation: a sibling staging files between your add and commit will be swept into your commit.

**Troubleshooting**: if every commit fails with `ModuleNotFoundError: No module named 'specify_cli'`, the generated `.git/hooks/pre-commit` is pointing at a base interpreter rather than the one spec-kitty is installed into. Correct the interpreter path in the hook.
