# Contributing to kenaz-ml

Thanks for your interest in contributing to kenaz-ml.

## Before You Start

1. **Understand the architecture.** kenaz-ml is a sidecar — it reads events
   written by [sigild](https://github.com/kameas-ai/sigil) and writes
   predictions back to the same SQLite database. It never writes to tables
   owned by the Go daemon.

2. **Open an issue first.** For anything beyond a typo fix, open an issue
   describing what you want to change and why. This saves everyone time if
   the change conflicts with the project's direction.

3. **One logical change per PR.** Don't bundle unrelated fixes. Each PR should
   be reviewable in isolation.

## Development Setup

```bash
git clone https://github.com/kameas-ai/kenaz-ml.git
cd kenaz-ml
uv sync                   # installs the pinned Python and the locked dependencies
uv run pytest tests/ -v   # must pass before submitting
```

Requires [uv](https://docs.astral.sh/uv/). If you change a dependency, run `uv lock`
and commit `uv.lock` in the same PR; CI fails on a stale lock.

## Code Standards

- **Keep dependencies minimal.** The project uses only `fastapi`, `uvicorn`,
  `scikit-learn`, `joblib`, and `numpy`. Do not add new dependencies without
  discussion in an issue first.
- **Type hints** on all public function signatures.
- **Tests** for every new model, feature extractor, or endpoint. Use pytest
  fixtures with temporary SQLite databases — no sigild dependency in tests.
- **No network calls.** kenaz-ml is local-only. Feature extraction and
  prediction must never contact external services.
- `uv run pytest tests/ -v` must pass. No exceptions.

## Database Contract

kenaz-ml communicates with sigild exclusively through SQLite. These invariants
must be preserved:

1. Every SQLite connection must set `PRAGMA journal_mode=WAL` and
   `PRAGMA busy_timeout=5000`.
2. Model names in `ml_predictions.model` must exactly match what the Go daemon
   queries: `"stuck"`, `"suggest"`, `"duration"`, `"quality"`.
3. Python never writes to `events`, `tasks`, `patterns`, or `suggestions` —
   those tables are owned by the Go daemon.
4. The HTTP server's default port stays `7774` — `sigild` and `sigilctl`
   depend on it.

## Adding a New Model

1. Create `src/kenaz_ml/models/your_model.py` following the pattern in
   `stuck.py` or `duration.py` — define `FEATURE_NAMES`, implement `predict()`,
   `train()`, `is_trained`, and weight persistence via `joblib`.
2. Add a feature extractor in `features.py`.
3. Wire the model into `poller.py` and `server.py`.
4. Add synthetic data generation in `training/synthetic.py` for cold-start.
5. Add tests covering training, prediction, and persistence.
6. Update `CLAUDE.md` with the new model name in the invariants table.

## Commit Messages

```
feat: short description
fix: short description
refactor: short description
test: short description
docs: short description
```

## Cutting a Release

Releases are tags. CI builds and publishes everything from the tag; nothing
is built or uploaded by hand.

1. Bump `version` in `pyproject.toml` (it is the engine's single version
   source: the freeze stamps it into the bundle and `/health` reports it).
   Merge that to `main`.
2. Tag the merge commit `v<version>` and push the tag. The freeze job
   refuses a tag that disagrees with `pyproject.toml`.
3. CI then: freezes the engine for macOS arm64, Linux x86_64/arm64 and
   Windows x86_64 and runs the frozen smoke tests on each; signs, notarizes
   and staples the macOS `.dmg` (a release build **fails** without the Apple
   credentials — never skips); attaches every bundle plus `SHA256SUMS` to the
   GitHub Release; and, once the publication secrets and infrastructure
   exist, signs the manifest and publishes to the release bucket for the
   harness's pinned install (until then that job reports NOT RUN).
4. `scripts/install.sh` / `scripts/install.ps1` pick the new release up
   automatically; the kenaz app and the harness pin releases explicitly.

## License

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0.
