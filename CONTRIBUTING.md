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

Releases are cut automatically. Nothing is tagged, built or uploaded by hand.

1. **PR titles are conventional commits**, enforced by the "Conventional
   Commits" check. Merges are squash-only and the PR title becomes the
   commit subject on `main`, which is what decides the release:

   | Title prefix | Bump | Release? |
   |---|---|---|
   | `feat:` / `feat(scope):` | minor (`0.x.0`) | yes |
   | `fix:`, `perf:`, `revert:`, `deps:` | patch (`0.x.y`) | yes |
   | `feat!:` or `BREAKING CHANGE:` in the body | major (capped to minor while `< 1.0.0`) | yes |
   | `docs:`, `chore:`, `ci:`, `style:`, `refactor:`, `test:`, `build:` | none | no |

2. **A release-worthy PR must carry the matching `version` bump in
   `pyproject.toml`** (the engine's single version source: the freeze stamps
   it into the bundle and `/health` reports it). `tag-on-merge` computes the
   next tag from the latest `vX.Y.Z` tag and fails, without tagging, if
   `pyproject.toml` disagrees.
3. On merge, `tag-on-merge.yml` tags `vX.Y.Z`, creates the GitHub Release
   with auto-generated notes, and dispatches the release builds at the tag:
   `ci.yml` freezes the engine for macOS arm64 and x86_64, Linux x86_64 and
   arm64, and Windows x86_64, runs the frozen smoke tests on each, signs,
   notarizes and staples both macOS `.dmg`s (a release build **fails**
   without the Apple credentials, never skips), attaches every bundle plus
   `SHA256SUMS` to the Release, signs each bundle's manifest with the engine
   release key and publishes to the prod release bucket and
   `downloads.kameas.ai`; `release.yml` attaches the sdist and wheel.
4. `scripts/install.sh` / `scripts/install.ps1` pick the new release up
   automatically; the kenaz app and the harness pin releases explicitly
   (the harness verifies the manifest signature against its baked copy of
   the release public key).

Pushes to `main` that do not cut a release still build and publish a
`<version>-dev.<sha>` label to the dev bucket. A `vX.Y.Z-rc.N` tag pushed by
hand publishes to stage.

## License

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0.
