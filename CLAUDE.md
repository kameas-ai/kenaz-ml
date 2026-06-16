# CLAUDE.md — kenaz-ml

## What kenaz-ml Is

`kenaz-ml` is the ML sidecar for [`sigil`](https://github.com/wambozi/sigil) — a background daemon that observes developer workflow signals and surfaces productivity suggestions. The ML service runs locally on the user's laptop.

**Core principle: security-first, local-only.** No data leaves the machine.

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
4. No heavyweight dependencies — `scikit-learn`, `numpy`, `fastapi`, `uvicorn`, `joblib` only

## Build & Test

```bash
pip install -e ".[dev]"
kenaz-ml serve           # start server with poller
pytest tests/            # run tests
```
