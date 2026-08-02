<h1 align="center">kenaz-ml</h1>

<p align="center">
  <strong>ML prediction sidecar for <a href="https://github.com/kameas-ai/sigil">Sigil</a>.</strong><br />
  Learns your workflow patterns. Predicts when you're stuck. Suggests what to do next.
</p>

<p align="center">
  <a href="https://github.com/kameas-ai/kenaz-ml/actions/workflows/ci.yml"><img src="https://github.com/kameas-ai/kenaz-ml/actions/workflows/ci.yml/badge.svg" alt="Tests" /></a>
  <a href="https://github.com/kameas-ai/kenaz-ml/actions/workflows/release.yml"><img src="https://github.com/kameas-ai/kenaz-ml/actions/workflows/release.yml/badge.svg" alt="Release" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+" /></a>
</p>

---

## Philosophy

Sigil's intelligence works in layers. The [core daemon](https://github.com/kameas-ai/sigil) watches what you do — file edits, terminal commands, git activity, test results — and runs 20+ heuristic pattern detectors written in pure Go. These heuristics are fast and always available, but they only look at the present.

**kenaz-ml adds memory.** It learns from your history to predict what's coming next: when you're about to get stuck, how long a task will take, and which nudge will actually help. The models start simple and get sharper as they observe more of your work.

The system earns trust in stages. At Level 2 (ambient), it shows passive toasts. At Level 3 (conversational), it offers action buttons. At Level 4 (autonomous), it acts on your behalf — but only after the models have demonstrated calibrated, high-confidence predictions over time. **No model skips the line.** Autonomy is earned, not assumed.

Everything runs locally. No data leaves your machine. The models are lightweight scikit-learn classifiers that train on your SQLite event history in seconds.

## How It Works

kenaz-ml runs alongside `sigild` as a local sidecar service. They share a SQLite database — the daemon writes events, kenaz-ml reads them, runs predictions, and writes results back for the daemon to surface.

```
sigild (Go)                          kenaz-ml (Python)
  │                                      │
  ├── writes events ─────────────────────┤ polls for new events
  ├── writes tasks  ─────────────────────┤ extracts features
  │                                      │
  ├── reads predictions ◄────────────────┤ writes predictions
  │   └── surfaces via notifications,    │   └── stuck, suggest,
  │       MCP tools, sigilctl            │       duration, quality
  │                                      │
  └── shared: ~/.local/share/sigild/data.db (SQLite WAL)
```

No HTTP calls between them. No message queues. Just a shared database with clear table ownership.

## Models

### Stuck Predictor

GradientBoosting classifier that predicts when you're stuck on a task. Features include test failure count, time in current phase, edit velocity, file switch rate, and time since last commit. When the model's probability exceeds 0.7, the daemon escalates the task to "stuck" phase early — before the 3-failure heuristic would trigger.

### Suggestion Policy

Thompson Sampling bandit that learns which nudges help. Chooses from 11 actions — `suggest_commit`, `suggest_break`, `suggest_step_back`, `suggest_run_tests_now`, `stay_silent`, and more. Each action maintains a Beta distribution that updates from your accept/dismiss feedback. Over time, the policy learns that *you* respond better to "take a break" than "step back" when stuck, and adjusts.

### Duration Estimator

GradientBoosting regressor that estimates how long a task will take based on file count, edit volume, time of day, and branch complexity. Returns a point estimate with a confidence interval derived from individual tree predictions.

### Quality Estimator

Weighted scoring model that computes a rolling 30-minute work quality score (0–100) from five components: test pass rate, edit focus, velocity vs. baseline, commit frequency, and revert penalty. Scores below 40 trigger "degraded" status with an actionable suggestion. Component weights are learnable from task outcomes.

## Install

### Homebrew

```bash
brew tap kameas-ai/sigil
brew install kenaz-ml
```

### From source

Requires Python 3.10+.

```bash
git clone https://github.com/kameas-ai/kenaz-ml.git && cd kenaz-ml
pip install -e ".[dev]"
```

## Usage

### Start the sidecar

```bash
kenaz-ml serve
```

The server starts on `127.0.0.1:7774` and immediately begins polling the sigild database for events. Predictions are written back automatically.

If `sigild` manages the sidecar lifecycle (the default), you don't need to start it manually — the daemon launches `kenaz-ml serve` as a subprocess and monitors its health.

### Frozen, self-contained binary (FR-3)

For shipping inside the kenaz desktop control plane, `kenaz-ml` can be frozen
into a single self-contained executable that carries its own interpreter +
scikit-learn + numpy + uvicorn — no system Python, no `pip install` on the
user's machine. kenaz supervises this binary directly (spawn-if-absent,
health-check, bounded restart), so the app never silently drops to a fake ML
backend. See `.specify/decisions/ADR-ml-packaging.md`.

```bash
pip install -e ".[freeze]"
make freeze            # → dist/kenaz-ml/ (onedir bundle: kenaz-ml exe + _internal/; current host platform)
make freeze-smoke      # boots the frozen binary, asserts /predict/stuck returns a real prediction
```

The frozen binary's runtime behaviour and the SQLite WAL data contract
(`CLAUDE.md`) are unchanged — freezing changes delivery, not behaviour.

### Train models manually

```bash
kenaz-ml train                    # from default sigild database
kenaz-ml train --db /path/to.db   # from a specific database
```

Models retrain automatically in the background after 10 completed tasks (minimum 1-hour interval). Manual training is useful for bootstrapping or after a fresh install.

### Health check

```bash
kenaz-ml health-check
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Model readiness, uptime, loaded model status |
| `/status` | GET | Poller cursor position, latest predictions |
| `/predict/stuck` | POST | Stuck probability for a task or feature set |
| `/predict/suggest` | POST | Next best suggestion action |
| `/predict/duration` | POST | Estimated task duration with confidence interval |
| `/predict/quality` | POST | Rolling work quality score with components |
| `/train` | POST | Trigger background retraining |

## Architecture

```
kenaz-ml/
  src/kenaz_ml/
    config.py              # XDG-aware path discovery
    schema.py              # Database table bootstrap
    features.py            # Feature extraction from SQLite events
    poller.py              # Event polling loop + prediction writer
    server.py              # FastAPI server + CLI entry point
    models/
      stuck.py             # GradientBoostingClassifier — stuck detection
      suggest.py           # Thompson Sampling bandit — action selection
      duration.py          # GradientBoostingRegressor — time estimation
      quality.py           # Weighted scorer — rolling quality signal
    training/
      trainer.py           # Orchestrated model retraining
      scheduler.py         # Background retrain trigger
      synthetic.py         # Synthetic data for cold-start
```

### Prediction Pipeline

1. **Poll** — every 500ms, check for new events since the last cursor position
2. **Buffer** — maintain a rolling window of recent events in memory
3. **Trigger** — predict when 3+ new events arrive and 60+ seconds have elapsed
4. **Extract** — compute features from the current task + event history
5. **Predict** — run all four models (stuck, suggest, duration, quality)
6. **Write** — insert predictions to `ml_predictions` with TTL (90–120s)
7. **Audit** — log prediction latency to `ml_events`

### Cold Start

When fewer than 10 completed tasks exist, models train on synthetic data with realistic distributions. The synthetic generator produces 500 samples per model with appropriate noise. As real data accumulates, models automatically retrain on actual workflow patterns.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Tests use temporary SQLite databases and isolated model directories — no sigild dependency required.

## Privacy

kenaz-ml reads from and writes to a local SQLite database. It makes no network calls. No telemetry. No external APIs. Your workflow data never leaves your machine.

See the [Sigil privacy policy](https://github.com/kameas-ai/sigil/blob/main/PRIVACY.md) for the full data inventory.

## License

Apache 2.0 — see [LICENSE](LICENSE).
