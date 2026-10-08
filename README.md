<h1 align="center">kenaz-ml</h1>

<p align="center">
  <strong>ML prediction sidecar for <a href="https://github.com/kameas-ai/sigil">Sigil</a>.</strong><br />
  Learns your workflow patterns. Predicts when you're stuck. Suggests what to do next.
</p>

<p align="center">
  <a href="https://github.com/kameas-ai/kenaz-ml/actions/workflows/ci.yml"><img src="https://github.com/kameas-ai/kenaz-ml/actions/workflows/ci.yml/badge.svg" alt="Tests" /></a>
  <a href="https://github.com/kameas-ai/kenaz-ml/actions/workflows/release.yml"><img src="https://github.com/kameas-ai/kenaz-ml/actions/workflows/release.yml/badge.svg" alt="Release" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.14+-blue.svg" alt="Python 3.14+" /></a>
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

### Binary (no Python needed)

Every release ships the engine as a self-contained, frozen bundle for macOS
(Apple silicon, signed and notarized), Linux (x86_64, arm64) and Windows
(x86_64). The install scripts download the bundle for your machine from the
GitHub Release, verify it against the release's `SHA256SUMS`, and put
`kenaz-ml` on your PATH:

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/kameas-ai/kenaz-ml/main/scripts/install.sh | sh
```

```powershell
# Windows (PowerShell)
irm https://raw.githubusercontent.com/kameas-ai/kenaz-ml/main/scripts/install.ps1 | iex
```

Set `KENAZ_ML_VERSION=1.2.3` to pin a version. The bundles themselves are on
the [releases page](https://github.com/kameas-ai/kenaz-ml/releases) as
`kenaz-ml-<version>-<os>-<arch>.{dmg,zip}`; each unpacks to a `kameas-ml/`
directory whose `kameas-ml` executable is the engine. The kenaz desktop app and
kenaz-harness install the same bundles on their own.

### From source

Requires [uv](https://docs.astral.sh/uv/), which installs the pinned Python (3.14) and the locked dependencies for you.

```bash
git clone https://github.com/kameas-ai/kenaz-ml.git && cd kenaz-ml
uv sync
```

Without uv, `pip install -e . --group dev` (pip 25.1+, Python 3.14+) works too, but resolves its own versions rather than the committed `uv.lock`.

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
make freeze            # → dist/kameas-ml/ (onedir bundle: kameas-ml exe + _internal/; current host platform)
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
uv sync
uv run pytest tests/ -v
```

Tests use temporary SQLite databases and isolated model directories — no sigild dependency required.

## Privacy

**Local by default.** The open-source Kenaz engine records your workflow and produces its suggestions on your machine. Nothing leaves your machine. There is no telemetry, no account requirement and no cloud dependency in the open-source engine. Organizations that subscribe to Kameas Fleet can separately enable a paid, opt-in service called Offload; that is a different thing, it is off unless an organization turns it on, and it is described below.

kenaz-ml reads from and writes to a local SQLite database. It makes no network calls. No telemetry. No external APIs. The test suite enforces this: `tests/test_no_egress.py` fails the build if any code path opens a socket or resolves a name.

### Offload, a separate paid service

Kameas also offers hosted inference for organizations that subscribe to Kameas Fleet. Under that service, called Offload, an organization's administrator can choose to send its developers' workflow activity to Kameas's cloud so predictions are computed there, including from models trained on the whole team's history.

- Offload exists only when the engine is connected to a Kameas Fleet account on which an organization administrator has enabled it. The open-source engine on its own never sends anything.
- Before anything is first sent from a developer's machine, the developer sees a notice in the product describing exactly what is sent.
- What is sent: workflow events (file paths, shell commands run, git branch and repository names, active application or window, browser activity at domain level, power state, timestamps) and task summaries derived from them.
- What is never sent, under Offload or otherwise: file contents, source code, prompts or AI conversations, screenshots, keystrokes. Strings that look like credentials are redacted before sending.
- Where it goes: Kameas's AWS environment in Ohio (us-east-2), in a database schema dedicated to the organization. Predictions come back only to the developer's own tools.

The terms that govern Offload are in the Kameas Fleet Subscription Agreement (§3.5 and the Data Processing Addendum) at [kenaz.kameas.ai/subscription-agreement](https://kenaz.kameas.ai/subscription-agreement), and the [Kameas Privacy Policy](https://kameas.ai/privacy), section 13, explains what it means for an individual developer.

**If you want to be sure:** run the engine without a Fleet account, or with Offload disabled in your organization, and nothing leaves your machine. That is the guarantee the open-source project makes, and this section is where we will say so if that ever changes.

See the [Sigil privacy policy](https://github.com/kameas-ai/sigil/blob/main/PRIVACY.md) for the full data inventory.

## License

Apache 2.0 — see [LICENSE](LICENSE).
