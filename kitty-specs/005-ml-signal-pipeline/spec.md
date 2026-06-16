# Feature Specification: ML Signal Pipeline

**Feature Branch**: `005-ml-signal-pipeline`
**Created**: 2026-03-30
**Status**: Draft
**Input**: Build an event-driven ML signal system that learns each user's actual tools, workflows, and patterns from observed event data, then detects noteworthy moments and predicts behavior. Signals are rendered into human-readable suggestions by the LLM. Go heuristics remain as fallback when kenaz-ml is unavailable.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Personalized Pattern Detection (Priority: P1)

A developer works normally — editing files, running tests, committing code, switching between tools. kenaz-ml silently learns their behavioral patterns: which tools they use, their typical editing rhythms, how long they spend before committing, their test cadence. When something deviates from their personal norm — e.g., they've been editing the same file for 20 minutes without testing when they normally test every 8 minutes — kenaz-ml emits a signal. The Go daemon passes the signal to the LLM, which renders it as: "You've been on routes.py for 20 minutes without running tests — that's unusual for you. Want to run your test suite?"

The system never suggests tools the user doesn't use. It never fires on behavior that's normal *for this person*, even if it would be unusual for someone else.

**Why this priority**: This is the core value proposition — ML that adapts to the individual rather than applying generic rules. Without personalized baselines, signals are just noise.

**Independent Test**: Collect 1000+ events for a user. Build their behavior profile. Inject a sequence of events that deviates from their observed patterns. Verify a signal is emitted with structured evidence. Verify no signal is emitted for sequences that match their normal behavior.

**Acceptance Scenarios**:

1. **Given** a user with 1000+ observed events establishing a baseline, **When** their recent behavior deviates significantly from their personal norm, **Then** a signal is written to `ml_signals` with structured evidence describing the deviation.
2. **Given** a user who never uses a particular tool (e.g., Docker), **When** pattern analysis runs, **Then** no signals reference or recommend that tool.
3. **Given** a user who context-switches between 10 repos hourly as their normal pattern, **When** they continue context-switching at that rate, **Then** no "excessive context switching" signal is emitted (even though a fixed threshold would fire).
4. **Given** a signal is emitted, **When** the Go daemon reads it, **Then** the signal contains structured evidence (observed behavior, baseline behavior, deviation magnitude) sufficient for the LLM to generate a specific, actionable suggestion.

---

### User Story 2 - Next-Action Prediction and Divergence Signals (Priority: P1)

The system learns the user's typical action sequences — e.g., "after editing 5+ files, this user usually runs tests" or "after a commit, this user usually switches branches." When the user's current behavior diverges from their predicted next action, a signal is emitted. The LLM turns this into a timely nudge: "You usually run tests after editing this many files — ready to test?"

**Why this priority**: Predicting what the user will do next is the foundation for proactive assistance. Divergence from prediction is the most natural signal that something is worth mentioning.

**Independent Test**: Train the next-action model on a user's event history. Feed a live sequence where the predicted next action is "test" but the user keeps editing. Verify a divergence signal is emitted. Feed a sequence where the user follows their normal pattern. Verify no signal.

**Acceptance Scenarios**:

1. **Given** a trained next-action model for a user, **When** the user's actual next action matches the prediction, **Then** no signal is emitted.
2. **Given** the model predicts "test" as the likely next action with high confidence, **When** the user continues editing past their normal threshold, **Then** a divergence signal is emitted with the predicted action and actual behavior.
3. **Given** insufficient training data (cold start), **When** the model cannot make confident predictions, **Then** no signals are emitted (silence over noise).

---

### User Story 3 - File and Context Recommendation (Priority: P2)

Based on historical co-occurrence patterns — which files are typically edited together, which tools follow which files — the system predicts relevant files and contexts. When a user opens `app.py`, the system knows they usually also touch `routes.py` and `test_server.py` in the same session. This feeds into the LLM's context when rendering suggestions: "You're working on app.py — you usually also update routes.py and test_server.py when making changes here."

**Why this priority**: File recommendation improves suggestion quality but depends on having the pattern detection (P1) and profile infrastructure in place first.

**Independent Test**: Build co-occurrence matrix from a user's completed tasks. Start a new editing session on a file with known co-occurring files. Verify the system identifies the associated files in its signal evidence.

**Acceptance Scenarios**:

1. **Given** a user has completed tasks where files A, B, C were consistently edited together, **When** the user starts editing file A in a new session, **Then** files B and C appear in a recommendation signal as likely related files.
2. **Given** two files have never been edited in the same task, **When** one is being edited, **Then** the other is not recommended.
3. **Given** a user works across multiple repositories, **When** file recommendations are generated, **Then** they are scoped to the current repository context.

---

### User Story 4 - User Behavior Profile (Priority: P1)

kenaz-ml maintains a continuously updated profile of each user's observed behavior: tools and applications used (with frequency), file types and languages worked on, workflow rhythms (when they commit, test, take breaks), active plugins and event sources. This profile is stored locally (and optionally synced to cloud for aggregate training). All models and signals are filtered through this profile — ensuring recommendations are grounded in what the user actually does.

**Why this priority**: The profile is the foundation that all three models depend on. Without knowing what tools a user uses, patterns can't be personalized and recommendations can't be filtered.

**Independent Test**: Process a user's full event history. Verify the profile accurately reflects their tool usage (top apps, languages, test frameworks, commit frequency). Verify the profile updates incrementally as new events arrive.

**Acceptance Scenarios**:

1. **Given** a user's event history, **When** the behavior profile is built, **Then** it accurately reflects their top tools, languages, typical session length, commit frequency, and test cadence.
2. **Given** a user starts using a new tool (e.g., a new test framework), **When** enough events accumulate, **Then** the profile updates to include the new tool.
3. **Given** a user stops using a tool for an extended period, **When** the profile is refreshed, **Then** the tool's weight decays over time.
4. **Given** cloud mode with opt-in, **When** the profile is synced, **Then** only aggregated, anonymized behavioral patterns are shared — no file paths, repo names, or code content.

---

### User Story 5 - Training Flywheel: Local → Cloud → Base Model (Priority: P2)

A free-tier user runs locally with rule-based cold-start models. As they accumulate data and provide feedback (accepting or dismissing suggestions), their local models improve. A cloud-tier user opts in to aggregate training. Their anonymized behavioral patterns (not code, not file paths — just event sequences, tool usage frequencies, and feedback labels) contribute to a shared base model. The improved base model is distributed back to all users, giving even new users better starting predictions.

**Why this priority**: The training flywheel is the long-term competitive moat, but it builds on the local signal infrastructure that must work first.

**Independent Test**: Train a local model from a user's data. Verify it improves prediction accuracy over the cold-start baseline. Simulate aggregate training from multiple users' anonymized data. Verify the base model generalizes better than any single user's local model.

**Acceptance Scenarios**:

1. **Given** a user with 500+ feedback events (accepted/dismissed suggestions), **When** local training runs, **Then** the pattern detector's signal precision improves measurably over the rule-based baseline.
2. **Given** 10+ opted-in cloud users with diverse workflows, **When** aggregate training runs, **Then** the resulting base model performs better on new-user cold start than the rule-based default.
3. **Given** a new base model is available, **When** a local user updates, **Then** their models are initialized from the new base while preserving any local fine-tuning.
4. **Given** a user opts out of aggregate training, **When** cloud training runs, **Then** none of their data is included.

---

### User Story 6 - Feedback Collection for Model Improvement (Priority: P1)

When a suggestion is shown to the user (via the notification system), their response — accepted, dismissed, or ignored — is recorded as feedback. This feedback is the training signal: accepted suggestions confirm that the ML signal was valuable; dismissed suggestions indicate noise. The feedback flows back to kenaz-ml via the shared database and is used to improve signal precision over time.

**Why this priority**: Without feedback, models can't learn. This is the data collection mechanism that enables all future model improvement.

**Independent Test**: Emit a signal, have the Go daemon surface it as a suggestion, record the user's acceptance/dismissal. Verify the feedback is written to the database in a format that kenaz-ml can use for training.

**Acceptance Scenarios**:

1. **Given** a suggestion was surfaced from an ML signal, **When** the user accepts or dismisses it, **Then** the feedback is recorded with a reference to the originating signal ID.
2. **Given** accumulated feedback data, **When** training runs, **Then** the feedback is used as labels (accepted = positive, dismissed = negative) for model improvement.
3. **Given** a suggestion was surfaced but the user took no action for a configurable period, **Then** it is recorded as "ignored" — a weaker negative signal than explicit dismissal.

---

### Edge Cases

- What happens during cold start with no event history? Rule-based fallback signals only, with low confidence. No ML signals emitted until minimum data threshold is met.
- What happens when event volume is very low (e.g., user only commits once a day)? The profile reflects this low-activity pattern. Signals adapt to the user's actual cadence. No false "you haven't committed in a while" signals.
- What happens when multiple users share a machine? Each user's profile is separate, keyed by tenant ID (cloud) or system user (local).
- What happens when the user's behavior changes dramatically (new project, new language)? The profile uses exponential decay — recent behavior is weighted more heavily. Transition periods generate fewer signals as the model adapts.
- What happens when kenaz-ml is not running? Go heuristic pattern detectors (existing 21 detectors in patterns.go) continue operating as fallback. Signals from `ml_signals` table are simply absent; the daemon continues with heuristic-only suggestions.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST maintain a per-user behavior profile derived from observed events, including: tool/application usage frequency, file type distribution, workflow rhythm metrics (commit cadence, test cadence, session patterns), and active event sources.
- **FR-002**: System MUST build and update the behavior profile incrementally as new events arrive, without requiring a full reprocessing of historical data.
- **FR-003**: System MUST implement a Pattern Detector model that learns each user's behavioral baselines and emits signals when current behavior deviates significantly from personal norms.
- **FR-004**: Pattern detection MUST NOT use hardcoded signal categories or fixed thresholds — pattern types and sensitivity thresholds MUST be learned from data.
- **FR-005**: System MUST implement a Next-Action Predictor that predicts the likely next action type from a sliding window of recent events and emits divergence signals when actual behavior differs.
- **FR-006**: System MUST implement a File Recommender that predicts likely related files based on co-occurrence patterns within task sessions.
- **FR-007**: All models MUST write structured signals to an `ml_signals` table in the shared SQLite database, event-driven (written immediately on detection, not batched with the prediction polling cycle).
- **FR-008**: Each signal MUST include: signal evidence (structured JSON with observed values, baseline values, and deviation context), a confidence score, a suggested action type (generic, for LLM interpretation), creation timestamp, and expiration timestamp.
- **FR-009**: Signals MUST be filtered through the user's behavior profile — the system MUST NOT reference tools, files, languages, or workflows the user has never used.
- **FR-010**: System MUST support rule-based cold start when insufficient data exists for ML models, matching the existing model cold-start pattern (e.g., ActivityClassifier rules → ML upgrade).
- **FR-011**: System MUST support local training when a user accumulates sufficient feedback data (accepted/dismissed suggestions linked to signal IDs).
- **FR-012**: System MUST support cloud aggregate training from opted-in users using the existing cloud training pipeline (Feature 004) infrastructure.
- **FR-013**: Cloud aggregate training MUST use only anonymized behavioral patterns — no file paths, repository names, code content, or personally identifiable information.
- **FR-014**: System MUST support distributing trained base models to local instances for fine-tuning, using the existing ModelStore infrastructure (Feature 003).
- **FR-015**: System MUST record feedback linkage: when a suggestion derived from an ML signal is accepted or dismissed, the feedback MUST be traceable to the originating signal for use as training labels.
- **FR-016**: All existing kenaz-ml behavior (predictions, polling, training, local mode, cloud mode) MUST remain unchanged — the signal pipeline is additive.
- **FR-017**: The `ml_signals` table MUST be Python-owned (kenaz-ml creates and writes; Go daemon reads only), following the same ownership convention as `ml_predictions`.

### Key Entities

- **Behavior Profile**: Per-user summary of observed tool usage, workflow rhythms, file type distribution, and event source activity. Updated incrementally. Used to personalize all model outputs.
- **ML Signal**: A structured event written to `ml_signals` when a model detects something noteworthy. Contains evidence, confidence, and expiry. Read by the Go daemon for LLM rendering.
- **Signal Feedback**: A record linking a user's response (accept/dismiss/ignore) to the ML signal that produced the suggestion. Used as training labels.
- **Pattern Detector Model**: Learns per-user behavioral baselines and detects deviations. No hardcoded categories — patterns emerge from data.
- **Next-Action Model**: Predicts likely next action from recent event sequence. Emits divergence signals.
- **File Recommender Model**: Predicts co-occurring files from task session history.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: ML signals achieve a higher suggestion acceptance rate than the existing Go heuristic patterns within 30 days of deployment for users with 1000+ events.
- **SC-002**: Zero signals reference tools, files, or workflows the user has never used.
- **SC-003**: Signal latency from event detection to `ml_signals` write is under 500ms (event-driven, not poll-batched).
- **SC-004**: Cold-start users receive reasonable rule-based signals within the first session, with ML signals activating after accumulating sufficient data.
- **SC-005**: The system operates within the existing resource constraints — ML signal processing adds less than 50MB memory and less than 5% CPU overhead on the developer's laptop.
- **SC-006**: A cloud base model trained from 10+ opted-in users produces measurably better cold-start predictions than the rule-based default.
- **SC-007**: All existing predictions (stuck, suggest, duration, activity, quality) continue operating unchanged — zero regression in existing behavior.

## Assumptions

- The Go daemon (sigild) will be updated to read from the `ml_signals` table and pass signals to the LLM for rendering (to be specified as Feature 021 in the sigil repo).
- The existing suggestion feedback mechanism (accepted/dismissed/ignored status in the `suggestions` table) can be extended to include a reference to the originating `ml_signals` row ID.
- The existing cloud training pipeline (Feature 004) can be extended to train signal models alongside existing models.
- Users typically accumulate enough event data (1000+ events) within 1-2 days of normal development work to build initial behavioral baselines, based on the observed rate of ~142k events across approximately 5 days.
