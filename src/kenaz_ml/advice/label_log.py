"""Per-kind label log and the ``/v1/labels/{kind}`` ingest (two-client-engine-01MSK2EN WP04).

The frozen ingest contract (Amendment A3.3, ruled 2026-09-30)
-------------------------------------------------------------
* Every row carries a monotonic ``revision``.
* The idempotency key is ``(client, kind, features_hash, ts)`` with
  **revision-based upsert**: a higher ``revision`` replaces the stored row (how
  a post-push ``user_action`` change lands); an equal or lower one is a
  duplicate and is ignored.
* The **ack is a cursor over ``(ts, revision)``**: rows are processed in
  ``(ts, revision)`` order and the ack advances through the contiguous prefix of
  rows that were accepted, superseded or duplicate — a client resumes at the
  first refused row.
* Propensity fields (``shown``, ``features_complete``, ``model_id``,
  ``user_action``, ``rung``, ``prompt_version``, confidence-at-decision-time)
  live in **this label log**, never in ``retained.Example`` (whose shape is
  unchanged). ``harness-recommendation-models-01MSK2RM`` joins on
  ``(features_hash, ts)`` at evaluation time and adds its own shadow records to
  the same file under a different ``record`` discriminator.
* Training mapping: ``accepted``/``auto_acted`` -> ``y=1``, ``dismissed`` ->
  ``y=0``, ``ignored`` -> recorded, not trained. A row with
  ``features_complete=false`` never reaches the retained set.

Storage
-------
``<retained_data_dir>/<kind>.labels.jsonl`` — beside the retained sets (never in
the base slot), one JSON object per line. Label records carry
``"record": "label"``; any other ``record`` value (the sibling mission's shadow
records) is preserved untouched by every rewrite here. The file holds **one
label record per key**, the highest revision: a superseding batch rewrites the
file atomically (temp file + ``os.replace``), replacing the old record in place.
The file is bounded (default 50 MB) with the retained set's policy: contiguous
oldest-first eviction.

**After eviction**, a key that was evicted is unknown again, so a cursor-zero
re-push re-appends it — at the *end* of the file, out of ``ts`` order. Readers
must therefore not assume file order is ``ts`` order; the sibling joins on
``(features_hash, ts)`` and never relies on position.

The retained mirror
-------------------
Trainable rows are appended through the registry's existing
``append_examples`` — cap, eviction, header stamping and contract-mismatch
refusal unchanged. The retained set is a **rebuildable mirror** of the label
log: when a revision changes an already-retained row's label (or makes it
untrainable), the append-only retained file cannot be edited in place, so the
kind's retained set is rebuilt from the label log through the existing
``reset_retained`` + ``append_examples`` API, once per batch. That bumps the
retained generation, which the ingest result reports.

Nothing here raises to its caller: I/O failures come back as refusals.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RECORD_LABEL = "label"
LABEL_LOG_SUFFIX = ".labels.jsonl"
DEFAULT_MAX_BYTES = 50 * 1024 * 1024

#: user_action -> training label. ``None`` = recorded in the label log, not trained.
USER_ACTION_LABELS: dict[str, float | None] = {
    "accepted": 1.0,
    "auto_acted": 1.0,
    "dismissed": 0.0,
    "ignored": None,
}

STATUS_ACCEPTED = "accepted"
STATUS_SUPERSEDED = "superseded"
STATUS_DUPLICATE = "duplicate"
STATUS_REFUSED = "refused"

# Batch refusal reasons (nothing written).
REASON_CONTRACT_MISMATCH = "contract_mismatch"
REASON_NAMES_MISMATCH = "names_mismatch"
REASON_RETAINED_REFUSED = "retained_refused"
REASON_IO_ERROR = "io_error"

# Per-row refusal reasons.
ROW_UNKNOWN_USER_ACTION = "unknown_user_action"
ROW_CLIENT_MISMATCH = "client_mismatch"
ROW_KIND_MISMATCH = "kind_mismatch"
ROW_FEATURES_INVALID = "features_invalid"

Key = tuple[str, str, str, int]


# ---------------------------------------------------------------------------
# Paths, reading, writing
# ---------------------------------------------------------------------------


def label_log_path(kind: str, *, directory: Path | str | None = None) -> Path:
    """``<directory>/<kind>.labels.jsonl``; ``directory`` defaults to the retained dir."""
    if directory is None:
        from kenaz_ml import config

        base = Path(config.retained_data_dir())
    else:
        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)
    return base / f"{kind}{LABEL_LOG_SUFFIX}"


def label_key(record: Mapping[str, Any]) -> Key:
    return (str(record["client"]), str(record["kind"]), str(record["features_hash"]), int(record["ts"]))


@dataclass
class LabelLogRead:
    """A tolerant read. ``records`` is every parsed line in file order (all record types)."""

    path: Path
    records: list[dict[str, Any]] = field(default_factory=list)
    skipped_lines: int = 0
    truncated_final_line: bool = False

    def labels(self) -> dict[Key, dict[str, Any]]:
        """Label records by key, highest revision winning."""
        out: dict[Key, dict[str, Any]] = {}
        for record in self.records:
            if record.get("record") != RECORD_LABEL:
                continue
            try:
                key = label_key(record)
            except (KeyError, TypeError, ValueError):
                continue
            prior = out.get(key)
            if prior is None or int(record.get("revision", 0)) > int(prior.get("revision", 0)):
                out[key] = record
        return out


def read_label_log(kind: str, *, directory: Path | str | None = None) -> LabelLogRead:
    """Read the kind's label log. Skips (and counts) malformed lines; never raises."""
    path = label_log_path(kind, directory=directory)
    result = LabelLogRead(path=path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return result
    except OSError as exc:
        logger.warning("label_log: cannot read %s (%s)", path, exc)
        return result
    lines = raw.split(b"\n")
    ends_clean = raw.endswith(b"\n") or not raw
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("not an object")
        except ValueError:
            if index == len(lines) - 1 and not ends_clean:
                result.truncated_final_line = True
            else:
                result.skipped_lines += 1
            continue
        result.records.append(record)
    return result


def _dumps(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_write(path: Path, lines: Sequence[str]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    payload = "".join(f"{line}\n" for line in lines).encode("utf-8")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _append(path: Path, lines: Sequence[str]) -> None:
    if not lines:
        return
    # Repair a truncated final line first so the new record starts on its own line.
    needs_newline = False
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() > 0:
                handle.seek(-1, os.SEEK_END)
                needs_newline = handle.read(1) != b"\n"
    except FileNotFoundError:
        pass
    with path.open("ab") as handle:
        if needs_newline:
            handle.write(b"\n")
        handle.write("".join(f"{line}\n" for line in lines).encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def enforce_bound(path: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> int:
    """Evict whole lines oldest-first until ``path`` fits ``max_bytes``. Returns lines dropped."""
    if max_bytes is None or max_bytes <= 0:
        return 0
    try:
        if path.stat().st_size <= max_bytes:
            return 0
        raw = path.read_bytes()
    except OSError:
        return 0
    lines = [line for line in raw.split(b"\n") if line]
    kept: list[bytes] = []
    used = 0
    for line in reversed(lines):
        cost = len(line) + 1
        if used + cost > max_bytes:
            break
        kept.append(line)
        used += cost
    kept.reverse()
    dropped = len(lines) - len(kept)
    if dropped <= 0:
        return 0
    try:
        _atomic_write(path, [line.decode("utf-8") for line in kept])
    except OSError as exc:
        logger.warning("label_log: eviction of %s failed (%s); unchanged", path, exc)
        return 0
    logger.info("label_log: %s exceeded %d bytes; evicted %d oldest line(s)", path, max_bytes, dropped)
    return dropped


def append_records(
    kind: str,
    records: Iterable[Mapping[str, Any]],
    *,
    directory: Path | str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> int:
    """Append records of any type (the sibling's shadow records use this). Returns lines evicted.

    Invalidates the in-memory key index for the kind. Raises ``OSError`` only;
    callers that must not raise wrap it.
    """
    path = label_log_path(kind, directory=directory)
    lock = _lock_for(path)
    with lock:
        _append(path, [_dumps(r) for r in records])
        evicted = enforce_bound(path, max_bytes=max_bytes)
        _INDEXES.pop(path, None)
        return evicted


# ---------------------------------------------------------------------------
# Training mapping
# ---------------------------------------------------------------------------


def training_label(record: Mapping[str, Any]) -> float | None:
    """The retained ``y`` for a label record, or ``None`` when it must not train."""
    if not record.get("features_complete", False):
        return None
    return USER_ACTION_LABELS.get(str(record.get("user_action")))


def example_for(record: Mapping[str, Any], names: Sequence[str]) -> Any | None:
    """``Example(x, y, as_of_ms)`` for a trainable record, else ``None``."""
    y = training_label(record)
    if y is None:
        return None
    from kenaz_ml.modelstore.registry import Example

    features = record.get("features") or {}
    x = tuple(float(features[name]) for name in names)
    as_of = record.get("as_of_ms")
    return Example(x=x, y=y, as_of_ms=int(as_of) if as_of is not None else int(record["ts"]))


# ---------------------------------------------------------------------------
# Per-kind index and locks
# ---------------------------------------------------------------------------

_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
#: path -> key -> stored label record (highest revision). Built lazily from disk.
_INDEXES: dict[Path, dict[Key, dict[str, Any]]] = {}


def _lock_for(path: Path) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[path] = lock
        return lock


def _index(path: Path, kind: str, directory: Path | str | None) -> dict[Key, dict[str, Any]]:
    index = _INDEXES.get(path)
    if index is None:
        index = read_label_log(kind, directory=directory).labels()
        _INDEXES[path] = index
    return index


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


@dataclass
class RowOutcome:
    index: int
    status: str
    features_hash: str
    ts: int
    revision: int
    reason: str | None = None


@dataclass
class IngestResult:
    """Outcome of one batch. ``refusal`` set means nothing was written."""

    kind: str
    ack: tuple[int, int] | None = None
    outcomes: list[RowOutcome] = field(default_factory=list)
    refusal: str | None = None
    refusal_detail: str | None = None
    retained_appended: int = 0
    retained_rebuilt: bool = False
    retained_generation: str | None = None
    evicted: int = 0

    def count(self, status: str) -> int:
        return sum(1 for o in self.outcomes if o.status == status)


def _row_refusal(row: Mapping[str, Any], client: str, kind: str, names: Sequence[str]) -> str | None:
    if str(row.get("client")) != client:
        return ROW_CLIENT_MISMATCH
    if str(row.get("kind")) != kind:
        return ROW_KIND_MISMATCH
    if str(row.get("user_action")) not in USER_ACTION_LABELS:
        return ROW_UNKNOWN_USER_ACTION
    features = row.get("features") or {}
    for value in features.values():
        try:
            if not math.isfinite(float(value)):
                return ROW_FEATURES_INVALID
        except (TypeError, ValueError):
            return ROW_FEATURES_INVALID
    return None


def _contract_refusal(rows: Sequence[Mapping[str, Any]], contract: Any) -> tuple[str, str] | None:
    """Batch-level contract check. Any mismatch refuses the whole batch."""
    version = contract.service_version
    names = tuple(contract.names)
    known = set(names)
    for row in rows:
        posted = row.get("feature_contract_version")
        if posted != version:
            return REASON_CONTRACT_MISMATCH, f"contract {posted} != {version} for kind {contract.service!r}"
        features = row.get("features") or {}
        unexpected = sorted(set(features) - known)
        missing = [n for n in names if n not in features]
        if unexpected or (row.get("features_complete") and missing):
            parts = []
            if missing and row.get("features_complete"):
                parts.append(f"missing {missing}")
            if unexpected:
                parts.append(f"unexpected {unexpected}")
            return REASON_NAMES_MISMATCH, f"features do not match contract {version}: " + "; ".join(parts)
    return None


def ingest(
    kind: str,
    client: str,
    rows: Sequence[Mapping[str, Any]],
    contract: Any,
    *,
    directory: Path | str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    retained_max_bytes: int | None = None,
    now_ms: int | None = None,
) -> IngestResult:
    """Ingest one batch under the frozen contract. Never raises."""
    result = IngestResult(kind=kind)
    try:
        return _ingest(kind, client, rows, contract, result, directory, max_bytes, retained_max_bytes, now_ms)
    except OSError as exc:
        logger.warning("label_log: ingest for %r failed", kind, exc_info=True)
        result.outcomes = []
        result.ack = None
        result.refusal = REASON_IO_ERROR
        result.refusal_detail = f"{type(exc).__name__}: {exc}"
        return result


def _ingest(
    kind: str,
    client: str,
    rows: Sequence[Mapping[str, Any]],
    contract: Any,
    result: IngestResult,
    directory: Path | str | None,
    max_bytes: int,
    retained_max_bytes: int | None,
    now_ms: int | None,
) -> IngestResult:
    from kenaz_ml.modelstore.registry import append_examples

    names = tuple(contract.names)
    batch_refusal = _contract_refusal(rows, contract)
    if batch_refusal is not None:
        result.refusal, result.refusal_detail = batch_refusal
        return result

    retained_kwargs: dict[str, Any] = {"directory": directory}
    if retained_max_bytes is not None:
        retained_kwargs["max_bytes"] = retained_max_bytes

    path = label_log_path(kind, directory=directory)
    with _lock_for(path):
        # The retained header must agree with the contract before anything is written.
        probe = append_examples(kind, [], contract, **retained_kwargs)
        if not probe.ok:
            result.refusal = REASON_RETAINED_REFUSED
            result.refusal_detail = probe.reason
            return result

        index = _index(path, kind, directory)
        stamp = now_ms if now_ms is not None else int(time.time() * 1000)

        order = sorted(range(len(rows)), key=lambda i: (int(rows[i]["ts"]), int(rows[i]["revision"])))
        pending_new: dict[Key, dict[str, Any]] = {}
        replaced: dict[Key, dict[str, Any]] = {}
        appends: list[dict[str, Any]] = []
        needs_rebuild = False
        prefix_open = True

        for i in order:
            row = rows[i]
            outcome = RowOutcome(
                index=i,
                status=STATUS_REFUSED,
                features_hash=str(row["features_hash"]),
                ts=int(row["ts"]),
                revision=int(row["revision"]),
            )
            reason = _row_refusal(row, client, kind, names)
            if reason is not None:
                outcome.reason = reason
                result.outcomes.append(outcome)
                prefix_open = False
                continue

            record = {**dict(row), "record": RECORD_LABEL, "ingested_at_ms": stamp}
            key = label_key(record)
            stored = pending_new.get(key) or replaced.get(key) or index.get(key)
            if stored is None:
                outcome.status = STATUS_ACCEPTED
                pending_new[key] = record
            elif outcome.revision > int(stored.get("revision", 0)):
                outcome.status = STATUS_SUPERSEDED
                old_y, new_y = training_label(stored), training_label(record)
                if key in pending_new:
                    pending_new[key] = record
                else:
                    replaced[key] = record
                    if old_y is not None and old_y != new_y:
                        needs_rebuild = True
                    elif old_y is None and new_y is not None:
                        appends.append(record)
            else:
                outcome.status = STATUS_DUPLICATE
            result.outcomes.append(outcome)
            if prefix_open:
                result.ack = (outcome.ts, outcome.revision)

        if not pending_new and not replaced:
            return result

        # --- label log write -------------------------------------------------
        if replaced:
            existing = read_label_log(kind, directory=directory)
            lines: list[str] = []
            seen: set[Key] = set()
            for rec in existing.records:
                if rec.get("record") == RECORD_LABEL:
                    try:
                        k = label_key(rec)
                    except (KeyError, TypeError, ValueError):
                        lines.append(_dumps(rec))
                        continue
                    if k in seen:
                        continue  # collapse any stale duplicate lines to the one current record
                    seen.add(k)
                    lines.append(_dumps(replaced.get(k, index.get(k, rec))))
                else:
                    lines.append(_dumps(rec))
            lines.extend(_dumps(r) for r in pending_new.values())
            _atomic_write(path, lines)
        else:
            _append(path, [_dumps(r) for r in pending_new.values()])
        result.evicted = enforce_bound(path, max_bytes=max_bytes)

        index.update(replaced)
        index.update(pending_new)
        if result.evicted:
            _INDEXES.pop(path, None)

        # --- retained mirror -------------------------------------------------
        if needs_rebuild:
            _rebuild_retained(kind, contract, directory, result, retained_kwargs)
        else:
            examples = [ex for ex in (example_for(r, names) for r in [*pending_new.values(), *appends]) if ex]
            if examples:
                appended = append_examples(kind, examples, contract, **retained_kwargs)
                if appended.ok:
                    result.retained_appended = appended.written
                    result.retained_generation = appended.generation
                else:  # pragma: no cover - the probe above already checked the header
                    logger.warning("label_log: retained append for %r refused: %s", kind, appended.reason)
    return result


def _rebuild_retained(
    kind: str, contract: Any, directory: Path | str | None, result: IngestResult, retained_kwargs: dict[str, Any]
) -> None:
    """Rebuild the kind's retained set from the label log (existing registry APIs only)."""
    from kenaz_ml.modelstore.registry import append_examples, reset_retained

    names = tuple(contract.names)
    labels = read_label_log(kind, directory=directory).labels()
    examples = [ex for ex in (example_for(r, names) for r in labels.values()) if ex]
    reset = reset_retained(kind, contract=contract, directory=directory)
    appended = append_examples(kind, examples, contract, generation=reset.next_generation, **retained_kwargs)
    result.retained_rebuilt = True
    result.retained_appended = appended.written
    result.retained_generation = appended.generation or reset.next_generation
    logger.info(
        "label_log: rebuilt retained set for %r from the label log after a label-changing revision "
        "(generation %s -> %s, %d example(s))",
        kind,
        reset.previous_generation,
        result.retained_generation,
        appended.written,
    )
