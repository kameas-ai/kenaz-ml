"""Row shaping for the hosted-advice event kinds, shared by both stores.

The harness records advice snapshots, labels and model-size signals as
``events`` rows (``advice.features``, ``advice.label``, ``agent.turn_profile``,
``model.switch``). Both ``DataStore`` implementations select those rows with
their own SQL and hand them here, so the two deployments shape them
identically. Nothing in this module touches a database.

A row whose payload is not the documented shape is skipped, never repaired:
these rows feed training and serving, and a guessed field is worse than a
missing example.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

FEATURES_KIND = "advice.features"
LABEL_KIND = "advice.label"
TURN_PROFILE_KIND = "agent.turn_profile"
SWITCH_KIND = "model.switch"

#: The event kinds the sizing labeler reads.
SIZING_KINDS: tuple[str, ...] = (TURN_PROFILE_KIND, SWITCH_KIND)

#: LabelRow fields a label payload must carry (``model`` is never shipped).
_LABEL_REQUIRED = ("kind", "features_hash", "ts", "revision", "user_action", "features_complete", "shown")


def parse_payload(raw: Any) -> dict[str, Any] | None:
    """A payload as a dict: JSON text is decoded, a dict (Postgres JSONB) is taken as is."""
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            value = json.loads(raw)
        except ValueError:
            return None
        return value if isinstance(value, dict) else None
    return None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def feature_snapshots(
    rows: Iterable[Mapping[str, Any]],
    *,
    kind: str | None = None,
    task: str | None = None,
    latest_only: bool = False,
) -> list[dict[str, Any]]:
    """Shape ``advice.features`` rows, ordered by ``(ts, event_id)``.

    Each snapshot is ``{event_id, ts, task, kind, feature_contract_version,
    features, features_hash}``; ``features`` is carried verbatim (a numeric
    array on the wire, or an object keyed by feature name). ``latest_only``
    keeps the newest snapshot per ``(task, kind)``.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        payload = parse_payload(row.get("payload"))
        if payload is None:
            continue
        snap_task, snap_kind, snap_hash = payload.get("task"), payload.get("kind"), payload.get("features_hash")
        features = payload.get("features")
        version = payload.get("feature_contract_version")
        if not (_nonempty_str(snap_task) and _nonempty_str(snap_kind) and _nonempty_str(snap_hash)):
            continue
        if not isinstance(features, (list, dict)) or (version is not None and not isinstance(version, str)):
            continue
        if (kind is not None and snap_kind != kind) or (task is not None and snap_task != task):
            continue
        out.append(
            {
                "event_id": int(row["id"]),
                "ts": int(row["ts"]),
                "task": snap_task,
                "kind": snap_kind,
                "feature_contract_version": version,
                "features": features,
                "features_hash": snap_hash,
            }
        )
    out.sort(key=lambda s: (s["ts"], s["event_id"]))
    if latest_only:
        newest: dict[tuple[str, str], dict[str, Any]] = {}
        for snap in out:
            newest[(snap["task"], snap["kind"])] = snap
        out = sorted(newest.values(), key=lambda s: (s["ts"], s["event_id"]))
    return out


def label_rows(rows: Iterable[Mapping[str, Any]], *, kind: str | None = None) -> list[dict[str, Any]]:
    """Shape ``advice.label`` rows: one LabelRow per ``(kind, features_hash, ts)``.

    ``ts`` is the label's own creation time from the payload, not the event's.
    The highest ``revision`` wins; an equal or lower revision is stale and the
    first one seen (lowest event id) is kept, matching the label-log ingest.
    Each result is the LabelRow fields minus ``model``, plus ``task`` (or
    ``None`` when the payload carries none). ``features`` given as JSON text is
    decoded to an object. Ordered by ``(ts, kind, features_hash)``.
    """
    from kenaz_ml.typewriter.models import LabelRow

    best: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: int(r["id"])):
        payload = parse_payload(row.get("payload"))
        if payload is None or any(name not in payload for name in _LABEL_REQUIRED):
            continue
        if not (_is_int(payload["ts"]) and _is_int(payload["revision"])):
            continue
        try:
            label = LabelRow.model_validate({k: v for k, v in payload.items() if k not in ("task", "model")})
        except ValueError:
            continue
        if kind is not None and label.kind != kind:
            continue
        record = label.model_dump(exclude={"model"})
        if isinstance(record["features"], str):
            decoded = parse_payload(record["features"])
            if decoded is None:
                continue
            record["features"] = decoded
        task = payload.get("task")
        record["task"] = task if _nonempty_str(task) else None
        key = (record["kind"], record["features_hash"], record["ts"])
        current = best.get(key)
        if current is None or record["revision"] > current["revision"]:
            best[key] = record
    return sorted(best.values(), key=lambda r: (r["ts"], r["kind"], r["features_hash"]))


def sizing_events(rows: Iterable[Mapping[str, Any]], *, task: str | None = None) -> list[dict[str, Any]]:
    """Shape ``agent.turn_profile`` and ``model.switch`` rows for the sizing labeler.

    Each is ``{event_id, kind, ts, task, payload}`` with the payload decoded,
    ordered by ``(ts, event_id)``. The labeler joins turn profiles on their
    payload ``turn_ts``, which is carried untouched; ``ts`` here is the event's.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("kind") not in SIZING_KINDS:
            continue
        payload = parse_payload(row.get("payload"))
        if payload is None or not _nonempty_str(payload.get("task")):
            continue
        if task is not None and payload["task"] != task:
            continue
        out.append(
            {
                "event_id": int(row["id"]),
                "kind": row["kind"],
                "ts": int(row["ts"]),
                "task": payload["task"],
                "payload": payload,
            }
        )
    out.sort(key=lambda e: (e["ts"], e["event_id"]))
    return out
