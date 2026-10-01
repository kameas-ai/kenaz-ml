"""One commit-recency definition behind both stuck extractors (feature-vocabulary-refresh WP01).

Raw ``git`` and ``commit`` are the same signal; a pushed commit timestamp is one
more candidate; nothing after the reference time is ever used.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from kenaz_ml import features
from kenaz_ml.features import extract_features_from_buffer, extract_stuck_features_from_data

T0 = 1_700_000_000_000
NOW = T0 + 1_000_000  # 1000 s after T0
TASK = {"started_at": T0, "last_active": NOW, "test_fails": 0}


def _ev(kind: str, ts: int) -> dict:
    return {"kind": kind, "source": "x", "payload": {}, "ts": ts}


def _both(events: list[dict], **kw) -> tuple[float, float]:
    buf = extract_features_from_buffer(events, as_of_ms=NOW, **kw)["time_since_last_commit_sec"]
    task = extract_stuck_features_from_data(TASK, events, as_of_ms=NOW, **kw)["time_since_last_commit_sec"]
    return buf, task


def _base() -> list[dict]:
    return [_ev("file", T0), _ev("terminal", NOW - 10_000)]


@pytest.mark.parametrize("kind", ["commit", "git"])
def test_either_raw_kind_is_a_commit_on_both_extractors(kind: str) -> None:
    events = _base() + [_ev(kind, NOW - 300_000)]
    assert _both(events) == (300.0, 300.0)


def test_both_spellings_latest_wins() -> None:
    events = _base() + [_ev("git", NOW - 500_000), _ev("commit", NOW - 200_000)]
    assert _both(events) == (200.0, 200.0)


def test_no_signal_falls_back_to_session_length() -> None:
    buf, task = _both(_base())
    assert buf == extract_features_from_buffer(_base(), as_of_ms=NOW)["session_length_sec"]
    assert task == extract_stuck_features_from_data(TASK, _base(), as_of_ms=NOW)["session_length_sec"]


def test_pushed_timestamp_only() -> None:
    assert _both(_base(), pushed_commit_ts_ms=NOW - 120_000) == (120.0, 120.0)


def test_pushed_and_event_latest_wins() -> None:
    events = _base() + [_ev("git", NOW - 300_000)]
    assert _both(events, pushed_commit_ts_ms=NOW - 50_000) == (50.0, 50.0)
    assert _both(events, pushed_commit_ts_ms=NOW - 900_000) == (300.0, 300.0)


def test_pushed_after_reference_time_is_ignored() -> None:
    events = _base() + [_ev("git", NOW - 300_000)]
    assert _both(events, pushed_commit_ts_ms=NOW + 1) == (300.0, 300.0)
    # Inclusive boundary, same as _events_at_or_before.
    assert _both(events, pushed_commit_ts_ms=NOW) == (0.0, 0.0)


def test_pushed_with_empty_events_creates_no_vector() -> None:
    assert extract_features_from_buffer([], as_of_ms=NOW, pushed_commit_ts_ms=NOW - 1000) == {
        "test_failure_count": 0.0,
        "time_in_phase_sec": 0.0,
        "edit_velocity": 0.0,
        "file_switch_rate": 0.0,
        "session_length_sec": 0.0,
        "time_since_last_commit_sec": 0.0,
    }
    # No commit events on a task: degrades to the fallback path plus the push, no crash.
    assert extract_stuck_features_from_data(TASK, [], as_of_ms=NOW, pushed_commit_ts_ms=NOW - 1000)[
        "time_since_last_commit_sec"
    ] == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [0, -5, 1.5, "123", True, None])
def test_invalid_pushed_values_are_ignored(bad: object) -> None:
    assert _both(_base(), pushed_commit_ts_ms=bad) == _both(_base())  # type: ignore[arg-type]


def test_both_extractors_agree_on_identical_input() -> None:
    events = _base() + [_ev("commit", NOW - 42_000)]
    buf, task = _both(events, pushed_commit_ts_ms=NOW - 10_000)
    assert buf == task == 10.0


# --- structural: one definition, one normalisation -------------------------


def _tree(fn) -> ast.AST:
    import textwrap

    return ast.parse(textwrap.dedent(inspect.getsource(fn)))


@pytest.mark.parametrize("fn", [extract_features_from_buffer, extract_stuck_features_from_data])
def test_extractors_compute_no_commit_arithmetic_inline(fn) -> None:
    src = inspect.getsource(fn)
    assert "_time_since_last_commit_sec(" in src
    strings = {n.value for n in ast.walk(_tree(fn)) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not ({"git", "commit"} & strings), "raw commit kinds must not be compared inline"


def test_git_to_commit_is_decided_in_exactly_one_function() -> None:
    tree = ast.parse(inspect.getsource(features))
    owners = set()
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        if any(isinstance(n, ast.Constant) and n.value == "git" for n in ast.walk(fn)):
            owners.add(fn.name)
    # infer_tool maps a *tool name* "git" (unrelated to event-kind commit detection).
    assert owners - {"infer_tool"} == {"_normalise_kind"}


# --- live caller reads the WP09 accessor ------------------------------------


def test_live_buffer_path_honours_the_pushed_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    from kenaz_ml import features_push
    from kenaz_ml.feature_store.resolve import resolve_stuck_features_from_buffer

    now = int(time.time() * 1000)
    events = [_ev("file", now - 600_000), _ev("terminal", now - 5_000)]
    baseline = resolve_stuck_features_from_buffer(events)["time_since_last_commit_sec"]
    assert baseline == pytest.approx(595.0)  # no signal: session length

    store = features_push.FeaturePushStore()
    store.push("test", [{"event_class": "commit", "ts_ms": now - 60_000}])
    monkeypatch.setattr(features_push, "_CURRENT", store)
    pushed = resolve_stuck_features_from_buffer(events)["time_since_last_commit_sec"]
    assert pushed == pytest.approx(60.0, abs=2.0)


def test_live_path_with_no_lane_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    from kenaz_ml import features_push
    from kenaz_ml.feature_store.resolve import _pushed_commit_ts_ms

    monkeypatch.setattr(features_push, "_CURRENT", None)
    assert _pushed_commit_ts_ms("t1") is None
    assert _pushed_commit_ts_ms(None) is None
