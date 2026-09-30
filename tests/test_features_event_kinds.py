"""The activity one-hot vocabulary: six observed kinds plus an explicit ``other`` (WP01, R7d)."""

from __future__ import annotations

import pytest

from kenaz_ml.features import _EVENT_KINDS, extract_activity_features

KIND_KEYS = ["kind_browser", "kind_file", "kind_hyprland", "kind_other", "kind_power", "kind_process", "kind_terminal"]


def _kinds(kind: str) -> dict[str, float]:
    feats = extract_activity_features({"kind": kind, "payload": {}})
    return {k: v for k, v in feats.items() if k.startswith("kind_")}


def test_vocabulary_is_the_six_observed_kinds() -> None:
    assert set(_EVENT_KINDS) == {"file", "process", "hyprland", "browser", "terminal", "power"}


def test_key_set_and_sorted_order_are_pinned() -> None:
    # models/activity.py builds its vector from sorted(keys); a future edit must be visible here.
    feats = extract_activity_features({"kind": "file", "payload": {}})
    assert sorted(k for k in feats if k.startswith("kind_")) == KIND_KEYS
    assert "kind_ai" not in feats and "kind_git" not in feats


@pytest.mark.parametrize("kind", ["browser", "power", "file", "process", "hyprland", "terminal"])
def test_observed_kinds_get_their_own_dimension(kind: str) -> None:
    hot = {k for k, v in _kinds(kind).items() if v == 1.0}
    assert hot == {f"kind_{kind}"}


@pytest.mark.parametrize("kind", ["ai", "commit", "git", "never-seen", "", "phase_change"])
def test_everything_else_lands_in_other(kind: str) -> None:
    hot = {k for k, v in _kinds(kind).items() if v == 1.0}
    assert hot == {"kind_other"}


def test_missing_kind_is_other_not_all_zeros() -> None:
    feats = extract_activity_features({"payload": {}})
    assert feats["kind_other"] == 1.0
