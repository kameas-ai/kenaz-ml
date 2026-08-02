"""FR-014 — the ``SIGIL_* -> KENAZ_*`` environment-variable deprecation shim.

The kenaz-ml rebrand renamed this product's own environment variables. Renaming
alone would have broken existing deployments *silently*, because an unrecognised
environment variable is simply ignored: a container still setting
``SIGIL_MODE=cloud`` would have quietly fallen back to local mode. The shim in
:mod:`kenaz_ml.config` turns that silent fallback into a visible warning.

Two families were renamed, and both are covered here:

===================  ====================
old                  new
===================  ====================
``SIGIL_ML_<X>``     ``KENAZ_ML_<X>``
``SIGIL_<X>``        ``KENAZ_<X>``
===================  ====================

``SIGIL_MODE`` and ``SIGIL_ML_MODE`` duplicate each other — they are two
switches for one local-vs-cloud decision. That is a real defect, filed
separately. This mission renamed both and **preserved the duplication**;
consolidating them is a behaviour change and was deliberately kept out. The
test at the bottom of this module guards that they stay distinct.
"""

from __future__ import annotations

import logging

import pytest

from kenaz_ml import config


@pytest.fixture(autouse=True)
def _clear_warned_cache():
    """The shim warns once per legacy name; reset that between tests."""
    config._ENV_DEPRECATION_WARNED.clear()
    yield
    config._ENV_DEPRECATION_WARNED.clear()


# Both families, and for each: the config accessor that reads it (or None when
# the variable is read directly at a call site rather than through an accessor).
BOTH_FAMILIES = [
    pytest.param("KENAZ_ML_MODE", "SIGIL_ML_MODE", id="SIGIL_ML_ family"),
    pytest.param("KENAZ_MODE", "SIGIL_MODE", id="SIGIL_ family"),
]


@pytest.mark.parametrize(("new", "old"), BOTH_FAMILIES)
def test_old_name_only_still_works_and_warns(new, old, monkeypatch, caplog):
    monkeypatch.delenv(new, raising=False)
    monkeypatch.setenv(old, "cloud")

    with caplog.at_level(logging.WARNING, logger="kenaz_ml.config"):
        assert config.env(new) == "cloud"

    assert len(caplog.records) == 1, "reading a deprecated name must warn exactly once"
    message = caplog.records[0].getMessage()
    assert old in message and new in message, "the warning must name both the old and the new variable"


@pytest.mark.parametrize(("new", "old"), BOTH_FAMILIES)
def test_new_name_only_works_silently(new, old, monkeypatch, caplog):
    monkeypatch.delenv(old, raising=False)
    monkeypatch.setenv(new, "cloud")

    with caplog.at_level(logging.WARNING, logger="kenaz_ml.config"):
        assert config.env(new) == "cloud"

    assert caplog.records == [], "the supported name must not warn"


@pytest.mark.parametrize(("new", "old"), BOTH_FAMILIES)
def test_new_name_wins_when_both_are_set(new, old, monkeypatch, caplog):
    monkeypatch.setenv(old, "cloud")
    monkeypatch.setenv(new, "local")

    with caplog.at_level(logging.WARNING, logger="kenaz_ml.config"):
        assert config.env(new) == "local", "the new name must take precedence"

    assert len(caplog.records) == 1
    assert old in caplog.records[0].getMessage()


@pytest.mark.parametrize(("new", "old"), BOTH_FAMILIES)
def test_default_is_returned_when_neither_is_set(new, old, monkeypatch, caplog):
    monkeypatch.delenv(old, raising=False)
    monkeypatch.delenv(new, raising=False)

    with caplog.at_level(logging.WARNING, logger="kenaz_ml.config"):
        assert config.env(new, "fallback") == "fallback"

    assert caplog.records == []


def test_the_warning_fires_only_once_per_legacy_name(monkeypatch, caplog):
    monkeypatch.delenv("KENAZ_MODE", raising=False)
    monkeypatch.setenv("SIGIL_MODE", "cloud")

    with caplog.at_level(logging.WARNING, logger="kenaz_ml.config"):
        for _ in range(5):
            config.env("KENAZ_MODE")

    assert len(caplog.records) == 1, "a variable read in a loop must not spam the log"


# ---------------------------------------------------------------------------
# The names the shim maps, and the ones it must leave alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("new", "expected_old"),
    [
        # SIGIL_ML_ family — every member the codebase reads.
        ("KENAZ_ML_MODE", "SIGIL_ML_MODE"),
        ("KENAZ_ML_LOCK_TIMEOUT_SEC", "SIGIL_ML_LOCK_TIMEOUT_SEC"),
        ("KENAZ_ML_TRAIN_MIN_TASKS", "SIGIL_ML_TRAIN_MIN_TASKS"),
        ("KENAZ_ML_TRAIN_MIN_INTERVAL", "SIGIL_ML_TRAIN_MIN_INTERVAL"),
        ("KENAZ_ML_TRAIN_MAX_TASKS_PER_TENANT", "SIGIL_ML_TRAIN_MAX_TASKS_PER_TENANT"),
        # SIGIL_ family.
        ("KENAZ_MODE", "SIGIL_MODE"),
        ("KENAZ_POSTGRES_URL", "SIGIL_POSTGRES_URL"),
        ("KENAZ_TENANT", "SIGIL_TENANT"),
        ("KENAZ_TENANT_HEADER", "SIGIL_TENANT_HEADER"),
        ("KENAZ_S3_BUCKET", "SIGIL_S3_BUCKET"),
        ("KENAZ_S3_ENDPOINT_URL", "SIGIL_S3_ENDPOINT_URL"),
        ("KENAZ_MODEL_CACHE_TTL", "SIGIL_MODEL_CACHE_TTL"),
    ],
)
def test_legacy_name_mapping(new, expected_old):
    assert config._legacy_env_name(new) == expected_old


def test_ml_prefix_is_matched_before_the_bare_prefix():
    """The longest prefix must be tried first, and that ordering is load-bearing.

    ``KENAZ_ML_MODE`` starts with both ``KENAZ_ML_`` and ``KENAZ_``. If the bare
    prefix were checked first it would yield ``SIGIL_ML_MODE`` anyway — the two
    happen to agree here — but ``KENAZ_ML_TRAIN_MIN_TASKS`` would then map to
    ``SIGIL_ML_TRAIN_MIN_TASKS`` by the same accident while any future name where
    the families diverge would map wrong. Pin the ordering rather than rely on it.
    """
    assert config._ENV_PREFIX_PAIRS[0][0] == "KENAZ_ML_"
    assert config._legacy_env_name("KENAZ_ML_ANYTHING") == "SIGIL_ML_ANYTHING"
    assert config._legacy_env_name("KENAZ_ANYTHING") == "SIGIL_ANYTHING"


@pytest.mark.parametrize("name", ["SIGILD_PLUGIN_URL", "XDG_DATA_HOME", "AWS_REGION", "MODEL_CACHE_TTL_SECONDS"])
def test_variables_this_product_does_not_own_have_no_legacy_mapping(name):
    """The shim must not invent a rename for a variable belonging to someone else.

    ``SIGILD_PLUGIN_URL`` is the sigil daemon's, and the daemon was not renamed
    by anything in this mission. ``XDG_DATA_HOME`` and ``AWS_REGION`` are
    third-party conventions.
    """
    assert config._legacy_env_name(name) is None


def test_sigild_plugin_url_is_read_under_its_own_name(monkeypatch):
    """FR-007 — the daemon-facing variable is untouched by the rebrand."""
    monkeypatch.setenv("SIGILD_PLUGIN_URL", "http://127.0.0.1:9999")
    assert config.sigild_plugin_url() == "http://127.0.0.1:9999"


# ---------------------------------------------------------------------------
# The two mode switches stay distinct (deliberately NOT consolidated)
# ---------------------------------------------------------------------------


def test_the_two_mode_switches_remain_independent(monkeypatch):
    """``KENAZ_ML_MODE`` and ``KENAZ_MODE`` duplicate each other, and stay that way.

    Consolidating them is a behaviour change, so this mission renamed both and
    preserved the duplication. If someone later merges them, this fails and
    they can make that call deliberately rather than as a side effect.
    """
    monkeypatch.delenv("SIGIL_MODE", raising=False)
    monkeypatch.delenv("SIGIL_ML_MODE", raising=False)
    monkeypatch.setenv("KENAZ_ML_MODE", "cloud")
    monkeypatch.setenv("KENAZ_MODE", "local")

    assert config.resolve_mode() is config.ServingMode.CLOUD
    assert config.operating_mode() == "local"


def test_accessors_honour_the_deprecated_names_end_to_end(monkeypatch, caplog):
    """Not just ``env()`` — the real accessors must route through the shim."""
    for new in ("KENAZ_MODE", "KENAZ_ML_MODE", "KENAZ_POSTGRES_URL", "KENAZ_TENANT", "KENAZ_S3_BUCKET"):
        monkeypatch.delenv(new, raising=False)
    monkeypatch.setenv("SIGIL_MODE", "cloud")
    monkeypatch.setenv("SIGIL_ML_MODE", "cloud")
    monkeypatch.setenv("SIGIL_POSTGRES_URL", "postgresql://localhost/legacy")
    monkeypatch.setenv("SIGIL_TENANT", "legacy-tenant")
    monkeypatch.setenv("SIGIL_S3_BUCKET", "legacy-bucket")

    with caplog.at_level(logging.WARNING, logger="kenaz_ml.config"):
        assert config.operating_mode() == "cloud"
        assert config.resolve_mode() is config.ServingMode.CLOUD
        assert config.postgres_url() == "postgresql://localhost/legacy"
        assert config.tenant_id() == "legacy-tenant"
        assert config.s3_bucket() == "legacy-bucket"

    warned = {r.getMessage() for r in caplog.records}
    assert len(warned) == 5, f"expected one warning per deprecated variable, got {warned}"
