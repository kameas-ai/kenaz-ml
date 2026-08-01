"""Deployment configuration for the Feast feature store.

Two configurations live beside this module — ``feature_store.local.yaml`` and
``feature_store.cloud.yaml`` — and this module decides which one a process may
see, based on :func:`sigil_ml.config.operating_mode`, the same switch that
already chooses between ``SqliteStore`` and ``PostgresStore``.

Why this module is defensive
----------------------------
C-001 is a hard product requirement: the local install has no network path to
the cloud feature store. Feast, however, *supports* remote registries, remote
online stores and remote providers purely by configuration. A guarantee that
rests on "we did not configure that" is one edit away from being false, and the
edit would look innocuous in review. So the separation is built three ways, and
each one has to be defeated independently:

1. **The cloud path is not computed in local mode.** :func:`cloud_config_path`
   raises before it builds the path, so the local flow cannot read the cloud
   file, cannot read-then-ignore it, and cannot even name it.
2. **The local loader refuses network-shaped configuration text.** Whatever file
   it is handed — the shipped one, a future edit of it, or the cloud file if
   someone rewires the dispatch — it lints the text first and raises
   :class:`NetworkSurfaceError` on any match. This holds at runtime, in a user's
   install, not only under test.
3. **The test suite proves it structurally.** ``tests/test_no_egress.py`` runs
   the full local flow with socket creation blocked at the interpreter level and
   asserts no socket was ever created, by any path, anticipated or not.

Path resolution
---------------
Two roots, deliberately kept apart (D-001, FR-013):

* the **bundle** directory is read-only — inside the frozen ``onedir`` bundle it
  is under ``sys._MEIPASS``, and in a source or wheel install it is this
  package's own directory. The registry ships pre-applied there, so no feature
  operation writes into the application directory;
* the **user data** directory is writable, and is the same
  ``~/.local/share/sigild`` that already holds ``data.db``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml

from sigil_ml import config as sigil_config

__all__ = [
    "CLOUD_CONFIG_FILENAME",
    "LOCAL_CONFIG_FILENAME",
    "NETWORK_SURFACE_PATTERNS",
    "CloudConfigUnreachableError",
    "NetworkSurfaceError",
    "UnknownOperatingModeError",
    "assert_no_network_surface",
    "bundle_dir",
    "cloud_config_path",
    "load_cloud_repo_config",
    "load_local_repo_config",
    "load_repo_config",
    "local_config_path",
    "local_config_text",
    "online_store_path",
    "registry_path",
    "render_local_config",
    "user_data_dir",
]

LOCAL_CONFIG_FILENAME = "feature_store.local.yaml"
CLOUD_CONFIG_FILENAME = "feature_store.cloud.yaml"

REGISTRY_FILENAME = "registry.db"
ONLINE_STORE_FILENAME = "feast_online.db"

#: Substrings that must never appear anywhere in the local configuration, in a
#: value or in a comment. The list is intentionally broader than "things that
#: are definitely a URL": ``host`` and ``port`` on their own are how Feast names
#: the fields of every remote store it supports, so catching the field name
#: catches the edit at the moment it is written rather than the moment it
#: connects. Matching is case-insensitive and by substring, which is why the
#: local YAML's own prose is written around these words.
NETWORK_SURFACE_PATTERNS: tuple[str, ...] = (
    "http",
    "://",
    "host",
    "port",
    "password",
    "token",
    "endpoint",
)

_BUNDLE_DIR_MARKER = "${BUNDLE_DIR}"
_USER_DATA_DIR_MARKER = "${USER_DATA_DIR}"


class NetworkSurfaceError(RuntimeError):
    """Raised when configuration bound for the local deployment looks remote.

    This is the runtime half of the C-001 guarantee. It fires before Feast sees
    the configuration, so a network-shaped value never reaches a client.
    """


class CloudConfigUnreachableError(RuntimeError):
    """Raised when the cloud configuration is requested from a non-cloud process.

    Deliberately raised *before* the cloud path is constructed, so that in local
    mode the cloud configuration is unreachable rather than merely unused.
    """


class UnknownOperatingModeError(RuntimeError):
    """Raised when the operating mode is not one this module knows how to serve.

    There is no default. Falling back to anything would mean picking a
    deployment on the user's behalf, and the only two candidates are "silently
    local" (surprising) and "silently network-capable" (a C-001 breach).
    """


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def bundle_dir() -> Path:
    """Return the read-only directory holding the shipped feature-store assets.

    Resolves for both distribution forms. Under PyInstaller the interpreter
    exposes the unpacked bundle root as ``sys._MEIPASS`` and the collected
    package data sits beneath it at its import path; in a source or wheel
    install the assets sit beside this module. The attribute is absent outside a
    frozen build, which is why it is read with ``getattr`` rather than imported.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        return Path(meipass) / "sigil_ml" / "feature_store"
    return Path(__file__).resolve().parent


def user_data_dir() -> Path:
    """Return the writable sigild data directory.

    Derived from :func:`sigil_ml.config.db_path` rather than recomputed, so the
    online store cannot drift away from the shared database it sits beside if
    ``XDG_DATA_HOME`` moves.
    """
    return sigil_config.db_path().parent


def local_config_path() -> Path:
    """Return the path of the shipped local configuration."""
    return bundle_dir() / LOCAL_CONFIG_FILENAME


def cloud_config_path() -> Path:
    """Return the path of the shipped cloud configuration.

    Raises:
        CloudConfigUnreachableError: If the process is not in cloud mode. The
            check runs before the path is built, so a local process cannot
            obtain the location of the cloud configuration at all.
    """
    mode = sigil_config.operating_mode()
    if mode != "cloud":
        raise CloudConfigUnreachableError(
            f"The cloud feature-store configuration is not reachable in {mode!r} mode. "
            "The local deployment has no network path to the cloud feature store (C-001); "
            "if you meant to run the cloud deployment, set SIGIL_MODE=cloud."
        )
    return bundle_dir() / CLOUD_CONFIG_FILENAME


def registry_path(*, bundle: Path | None = None) -> Path:
    """Return the path of the shipped, read-only local registry."""
    return (bundle or bundle_dir()) / REGISTRY_FILENAME


def online_store_path(*, user_data: Path | None = None) -> Path:
    """Return the path of the writable local SQLite online store."""
    return (user_data or user_data_dir()) / ONLINE_STORE_FILENAME


# ---------------------------------------------------------------------------
# The lint
# ---------------------------------------------------------------------------


def assert_no_network_surface(text: str, *, source: str) -> None:
    """Raise unless ``text`` is free of every network-shaped pattern.

    Args:
        text: The configuration text to scan, comments included.
        source: A human-readable name for the text, used in the diagnostic.

    Raises:
        NetworkSurfaceError: On the first matching pattern, naming the line so
            the diagnostic points at the edit rather than at the file.
    """
    lowered = text.lower()
    for pattern in NETWORK_SURFACE_PATTERNS:
        index = lowered.find(pattern)
        if index == -1:
            continue
        line_number = text.count("\n", 0, index) + 1
        line = text.splitlines()[line_number - 1].strip()
        raise NetworkSurfaceError(
            f"{source} line {line_number} matches the network-shaped pattern {pattern!r}: {line!r}. "
            "The local deployment must reference no remote registry, remote online store or "
            "remote provider (FR-005, C-001). If this is prose in a comment, reword it — the "
            "scan covers comments on purpose, because a value parked in a comment is one "
            "keystroke from being live."
        )


#: A URL scheme at the head of a value that is supposed to be a directory.
#: Two or more characters before the colon, so a Windows drive letter (``C:``)
#: is not mistaken for one.
_URL_SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]+:", re.IGNORECASE)


def _assert_local_directory(value: str | Path, *, what: str) -> Path:
    """Validate a directory substituted into the local configuration.

    Substituted values are checked with a narrower rule than
    :func:`assert_no_network_surface`: a real filesystem path can perfectly well
    contain the letters ``port`` (``/Library/Application Support``), so applying
    the full substring scan to a resolved path would fail on machines rather
    than on edits. What actually matters for a path is that it is a local
    absolute location and not a URL, and that is what is asserted.

    The rejection is by URL *scheme* rather than by the ``://`` separator,
    because ``pathlib`` normalises a doubled slash away the moment a value
    becomes a ``Path``: ``str(Path("s3://bucket"))`` is already ``"s3:/bucket"``,
    which would otherwise sail through as a plausible relative directory.

    Returns:
        The resolved absolute directory.

    Raises:
        NetworkSurfaceError: If the value carries a URL scheme.
        ValueError: If the value does not resolve to an absolute path.
    """
    given = str(value)
    if "://" in given or _URL_SCHEME.match(given):
        raise NetworkSurfaceError(
            f"The {what} for the local feature store must be a local directory, not a URL: {given!r}."
        )
    resolved = Path(given).resolve()
    if not resolved.is_absolute():
        raise ValueError(f"The {what} for the local feature store must be an absolute path, got {given!r}.")
    return resolved


# ---------------------------------------------------------------------------
# Local configuration
# ---------------------------------------------------------------------------


def local_config_text() -> str:
    """Read the shipped local configuration and lint it.

    Raises:
        NetworkSurfaceError: If the shipped file has acquired a network-shaped
            value or comment since it was written.
    """
    path = local_config_path()
    text = path.read_text(encoding="utf-8")
    assert_no_network_surface(text, source=str(path))
    return text


def render_local_config(
    *,
    bundle: str | Path | None = None,
    user_data: str | Path | None = None,
) -> str:
    """Return the local configuration with its two directory markers filled in.

    Args:
        bundle: Overrides the read-only bundle directory. Used by the build step
            that applies the registry, and by tests. Never a source of network
            surface: it is validated as a local absolute directory.
        user_data: Overrides the writable user data directory, same conditions.

    Raises:
        NetworkSurfaceError: If the template or either directory looks remote.
    """
    text = local_config_text()
    bundle_root = _assert_local_directory(bundle or bundle_dir(), what="bundle directory")
    user_data_root = _assert_local_directory(user_data or user_data_dir(), what="user data directory")
    return text.replace(_BUNDLE_DIR_MARKER, str(bundle_root)).replace(_USER_DATA_DIR_MARKER, str(user_data_root))


def load_local_repo_config(
    *,
    bundle: str | Path | None = None,
    user_data: str | Path | None = None,
) -> Any:
    """Build the Feast ``RepoConfig`` for the local deployment.

    This function names only :data:`LOCAL_CONFIG_FILENAME`. It has no branch,
    parameter or fallback that could reach the cloud configuration, and it does
    not call :func:`cloud_config_path`.

    The writable directory is created if absent; the bundle directory is never
    created, because it belongs to the read-only installed application (FR-013)
    and a missing registry there is a packaging failure that should surface as
    one rather than be papered over with an empty directory.

    Returns:
        A ``feast.repo_config.RepoConfig``. Typed as ``Any`` so that importing
        this module does not require Feast to be importable.

    Raises:
        NetworkSurfaceError: If the configuration text looks remote.
    """
    from feast.repo_config import RepoConfig

    rendered = render_local_config(bundle=bundle, user_data=user_data)
    parsed = yaml.safe_load(rendered)

    if parsed.get("provider") != "local":
        raise NetworkSurfaceError(f"The local feature store requires provider 'local', got {parsed.get('provider')!r}.")
    if "offline_store" in parsed:
        raise NetworkSurfaceError(
            "The local configuration declares an offline store. Local training reads the shared "
            "database directly (D-002); an offline store here is dead configuration whose only "
            "future is a remote value."
        )

    online_path = Path(parsed["online_store"]["path"])
    online_path.parent.mkdir(parents=True, exist_ok=True)

    repo_path = Path(parsed["registry"]["path"]).parent
    return RepoConfig(repo_path=str(repo_path), **parsed)


# ---------------------------------------------------------------------------
# Cloud configuration
# ---------------------------------------------------------------------------


def _postgres_connection() -> dict[str, Any]:
    """Derive Postgres connection settings from the environment.

    Read from :func:`sigil_ml.config.postgres_url`, the same variable
    ``PostgresStore`` already uses, so the feature store and the data store
    cannot end up pointed at different databases.

    Raises:
        RuntimeError: If the URL is unset or carries no database name.
    """
    url = sigil_config.postgres_url()
    if not url:
        raise RuntimeError(
            "SIGIL_POSTGRES_URL is not set. The cloud feature store needs it for both the "
            "offline and the online store (FR-006)."
        )
    parts = urlparse(url)
    database = parts.path.lstrip("/")
    if not database:
        raise RuntimeError(f"SIGIL_POSTGRES_URL names no database: {parts.scheme}://.../<database>")

    settings: dict[str, Any] = {"database": database}
    if parts.hostname:
        settings["host"] = parts.hostname
    if parts.port:
        settings["port"] = parts.port
    if parts.username:
        settings["user"] = unquote(parts.username)
    if parts.password:
        settings["password"] = unquote(parts.password)
    return settings


def load_cloud_repo_config() -> Any:
    """Build the Feast ``RepoConfig`` for the cloud deployment.

    Connection settings are injected here from the environment; the shipped YAML
    declares store types only, and holds no credential.

    Returns:
        A ``feast.repo_config.RepoConfig`` with Postgres on both halves of the
        store and a SQL registry.

    Raises:
        CloudConfigUnreachableError: If the process is not in cloud mode.
        RuntimeError: If the Postgres URL is unset or incomplete.
    """
    from feast.repo_config import RepoConfig

    path = cloud_config_path()
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    connection = _postgres_connection()

    parsed["registry"] = {**parsed.get("registry", {}), "path": sigil_config.postgres_url()}
    parsed["offline_store"] = {**parsed["offline_store"], **connection}
    parsed["online_store"] = {**parsed["online_store"], **connection}
    return RepoConfig(**parsed)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def load_repo_config(**local_overrides: str | Path | None) -> Any:
    """Return the Feast ``RepoConfig`` for the current operating mode.

    Args:
        **local_overrides: Forwarded to :func:`load_local_repo_config` in local
            mode. Passing them in cloud mode is an error rather than a silent
            no-op, so a caller cannot believe it redirected a cloud store.

    Raises:
        UnknownOperatingModeError: If the mode is neither ``local`` nor
            ``cloud``. There is no fallback, and in particular no fallback to
            anything network-capable.
    """
    mode = sigil_config.operating_mode()
    if mode == "local":
        return load_local_repo_config(**local_overrides)
    if mode == "cloud":
        if local_overrides:
            raise UnknownOperatingModeError(
                f"Local directory overrides {sorted(local_overrides)} do not apply in cloud mode."
            )
        return load_cloud_repo_config()
    raise UnknownOperatingModeError(
        f"Unrecognised operating mode {mode!r}. Expected 'local' or 'cloud'. "
        "No default is chosen: the local deployment must never fall back to a network-capable "
        "configuration (C-001)."
    )
