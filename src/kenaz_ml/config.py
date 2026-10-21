"""Configuration and path discovery for kenaz-ml."""

from __future__ import annotations

import enum
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import overload

_log = logging.getLogger(__name__)

#: Old-name -> new-name prefix pairs for this product's own environment
#: variables, longest first so ``KENAZ_ML_`` is matched before ``KENAZ_``.
#:
#: NOTE what is deliberately absent: ``SIGILD_*``. Those belong to the sigil
#: daemon, a separate product, and are not renamed by anything here.
_ENV_PREFIX_PAIRS = (("KENAZ_ML_", "SIGIL_ML_"), ("KENAZ_", "SIGIL_"))

#: Legacy names already warned about, so a variable read in a loop warns once.
_ENV_DEPRECATION_WARNED: set[str] = set()


def _legacy_env_name(name: str) -> str | None:
    """Return the pre-rebrand name for ``name``, or None if it has no predecessor."""
    for new_prefix, old_prefix in _ENV_PREFIX_PAIRS:
        if name.startswith(new_prefix):
            return old_prefix + name[len(new_prefix) :]
    return None


@overload
def env(name: str) -> str | None: ...
@overload
def env(name: str, default: str) -> str: ...
@overload
def env(name: str, default: None) -> str | None: ...
def env(name: str, default: str | None = None) -> str | None:
    """Read environment variable ``name``, falling back to its pre-rebrand name.

    The kenaz-ml rebrand renamed this product's own variables
    ``SIGIL_ML_* -> KENAZ_ML_*`` and ``SIGIL_* -> KENAZ_*`` (FR-014). Renaming
    alone would break existing deployments *silently*: an unrecognised
    environment variable is simply ignored, so a container still setting
    ``SIGIL_MODE=cloud`` would quietly fall back to local mode. This shim turns
    that silent fallback into a visible deprecation warning.

    Precedence: the new name wins whenever it is set. The old name is honoured
    only when the new one is unset, and reading it warns, naming both.

    Only variables this product owns are covered. ``SIGILD_PLUGIN_URL`` belongs
    to the sigil daemon and has no ``KENAZ_`` form; ``XDG_DATA_HOME`` and
    ``AWS_REGION`` are third-party conventions. All three are read directly.
    """
    value = os.environ.get(name)
    legacy = _legacy_env_name(name)
    legacy_value = os.environ.get(legacy) if legacy is not None else None

    if legacy is not None and legacy_value is not None and legacy not in _ENV_DEPRECATION_WARNED:
        _ENV_DEPRECATION_WARNED.add(legacy)
        if value is not None:
            _log.warning(
                "kenaz-ml: environment variable %s is deprecated, use %s instead. "
                "Both are set, so %s wins and %s is ignored.",
                legacy,
                name,
                name,
                legacy,
            )
        else:
            _log.warning(
                "kenaz-ml: environment variable %s is deprecated and will be removed, use %s instead.",
                legacy,
                name,
            )

    if value is not None:
        return value
    if legacy_value is not None:
        return legacy_value
    return default


#: Directory name of the read-only base-model slot, relative to the installed
#: ``kenaz_ml`` package. Base artifacts ship *inside* the distribution and are
#: read in place (D-001), so this name appears in the PyInstaller collection
#: rules as well as here; the two must agree.
BASE_MODELS_DIRNAME = "ml-base"

#: Directory name of the retained training set, under :func:`models_dir`.
RETAINED_DIRNAME = "retained"


class ServingMode(str, enum.Enum):  # noqa: UP042 - StrEnum would change str(mode)
    """Operating mode for the kenaz-ml service.

    LOCAL: Default. Poller, SQLite, local models. Current behavior.
    CLOUD: Stateless. No poller, no SQLite, tenant-aware model loading.
    """

    LOCAL = "local"
    CLOUD = "cloud"


def resolve_mode(cli_mode: str | None = None) -> ServingMode:
    """Resolve the serving mode from CLI flag or environment.

    Priority:
      1. cli_mode argument (from --mode flag)
      2. KENAZ_ML_MODE environment variable (or the deprecated SIGIL_ML_MODE)
      3. Default: LOCAL

    Raises:
        SystemExit: If the provided mode value is invalid.
    """
    raw = cli_mode or env("KENAZ_ML_MODE", "local")
    if not raw or not raw.strip():
        raw = "local"
    try:
        return ServingMode(raw.strip().lower())
    except ValueError:
        raise SystemExit(f"Invalid serving mode: {raw!r}. Must be 'local' or 'cloud'.") from None


def _data_home() -> Path:
    """Return the XDG data home, defaulting to ~/.local/share."""
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))


def db_path() -> Path:
    """Return the path to the sigild SQLite database."""
    return _data_home() / "sigild" / "data.db"


# ---------------------------------------------------------------------------
# The harness's local activity store
# ---------------------------------------------------------------------------

#: The absolute path of the harness's local activity database. The spawning
#: client sets it; the engine never derives one. ``serve --activity-db`` wins
#: over it (the CLI bridges the flag through this variable, like ``--mode``).
#: Read with ``os.environ`` on purpose: the name is new and has no legacy
#: ``SIGIL_`` spelling to fall back to.
ACTIVITY_DB_ENV = "KENAZ_ML_ACTIVITY_DB"


class ActivityDbConfigError(ValueError):
    """The activity database setting is unusable; the engine must not start."""


def activity_db_path(cli_value: str | None = None) -> Path | None:
    """Return the configured activity database path, or ``None`` when none is set.

    ``cli_value`` (the ``--activity-db`` flag) wins over ``KENAZ_ML_ACTIVITY_DB``.
    An empty value counts as unset. A relative path is refused rather than
    resolved against whatever the working directory happens to be.

    Raises:
        ActivityDbConfigError: the value is not an absolute path.
    """
    raw = cli_value if cli_value is not None and cli_value.strip() else os.environ.get(ACTIVITY_DB_ENV)
    if raw is None or not raw.strip():
        return None
    path = Path(raw.strip())
    if not path.is_absolute():
        raise ActivityDbConfigError(
            f"{ACTIVITY_DB_ENV} / --activity-db must be an absolute path, got {raw!r}; refusing to start"
        )
    return path


def local_db_path() -> Path:
    """The SQLite file the local deployment reads: the activity DB when set, else sigild's."""
    return activity_db_path() or db_path()


class LocalityError(RuntimeError):
    """An activity database is configured on a path that could move its data off the machine."""


def check_activity_db_locality(*, serving_mode: str | None = None, model_store: object | None = None) -> None:
    """Refuse to run with an activity database anywhere but the local, no-egress path.

    The activity database holds one person's workflow on one machine. It is
    read only by the local deployment: a cloud serving or operating mode, or a
    model store that writes to S3, would put an engine that reads it next to a
    path that leaves the machine. Any of those with an activity database set
    is a misconfiguration, so the engine stops instead of choosing for it.

    Raises:
        ActivityDbConfigError: the activity DB path is not absolute.
        LocalityError: an activity DB is set together with cloud mode or an S3 model store.
    """
    path = activity_db_path()
    if path is None:
        return
    reasons: list[str] = []
    if serving_mode is not None and str(getattr(serving_mode, "value", serving_mode)).lower() == "cloud":
        reasons.append("serving mode is cloud")
    if operating_mode() == "cloud":
        reasons.append("KENAZ_MODE is cloud")
    if model_store is not None and _is_s3_model_store(model_store):
        reasons.append("the model store is S3")
    if reasons:
        raise LocalityError(
            f"{ACTIVITY_DB_ENV} is set but {' and '.join(reasons)}: the local activity database is only read "
            "by the local, on-device deployment. Unset it, or run in local mode with the filesystem model store."
        )


def _is_s3_model_store(model_store: object) -> bool:
    from kenaz_ml.modelstore import CachedModelStore, S3ModelStore

    inner = getattr(model_store, "_inner", None) if isinstance(model_store, CachedModelStore) else None
    return isinstance(model_store, S3ModelStore) or isinstance(inner, S3ModelStore)


#: Minutes of inactivity after which an uncompleted task stops being the active one.
ACTIVE_TASK_WINDOW_ENV = "KENAZ_ACTIVE_TASK_WINDOW_MIN"
DEFAULT_ACTIVE_TASK_WINDOW_MIN = 30


def active_task_window_ms() -> int:
    """The active-task window in ms (``KENAZ_ACTIVE_TASK_WINDOW_MIN``, default 30 minutes).

    A task is active when it is not completed and its ``last_active`` falls
    inside this window. Anything but a positive whole number of minutes falls
    back to the default with a warning.
    """
    raw = os.environ.get(ACTIVE_TASK_WINDOW_ENV)
    minutes = DEFAULT_ACTIVE_TASK_WINDOW_MIN
    if raw is not None and raw.strip():
        text = raw.strip()
        if re.fullmatch(r"[0-9]+", text) and int(text) > 0:
            minutes = int(text)
        else:
            _log.warning(
                "kenaz-ml: %s=%r is not a positive whole number of minutes; using %d",
                ACTIVE_TASK_WINDOW_ENV,
                raw,
                DEFAULT_ACTIVE_TASK_WINDOW_MIN,
            )
    return minutes * 60_000


#: Local retention of the engine's own rows (``ml_predictions``, ``ml_events``).
ACTIVITY_RETENTION_ENV = "KENAZ_ML_ACTIVITY_RETENTION_DAYS"
RETENTION_MIN_DAYS = 1
RETENTION_MAX_DAYS = 3650


@dataclass(frozen=True)
class RetentionSetting:
    """The parsed retention variable.

    ``days`` is set only when ``reason == "valid"``; every other reason means
    "do not prune this run". ``warn`` is whether that outcome deserves a
    warning: everything except a valid value and an explicit ``hold``.
    """

    days: int | None
    reason: str  # valid | hold | missing | empty | invalid | no_activity_db

    @property
    def warn(self) -> bool:
        return self.reason not in ("valid", "hold", "no_activity_db")


def parse_activity_retention(raw: str | None) -> RetentionSetting:
    """Parse ``KENAZ_ML_ACTIVITY_RETENTION_DAYS`` exactly, never clamping and never defaulting.

    - a decimal integer 1..3650 after trimming whitespace: prune rows older than that many days;
    - ``hold`` (any case): skip;
    - missing, empty, or anything else (0, negative, >3650, non-integer): skip, with a warning.
    """
    if raw is None:
        return RetentionSetting(None, "missing")
    text = raw.strip()
    if not text:
        return RetentionSetting(None, "empty")
    if text.lower() == "hold":
        return RetentionSetting(None, "hold")
    if re.fullmatch(r"[0-9]+", text):
        days = int(text)
        if RETENTION_MIN_DAYS <= days <= RETENTION_MAX_DAYS:
            return RetentionSetting(days, "valid")
    return RetentionSetting(None, "invalid")


def activity_retention() -> RetentionSetting:
    """Read and parse the retention variable once, logging one warning when it cannot be used.

    Retention applies only to the harness activity DB. With none configured
    (sigild only, whose ml_* tables are Go-owned) it is off whatever the
    variable says: reason ``"no_activity_db"``, a debug log, never a warning.
    Otherwise a missing or unusable value warns once.
    """
    raw = os.environ.get(ACTIVITY_RETENTION_ENV)
    if not _activity_db_configured():
        _log.debug("kenaz-ml: no activity DB configured; %s ignored, nothing is pruned", ACTIVITY_RETENTION_ENV)
        return RetentionSetting(None, "no_activity_db")
    setting = parse_activity_retention(raw)
    if setting.warn:
        _log.warning(
            "kenaz-ml: %s is %s (%s); local ml_predictions / ml_events are not pruned this run. "
            "Expected a whole number of days %d..%d, or 'hold'.",
            ACTIVITY_RETENTION_ENV,
            setting.reason,
            "unset" if raw is None else repr(raw),
            RETENTION_MIN_DAYS,
            RETENTION_MAX_DAYS,
        )
    return setting


def _activity_db_configured() -> bool:
    """Whether an activity DB is set at all (a relative one counts; startup refuses it elsewhere)."""
    try:
        return activity_db_path() is not None
    except ActivityDbConfigError:
        return True


def models_dir() -> Path:
    """Return the directory for ML model weights, creating it if needed."""
    d = _data_home() / "sigild" / "ml-models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def weights_path(model_name: str) -> Path:
    """Return the path to a specific model's weight file."""
    return models_dir() / f"{model_name}.joblib"


def base_models_dir() -> Path:
    """Return the read-only directory holding shipped base model artifacts.

    Resolves by distribution form, the same problem
    :func:`kenaz_ml.feature_store.config.bundle_dir` already solves and by the
    same route: under a PyInstaller bundle the interpreter exposes the unpacked
    bundle root as ``sys._MEIPASS`` and collected package data sits beneath the
    import path, while in a source or wheel install it sits beside this module.

    **This function does not create the directory, and must not start.** Unlike
    :func:`models_dir` the base slot is read-only: it belongs to the installed
    application, it is covered by the same notarization as the binary (D-001),
    and inside a signed bundle a ``mkdir`` would fail rather than help. It is
    also legitimately absent — no base models have shipped yet, so every install
    today runs with this directory missing. The path is returned regardless;
    whether anything is there is the caller's question, answered with
    ``.exists()`` rather than by this function refusing to answer.
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass is not None:
            return Path(meipass) / "kenaz_ml" / BASE_MODELS_DIRNAME
        # A frozen build that exposes no _MEIPASS: resolving through __file__
        # would point inside the archive, so anchor on the executable instead.
        return Path(sys.executable).resolve().parent / "kenaz_ml" / BASE_MODELS_DIRNAME
    return Path(__file__).resolve().parent / BASE_MODELS_DIRNAME


def retained_data_dir() -> Path:
    """Return the directory holding retained training sets, creating it if needed.

    Lives under :func:`models_dir` because it is user data in the writable slot,
    and is created on demand for the same reason that one is.
    """
    d = models_dir() / RETAINED_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Shared install root (two-client-engine-01MSK2EN WP05, plan D-A1)
# ---------------------------------------------------------------------------

#: Environment variable the spawning client sets to name the shared install
#: root (owner-ruled D-A1). Read through :func:`env`, so the legacy
#: ``SIGIL_ML_INSTALL_ROOT`` spelling is honoured with a deprecation warning.
INSTALL_ROOT_ENV = "KENAZ_ML_INSTALL_ROOT"

#: The client-owned lease directory under the install root.
LEASE_DIRNAME = "lease"

#: The admin-shutdown token file inside ``lease/``. A cross-repo interface: the
#: spawning client (kenaz-harness ``laya-advisors-01LAYA001`` WP12) writes it,
#: user-read-only; kenaz-ml only ever reads it.
SHUTDOWN_TOKEN_FILENAME = "shutdown.token"


def install_root() -> Path:
    """Return the shared engine install root. **Never creates anything.**

    ``KENAZ_ML_INSTALL_ROOT`` when set; otherwise the design doc's stated path —
    ``~/Library/Application Support/kameas/ml`` on macOS,
    ``$XDG_DATA_HOME/kameas/ml`` (``~/.local/share/kameas/ml``) elsewhere. The
    tree belongs to the client (C-004): it creates, verifies and swaps it. A
    developer's standalone ``kenaz-ml serve`` must not litter one, so unlike
    :func:`models_dir` this function does not ``mkdir``.
    """
    override = env(INSTALL_ROOT_ENV)
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "kameas" / "ml"
    return Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")) / "kameas" / "ml"


def lease_dir() -> Path:
    """``<install_root>/lease`` — read-only from this process; never created here."""
    return install_root() / LEASE_DIRNAME


def shutdown_token_path() -> Path:
    """``<install_root>/lease/<SHUTDOWN_TOKEN_FILENAME>`` — read, never written, by kenaz-ml."""
    return lease_dir() / SHUTDOWN_TOKEN_FILENAME


def sigild_plugin_url() -> str:
    """Return the URL for the sigild plugin ingest/capabilities API."""
    return os.environ.get("SIGILD_PLUGIN_URL", "http://127.0.0.1:7775")


def operating_mode() -> str:
    """Return the operating mode: 'local' or 'cloud'.

    Reads from the KENAZ_MODE environment variable (or the deprecated SIGIL_MODE).
    Defaults to 'local' if not set.
    """
    mode = (env("KENAZ_MODE", "local") or "local").lower()
    if mode not in ("local", "cloud"):
        raise ValueError(f"Invalid KENAZ_MODE: {mode!r}. Must be 'local' or 'cloud'.")
    return mode


def postgres_url() -> str | None:
    """Return the Postgres connection URL, or None if not configured.

    Set via the KENAZ_POSTGRES_URL environment variable.
    Required when KENAZ_MODE=cloud.
    """
    return env("KENAZ_POSTGRES_URL")


def tenant_id() -> str:
    """Return the tenant identifier for multi-tenant Postgres schemas.

    Set via the KENAZ_TENANT environment variable.
    Defaults to 'public' if not set.
    """
    return env("KENAZ_TENANT", "public")


def serving_mode() -> str:
    """Return the serving mode: 'local' or 'cloud'. Alias for operating_mode()."""
    return operating_mode()


def s3_bucket() -> str | None:
    """Return the S3 bucket for model storage, or None if not configured."""
    return env("KENAZ_S3_BUCKET")


def s3_endpoint_url() -> str | None:
    """Return the S3 endpoint URL (for MinIO), or None for AWS default."""
    return env("KENAZ_S3_ENDPOINT_URL")


def aws_region() -> str | None:
    """Return the AWS region, or None for default."""
    return os.environ.get("AWS_REGION")


def model_cache_ttl() -> float:
    """Return the model cache TTL in seconds. Default 300."""
    return float(env("KENAZ_MODEL_CACHE_TTL", "300") or "300")


_TENANT_ID_RE = re.compile(r"^[a-z0-9_-]{1,63}$")


def validate_tenant_id(tenant_id: str) -> bool:
    """Return True if tenant_id matches the allowed format.

    Valid: 1-63 characters of lowercase alphanumeric, hyphens, underscores.
    """
    return bool(_TENANT_ID_RE.match(tenant_id))
