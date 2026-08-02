"""Configuration and path discovery for kenaz-ml."""

from __future__ import annotations

import enum
import logging
import os
import re
import sys
from pathlib import Path

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


class ServingMode(str, enum.Enum):
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
