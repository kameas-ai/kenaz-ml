"""Configuration and path discovery for kameas-ml."""

from __future__ import annotations

import enum
import os
import re
import sys
from pathlib import Path

#: Directory name of the read-only base-model slot, relative to the installed
#: ``sigil_ml`` package. Base artifacts ship *inside* the distribution and are
#: read in place (D-001), so this name appears in the PyInstaller collection
#: rules as well as here; the two must agree.
BASE_MODELS_DIRNAME = "ml-base"

#: Directory name of the retained training set, under :func:`models_dir`.
RETAINED_DIRNAME = "retained"


class ServingMode(str, enum.Enum):
    """Operating mode for the kameas-ml service.

    LOCAL: Default. Poller, SQLite, local models. Current behavior.
    CLOUD: Stateless. No poller, no SQLite, tenant-aware model loading.
    """

    LOCAL = "local"
    CLOUD = "cloud"


def resolve_mode(cli_mode: str | None = None) -> ServingMode:
    """Resolve the serving mode from CLI flag or environment.

    Priority:
      1. cli_mode argument (from --mode flag)
      2. SIGIL_ML_MODE environment variable
      3. Default: LOCAL

    Raises:
        SystemExit: If the provided mode value is invalid.
    """
    raw = cli_mode or os.environ.get("SIGIL_ML_MODE", "local")
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
    :func:`sigil_ml.feature_store.config.bundle_dir` already solves and by the
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
            return Path(meipass) / "sigil_ml" / BASE_MODELS_DIRNAME
        # A frozen build that exposes no _MEIPASS: resolving through __file__
        # would point inside the archive, so anchor on the executable instead.
        return Path(sys.executable).resolve().parent / "sigil_ml" / BASE_MODELS_DIRNAME
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

    Reads from SIGIL_MODE environment variable.
    Defaults to 'local' if not set.
    """
    mode = os.environ.get("SIGIL_MODE", "local").lower()
    if mode not in ("local", "cloud"):
        raise ValueError(f"Invalid SIGIL_MODE: {mode!r}. Must be 'local' or 'cloud'.")
    return mode


def postgres_url() -> str | None:
    """Return the Postgres connection URL, or None if not configured.

    Set via SIGIL_POSTGRES_URL environment variable.
    Required when SIGIL_MODE=cloud.
    """
    return os.environ.get("SIGIL_POSTGRES_URL")


def tenant_id() -> str:
    """Return the tenant identifier for multi-tenant Postgres schemas.

    Set via SIGIL_TENANT environment variable.
    Defaults to 'public' if not set.
    """
    return os.environ.get("SIGIL_TENANT", "public")


def serving_mode() -> str:
    """Return the serving mode: 'local' or 'cloud'. Alias for operating_mode()."""
    return operating_mode()


def s3_bucket() -> str | None:
    """Return the S3 bucket for model storage, or None if not configured."""
    return os.environ.get("SIGIL_S3_BUCKET")


def s3_endpoint_url() -> str | None:
    """Return the S3 endpoint URL (for MinIO), or None for AWS default."""
    return os.environ.get("SIGIL_S3_ENDPOINT_URL")


def aws_region() -> str | None:
    """Return the AWS region, or None for default."""
    return os.environ.get("AWS_REGION")


def model_cache_ttl() -> float:
    """Return the model cache TTL in seconds. Default 300."""
    return float(os.environ.get("SIGIL_MODEL_CACHE_TTL", "300"))


_TENANT_ID_RE = re.compile(r"^[a-z0-9_-]{1,63}$")


def validate_tenant_id(tenant_id: str) -> bool:
    """Return True if tenant_id matches the allowed format.

    Valid: 1-63 characters of lowercase alphanumeric, hyphens, underscores.
    """
    return bool(_TENANT_ID_RE.match(tenant_id))
