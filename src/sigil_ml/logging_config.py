"""Logging configuration for kenaz-ml.

Configures both console and file logging. Log file is written to
~/.local/share/sigild/logs/kenaz-ml.log alongside sigild's logs.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _log_dir() -> Path:
    """Return the shared sigild logs directory."""
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    log_dir = data_home / "sigild" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for kenaz-ml with console and file output.

    File: ~/.local/share/sigild/logs/kenaz-ml.log (5MB rotate, 3 backups)
    Console: standard uvicorn-style output
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Root sigil_ml logger
    logger = logging.getLogger("sigil_ml")
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers on reload
    if logger.handlers:
        return

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # File handler — rotating, shared logs directory
    log_file = _log_dir() / "kenaz-ml.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("kenaz-ml logging initialized: file=%s level=%s", log_file, level)
