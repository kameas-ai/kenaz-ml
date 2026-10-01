"""The engine's version — ``pyproject.toml`` is the single source (Amendment A5).

Before A5 every build reported a hard-coded ``0.1.0``, so a spawning client
could not label its ``versions/<semver>/`` directories honestly. Now:

1. **Frozen** (PyInstaller onedir): the ``VERSION`` file the freeze spec writes
   from ``pyproject.toml`` at build time — ``<_MEIPASS>/kenaz_ml/VERSION``, with
   the onedir-root ``VERSION`` (beside the launcher binary) as a second
   carrier a client can read without executing anything.
2. **Source checkout / editable install**: ``pyproject.toml`` read directly, so a
   bumped version is reported without a reinstall.
3. **Installed wheel**: the distribution metadata (written from pyproject).
4. Otherwise ``0+unknown`` — honest, never a made-up release number.

Release discipline: CI refuses a ``v*`` tag whose version differs from
``pyproject.toml``'s, so a tagged build always reports its tag.

Standard library only; nothing here touches the network.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DIST_NAME = "kenaz-ml"
VERSION_FILENAME = "VERSION"
UNKNOWN_VERSION = "0+unknown"

_PROJECT_TABLE = re.compile(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", re.MULTILINE | re.DOTALL)
_KEY = re.compile(r'^\s*{key}\s*=\s*"([^"]+)"\s*$', re.MULTILINE)


def pyproject_version(path: Path) -> str | None:
    """``[project].version`` from ``path`` when its ``[project].name`` is kenaz-ml.

    A small regex reader rather than ``tomllib`` so Python 3.10 (no tomllib)
    reads it identically.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    table = _PROJECT_TABLE.search(text)
    if table is None:
        return None
    body = table.group(1)
    name = re.search(_KEY.pattern.format(key="name"), body, re.MULTILINE)
    version = re.search(_KEY.pattern.format(key="version"), body, re.MULTILINE)
    if name is None or version is None or name.group(1) != DIST_NAME:
        return None
    return version.group(1).strip() or None


def _read_version_file(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def resolve_version() -> str:
    if getattr(sys, "frozen", False):
        candidates = []
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "kenaz_ml" / VERSION_FILENAME)
        candidates.append(Path(sys.executable).resolve().parent / VERSION_FILENAME)
        for candidate in candidates:
            found = _read_version_file(candidate)
            if found:
                return found
        return UNKNOWN_VERSION

    source_tree = Path(__file__).resolve().parents[2] / "pyproject.toml"
    found = pyproject_version(source_tree)
    if found:
        return found

    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(DIST_NAME)
    except PackageNotFoundError:
        return UNKNOWN_VERSION
    except Exception:  # pragma: no cover - metadata backends vary
        return UNKNOWN_VERSION


__version__ = resolve_version()
