#!/usr/bin/env python3
"""Package a frozen onedir as the release zip, identically on every platform.

Usage:
    python scripts/package_bundle.py --onedir dist/kameas-ml --out release/kenaz-ml-<ver>-<target>.zip

Why not ``zip -r`` / ``Compress-Archive``: the two produce different archives.
PowerShell's ``Compress-Archive`` writes entry names with backslashes, which
every non-Windows unzipper (and Go's ``archive/zip``, which the harness
installer uses) treats as literal file names; ``zip`` keeps Unix modes but is
not on the Windows runner. This writes one layout everywhere:

* entry names use forward slashes and start with the onedir's name
  (``kameas-ml/…``), so the archive root is the ``kameas-ml/`` directory the
  installers expect — the same layout as the macOS ``.dmg`` volume;
* Unix mode bits are recorded for every entry (on Windows, where there are
  none, the launcher and shared libraries get 0755 so a cross-platform
  unpacker still marks them executable);
* symlinks are followed (their targets' bytes are stored), never stored as
  links: a zip reader that does not understand symlink entries would
  otherwise write the link text as a file.

Standard library only; it runs on the build runner, never in the engine.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
import zipfile
from pathlib import Path

EXECUTABLE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd", ".exe"}


def unix_mode(path: Path, *, name: str) -> int:
    """The mode bits to record for ``path``: real ones on POSIX, sensible ones on Windows."""
    if os.name != "nt":
        return stat.S_IMODE(path.stat().st_mode)
    executable = Path(name).suffix.lower() in EXECUTABLE_SUFFIXES or "." not in Path(name).name
    return 0o755 if executable else 0o644


def package(onedir: Path, out: Path) -> int:
    onedir = onedir.resolve()
    if not onedir.is_dir():
        raise SystemExit(f"{onedir} is not a directory")
    if not any((onedir / n).exists() for n in ("kameas-ml", "kameas-ml.exe")):
        raise SystemExit(f"{onedir} has no launcher; is it a frozen onedir?")
    out.parent.mkdir(parents=True, exist_ok=True)
    root = onedir.name
    count = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for dirpath, dirnames, filenames in os.walk(onedir, followlinks=True):
            dirnames.sort()
            rel_dir = Path(dirpath).relative_to(onedir)
            for filename in sorted(filenames):
                src = Path(dirpath) / filename
                rel = (rel_dir / filename).as_posix()
                name = f"{root}/{rel}"
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (unix_mode(src, name=name) | stat.S_IFREG) << 16
                with open(src, "rb") as fh:
                    zf.writestr(info, fh.read())
                count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--onedir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    n = package(args.onedir, args.out)
    print(f"wrote {args.out} ({n} files, {args.out.stat().st_size // (1024 * 1024)} MB)")


if __name__ == "__main__":
    sys.exit(main())
