"""scripts/package_bundle.py writes the one release-zip layout every installer expects."""

from __future__ import annotations

import importlib.util
import os
import stat
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "package_bundle.py"
#: The freeze output, as `make freeze` leaves it; the onedir's name is the artifact name.
ONEDIR = "dist/kameas-ml"
NAME = Path(ONEDIR).name


def _load():
    spec = importlib.util.spec_from_file_location("package_bundle", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _fake_onedir(root: Path) -> Path:
    onedir = Path(root, ONEDIR)
    (onedir / "_internal" / "nested").mkdir(parents=True)
    (onedir / NAME).write_bytes(b"#!/bin/sh\n")
    (onedir / NAME).chmod(0o755)
    (onedir / "VERSION").write_text("0.1.0\n")
    (onedir / "_internal" / "lib.so").write_bytes(b"\x7fELF")
    (onedir / "_internal" / "lib.so").chmod(0o755)
    (onedir / "_internal" / "nested" / "data.txt").write_text("x")
    if os.name != "nt":
        os.symlink("lib.so", onedir / "_internal" / "lib.link.so")
    return onedir


def test_zip_root_is_the_onedir_with_forward_slashes_and_modes(tmp_path: Path) -> None:
    pb = _load()
    onedir = _fake_onedir(tmp_path)
    out = tmp_path / "release" / "kenaz-ml-0.1.0-test.zip"
    n = pb.package(onedir, out)
    assert n >= 4
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert all(n.startswith(f"{NAME}/") for n in names), names
        assert not any("\\" in n for n in names), names
        assert f"{NAME}/{NAME}" in names
        assert f"{NAME}/_internal/nested/data.txt" in names
        modes = {i.filename: stat.S_IMODE(i.external_attr >> 16) for i in zf.infolist()}
        assert modes[f"{NAME}/{NAME}"] == 0o755
        assert modes[f"{NAME}/_internal/lib.so"] == 0o755
        assert modes[f"{NAME}/VERSION"] & 0o111 == 0
        if os.name != "nt":
            # The symlink is stored as its target's bytes, never as a link.
            info = zf.getinfo(f"{NAME}/_internal/lib.link.so")
            assert not stat.S_ISLNK(info.external_attr >> 16)
            assert zf.read(info) == b"\x7fELF"


def test_refuses_a_directory_that_is_not_a_onedir(tmp_path: Path) -> None:
    pb = _load()
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        pb.package(tmp_path / "empty", tmp_path / "out.zip")
