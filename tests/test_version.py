"""Amendment A5 — real version stamping: pyproject.toml is the single source."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import kenaz_ml
from kenaz_ml import _version

REPO = Path(__file__).resolve().parents[1]


def test_version_comes_from_pyproject() -> None:
    assert kenaz_ml.__version__ == _version.pyproject_version(REPO / "pyproject.toml")
    assert kenaz_ml.__version__ not in ("", _version.UNKNOWN_VERSION)


def test_cli_version_flag_prints_the_bare_version(monkeypatch: pytest.MonkeyPatch, capsys, tmp_path: Path) -> None:
    from kenaz_ml import cli

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["kenaz-ml", "--version"])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == kenaz_ml.__version__


def test_health_root_and_openapi_report_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kenaz_ml.app import create_app

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("KENAZ_ML_INSTALL_ROOT", str(tmp_path / "install"))
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["sidecar_version"] == kenaz_ml.__version__
        assert client.get("/").json()["version"] == kenaz_ml.__version__
        assert client.get("/openapi.json").json()["info"]["version"] == kenaz_ml.__version__


def test_pyproject_reader_requires_the_kenaz_ml_project(tmp_path: Path) -> None:
    other = tmp_path / "pyproject.toml"
    other.write_text('[project]\nname = "something-else"\nversion = "9.9.9"\n')
    assert _version.pyproject_version(other) is None
    mine = tmp_path / "mine.toml"
    mine.write_text(
        '[build-system]\nrequires = ["x"]\n\n[project]\nname = "kenaz-ml"\nversion = "1.4.2"\n\n[tool.x]\nversion = "0"\n'
    )
    assert _version.pyproject_version(mine) == "1.4.2"
    assert _version.pyproject_version(tmp_path / "absent.toml") is None


def test_frozen_reads_the_bundled_version_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "kenaz_ml").mkdir()
    (tmp_path / "kenaz_ml" / "VERSION").write_text("2.3.4\n")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert _version.resolve_version() == "2.3.4"


def test_frozen_falls_back_to_the_onedir_root_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "VERSION").write_text("2.3.5\n")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "_internal"), raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "kameas-launcher"))
    assert _version.resolve_version() == "2.3.5"


def test_frozen_without_a_version_file_is_honestly_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "launcher"))
    assert _version.resolve_version() == _version.UNKNOWN_VERSION
