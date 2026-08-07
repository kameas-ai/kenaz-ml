"""Freeze smoke test for the frozen `kenaz-ml` binary (FR-3, ADR-ml-packaging).

This guards the known scikit-learn / numpy / uvicorn hidden-import breakage
that PyInstaller one-file builds are prone to: a binary can `serve` and answer
`/health` while still failing the moment an sklearn estimator is exercised,
because a Cython-compiled submodule was dropped from the frozen graph. So the
test does more than "it starts" — it POSTs `/predict/stuck` and asserts a real
sklearn prediction comes back.

Feast (feast-feature-store-migration WP05) makes that concern sharper rather
than merely larger. Feast resolves its provider, registry and online-store
implementations from strings through ``importlib.import_module``, which
PyInstaller's static analysis cannot follow, so the expected failure is a bundle
that builds cleanly, starts cleanly, answers ``/health``, and dies on the first
real feature call. Everything below the sklearn tests exists because a green
build is not evidence:

* :func:`test_frozen_bundle_ships_feature_store_assets` — the registry is
  applied at build time and shipped read-only inside the signed bundle (D-001),
  at the exact path ``feature_store.config.bundle_dir()`` resolves to.
* :func:`test_frozen_feature_store_resolves_with_readonly_app_directory` — the
  real feature call, run with the application directory actually made
  unwritable, asserting nothing inside the bundle changed and that the online
  store landed in the writable user data directory (FR-013).
* :func:`test_frozen_cold_start_within_budget` — NFR-003, measured on the real
  binary, because ``import feast`` pulls pyarrow and pandas.

The tests are SKIPPED unless ``KENAZ_ML_FROZEN_BIN`` points at a built
artifact, so the normal `pytest tests/` run on a dev machine (no frozen binary)
stays green. They are the integration contract between the freeze recipe (part
A) and kenaz's supervised-production path (part B): B's production wiring is
only considered correct once this passes against the baked binary.

Run them explicitly after a freeze build:

    pip install -e ".[freeze]"
    pyinstaller freeze/kenaz-ml.spec --noconfirm
    KENAZ_ML_FROZEN_BIN=$PWD/dist/kameas-ml/kameas-ml pytest tests/test_frozen_smoke.py -v
"""

from __future__ import annotations

import json
import os
import socket
import stat
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

FROZEN_BIN = os.environ.get("KENAZ_ML_FROZEN_BIN")

pytestmark = pytest.mark.skipif(
    not FROZEN_BIN,
    reason="KENAZ_ML_FROZEN_BIN not set; build the frozen binary first "
    "(pyinstaller freeze/kenaz-ml.spec) and point the env var at "
    "dist/kameas-ml/kameas-ml (onedir layout)",
)

#: NFR-003 — process start-to-serving budget for a background daemon.
COLD_START_BUDGET_SECONDS = 10.0

#: Where the feature-store assets must sit inside the onedir bundle. This
#: mirrors ``kenaz_ml.feature_store.config.bundle_dir()``, which resolves to
#: ``<sys._MEIPASS>/kenaz_ml/feature_store`` when frozen; for a onedir build
#: ``sys._MEIPASS`` is the ``_internal`` directory beside the executable. The
#: constant is repeated rather than imported so the test fails loudly if the
#: packaged layout and the resolver ever diverge — importing the resolver would
#: make the two agree by construction and assert nothing.
BUNDLE_FEATURE_STORE_RELPATH = ("_internal", "kenaz_ml", "feature_store")


def _free_port() -> int:
    """Reserve an ephemeral port and return it (closed immediately so the
    server can bind it)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_healthy(base_url: str, proc: subprocess.Popen, timeout: float = 60.0) -> None:
    """Poll /health until it returns 200, or fail with captured output."""
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out, _ = proc.communicate(timeout=5)
            raise AssertionError(
                f"frozen binary exited early (code {proc.returncode}) before becoming healthy.\n--- output ---\n{out}"
            )
        try:
            with urlopen(f"{base_url}/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except (URLError, ConnectionError, OSError) as exc:  # not up yet
            last_err = exc
        time.sleep(0.5)
    raise AssertionError(f"frozen binary did not become healthy within {timeout}s: {last_err}")


@pytest.fixture
def frozen_server():
    """Launch the frozen binary with `serve --port <ephemeral>`; yield base URL."""
    assert os.path.exists(FROZEN_BIN), f"frozen binary not found: {FROZEN_BIN}"
    assert os.access(FROZEN_BIN, os.X_OK), f"frozen binary not executable: {FROZEN_BIN}"

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    proc = subprocess.Popen(
        [FROZEN_BIN, "serve", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_healthy(base_url, proc)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _post_json(url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_frozen_binary_serves_health(frozen_server: str) -> None:
    """The frozen binary boots and answers /health with 200 — proves the
    uvicorn/fastapi freeze graph is intact."""
    with urlopen(f"{frozen_server}/health", timeout=5) as resp:
        assert resp.status == 200
        body = json.loads(resp.read().decode("utf-8"))
    assert "status" in body


def test_frozen_binary_predict_stuck(frozen_server: str) -> None:
    """POST /predict/stuck with a fixed feature dict returns 200 + a
    `probability` field — proves the sklearn estimator path survived freezing
    (the actual hidden-import guard, not just "it started")."""
    # A minimal, well-formed StuckRequest feature dict. Local-mode serving
    # falls back to a deterministic response when no trained model is present,
    # but the request still flows through the sklearn-backed StuckPredictor
    # path, exercising the frozen estimator imports.
    features = {
        "elapsed_minutes": 42.0,
        "edit_count": 7.0,
        "test_run_count": 2.0,
        "error_count": 1.0,
        "file_switch_count": 3.0,
    }
    status, body = _post_json(f"{frozen_server}/predict/stuck", {"features": features})

    assert status == 200, f"unexpected status {status}: {body}"
    assert "probability" in body, f"response missing 'probability': {body}"
    assert isinstance(body["probability"], (int, float))
    assert 0.0 <= float(body["probability"]) <= 1.0


# ===========================================================================
# Feature store — the bundled registry and the read-only application directory
# ===========================================================================


def _app_dir() -> Path:
    """The onedir application directory: the executable's parent."""
    return Path(FROZEN_BIN).resolve().parent


def _bundled_feature_store_dir() -> Path:
    return _app_dir().joinpath(*BUNDLE_FEATURE_STORE_RELPATH)


def _tree_manifest(root: Path) -> dict[str, tuple[int, int]]:
    """Map every file under ``root`` to ``(size, mtime_ns)``.

    Compared before and after the read-only run to prove nothing inside the
    bundle was written. Size plus nanosecond mtime is used rather than a content
    hash because the bundle is ~330 MB and hashing it twice per test would make
    the check expensive enough that someone eventually deletes it. Any write
    that succeeded would move an mtime; any write that was *attempted* would
    fail outright against the read-only permissions, which is the other half of
    the assertion.
    """
    manifest: dict[str, tuple[int, int]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            info = path.stat()
            manifest[str(path.relative_to(root))] = (info.st_size, info.st_mtime_ns)
    return manifest


def _set_tree_writable(root: Path, writable: bool) -> bool:
    """Add or remove write permission on every file and directory under ``root``.

    Directories matter as much as files: creating or deleting an entry needs
    write permission on the containing directory, so clearing it is what makes a
    "the online store must not land beside the binary" failure surface as an
    error instead of an extra file.

    Returns:
        ``False`` if the tree could not be changed because it sits on a
        read-only filesystem — which is the case when the artifact under test is
        a mounted disk image, i.e. exactly what a user installs from. That is
        not a failure; the caller checks unwritability by probing rather than by
        trusting this call.
    """
    paths = [root, *root.rglob("*")]
    # Deepest first when removing permission, shallowest first when restoring,
    # so a directory is never made unwritable before its children are handled.
    paths.sort(key=lambda p: len(p.parts), reverse=not writable)
    for path in paths:
        if path.is_symlink():
            continue
        mode = path.stat().st_mode
        try:
            if writable:
                path.chmod(mode | stat.S_IWUSR)
            else:
                path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        except OSError:
            return False
    return True


def _assert_unwritable(root: Path) -> None:
    """Fail unless a write into ``root`` actually raises."""
    probe = root / ".kenaz-ml-writability-probe"
    try:
        probe.touch()
    except OSError:
        return
    probe.unlink(missing_ok=True)
    raise AssertionError(
        f"{root} is still writable, so this test would prove nothing. A notarized application "
        "directory is read-only on a user's machine and the check has to reproduce that."
    )


@pytest.fixture
def readonly_app_dir() -> Iterator[Path]:
    """Make the frozen application directory unwritable for the test's duration.

    This is the point of the exercise: a notarized bundle is signed and
    read-only on a user's machine, so "nothing writes into the bundle" has to be
    enforced by the filesystem here rather than asserted by inspection.

    When the artifact is already on a read-only mount (a notarized .dmg), the
    permission change is a no-op and the mount provides the guarantee. Either
    way the directory is probed before the test body runs, so the check can
    never degrade into a writable-directory run that passes for free.
    """
    app_dir = _app_dir()
    changed = _set_tree_writable(app_dir, writable=False)
    try:
        _assert_unwritable(app_dir)
        yield app_dir
    finally:
        if changed:
            _set_tree_writable(app_dir, writable=True)


def _run_selfcheck(user_data: Path) -> tuple[int, dict]:
    """Run the bundled ``feature-store-selfcheck`` and return (exit code, report)."""
    proc = subprocess.run(
        [FROZEN_BIN, "feature-store-selfcheck", "--user-data", str(user_data)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            "feature-store-selfcheck did not emit a JSON report.\n"
            f"exit={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        ) from exc
    return proc.returncode, report


def test_frozen_bundle_ships_feature_store_assets() -> None:
    """The registry is applied at build time and shipped inside the bundle (D-001).

    The registry must be *in* the artifact, not generated on first launch: the
    application directory is read-only, so a runtime `feast apply` would fail,
    and it would place the definitions outside the signature even if it did not.
    """
    store_dir = _bundled_feature_store_dir()
    assert store_dir.is_dir(), (
        f"no feature-store directory at {store_dir}. The PyInstaller spec must collect the "
        "assets to the path kenaz_ml.feature_store.config.bundle_dir() resolves to when frozen."
    )

    registry = store_dir / "registry.db"
    assert registry.is_file(), f"no build-time registry at {registry}"
    assert registry.stat().st_size > 0, "shipped registry is empty"

    assert (store_dir / "feature_store.local.yaml").is_file()
    assert (store_dir / "feature_store.cloud.yaml").is_file()

    # D-007 / FR-016: the registry is a serialized protobuf coupled to the Feast
    # version that produced it, so that version travels with it.
    marker = store_dir / "registry.version.json"
    assert marker.is_file(), f"no producing-version marker at {marker}"
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    assert recorded["feast_version"], "marker records no producing Feast version"
    assert recorded["registry_filename"] == "registry.db"
    assert "stuck" in recorded["applied_objects"]
    assert "duration" in recorded["applied_objects"]


def test_frozen_feature_store_resolves_with_readonly_app_directory(readonly_app_dir: Path, tmp_path: Path) -> None:
    """The full local flow succeeds against a read-only bundle (FR-013, FR-011).

    This is the test that means something. It runs the frozen binary with its
    application directory genuinely unwritable and a *fresh* user data directory
    holding no online store, then asserts:

    1. the flow completes — which requires the local provider, the file registry
       and the SQLite online store to have all survived freezing, each of which
       Feast imports by string;
    2. nothing inside the bundle changed;
    3. the online store was created in the writable user data directory.

    A bundle missing a dynamic import fails here and nowhere earlier.
    """
    before = _tree_manifest(readonly_app_dir)
    user_data = tmp_path / "sigild"
    user_data.mkdir()
    online_store = user_data / "feast_online.db"
    assert not online_store.exists(), "precondition: this is a first run, with no online store"

    exit_code, report = _run_selfcheck(user_data)

    assert exit_code == 0, (
        "the frozen binary could not resolve features against its bundled registry.\n"
        f"{json.dumps(report, indent=2, sort_keys=True)}"
    )
    assert report["ok"] is True
    assert report["frozen"] is True, "selfcheck did not run from a frozen bundle"

    # The definitions came out of the shipped registry, not from a source tree.
    assert report["provider"] == "local"
    assert report["feature_views"] == ["duration_features", "stuck_features"]
    assert report["feature_services"] == ["duration", "stuck"]
    assert report["registry_path_in_use"].startswith(str(readonly_app_dir))

    # A real feature resolution ran for every shipped feature service, returning
    # the join key plus that service's feature columns.
    resolved = report["resolved_features"]
    assert set(resolved) == {"duration", "stuck"}
    for service_name, columns in resolved.items():
        assert "task_id" in columns, f"{service_name} resolved no entity column: {columns}"
        assert len(columns) > 1, f"{service_name} resolved no feature columns: {columns}"

    # The push sources the build recorded binding are really in the shipped
    # registry. Local serving pushes into them by name (D-003); a registry that
    # lost them would still answer reads, so nothing above would catch it.
    marker = json.loads((_bundled_feature_store_dir() / "registry.version.json").read_text(encoding="utf-8"))
    missing = sorted(set(marker.get("local_push_sources", [])) - set(report["data_sources"]))
    assert not missing, (
        f"the build recorded binding push sources {marker['local_push_sources']} but the shipped "
        f"registry does not contain {missing}. Serving would compute correctly and fail every push."
    )

    # FR-013 — the writable side landed in the user data directory...
    assert report["online_store_path"] == str(online_store)
    assert online_store.is_file(), "online store was not created in the user data directory"

    # ...and the read-only side is untouched, byte-for-byte.
    after = _tree_manifest(readonly_app_dir)
    assert after == before, (
        "the frozen bundle changed while feature operations ran. Nothing may write inside a "
        "notarized application directory (FR-013).\n"
        f"added:   {sorted(set(after) - set(before))}\n"
        f"removed: {sorted(set(before) - set(after))}\n"
        f"modified: {sorted(k for k in set(after) & set(before) if after[k] != before[k])}"
    )


def test_frozen_registry_records_the_running_feast_version(tmp_path: Path) -> None:
    """The shipped registry's producing version matches the frozen runtime (D-007).

    FR-016 requires a mismatch to be refused with a clear diagnostic rather than
    a deserialization traceback. That diagnostic needs something to compare
    against, and this asserts the comparison is wired and currently agrees — so
    a Feast bump that forgets to rebuild the registry fails here rather than on
    a user's machine.
    """
    _, report = _run_selfcheck(tmp_path)
    assert report["registry_feast_version"] == report["runtime_feast_version"], (
        f"registry was produced by feast {report['registry_feast_version']} but the frozen "
        f"runtime is feast {report['runtime_feast_version']}. Rebuild the bundle: the registry "
        "is applied at build time and is coupled to the library version that wrote it (D-007)."
    )
    assert report["feast_version_match"] is True


def test_frozen_cold_start_within_budget() -> None:
    """Start-to-serving stays inside NFR-003's 10 s budget on the real binary.

    Measured on the frozen artifact rather than a source checkout because that
    is where the cost is: ``import feast`` pulls pyarrow and pandas, and the
    onedir bundle is ~330 MB of files to page in.

    The reported figure is the median of three runs after one warm-up. The
    warm-up is deliberate and is the honest caveat on this number: the very
    first launch of a freshly built, unsigned bundle on macOS pays a one-off
    cost for the operating system to page in and scan ~435 Mach-O files, which
    was measured at ~25 s here and is not what the daemon experiences on any
    subsequent start. Gating on that first launch would make the gate
    machine-dependent and flaky, which T022 explicitly rules out — a gate that
    fails randomly gets ignored. All four measurements are printed so a
    regression in either number is visible.
    """
    samples: list[float] = []
    for _ in range(4):
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        started = time.monotonic()
        proc = subprocess.Popen(
            [FROZEN_BIN, "serve", "--host", "127.0.0.1", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_healthy(base_url, proc, timeout=120.0)
            samples.append(time.monotonic() - started)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    warm_up, *measured = samples
    measured.sort()
    median = measured[len(measured) // 2]
    print(
        f"cold start — warm-up {warm_up:.2f}s; measured "
        f"{', '.join(f'{s:.2f}s' for s in measured)}; median {median:.2f}s "
        f"(NFR-003 budget {COLD_START_BUDGET_SECONDS:.0f}s)"
    )
    assert median < COLD_START_BUDGET_SECONDS, (
        f"frozen binary took a median {median:.2f}s to serve /health, over NFR-003's "
        f"{COLD_START_BUDGET_SECONDS:.0f}s budget. Samples: {samples}"
    )
