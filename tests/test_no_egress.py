"""Structural proof that the local deployment opens no socket (NFR-004, SC-002).

C-001 is a hard product requirement, not a default: the local install has no
network path to the cloud feature store. Feast *supports* remote registries,
remote online stores and remote providers purely by configuration, so a
guarantee resting on "we did not configure that" is one edit away from being
false — and the edit would look innocuous in review.

This module therefore asserts the negative structurally rather than by
inspection, in three layers:

* **No socket at all.** :func:`no_network` records and blocks *every* socket the
  interpreter creates while the full local flow runs, and the flow is asserted
  to leave the record empty. This is deliberately not an allow-list of known
  upload functions: an allow-list proves only that the paths someone thought of
  are quiet, whereas a structural assertion catches a path nobody anticipated,
  which is the entire point.
* **No network-shaped configuration.** The shipped local YAML is linted against
  every pattern in ``NETWORK_SURFACE_PATTERNS``, so a future edit introducing a
  host or a URL fails this suite rather than shipping.
* **No reachable cloud configuration.** Every file the local flow opens is
  recorded, and the cloud configuration is asserted absent from that record.

The primary mechanism is a :func:`sys.addaudithook` hook. That matters: audit
events are raised from CPython's C implementation of the socket type, so the
hook sees sockets built through ``_socket`` directly, through a reference to
``socket.socket`` captured before the test started, and through any C extension
in Feast's dependency tree. A monkeypatch of the ``socket`` module alone would
miss all three. The monkeypatch layer is kept as an independent second line and
is exercised on its own below, so neither layer can rot unnoticed.
"""

from __future__ import annotations

import _socket
import ast
import socket
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from feast import FeatureStore, FileSource, PushSource
from feast.data_source import PushMode
from sklearn.ensemble import GradientBoostingClassifier

from kenaz_ml.feature_store import config as fsc
from kenaz_ml.feature_store.definitions import stuck_feature_view, task
from kenaz_ml.models.stuck import FEATURE_NAMES, StuckPredictor
from kenaz_ml.modelstore import LocalModelStore

# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

# Every audit event CPython raises on the way out of the machine. `socket.__new__`
# fires when a socket object is constructed at all, which is the assertion this
# package actually wants: not "no connection succeeded" but "no socket existed".
_SOCKET_AUDIT_EVENTS = frozenset(
    {
        "socket.__new__",
        "socket.bind",
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.gethostname",
        "socket.getnameinfo",
        "socket.getservbyname",
        "socket.getservbyport",
        "socket.sendmsg",
        "socket.sendto",
        "socket.sethostname",
    }
)


class EgressAttempted(AssertionError):
    """Raised the instant anything tries to build or use a socket.

    Raising rather than merely recording is what makes the flow run with
    outbound network genuinely *unavailable* rather than merely unused: a code
    path that would have connected fails here instead of succeeding quietly on a
    developer machine that happens to be online.
    """


@dataclass
class Recorder:
    """What happened inside a :func:`no_network` block."""

    socket_attempts: list[str] = field(default_factory=list)
    opened_paths: list[str] = field(default_factory=list)

    def opened(self, name: str) -> list[str]:
        """Return every recorded open whose final path component is ``name``."""
        return [p for p in self.opened_paths if p.rsplit("/", 1)[-1] == name]


# Armed only inside a guard. The audit hook is installed once per process and
# cannot be uninstalled, so it returns on the first line when disarmed.
_recorder: Recorder | None = None
_record_opens = False
_audit_hook_installed = False


def _audit(event: str, args: tuple[Any, ...]) -> None:
    recorder = _recorder
    if recorder is None:
        return
    if event in _SOCKET_AUDIT_EVENTS:
        detail = f"{event}{args!r}"
        recorder.socket_attempts.append(detail)
        raise EgressAttempted(
            f"The local deployment attempted network access: {detail}. "
            "The local install has no network path to the cloud feature store (C-001)."
        )
    if _record_opens and event == "open":
        path = args[0]
        if isinstance(path, (str, bytes, Path)):
            recorder.opened_paths.append(str(path))


@contextmanager
def audit_guard(recorder: Recorder, *, record_opens: bool = False) -> Iterator[None]:
    """Arm the interpreter-level audit hook for the duration of the block."""
    global _recorder, _record_opens, _audit_hook_installed
    if not _audit_hook_installed:
        sys.addaudithook(_audit)
        _audit_hook_installed = True
    previous, previous_opens = _recorder, _record_opens
    _recorder, _record_opens = recorder, record_opens
    try:
        yield
    finally:
        _recorder, _record_opens = previous, previous_opens


@contextmanager
def patch_guard(recorder: Recorder) -> Iterator[None]:
    """Replace the ``socket`` module's constructors and helpers with refusals.

    Independent second line of defence. It is patched onto the ``socket.socket``
    *class* rather than onto the module name, so a module that did
    ``from socket import socket`` at import time is still covered.
    """

    def refuse(label: str) -> Any:
        def guard(*args: Any, **kwargs: Any) -> Any:
            detail = f"{label}(args={args[1:] if label.startswith('socket.socket.') else args})"
            recorder.socket_attempts.append(detail)
            raise EgressAttempted(
                f"The local deployment attempted network access: {detail}. "
                "The local install has no network path to the cloud feature store (C-001)."
            )

        return guard

    originals: list[tuple[Any, str, Any]] = [
        (socket.socket, "__init__", socket.socket.__init__),
        (socket.socket, "connect", socket.socket.connect),
        (socket.socket, "connect_ex", socket.socket.connect_ex),
        (socket, "create_connection", socket.create_connection),
        (socket, "getaddrinfo", socket.getaddrinfo),
        (socket, "socketpair", socket.socketpair),
    ]
    for owner, name, _original in originals:
        label = f"socket.socket.{name}" if owner is socket.socket else f"socket.{name}"
        setattr(owner, name, refuse(label))
    try:
        yield
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


@contextmanager
def no_network(*, record_opens: bool = False) -> Iterator[Recorder]:
    """Run a block with every socket recorded and refused."""
    recorder = Recorder()
    with audit_guard(recorder, record_opens=record_opens), patch_guard(recorder):
        yield recorder


# ---------------------------------------------------------------------------
# The full local flow
# ---------------------------------------------------------------------------

_PUSH_SOURCE_NAME = "stuck_push"
_FEATURE_VALUES = {name: float(index + 1) for index, name in enumerate(FEATURE_NAMES)}


def _trained_stuck_model() -> GradientBoostingClassifier:
    """Fit a tiny stuck classifier. Deliberately built outside any guard."""
    rows = 24
    rng = np.random.default_rng(0)
    x = rng.normal(size=(rows, len(FEATURE_NAMES)))
    y = np.array([index % 2 for index in range(rows)])
    model = GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=0)
    model.fit(x, y)
    return model


def run_local_flow(bundle: Path, user_data: Path, model: GradientBoostingClassifier) -> dict[str, Any]:
    """Execute every local feature operation there is, end to end.

    Load the configuration, apply the registry, reopen the store so the registry
    is genuinely read back from disk rather than served from the object that
    wrote it, push a feature vector into the online store, resolve it, and serve
    a prediction from the resolved values.

    Returns:
        The resolved feature vector and the prediction served from it.
    """
    repo_config = fsc.load_repo_config(bundle=bundle, user_data=user_data)

    batch_source = FileSource(
        name="stuck_batch",
        path=str(bundle / "stuck.parquet"),
        timestamp_field="event_timestamp",
    )
    view = stuck_feature_view(source=PushSource(name=_PUSH_SOURCE_NAME, batch_source=batch_source))

    FeatureStore(config=repo_config).apply([task, view])

    # A second store, built from a second read of the configuration, so the
    # registry is loaded from the file the first store wrote.
    store = FeatureStore(config=fsc.load_repo_config(bundle=bundle, user_data=user_data))
    assert [v.name for v in store.list_feature_views()] == ["stuck_features"]

    now = pd.Timestamp.now(tz="UTC")
    frame = pd.DataFrame(
        {
            "task_id": ["task-1"],
            "event_timestamp": [now],
            "created": [now],
            **{k: [v] for k, v in _FEATURE_VALUES.items()},
        }
    )
    store.push(_PUSH_SOURCE_NAME, frame, to=PushMode.ONLINE)

    resolved = store.get_online_features(
        features=[f"stuck_features:{name}" for name in FEATURE_NAMES],
        entity_rows=[{"task_id": "task-1"}],
    ).to_dict()

    features = {name: resolved[name][0] for name in FEATURE_NAMES}
    predictor = StuckPredictor.from_trained_model(model, store=LocalModelStore(base_dir=user_data / "models"))
    return {"features": features, "prediction": predictor.predict(features)}


@pytest.fixture(scope="module")
def stuck_model() -> GradientBoostingClassifier:
    return _trained_stuck_model()


@pytest.fixture
def local_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A read-only-in-spirit bundle directory and a writable data directory.

    They are separate directories on purpose (FR-013): collapsing them would let
    a feature operation write into what ships as the signed application
    directory and the test would stop noticing.
    """
    monkeypatch.delenv("KENAZ_MODE", raising=False)
    bundle = tmp_path / "bundle"
    user_data = tmp_path / "data"
    bundle.mkdir()
    user_data.mkdir()
    return bundle, user_data


# ---------------------------------------------------------------------------
# The guard is not vacuous
# ---------------------------------------------------------------------------


class TestGuardCatchesEgress:
    """If these fail, every assertion below is worthless."""

    def test_audit_layer_catches_a_raw_socket(self) -> None:
        # `_socket.socket` bypasses the `socket` module entirely, so only the
        # interpreter-level hook can see it. This is why the hook is the primary
        # mechanism rather than a monkeypatch.
        with no_network() as recorder, pytest.raises(EgressAttempted):
            _socket.socket()
        assert recorder.socket_attempts
        assert recorder.socket_attempts[0].startswith("socket.__new__")

    def test_audit_layer_catches_name_resolution(self) -> None:
        recorder = Recorder()
        with audit_guard(recorder), pytest.raises(EgressAttempted):
            socket.getaddrinfo("example.invalid", 80)
        assert any("getaddrinfo" in attempt for attempt in recorder.socket_attempts)

    def test_audit_layer_catches_an_outbound_connection(self) -> None:
        recorder = Recorder()
        with audit_guard(recorder), pytest.raises(EgressAttempted):
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)
        assert recorder.socket_attempts

    def test_patch_layer_is_armed_independently(self) -> None:
        # Exercised with the audit hook disarmed, so the second line of defence
        # is proven to work on its own rather than being shadowed by the first.
        before = (socket.socket.__init__, socket.socket.connect, socket.create_connection)
        recorder = Recorder()
        with patch_guard(recorder):
            with pytest.raises(EgressAttempted):
                socket.socket()
            with pytest.raises(EgressAttempted):
                socket.create_connection(("127.0.0.1", 9))
        assert len(recorder.socket_attempts) == 2
        # ...and the module is put back exactly as it was, so this guard cannot
        # leak into the rest of the suite.
        assert (socket.socket.__init__, socket.socket.connect, socket.create_connection) == before
        socket.socket().close()

    def test_guard_records_nothing_when_nothing_happens(self) -> None:
        with no_network() as recorder:
            pass
        assert recorder.socket_attempts == []


# ---------------------------------------------------------------------------
# NFR-004 / SC-002 — the assertion this package exists for
# ---------------------------------------------------------------------------


class TestNoEgress:
    def test_full_local_flow_opens_no_socket(
        self, local_dirs: tuple[Path, Path], stuck_model: GradientBoostingClassifier
    ) -> None:
        """Not "no known uploader ran" — no socket was created, by any path."""
        bundle, user_data = local_dirs
        with no_network() as recorder:
            run_local_flow(bundle, user_data, stuck_model)
        assert recorder.socket_attempts == [], "The local feature flow attempted network access:\n  " + "\n  ".join(
            recorder.socket_attempts
        )

    def test_local_flow_succeeds_with_network_unavailable(
        self, local_dirs: tuple[Path, Path], stuck_model: GradientBoostingClassifier
    ) -> None:
        """US2 scenario 3: behaviour is unchanged when outbound network is gone."""
        bundle, user_data = local_dirs
        with no_network():
            result = run_local_flow(bundle, user_data, stuck_model)

        assert result["features"] == pytest.approx(_FEATURE_VALUES)
        assert 0.0 <= result["prediction"]["probability"] <= 1.0
        assert result["prediction"]["confidence"] in {"weak", "moderate", "strong"}

    def test_registry_is_read_back_from_disk(self, local_dirs: tuple[Path, Path]) -> None:
        """The flow above is only meaningful if a real registry file is involved."""
        bundle, user_data = local_dirs
        with no_network():
            run_local_flow(bundle, user_data, _trained_stuck_model())
        assert fsc.registry_path(bundle=bundle).is_file()
        assert fsc.online_store_path(user_data=user_data).is_file()


# ---------------------------------------------------------------------------
# FR-004 — the cloud configuration is unreachable, not merely unused
# ---------------------------------------------------------------------------


class TestCloudConfigUnreachable:
    def test_cloud_config_is_never_opened_during_the_local_flow(
        self, local_dirs: tuple[Path, Path], stuck_model: GradientBoostingClassifier
    ) -> None:
        bundle, user_data = local_dirs
        with no_network(record_opens=True) as recorder:
            run_local_flow(bundle, user_data, stuck_model)

        # Vacuity check first: the recorder must actually see YAML opens, or the
        # assertion below would pass on an empty record.
        assert recorder.opened(fsc.LOCAL_CONFIG_FILENAME), "the open recorder saw nothing"
        assert recorder.opened(fsc.CLOUD_CONFIG_FILENAME) == []

    def test_cloud_config_path_is_refused_in_local_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KENAZ_MODE", "local")
        with pytest.raises(fsc.CloudConfigUnreachableError):
            fsc.cloud_config_path()

    def test_cloud_config_path_is_refused_when_mode_is_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KENAZ_MODE", raising=False)
        with pytest.raises(fsc.CloudConfigUnreachableError):
            fsc.cloud_config_path()

    def test_loading_the_cloud_config_is_refused_in_local_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KENAZ_MODE", "local")
        monkeypatch.setenv("KENAZ_POSTGRES_URL", "postgresql://u:p@db.example.com:5432/sigil")
        with pytest.raises(fsc.CloudConfigUnreachableError):
            fsc.load_cloud_repo_config()

    def test_local_loader_names_no_cloud_config(self) -> None:
        """The local loader has no branch that could reach the cloud file.

        Read from the parsed syntax tree with the docstring dropped, so this
        asserts something about the executable code rather than about prose that
        happens to mention the word.
        """
        module = ast.parse(Path(fsc.__file__).read_text(encoding="utf-8"))
        loader = next(
            node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "load_local_repo_config"
        )
        body = loader.body[1:] if ast.get_docstring(loader) else loader.body
        identifiers = {
            node.id if isinstance(node, ast.Name) else node.attr
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, (ast.Name, ast.Attribute))
        }
        literals = {
            node.value
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        for symbol in identifiers | literals:
            assert "cloud" not in symbol.lower(), f"the local loader references {symbol!r}"

    def test_unknown_mode_fails_loudly_with_no_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KENAZ_MODE", "remote")
        with pytest.raises((ValueError, fsc.UnknownOperatingModeError)) as excinfo:
            fsc.load_repo_config()
        assert "remote" in str(excinfo.value)

    def test_unknown_mode_is_refused_even_if_the_mode_switch_loosens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The dispatch carries its own backstop rather than trusting its caller."""
        monkeypatch.setattr(fsc.kenaz_config, "operating_mode", lambda: "hybrid")
        with pytest.raises(fsc.UnknownOperatingModeError):
            fsc.load_repo_config()


# ---------------------------------------------------------------------------
# FR-005 — the configuration lint
# ---------------------------------------------------------------------------


class TestLocalConfigLint:
    @pytest.fixture
    def local_text(self) -> str:
        return (Path(fsc.__file__).parent / fsc.LOCAL_CONFIG_FILENAME).read_text(encoding="utf-8")

    @pytest.mark.parametrize("pattern", fsc.NETWORK_SURFACE_PATTERNS)
    def test_shipped_local_config_matches_no_network_pattern(self, local_text: str, pattern: str) -> None:
        assert pattern not in local_text.lower(), (
            f"{fsc.LOCAL_CONFIG_FILENAME} contains {pattern!r}. The local deployment must reference "
            "no remote registry, remote online store or remote provider (FR-005)."
        )

    def test_the_lint_agrees(self, local_text: str) -> None:
        fsc.assert_no_network_surface(local_text, source=fsc.LOCAL_CONFIG_FILENAME)

    @pytest.mark.parametrize(
        "edit",
        [
            "\nonline_store:\n  host: features.internal\n",
            "\nregistry:\n  path: s3://bucket/registry.db\n",
            "\n# TODO: point at http://cloud/registry when ready\n",
            "\nonline_store:\n  port: 5432\n",
            "\nonline_store:\n  password: hunter2\n",
            "\nauth:\n  token: abc\n",
            "\nregistry:\n  endpoint: registry.internal\n",
        ],
    )
    def test_the_lint_rejects_a_future_edit(self, local_text: str, edit: str) -> None:
        """Including one that is only a comment — a value parked in a comment is
        one keystroke from being live, and reviewers skim comments."""
        with pytest.raises(fsc.NetworkSurfaceError):
            fsc.assert_no_network_surface(local_text + edit, source="edited")

    def test_the_loader_refuses_a_network_shaped_config_at_runtime(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guarantee holds in a user's install, not only in this suite."""
        fake_bundle = tmp_path / "bundle"
        fake_bundle.mkdir()
        doctored = (Path(fsc.__file__).parent / fsc.LOCAL_CONFIG_FILENAME).read_text(encoding="utf-8")
        doctored += "\nonline_store_override:\n  host: features.internal\n"
        (fake_bundle / fsc.LOCAL_CONFIG_FILENAME).write_text(doctored, encoding="utf-8")
        monkeypatch.setattr(fsc, "bundle_dir", lambda: fake_bundle)

        with pytest.raises(fsc.NetworkSurfaceError):
            fsc.load_local_repo_config(bundle=fake_bundle, user_data=tmp_path / "data")

    def test_substituted_directories_may_not_be_urls(self, local_dirs: tuple[Path, Path]) -> None:
        _, user_data = local_dirs
        with pytest.raises(fsc.NetworkSurfaceError):
            fsc.render_local_config(bundle=Path("s3://bucket/registry"), user_data=user_data)


# ---------------------------------------------------------------------------
# FR-003 / FR-005 — the local configuration's shape
# ---------------------------------------------------------------------------


class TestLocalConfigShape:
    @pytest.fixture
    def parsed(self, local_dirs: tuple[Path, Path]) -> dict[str, Any]:
        bundle, user_data = local_dirs
        return yaml.safe_load(fsc.render_local_config(bundle=bundle, user_data=user_data))

    def test_provider_is_local(self, parsed: dict[str, Any]) -> None:
        assert parsed["provider"] == "local"

    def test_registry_is_a_file(self, parsed: dict[str, Any]) -> None:
        assert parsed["registry"]["registry_type"] == "file"

    def test_online_store_is_sqlite(self, parsed: dict[str, Any]) -> None:
        assert parsed["online_store"]["type"] == "sqlite"

    def test_no_offline_store_is_declared(self, parsed: dict[str, Any]) -> None:
        assert "offline_store" not in parsed

    def test_registry_and_online_store_do_not_share_a_root(
        self, parsed: dict[str, Any], local_dirs: tuple[Path, Path]
    ) -> None:
        bundle, user_data = local_dirs
        registry = Path(parsed["registry"]["path"])
        online = Path(parsed["online_store"]["path"])
        assert registry.parent == bundle.resolve()
        assert online.parent == user_data.resolve()
        assert registry.parent != online.parent

    def test_the_registry_directory_is_never_created_by_a_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FR-013: no feature operation writes into the application directory."""
        monkeypatch.delenv("KENAZ_MODE", raising=False)
        absent_bundle = tmp_path / "absent-bundle"
        fsc.load_local_repo_config(bundle=absent_bundle, user_data=tmp_path / "data")
        assert not absent_bundle.exists()


# ---------------------------------------------------------------------------
# T007 — path resolution across both distribution forms
# ---------------------------------------------------------------------------


class TestPathResolution:
    def test_source_install_resolves_beside_this_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)
        resolved = fsc.bundle_dir()
        assert resolved == Path(fsc.__file__).resolve().parent
        # Not merely a plausible path — the shipped configuration is really there.
        assert (resolved / fsc.LOCAL_CONFIG_FILENAME).is_file()

    def test_frozen_bundle_resolves_under_meipass(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
        assert fsc.bundle_dir() == tmp_path / "kenaz_ml" / "feature_store"
        assert fsc.registry_path() == tmp_path / "kenaz_ml" / "feature_store" / "registry.db"

    def test_online_store_follows_the_shared_database(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The online store sits beside data.db so the two cannot drift apart."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        assert fsc.user_data_dir() == tmp_path / "sigild"
        assert fsc.online_store_path().parent == tmp_path / "sigild"


# ---------------------------------------------------------------------------
# FR-006 — the cloud configuration
# ---------------------------------------------------------------------------


class TestCloudConfigShape:
    @pytest.fixture
    def cloud_text(self) -> str:
        return (Path(fsc.__file__).parent / fsc.CLOUD_CONFIG_FILENAME).read_text(encoding="utf-8")

    @pytest.fixture
    def parsed(self, cloud_text: str) -> dict[str, Any]:
        return yaml.safe_load(cloud_text)

    def test_postgres_on_both_halves(self, parsed: dict[str, Any]) -> None:
        assert parsed["offline_store"]["type"] == "postgres"
        assert parsed["online_store"]["type"] == "postgres"

    def test_registry_is_sql(self, parsed: dict[str, Any]) -> None:
        assert parsed["registry"]["registry_type"] == "sql"

    def test_no_hard_coded_credentials(self, parsed: dict[str, Any]) -> None:
        for section in ("registry", "offline_store", "online_store"):
            declared = set(parsed[section])
            assert declared & {"host", "port", "user", "password", "path"} == set(), (
                f"{fsc.CLOUD_CONFIG_FILENAME} hard-codes connection details in {section!r}; "
                "they are injected from the environment at load time."
            )

    def test_no_shared_include_with_the_local_config(self, cloud_text: str) -> None:
        local_text = (Path(fsc.__file__).parent / fsc.LOCAL_CONFIG_FILENAME).read_text(encoding="utf-8")
        for name, text in (("cloud", cloud_text), ("local", local_text)):
            assert "!include" not in text, f"{name} configuration pulls in another file"
            assert "<<:" not in text, f"{name} configuration merges another document"
            body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
            assert "&" not in body and "*" not in body, f"{name} configuration uses YAML anchors"

    def test_connection_settings_come_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KENAZ_POSTGRES_URL", "postgresql://sigil:s%40cret@db.internal:6543/kenaz_features")
        settings = fsc._postgres_connection()
        assert settings == {
            "database": "kenaz_features",
            "host": "db.internal",
            "port": 6543,
            "user": "sigil",
            "password": "s@cret",
        }

    def test_a_missing_postgres_url_fails_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KENAZ_POSTGRES_URL", raising=False)
        with pytest.raises(RuntimeError, match="KENAZ_POSTGRES_URL"):
            fsc._postgres_connection()

    def test_local_overrides_are_refused_in_cloud_mode(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("KENAZ_MODE", "cloud")
        with pytest.raises(fsc.UnknownOperatingModeError):
            fsc.load_repo_config(bundle=tmp_path)


# ===========================================================================
# two-client-engine-01MSK2EN WP06 — the fence grows with the new surface
# ===========================================================================
#
# New tests only; everything above is unchanged. Two things are added:
#
# * FR-018 clause 1 (T027): the loopback bind is an *enforced* invariant. The
#   enforcement lives in ``kenaz_ml.cli`` (``is_loopback_host`` guarding the
#   ``serve`` branch, WP05 T041); these tests prove (a) the default host is
#   127.0.0.1, (b) the guard refuses non-loopback hosts without
#   ``--dev-allow-remote``, and (c) no source path hands the server a host
#   except through that guard (AST scan, with a planted-violation self-test).
# * FR-018 clause 2 (T028): a full request cycle through every new route, plus
#   one lease sweep and the token read, opens zero sockets.
#
# The event-loop trap: asyncio's selector loop builds an AF_UNIX socketpair for
# its self-pipe. It is handled by *ordering*, not by exempting anything: the
# TestClient (and so its loop and self-pipe) is entered before the guard is
# armed, and requests are then driven through that already-running loop. The
# guard itself is unchanged — every AF_INET/AF_INET6/AF_UNIX construction
# inside the block still raises, which ``TestNewSurfaceGuardIsNotVacuous``
# proves by planting violations on a live request path.

import os as _os  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_KENAZ_SRC = Path(__file__).resolve().parents[1] / "src" / "kenaz_ml"


def bind_host_violations(source: str, filename: str) -> list[str]:
    """AST scan: every server-start call must take its host from the guarded CLI arg.

    Flags (1) any ``uvicorn.run``/``uvicorn.Config``/``uvicorn.Server`` call
    whose ``host=`` is anything but ``args.host``, (2) a ``host=`` literal that
    is not a loopback address anywhere in a call, and (3) a ``uvicorn.run``
    call in a function that never calls ``is_loopback_host``.
    """
    tree = ast.parse(source)
    problems: list[str] = []
    for func in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))] + [tree]:
        body_nodes = list(ast.walk(func))
        guarded = any(
            isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "is_loopback_host"
            for n in body_nodes
        )
        for node in body_nodes:
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            host_kw = next((k for k in node.keywords if k.arg == "host"), None)
            if host_kw is not None and isinstance(host_kw.value, ast.Constant):
                if host_kw.value.value not in ("127.0.0.1", "::1", "localhost"):
                    problems.append(f"{filename}:{node.lineno}: {name}(host={host_kw.value.value!r})")
            if name in ("uvicorn.run", "uvicorn.Config", "uvicorn.Server", "run", "Config"):
                if host_kw is None or ast.unparse(host_kw.value) != "args.host":
                    if name.startswith("uvicorn") or host_kw is not None:
                        problems.append(f"{filename}:{node.lineno}: {name} host not taken from guarded args.host")
                if name == "uvicorn.run" and isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) and not guarded:
                    problems.append(f"{filename}:{node.lineno}: {name} in {func.name}() without is_loopback_host")
    return problems


class TestLoopbackBindIsEnforced:
    """FR-018 clause 1. The guard in cli.py enforces; these tests prove it."""

    def _serve(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *args: str) -> list[dict]:
        import uvicorn

        from kenaz_ml import cli

        calls: list[dict] = []
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        monkeypatch.setenv("KENAZ_ML_MODE", "local")
        monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append(kw))
        monkeypatch.setattr(sys, "argv", ["kenaz-ml", "serve", *args])
        cli.main()
        return calls

    def test_default_host_is_loopback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        calls = self._serve(monkeypatch, tmp_path)
        assert calls and calls[0]["host"] == "127.0.0.1"

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", ""])
    def test_non_loopback_is_refused_without_the_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, host: str
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            self._serve(monkeypatch, tmp_path, "--host", host)
        assert excinfo.value.code != 0

    def test_non_loopback_only_with_the_explicit_flag(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        calls = self._serve(monkeypatch, tmp_path, "--host", "0.0.0.0", "--dev-allow-remote")
        assert calls[0]["host"] == "0.0.0.0"

    def test_no_source_path_hands_the_server_an_unguarded_host(self) -> None:
        problems: list[str] = []
        for path in sorted(_KENAZ_SRC.rglob("*.py")):
            problems += bind_host_violations(path.read_text(encoding="utf-8"), str(path.relative_to(_KENAZ_SRC)))
        assert problems == [], "\n".join(problems)

    def test_the_serve_branch_is_guarded(self) -> None:
        source = (_KENAZ_SRC / "cli.py").read_text(encoding="utf-8")
        assert "is_loopback_host(args.host)" in source
        assert source.index("is_loopback_host(args.host)") < source.index("uvicorn.run(")

    @pytest.mark.parametrize(
        "planted",
        [
            'import uvicorn\ndef serve(args):\n    uvicorn.run("kenaz_ml.app:app", host="0.0.0.0", port=1)\n',
            'import uvicorn\ndef serve(args):\n    uvicorn.run("kenaz_ml.app:app", host=args.host, port=1)\n',
            "import uvicorn\ndef serve(args, h):\n    if is_loopback_host(args.host):\n        uvicorn.run('a:b', host=h)\n",
        ],
    )
    def test_the_scan_catches_a_planted_violation(self, planted: str) -> None:
        assert bind_host_violations(planted, "planted.py")


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """A local engine over tmp dirs, its TestClient (and loop) entered *before* any guard."""
    from kenaz_ml import config as kconfig
    from kenaz_ml.app import AppState
    from kenaz_ml.routes import register_routes

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    root = tmp_path / "install"
    monkeypatch.setenv("KENAZ_ML_INSTALL_ROOT", str(root))
    base = tmp_path / "base"
    monkeypatch.setattr(kconfig, "base_models_dir", lambda: base)
    lease = root / "lease"
    lease.mkdir(parents=True)
    token = "egress-fence-token"
    (lease / kconfig.SHUTDOWN_TOKEN_FILENAME).write_text(token + "\n")
    (lease / kconfig.SHUTDOWN_TOKEN_FILENAME).chmod(0o600)

    state = AppState()
    exits: list[str] = []
    state.exit_fn = lambda: exits.append("exit")
    app = FastAPI()
    register_routes(app, state)
    with TestClient(app) as client:
        yield {"client": client, "state": state, "exits": exits, "token": token, "app": app, "base": base}


def _drive_new_surface(engine: dict[str, Any]) -> None:
    """Every new route, one lease sweep, and both shutdown outcomes."""
    from kenaz_ml.advice.contracts import contract_for
    from kenaz_ml.advice.dispatch import build_table
    from kenaz_ml.lifecycle.leases import is_managed
    from tests.fixtures.advice_backends import FIXTURE_KIND, fixture_contract, fixture_features, register_fixtures

    client, state = engine["client"], engine["state"]
    state.dispatch_table = build_table(None, engine["base"])
    register_fixtures(state.dispatch_table)

    routes = {getattr(r, "path", None) for r in engine["app"].routes}
    for path in (
        "/v1/recommend/{kind}",
        "/v1/labels/{kind}",
        "/v1/features",
        "/v1/contracts",
        "/v1/clients/lease",
        "/v1/admin/shutdown",
    ):
        assert path in routes, f"{path} is not registered; the fence would pass vacuously"

    served = client.post(
        f"/v1/recommend/{FIXTURE_KIND}",
        json={"features": fixture_features(), "feature_contract_version": fixture_contract().service_version},
    )
    assert served.status_code == 200
    compact = contract_for("compact_now")
    refused = client.post(
        "/v1/recommend/compact_now",
        json={"features": dict.fromkeys(compact.names, 0.0), "feature_contract_version": compact.service_version},
    )
    assert refused.json()["refusal"]["reason"] == "kind_not_served"

    row = {
        "client": "harness",
        "kind": "compact_now",
        "features_hash": "h1",
        "ts": 1000,
        "revision": 1,
        "feature_contract_version": compact.service_version,
        "features": dict.fromkeys(compact.names, 1.0),
        "features_complete": True,
        "shown": False,
        "confidence": 40,
        "user_action": "dismissed",
    }
    for _ in range(2):  # a batch, then the same batch again
        assert client.post("/v1/labels/compact_now", json={"client": "harness", "rows": [row]}).status_code == 200

    feats = client.post(
        "/v1/features",
        json={
            "client": "kenaz",
            "events": [
                {"event_class": "commit", "ts_ms": 1_700_000_000_000, "session_id": "s"},
                {"event_class": "teleport", "ts_ms": 1},
            ],
        },
    )
    assert feats.status_code == 200 and feats.json()["accepted"] == 1 and feats.json()["refused"] == 1

    assert client.get("/v1/contracts").status_code == 200
    assert (
        client.post(
            "/v1/clients/lease", json={"client": "harness", "pid": _os.getpid(), "client_version": "1"}
        ).status_code
        == 200
    )
    assert client.get("/health").status_code == 200

    state.leases.sweep()  # one sweep iteration (pid liveness via signal 0)
    state.leases.should_exit(is_managed())

    wrong = client.post("/v1/admin/shutdown", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 403
    ok = client.post("/v1/admin/shutdown", headers={"Authorization": f"Bearer {engine['token']}"})
    assert ok.status_code == 202


class TestNewSurfaceOpensNoSocket:
    """FR-018 clause 2 / SC-010."""

    def test_full_request_cycle_through_every_new_route(self, engine: dict[str, Any]) -> None:
        with no_network() as recorder:
            _drive_new_surface(engine)
        assert recorder.socket_attempts == [], "The new surface attempted network access:\n  " + "\n  ".join(
            recorder.socket_attempts
        )
        assert engine["exits"] == ["exit"], "the accepted shutdown never reached the (stubbed) exit hook"

    def test_the_lease_sweep_and_token_read_open_no_socket(self, engine: dict[str, Any]) -> None:
        from kenaz_ml import config as kconfig
        from kenaz_ml.lifecycle.shutdown import check_token

        with no_network() as recorder:
            engine["state"].leases.renew("harness", _os.getpid(), "1")
            engine["state"].leases.sweep()
            assert check_token(engine["token"], kconfig.shutdown_token_path()).ok
            assert not check_token("wrong", kconfig.shutdown_token_path()).ok
        assert recorder.socket_attempts == []


class TestNewSurfaceGuardIsNotVacuous:
    """Planted violations on live new paths must be caught — with the loop already running."""

    def test_an_af_inet_socket_is_still_caught_while_the_loop_runs(self, engine: dict[str, Any]) -> None:
        with no_network() as recorder, pytest.raises(EgressAttempted):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        assert recorder.socket_attempts

    def test_an_af_unix_socketpair_is_still_caught_too(self, engine: dict[str, Any]) -> None:
        # Nothing is exempted for the loop's benefit: a *new* socketpair is caught.
        with no_network() as recorder, pytest.raises(EgressAttempted):
            socket.socketpair()
        assert recorder.socket_attempts

    def test_a_planted_lookup_in_the_features_lane_fails_the_fence(
        self, engine: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kenaz_ml import features_push

        real = features_push.validate_event

        def leaky(event: Any) -> Any:
            socket.getaddrinfo("telemetry.example.invalid", 443)
            return real(event)

        monkeypatch.setattr(features_push, "validate_event", leaky)
        with no_network() as recorder, pytest.raises(EgressAttempted):
            engine["client"].post(
                "/v1/features", json={"client": "k", "events": [{"event_class": "commit", "ts_ms": 1}]}
            )
        assert any("getaddrinfo" in a for a in recorder.socket_attempts)

    def test_a_planted_connection_in_a_new_route_fails_the_fence(self, engine: dict[str, Any]) -> None:
        @engine["app"].post("/v1/planted-leak")
        async def leak() -> dict:
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)
            return {}

        with no_network() as recorder, pytest.raises(EgressAttempted):
            engine["client"].post("/v1/planted-leak")
        assert recorder.socket_attempts

    def test_a_planted_socket_in_label_ingest_fails_the_fence(
        self, engine: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kenaz_ml.advice import label_log

        real = label_log.ingest

        def leaky(*args: Any, **kwargs: Any) -> Any:
            _socket.socket()
            return real(*args, **kwargs)

        from kenaz_ml.advice.dispatch import build_table

        engine["state"].dispatch_table = build_table(None, engine["base"])
        monkeypatch.setattr(label_log, "ingest", leaky)
        with no_network() as recorder, pytest.raises(EgressAttempted):
            engine["client"].post("/v1/labels/compact_now", json={"client": "h", "rows": []})
        assert recorder.socket_attempts
