"""two-client-engine-01MSK2EN WP05 — leases, self-termination, shutdown, bind guard.

Both directions of the D-A2 asymmetry are tested separately: fail-OPEN on
availability (no ``lease/`` directory: the engine starts, serves, and never
self-terminates) and fail-CLOSED on shutdown authorization (only the correct
token exits the process). No test sleeps anywhere near 120 s: the unit tests
use an injected clock, and the integration tests shrink the windows through
the D-A6 environment variables.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from kenaz_ml import config
from kenaz_ml.lifecycle import leases as leases_mod
from kenaz_ml.lifecycle import shutdown as shutdown_mod
from kenaz_ml.lifecycle.leases import LeaseTable, is_managed, pid_alive

TOKEN = "s3cret-token-value-0123456789"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------------------
# T021 — install-root configuration (D-A1)
# ---------------------------------------------------------------------------


class TestInstallRoot:
    def test_env_var_wins_and_nothing_is_created(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        root = tmp_path / "root"
        monkeypatch.setenv("KENAZ_ML_INSTALL_ROOT", str(root))
        assert config.install_root() == root
        assert config.lease_dir() == root / "lease"
        assert config.shutdown_token_path() == root / "lease" / config.SHUTDOWN_TOKEN_FILENAME
        assert not root.exists(), "install_root()/lease_dir()/shutdown_token_path() must not create anything"

    def test_legacy_env_name_is_honoured(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KENAZ_ML_INSTALL_ROOT", raising=False)
        monkeypatch.setenv("SIGIL_ML_INSTALL_ROOT", str(tmp_path / "legacy"))
        assert config.install_root() == tmp_path / "legacy"

    def test_macos_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("KENAZ_ML_INSTALL_ROOT", raising=False)
        monkeypatch.delenv("SIGIL_ML_INSTALL_ROOT", raising=False)
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("HOME", str(tmp_path))
        assert config.install_root() == tmp_path / "Library" / "Application Support" / "kameas" / "ml"

    def test_linux_default_follows_xdg(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("KENAZ_ML_INSTALL_ROOT", raising=False)
        monkeypatch.delenv("SIGIL_ML_INSTALL_ROOT", raising=False)
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
        assert config.install_root() == tmp_path / "xdg" / "kameas" / "ml"
        assert not (tmp_path / "xdg").exists()


# ---------------------------------------------------------------------------
# T022 — leases and the pid-liveness sweep
# ---------------------------------------------------------------------------


class TestLeaseTable:
    def test_dead_pid_is_discarded_in_one_sweep_live_pid_is_kept(self) -> None:
        alive = {111: True, 222: True}
        table = LeaseTable(clock=FakeClock(), pid_alive_fn=lambda pid: alive.get(pid, False))
        table.renew("harness", 111, "1.0")
        table.renew("kenaz", 222, "2.0")
        assert table.live_count() == 2

        alive[111] = False
        discarded = table.sweep()
        assert [lease.client for lease in discarded] == ["harness"]
        assert [lease.client for lease in table.explicit_leases()] == ["kenaz"]
        table.sweep()
        assert [lease.client for lease in table.explicit_leases()] == ["kenaz"]

    def test_renewal_is_idempotent(self) -> None:
        clock = FakeClock()
        table = LeaseTable(clock=clock, pid_alive_fn=lambda pid: True)
        table.renew("harness", 111, "1.0")
        clock.now += 30
        table.renew("harness", 111, "1.0")
        assert len(table.explicit_leases()) == 1
        assert table.explicit_leases()[0].renewed_at == clock.now

    def test_a_real_killed_process_is_swept(self) -> None:
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            table = LeaseTable()
            table.renew("harness", proc.pid, "1.0")
            table.sweep()
            assert len(table.explicit_leases()) == 1
        finally:
            proc.kill()
            proc.wait()
        table.sweep()
        assert table.explicit_leases() == []

    def test_per_client_lease_count_is_capped_oldest_replaced(self) -> None:
        clock = FakeClock()
        table = LeaseTable(clock=clock, pid_alive_fn=lambda pid: True)
        cap = leases_mod.MAX_LEASES_PER_CLIENT
        for pid in range(100, 100 + cap):
            table.renew("harness", pid, "1")
            clock.now += 1
        table.renew("harness", 100, "1")  # renewing an existing lease never evicts
        clock.now += 1
        assert len(table.explicit_leases()) == cap
        table.renew("harness", 999, "1")  # at the cap: the least-recently-renewed (pid 101) goes
        pids = {lease.pid for lease in table.explicit_leases()}
        assert len(pids) == cap and 999 in pids and 101 not in pids and 100 in pids
        table.renew("kenaz", 5, "1")  # another client's leases are unaffected
        assert len(table.explicit_leases()) == cap + 1

    @pytest.mark.parametrize("pid", [0, -1, -12345, 10**9, True])
    def test_absurd_pids_are_never_signalled(self, pid: int) -> None:
        assert pid_alive(pid) is False

    def test_this_process_is_alive(self) -> None:
        assert pid_alive(os.getpid()) is True


class TestIdleTimer:
    def _table(self, clock: FakeClock, **kw: float) -> LeaseTable:
        return LeaseTable(clock=clock, pid_alive_fn=lambda pid: False, implicit_lease_sec=90, idle_exit_sec=120, **kw)

    def test_zero_leases_for_the_window_exits_when_managed(self) -> None:
        clock = FakeClock()
        table = self._table(clock)
        clock.now += 119
        assert not table.should_exit(True)
        clock.now += 1
        assert table.should_exit(True)

    def test_a_slow_startup_does_not_spend_the_window(self) -> None:
        # Review fix: startup runs before uvicorn accepts a connection, so no
        # client could lease during it. Serving begins -> the countdown restarts.
        clock = FakeClock()
        table = self._table(clock)
        clock.now += 300  # a startup rebuild longer than the whole window
        assert table.should_exit(True)
        table.restart_countdown()
        assert not table.should_exit(True)
        clock.now += 119
        assert not table.should_exit(True)
        clock.now += 1
        assert table.should_exit(True)

    def test_degraded_mode_never_self_terminates(self) -> None:
        clock = FakeClock()
        table = self._table(clock)
        clock.now += 10_000
        assert not table.should_exit(False)

    def test_health_polling_alone_keeps_the_engine_alive(self) -> None:
        clock = FakeClock()
        table = self._table(clock)
        for _ in range(20):  # 20 minutes of 60 s polls, never an explicit lease
            table.note_health_poll()
            clock.now += 60
            table.sweep()
            assert not table.should_exit(True)
        # Polls stop: live for the 90 s implicit window, then 120 s idle.
        last_poll = clock.now - 60
        clock.now = last_poll + 90 + 119
        assert not table.should_exit(True)
        clock.now = last_poll + 90 + 120
        assert table.should_exit(True)

    def test_crashed_client_starts_the_countdown(self) -> None:
        clock = FakeClock()
        alive = {111: True}
        table = LeaseTable(clock=clock, pid_alive_fn=lambda pid: alive[pid], implicit_lease_sec=90, idle_exit_sec=120)
        table.renew("harness", 111, "1.0")
        clock.now += 500
        table.sweep()
        assert not table.should_exit(True)
        alive[111] = False
        table.sweep()
        clock.now += 119
        assert not table.should_exit(True)
        clock.now += 1
        assert table.should_exit(True)

    def test_surviving_client_keeps_the_engine_alive(self) -> None:
        clock = FakeClock()
        alive = {111: True, 222: True}
        table = LeaseTable(clock=clock, pid_alive_fn=lambda pid: alive[pid], idle_exit_sec=120)
        table.renew("harness", 111, "1.0")
        table.renew("kenaz", 222, "1.0")
        alive[111] = False
        table.sweep()
        clock.now += 1000
        table.sweep()
        assert not table.should_exit(True)

    def test_windows_are_env_overridable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KENAZ_ML_IDLE_EXIT_SEC", "7")
        monkeypatch.setenv("KENAZ_ML_IMPLICIT_LEASE_SEC", "3")
        table = LeaseTable()
        assert table.idle_exit_sec == 7 and table.implicit_lease_sec == 3
        monkeypatch.delenv("KENAZ_ML_IDLE_EXIT_SEC")
        assert LeaseTable().idle_exit_sec == 120


# ---------------------------------------------------------------------------
# T023 — self-termination end to end (short windows), and training drain
# ---------------------------------------------------------------------------


@pytest.fixture
def exits(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(shutdown_mod, "request_process_exit", lambda: calls.append("exit"))
    return calls


@pytest.fixture
def engine_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    root = tmp_path / "install"
    monkeypatch.setenv("KENAZ_ML_INSTALL_ROOT", str(root))
    monkeypatch.setenv("KENAZ_ML_IDLE_EXIT_SEC", "0.4")
    monkeypatch.setenv("KENAZ_ML_IMPLICIT_LEASE_SEC", "0.2")
    monkeypatch.setenv("KENAZ_ML_LEASE_SWEEP_SEC", "0.05")
    return root


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_managed_engine_with_no_lease_self_terminates(engine_env: Path, exits: list[str]) -> None:
    (engine_env / "lease").mkdir(parents=True)
    from kenaz_ml.app import create_app

    with TestClient(create_app()):
        assert _wait_for(lambda: exits == ["exit"]), "managed engine never self-terminated"


def test_the_countdown_restarts_when_serving_begins(
    engine_env: Path, exits: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from kenaz_ml.app import create_app
    from kenaz_ml.lifecycle.leases import LeaseTable

    calls: list[str] = []
    real = LeaseTable.restart_countdown
    monkeypatch.setattr(LeaseTable, "restart_countdown", lambda self: (calls.append("restart"), real(self)))
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        assert _wait_for(lambda: calls == ["restart"]), "the lifecycle loop never restarted the countdown"


def test_degraded_engine_starts_serves_and_never_self_terminates(engine_env: Path, exits: list[str]) -> None:
    assert not engine_env.exists()
    from kenaz_ml.app import create_app

    with TestClient(create_app()) as client:
        time.sleep(1.0)  # 2.5x the shrunken idle window
        assert client.get("/health").status_code == 200
        assert exits == []
    assert not engine_env.exists(), "the engine created the client-owned install root"


def test_health_polling_keeps_a_managed_engine_alive(engine_env: Path, exits: list[str]) -> None:
    (engine_env / "lease").mkdir(parents=True)
    from kenaz_ml.app import create_app

    with TestClient(create_app()) as client:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            assert client.get("/health").status_code == 200
            time.sleep(0.05)
        assert exits == []
        assert _wait_for(lambda: exits == ["exit"]), "polling stopped but the engine never left"


def test_drain_waits_for_in_flight_training() -> None:
    calls: list[float] = []
    state = SimpleNamespace(training_in_progress=True, poller=None, exit_fn=None)

    async def run() -> None:
        async def finish_training() -> None:
            await asyncio.sleep(0.2)
            state.training_in_progress = False

        asyncio.get_running_loop().create_task(finish_training())
        started = time.monotonic()
        await shutdown_mod.drain_and_exit(
            state, exit_fn=lambda: calls.append(time.monotonic() - started), poll_sec=0.02
        )

    asyncio.run(run())
    assert len(calls) == 1 and calls[0] >= 0.2


def test_drain_is_bounded(caplog: pytest.LogCaptureFixture) -> None:
    calls: list[str] = []
    stopped: list[str] = []
    state = SimpleNamespace(training_in_progress=True, poller=SimpleNamespace(stop=lambda: stopped.append("x")))
    with caplog.at_level(logging.WARNING):
        asyncio.run(
            shutdown_mod.drain_and_exit(state, exit_fn=lambda: calls.append("exit"), drain_sec=0.1, poll_sec=0.02)
        )
    assert calls == ["exit"] and stopped == ["x"]
    assert any("training still running" in r.getMessage() for r in caplog.records)


def test_is_managed_follows_the_lease_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KENAZ_ML_INSTALL_ROOT", str(tmp_path))
    assert not is_managed()
    (tmp_path / "lease").mkdir()
    assert is_managed()


# ---------------------------------------------------------------------------
# T024 — token-authorized shutdown: fail closed
# ---------------------------------------------------------------------------


def _write_token(root: Path, value: str = TOKEN) -> Path:
    lease = root / "lease"
    lease.mkdir(parents=True, exist_ok=True)
    path = lease / config.SHUTDOWN_TOKEN_FILENAME
    path.write_text(value + "\n")
    path.chmod(0o600)
    return path


@pytest.fixture
def shutdown_client(engine_env: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Long idle window here: these tests are about the token, not the timer.
    monkeypatch.setenv("KENAZ_ML_IDLE_EXIT_SEC", "3600")
    from kenaz_ml.app import create_app

    with TestClient(create_app()) as client:
        yield client


def _shutdown(client: TestClient, token: str | None):
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    return client.post("/v1/admin/shutdown", headers=headers)


def test_correct_token_shuts_down(shutdown_client: TestClient, engine_env: Path, exits: list[str]) -> None:
    _write_token(engine_env)
    resp = _shutdown(shutdown_client, TOKEN)
    assert resp.status_code == 202
    assert resp.json() == {"status": "shutting_down"}
    assert _wait_for(lambda: exits == ["exit"])


@pytest.mark.parametrize(
    ("setup", "presented", "status", "reason"),
    [
        ("token", None, 401, "no_token_presented"),
        ("token", "wrong-token", 403, "token_mismatch"),
        ("lease_only", TOKEN, 403, "token_file_missing"),
        ("empty", TOKEN, 403, "token_file_empty"),
        ("nothing", TOKEN, 403, "lease_dir_missing"),
    ],
)
def test_every_negative_case_leaves_the_engine_running(
    shutdown_client: TestClient,
    engine_env: Path,
    exits: list[str],
    caplog: pytest.LogCaptureFixture,
    setup: str,
    presented: str | None,
    status: int,
    reason: str,
) -> None:
    if setup == "token":
        _write_token(engine_env)
    elif setup == "lease_only":
        (engine_env / "lease").mkdir(parents=True)
    elif setup == "empty":
        _write_token(engine_env, "")
    with caplog.at_level(logging.DEBUG):
        resp = _shutdown(shutdown_client, presented)
    assert resp.status_code == status
    assert reason in resp.json()["detail"]
    time.sleep(0.1)
    assert exits == []
    assert shutdown_client.get("/health").status_code == 200
    everything = resp.text + "\n".join(r.getMessage() for r in caplog.records)
    assert TOKEN not in everything and "wrong-token" not in everything


def test_unreadable_token_file_refuses(tmp_path: Path) -> None:
    path = _write_token(tmp_path)
    path.chmod(0)
    try:
        if os.access(path, os.R_OK):  # running as root
            pytest.skip("cannot make a file unreadable as this user")
        assert shutdown_mod.check_token(TOKEN, path).reason == "token_file_unreadable"
    finally:
        path.chmod(0o600)


def test_permissive_token_file_warns_but_is_honoured(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _write_token(tmp_path)
    path.chmod(0o644)
    with caplog.at_level(logging.WARNING):
        assert shutdown_mod.check_token(TOKEN, path).ok
    assert any("group/world" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("header", ["", "Bearer", "Basic abc", "bearer  "])
def test_malformed_authorization_headers_present_no_token(header: str) -> None:
    assert shutdown_mod.presented_token(header) is None


# ---------------------------------------------------------------------------
# T041 — loopback bind guard
# ---------------------------------------------------------------------------


@pytest.fixture
def serve_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    import uvicorn

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    calls: list[dict] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append(kw))
    monkeypatch.setenv("KENAZ_ML_MODE", "local")  # cli writes it; setenv makes teardown restore it
    return calls


def _serve(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    from kenaz_ml import cli

    monkeypatch.setattr(sys, "argv", ["kenaz-ml", "serve", *args])
    cli.main()


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "", "10.0.0.5", "192.168.1.2", "example.com", "2001:db8::1"])
def test_non_loopback_host_is_refused(
    serve_calls: list[dict], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, host: str
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _serve(monkeypatch, "--host", host)
    assert excinfo.value.code != 0
    assert serve_calls == []
    assert "--dev-allow-remote" in capsys.readouterr().err


def test_dev_allow_remote_is_accepted_with_a_warning(
    serve_calls: list[dict], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        _serve(monkeypatch, "--host", "0.0.0.0", "--dev-allow-remote")
    assert serve_calls and serve_calls[0]["host"] == "0.0.0.0"
    assert any("dev-allow-remote" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "127.0.0.2"])
def test_loopback_hosts_are_accepted(serve_calls: list[dict], monkeypatch: pytest.MonkeyPatch, host: str) -> None:
    _serve(monkeypatch, "--host", host)
    assert serve_calls[0]["host"] == host


@pytest.mark.parametrize("port", ["7774", "7785", "7786", "7794", "7825"])
def test_port_flag_is_respected(serve_calls: list[dict], monkeypatch: pytest.MonkeyPatch, port: str) -> None:
    _serve(monkeypatch, "--port", port)
    assert serve_calls[0]["port"] == int(port)
    assert serve_calls[0]["host"] == "127.0.0.1"


@pytest.mark.parametrize("port", ["7774", "7785", "7786", "7794", "7825"])
def test_loopback_guard_holds_on_every_port(
    serve_calls: list[dict], monkeypatch: pytest.MonkeyPatch, port: str
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _serve(monkeypatch, "--host", "0.0.0.0", "--port", port)
    assert excinfo.value.code != 0
    assert serve_calls == []


def test_default_serve_is_unchanged(serve_calls: list[dict], monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch)
    assert serve_calls[0]["host"] == "127.0.0.1"
    assert serve_calls[0]["port"] == 7774


def test_module_constants_are_the_shipped_defaults() -> None:
    assert leases_mod.IDLE_EXIT_SEC == 120
    assert leases_mod.IMPLICIT_LEASE_SEC == 90
