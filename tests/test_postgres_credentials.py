"""Credentials a host can rotate under a Postgres store.

A cloud host hands the store a provider instead of a fixed URL. The store asks
it for credentials on every new connection and, when the server refuses them,
asks once more with ``refresh=True`` and retries a single time.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from kenaz_ml.datastore.postgres import PostgresStore
from tests.test_postgres_stream_scope import FakeConn, fake_psycopg2  # noqa: F401  (fixture)


class OperationalError(Exception):
    def __init__(self, msg: str, pgcode: str | None = None) -> None:
        super().__init__(msg)
        self.pgcode = pgcode


@pytest.fixture
def connects(fake_psycopg2: dict[str, Any]) -> list[str]:  # noqa: F811
    """Make the fake refuse every conninfo containing 'old'."""
    mod = sys.modules["psycopg2"]
    mod.OperationalError = OperationalError  # type: ignore[attr-defined]
    seen: list[str] = []

    def connect(dsn: str) -> FakeConn:
        seen.append(dsn)
        if "old" in dsn:
            raise OperationalError('FATAL:  password authentication failed for user "x"')
        return FakeConn(fake_psycopg2["log"], deny_ddl=False, tables_present=(), fail_on=())

    mod.connect = connect  # type: ignore[attr-defined]
    return seen


def test_a_refused_login_is_retried_once_with_refreshed_credentials(connects: list[str]) -> None:
    asked: list[bool] = []

    def provider(refresh: bool) -> str:
        asked.append(refresh)
        return "password=new" if refresh else "password=old"

    PostgresStore(provider, tenant="tenant_a", stream="u1").get_cursor()
    assert asked == [False, True]
    assert connects == ["password=old", "password=new"]


def test_a_second_refusal_is_raised_not_looped(connects: list[str]) -> None:
    calls: list[bool] = []

    def provider(refresh: bool) -> str:
        calls.append(refresh)
        return "password=old"

    with pytest.raises(OperationalError):
        PostgresStore(provider, tenant="tenant_a").get_cursor()
    assert calls == [False, True]


def test_a_non_auth_failure_is_not_retried(fake_psycopg2: dict[str, Any]) -> None:  # noqa: F811
    mod = sys.modules["psycopg2"]
    mod.OperationalError = OperationalError  # type: ignore[attr-defined]

    def connect(dsn: str) -> FakeConn:
        raise OperationalError("could not connect to server: Connection timed out")

    mod.connect = connect  # type: ignore[attr-defined]
    calls: list[bool] = []
    with pytest.raises(OperationalError, match="timed out"):
        PostgresStore(lambda r: calls.append(r) or "x", tenant="tenant_a").get_cursor()
    assert calls == [False]


def test_an_auth_failure_with_a_sqlstate_is_retried(connects: list[str]) -> None:
    mod = sys.modules["psycopg2"]
    first = True

    def connect(dsn: str) -> FakeConn:
        nonlocal first
        if first:
            first = False
            raise OperationalError("refused", pgcode="28P01")
        return FakeConn([], deny_ddl=False, tables_present=(), fail_on=())

    mod.connect = connect  # type: ignore[attr-defined]
    asked: list[bool] = []
    PostgresStore(lambda r: asked.append(r) or "x", tenant="tenant_a").get_cursor()
    assert asked == [False, True]


def test_a_plain_url_still_works(fake_psycopg2: dict[str, Any]) -> None:  # noqa: F811
    PostgresStore("postgresql://x", tenant="tenant_a").get_cursor()
    assert fake_psycopg2["conn"] is not None
