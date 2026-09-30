"""Token-authorized graceful shutdown (design doc §3.5, repair R6; FR-015).

The token lives in the client-owned ``lease/`` directory
(:func:`kenaz_ml.config.shutdown_token_path`), written user-read-only by the
spawning client. kenaz-ml **reads it on every request** (a newer client may
rotate it) and never creates it.

Fail closed on authorization: a missing ``lease/`` directory, a missing, empty
or unreadable token file, no presented token, or a wrong one all refuse. A
missing token is never read as "no token required". Neither token value is
ever logged or echoed.

Presented as ``Authorization: Bearer <token>`` (:data:`TOKEN_SCHEME`).

Exiting
-------
With ``uvicorn.run`` called on a string import there is no server handle, so
the exit path is the standard one: ``SIGTERM`` to this process, which uvicorn
turns into ``should_exit`` and the app's lifespan shutdown (poller stop, store
close) runs. Before that, :func:`drain_and_exit` waits — bounded — for any
in-flight training run so no artifact is left half-written.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import signal
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TOKEN_HEADER = "Authorization"
TOKEN_SCHEME = "Bearer"

REASON_NO_TOKEN = "no_token_presented"
REASON_LEASE_DIR_MISSING = "lease_dir_missing"
REASON_TOKEN_FILE_MISSING = "token_file_missing"
REASON_TOKEN_FILE_UNREADABLE = "token_file_unreadable"
REASON_TOKEN_FILE_EMPTY = "token_file_empty"
REASON_TOKEN_MISMATCH = "token_mismatch"


@dataclass(frozen=True)
class TokenCheck:
    ok: bool
    reason: str | None = None


def presented_token(header_value: str | None) -> str | None:
    """Extract the bearer token from an ``Authorization`` header value."""
    if not header_value:
        return None
    scheme, _, value = header_value.partition(" ")
    if scheme.lower() != TOKEN_SCHEME.lower() or not value:
        return None
    return value.strip() or None


def check_token(presented: str | None, token_path: Path) -> TokenCheck:
    """Compare ``presented`` with the token file. Fails closed on every doubt."""
    if not token_path.parent.is_dir():
        return TokenCheck(False, REASON_LEASE_DIR_MISSING)
    if not token_path.exists():
        return TokenCheck(False, REASON_TOKEN_FILE_MISSING)
    try:
        raw = token_path.read_bytes()
        mode = token_path.stat().st_mode
    except OSError:
        return TokenCheck(False, REASON_TOKEN_FILE_UNREADABLE)
    try:
        expected = raw.decode("utf-8").rstrip()
    except UnicodeDecodeError:
        return TokenCheck(False, REASON_TOKEN_FILE_UNREADABLE)
    if not expected:
        return TokenCheck(False, REASON_TOKEN_FILE_EMPTY)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        # The client owns the file's permissions; warn, do not refuse.
        logger.warning("lifecycle: shutdown token file %s is group/world accessible", token_path)
    if presented is None:
        return TokenCheck(False, REASON_NO_TOKEN)
    if not secrets.compare_digest(presented.encode("utf-8"), expected.encode("utf-8")):
        return TokenCheck(False, REASON_TOKEN_MISMATCH)
    return TokenCheck(True)


def request_process_exit() -> None:
    """Ask this process to exit through uvicorn's graceful SIGTERM path."""
    os.kill(os.getpid(), signal.SIGTERM)


async def drain_and_exit(
    state: Any,
    *,
    exit_fn: Callable[[], None] | None = None,
    drain_sec: float | None = None,
    poll_sec: float = 0.5,
    reason: str = "shutdown",
) -> None:
    """Wait (bounded) for in-flight training, stop the poller, then exit.

    The poller commits its cursor per batch; stopping it lets the current batch
    finish. The lifespan shutdown block also stops it and closes the store.
    """
    from kenaz_ml.lifecycle.leases import TRAINING_DRAIN_SEC

    budget = TRAINING_DRAIN_SEC if drain_sec is None else drain_sec
    waited = 0.0
    while getattr(state, "training_in_progress", False) and waited < budget:
        await asyncio.sleep(poll_sec)
        waited += poll_sec
    if getattr(state, "training_in_progress", False):
        logger.warning("lifecycle: exiting (%s) with training still running after %.0fs", reason, waited)
    poller = getattr(state, "poller", None)
    if poller is not None:
        poller.stop()
    logger.info("lifecycle: exiting (%s)", reason)
    (exit_fn or getattr(state, "exit_fn", None) or request_process_exit)()
