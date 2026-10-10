"""The cloud worker: the engine's local loop, run per device stream.

The local engine is one person's loop: it tails one ``events`` ledger from
one cursor, keeps one active task, and writes that person's predictions.
In the cloud the same loop runs once per *stream* -- an ``(org, user)`` pair
-- against a :class:`~kenaz_ml.datastore.postgres.PostgresStore` scoped to
that stream, with the org's models from the shared model store. Nothing
about features, models, the poller or the training scheduler is different;
this module only hosts them outside the HTTP app and multiplies them.

The serving app stays stateless (``ServingMode.CLOUD``); a deployment runs
this worker as its own process in the same image::

    kenaz-ml worker --tenants org_a,org_b

discovers the streams in each tenant schema and runs one
:class:`StreamWorker` per stream until SIGTERM.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from kenaz_ml.app import AppState, build_signal_engine
from kenaz_ml.config import ServingMode
from kenaz_ml.datastore import DataStore
from kenaz_ml.modelstore import ModelStore
from kenaz_ml.poller import POLL_INTERVAL_SEC, EventPoller
from kenaz_ml.training.scheduler import TrainingScheduler

logger = logging.getLogger(__name__)


def stream_retained_dir(tenant: str, user: str) -> Path:
    """Return the retained-set directory of one ``(tenant, user)`` stream.

    One directory per stream, so no file ever holds two organizations' (or two
    members') examples, and erasing a member is removing one directory.
    """
    from kenaz_ml import config
    from kenaz_ml.datastore.postgres import validate_stream_id

    unsafe = user in (".", "..") or "/" in user or "\\" in user
    if not config.validate_tenant_id(tenant) or not validate_stream_id(user) or unsafe:
        raise ValueError(f"refusing a retained-set directory for stream {tenant!r}/{user!r}")
    return config.retained_data_dir() / tenant / user


#: How often a stream's worker asks the training scheduler whether a retrain
#: is due; the scheduler applies its own minimum interval on top.
DEFAULT_RETRAIN_CHECK_SEC = 600.0


@dataclass
class StreamWorker:
    """One stream's loop: poller plus training scheduler over one store.

    ``store`` must already be scoped to the stream (for Postgres,
    ``PostgresStore(url, tenant=org, stream=user)``); the worker never adds
    scoping of its own. ``model_store`` is the org's, shared by its streams.
    ``retained_dir`` is this stream's own retained-set directory; with none,
    training over a non-filesystem model store retains nothing.
    """

    name: str
    store: DataStore
    model_store: ModelStore
    retained_dir: Path | None = None
    poll_interval_sec: float = POLL_INTERVAL_SEC
    retrain_check_sec: float = DEFAULT_RETRAIN_CHECK_SEC
    state: AppState = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # The container the local app uses, so load/reload paths are shared.
        self.state = AppState(ServingMode.LOCAL)
        self.state.store = self.store
        self.state.model_store = self.model_store

    def setup(self) -> None:
        """Blocking setup: tables, registry refresh, models, pipeline, poller.

        Runs off the event loop. Raises on a misconfigured store: a worker
        that cannot reach its schema must fail loudly, not poll quietly.
        """
        self.store.ensure_tables()
        self.state.refresh_registry(self.model_store)
        self.state.load_models(self.model_store)
        self.state.signal_engine = build_signal_engine(self.store, self.model_store)
        self.state.poller = EventPoller(
            store=self.store,
            models={
                "stuck": self.state.stuck,
                "activity": self.state.activity,
                "workflow": self.state.workflow,
                "duration": self.state.duration,
                "quality": self.state.quality,
            },
            signal_engine=self.state.signal_engine,
            poll_interval_sec=self.poll_interval_sec,
        )
        self._scheduler = TrainingScheduler(
            self.store,
            model_store=self.model_store,
            reload_callback=self.state.reload_models_into_poller,
            retained_dir=self.retained_dir,
        )
        logger.info("worker[%s]: models loaded, poller ready", self.name)

    async def _retrain_loop(self, stop: asyncio.Event) -> None:
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            try:
                await loop.run_in_executor(None, self._scheduler.check_and_retrain)
            except Exception:
                logger.warning("worker[%s]: retrain check failed (will retry)", self.name, exc_info=True)
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.retrain_check_sec)
            except TimeoutError:
                continue

    async def run(self, stop: asyncio.Event) -> None:
        """Run until ``stop`` is set, then stop the poller and close the store."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.setup)
        poller = self.state.poller
        assert poller is not None
        tasks = [
            asyncio.create_task(poller.run(), name=f"poller:{self.name}"),
            asyncio.create_task(self._retrain_loop(stop), name=f"retrain:{self.name}"),
        ]
        try:
            await stop.wait()
        finally:
            poller.stop()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.store.close()
            logger.info("worker[%s]: stopped", self.name)


async def run_workers(workers: Iterable[StreamWorker], stop: asyncio.Event) -> None:
    """Run every worker concurrently until ``stop`` is set.

    One stream's failure is logged and does not stop the others.
    """
    results = await asyncio.gather(*(w.run(stop) for w in workers), return_exceptions=True)
    for worker, result in zip(workers, results):
        if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
            logger.error("worker[%s]: exited with %r", worker.name, result)


def install_stop_signals(stop: asyncio.Event) -> None:
    """SIGTERM and SIGINT set ``stop``; a no-op where the loop lacks signal support."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError, RuntimeError:  # pragma: no cover - Windows
            return


def build_postgres_workers(
    connection_url: str,
    bucket: str,
    *,
    tenants: Iterable[str] = (),
    streams: Iterable[tuple[str, str]] = (),
    poll_interval_sec: float = POLL_INTERVAL_SEC,
    retrain_check_sec: float = DEFAULT_RETRAIN_CHECK_SEC,
) -> list[StreamWorker]:
    """One worker per stream, from explicit ``(tenant, user)`` pairs and/or
    every stream discovered in each of ``tenants``.

    Requires the ``cloud`` extra. Discovery is at build time; a stream that
    appears later is picked up on the next worker start.
    """
    from kenaz_ml import config
    from kenaz_ml.datastore.postgres import PostgresStore
    from kenaz_ml.modelstore import S3ModelStore

    wanted: list[tuple[str, str]] = list(streams)
    for tenant in tenants:
        discovery = PostgresStore(connection_url, tenant=tenant)
        try:
            found = discovery.list_streams()
        finally:
            discovery.close()
        if not found:
            logger.warning("worker: tenant %s has no streams (no events with a user_id yet)", tenant)
        wanted.extend((tenant, user) for user in found)

    seen: set[tuple[str, str]] = set()
    workers: list[StreamWorker] = []
    model_stores: dict[str, ModelStore] = {}
    for tenant, user in wanted:
        if (tenant, user) in seen:
            continue
        seen.add((tenant, user))
        if tenant not in model_stores:
            model_stores[tenant] = S3ModelStore(
                bucket=bucket,
                tenant_id=tenant,
                endpoint_url=config.s3_endpoint_url(),
                region=config.aws_region(),
            )
        workers.append(
            StreamWorker(
                name=f"{tenant}/{user}",
                store=PostgresStore(connection_url, tenant=tenant, stream=user),
                model_store=model_stores[tenant],
                retained_dir=stream_retained_dir(tenant, user),
                poll_interval_sec=poll_interval_sec,
                retrain_check_sec=retrain_check_sec,
            )
        )
    return workers
