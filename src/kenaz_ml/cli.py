"""CLI entry point for kenaz-ml."""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
import sys
from typing import TYPE_CHECKING

import uvicorn

from kenaz_ml.config import env, resolve_mode
from kenaz_ml.datastore import create_store
from kenaz_ml.datastore.sqlite import SqliteStore
from kenaz_ml.logging_config import setup_logging
from kenaz_ml.modelstore import model_store_factory
from kenaz_ml.training.trainer import Trainer

if TYPE_CHECKING:
    from kenaz_ml.datastore import DataStore
    from kenaz_ml.modelstore import ModelStore
    from kenaz_ml.training.models import CloudTrainingConfig


def is_loopback_host(host: str | None) -> bool:
    """True only for a loopback bind address (FR-018).

    ``localhost`` is accepted by name; anything else must be an IP literal whose
    address is loopback. ``0.0.0.0``, ``::``, an empty string and every other
    literal or hostname are refused. Nothing is resolved over the network.
    """
    if not host:
        return False
    candidate = host.strip()
    if candidate.lower() == "localhost":
        return True
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def main() -> None:
    """Entry point for the kenaz-ml CLI."""
    from kenaz_ml import __version__

    parser = argparse.ArgumentParser(description="kenaz-ml — the ML sidecar for Sigil")
    # Prints the bare version (single source: pyproject.toml; Amendment A5) so a
    # spawning client can label its version directories from it.
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="Start the ML server")
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. Must be loopback (127.0.0.1, ::1, localhost) unless --dev-allow-remote is given.",
    )
    serve_parser.add_argument(
        "--dev-allow-remote",
        action="store_true",
        default=False,
        help="DEVELOPMENT ONLY: allow binding a non-loopback address. Exposes the engine beyond this "
        "machine; never set by a spawning client.",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=7774,
        help="Loopback port (default 7774, the daemon's contract — see CLAUDE.md invariant 3). A spawning client passes the "
        "port it chose from its env's lane (design A5.3: base prod 7774, dev 7785, test 7786, falling back to "
        "base+10k when a port is taken) and records it in the install root's engine.port; the engine binds "
        "whatever it is given. The loopback bind guard applies on every port.",
    )
    serve_parser.add_argument(
        "--mode",
        choices=["local", "cloud"],
        default=None,
        help="Serving mode: 'local' (default, with poller) or 'cloud' (stateless, no SQLite)",
    )

    train_parser = sub.add_parser("train", help="Train models from local data")
    train_parser.add_argument("--db", help="Path to sigild SQLite database")
    train_parser.add_argument(
        "--mode",
        choices=["local", "cloud"],
        default="local",
        help="Training mode: local (SQLite) or cloud (Postgres/S3)",
    )
    train_parser.add_argument(
        "--tenant",
        type=str,
        default=None,
        help="Train models for a specific tenant ID (cloud mode only)",
    )
    train_parser.add_argument(
        "--all-tenants",
        action="store_true",
        default=False,
        help="Discover and train all eligible tenants (cloud mode only)",
    )
    train_parser.add_argument(
        "--aggregate",
        action="store_true",
        default=False,
        help="Train aggregate model from pooled opted-in data (cloud mode only)",
    )
    train_parser.add_argument(
        "--min-interval",
        type=int,
        default=None,
        help="Minimum seconds between retraining a tenant (default: 3600)",
    )
    train_parser.add_argument(
        "--min-tasks",
        type=int,
        default=None,
        help="Minimum completed tasks for ML training (default: 10)",
    )
    train_parser.add_argument(
        "--max-tasks-per-tenant",
        type=int,
        default=None,
        help="Cap per-tenant tasks for aggregate training (default: 1000)",
    )
    train_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Force compact JSON output (default for non-TTY)",
    )

    sub.add_parser("health-check", help="Check if server is running")

    args = parser.parse_args()

    # Initialize file + console logging for all commands
    setup_logging()

    if args.command == "serve":
        if not is_loopback_host(args.host):
            if not args.dev_allow_remote:
                print(
                    f"kenaz-ml: refusing to bind non-loopback host {args.host!r}. The engine binds loopback only "
                    "(127.0.0.1, ::1, localhost); pass --dev-allow-remote to override for development.",
                    file=sys.stderr,
                )
                sys.exit(2)
            logging.getLogger("kenaz_ml").warning(
                "kenaz-ml: --dev-allow-remote set; binding non-loopback host %r (development only)", args.host
            )
        mode = resolve_mode(args.mode)
        # Bridge mode to create_app() via env var (uvicorn string import cannot pass args)
        os.environ["KENAZ_ML_MODE"] = mode.value
        uvicorn.run(
            "kenaz_ml.app:app",
            host=args.host,
            port=args.port,
            log_level="info",
        )
    elif args.command == "train":
        if args.mode == "cloud":
            _handle_cloud_training(args)
        else:
            # Existing local training path -- COMPLETELY UNCHANGED
            if args.db:
                from pathlib import Path

                store = SqliteStore(Path(args.db))
            else:
                store = create_store()
            ms = model_store_factory()
            print(f"Training models using {type(store).__name__} + {type(ms).__name__} ...")
            trainer = Trainer(store, model_store=ms)
            result = trainer.train_all()
            print(f"Done: {result}")
    elif args.command == "health-check":
        try:
            import httpx
        except ImportError:
            print("httpx is required for health-check: pip install httpx", file=sys.stderr)
            sys.exit(1)

        try:
            resp = httpx.get("http://127.0.0.1:7774/health", timeout=5)
            data = resp.json()
            print(f"Status: {data['status']}")
            for model, state in data.get("models", {}).items():
                print(f"  {model}: {state}")
        except Exception as e:
            print(f"Server not reachable: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


def _handle_cloud_training(args: argparse.Namespace) -> None:
    """Handle all cloud training modes. Lazy-imports cloud modules."""
    # Validate cloud flags: at least one target required
    cloud_actions = [args.tenant, args.all_tenants, args.aggregate]
    if not any(cloud_actions):
        print(
            "Error: Cloud mode requires --tenant, --all-tenants, or --aggregate",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate mutual exclusivity
    if sum(bool(a) for a in cloud_actions) > 1:
        print(
            "Error: --tenant, --all-tenants, and --aggregate are mutually exclusive",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate cloud-only flags not used with local mode
    # (already handled by routing -- only called when mode == "cloud")

    # Validate required environment variables
    db_url = env("KENAZ_POSTGRES_URL")
    s3_bucket = env("KENAZ_S3_BUCKET")
    if not db_url or not s3_bucket:
        print(
            "Error: KENAZ_POSTGRES_URL and KENAZ_S3_BUCKET environment variables are required for cloud mode",
            file=sys.stderr,
        )
        sys.exit(1)

    # Construct stores from config (lazy imports)
    from kenaz_ml.training.cloud_trainer import CloudTrainer

    data_store = _create_data_store(db_url)
    model_store = _create_model_store(s3_bucket)

    cfg = _build_cloud_training_config(
        min_interval=args.min_interval,
        min_tasks=args.min_tasks,
        max_tasks_per_tenant=args.max_tasks_per_tenant,
    )

    trainer = CloudTrainer(data_store, model_store, cfg)

    use_compact_json = not sys.stdout.isatty() or args.json

    if args.tenant:
        result = trainer.train_tenant(args.tenant)
        if use_compact_json:
            print(json.dumps(result.to_dict()))
        else:
            print(json.dumps(result.to_dict(), indent=2))
        sys.exit(0 if result.status != "failed" else 1)

    elif args.all_tenants:
        batch = trainer.train_all_tenants()
        if use_compact_json:
            print(json.dumps(batch.to_dict()))
        else:
            print("\n=== Batch Training Summary ===")
            print(f"Total tenants: {batch.total}")
            print(f"  Trained: {batch.trained}")
            print(f"  Skipped: {batch.skipped}")
            print(f"  Failed:  {batch.failed}")
            print(f"Duration: {batch.total_duration_ms}ms")
            if batch.failed > 0:
                print("\nFailed tenants:")
                for run in batch.runs:
                    if run.status == "failed":
                        print(f"  - {run.tenant_id}: {run.error}")
            print("\nFull JSON:")
            print(json.dumps(batch.to_dict(), indent=2))
        sys.exit(0 if batch.failed == 0 else 1)

    elif args.aggregate:
        result = trainer.train_aggregate()
        if use_compact_json:
            print(json.dumps(result.to_dict()))
        else:
            print("\n=== Aggregate Training Summary ===")
            print(f"Status: {result.status}")
            print(f"Samples: {result.sample_count}")
            print(f"Models trained: {', '.join(result.models_trained) or 'none'}")
            print(f"Duration: {result.duration_ms}ms")
            if result.error:
                print(f"Note: {result.error}")
            print("\nFull JSON:")
            print(json.dumps(result.to_dict(), indent=2))
        sys.exit(0 if result.status != "failed" else 1)


def _create_data_store(db_url: str) -> DataStore:
    """Create a DataStore from the Postgres URL."""
    try:
        from kenaz_ml import config
        from kenaz_ml.datastore.postgres import PostgresStore

        tenant = config.tenant_id()
        return PostgresStore(connection_url=db_url, tenant=tenant)
    except ImportError:
        raise SystemExit("Error: PostgresStore not available. Install with: pip install kenaz-ml[cloud]") from None


def _create_model_store(s3_bucket_name: str) -> ModelStore:
    """Create a ModelStore from the S3 bucket config."""
    try:
        from kenaz_ml import config
        from kenaz_ml.modelstore import S3ModelStore

        return S3ModelStore(
            bucket=s3_bucket_name,
            tenant_id=config.tenant_id(),
            endpoint_url=config.s3_endpoint_url(),
            region=config.aws_region(),
        )
    except ImportError:
        raise SystemExit("Error: S3ModelStore not available. Install with: pip install kenaz-ml[cloud]") from None


def _build_cloud_training_config(
    min_interval: int | None = None,
    min_tasks: int | None = None,
    max_tasks_per_tenant: int | None = None,
) -> CloudTrainingConfig:
    """Build a CloudTrainingConfig from env vars with CLI overrides."""
    from kenaz_ml.training.models import CloudTrainingConfig

    return CloudTrainingConfig(
        min_interval_sec=min_interval if min_interval is not None else int(env("KENAZ_ML_TRAIN_MIN_INTERVAL", "3600")),
        min_tasks=min_tasks if min_tasks is not None else int(env("KENAZ_ML_TRAIN_MIN_TASKS", "10")),
        max_tasks_per_tenant=max_tasks_per_tenant
        if max_tasks_per_tenant is not None
        else int(env("KENAZ_ML_TRAIN_MAX_TASKS_PER_TENANT", "1000")),
    )


if __name__ == "__main__":
    main()
