"""CLI entry point for kameas-ml."""

from __future__ import annotations

import argparse
import json
import os
import sys

import uvicorn

from sigil_ml.config import resolve_mode
from sigil_ml.logging_config import setup_logging
from sigil_ml.storage.model_store import model_store_factory
from sigil_ml.store import create_store
from sigil_ml.store_sqlite import SqliteStore
from sigil_ml.training.trainer import Trainer


def main() -> None:
    """Entry point for the kameas-ml CLI."""
    parser = argparse.ArgumentParser(description="Sigil ML sidecar")
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="Start the ML server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=7774)
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
        mode = resolve_mode(args.mode)
        # Bridge mode to create_app() via env var (uvicorn string import cannot pass args)
        os.environ["SIGIL_ML_MODE"] = mode.value
        uvicorn.run(
            "sigil_ml.app:app",
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
    db_url = os.environ.get("SIGIL_POSTGRES_URL")
    s3_bucket = os.environ.get("SIGIL_S3_BUCKET")
    if not db_url or not s3_bucket:
        print(
            "Error: SIGIL_POSTGRES_URL and SIGIL_S3_BUCKET environment variables are required for cloud mode",
            file=sys.stderr,
        )
        sys.exit(1)

    # Construct stores from config (lazy imports)
    from sigil_ml.training.cloud_trainer import CloudTrainer

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


def _create_data_store(db_url: str) -> object:
    """Create a DataStore from the Postgres URL."""
    try:
        from sigil_ml import config
        from sigil_ml.store_postgres import PostgresStore

        tenant = config.tenant_id()
        return PostgresStore(connection_url=db_url, tenant=tenant)
    except ImportError:
        raise SystemExit("Error: PostgresStore not available. Install with: pip install kameas-ml[cloud]") from None


def _create_model_store(s3_bucket_name: str) -> object:
    """Create a ModelStore from the S3 bucket config."""
    try:
        from sigil_ml import config
        from sigil_ml.storage.model_store import S3ModelStore

        return S3ModelStore(
            bucket=s3_bucket_name,
            tenant_id=config.tenant_id(),
            endpoint_url=config.s3_endpoint_url(),
            region=config.aws_region(),
        )
    except ImportError:
        raise SystemExit("Error: S3ModelStore not available. Install with: pip install kameas-ml[cloud]") from None


def _build_cloud_training_config(
    min_interval: int | None = None,
    min_tasks: int | None = None,
    max_tasks_per_tenant: int | None = None,
) -> object:
    """Build a CloudTrainingConfig from env vars with CLI overrides."""
    from sigil_ml.training.models import CloudTrainingConfig

    return CloudTrainingConfig(
        min_interval_sec=min_interval
        if min_interval is not None
        else int(os.environ.get("SIGIL_ML_TRAIN_MIN_INTERVAL", "3600")),
        min_tasks=min_tasks if min_tasks is not None else int(os.environ.get("SIGIL_ML_TRAIN_MIN_TASKS", "10")),
        max_tasks_per_tenant=max_tasks_per_tenant
        if max_tasks_per_tenant is not None
        else int(os.environ.get("SIGIL_ML_TRAIN_MAX_TASKS_PER_TENANT", "1000")),
    )


if __name__ == "__main__":
    main()
