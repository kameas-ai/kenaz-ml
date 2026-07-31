"""Frozen-binary entrypoint for `kameas-ml`.

PyInstaller freezes a real script file rather than a console_scripts shim, so
this module exists purely to invoke the unchanged CLI ``main()``. The runtime
behaviour is identical to ``kameas-ml`` installed via pip — same subcommands
(``serve --port``, ``train``, ``health-check``), same routes, same WAL
contract (kenaz-ml/CLAUDE.md). See ADR-ml-packaging.md.

One subcommand exists **only here**, in the frozen artifact:
``feature-store-selfcheck``. It is packaging verification, not product
behaviour, which is why it lives in the freeze entrypoint rather than in
``sigil_ml.cli`` — a pip install has no bundle to verify and no read-only
application directory to assert against.

Why it has to exist (WP05 T020/T021, research.md D-005): Feast resolves its
provider, registry and online-store implementations from strings, through
``importlib.import_module``. PyInstaller's static analysis cannot follow that,
so a bundle missing those modules builds cleanly, starts cleanly, answers
``/health``, and then fails on the first actual feature call — on a user's
machine. A successful build is therefore not evidence. This subcommand performs
a real feature resolution against the shipped read-only registry and reports
what it found as JSON, so ``tests/test_frozen_smoke.py`` can assert on it.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import traceback

from sigil_ml.cli import main

#: Argv token that selects the packaging self-check instead of the product CLI.
SELFCHECK_COMMAND = "feature-store-selfcheck"

#: Entity id used for the probe lookup. It is not expected to resolve to
#: anything — the point is that the lookup *executes*, which requires the local
#: provider, the file registry and the SQLite online store all to have survived
#: freezing. A miss returns nulls; a missing dynamic import raises.
_PROBE_ENTITY_ID = "kameas-ml-freeze-selfcheck"


def _feature_store_selfcheck(argv: list[str]) -> int:
    """Exercise the full local feature flow against the bundled registry.

    Prints a single JSON object to stdout and returns a process exit code.

    The flow deliberately mirrors what a real local feature operation does, in
    order:

    1. Render the shipped local configuration, which reads
       ``feature_store.local.yaml`` out of the read-only bundle and runs the
       network-surface lint over it (FR-005) — so the lint is exercised in the
       frozen artifact, not only under test.
    2. Read the provenance marker written by the build-time ``feast apply`` and
       compare its Feast version with the running one (D-007, FR-016).
    3. Open the registry. This deserializes the shipped protobuf and resolves
       ``registry_type: file`` by name.
    4. Create online-store infrastructure. On a first run the SQLite tables do
       not exist yet, and they must be created in the **writable user data
       directory** — never beside the binary (FR-013). Note this is *not* an
       apply: it touches the online store only and never writes the registry.
    5. Resolve features through a feature service. This is the real feature
       call: ``provider: local`` and ``online_store: sqlite`` are both imported
       by string here, and this is where a bundle with missing hidden imports
       dies.

    Args:
        argv: Arguments after the subcommand token.

    Returns:
        ``0`` on success, ``1`` on any failure.
    """
    parser = argparse.ArgumentParser(
        prog=f"kameas-ml {SELFCHECK_COMMAND}",
        description=(
            "Verify that the bundled Feast registry and the dynamically-imported "
            "local provider and SQLite online store survived freezing."
        ),
    )
    parser.add_argument(
        "--user-data",
        default=None,
        help=(
            "Writable directory to use for the online store instead of the sigild "
            "data directory. Used by the freeze smoke test so the check does not "
            "touch the developer's real store."
        ),
    )
    args = parser.parse_args(argv)

    report: dict[str, object] = {"ok": False}
    try:
        import feast
        from feast import FeatureStore

        from sigil_ml.feature_store import config as fs_config

        bundle = fs_config.bundle_dir()
        registry = fs_config.registry_path()
        report["frozen"] = hasattr(sys, "_MEIPASS")
        report["bundle_dir"] = str(bundle)
        report["registry_path"] = str(registry)
        report["registry_exists"] = registry.is_file()
        if not registry.is_file():
            raise RuntimeError(
                f"No registry at {registry}. The registry is applied at build time and shipped "
                "inside the signed bundle (D-001); its absence means the PyInstaller spec did not "
                "collect it to the path bundle_dir() resolves to."
            )
        report["registry_bytes"] = registry.stat().st_size

        # --- 2. provenance (D-007 / FR-016) --------------------------------
        report["runtime_feast_version"] = feast.__version__
        marker = bundle / "registry.version.json"
        report["registry_version_marker"] = str(marker)
        if marker.is_file():
            recorded = json.loads(marker.read_text(encoding="utf-8"))
            report["registry_feast_version"] = recorded.get("feast_version")
            report["registry_applied_objects"] = recorded.get("applied_objects")
            report["feast_version_match"] = recorded.get("feast_version") == feast.__version__
        else:
            report["registry_feast_version"] = None
            report["feast_version_match"] = False

        # --- 1./3. configuration and registry ------------------------------
        repo_config = fs_config.load_local_repo_config(user_data=args.user_data)
        report["provider"] = repo_config.provider
        # Read back off the parsed configuration rather than recomputing it, so
        # the reported location is the one Feast will actually use.
        report["online_store_path"] = str(repo_config.online_store.path)
        report["registry_path_in_use"] = str(repo_config.registry.path)

        store = FeatureStore(config=repo_config)
        feature_views = store.list_feature_views()
        entities = store.list_entities()
        report["feature_views"] = sorted(view.name for view in feature_views)
        report["entities"] = sorted(entity.name for entity in entities)
        report["feature_services"] = sorted(service.name for service in store.list_feature_services())
        # Reported so the smoke test can confirm the registry actually carries
        # the push sources the build recorded binding — the local serving path
        # pushes into them by name (D-003), and a registry without them serves
        # reads while failing every push.
        report["data_sources"] = sorted(source.name for source in store.list_data_sources())

        # --- 4. online-store infrastructure in the writable directory ------
        # `update_infra` is the online-store half of what `apply` does. Calling
        # `apply` here would rewrite the registry, which lives in the read-only
        # bundle — it would fail, and it would be wrong even if it succeeded,
        # because the shipped registry is part of the signed artifact.
        provider = store._get_provider()
        provider.update_infra(
            project=repo_config.project,
            tables_to_delete=[],
            tables_to_keep=feature_views,
            entities_to_delete=[],
            entities_to_keep=entities,
            partial=True,
        )

        # --- 5. the real feature call --------------------------------------
        resolved: dict[str, list[object]] = {}
        for service_name in report["feature_services"]:  # type: ignore[union-attr]
            service = store.get_feature_service(service_name)
            resolved[service_name] = sorted(
                store.get_online_features(
                    features=service,
                    entity_rows=[{"task_id": _PROBE_ENTITY_ID}],
                ).to_dict()
            )
        report["resolved_features"] = resolved
        report["ok"] = True
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 — the report is the interface
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    # PyInstaller one-file builds re-exec the bootloader for child processes;
    # freeze_support() makes any multiprocessing-based worker (joblib/loky,
    # uvicorn reload) behave correctly in the frozen binary.
    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == SELFCHECK_COMMAND:
        raise SystemExit(_feature_store_selfcheck(sys.argv[2:]))
    main()
