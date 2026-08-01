# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the frozen, self-contained `kameas-ml` executable.
#
# FR-3 (ADR-ml-packaging.md, Option 2): kenaz supervises a frozen Python ML
# sidecar instead of dropping to a fake backend. This spec produces a single
# self-contained bundle that carries its own interpreter + scikit-learn +
# numpy + uvicorn + joblib + fastapi + Feast — no system Python, no
# `pip install` on the user's machine.
#
# Build (from the kenaz-ml repo root):
#     pip install -e ".[freeze]"
#     pyinstaller freeze/kameas-ml.spec --noconfirm
#     # → dist/kameas-ml/  (ONEDIR bundle: dist/kameas-ml/kameas-ml + _internal/)
#
# ONEDIR, not onefile (spec 069 LD-3 / FR-002): a onefile build self-extracts
# its (unsigned copies of) dylibs to a temp dir at runtime, which Apple
# notarization + hardened runtime reject — the extracted code is outside the
# signed/notarized envelope. The onedir layout keeps every Mach-O on disk in
# the app bundle where kenaz's inside-out signing pass
# (ADR-nested-binary-signing) can sign each one individually before
# notarization.
#
# Freeze arch matrix per LD-3: macOS arm64 only (sklearn/scipy wheels are
# single-arch; the darwin-amd64 leg is dropped for v1); Linux x86_64 + arm64.
#
# The runtime behaviour and the WAL data contract (kenaz-ml/CLAUDE.md) are
# UNCHANGED — freezing changes delivery, not behaviour.
#
# ---------------------------------------------------------------------------
# Feast (feast-feature-store-migration, WP05 / D-001, D-005, D-007)
# ---------------------------------------------------------------------------
# Two things about Feast make this spec more than a dependency list.
#
# 1. **The registry is applied here, at build time** (T019 / D-001). The
#    installed application directory is read-only and signed; `feast apply`
#    writes the registry, so it cannot run at first launch. Applying it here
#    also folds the definitions into the signed artifact, so they inherit
#    notarization's integrity guarantee. If the apply fails, this spec raises
#    and the build stops — a binary that shipped without a registry would fail
#    at the first feature call, on a user's machine.
#
# 2. **Feast resolves providers and stores by string at runtime** (T020 /
#    D-005). `feast.importer.import_class` does `importlib.import_module(name)`
#    on values looked up from `PROVIDERS_CLASS_FOR_TYPE` and
#    `ONLINE_STORE_CLASS_FOR_TYPE` in `feast/repo_config.py`. PyInstaller's
#    static analysis cannot see through that, so the failure shape is a binary
#    that builds cleanly, starts cleanly, and dies on the first actual feature
#    call. A green build proves nothing here; the guard is
#    `tests/test_frozen_smoke.py`, which runs the bundled
#    `feature-store-selfcheck` and asserts a real feature resolution against the
#    shipped registry.

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

REPO_ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 — SPECPATH is injected by PyInstaller
SIGIL_ML_SRC = REPO_ROOT / "src" / "sigil_ml"
FEATURE_STORE_SRC = SIGIL_ML_SRC / "feature_store"

# Where the collected feature-store assets must land inside the bundle. This is
# not a free choice: `sigil_ml.feature_store.config.bundle_dir()` resolves to
# `<sys._MEIPASS>/sigil_ml/feature_store` when frozen, and every path in the
# local configuration (`registry.db`) hangs off it. Changing this string without
# changing `bundle_dir()` produces a bundle whose registry cannot be found.
BUNDLE_FEATURE_STORE_DIR = os.path.join("sigil_ml", "feature_store")

# Filename of the build-time provenance marker written beside the registry
# (T019 step 5 / D-007). The registry is a serialized protobuf coupled to the
# Feast version that wrote it, so FR-016's "refuse a mismatched registry with a
# clear diagnostic" needs a recorded producing version to compare the running
# one against. Kept next to the registry so the two travel together and are
# covered by the same signature.
REGISTRY_VERSION_FILENAME = "registry.version.json"


# ===========================================================================
# T019 — build-time `feast apply`
# ===========================================================================


def _apply_registry(staging_dir):
    """Run `feast apply` against the shipped definitions; return the artifacts.

    The apply is done through Feast's Python API rather than the `feast` CLI
    because the CLI wants a feature-repo directory containing a
    `feature_store.yaml` and importable definition modules, which would mean
    duplicating the configuration this repo already renders from
    `sigil_ml.feature_store.config`. The API path uses exactly the shipped
    configuration and the shipped definitions, so what is applied here is what
    the frozen binary reads.

    Two directories are deliberately kept apart, mirroring the runtime split:

    * ``staging_dir`` stands in for the read-only bundle directory. The registry
      is written here and then collected into the bundle.
    * a throwaway temp directory stands in for the writable user data
      directory. `apply` creates online-store infrastructure, and it must not
      land in the artifact — the online store belongs in the user's data
      directory at runtime (FR-013), never beside the binary.

    Raises:
        RuntimeError: If the apply fails, with the original exception chained.
            The build stops here on purpose (T019 step 4).
    """
    # Imported inside the function so that a missing/broken Feast install
    # produces this function's diagnostic rather than a bare ImportError at
    # spec-parse time.
    import feast
    from feast import FeatureStore

    from sigil_ml.feature_store import config as fs_config
    from sigil_ml.feature_store import definitions as fs_definitions

    staging_dir = Path(staging_dir)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    # The views are taken source-bound from `resolve.local_feature_views()`
    # rather than bare from `definitions.FEATURE_VIEWS`. This matters and is not
    # a stylistic choice: the local serving path pushes computed vectors with
    # `store.push(<push source name>, ...)` (D-003), and a registry whose views
    # carry no `PushSource` has no such source to push into. Applying the
    # unbound views would produce a bundle that resolves features for reads and
    # fails every push — a half-working store, which is worse than an obviously
    # broken one because the prediction still returns.
    #
    # `definitions.py` declares shape only and deliberately binds no source, so
    # the binding has to come from the deployment. The fallback exists so this
    # build does not hard-depend on the local resolver being present; when it is
    # taken, the artifact records that its registry carries no push sources
    # rather than pretending otherwise.
    try:
        from sigil_ml.feature_store.resolve import local_feature_views

        feature_views = local_feature_views()
        push_sources = sorted(view.stream_source.name for view in feature_views if view.stream_source)
    except ImportError:
        print(
            "[kameas-ml.spec] WARNING: sigil_ml.feature_store.resolve.local_feature_views is not "
            "available; applying the unbound feature views. The registry will support reads but "
            "no push path."
        )
        feature_views = list(fs_definitions.FEATURE_VIEWS)
        push_sources = []

    objects = [
        *fs_definitions.ENTITIES,
        *feature_views,
        *fs_definitions.FEATURE_SERVICES.values(),
    ]

    with tempfile.TemporaryDirectory(prefix="kameas-ml-freeze-userdata-") as throwaway_user_data:
        try:
            repo_config = fs_config.load_local_repo_config(
                bundle=staging_dir,
                user_data=throwaway_user_data,
            )
            store = FeatureStore(config=repo_config)
            store.apply(objects)
        except Exception as exc:  # noqa: BLE001 — re-raised below with context
            raise RuntimeError(
                "feast apply failed during the frozen build, so no registry was produced.\n"
                "The build is stopped deliberately (WP05 T019): a bundle shipped without a "
                "registry builds and starts fine and then fails at the first feature call, on a "
                f"user's machine.\nStaging directory: {staging_dir}\nCause: {type(exc).__name__}: {exc}"
            ) from exc

    registry = fs_config.registry_path(bundle=staging_dir)
    if not registry.is_file():
        raise RuntimeError(
            f"feast apply reported success but wrote no registry at {registry}. "
            "Refusing to build a bundle without one (WP05 T019)."
        )

    marker = staging_dir / REGISTRY_VERSION_FILENAME
    marker.write_text(
        json.dumps(
            {
                # D-007: the version that produced the protobuf on disk. Compared
                # against the running `feast.__version__` by the frozen
                # `feature-store-selfcheck`, and the input FR-016's mismatch
                # diagnostic needs.
                "feast_version": feast.__version__,
                "project": repo_config.project,
                "registry_filename": registry.name,
                "entity_key_serialization_version": repo_config.entity_key_serialization_version,
                "applied_objects": sorted(obj.name for obj in objects),
                # The push sources the shipped registry carries. The frozen
                # smoke test cross-checks these against what the registry
                # actually contains, so a build that silently applied the
                # unbound views is visible in the artifact.
                "local_push_sources": push_sources,
                "applied_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[kameas-ml.spec] feast apply OK — registry {registry} "
        f"({registry.stat().st_size} bytes) produced by feast {feast.__version__}"
    )
    return registry, marker


# An editable install already puts `sigil_ml` on the path; a plain source
# checkout does not. Adding `src` makes the build step work either way.
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

_registry_path, _registry_marker_path = _apply_registry(REPO_ROOT / "build" / "feature-store")


# ===========================================================================
# T020 — hidden imports
# ===========================================================================
# Freezing scikit-learn / numpy / uvicorn has well-known hidden-import sharp
# edges: PyInstaller's static analysis misses modules pulled in dynamically
# (sklearn's Cython-compiled estimators, uvicorn's loop/protocol plugins,
# joblib's loky backend). Pin them explicitly. The freeze smoke test
# (tests/test_frozen_smoke.py) is the guard that these are sufficient: it
# launches the frozen binary and POSTs /predict/stuck, which exercises the
# sklearn estimator path end-to-end.
hiddenimports = []
hiddenimports += collect_submodules("sklearn")
hiddenimports += collect_submodules("sklearn.utils")
hiddenimports += collect_submodules("sklearn.tree")
hiddenimports += collect_submodules("sklearn.ensemble")
hiddenimports += collect_submodules("sklearn.neighbors")
hiddenimports += collect_submodules("scipy")
hiddenimports += collect_submodules("numpy")
hiddenimports += collect_submodules("joblib")
hiddenimports += [
    # uvicorn dynamically imports its event-loop / HTTP / websocket plugins by
    # string name; PyInstaller cannot see these from the import graph.
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    # The app is loaded by uvicorn via the string "sigil_ml.app:app", so the
    # whole package must be reachable in the frozen graph.
    "sigil_ml.app",
]
hiddenimports += collect_submodules("sigil_ml")

# --- Feast: dynamically-resolved backends (D-005) ---------------------------
# Everything in this block exists for one reason: Feast names its
# implementations in strings and imports them with `importlib.import_module`.
# None of it is reachable from the static import graph.
#
# The backends the frozen LOCAL bundle must never need are filtered out rather
# than collected. Two reasons, in order of importance:
#
#   * C-001 — the open-source install has no network path to a feature store.
#     Not shipping `feast.infra.*.remote`, the SQL/Postgres registry, or the
#     cloud object-store registries means the local artifact does not merely
#     decline to dial out, it carries no code that could. The runtime lint in
#     `feature_store/config.py` and the socket-level test in
#     `tests/test_no_egress.py` are the other two independent layers.
#   * every collected module is signable surface, and Feast already costs
#     +309 MB / +338 native libraries (spec C-003).
#
# The filter only decides what is ADDED as a hidden import; anything statically
# reachable is still collected by Analysis. So an over-aggressive entry here
# cannot silently remove something Feast imports normally — but it CAN remove
# something Feast imports dynamically, which is precisely the failure this
# package exists to prevent. That is why the guard is a real feature call in
# the frozen binary (`feature-store-selfcheck`), not a successful build.
FEAST_EXCLUDED_BACKEND_TOKENS = (
    # Remote / client-server surface — excluded for C-001, see above.
    "remote",
    "registry_server",
    "feature_server",
    "offline_server",
    "grpc",
    "rest",
    "mcp",
    "ui",
    # Cloud and third-party stores. The local deployment pins
    # `provider: local`, `registry_type: file`, `online_store: sqlite`
    # (D-004); none of these can be selected by any configuration the local
    # package can reach.
    "aerospike",
    "athena",
    "bigquery",
    "bigtable",
    "cassandra",
    "clickhouse",
    "couchbase",
    "datastore",
    "dynamodb",
    "elasticsearch",
    "gcs",
    "hazelcast",
    "hbase",
    "hybrid",
    "ikv",
    "milvus",
    "mongodb",
    "mssql",
    "mysql",
    "postgres",
    "qdrant",
    "redis",
    "redshift",
    "s3",
    "scylladb",
    "singlestore",
    "snowflake",
    "spark",
    "trino",
    # Distributed compute engines and vector search — not used locally, and
    # each drags a large native tree of its own.
    "ray",
    "faiss",
    "vector",
    "contrib",
    "templates",
    "embedded_go",
    "transformation_servers",
)


def _feast_local_backend(module_name):
    """True for Feast modules the local frozen bundle may need."""
    return not any(token in module_name for token in FEAST_EXCLUDED_BACKEND_TOKENS)


hiddenimports += collect_submodules("feast", filter=_feast_local_backend)
hiddenimports += [
    # The three names the local configuration resolves by string, spelled out
    # so that a future change to the filter above cannot drop them silently.
    # `provider: local`  → feast/infra/provider.py PROVIDERS_CLASS_FOR_TYPE
    "feast.infra.passthrough_provider",
    # `online_store.type: sqlite` → feast/repo_config.py ONLINE_STORE_CLASS_FOR_TYPE
    "feast.infra.online_stores.sqlite",
    # `registry.registry_type: file` → feast/repo_config.py REGISTRY_CLASS_FOR_TYPE
    "feast.infra.registry.registry",
    "feast.infra.registry.file",
    # `RepoConfig` defaults `offline_store` to "dask" when none is declared,
    # and the local configuration declares none on purpose (D-002 — local
    # training reads the shared database directly). The class is therefore
    # still resolved by name even though the store is never queried.
    "feast.infra.offline_stores.dask",
    "feast.infra.offline_stores.file_source",
    # Feast's own protobuf modules are imported by generated code paths that
    # PyInstaller follows inconsistently.
    "feast.protos.feast.core",
    "feast.protos.feast.serving",
    "feast.protos.feast.types",
]

# `pyarrow` and `pandas` are pulled in by Feast rather than by anything in
# `sigil_ml`, and pyarrow in particular loads its compiled `_*.so` modules
# lazily from `pyarrow.__init__`.
hiddenimports += collect_submodules("pyarrow")
hiddenimports += ["pandas", "dask", "dask.dataframe"]

# `grpcio` is not part of Feast 0.65.0's local dependency closure — it arrives
# only with the client-server extras this bundle deliberately excludes. It is
# collected if present so that a future dependency change does not silently
# produce a bundle missing its native extensions, and skipped if not, so the
# build does not fail on a package that was never installed.
try:
    hiddenimports += collect_submodules("grpc")
except Exception:  # noqa: BLE001 — absence is the expected case
    pass


# ===========================================================================
# Data files and native libraries
# ===========================================================================
# sklearn / scipy ship compiled data + metadata that must be bundled.
datas = []
datas += collect_data_files("sklearn")
datas += collect_data_files("scipy")

# Feast package data (proto descriptors, feature-repo templates read at
# runtime) and dask's shipped `dask.yaml` defaults, which dask reads on import.
#
# `include_py_files=True` is not a nicety — without it the frozen binary dies on
# `import feast`, before any feature call. `feast/field.py` and its neighbours
# are decorated with typeguard's `@typechecked`, which instruments at import
# time by calling `inspect.getsource()` on the defining module. PyInstaller
# compiles modules into the PYZ archive and ships no `.py`, so `getsource`
# raises `OSError: could not get source code` and the import fails. Collecting
# the sources as data puts them back at the exact path the frozen `__file__`
# already points at (`<sys._MEIPASS>/feast/...`), so `linecache` finds them.
#
# The alternative — building with `optimize=1` so `__debug__` is false and
# typeguard's `@typechecked` short-circuits — was rejected: it would also strip
# every `assert` from every bundled package, changing runtime behaviour far
# outside Feast to work around a packaging problem. This costs ~5 MB.
datas += collect_data_files("feast", include_py_files=True)
datas += collect_data_files("dask")
datas += collect_data_files("pyarrow")

# Distribution metadata (`*.dist-info`), which PyInstaller does not collect by
# default. Feast's dependency tree gates optional imports on
# `importlib.metadata.distribution(name)` rather than on the import itself:
# `dask._compatibility.import_optional_dependency` looks up pandas' version
# this way, and a bundle without the `.dist-info` raises
# `PackageNotFoundError: No package metadata was found for pandas` from inside
# `feast.infra.offline_stores.dask` — again at feature-resolution time, not at
# build or start. `recursive=True` covers the whole tree rather than the
# handful of packages that happen to trip it today.
datas += copy_metadata("feast", recursive=True)
datas += copy_metadata("pandas")
datas += copy_metadata("dask")

# The shipped feature-store assets, all three landing at the one path
# `bundle_dir()` resolves to. The YAML pair is the configuration surface; the
# registry and its provenance marker are the build-time apply output.
datas += [
    (str(FEATURE_STORE_SRC / "feature_store.local.yaml"), BUNDLE_FEATURE_STORE_DIR),
    (str(FEATURE_STORE_SRC / "feature_store.cloud.yaml"), BUNDLE_FEATURE_STORE_DIR),
    (str(_registry_path), BUNDLE_FEATURE_STORE_DIR),
    (str(_registry_marker_path), BUNDLE_FEATURE_STORE_DIR),
]

# Native extensions that `collect_submodules` alone does not bring: pyarrow
# ships `libarrow*.dylib` / `libparquet*.dylib` alongside its extension modules
# and loads them through its own loader, and grpcio (when present) carries a
# statically-linked `cygrpc` extension.
binaries = []
binaries += collect_dynamic_libs("pyarrow")
try:
    binaries += collect_dynamic_libs("grpc")
except Exception:  # noqa: BLE001 — see the grpc note above
    pass

block_cipher = None

a = Analysis(
    ["entrypoint.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Trim optional/cloud-only deps that are not part of the frozen local
    # sidecar (psycopg2/boto3 are the `cloud` extra; matplotlib/IPython are
    # pulled transitively but unused at serve time).
    excludes=[
        "psycopg2",
        "boto3",
        "botocore",
        "matplotlib",
        "IPython",
        "tkinter",
        "pytest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ONEDIR: the EXE is the thin bootloader only (exclude_binaries=True); the
# interpreter + all compiled deps land beside it via COLLECT in
# dist/kameas-ml/_internal/. Signing is deliberately NOT done here
# (codesign_identity=None): kenaz's `make sign-macos` signs the staged tree
# inside-out with the release identity + per-file hardened runtime, so the
# freeze stays identity-agnostic and reproducible across dev/CI.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="kameas-ml",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="kameas-ml",
)
