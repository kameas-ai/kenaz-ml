# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the frozen, self-contained `kameas-ml` executable.
#
# FR-3 (ADR-ml-packaging.md, Option 2): kenaz supervises a frozen Python ML
# sidecar instead of dropping to a fake backend. This spec produces a single
# self-contained executable that carries its own interpreter + scikit-learn +
# numpy + uvicorn + joblib + fastapi — no system Python, no `pip install` on
# the user's machine.
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

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# --- Hidden imports ---------------------------------------------------------
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

# sklearn / scipy ship compiled data + metadata that must be bundled.
datas = []
datas += collect_data_files("sklearn")
datas += collect_data_files("scipy")

block_cipher = None

a = Analysis(
    ["entrypoint.py"],
    pathex=[],
    binaries=[],
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
