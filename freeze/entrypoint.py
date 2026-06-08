"""Frozen-binary entrypoint for `kameas-ml`.

PyInstaller freezes a real script file rather than a console_scripts shim, so
this module exists purely to invoke the unchanged CLI ``main()``. The runtime
behaviour is identical to ``kameas-ml`` installed via pip — same subcommands
(``serve --port``, ``train``, ``health-check``), same routes, same WAL
contract (kenaz-ml/CLAUDE.md). See ADR-ml-packaging.md.
"""

from __future__ import annotations

import multiprocessing

from sigil_ml.cli import main

if __name__ == "__main__":
    # PyInstaller one-file builds re-exec the bootloader for child processes;
    # freeze_support() makes any multiprocessing-based worker (joblib/loky,
    # uvicorn reload) behave correctly in the frozen binary.
    multiprocessing.freeze_support()
    main()
