#!/usr/bin/env python3
"""Generate the OpenAPI spec from the FastAPI app and write it to docs/openapi.yaml.

Usage:
    python scripts/gen_openapi.py          # write to docs/openapi.yaml
    python scripts/gen_openapi.py --check  # exit 1 if spec is out of date
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Import the app to get the auto-generated schema.
from kenaz_ml.app import create_app
from kenaz_ml.config import ServingMode

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "docs" / "openapi.json"


def generate() -> str:
    app = create_app(mode=ServingMode.LOCAL)
    schema = app.openapi()
    return json.dumps(schema, indent=2, sort_keys=False) + "\n"


def main() -> None:
    spec = generate()

    if "--check" in sys.argv:
        if not SPEC_PATH.exists():
            print(f"FAIL: {SPEC_PATH} does not exist. Run 'make openapi' to generate it.")
            sys.exit(1)
        existing = SPEC_PATH.read_text()
        if existing != spec:
            print(f"FAIL: {SPEC_PATH} is out of date. Run 'make openapi' to update it.")
            sys.exit(1)
        print(f"OK: {SPEC_PATH} is up to date.")
        return

    SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPEC_PATH.write_text(spec)
    print(f"Wrote {SPEC_PATH} ({len(spec)} bytes)")


if __name__ == "__main__":
    main()
