.PHONY: openapi openapi-check install lint format test build clean freeze freeze-smoke

# Generate the OpenAPI spec from the FastAPI app
openapi:
	python scripts/gen_openapi.py

# Verify the committed spec matches the code (used by CI)
openapi-check:
	python scripts/gen_openapi.py --check

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

test:
	pytest tests/ -v

build:
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info

# FR-3 (ADR-ml-packaging.md): produce the frozen, self-contained `kameas-ml`
# executable for the CURRENT HOST PLATFORM. Requires the build-time freeze
# extra (`pip install -e ".[freeze]"`). The multi-platform matrix is ML-DEBT-2.
freeze:
	pyinstaller freeze/kameas-ml.spec --noconfirm --clean

# Run the freeze smoke test against a built artifact. Boots dist/kameas-ml on
# an ephemeral port and asserts /predict/stuck returns a real sklearn
# prediction — the guard for the known sklearn/numpy/uvicorn hidden-import
# breakage. Run `make freeze` first.
freeze-smoke:
	KAMEAS_ML_FROZEN_BIN=$(PWD)/dist/kameas-ml pytest tests/test_frozen_smoke.py -v
