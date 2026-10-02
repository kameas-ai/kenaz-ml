.PHONY: openapi openapi-check typewriter typewriter-check typewriter-test install lint format test build clean freeze freeze-smoke

# Generate the OpenAPI spec from the FastAPI app
openapi:
	uv run python scripts/gen_openapi.py

# Verify the committed spec matches the code (used by CI)
openapi-check:
	uv run python scripts/gen_openapi.py --check

# Generate the Python models and the Go module from typewriter/spec.yaml
typewriter:
	uv run python scripts/gen_typewriter.py

# Verify the generated files match the spec (used by CI; needs gofmt)
typewriter-check:
	uv run python scripts/gen_typewriter.py --check --strict

# Run the Go module's conformance tests against the shared fixtures
typewriter-test:
	cd typewriter && go vet ./... && go test ./...

# Create .venv on the pinned Python and install the locked dependencies
install:
	uv sync

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

test:
	uv run pytest tests/ -v

build:
	uv build

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info

# FR-3 (ADR-ml-packaging.md) + spec 069 LD-3/FR-002: produce the frozen,
# self-contained ML sidecar ONEDIR bundle for the CURRENT HOST PLATFORM:
#   dist/kameas-ml/kameas-ml  (bootloader exe)
#   dist/kameas-ml/_internal/ (interpreter + sklearn/numpy/... dylibs)
# The artifact's name differs from the product's on purpose: kenaz resolves the
# sidecar by it (spec 069 LD-3, resolveMLBinary()). It is a cross-repo
# interface, not branding — read the freeze spec's header before "fixing" it.
# Onedir (not onefile) because notarization rejects onefile's runtime
# self-extraction of unsigned dylibs — see freeze/kenaz-ml.spec header.
# Needs only uv on PATH: the recipe requests the build-time `freeze` group itself.
freeze:
	uv run --group freeze pyinstaller freeze/kenaz-ml.spec --noconfirm --clean

# Run the freeze smoke test against a built artifact. Boots the onedir
# executable on an ephemeral port and asserts /predict/stuck returns a real
# sklearn prediction — the guard for the known sklearn/numpy/uvicorn
# hidden-import breakage. Run `make freeze` first.
freeze-smoke:
	KENAZ_ML_FROZEN_BIN=$(PWD)/dist/kameas-ml/kameas-ml uv run pytest tests/test_frozen_smoke.py -v
