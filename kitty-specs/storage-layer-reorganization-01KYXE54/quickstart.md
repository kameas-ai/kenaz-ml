# Quickstart: Storage Layer Reorganization

**Date**: 2026-08-01 | **Plan**: [plan.md](./plan.md)

Verification in the order a reviewer should check it. This is a behaviour-preserving move, so the tests that matter are the ones proving *nothing changed*.

## Setup

```bash
uv venv .venv-check && uv pip install -q --python .venv-check/bin/python -e ".[dev]"
.venv-check/bin/pytest tests/ -q
```

No venv is checked in, and any left from an earlier session predates Feast and will fail collection.

---

## 1. Nothing changed at runtime (SC-003, FR-005) — the whole point

```bash
pytest tests/     # expect no count regression: >= 487 passed, 9 skipped
```

The count should *rise* by the new Stack B coverage, never fall. A dropped test is a missed import that silently stopped exercising something.

Then the behavioural check that a test count cannot give you:

```bash
# same fixture, same pinned clock, before and after
curl -sf -X POST localhost:7774/predict/stuck -d '{"task_id":"..."}' | diff - expected.json
```

## 2. No module survives at its old path (SC-007, FR-004)

```python
for old in ("sigil_ml.store", "sigil_ml.store_sqlite", "sigil_ml.store_postgres",
            "sigil_ml.storage.model_store", "sigil_ml.loader", "sigil_ml.cache"):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old)
```

A shim left behind would let stale imports keep working and reintroduce the ambiguity in a new form (D-004).

## 3. The public surface is what the plan says (FR-010)

```python
from sigil_ml.datastore import DataStore, create_store
from sigil_ml.modelstore import (
    ModelStore, LocalModelStore, S3ModelStore, CachedModelStore, model_store_factory,
    ModelLoader, FilesystemModelLoader, ModelCache, create_model_cache,
)
```

Every name must resolve from the *package*, not only from a submodule — the follow-on registry mission imports these.

## 4. Stack B is now tested (SC-006, FR-008/009)

The coverage that did not exist before. Each must fail if the behaviour regresses:

- **TTL expiry** — an entry past its TTL reads as a miss
- **LRU eviction** — at capacity, the least-recently-used entry goes
- **Tenant scoping** — two tenants requesting the same model name never receive each other's artifact
- **Shared fallback** — with no tenant-specific artifact, the shared one is used
- **`None` on missing** — a missing artifact returns `None`, never raises, per the protocol docstring

Prove they bite rather than trusting them:

```bash
# break each behaviour in a scratch copy and confirm the test fails
```

## 5. Old artifacts still load (SC-005, FR-007)

Write a `.joblib` with the pre-change code, load it with the post-change code, assert the model is equivalent. Nothing about the artifact format is intentionally changing, so a failure here means something moved that should not have.

## 6. The frozen binary still builds (SC-004, FR-006)

```bash
make freeze
```

`collect_submodules("sigil_ml")` should reach `datastore/` and `modelstore/` without a spec change — but verify rather than assume, and exercise a real prediction in the built binary rather than accepting a clean build. The Feast mission's WP05 found two packaging bugs that only appeared on first *use*.

## 7. Import time did not regress (NFR-003)

```bash
python -X importtime -c "import sigil_ml.app" 2>&1 | tail -1
```

Within 10% of the pre-change figure. Two new `__init__.py` files doing re-exports could plausibly pull more at import time than the flat modules did.

## 8. The occurrence map was honoured (NFR-004, C-006)

Cross-check `occurrence_map.yaml` against the final diff. Every listed occurrence should be accounted for, and the diff should contain no moved-symbol edit the map did not anticipate. An unlisted edit means the map was incomplete — worth knowing for the next bulk edit.

---

## What this mission deliberately does not do

- **Reconcile the two caches.** `ModelCache` and `CachedModelStore` still coexist. Merging them changes behaviour (C-002).
- **Delete either model stack.** Both remain (C-003).
- **Reorganize `feature_store/`, `models/`, `signals/`, `training/`.**
- **Change any protocol.** `DataStore` and `ModelStore` are moved, not edited.

## Rollback

Revert the commit. No state, schema, or artifact format changed, so there is nothing to unwind — the only risk of reverting is stale `__pycache__`, which `find . -name __pycache__ -exec rm -rf {} +` clears.
