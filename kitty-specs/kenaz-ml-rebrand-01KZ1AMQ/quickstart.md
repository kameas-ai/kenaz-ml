# Quickstart: kenaz-ml Rebrand

**Date**: 2026-08-02 | **Plan**: [plan.md](./plan.md)

Verification in the order a reviewer should check it. The first item is the one that could cause real damage; the rest confirm the rename is complete and inert.

---

## 1. The Sigil integration is untouched (SC-004, FR-007) — check this first

```bash
git grep -c "sigild" | awk -F: '{s+=$2} END {print s}'    # expect 138, unchanged
git grep -n "SIGILD_PLUGIN_URL" src/                       # variable name unchanged
git grep -n "share/sigild" src/                            # data path unchanged
```

Then prove it at runtime, since a path can be textually present and still wrong:

```python
from kenaz_ml import config
assert config.db_path() == Path.home()/".local/share/sigild/data.db"
```

`kenaz-ml` is a sidecar for a **separate** Go product. If the data path moved, every install now points at a database that does not exist — and it would import perfectly while doing so.

## 2. Nothing behaves differently (SC-005, FR-008)

```bash
pytest tests/      # expect no count regression: 959 passed, 9 skipped
```

The count must not fall. A dropped test is a module that silently stopped being collected.

Then the check a count cannot give you — same fixture, same pinned clock, before and after:

```bash
curl -sf -X POST localhost:7774/predict/stuck -d '{"task_id":"..."}' | diff - expected.json
```

## 3. The old names are gone, the new ones resolve (SC-001, SC-008)

```python
import kenaz_ml                                    # succeeds
with pytest.raises(ModuleNotFoundError):
    import sigil_ml                                # gone
```

```bash
git grep -l "sigil_ml" -- src tests freeze scripts docs Makefile pyproject.toml
# expect: only tests/test_migration_regression.py (the frozen-history alias, D-004)

git grep -l "kameas-ml" -- src tests freeze scripts .github Makefile pyproject.toml
# expect: empty
```

`kitty-specs/` for merged missions is deliberately excluded — those are historical records (D-003).

## 4. Distribution, CLI and metadata (SC-002, SC-003)

```bash
pip install -e ".[dev]"
kenaz-ml --version                    # the command is kenaz-ml
python -c "from importlib.metadata import metadata; m=metadata('kenaz-ml'); print(m['Name'], m['Home-page'])"
```

URLs must resolve to the real repository — `github.com/kameas-ai/kenaz-ml`. The organisation stays `kameas-ai` (D-007); only the repository name was stale.

## 5. The string-resolved references (FR-011, D-005)

These fail at *runtime*, not at import, so the suite passing proves nothing about them:

```bash
grep -n "kenaz_ml.app:app" src/kenaz_ml/cli.py           # uvicorn target
grep -n "collect_submodules\|hiddenimports" freeze/*.spec # PyInstaller
grep -n "kenaz-ml = " pyproject.toml                      # entry point
```

Then exercise each: start the server through the CLI (not `uvicorn` directly), and build the binary.

## 6. The frozen binary (SC-006, FR-009)

```bash
make freeze
```

**Exercise a real prediction in the built binary**, not just startup. Two prior missions found packaging bugs that appeared only on first *use* — "builds clean, dies on first call" is this codebase's characteristic failure. Confirm the spec file's own name change did not break the Makefile target that invokes it.

## 7. Pre-rename artifacts still load (SC-007, FR-010)

Load a `.joblib` written before the rename. It should load unchanged — the pickled objects are plain sklearn estimators with no project module path recorded. If this fails, something embedded the package name that the probe said did not.

## 8. The frozen-history alias (FR-012, D-004)

```bash
pytest tests/test_migration_regression.py -k baselines_were_recorded
```

`sys.modules["sigil_ml.store"]` **must keep the old key** — it matches text frozen in git object `ef67e05`, whose SHA256 is asserted. Only the value it points at follows the rename. A reviewer will see `sigil_ml` there and should recognise it as correct, not as a leftover.

## 9. Import time (NFR-004)

```bash
python -X importtime -c "import kenaz_ml.app" 2>&1 | tail -1
```

Within 10% of the pre-change figure.

---

## What this mission deliberately does not do

- **Rename anything belonging to the Sigil daemon** — `sigild`, its data path, its config variables, or documentation describing it.
- **Rename the GitHub organisation** — `kameas-ai` is correct.
- **Rewrite merged missions' specs** — historical records (D-003).
- **Move or restructure any module** — this is a rename; the storage reorganization already settled the layout.

## Rollback

Revert the commit and reinstall (`pip install -e ".[dev]"`) so the editable `.pth` and entry point return to the old distribution name. No state, schema or artifact format changed, so nothing else needs unwinding. Clear stale bytecode with `find . -name __pycache__ -exec rm -rf {} +`.
