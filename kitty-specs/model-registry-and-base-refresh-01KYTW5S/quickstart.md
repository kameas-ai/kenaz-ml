# Quickstart: Model Registry and Base Refresh

**Date**: 2026-07-31 | **Plan**: [plan.md](./plan.md)

How to verify this feature, in the order a reviewer should check it.

## Setup

```bash
pip install -e ".[dev]"
pytest tests/test_registry_*.py -v
```

No new dependencies. No migrations. No base models exist yet, so the default state of every install is "base slot empty" — check that path first.

---

## 1. Nothing shipped, nothing broken (SC-008)

The most important test today, because it is the state every install is actually in.

```bash
rm -rf ~/.local/share/sigild/ml-models        # no local models either
pytest tests/ -k "cold_start or no_base" -v
```

Behavior must be identical to pre-feature: resolution falls through both slots to existing cold-start behavior without raising, and training proceeds as it does today.

## 2. Integrity refusal (SC-002)

```python
write_artifact_and_manifest(name="stuck")
tamper_one_byte(artifact_path)

result = registry.load("stuck")
assert result.refused
assert result.reason == "integrity"
assert not deserialization_occurred()      # the point: refused BEFORE joblib.load
```

The ordering assertion is the substance here. A checksum verified after deserialization provides no protection at all.

## 3. Contract refusal (SC-001)

```python
manifest["feature_contract"]["names"].remove("edit_velocity")
result = registry.load("stuck")
assert result.refused and result.reason == "contract"
assert "edit_velocity" in result.detail     # names the disagreement
```

Also assert the reordering case — same names, different order — is refused. That is the failure a set comparison would miss, and it silently permutes every model input.

## 4. Base is never written (SC-005)

```python
base_hash = sha256(base_artifact_path)
run_local_training(); run_local_training(); run_local_training()
assert sha256(base_artifact_path) == base_hash
assert manifest("local", "stuck")["provenance"]["n_local_extensions"] == 3
```

## 5. Same-contract refresh preserves personalization (SC-003)

The headline behavior of the chosen policy.

```python
build_local_from(base_version="1", retained=KNOWN_SET)
local_before = predictions(local_model, HOLDOUT)

ship_base(version="2", contract_version="2")   # contract UNCHANGED
registry.refresh()

after = predictions(load("stuck"), HOLDOUT)
bare  = predictions(bare_base_v2(), HOLDOUT)
assert distance(after, local_before) < distance(bare, local_before)
```

The inequality is the assertion that matters — it proves retained data actually influenced the rebuild, rather than the new base merely being served.

## 6. Changed-contract refresh resets honestly (SC-004)

```python
ship_base(version="2", contract_version="3")   # contract CHANGED
registry.refresh()

assert retained_set("stuck").is_empty()
assert manifest("local","stuck")["provenance"]["reset_reason"] == "contract_version_changed"
assert manifest("local","stuck")["provenance"]["n_local_extensions"] == 0
```

## 7. Failed refresh is survivable (FR-019)

Inject a failure mid-rebuild; assert the previous local artifact and manifest are byte-identical afterward and still served. Temp-write-then-move is what makes this hold.

## 8. Retained data is local, inspectable, deletable (SC-006, FR-018)

```bash
head -2 ~/.local/share/sigild/ml-models/retained/stuck.jsonl   # header + one example, readable
```

```python
assert no_network_egress_during(run_local_training)
delete_retained("stuck")
assert registry.load("stuck").ok          # still serving
assert retained_set("stuck").is_empty()   # accumulation restarts
```

The egress assertion should be structural — assert no socket is opened by the training path — not merely that no known upload function was called.

## 9. Provenance is answerable (SC-007)

```python
p = registry.provenance("stuck")
assert p.base_version and p.n_local_extensions is not None and p.contract_version
```

## 10. Bounds hold (NFR-004)

Append past the configured cap; assert the file stops growing, that the newest examples survive, and that the header remains valid.

---

## What this feature deliberately does not do

- **No discovery surface.** `/introspect` is not extended here; that is the next mission. The registry exposes the data it will read.
- **No feature selection.**
- **No warm-start.** Rebuild is full retraining (D-003).
- **No cloud registry, no MLflow, no export job.** The manifest schema is designed to be writable by that job, but the job is not built here.
- **No base models.** They remain manual central work; this feature makes shipping them safe.
- **No signing.** Checksums only.

## Rollback

Revert the commit. Local artifacts written under the new layout remain readable by the old code, since the local slot keeps the existing `{name}.joblib` path — only the sidecar manifests and the `retained/` directory become orphaned files, which are inert.
