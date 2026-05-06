---
work_package_id: "WP02"
title: "S3ModelStore Implementation"
lane: "planned"
dependencies: ["WP01"]
subtasks:
  - "T007"
  - "T008"
  - "T009"
  - "T010"
  - "T011"
phase: "Phase 2 - Cloud Storage"
assignee: ""
agent: ""
shell_pid: ""
review_status: ""
reviewed_by: ""
history:
  - timestamp: "2026-03-30T01:45:11Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt generated via /spec-kitty.tasks"
requirement_refs:
  - "FR-005"
  - "FR-006"
---

# Work Package Prompt: WP02 -- S3ModelStore Implementation

## Review Feedback

> **Populated by `/spec-kitty.review`** -- Reviewers add detailed feedback here when work needs changes.

*[This section is empty initially.]*

---

## Markdown Formatting
Wrap HTML/XML tags in backticks: `` `<div>` ``, `` `<script>` ``
Use language identifiers in code blocks: ````python`, ````bash`

---

## Objectives & Success Criteria

- Implement `S3ModelStore` that reads and writes model weight files from an S3-compatible object store with per-tenant key prefix.
- Use timestamp-based versioning with a `latest` pointer per plan.md design decision D5.
- Handle all S3 error conditions gracefully per spec edge cases.
- Add `boto3` as an optional dependency under a `cloud` extras group.
- Wire the S3 backend into the existing `model_store_factory()`.

**Success Criteria**:
- `S3ModelStore.save("stuck", data)` writes to `s3://{bucket}/{tenant_id}/models/stuck/{timestamp}/model.joblib` and updates the `latest` pointer.
- `S3ModelStore.load("stuck")` reads the `latest` pointer, fetches the version, and returns the bytes. Returns `None` if the key doesn't exist.
- Missing keys return `None` without raising exceptions.
- Invalid credentials, missing bucket, or network errors produce clear, actionable error messages.
- `pip install kenaz-ml[cloud]` installs `boto3`. Local install does NOT require `boto3`.

## Context & Constraints

- **Spec**: `kitty-specs/003-model-storage-abstraction/spec.md` -- FR-005, FR-006.
- **Plan**: Design decisions D4 (tenant ID handling), D5 (model versioning), D6 (boto3 as optional).
- **Depends on WP01**: `ModelStore` protocol, `model_store_factory()`, and config helpers are defined in WP01.
- **Key files from WP01**:
  - `src/sigil_ml/storage/model_store.py` -- protocol, `LocalModelStore`, factory (with S3 stub)
  - `src/sigil_ml/config.py` -- `s3_bucket()`, `s3_endpoint_url()`, `aws_region()`, `tenant_id()`
- **Constraint**: `boto3` must be optional -- import guarded with try/except.
- **Constraint**: Synchronous API (boto3's default client is sync).
- **Constraint**: No locking for concurrent saves (S3 last-writer-wins per spec edge cases).
- **Constraint**: S3 temporary unreachability causes `load()` to return `None` (serving falls back to rule-based).

## Subtasks & Detailed Guidance

### Subtask T007 -- Implement S3ModelStore in `src/sigil_ml/storage/model_store.py`

- **Purpose**: Provide the S3-backed storage backend for cloud deployments with per-tenant model isolation and version tracking.

- **Steps**:
  1. Add `S3ModelStore` to `src/sigil_ml/storage/model_store.py` (same file as protocol and LocalModelStore per plan.md):
     ```python
     import logging
     from datetime import datetime, timezone

     logger = logging.getLogger(__name__)


     class S3ModelStore:
         """ModelStore backed by S3-compatible object storage.

         Key structure per plan.md D5:
           {tenant_id}/models/{model_name}/{version}/model.joblib
           {tenant_id}/models/{model_name}/latest  (contains version string)

         The tenant_id is set at construction time (one instance per tenant).
         """

         def __init__(
             self,
             bucket: str,
             tenant_id: str,
             endpoint_url: str | None = None,
             region: str | None = None,
         ) -> None:
             try:
                 import boto3
                 from botocore.config import Config as BotoConfig
             except ImportError:
                 raise ImportError(
                     "boto3 is required for S3 model storage. "
                     "Install with: pip install kenaz-ml[cloud]"
                 ) from None

             self._bucket = bucket
             self._tenant_id = tenant_id
             self._client = boto3.client(
                 "s3",
                 region_name=region or "us-east-1",
                 endpoint_url=endpoint_url,
                 config=BotoConfig(
                     connect_timeout=5,
                     read_timeout=10,
                     retries={"max_attempts": 2},
                 ),
             )
             self._validate_bucket()

         def _validate_bucket(self) -> None:
             """Verify bucket exists and is accessible (fail-fast)."""
             from botocore.exceptions import ClientError, NoCredentialsError

             try:
                 self._client.head_bucket(Bucket=self._bucket)
                 logger.info(
                     "S3ModelStore: connected to bucket=%s tenant=%s",
                     self._bucket, self._tenant_id,
                 )
             except ClientError as e:
                 code = e.response["Error"]["Code"]
                 if code == "404":
                     raise ValueError(
                         f"S3 bucket '{self._bucket}' does not exist. "
                         f"Create it or check SIGIL_S3_BUCKET."
                     ) from e
                 if code == "403":
                     raise PermissionError(
                         f"Access denied to S3 bucket '{self._bucket}'. "
                         f"Check IAM permissions."
                     ) from e
                 raise
             except NoCredentialsError as e:
                 raise PermissionError(
                     "No AWS credentials found. Configure "
                     "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, "
                     "or use an IAM role."
                 ) from e

         def _latest_key(self, model_name: str) -> str:
             return f"{self._tenant_id}/models/{model_name}/latest"

         def _version_key(self, model_name: str, version: str) -> str:
             return f"{self._tenant_id}/models/{model_name}/{version}/model.joblib"

         def load(self, model_name: str) -> bytes | None:
             from botocore.exceptions import ClientError

             # Read the latest pointer
             try:
                 resp = self._client.get_object(
                     Bucket=self._bucket, Key=self._latest_key(model_name)
                 )
                 version = resp["Body"].read().decode("utf-8").strip()
             except ClientError as e:
                 if e.response["Error"]["Code"] == "NoSuchKey":
                     logger.debug("S3ModelStore: no latest pointer for %s", model_name)
                     return None
                 logger.warning("S3ModelStore: error reading latest for %s: %s", model_name, e)
                 return None
             except Exception:
                 logger.warning("S3ModelStore: unexpected error loading %s", model_name, exc_info=True)
                 return None

             # Fetch the versioned model
             try:
                 resp = self._client.get_object(
                     Bucket=self._bucket, Key=self._version_key(model_name, version)
                 )
                 data = resp["Body"].read()
                 logger.info("S3ModelStore: loaded %s version=%s (%d bytes)", model_name, version, len(data))
                 return data
             except ClientError as e:
                 if e.response["Error"]["Code"] == "NoSuchKey":
                     logger.warning("S3ModelStore: latest points to missing version %s for %s", version, model_name)
                     return None
                 logger.warning("S3ModelStore: error loading %s: %s", model_name, e)
                 return None
             except Exception:
                 logger.warning("S3ModelStore: unexpected error loading %s", model_name, exc_info=True)
                 return None

         def save(self, model_name: str, data: bytes) -> None:
             version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

             # Write the versioned model object
             self._client.put_object(
                 Bucket=self._bucket,
                 Key=self._version_key(model_name, version),
                 Body=data,
             )

             # Update the latest pointer
             self._client.put_object(
                 Bucket=self._bucket,
                 Key=self._latest_key(model_name),
                 Body=version.encode("utf-8"),
             )

             logger.info("S3ModelStore: saved %s version=%s (%d bytes)", model_name, version, len(data))

         def exists(self, model_name: str) -> bool:
             from botocore.exceptions import ClientError

             try:
                 self._client.head_object(
                     Bucket=self._bucket, Key=self._latest_key(model_name)
                 )
                 return True
             except ClientError:
                 return False
     ```

- **Files**:
  - Update: `src/sigil_ml/storage/model_store.py`

- **Parallel?**: No -- core implementation.

- **Notes**:
  - `boto3` and `botocore` are imported locally (inside methods and `__init__`) to maintain the lazy-import guarantee.
  - `_validate_bucket()` is called at construction time (fail-fast at startup, not first request).
  - `load()` is a two-step process: read `latest` pointer, then fetch versioned object. Both steps catch errors individually.
  - `save()` re-raises exceptions on failure -- training pipeline must be aware of save failures.
  - `load()` catches all exceptions and returns `None` -- serving falls back to rule-based predictions.
  - The `BotoConfig` sets reasonable timeouts to prevent hanging on network issues.

### Subtask T008 -- Add boto3 as optional dependency in pyproject.toml

- **Purpose**: Allow cloud users to install S3 support without burdening local-mode users.

- **Steps**:
  1. Update `pyproject.toml` optional-dependencies:
     ```toml
     [project.optional-dependencies]
     dev = ["pytest>=8.0", "httpx>=0.27", "ruff>=0.4", "pyre-check>=0.9.18", "moto[s3]>=5.0"]
     cloud = ["boto3>=1.34"]
     ```
  2. `moto[s3]>=5.0` is added to `dev` extras for mocking S3 in tests.
  3. Verify `pip install kenaz-ml` does NOT install `boto3`.
  4. Verify `pip install kenaz-ml[cloud]` installs `boto3` and its transitive dependencies.

- **Files**:
  - Update: `pyproject.toml`

- **Parallel?**: Yes -- independent of the implementation code.

- **Notes**:
  - `boto3>=1.34` pins to a recent stable release.
  - Do NOT add `botocore` separately -- it is a transitive dependency of `boto3`.
  - `moto[s3]` is only needed for tests, not production.

### Subtask T009 -- Handle S3 error conditions

- **Purpose**: Ensure all S3 failure modes are handled gracefully per the spec's edge cases section.

- **Steps**:
  1. **Missing bucket**: `_validate_bucket()` raises `ValueError` with actionable message mentioning `SIGIL_S3_BUCKET`. (Already in T007.)
  2. **Invalid credentials**: `_validate_bucket()` raises `PermissionError` with message about IAM permissions. (Already in T007.)
  3. **Network timeout**: `load()` catches all exceptions, logs warning, returns `None`. The `BotoConfig` with `connect_timeout=5, read_timeout=10, retries={"max_attempts": 2}` prevents indefinite hangs. (Already in T007.)
  4. **Corrupted object bytes**: `load()` returns raw bytes without validation. The model class's `joblib.load(BytesIO(data))` will raise on corruption -- that error is caught in the model's `_load()` method (WP04) and falls back to rule-based.
  5. **Missing `latest` pointer**: `load()` gets `NoSuchKey` on the `latest` key, returns `None`. Treated as "model not yet trained".
  6. **Dangling version pointer**: `latest` points to a version that doesn't exist. `load()` catches `NoSuchKey` on the versioned key, logs warning, returns `None`.
  7. **Concurrent saves**: S3 last-writer-wins. `save()` does not use conditional writes or locking.

- **Files**:
  - Verify/update: `src/sigil_ml/storage/model_store.py`

- **Parallel?**: No -- refinements to T007 implementation.

- **Notes**:
  - The error handling strategy matches plan.md's error handling table.
  - With `retries={"max_attempts": 2}`, a request may take up to ~25 seconds on a completely down network (5s connect * 2 retries + 10s read * 2 retries). This is acceptable for a background operation; the serving layer returns fallback predictions in the meantime.

### Subtask T010 -- Update model_store_factory() for S3 backend

- **Purpose**: Replace the `NotImplementedError` stub from WP01 with actual S3 backend instantiation.

- **Steps**:
  1. Update the `s3` branch in `model_store_factory()`:
     ```python
     if backend == "s3":
         bucket = config.s3_bucket()
         if not bucket:
             raise ValueError(
                 "SIGIL_S3_BUCKET must be set for S3 model storage backend."
             )
         tid = tenant_id or config.tenant_id()
         return S3ModelStore(
             bucket=bucket,
             tenant_id=tid,
             endpoint_url=config.s3_endpoint_url(),
             region=config.aws_region(),
         )
     ```
  2. The S3 import is NOT deferred here because `S3ModelStore` is in the same module. The lazy `boto3` import is inside `S3ModelStore.__init__`.
  3. Validate that `SIGIL_S3_BUCKET` is set -- empty string is not valid.

- **Files**:
  - Update: `src/sigil_ml/storage/model_store.py`

- **Parallel?**: No -- depends on T007.

- **Notes**:
  - `tenant_id` parameter takes precedence over config default. This supports per-request tenant switching in cloud mode.
  - The bucket validation here is separate from `_validate_bucket()` -- this catches "not configured" while `_validate_bucket()` catches "configured but inaccessible".

### Subtask T011 -- Add startup bucket validation

- **Purpose**: Verify that the S3 bucket is accessible at `S3ModelStore` construction time, producing clear errors for operators.

- **Steps**:
  1. The `_validate_bucket()` method is already defined in T007. This subtask verifies it covers all scenarios:
     - Bucket exists and is accessible: logs success message.
     - Bucket does not exist (404): raises `ValueError` with "Create it or check SIGIL_S3_BUCKET."
     - Access denied (403): raises `PermissionError` with "Check IAM permissions."
     - No credentials: raises `PermissionError` with "Configure AWS_ACCESS_KEY_ID..."
  2. Verify the validation happens at construction time, not on first `load()`/`save()`.
  3. Verify the error messages include the specific env var names to check.

- **Files**:
  - Verify: `src/sigil_ml/storage/model_store.py`

- **Parallel?**: No -- depends on T007.

- **Notes**:
  - This is a verification subtask. The implementation is already part of T007's `_validate_bucket()`.
  - The fail-fast design means cloud deployments surface configuration issues at startup, not at the first prediction request.
  - In production K8s deployments, this causes the pod to crash and be restarted, which is the desired behavior for misconfigured services.

## Risks & Mitigations

- **Risk**: `boto3` dependency size (~100MB installed). **Mitigation**: Optional extras group. Local users never see it.
- **Risk**: AWS credential management complexity. **Mitigation**: `boto3` supports the standard credential chain (env vars, IAM role, shared credentials file). No custom management needed.
- **Risk**: S3 strong read-after-write consistency concern. **Mitigation**: S3 provides strong read-after-write consistency for PUTs since December 2020. Non-issue.
- **Risk**: MinIO compatibility differences. **Mitigation**: MinIO implements the S3 API. The `endpoint_url` parameter enables pointing to any S3-compatible store.
- **Risk**: Versioning creates many objects per model. **Mitigation**: Old versions are not cleaned up in this feature. A retention policy / lifecycle rule can be added to the S3 bucket independently.

## Review Guidance

- Verify `S3ModelStore` satisfies the `ModelStore` protocol (has `load`, `save`, `exists` with correct signatures).
- Verify `latest` pointer versioning: `save()` writes version + pointer, `load()` reads pointer then version.
- Verify bucket validation at construction time with clear, actionable error messages.
- Verify `load()` NEVER raises exceptions -- always returns `bytes | None`.
- Verify `save()` DOES raise on failure (training pipeline awareness).
- Verify `boto3` import is lazy (inside `__init__`, not at module level).
- Verify `pyproject.toml` has `cloud` extras group with `boto3>=1.34`.
- Verify `BotoConfig` timeout and retry settings are present.

## Implementation Command

```bash
spec-kitty implement WP02 --base WP01
```

## Activity Log

- 2026-03-30T01:45:11Z -- system -- lane=planned -- Prompt created.
