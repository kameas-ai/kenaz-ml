"""Retained training sets kept in S3, for the cloud worker.

The retained-set format and its rules live in
:mod:`kenaz_ml.modelstore.registry.retained`, which is filesystem-only by
design. A cloud worker has no durable disk, so it keeps each stream's retained
sets in S3 under that stream's own prefix and works on them in a scratch
directory: :meth:`S3RetainedMirror.pull` before a training run,
:meth:`S3RetainedMirror.push` after it.

The bucket is versioned. Every rewrite of a retained file would otherwise
leave the examples it just expired or evicted behind in a noncurrent version,
and a deletion would leave only a delete marker. So :meth:`push` removes every
noncurrent version under the prefix, and :func:`purge_prefix` removes every
version and delete marker: when either returns, the data is gone, not hidden.

Cloud only. Nothing in the local deployment imports this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUFFIX = ".jsonl"
_DELETE_BATCH = 1000


def _versions(client: Any, bucket: str, prefix: str) -> list[dict[str, Any]]:
    """Every version and delete marker under ``prefix``: {Key, VersionId, IsLatest, LastModified}.

    ``LastModified`` is S3's timezone-aware datetime, or ``None`` when absent.
    """
    found: list[dict[str, Any]] = []
    paginator = client.get_paginator("list_object_versions")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for entry in [*page.get("Versions", []), *page.get("DeleteMarkers", [])]:
            if not entry["Key"].startswith(prefix):
                raise AssertionError(f"S3 listed {entry['Key']!r} outside {prefix!r}")
            found.append(
                {
                    "Key": entry["Key"],
                    "VersionId": entry["VersionId"],
                    "IsLatest": entry["IsLatest"],
                    "LastModified": entry.get("LastModified"),
                }
            )
    return found


def _delete_versions(client: Any, bucket: str, entries: list[dict[str, Any]]) -> int:
    objects = [{"Key": e["Key"], "VersionId": e["VersionId"]} for e in entries]
    for start in range(0, len(objects), _DELETE_BATCH):
        batch = objects[start : start + _DELETE_BATCH]
        resp = client.delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})
        if resp.get("Errors"):
            first = resp["Errors"][0]
            raise RuntimeError(f"S3 refused to delete {len(resp['Errors'])} version(s), first {first}")
    return len(objects)


def purge_prefix(client: Any, bucket: str, prefix: str) -> int:
    """Permanently delete every object version and delete marker under ``prefix``.

    ``prefix`` must name a directory (end in ``/``) so a sibling sharing its
    leading characters can never match. Returns the number of versions removed.
    """
    if not prefix.endswith("/") or prefix == "/":
        raise ValueError(f"refusing to purge {prefix!r}: a prefix must end in '/' and not be the root")
    removed = _delete_versions(client, bucket, _versions(client, bucket, prefix))
    logger.info("s3: purged %d version(s) under %s", removed, prefix)
    return removed


class S3RetainedMirror:
    """One stream's retained sets in S3 at ``prefix`` (ending in ``/``)."""

    def __init__(self, client: Any, bucket: str, prefix: str) -> None:
        if not prefix.endswith("/"):
            raise ValueError(f"retained prefix {prefix!r} must end in '/'")
        self._s3 = client
        self._bucket = bucket
        self._prefix = prefix

    @property
    def prefix(self) -> str:
        return self._prefix

    def pull(self, directory: Path) -> int:
        """Replace ``directory``'s retained files with the S3 copies. Returns files pulled."""
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob(f"*{SUFFIX}"):
            stale.unlink()
        pulled = 0
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
            for obj in page.get("Contents", []):
                name = obj["Key"][len(self._prefix) :]
                if "/" in name or not name.endswith(SUFFIX):
                    continue
                body = self._s3.get_object(Bucket=self._bucket, Key=obj["Key"])["Body"].read()
                (directory / name).write_bytes(body)
                pulled += 1
        return pulled

    def push(self, directory: Path) -> int:
        """Make S3 hold exactly ``directory``'s retained files, with no older versions.

        Uploads each file, deletes objects with no local file, then removes
        every noncurrent version under the prefix. Returns files uploaded.
        """
        local = {p.name: p for p in directory.glob(f"*{SUFFIX}")} if directory.exists() else {}
        for name, path in sorted(local.items()):
            self._s3.put_object(Bucket=self._bucket, Key=self._prefix + name, Body=path.read_bytes())
        entries = _versions(self._s3, self._bucket, self._prefix)
        gone = [e for e in entries if e["Key"][len(self._prefix) :] not in local]
        noncurrent = [e for e in entries if not e["IsLatest"] and e not in gone]
        _delete_versions(self._s3, self._bucket, gone + noncurrent)
        return len(local)

    def purge(self) -> int:
        """Permanently delete this stream's retained sets (member erasure)."""
        return purge_prefix(self._s3, self._bucket, self._prefix)
