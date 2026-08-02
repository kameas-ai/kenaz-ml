"""Retained training set — the local examples a personalized model was built from.

One JSONL file per model at ``{retained_data_dir()}/{name}.jsonl``: a single
header record, then one example per line (D-002, data-model.md §Retained
Training Set)::

    {"record":"header","contract_version":"9f2c…","generation":"2","names":["test_failure_count",…],"created_at":1753900000000}
    {"record":"example","x":[5.0,1800.0,2.3,0.4,3600.0,900.0],"y":1.0,"as_of_ms":1753812345678}
    {"record":"example","x":[1.0,300.0,0.8,0.9,1200.0,1200.0],"y":0.0,"as_of_ms":1753814445678}

Why a text format
-----------------
FR-018 requires the user be able to *inspect* what is retained, and FR-011/C-003
require it never leave the machine. "We keep some of your data locally, in a
binary blob" is a materially weaker claim than a file the user can open. At
roughly 100 bytes per six-float row the 50 MB default bound holds on the order
of 500,000 examples — orders of magnitude past what one install accumulates —
so compactness was never the binding constraint (D-002). ``head -2`` on any
retention file shows a header and an example; if it does not, this module has
failed its purpose.

``names`` is duplicated from the feature contract so the file is self-describing
when read on its own. ``as_of_ms`` is the reference time the vector was computed
at, carried through from the ``feature-extraction-correctness`` work — it is
recorded, never recomputed.

Local-only, structurally
------------------------
This module imports the standard library only, and only the parts of it that
touch the filesystem. It opens no socket, imports nothing that can, and must
never gain a code path that could. WP06 verifies that structurally; nothing here
should give it anything to find.

Retention policy (FR-010, NFR-004)
----------------------------------
The policy is deliberately simple enough to state in one sentence, because
"predictable" means a user can reason about what is kept without reading this
code:

    **Each model's retention file is capped (50 MB by default). When an append
    pushes it past the cap, the oldest examples are dropped — and only the
    oldest — until the remainder fits. The header is always preserved.**

Consequences worth stating plainly:

* Eviction is oldest-first and contiguous. The file is always a suffix of
  everything ever appended, never a sample of it.
* The cap counts bytes on disk, header included, so the file never exceeds it.
* An append is never rejected for being too large; it is admitted and older
  examples make room. A single example larger than the cap leaves the file with
  a header and no examples rather than an over-cap file.
* Eviction rewrites the file through a temporary file and :func:`os.replace`, so
  an eviction interrupted at any point leaves either the pre-eviction file or
  the post-eviction file — never a truncated one.

Reading never raises
--------------------
A retention file is written incrementally and can be interrupted by anything
that ends a process. Reading it must therefore never be the thing that breaks
training: :func:`read_retained` returns an empty result instead of raising, in
every failure mode (missing or unparseable header, truncated final line, empty
file, corruption mid-file). An unreadable retention file costs personalization,
not availability.

Appending, by contrast, verifies that the existing header's ``contract_version``
matches the contract being appended under, and *refuses* on mismatch. That is a
programming error at this layer, not a data condition — resets are WP04's job,
and silently mixing vectors computed under two different contracts into one file
would poison every model trained from it.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kenaz_ml.modelstore.registry.manifest import FeatureContract

__all__ = [
    "DEFAULT_MAX_BYTES",
    "INITIAL_GENERATION",
    "RECORD_EXAMPLE",
    "RECORD_HEADER",
    "AppendResult",
    "Example",
    "ResetResult",
    "RetainedHeader",
    "RetainedSet",
    "RetainedSummary",
    "append_examples",
    "delete_retained",
    "next_generation",
    "read_retained",
    "reset_retained",
    "retained_path",
    "summarize_all",
    "summarize_retained",
]

logger = logging.getLogger(__name__)

RECORD_HEADER = "header"
RECORD_EXAMPLE = "example"

#: NFR-004. Per model, header included. Override per call via ``max_bytes``.
DEFAULT_MAX_BYTES = 50 * 1024 * 1024

#: The generation a set starts at. Increments on every reset (FR-014).
INITIAL_GENERATION = "1"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Example:
    """One retained training example.

    ``x`` is the feature vector *in the header's ``names`` order* — the same
    positional contract the trainers build against. ``as_of_ms`` is the
    reference time the vector was computed at; it is carried from the extractor
    that produced it and is never recomputed at write time, because recomputing
    it against the wall clock would make elapsed features measure the age of the
    record rather than the behaviour it describes.
    """

    x: tuple[float, ...]
    y: float
    as_of_ms: int | None = None


@dataclass(frozen=True)
class RetainedHeader:
    """The first record in a retention file.

    ``contract_version`` is :attr:`FeatureContract.service_version` — the hash
    that changes exactly when the contract does. Refresh compares it against the
    shipped base's contract to decide rebuild versus reset (FR-013/FR-014).
    """

    contract_version: str
    generation: str
    names: tuple[str, ...] = ()
    created_at: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetainedSet:
    """What a read produced. Never carries an exception; see :func:`read_retained`."""

    path: Path
    header: RetainedHeader | None = None
    examples: tuple[Example, ...] = ()
    skipped_lines: int = 0
    truncated_final_line: bool = False
    reason: str | None = None

    @property
    def ok(self) -> bool:
        """True when a usable header was read. Examples may still be empty."""
        return self.header is not None

    @property
    def contract_version(self) -> str | None:
        """The contract the retained vectors were computed under, if readable."""
        return self.header.contract_version if self.header else None

    @property
    def generation(self) -> str | None:
        """The generation of this set, if readable."""
        return self.header.generation if self.header else None

    @property
    def names(self) -> tuple[str, ...]:
        """The ordered feature names the file declares. Empty when unreadable."""
        return self.header.names if self.header else ()

    def __len__(self) -> int:
        return len(self.examples)


@dataclass(frozen=True)
class AppendResult:
    """Outcome of an append. ``ok`` is False only for a refusal, never for I/O success."""

    path: Path
    ok: bool
    written: int = 0
    evicted: int = 0
    header_written: bool = False
    generation: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ResetResult:
    """Outcome of a reset: what the generation was, and what it now is."""

    path: Path
    previous_generation: str | None
    next_generation: str
    deleted: bool
    header_written: bool = False


@dataclass(frozen=True)
class RetainedSummary:
    """Inspection surface for FR-018 — what is retained, how much, under what stamp."""

    name: str
    path: Path
    exists: bool
    size_bytes: int = 0
    n_examples: int = 0
    contract_version: str | None = None
    generation: str | None = None
    created_at: int | None = None
    names: tuple[str, ...] = ()
    skipped_lines: int = 0
    max_bytes: int = DEFAULT_MAX_BYTES
    readable: bool = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _default_directory() -> Path:
    """Resolve the retained-data directory, creating it if needed.

    Prefers ``config.retained_data_dir()``. That function is owned by the slot
    layout work; until it lands this falls back to the same location it defines
    (``models_dir() / "retained"``), so this module never depends on the order
    the two land in.
    """
    from kenaz_ml import config

    resolver = getattr(config, "retained_data_dir", None)
    if callable(resolver):
        directory = Path(resolver())
    else:
        directory = Path(config.models_dir()) / "retained"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def retained_path(name: str, *, directory: Path | str | None = None) -> Path:
    """Return the retention file path for ``name``, creating its directory."""
    if directory is None:
        return _default_directory() / f"{name}.jsonl"
    resolved = Path(directory)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved / f"{name}.jsonl"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _dumps(payload: dict[str, Any]) -> str:
    """One compact JSON object per line, strict about non-finite floats.

    ``allow_nan=False`` matters: Python's ``json`` emits bare ``NaN``/``Infinity``
    by default, which is not JSON and which any other tool reading this file
    would reject. Keeping the file strictly valid is part of it being inspectable.
    """
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _header_line(header: RetainedHeader) -> str:
    payload: dict[str, Any] = {
        "record": RECORD_HEADER,
        "contract_version": header.contract_version,
        "generation": header.generation,
        "names": list(header.names),
        "created_at": header.created_at,
    }
    payload.update(header.extra)
    return _dumps(payload)


def _example_line(example: Example) -> str:
    payload: dict[str, Any] = {
        "record": RECORD_EXAMPLE,
        "x": [float(v) for v in example.x],
        "y": float(example.y),
        "as_of_ms": example.as_of_ms,
    }
    return _dumps(payload)


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# T011 — tolerant reader. No branch of this section may raise.
# ---------------------------------------------------------------------------


def _parse_header(line: str) -> RetainedHeader | None:
    """Parse a header record, or return None if it is not one."""
    try:
        record = json.loads(line)
    except Exception:
        return None
    if not isinstance(record, dict) or record.get("record") != RECORD_HEADER:
        return None
    contract_version = record.get("contract_version")
    generation = record.get("generation")
    if not isinstance(contract_version, str) or not isinstance(generation, str):
        return None
    raw_names = record.get("names")
    names: tuple[str, ...] = ()
    if isinstance(raw_names, list) and all(isinstance(n, str) for n in raw_names):
        names = tuple(raw_names)
    created_at = record.get("created_at")
    if not isinstance(created_at, int):
        created_at = None
    known = {"record", "contract_version", "generation", "names", "created_at"}
    extra = {k: v for k, v in record.items() if k not in known}
    return RetainedHeader(
        contract_version=contract_version,
        generation=generation,
        names=names,
        created_at=created_at,
        extra=extra,
    )


def _parse_example(line: str) -> Example | None:
    """Parse an example record, or return None if the line is unusable."""
    try:
        record = json.loads(line)
    except Exception:
        return None
    if not isinstance(record, dict) or record.get("record") != RECORD_EXAMPLE:
        return None
    raw_x = record.get("x")
    raw_y = record.get("y")
    if not isinstance(raw_x, list) or isinstance(raw_y, bool) or not isinstance(raw_y, (int, float)):
        return None
    values: list[float] = []
    for value in raw_x:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        as_float = float(value)
        if not math.isfinite(as_float):
            return None
        values.append(as_float)
    y = float(raw_y)
    if not math.isfinite(y):
        return None
    as_of_ms = record.get("as_of_ms")
    if isinstance(as_of_ms, bool) or not isinstance(as_of_ms, int):
        as_of_ms = None
    return Example(x=tuple(values), y=y, as_of_ms=as_of_ms)


def read_retained(name: str, *, directory: Path | str | None = None) -> RetainedSet:
    """Read a retention file. **Never raises.**

    Tolerance, per data-model.md §Retained Training Set:

    * **Missing file, or empty file** — no retained data.
    * **Missing or unparseable header** — the whole set is treated as no
      retained data, because nothing in it can be interpreted without the
      contract stamp and the name ordering.
    * **Truncated final line** — discarded; every earlier example survives. This
      is the ordinary shape of a process killed mid-append.
    * **An unparseable line mid-file** — skipped and counted; the count is
      logged once, not once per line.

    Every one of those returns a :class:`RetainedSet`. The caller inspects
    ``ok``, ``examples`` and ``reason``; it never handles an exception, because
    an unreadable retention file costs personalization, not availability.
    """
    path = _safe_path(name, directory)
    if path is None:
        return RetainedSet(path=Path(f"{name}.jsonl"), reason="path-unresolvable")

    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return RetainedSet(path=path, reason="missing")
    except Exception as exc:  # unreadable for any other reason — permissions, a directory, I/O
        logger.warning("retained: cannot read %s (%s); treating as no retained data", path, exc)
        return RetainedSet(path=path, reason="unreadable")

    if not raw.strip():
        return RetainedSet(path=path, reason="empty")

    try:
        text = raw.decode("utf-8", errors="replace")
        lines = text.split("\n")

        # A file that does not end in a newline has a possibly-truncated final
        # segment. Attempt it anyway: a complete last line without its newline
        # is common and perfectly usable. Only an unparseable tail is discarded.
        trailing_incomplete = not text.endswith("\n") and bool(lines[-1])
        if not lines[-1]:
            lines.pop()

        header: RetainedHeader | None = None
        examples: list[Example] = []
        skipped = 0
        truncated = False

        for index, line in enumerate(lines):
            stripped = line.strip()
            is_final = index == len(lines) - 1
            if not stripped:
                continue
            if header is None:
                header = _parse_header(stripped)
                if header is None:
                    logger.warning(
                        "retained: %s has no parseable header; treating as no retained data",
                        path,
                    )
                    return RetainedSet(path=path, reason="no-header")
                continue
            parsed = _parse_example(stripped)
            if parsed is None:
                if is_final and trailing_incomplete:
                    truncated = True
                else:
                    skipped += 1
                continue
            examples.append(parsed)

        if header is None:
            return RetainedSet(path=path, reason="no-header")
        if skipped:
            logger.warning("retained: skipped %d unparseable line(s) in %s", skipped, path)
        if truncated:
            logger.info("retained: discarded a truncated final line in %s", path)

        return RetainedSet(
            path=path,
            header=header,
            examples=tuple(examples),
            skipped_lines=skipped,
            truncated_final_line=truncated,
        )
    except Exception as exc:  # defence in depth: the reader must not raise, ever
        logger.warning("retained: unexpected failure reading %s (%s); treating as no retained data", path, exc)
        return RetainedSet(path=path, reason="unreadable")


def _safe_path(name: str, directory: Path | str | None) -> Path | None:
    """Resolve a path without raising, for use by the read/summary paths."""
    try:
        return retained_path(name, directory=directory)
    except Exception as exc:
        logger.warning("retained: cannot resolve retention path for %r (%s)", name, exc)
        return None


# ---------------------------------------------------------------------------
# T010 — writer
# ---------------------------------------------------------------------------


def _read_header_only(path: Path) -> RetainedHeader | None:
    """Read just the first line of an existing file. Returns None if unusable."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    return _parse_header(stripped)
        return None
    except Exception:
        return None


def append_examples(
    name: str,
    examples: Iterable[Example],
    contract: FeatureContract | None,
    *,
    directory: Path | str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    generation: str = INITIAL_GENERATION,
    created_at: int | None = None,
) -> AppendResult:
    """Append ``examples`` to ``name``'s retention file, writing the header first if new.

    ``contract`` supplies the stamp: ``service_version`` becomes the header's
    ``contract_version`` and ``names`` is duplicated into the header so the file
    is self-describing when read alone.

    **Contract-version mismatch is refused, not resolved.** If the file already
    has a header whose ``contract_version`` differs from ``contract``'s, this
    returns ``ok=False`` with a reason naming both versions and writes nothing.
    Mixing vectors computed under two contracts into one file would poison every
    model trained from it, and deciding to discard the old set is a policy call
    that belongs to refresh, not to the writer.

    ``generation`` is used only when a header is being written for the first
    time; an existing header's generation is authoritative and is carried
    forward. After a reset, pass :attr:`ResetResult.next_generation` here.

    The size cap is checked after the append. When exceeded, the oldest examples
    are evicted per the policy in the module docstring.
    """
    path = retained_path(name, directory=directory)

    if contract is None or not contract.service_version:
        return AppendResult(
            path=path,
            ok=False,
            reason="no-contract-version: refusing to retain examples that cannot be stamped with a contract version",
        )
    contract_version = contract.service_version
    contract_names = tuple(contract.names)

    batch = list(examples)

    existing = _read_header_only(path) if path.exists() else None
    header_written = False

    if existing is not None:
        if existing.contract_version != contract_version:
            return AppendResult(
                path=path,
                ok=False,
                generation=existing.generation,
                reason=(
                    f"contract-version-mismatch: retention file is stamped {existing.contract_version!r} "
                    f"but the current contract is {contract_version!r}; refusing to append. "
                    "Resetting the retained set is refresh's decision, not the writer's."
                ),
            )
        if existing.names and existing.names != contract_names:
            return AppendResult(
                path=path,
                ok=False,
                generation=existing.generation,
                reason=(
                    f"contract-names-mismatch: retention file declares {list(existing.names)} "
                    f"but the current contract declares {list(contract_names)}; refusing to append."
                ),
            )
        header = existing
    else:
        if path.exists() and path.stat().st_size > 0:
            # The file is there but its header is unreadable, which the reader
            # already treats as no retained data — so nothing usable is lost by
            # starting over, and leaving it in place would wedge accumulation.
            logger.warning("retained: %s has no usable header; starting a fresh set", path)
        header = RetainedHeader(
            contract_version=contract_version,
            generation=generation,
            names=contract_names,
            created_at=created_at if created_at is not None else _now_ms(),
        )
        header_written = True

    try:
        lines = [_example_line(example) for example in batch]
    except (TypeError, ValueError) as exc:
        return AppendResult(
            path=path,
            ok=False,
            generation=header.generation,
            reason=f"unserializable-example: {exc}",
        )

    payload = ""
    if header_written:
        payload += _header_line(header) + "\n"
    payload += "".join(line + "\n" for line in lines)

    if header_written:
        # A fresh header replaces whatever was there; write the whole file
        # atomically so a corrupt predecessor is never partially overwritten.
        _atomic_write_text(path, payload)
    elif payload:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)

    evicted = _enforce_bound(path, max_bytes=max_bytes)

    return AppendResult(
        path=path,
        ok=True,
        written=len(lines),
        evicted=evicted,
        header_written=header_written,
        generation=header.generation,
    )


# ---------------------------------------------------------------------------
# T012 — bound enforcement and eviction
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, payload: str) -> None:
    """Write ``payload`` to ``path`` via a temp file in the same directory.

    ``os.replace`` is atomic within a filesystem, so a reader — or a crash —
    sees either the whole previous file or the whole new one. The temp file
    shares the directory precisely so the replace never crosses a filesystem
    boundary and degrade into a copy.
    """
    tmp = path.with_name(f".{path.name}.tmp")
    encoded = payload.encode("utf-8")
    with tmp.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _enforce_bound(path: Path, *, max_bytes: int) -> int:
    """Evict oldest-first until the file fits ``max_bytes``. Returns examples dropped.

    Operates on the raw encoded lines rather than on parsed examples, so an
    eviction neither re-serializes surviving rows nor silently repairs them —
    what was on disk before is byte-identical to what is on disk after, minus a
    prefix.
    """
    if max_bytes is None or max_bytes <= 0:
        return 0
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    if size <= max_bytes:
        return 0

    try:
        raw = path.read_bytes()
    except OSError as exc:
        logger.warning("retained: cannot read %s to enforce the size bound (%s)", path, exc)
        return 0

    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    if not lines:
        return 0

    header_line = lines[0]
    example_lines = lines[1:]

    budget = max_bytes - (len(header_line) + 1)
    if budget < 0:
        budget = 0

    kept: list[bytes] = []
    used = 0
    for line in reversed(example_lines):
        cost = len(line) + 1
        if used + cost > budget:
            break
        kept.append(line)
        used += cost
    kept.reverse()

    dropped = len(example_lines) - len(kept)
    if dropped <= 0:
        return 0

    payload = b"\n".join([header_line, *kept]) + b"\n"
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("retained: eviction of %s failed (%s); the set is unchanged", path, exc)
        try:
            tmp.unlink()
        except OSError:
            pass
        return 0

    logger.info(
        "retained: %s exceeded %d bytes; evicted %d oldest example(s), %d retained",
        path,
        max_bytes,
        dropped,
        len(kept),
    )
    return dropped


# ---------------------------------------------------------------------------
# T013 — deletion, generation, and the inspection surface
# ---------------------------------------------------------------------------


def next_generation(current: str | None) -> str:
    """Return the generation following ``current``.

    Generations are decimal strings so they round-trip through JSON and the
    manifest unchanged. An unparseable or absent generation restarts the count
    rather than propagating garbage into the next header.
    """
    try:
        return str(int(str(current)) + 1)
    except (TypeError, ValueError):
        return INITIAL_GENERATION


def delete_retained(name: str, *, directory: Path | str | None = None) -> bool:
    """Delete a model's retention file entirely (FR-018). Returns True if one was removed.

    Deletion costs personalization and nothing else: the artifact and its
    manifest are untouched, so the install stays servable and simply has no
    local examples to rebuild from on the next base refresh. Accumulation
    restarts on the next append, which writes a fresh header.
    """
    path = _safe_path(name, directory)
    if path is None:
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("retained: could not delete %s (%s)", path, exc)
        return False
    logger.info("retained: deleted %s", path)
    return True


def reset_retained(
    name: str,
    *,
    contract: FeatureContract | None = None,
    directory: Path | str | None = None,
    created_at: int | None = None,
) -> ResetResult:
    """Discard a retained set and advance its generation (FR-014).

    The generation counter lives in the header and nowhere else — there is no
    sidecar file to fall out of step with the data it describes. So the counter
    is read out before the file goes away and handed back to the caller:

    * With ``contract``, the reset writes a fresh header-only file stamped with
      the new contract version and the incremented generation. The counter is
      durable immediately, and the next append accumulates under it.
    * Without ``contract``, the file is simply removed and
      :attr:`ResetResult.next_generation` must be passed to the next
      :func:`append_examples` call, which writes it into the header it creates.

    Either way the install stays servable; a reset removes examples, not models.
    """
    path = retained_path(name, directory=directory)
    previous = _read_header_only(path)
    previous_generation = previous.generation if previous else None
    following = next_generation(previous_generation)

    deleted = delete_retained(name, directory=directory)

    header_written = False
    if contract is not None and contract.service_version:
        header = RetainedHeader(
            contract_version=contract.service_version,
            generation=following,
            names=tuple(contract.names),
            created_at=created_at if created_at is not None else _now_ms(),
        )
        _atomic_write_text(path, _header_line(header) + "\n")
        header_written = True

    logger.info(
        "retained: reset %s (generation %s -> %s)",
        path,
        previous_generation or "-",
        following,
    )
    return ResetResult(
        path=path,
        previous_generation=previous_generation,
        next_generation=following,
        deleted=deleted,
        header_written=header_written,
    )


def summarize_retained(
    name: str,
    *,
    directory: Path | str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> RetainedSummary:
    """Describe what is retained for ``name`` — size, count, and stamp (FR-018).

    This is the inspection surface a discovery endpoint reads. Like the reader,
    it never raises: an unreadable file reports ``readable=False`` with whatever
    is knowable from the filesystem.
    """
    path = _safe_path(name, directory)
    if path is None:
        return RetainedSummary(name=name, path=Path(f"{name}.jsonl"), exists=False, max_bytes=max_bytes)

    try:
        size = path.stat().st_size
        exists = True
    except OSError:
        size = 0
        exists = False

    retained = read_retained(name, directory=directory)
    return RetainedSummary(
        name=name,
        path=path,
        exists=exists,
        size_bytes=size,
        n_examples=len(retained.examples),
        contract_version=retained.contract_version,
        generation=retained.generation,
        created_at=retained.header.created_at if retained.header else None,
        names=retained.names,
        skipped_lines=retained.skipped_lines,
        max_bytes=max_bytes,
        readable=retained.ok,
    )


def summarize_all(
    names: Sequence[str],
    *,
    directory: Path | str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[RetainedSummary, ...]:
    """Summarize several models at once, for a discovery surface listing the roster."""
    return tuple(summarize_retained(name, directory=directory, max_bytes=max_bytes) for name in names)
