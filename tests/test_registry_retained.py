"""Tests for the retained training set (WP03 — FR-009, FR-010, FR-011, FR-018, NFR-004).

Three properties carry most of the weight here:

* the file is **inspectable** — a header and an example, readable with ``head -2``;
* the reader **never raises**, in any of the four documented damage modes;
* nothing in the module can **open a socket** (FR-011, C-003).
"""

from __future__ import annotations

import ast
import json
import logging
import socket
from pathlib import Path

import pytest

from kenaz_ml.modelstore.registry import FeatureContract
from kenaz_ml.modelstore.registry import retained as retained_module
from kenaz_ml.modelstore.registry.retained import (
    DEFAULT_MAX_BYTES,
    INITIAL_GENERATION,
    RECORD_EXAMPLE,
    RECORD_HEADER,
    Example,
    append_examples,
    delete_retained,
    next_generation,
    read_retained,
    reset_retained,
    retained_path,
    summarize_all,
    summarize_retained,
)

NAMES = (
    "test_failure_count",
    "time_in_phase_sec",
    "edit_velocity",
    "file_switch_rate",
    "session_length_sec",
    "time_since_last_commit_sec",
)

CONTRACT = FeatureContract(
    service="stuck",
    service_version="9f2c4a71b3e05d18",
    names=NAMES,
    dtypes=("float64",) * 6,
)

OTHER_CONTRACT = FeatureContract(
    service="stuck",
    service_version="deadbeefdeadbeef",
    names=NAMES,
    dtypes=("float64",) * 6,
)


def make_example(index: int) -> Example:
    return Example(
        x=(float(index), 1800.0, 2.3, 0.4, 3600.0, 900.0),
        y=float(index % 2),
        as_of_ms=1_753_812_345_678 + index,
    )


def write_raw(path: Path, lines: list[str], *, trailing_newline: bool = True) -> None:
    payload = "\n".join(lines)
    if trailing_newline:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


# ---------------------------------------------------------------------------
# T010 — writer and format
# ---------------------------------------------------------------------------


class TestWriterFormat:
    def test_header_written_once_on_first_append(self, tmp_path: Path) -> None:
        first = append_examples("stuck", [make_example(0)], CONTRACT, directory=tmp_path)
        second = append_examples("stuck", [make_example(1)], CONTRACT, directory=tmp_path)

        assert first.ok and first.header_written
        assert second.ok and not second.header_written

        lines = (tmp_path / "stuck.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert [json.loads(line)["record"] for line in lines] == [
            RECORD_HEADER,
            RECORD_EXAMPLE,
            RECORD_EXAMPLE,
        ]

    def test_head_two_shows_a_header_and_an_example(self, tmp_path: Path) -> None:
        """FR-018: the point of JSONL is that a user can open the file (D-002)."""
        append_examples("stuck", [make_example(0), make_example(1)], CONTRACT, directory=tmp_path)

        head = (tmp_path / "stuck.jsonl").read_text(encoding="utf-8").splitlines()[:2]

        header = json.loads(head[0])
        assert header == {
            "record": "header",
            "contract_version": CONTRACT.service_version,
            "generation": INITIAL_GENERATION,
            "names": list(NAMES),
            "created_at": header["created_at"],
        }
        assert isinstance(header["created_at"], int)

        example = json.loads(head[1])
        assert example == {
            "record": "example",
            "x": [0.0, 1800.0, 2.3, 0.4, 3600.0, 900.0],
            "y": 0.0,
            "as_of_ms": 1_753_812_345_678,
        }

    def test_as_of_ms_is_carried_through_not_recomputed(self, tmp_path: Path) -> None:
        stale = Example(x=(1.0,) * 6, y=1.0, as_of_ms=1)
        append_examples("stuck", [stale], CONTRACT, directory=tmp_path)

        assert read_retained("stuck", directory=tmp_path).examples[0].as_of_ms == 1

    def test_examples_append_one_per_line(self, tmp_path: Path) -> None:
        append_examples("stuck", [make_example(i) for i in range(10)], CONTRACT, directory=tmp_path)

        lines = (tmp_path / "stuck.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 11

    def test_contract_version_mismatch_is_refused(self, tmp_path: Path) -> None:
        append_examples("stuck", [make_example(0)], CONTRACT, directory=tmp_path)
        before = (tmp_path / "stuck.jsonl").read_bytes()

        result = append_examples("stuck", [make_example(1)], OTHER_CONTRACT, directory=tmp_path)

        assert not result.ok
        assert "contract-version-mismatch" in (result.reason or "")
        assert CONTRACT.service_version in (result.reason or "")
        assert OTHER_CONTRACT.service_version in (result.reason or "")
        assert (tmp_path / "stuck.jsonl").read_bytes() == before

    def test_contract_names_mismatch_is_refused(self, tmp_path: Path) -> None:
        append_examples("stuck", [make_example(0)], CONTRACT, directory=tmp_path)
        permuted = FeatureContract(
            service=CONTRACT.service,
            service_version=CONTRACT.service_version,
            names=tuple(reversed(NAMES)),
            dtypes=CONTRACT.dtypes,
        )

        result = append_examples("stuck", [make_example(1)], permuted, directory=tmp_path)

        assert not result.ok
        assert "contract-names-mismatch" in (result.reason or "")

    def test_missing_contract_version_is_refused(self, tmp_path: Path) -> None:
        result = append_examples("stuck", [make_example(0)], FeatureContract(names=NAMES), directory=tmp_path)

        assert not result.ok
        assert "no-contract-version" in (result.reason or "")
        assert not (tmp_path / "stuck.jsonl").exists()

    def test_non_finite_values_are_refused_so_the_file_stays_valid_json(self, tmp_path: Path) -> None:
        bad = Example(x=(float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0), y=0.0, as_of_ms=1)

        result = append_examples("stuck", [bad], CONTRACT, directory=tmp_path)

        assert not result.ok
        assert "unserializable-example" in (result.reason or "")

    def test_module_imports_only_the_standard_library(self) -> None:
        source = Path(retained_module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])

        assert imported <= {
            "__future__",
            "json",
            "logging",
            "math",
            "os",
            "time",
            "collections",
            "dataclasses",
            "pathlib",
            "typing",
            "kenaz_ml",
        }, imported


# ---------------------------------------------------------------------------
# T011 — the reader must never raise
# ---------------------------------------------------------------------------


class TestTolerantReader:
    def test_round_trip(self, tmp_path: Path) -> None:
        written = [make_example(i) for i in range(25)]
        append_examples("stuck", written, CONTRACT, directory=tmp_path)

        result = read_retained("stuck", directory=tmp_path)

        assert result.ok
        assert result.examples == tuple(written)
        assert result.contract_version == CONTRACT.service_version
        assert result.generation == INITIAL_GENERATION
        assert result.names == NAMES
        assert result.skipped_lines == 0
        assert not result.truncated_final_line
        assert len(result) == 25

    def test_missing_file_is_no_retained_data(self, tmp_path: Path) -> None:
        result = read_retained("never-written", directory=tmp_path)

        assert not result.ok
        assert result.examples == ()
        assert result.reason == "missing"

    # --- tolerance case 1: missing / unparseable header ---------------------

    def test_missing_header_is_no_retained_data(self, tmp_path: Path) -> None:
        path = retained_path("stuck", directory=tmp_path)
        write_raw(path, [json.dumps({"record": "example", "x": [1.0], "y": 1.0, "as_of_ms": 1})] * 3)

        result = read_retained("stuck", directory=tmp_path)

        assert not result.ok
        assert result.examples == ()
        assert result.reason == "no-header"

    def test_unparseable_header_is_no_retained_data(self, tmp_path: Path) -> None:
        path = retained_path("stuck", directory=tmp_path)
        write_raw(
            path,
            ['{"record":"header","contract_ver'] + [json.dumps({"record": "example", "x": [1.0], "y": 1.0})] * 3,
        )

        result = read_retained("stuck", directory=tmp_path)

        assert not result.ok
        assert result.examples == ()
        assert result.reason == "no-header"

    def test_header_without_a_generation_is_rejected(self, tmp_path: Path) -> None:
        path = retained_path("stuck", directory=tmp_path)
        write_raw(path, [json.dumps({"record": "header", "contract_version": "v", "names": list(NAMES)})])

        assert read_retained("stuck", directory=tmp_path).reason == "no-header"

    # --- tolerance case 2: truncated final line -----------------------------

    def test_truncated_final_line_is_discarded_and_the_rest_survives(self, tmp_path: Path) -> None:
        append_examples("stuck", [make_example(i) for i in range(5)], CONTRACT, directory=tmp_path)
        path = retained_path("stuck", directory=tmp_path)

        raw = path.read_bytes()
        # Simulate a process killed mid-append: a partial sixth example.
        path.write_bytes(raw + b'{"record":"example","x":[9.0,1800.0,2.3')

        result = read_retained("stuck", directory=tmp_path)

        assert result.ok
        assert result.truncated_final_line
        assert result.skipped_lines == 0
        assert [e.x[0] for e in result.examples] == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_complete_final_line_without_a_newline_is_kept(self, tmp_path: Path) -> None:
        append_examples("stuck", [make_example(i) for i in range(3)], CONTRACT, directory=tmp_path)
        path = retained_path("stuck", directory=tmp_path)
        path.write_bytes(path.read_bytes().rstrip(b"\n"))

        result = read_retained("stuck", directory=tmp_path)

        assert result.ok
        assert not result.truncated_final_line
        assert len(result.examples) == 3

    # --- tolerance case 3: empty file ---------------------------------------

    def test_empty_file_is_no_retained_data(self, tmp_path: Path) -> None:
        retained_path("stuck", directory=tmp_path).write_bytes(b"")

        result = read_retained("stuck", directory=tmp_path)

        assert not result.ok
        assert result.examples == ()
        assert result.reason == "empty"

    def test_whitespace_only_file_is_no_retained_data(self, tmp_path: Path) -> None:
        retained_path("stuck", directory=tmp_path).write_text("\n\n  \n", encoding="utf-8")

        assert read_retained("stuck", directory=tmp_path).reason == "empty"

    # --- tolerance case 4: corruption mid-file ------------------------------

    def test_unparseable_mid_file_lines_are_skipped_counted_and_logged_once(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        append_examples("stuck", [make_example(i) for i in range(6)], CONTRACT, directory=tmp_path)
        path = retained_path("stuck", directory=tmp_path)
        lines = path.read_text(encoding="utf-8").splitlines()
        lines.insert(3, "}{ not json at all")
        lines.insert(5, '{"record":"example","x":"not-a-vector","y":1.0}')
        write_raw(path, lines)

        with caplog.at_level(logging.WARNING, logger=retained_module.__name__):
            result = read_retained("stuck", directory=tmp_path)

        assert result.ok
        assert result.skipped_lines == 2
        assert len(result.examples) == 6
        skip_logs = [r for r in caplog.records if "skipped" in r.getMessage()]
        assert len(skip_logs) == 1
        assert "skipped 2 unparseable line(s)" in skip_logs[0].getMessage()

    # --- the reader must not raise, whatever it is handed -------------------

    @pytest.mark.parametrize(
        "payload",
        [
            b"\x00\x01\x02\xff\xfe",
            b"[1, 2, 3]\n",
            b"null\n",
            b'{"record":"header"}\n',
            b'{"record":"header","contract_version":1,"generation":2}\n',
            b'{"record":"header","contract_version":"v","generation":"1"}\n' + b"\x80\x81\n",
        ],
    )
    def test_reader_never_raises_on_damaged_input(self, tmp_path: Path, payload: bytes) -> None:
        retained_path("stuck", directory=tmp_path).write_bytes(payload)

        result = read_retained("stuck", directory=tmp_path)

        assert result.examples == () or all(isinstance(e, Example) for e in result.examples)

    def test_reader_never_raises_when_the_path_is_a_directory(self, tmp_path: Path) -> None:
        (tmp_path / "stuck.jsonl").mkdir()

        result = read_retained("stuck", directory=tmp_path)

        assert not result.ok
        assert result.reason == "unreadable"

    def test_reader_never_raises_when_read_bytes_explodes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        append_examples("stuck", [make_example(0)], CONTRACT, directory=tmp_path)

        def boom(self: Path) -> bytes:
            raise OSError("disk fell over")

        monkeypatch.setattr(Path, "read_bytes", boom)

        result = read_retained("stuck", directory=tmp_path)

        assert not result.ok
        assert result.reason == "unreadable"


# ---------------------------------------------------------------------------
# T012 — bound enforcement and eviction
# ---------------------------------------------------------------------------


class TestBoundAndEviction:
    def test_default_cap_is_fifty_megabytes(self) -> None:
        assert DEFAULT_MAX_BYTES == 50 * 1024 * 1024

    def test_no_eviction_below_the_cap(self, tmp_path: Path) -> None:
        result = append_examples(
            "stuck", [make_example(i) for i in range(20)], CONTRACT, directory=tmp_path, max_bytes=DEFAULT_MAX_BYTES
        )

        assert result.evicted == 0
        assert len(read_retained("stuck", directory=tmp_path).examples) == 20

    def test_eviction_keeps_the_newest_and_preserves_the_header(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        append_examples("stuck", [make_example(i) for i in range(40)], CONTRACT, directory=tmp_path)
        path = retained_path("stuck", directory=tmp_path)

        lines = path.read_text(encoding="utf-8").splitlines()
        header_bytes = len(lines[0].encode()) + 1
        widest_example = max(len(line.encode()) for line in lines[1:]) + 1
        cap = header_bytes + widest_example * 10

        with caplog.at_level(logging.INFO, logger=retained_module.__name__):
            result = append_examples("stuck", [make_example(40)], CONTRACT, directory=tmp_path, max_bytes=cap)

        assert result.ok

        after = read_retained("stuck", directory=tmp_path)
        assert after.ok, "the header must survive an eviction"
        assert after.contract_version == CONTRACT.service_version
        assert after.generation == INITIAL_GENERATION
        assert after.names == NAMES
        assert path.stat().st_size <= cap

        # Oldest-first: the survivors are the newest contiguous suffix, nothing else.
        kept = len(after.examples)
        assert 9 <= kept <= 10, "the cap admits about ten examples"
        assert [e.x[0] for e in after.examples] == [float(i) for i in range(41 - kept, 41)]
        assert result.evicted == 41 - kept

        evict_logs = [r for r in caplog.records if "evicted" in r.getMessage()]
        assert len(evict_logs) == 1
        assert f"evicted {41 - kept} oldest example(s)" in evict_logs[0].getMessage()

    def test_surviving_lines_are_byte_identical(self, tmp_path: Path) -> None:
        append_examples("stuck", [make_example(i) for i in range(30)], CONTRACT, directory=tmp_path)
        path = retained_path("stuck", directory=tmp_path)
        before = path.read_text(encoding="utf-8").splitlines()
        cap = len(before[0].encode()) + 1 + (max(len(line.encode()) for line in before[1:]) + 1) * 5

        append_examples("stuck", [], CONTRACT, directory=tmp_path, max_bytes=cap)

        after = path.read_text(encoding="utf-8").splitlines()
        assert after[0] == before[0]
        assert after[1:] == before[-len(after) + 1 :]

    def test_an_example_larger_than_the_cap_leaves_a_header_only_file(self, tmp_path: Path) -> None:
        append_examples("stuck", [make_example(i) for i in range(5)], CONTRACT, directory=tmp_path)

        append_examples("stuck", [], CONTRACT, directory=tmp_path, max_bytes=1)

        result = read_retained("stuck", directory=tmp_path)
        assert result.ok, "the header is preserved even when nothing fits"
        assert result.examples == ()

    def test_eviction_is_atomic_via_a_temp_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An interrupted eviction leaves the pre-eviction file, never a truncated one."""
        append_examples("stuck", [make_example(i) for i in range(30)], CONTRACT, directory=tmp_path)
        path = retained_path("stuck", directory=tmp_path)
        before = path.read_bytes()
        cap = 500

        seen: list[tuple[str, str]] = []

        def interrupted(src: object, dst: object) -> None:
            seen.append((str(src), str(dst)))
            raise OSError("power cut between write and replace")

        monkeypatch.setattr(retained_module.os, "replace", interrupted)

        result = append_examples("stuck", [make_example(30)], CONTRACT, directory=tmp_path, max_bytes=cap)

        assert result.ok
        assert result.evicted == 0, "a failed eviction drops nothing"
        assert seen, "the rewrite went through a temp file, not the live file"
        assert seen[0][0].endswith(".stuck.jsonl.tmp")
        assert seen[0][1].endswith("stuck.jsonl")

        # The live file is exactly what it was, plus the appended example — the
        # eviction could not have destroyed it because it never wrote to it.
        assert path.read_bytes().startswith(before)
        survived = read_retained("stuck", directory=tmp_path)
        assert survived.ok
        assert len(survived.examples) == 31
        assert not list(tmp_path.glob(".*.tmp")), "the temp file is cleaned up"

    def test_no_stray_temp_file_remains_after_a_successful_eviction(self, tmp_path: Path) -> None:
        append_examples("stuck", [make_example(i) for i in range(30)], CONTRACT, directory=tmp_path)

        append_examples("stuck", [make_example(30)], CONTRACT, directory=tmp_path, max_bytes=600)

        assert [p.name for p in tmp_path.iterdir()] == ["stuck.jsonl"]

    def test_a_nonpositive_cap_disables_the_bound(self, tmp_path: Path) -> None:
        result = append_examples(
            "stuck", [make_example(i) for i in range(5)], CONTRACT, directory=tmp_path, max_bytes=0
        )

        assert result.evicted == 0
        assert len(read_retained("stuck", directory=tmp_path).examples) == 5


# ---------------------------------------------------------------------------
# T013 — deletion, generation, summary
# ---------------------------------------------------------------------------


class TestDeletionAndGeneration:
    def test_deletion_removes_the_file_entirely(self, tmp_path: Path) -> None:
        append_examples("stuck", [make_example(0)], CONTRACT, directory=tmp_path)
        path = retained_path("stuck", directory=tmp_path)

        assert delete_retained("stuck", directory=tmp_path) is True
        assert not path.exists()

    def test_deleting_nothing_is_not_an_error(self, tmp_path: Path) -> None:
        assert delete_retained("stuck", directory=tmp_path) is False

    def test_deletion_leaves_the_install_servable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """FR-018: deleting retained data costs personalization and nothing else."""
        models = tmp_path / "ml-models"
        models.mkdir()
        artifact = models / "stuck.joblib"
        manifest = models / "stuck.json"
        artifact.write_bytes(b"pretend-artifact")
        manifest.write_text('{"name":"stuck"}', encoding="utf-8")

        append_examples("stuck", [make_example(0)], CONTRACT, directory=models / "retained")
        delete_retained("stuck", directory=models / "retained")

        assert artifact.read_bytes() == b"pretend-artifact"
        assert manifest.exists()
        assert read_retained("stuck", directory=models / "retained").examples == ()

    def test_accumulation_restarts_after_deletion(self, tmp_path: Path) -> None:
        append_examples("stuck", [make_example(i) for i in range(4)], CONTRACT, directory=tmp_path)
        delete_retained("stuck", directory=tmp_path)

        result = append_examples("stuck", [make_example(99)], CONTRACT, directory=tmp_path)

        assert result.ok
        assert result.header_written
        after = read_retained("stuck", directory=tmp_path)
        assert after.ok
        assert [e.x[0] for e in after.examples] == [99.0]

    def test_next_generation_counts_and_tolerates_garbage(self) -> None:
        assert next_generation("1") == "2"
        assert next_generation("41") == "42"
        assert next_generation(None) == INITIAL_GENERATION
        assert next_generation("not-a-number") == INITIAL_GENERATION

    def test_generation_increments_across_a_reset_with_a_new_contract(self, tmp_path: Path) -> None:
        append_examples("stuck", [make_example(i) for i in range(3)], CONTRACT, directory=tmp_path)
        assert read_retained("stuck", directory=tmp_path).generation == "1"

        reset = reset_retained("stuck", contract=OTHER_CONTRACT, directory=tmp_path)

        assert reset.previous_generation == "1"
        assert reset.next_generation == "2"
        assert reset.header_written

        # The counter is durable in the header alone — no sidecar file.
        assert [p.name for p in tmp_path.iterdir()] == ["stuck.jsonl"]
        after_reset = read_retained("stuck", directory=tmp_path)
        assert after_reset.ok
        assert after_reset.generation == "2"
        assert after_reset.contract_version == OTHER_CONTRACT.service_version
        assert after_reset.examples == ()

        append_examples("stuck", [make_example(7)], OTHER_CONTRACT, directory=tmp_path)
        rebuilt = read_retained("stuck", directory=tmp_path)
        assert rebuilt.generation == "2"
        assert [e.x[0] for e in rebuilt.examples] == [7.0]

        third = reset_retained("stuck", contract=OTHER_CONTRACT, directory=tmp_path)
        assert third.next_generation == "3"
        assert read_retained("stuck", directory=tmp_path).generation == "3"

    def test_generation_increments_across_a_reset_without_a_contract(self, tmp_path: Path) -> None:
        append_examples("stuck", [make_example(0)], CONTRACT, directory=tmp_path)

        reset = reset_retained("stuck", directory=tmp_path)

        assert reset.deleted
        assert not reset.header_written
        assert not retained_path("stuck", directory=tmp_path).exists()

        append_examples(
            "stuck", [make_example(1)], OTHER_CONTRACT, directory=tmp_path, generation=reset.next_generation
        )
        assert read_retained("stuck", directory=tmp_path).generation == "2"

    def test_reset_of_a_never_written_set_starts_at_one(self, tmp_path: Path) -> None:
        reset = reset_retained("stuck", directory=tmp_path)

        assert reset.previous_generation is None
        assert reset.next_generation == INITIAL_GENERATION

    def test_an_unreadable_header_is_replaced_rather_than_wedging_accumulation(self, tmp_path: Path) -> None:
        retained_path("stuck", directory=tmp_path).write_text("garbage\ngarbage\n", encoding="utf-8")

        result = append_examples("stuck", [make_example(0)], CONTRACT, directory=tmp_path)

        assert result.ok and result.header_written
        after = read_retained("stuck", directory=tmp_path)
        assert after.ok
        assert len(after.examples) == 1


class TestSummary:
    def test_summary_reports_size_count_and_stamp(self, tmp_path: Path) -> None:
        append_examples("stuck", [make_example(i) for i in range(12)], CONTRACT, directory=tmp_path)

        summary = summarize_retained("stuck", directory=tmp_path)

        assert summary.name == "stuck"
        assert summary.exists and summary.readable
        assert summary.n_examples == 12
        assert summary.size_bytes == retained_path("stuck", directory=tmp_path).stat().st_size
        assert summary.contract_version == CONTRACT.service_version
        assert summary.generation == INITIAL_GENERATION
        assert summary.names == NAMES
        assert summary.max_bytes == DEFAULT_MAX_BYTES

    def test_summary_of_a_missing_set(self, tmp_path: Path) -> None:
        summary = summarize_retained("duration", directory=tmp_path)

        assert not summary.exists
        assert not summary.readable
        assert summary.n_examples == 0
        assert summary.size_bytes == 0

    def test_summarize_all_covers_the_roster(self, tmp_path: Path) -> None:
        append_examples("stuck", [make_example(0)], CONTRACT, directory=tmp_path)

        summaries = summarize_all(["stuck", "duration"], directory=tmp_path)

        assert [s.name for s in summaries] == ["stuck", "duration"]
        assert summaries[0].readable
        assert not summaries[1].readable


# ---------------------------------------------------------------------------
# FR-011 / C-003 — retained data never leaves the machine
# ---------------------------------------------------------------------------

_NETWORK_MODULES = {
    "aiohttp",
    "asyncio",
    "boto3",
    "botocore",
    "fastapi",
    "ftplib",
    "http",
    "httpx",
    "imaplib",
    "poplib",
    "requests",
    "smtplib",
    "socket",
    "socketserver",
    "ssl",
    "telnetlib",
    "urllib",
    "urllib3",
    "uvicorn",
    "webbrowser",
    "xmlrpc",
}


class TestNoEgress:
    def test_module_imports_no_network_capable_module(self) -> None:
        source = Path(retained_module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        offenders: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders |= {a.name.split(".")[0] for a in node.names} & _NETWORK_MODULES
            elif isinstance(node, ast.ImportFrom) and node.module:
                offenders |= {node.module.split(".")[0]} & _NETWORK_MODULES

        assert offenders == set(), f"retained.py must not import network-capable modules: {sorted(offenders)}"

    def test_source_mentions_no_network_primitive(self) -> None:
        source = Path(retained_module.__file__).read_text(encoding="utf-8")
        # Exclude this module's own prose about not opening sockets.
        code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("#", "*")))
        for needle in ("socket(", "urlopen", "connect(", "sendall", "http://", "https://"):
            assert needle not in code, f"retained.py must not reference {needle!r}"

    def test_the_whole_api_runs_with_sockets_disabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exercise every public entry point with socket creation booby-trapped."""

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("retained.py opened a socket — FR-011/C-003 forbid it")

        monkeypatch.setattr(socket, "socket", forbidden)
        monkeypatch.setattr(socket, "create_connection", forbidden)
        monkeypatch.setattr(socket, "socketpair", forbidden)

        append_examples("stuck", [make_example(i) for i in range(5)], CONTRACT, directory=tmp_path)
        append_examples("stuck", [make_example(5)], CONTRACT, directory=tmp_path, max_bytes=400)
        read_retained("stuck", directory=tmp_path)
        summarize_retained("stuck", directory=tmp_path)
        summarize_all(["stuck"], directory=tmp_path)
        reset_retained("stuck", contract=CONTRACT, directory=tmp_path)
        delete_retained("stuck", directory=tmp_path)
        retained_path("stuck", directory=tmp_path)

        assert read_retained("stuck", directory=tmp_path).examples == ()


# ---------------------------------------------------------------------------
# Directory resolution — must not depend on when config.retained_data_dir lands
# ---------------------------------------------------------------------------


class TestDirectoryResolution:
    def test_prefers_config_retained_data_dir_when_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kenaz_ml import config

        target = tmp_path / "configured"
        monkeypatch.setattr(config, "retained_data_dir", lambda: target, raising=False)

        assert retained_path("stuck") == target / "stuck.jsonl"
        assert target.is_dir()

    def test_falls_back_to_models_dir_retained(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from kenaz_ml import config

        monkeypatch.delattr(config, "retained_data_dir", raising=False)
        monkeypatch.setattr(config, "models_dir", lambda: tmp_path)

        assert retained_path("stuck") == tmp_path / "retained" / "stuck.jsonl"
        assert (tmp_path / "retained").is_dir()
