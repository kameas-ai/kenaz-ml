"""The kenaz-ml rebrand: what was renamed, and what must never be.

This module guards a rename whose only failure mode is silent. `kenaz-ml` is
the ML service for `sigild`, a **separate Go daemon** that owns
``~/.local/share/sigild/data.db``. A blanket ``sigil`` -> ``kenaz`` substitution
would rewrite that path, and the result would **import perfectly** and fail only
when a real install went looking for a database that does not exist.

So the tests here come in two halves:

* **Positive** -- the rename happened. ``kenaz_ml`` imports, ``sigil_ml`` does
  not, the distribution metadata says ``kenaz-ml``, and no ``sigil_ml`` or
  ``kameas-ml`` survives in the source tree except the documented exceptions.
* **Negative** -- the daemon was not touched. The ledger surface is intact by
  count, the load-bearing paths are intact exactly, and ``db_path()`` still
  resolves to sigild's database *at runtime*, because text can be present and
  still wrong.
"""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
from pathlib import Path

import pytest

from kenaz_ml import config

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Paths this mission must not rewrite, excluded from every measurement below.
#:
#: ``kitty-specs/`` holds already-merged missions' specifications. They record
#: what was true when they were written; rewriting them would make the audit
#: trail lie -- a reader would find a mission describing a move of
#: ``kenaz_ml/store.py``, a path that never existed under that name (D-003,
#: FR-013). ``.worktrees/`` is an untracked husk of a merged lane.
#:
#: **This file excludes itself, and that is not self-serving.** It necessarily
#: quotes every token it forbids -- ``sigil_ml``, ``kameas-ml``, ``KENAZD_`` --
#: in order to search for them. Counting itself would make every figure below
#: shift whenever this file is edited, and would have this test fail on its own
#: pattern strings. Revision 1 of the occurrence map made exactly this mistake,
#: counting its own 45 self-references, and was rejected for it.
EXCLUDED = (":!kitty-specs", ":!.worktrees", ":!tests/test_rebrand.py")


def _git_grep(pattern: str, *paths: str, ignore_case: bool = False) -> list[str]:
    """Return matching occurrences, one per match (``-o``), or skip if git can't."""
    flags = "-oPi" if ignore_case else "-oP"
    try:
        completed = subprocess.run(
            ["git", "grep", flags, pattern, "--", *(paths or ()), *EXCLUDED],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except OSError as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"git is unavailable: {exc}")
    if completed.returncode not in (0, 1):  # 1 == no matches, which is a real answer
        pytest.skip(f"git grep failed, likely not a work tree: {completed.stderr.strip()}")
    return completed.stdout.splitlines()


def _files(pattern: str, *paths: str, ignore_case: bool = False) -> set[str]:
    flags = "-lPi" if ignore_case else "-lP"
    try:
        completed = subprocess.run(
            ["git", "grep", flags, pattern, "--", *(paths or ()), *EXCLUDED],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except OSError as exc:  # pragma: no cover
        pytest.skip(f"git is unavailable: {exc}")
    if completed.returncode not in (0, 1):
        pytest.skip(f"git grep failed, likely not a work tree: {completed.stderr.strip()}")
    return set(completed.stdout.split())


# ===========================================================================
# The rename happened  (FR-001 .. FR-006)
# ===========================================================================


def test_the_package_imports_under_its_new_name() -> None:
    import kenaz_ml

    assert kenaz_ml.__name__ == "kenaz_ml"


def test_the_old_package_name_is_gone() -> None:
    """Not merely renamed on disk -- genuinely unimportable.

    A compatibility shim at the old path would make this pass while leaving the
    rename half-done, so assert the import raises rather than that a directory
    is absent.
    """
    with pytest.raises(ModuleNotFoundError):
        __import__("sigil_ml")


def test_the_distribution_is_named_kenaz_ml() -> None:
    assert importlib.metadata.distribution("kenaz-ml").metadata["Name"] == "kenaz-ml"


def test_distribution_urls_resolve_to_the_real_repository() -> None:
    """FR-004. These were stale *before* the rebrand: package metadata pointed
    at ``kameas-ai/kameas-ml``, a repository that no longer exists under that
    name. The organisation ``kameas-ai`` is correct and unchanged (D-007)."""
    urls = dict(
        entry.split(", ", 1) for entry in importlib.metadata.distribution("kenaz-ml").metadata.get_all("Project-URL")
    )
    assert urls["Homepage"] == "https://github.com/kameas-ai/kenaz-ml"
    assert urls["Repository"] == "https://github.com/kameas-ai/kenaz-ml"
    assert urls["Issues"] == "https://github.com/kameas-ai/kenaz-ml/issues"
    # The daemon's own repository, which is a different product.
    assert urls["Sigil Daemon"] == "https://github.com/kameas-ai/sigil"


def test_the_console_script_points_at_the_renamed_entry_point() -> None:
    """FR-003. Resolved by name by the installer, so a stale half fails only on
    invocation."""
    scripts = {ep.name: ep.value for ep in importlib.metadata.distribution("kenaz-ml").entry_points}
    assert scripts["kenaz-ml"] == "kenaz_ml.cli:main"
    assert "kameas-ml" not in scripts


#: The only places the old package name may still appear, and why.
#:
#: Both are in ``test_migration_regression.py`` and both concern text frozen in
#: git history, not code on disk (FR-012, D-004):
#:
#: * ``src/sigil_ml/features.py`` is an argument to ``git show <sha>:<path>``.
#:   It names a path *in commit ef67e05*, whose content's SHA256 is asserted
#:   before it is exec'd. ``test_serving_regression.py`` documents the same
#:   command for commit 5db84f7.
#: * ``sys.modules["sigil_ml.store"]`` is an alias KEY that must match an import
#:   statement inside that frozen source. Only its value follows the rename.
ALLOWED_OLD_NAME_PATTERNS = (
    r"src/sigil_ml/features\.py",
    r"sigil_ml\.store",
)

#: The FROZEN ARTIFACT is named ``kameas-ml``, and that is not a leftover.
#:
#: This is the second member of the class of names this mission had to be
#: careful about, and the original sweep found only the first. ``sigild`` was
#: correctly identified as a live contract wearing an old-looking name and
#: carved out. The name of the frozen executable is the same kind of thing, and
#: it was not: SC-008 renamed it to ``kenaz-ml`` along with the branding.
#:
#: The consequence was invisible from inside this repository. The freeze still
#: built, ``freeze-smoke`` still passed, and CI stayed green — because the
#: rename was internally consistent. It broke in **kenaz**, whose release build
#: stages the artifact by name. Eleven version tags between 2026-08-02 and
#: 2026-08-06 published no release at all before anyone noticed.
#:
#: The name is fixed by three ratified consumers:
#:   * workspace spec 069 LD-3 / FR-001 / FR-002 / SC-003 (Kenaz 1.0 GA);
#:   * kenaz ``main.go::resolveMLBinary()`` — probes ``kameas-ml/kameas-ml``,
#:     ``../Resources/kameas-ml/kameas-ml``, then PATH ``kameas-ml``;
#:   * kenaz's ``Makefile`` (stage-ml / sign-macos / verify-complete) and
#:     ``release.yml`` AppImage payload.
#:
#: Renaming it is therefore a coordinated cross-repo change, not a tidy-up.
ARTIFACT_NAME = "kameas-ml"

#: The freeze recipe names the artifact, and its header explains at length why.
#: Sweeping it for ``kameas-ml`` would forbid the very thing it must say.
ARTIFACT_EXEMPT_PATHS = ("freeze/kenaz-ml.spec",)

#: Everywhere else, the artifact may only appear PATH-SHAPED — a reference to
#: the freeze output, never a stray brand name. A bare ``kameas-ml`` in prose
#: outside the freeze recipe is still an offence.
ALLOWED_ARTIFACT_PATTERNS = (
    rf"dist/{ARTIFACT_NAME}(?:/{ARTIFACT_NAME})?",
    rf"\$MOUNT/{ARTIFACT_NAME}",
)


def _strip_allowed(line: str) -> str:
    for pattern in (*ALLOWED_OLD_NAME_PATTERNS, *ALLOWED_ARTIFACT_PATTERNS):
        line = re.sub(pattern, "", line)
    return line


@pytest.mark.parametrize("area", ["src", "tests", "freeze", "scripts", "docs", "Makefile", "pyproject.toml"])
def test_no_old_name_survives(area: str) -> None:
    """SC-008 -- neither ``sigil_ml`` nor ``kameas-ml`` remains, bar the exceptions.

    Matched over whole lines (``-n``), not bare tokens (``-o``): the exceptions
    are only recognisable in context, since ``sigil_ml`` on its own is
    indistinguishable from ``sigil_ml`` inside ``src/sigil_ml/features.py``.

    **``sigil_ml`` is matched CASE-SENSITIVELY, and that is not a detail.**
    Case-insensitively it also catches ``SIGIL_ML_MODE`` and its family --
    environment variables, not the package. FR-014 renamed those to
    ``KENAZ_ML_*`` but kept a deprecation shim, so the old spellings *must*
    still appear: in the mapping table, in the warning text, and in the tests
    that prove the fallback works. A case-insensitive assertion here would
    demand deleting the very code that makes the rename survivable.
    ``kameas`` never named an environment variable of its own, so it is matched
    case-insensitively to also catch ``KAMEAS_ML_FROZEN_BIN``.

    **``kameas-ml`` is no longer forbidden outright.** It is the name of the
    frozen artifact — a cross-repo interface, see ``ARTIFACT_NAME`` above — so
    it is permitted path-shaped, and permitted freely inside the freeze recipe
    that defines it. Everywhere else it is still an offence.
    """
    pattern = r"sigil_ml|(?i:kameas[-_]ml)"
    try:
        completed = subprocess.run(
            ["git", "grep", "-nP", pattern, "--", area, *EXCLUDED],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except OSError as exc:  # pragma: no cover
        pytest.skip(f"git is unavailable: {exc}")
    if completed.returncode not in (0, 1):
        pytest.skip(f"git grep failed: {completed.stderr.strip()}")

    offences = [
        line
        for line in completed.stdout.splitlines()
        if not line.startswith(ARTIFACT_EXEMPT_PATHS) and re.search(pattern, _strip_allowed(line))
    ]
    assert offences == [], "old names survive:\n  " + "\n  ".join(offences)


def test_the_documented_exceptions_are_still_present() -> None:
    """The inverse of the test above: if someone "tidies" the two frozen-history
    references away, that is also a defect, and a much subtler one -- the suite
    would fail somewhere unrelated with ModuleNotFoundError."""
    text = (REPO_ROOT / "tests" / "test_migration_regression.py").read_text(encoding="utf-8")
    assert 'sys.modules["sigil_ml.store"]' in text, (
        "the frozen-history alias KEY was renamed; it must match text inside commit ef67e05"
    )
    assert "src/sigil_ml/features.py" in text, "the git-object path was renamed; it names a path in history"


# ===========================================================================
# The frozen artifact keeps the name kenaz resolves it by  (spec 069 LD-3)
# ===========================================================================
#
# These are the inverse of the sweep above, and they exist because the sweep
# alone caused an outage. Forbidding a name is only half a policy; the other
# half is asserting the name that must be there. Without these, the next
# consistent-looking rename of the freeze recipe passes every test in this
# repository and breaks kenaz's release train again.

FREEZE_SPEC = REPO_ROOT / "freeze" / "kenaz-ml.spec"


def test_the_frozen_artifact_is_named_for_the_kenaz_contract() -> None:
    """Both PyInstaller output names are the artifact name, via one constant.

    ``EXE(name=...)`` sets the bootloader's filename and ``COLLECT(name=...)``
    the onedir directory's; kenaz needs ``dist/kameas-ml/kameas-ml``, so both
    must be this name and they must not be able to drift apart.
    """
    source = FREEZE_SPEC.read_text(encoding="utf-8")

    assert f'ARTIFACT_NAME = "{ARTIFACT_NAME}"' in source, (
        f"the freeze recipe no longer defines ARTIFACT_NAME = {ARTIFACT_NAME!r}; kenaz "
        "resolves the sidecar by that name (spec 069 LD-3, resolveMLBinary())"
    )

    names = re.findall(r"^\s{4}name=(.+),$", source, flags=re.MULTILINE)
    assert names == ["ARTIFACT_NAME", "ARTIFACT_NAME"], (
        f"expected EXE(name=ARTIFACT_NAME) and COLLECT(name=ARTIFACT_NAME), found {names}. "
        "Hard-coding either one lets the exe and its onedir directory drift apart."
    )


def test_the_build_paths_agree_with_the_artifact_name() -> None:
    """A correct freeze that nothing can find is still a broken release."""
    expected = f"dist/{ARTIFACT_NAME}/{ARTIFACT_NAME}"

    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert expected in makefile, f"the Makefile's freeze-smoke target does not point at {expected}"

    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert expected in ci, f"CI does not point at {expected}"
    assert "dist/kenaz-ml" not in ci, "CI still refers to dist/kenaz-ml — the freeze does not produce that path"


def test_the_reason_the_artifact_keeps_its_name_is_recorded_where_it_is_set() -> None:
    """The rationale has to live next to the constant, not only in this test.

    Whoever next runs a rename sweep will open the freeze recipe, not the test
    suite. The 2026-08-02 regression was a careful mission that simply had no
    way of knowing this name was load-bearing.
    """
    header = FREEZE_SPEC.read_text(encoding="utf-8")[:4000]
    assert "DO NOT REBRAND" in header, "the freeze recipe lost the warning that explains its artifact name"
    assert "resolveMLBinary" in header, "the freeze recipe no longer names kenaz's consumer of this artifact"


# ===========================================================================
# The Sigil daemon was NOT touched  (FR-007, C-002)  -- T010 invariant 1
# ===========================================================================

#: The ledger surface measured on the pre-rename tree (commit b86242b), using
#: ``git grep -oPi 'sigil(?!_ml\b)(?!-ml\b)(?!_[A-Z])'`` excluding kitty-specs/
#: and .worktrees/. WP01 rev 3. Do not "correct" these to 138, 242 or 524 --
#: all three are superseded figures from earlier revisions of the occurrence map.
#:
#: The ``(?!_[A-Z])`` lookahead is load-bearing. Without it this product's own
#: ``SIGIL_*`` configuration keys -- which FR-014 renamed to ``KENAZ_*`` -- get
#: counted as though they were the daemon's, and the figure inflates by ~65.
LEDGER_PATTERN = r"sigil(?!_ml\b)(?!-ml\b)(?!_[A-Z])"
LEDGER_BASELINE_OCCURRENCES = 163
LEDGER_BASELINE_FILES = 39
SIGILD_BASELINE_OCCURRENCES = 95
SIGILD_BASELINE_FILES = 32


def test_the_ledger_surface_did_not_shrink() -> None:
    """FR-007 -- no reference to the daemon was rewritten into this product's name.

    This is a FLOOR, not an equality, and the reason is recorded here rather
    than hidden in a loosened assertion.

    The baseline is 163 occurrences across 39 files. The current figure is
    higher because FR-014 -- mandated by this same mission -- added the
    ``SIGIL_* -> KENAZ_*`` deprecation shim, and that shim's implementation,
    its docstring and its tests necessarily *name the daemon* in order to say
    which variables are deliberately excluded from the rename
    (``SIGILD_PLUGIN_URL`` belongs to sigild). Three files account for the whole
    increase: ``config.py``, ``test_env_deprecation_shim.py`` and
    ``test_registry_guarantees.py``.

    A per-file check confirmed, at implementation time, that **no file lost a
    single occurrence**. The direction that matters is downward, and the tests
    below pin the load-bearing tokens to exact equality.
    """
    occurrences = len(_git_grep(LEDGER_PATTERN, ignore_case=True))
    files = len(_files(LEDGER_PATTERN, ignore_case=True))
    assert occurrences >= LEDGER_BASELINE_OCCURRENCES, (
        f"the ledger surface shrank from {LEDGER_BASELINE_OCCURRENCES} to {occurrences}; "
        "a reference to the sigil daemon was renamed"
    )
    assert files >= LEDGER_BASELINE_FILES


def test_the_sigild_surface_did_not_shrink() -> None:
    occurrences = len(_git_grep("sigild", ignore_case=True))
    files = len(_files("sigild", ignore_case=True))
    assert occurrences >= SIGILD_BASELINE_OCCURRENCES, (
        f"sigild references dropped from {SIGILD_BASELINE_OCCURRENCES} to {occurrences}"
    )
    assert files >= SIGILD_BASELINE_FILES


# The floor above tolerates prose drift, because FR-014's shim must *name* the
# daemon in order to exclude it. Excluding the three files that shim touches --
# plus this file, which quotes the tokens it forbids -- the surface is exactly
# equal on both trees. That is the assertion with teeth: it catches a ledger
# reference being renamed anywhere the shim does not legitimately reach.
LEDGER_STABLE_SURFACE = 153
_SHIM_AND_SELF = (
    ":!tests/test_rebrand.py",
    ":!src/kenaz_ml/config.py",
    ":!tests/test_env_deprecation_shim.py",
    ":!tests/test_registry_guarantees.py",
)


def test_the_ledger_surface_outside_the_shim_is_exactly_unchanged() -> None:
    """FR-007, and strictly stronger than the floor.

    Measured identically on ``main`` (163 - 6 - 4) and on this branch
    (181 - 12 - 10 - 6): both 153. A prose ledger reference renamed to ``kenaz``
    anywhere outside the shim files fails here, which the >= floor permits.
    """
    hits = _git_grep(LEDGER_PATTERN, *_SHIM_AND_SELF, ignore_case=True)
    assert len(hits) == LEDGER_STABLE_SURFACE, (
        f"the ledger surface outside the FR-014 shim moved from "
        f"{LEDGER_STABLE_SURFACE} to {len(hits)}. If this is a deliberate edit to "
        f"prose naming the daemon, update the constant; if a rename leaked, fix it."
    )


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        # The daemon's data directory. THE critical invariant: config.py
        # hardcodes _data_home()/"sigild"/"data.db", and renaming it points
        # every install at a database that does not exist.
        ("share/sigild", 15),
        # The daemon's CLI, which drives this service's :7774 endpoints.
        ("sigilctl", 11),
        # The daemon's repository. Note the organisation is kameas-ai for both
        # products; only the repository name distinguishes them.
        (r"github\.com/kameas-ai/sigil\b", 6),
    ],
)
def test_load_bearing_daemon_tokens_are_exactly_unchanged(pattern: str, expected: int) -> None:
    """Unlike the broad surface these are pinned to EQUALITY, because each is a
    live runtime or integration contract rather than prose."""
    assert len(_git_grep(pattern)) == expected


def test_no_daemon_name_was_rewritten_into_the_kenaz_namespace() -> None:
    """The signature of the catastrophic failure, asserted directly.

    If a blanket ``sigil`` -> ``kenaz`` substitution had been run, these tokens
    would exist. They must not, in any file, ever.
    """
    offences = _git_grep(r"kenazd\b|kenazctl|share/kenaz|KENAZD_", ignore_case=True)
    assert offences == [], "the daemon's name was rewritten:\n  " + "\n  ".join(offences)


# --- and the same invariant at runtime, because text can be right and still wrong


def test_db_path_still_resolves_to_the_sigild_database() -> None:
    """The single assertion most worth keeping forever."""
    assert config.db_path() == Path.home() / ".local" / "share" / "sigild" / "data.db"


def test_models_dir_still_lives_under_sigild() -> None:
    assert config.models_dir() == Path.home() / ".local" / "share" / "sigild" / "ml-models"


def test_the_daemon_plugin_url_variable_kept_its_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-007. SIGILD_PLUGIN_URL is the daemon's, so the FR-014 rename does not
    reach it and the shim invents no ``KENAZD_`` form for it."""
    monkeypatch.setenv("SIGILD_PLUGIN_URL", "http://127.0.0.1:7775")
    assert config.sigild_plugin_url() == "http://127.0.0.1:7775"
    assert config._legacy_env_name("SIGILD_PLUGIN_URL") is None


def test_the_feature_store_still_reads_the_sigild_user_directory() -> None:
    """The local Feast online store is written next to the daemon's data.db."""
    from kenaz_ml.feature_store import config as fs_config

    assert fs_config.user_data_dir() == Path.home() / ".local" / "share" / "sigild"


# ===========================================================================
# Historical records are not rewritten  (FR-013, C-006, D-003)
# ===========================================================================


def test_merged_missions_records_still_use_the_old_names() -> None:
    """kitty-specs/ describes what was true when it was written. If a bulk edit
    swept through it, this fails."""
    completed = subprocess.run(
        ["git", "grep", "-lP", "sigil_ml", "--", "kitty-specs"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):  # pragma: no cover
        pytest.skip("git grep unavailable")
    assert completed.stdout.split(), "kitty-specs/ no longer mentions sigil_ml; historical records were rewritten"
