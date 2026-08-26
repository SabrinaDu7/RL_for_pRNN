"""Every reference a live file makes must resolve: paths in git, modules importable.

The cheapest gate with the best record in the 2026-08 audit: 41 of 174
referenced paths were dead in a clean clone, including a copy-pasteable
`pytest` command naming a deleted test file. Four of them looked fine from the
machine that wrote them and were broken for everyone else, which is exactly
what a filesystem check would miss - so this tests against `git ls-files`, not
against `Path.exists()`.

SCOPE, and it is a real distinction. Only **live** files are checked: code,
configs, launchers, `README.md`, `CLAUDE.md`, `justfile`. Two trees are
deliberately exempt:

* `docs/claude_logs/` - dated session records. "X moved to Y" legitimately
  names a path that no longer exists; that is what a record is for.
* `throwaway/` - reference-only by `CLAUDE.md`'s own rule, and its contents
  name each other from before they moved.

Paths into another repository (`../curious-george-questions/...`) are skipped:
this gate can only speak for this one.
"""

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Top-level directories whose contents are version-controlled repo paths.
#: `outputs/` is deliberately NOT here: it is where artifacts are WRITTEN, and a
#: script naming its own output path is not a broken reference. This gate cannot
#: tell a write path from a read path, and flagging every generated figure would
#: drown the signal. Inputs read from `outputs/` are a real risk - the OMT golden
#: gate depended on an untracked checkpoint there - but that is caught by reading
#: the code, not by this matcher.
ROOTS = "docs|scripts|tests|curious_george|Configs|slurm|throwaway"

#: A repo-relative path. `(?<![\w./-])` stops the tail of a LONGER path from
#: matching - without it `../curious-george-questions/tests/x.py` reports a
#: dead `tests/x.py`.
PATH = re.compile(rf"(?<![\w./-])(?:{ROOTS})/[A-Za-z0-9_./-]+\.(?:py|md|yaml|yml|sh|png|json|in|pt|toml)")

LIVE_GLOBS = ("*.py", "*.yaml", "*.yml", "*.sh", "*.toml", "*.md", "justfile")
EXEMPT_PREFIXES = ("throwaway/", "docs/claude_logs/")

#: This file, which cites the defects it exists to catch BY NAME - a deleted
#: test, a climbing path - so it trips its own matcher. A gate that documents a
#: failure cannot be subject to itself. Nothing else may be added here without a
#: reason as specific as this one.
EXEMPT_FILES = ("tests/test_references_resolve.py",)


def _git(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True)
    return out.stdout.split()


def _live_files() -> list[str]:
    return [
        f for f in _git("ls-files", *LIVE_GLOBS)
        if not f.startswith(EXEMPT_PREFIXES) and f not in EXEMPT_FILES
    ]


def _references() -> dict[str, list[str]]:
    """path -> the live files that name it."""
    found: dict[str, list[str]] = {}
    for f in _live_files():
        try:
            text = (REPO / f).read_text(errors="ignore")
        except OSError:
            continue
        for match in PATH.findall(text):
            found.setdefault(match.rstrip(".,)`'\""), []).append(f)
    return found


def test_every_referenced_path_is_tracked():
    """A path named in live code or docs resolves in a clean clone."""
    tracked = set(_git("ls-files"))
    dead = {p: sorted(set(src)) for p, src in _references().items() if p not in tracked}
    assert not dead, "referenced but not in git:\n" + "\n".join(
        f"  {p}\n      named by: {', '.join(src)}" for p, src in sorted(dead.items())
    )


def test_the_gate_can_see_something():
    """A matcher that silently stops matching would make the test above pass
    forever. Pin that it still finds a substantial reference set."""
    refs = _references()
    assert len(refs) > 30, f"only {len(refs)} paths matched; the regex is probably broken"


def test_no_climbing_paths_in_live_files():
    """`../..` encodes where a file sits, so moving anything breaks it -
    silently, because a missing file reads as an absent value (CLAUDE.md).
    `tests/perf/compare_batch_learning.py` shipped a copy-pasteable
    `tests/perf/results/../compare_batch_learning.py` for exactly this reason.

    A leading `../` is fine: that is a deliberate reference to a SIBLING
    REPOSITORY, not a climb inside this one."""
    climbing = re.compile(rf"(?:{ROOTS})/[A-Za-z0-9_./-]*/\.\./")
    offenders = {
        f: sorted(set(climbing.findall((REPO / f).read_text(errors="ignore"))))
        for f in _live_files()
        if climbing.search((REPO / f).read_text(errors="ignore"))
    }
    assert not offenders, f"paths that climb: {offenders}"


@pytest.mark.parametrize("exempt", EXEMPT_PREFIXES)
def test_exemptions_are_real_directories(exempt):
    """An exemption for a directory that no longer exists silently widens the
    gate's blind spot."""
    assert (REPO / exempt).is_dir(), f"{exempt} is exempt from this gate but does not exist"


# ---------------------------------------------------------------------------
# Modules. The other half of "a reference resolves", and the half that was
# missing: `io/` -> `log_and_store/` left `curious_george.storage` in
# `slurm/train_fast.sh` and the then-`scripts/multienv/checkpoint_curve.py`. Both
# live, neither is imported by any test, and a shell heredoc is invisible to
# every linter here - so the rename shipped broken and the suite stayed green.

#: A dotted name under this package, as it appears in an import, a heredoc, or
#: prose. `(?<![\w.])` stops a longer dotted name's tail from matching.
MODULE = re.compile(r"(?<![\w.])curious_george(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


def _module_references() -> dict[str, list[str]]:
    """dotted name -> the live files that name it."""
    found: dict[str, list[str]] = {}
    for f in _live_files():
        try:
            text = (REPO / f).read_text(errors="ignore")
        except OSError:
            continue
        for match in MODULE.findall(text):
            found.setdefault(match, []).append(f)
    return found


def _resolves(dotted: str) -> bool:
    """True if the name is a module, or an attribute of one (`pkg.mod.SYMBOL`).

    Attributes are accepted without importing: this gate answers "does the
    MODULE PATH still exist", which is what a rename breaks. Whether a symbol
    inside it still exists is the type checker's job.
    """
    parts = dotted.split(".")
    while len(parts) > 1:
        try:
            if importlib.util.find_spec(".".join(parts)) is not None:
                return True
        except (ImportError, AttributeError, ValueError):
            pass
        parts.pop()
    return False


def test_every_referenced_module_resolves():
    """A `curious_george.x.y` named anywhere live is importable in a clean env."""
    dead = {m: sorted(set(src)) for m, src in _module_references().items() if not _resolves(m)}
    assert not dead, "named but not importable:\n" + "\n".join(
        f"  {m}\n      named by: {', '.join(src)}" for m, src in sorted(dead.items())
    )


def test_the_module_gate_can_see_something():
    """A matcher that stops matching would make the test above pass forever."""
    refs = _module_references()
    assert len(refs) > 20, f"only {len(refs)} modules matched; the regex is probably broken"
