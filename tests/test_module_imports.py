"""Every module in the package must import.

`curious_george.evaluation.probe` was unimportable for a week - cleanup 2
(`c8bc5fb`) stopped re-exporting `get_agent` from the package root and the module
kept importing it from there. 656 tests passed throughout, because nothing in
`tests/` imports that module: the questions repository does, against a PINNED
older commit, so the break reached no gate here and no user there until someone
ran `make dev`.

An import is the cheapest possible assertion and it covers the whole tree, which
is exactly the class of failure a per-feature test suite misses. Modules that
cost real time or hardware at import are excluded by name, with the reason.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import curious_george

#: Modules not imported here, and why. Keep this list short and justified: an
#: entry is a module whose import is NOT covered, which is the bug above.
SKIP: dict[str, str] = {}


def _module_names() -> list[str]:
    return sorted(
        info.name
        for info in pkgutil.walk_packages(
            curious_george.__path__, prefix="curious_george."
        )
        if info.name not in SKIP
    )


@pytest.mark.parametrize("name", _module_names())
def test_every_module_imports(name: str) -> None:
    """A module that cannot be imported is broken however green the suite is."""
    importlib.import_module(name)


def test_the_gate_can_see_something() -> None:
    """The walk found modules at all - an empty parametrisation passes vacuously."""
    names = _module_names()
    assert len(names) > 20, f"only {len(names)} modules found; the walk is not working"
    assert "curious_george.evaluation.probe" in names
