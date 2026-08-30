"""The launchers' command lines must PARSE, and that is checked here, not on a node.

Two jobs died on tyro syntax before this existed: `env.source` is a union, so its
member is a SUBCOMMAND, its fields only exist after it, and tyro applies every
option to the directly preceding subcommand - so preset-level flags must come
BEFORE it. Neither mistake is visible by reading the script, and both cost a GPU
allocation and ~4 minutes of `uv sync` to discover.
"""

import re
import shlex
from pathlib import Path

import pytest

from curious_george.configs import cli

SLURM = Path(__file__).resolve().parents[1] / "slurm"

#: Shell variables the launcher expands, and a legal value for each.
BINDINGS = {
    "SEED": "3", "NAME": "test-run", "N": "5",
    "FLAG": "--env.source.impassable", "ENT": "0.003", "OFFSET": "1",
    "POLICY_GRAPH": "", "RAMP": "",
}


def _invocation(script: Path) -> list[str]:
    """The `main_train.py ...` line, as argv, with shell variables bound."""
    text = script.read_text()
    m = re.search(r"^uv run python main_train\.py (.*?)(?=\n\s*>)", text, re.M | re.S)
    assert m, f"no main_train.py invocation found in {script.name}"
    # The captured text ends at the redirect line, so the last content line's
    # own line-continuation backslash is still attached.
    body = m.group(1).replace("\\\n", " ").rstrip().rstrip("\\").strip()
    # `${VAR:+text}` - emit `text` only when VAR is bound non-empty. parity.sh
    # uses this so an omitted argument passes NO flag and the preset's own
    # default stands; that behaviour is exactly what needs gating.
    def conditional(match: "re.Match") -> str:
        var, text = match.group(1), match.group(2)
        return text if BINDINGS.get(var) else ""

    body = re.sub(r"\$\{(\w+):\+([^}]*)\}", conditional, body)
    for name, value in BINDINGS.items():
        body = body.replace(f'"${name}"', value).replace(f"${name}", value)
    assert "$" not in body, f"unbound shell variable in {script.name}: {body}"
    return shlex.split(body)


@pytest.mark.parametrize("name", ["multienv.sh", "parity.sh"])
def test_the_launcher_command_line_parses(name):
    """A launcher whose arguments tyro refuses is a job that dies after the
    allocation, the clone and the sync - and says nothing until then."""
    argv = _invocation(SLURM / name)
    cfg = cli(argv)  # SystemExit here IS the failure
    assert cfg.run.exp_name == "test-run"
    assert cfg.run.seed == 3


def test_multienv_launcher_selects_the_room_set_it_claims():
    """Ordering silently changes MEANING, not just validity: the affordance flag
    must reach `env.source`, not be swallowed as an unrecognized option."""
    from curious_george.envs.layouts import Selected

    cfg = cli(_invocation(SLURM / "multienv.sh"))
    assert isinstance(cfg.env.source, Selected)
    assert cfg.env.source.n == 5
    assert cfg.env.source.impassable is True, "the --env.source.impassable flag was lost"
