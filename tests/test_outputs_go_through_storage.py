"""Everything this machine writes goes through RL_STORAGE.

`.env`'s RL_STORAGE is the single source of truth for where run output lands,
reached through `log_and_store/storage.py::get_storage_dir`. A hardcoded
`outputs/...` works on the machine that wrote it and silently writes to the
wrong place everywhere else - on the cluster it lands in $SLURM_TMPDIR and is
deleted when the job ends.

Two ways the rule was actually broken, both of which this gate catches:

* `check/wandb_compare.py` wrote to a literal `Path("outputs/summary")`.
* `wandb.init` without `dir=` writes a `wandb/` tree into the CURRENT WORKING
  DIRECTORY. `training/logging.py` passes `dir=run_ctx.model_dir` and is fine;
  the OMT/OTC task launchers did not, which is where this repo's stray 238 MB
  `wandb/` came from. The ported copy in the questions repo still omits it,
  under a comment claiming all task output lives under the storage root.

SCOPE: the library and its entry point. `tests/perf/` writes benchmark results
to caller-supplied paths and `throwaway/` is reference-only, so neither is
subject to this.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: The library, plus the example training entry point.
LIVE = sorted(REPO.glob("curious_george/**/*.py")) + [REPO / "main_train.py"]

#: A literal run-output path. `get_storage_dir()` is the only correct source.
HARDCODED = re.compile(r"""["']outputs/""")

#: An output path anchored to the user's home rather than the storage root.
#: `get_video_dir` wrote to $HOME/pRNN-RL/RLvideos, outside every rsync in
#: slurm/, so a cluster run's videos never left the node.
HOME_ANCHORED = re.compile(r"""environ\[.HOME.\]|Path\.home\(\)|expanduser""")

#: `wandb.init(` and everything up to its closing paren, so `dir=` is visible.
WANDB_INIT = re.compile(r"wandb\.init\((.*?)\)", re.DOTALL)


def _relative(p: Path) -> str:
    return str(p.relative_to(REPO))


@pytest.mark.parametrize("path", LIVE, ids=_relative)
def test_no_hardcoded_output_path(path):
    """No live file names `outputs/` directly; it asks the accessor."""
    hits = [
        f"{_relative(path)}:{i}: {line.strip()}"
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if HARDCODED.search(line) and "get_storage_dir" not in line
    ]
    assert not hits, (
        "hardcoded output path - use log_and_store.storage.get_storage_dir():\n"
        + "\n".join(hits)
    )


@pytest.mark.parametrize("path", LIVE, ids=_relative)
def test_no_home_anchored_output_path(path):
    """Run output does not hang off $HOME; slurm/ only rsyncs RL_STORAGE."""
    hits = [
        f"{_relative(path)}:{i}: {line.strip()}"
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if HOME_ANCHORED.search(line)
    ]
    assert not hits, "output path anchored to $HOME:\n" + "\n".join(hits)


@pytest.mark.parametrize("path", LIVE, ids=_relative)
def test_wandb_init_writes_under_the_storage_root(path):
    """`wandb.init` without `dir=` writes a wandb/ tree into the CWD."""
    bad = [
        call for call in WANDB_INIT.findall(path.read_text()) if "dir=" not in call
    ]
    assert not bad, (
        f"{_relative(path)}: wandb.init without dir= writes to the CWD, not "
        f"RL_STORAGE:\n  wandb.init({bad[0][:120].strip()}...)"
    )


def test_the_gate_can_see_the_files_it_guards():
    """A glob that stops matching would make both tests pass forever."""
    assert len(LIVE) > 30, f"only {len(LIVE)} live files matched"
    assert (REPO / "curious_george/log_and_store/storage.py") in LIVE
