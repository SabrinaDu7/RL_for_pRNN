"""The offline probe tools (`surprisal_timing`, `readout_probe`,
`checkpoint_series`, `prediction_figures`) score a checkpoint under the config
its run trained under. Two things make that hold without a run directory full
of checkpoints: a checkpoint path resolves to its run, and a `--positions`
subset maps onto the run's own rooms rather than onto a re-typed set.
"""


import pytest

from curious_george.configs import cli
from curious_george.envs.layouts import ROOMS_SELECTED, resolve_layouts
from curious_george.evaluation.surprisal_timing import rooms_at
from curious_george.utils.checkpoints import ARCHIVE_DIRNAME, PRNN_CKPT_FILENAME, run_dir_of


def test_run_dir_of_the_rolling_and_the_archived_checkpoint(tmp_path):
    """Both layouts `training/loop.py::save_checkpoint` writes resolve to the
    directory holding `provenance.json`."""
    run = tmp_path / "run"
    assert run_dir_of(run / PRNN_CKPT_FILENAME) == run.resolve()
    assert run_dir_of(run / ARCHIVE_DIRNAME / "predictiveNet_state_step0000065536.pt") == run.resolve()


def _selected(*positions: int):
    return cli(["multienv-fast", "--run.no-wandb", "env.source:selected",
                "--env.source.impassable", "--env.source.positions", *map(str, positions)])


def test_rooms_at_defaults_to_every_room_the_run_trained_on():
    cfg = _selected(0, 1, 2, 3, 5, 6, 7, 8)
    assert rooms_at(cfg, None) == resolve_layouts(cfg)


def test_rooms_at_maps_positions_onto_the_runs_rooms_not_onto_a_prefix():
    """Position 5 is the run's FIFTH room (index 4) in the CE plan's set; a
    prefix read would hand back position 4's room, which the run never saw."""
    cfg = _selected(0, 1, 2, 3, 5, 6, 7, 8)
    rooms = resolve_layouts(cfg)
    assert rooms_at(cfg, (5,)) == [rooms[4]]
    assert rooms_at(cfg, (8, 0)) == [rooms[7], rooms[0]]
    assert rooms_at(cfg, (5,))[0].landmarks == resolve_layouts(_selected(5))[0].landmarks


def test_rooms_at_refuses_a_room_the_run_never_trained_on():
    with pytest.raises(ValueError, match=r"positions \[4\] are not in the run's set"):
        rooms_at(_selected(0, 1, 2, 3, 5, 6, 7, 8), (4,))


def test_rooms_at_refuses_positions_on_a_non_selected_source():
    cfg = cli(["multienv-fast", "--run.no-wandb", "env.source:uniform"])
    with pytest.raises(ValueError, match="needs a Selected room set"):
        rooms_at(cfg, (0,))
    assert len(ROOMS_SELECTED) >= 9  # the positions above exist
