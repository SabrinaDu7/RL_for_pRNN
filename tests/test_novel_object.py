"""A configured novel object reaches the grid, and an unplaceable one is refused.

Q1 asks whether the agent goes to a novel object, so "the object is actually
there" is the premise of the whole question rather than a detail.

It was NOT there. `training/setup.py` read the position through
`getattr(cfg.env, "new_obj_pos", None)` while `EnvCfg` had no such field, so the
default turned a removed field into "no object requested". Every run built a
room with no novel object in it and said nothing. These tests are the check that
could not fire before.
"""

import pytest

from curious_george.configs import EnvCfg
from curious_george.envs.layouts import BASE_ROOM_ID, EnvShape
from curious_george.training.setup import setup_env
from tests.small_config import small_config

#: A walkable L-room cell. Asserted walkable below rather than trusted, so this
#: fixture cannot rot into "the object was outside the room all along".
OBJECT_CELL = (7, 2)


def _cell(env, xy):
    return env.env.unwrapped.grid.get(*xy)


def test_the_fixture_cell_is_actually_in_the_room():
    assert OBJECT_CELL in EnvShape(BASE_ROOM_ID).walkable


def test_a_configured_novel_object_is_on_the_grid():
    """The regression. Before the fix this cell came back empty."""
    cfg = small_config(env=EnvCfg(novel_object=OBJECT_CELL))
    env = setup_env(cfg)
    placed = _cell(env, OBJECT_CELL)
    assert placed is not None, "novel object was not placed"
    assert type(placed).__name__ == "FloorBright"


def test_no_novel_object_leaves_the_cell_as_it_was():
    """The negative half: without the field the room must be unchanged, or the
    positive test above would pass for the wrong reason."""
    env = setup_env(small_config(env=EnvCfg()))
    assert _cell(env, OBJECT_CELL) is None


def test_the_object_is_visible_in_an_observation():
    """On the grid is not the same as reachable by the network. Observations
    render at tile_size=1, so the object is one pixel and a placement the agent
    never sees would be a null result that looks like a real one."""
    import numpy as np

    with_obj = setup_env(small_config(env=EnvCfg(novel_object=OBJECT_CELL)))
    without = setup_env(small_config(env=EnvCfg()))
    for env in (with_obj, without):
        env.env.unwrapped.agent_pos = (OBJECT_CELL[0], OBJECT_CELL[1] + 2)
        env.env.unwrapped.agent_dir = 3  # facing the object

    frames = [
        e.env.unwrapped.get_frame(highlight=False, tile_size=1, agent_pov=True)
        for e in (with_obj, without)
    ]
    assert not np.array_equal(*frames), "the novel object changes no pixel the agent sees"


def test_a_cell_outside_the_room_is_refused():
    """An invisible object and a null result are indistinguishable downstream,
    so this has to fail at construction rather than at analysis."""
    with pytest.raises(ValueError, match="not a walkable cell"):
        EnvCfg(novel_object=(0, 0))


def test_the_position_round_trips_through_the_run_record():
    """`provenance.json` and the wandb config both go through `to_dict`. A
    manipulation absent from the record is unreproducible."""
    cfg = small_config(env=EnvCfg(novel_object=OBJECT_CELL))
    assert cfg.to_dict()["env"]["novel_object"] == list(OBJECT_CELL)
