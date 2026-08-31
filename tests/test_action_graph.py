"""The action graph is the machine the environments actually step.

Walkers and BFS distances mean something only if the table they consume is the
table the device backend steps, so fidelity is gated against the live MiniGrid
env, and the distance metric is pinned in ACTIONS - a turn costs a step, which
Manhattan distance would miss.
"""

import dataclasses

import gymnasium as gym
import numpy as np
import pytest

from curious_george.configs import PRESETS
from curious_george.envs.action_graph import (
    ActionGraph,
    FORWARD,
    categorical_walk,
    layout_tables,
    spawn_states,
    sweeper_walk,
    walk,
)
from curious_george.envs.layouts import (
    BASE_ROOM_ID,
    MULTI_ROOM_ID,
    Selected,
    base_walkable,
    resolve_rooms,
)

BASE = base_walkable(BASE_ROOM_ID)


def _rooms(*, impassable: bool):
    """The committed room set, exactly as an arm's training run resolves it."""
    entry = PRESETS["multienv-fast"]
    cfg = entry[1] if isinstance(entry, tuple) else entry
    env = dataclasses.replace(cfg.env, source=Selected(n=5, impassable=impassable))
    return resolve_rooms(
        shape=env.shape, content=env.content, source=env.source,
        room_rules=env.room_rules, set_rules=env.set_rules, indices=env.indices,
    )


@pytest.fixture(scope="module")
def rooms_impassable():
    return _rooms(impassable=True)


@pytest.fixture(scope="module")
def rooms_walkable():
    return _rooms(impassable=False)


@pytest.fixture(scope="module")
def table_impassable(rooms_impassable):
    return layout_tables(layouts=rooms_impassable[:1])[0]


@pytest.fixture(scope="module")
def table_walkable(rooms_walkable):
    return layout_tables(layouts=rooms_walkable[:1])[0]


# --- the table is the wrapper's table ---------------------------------------


def test_layout_tables_match_the_table_wrapper(rooms_impassable, tmp_path):
    """`layout_tables` skips the observation bank; the dynamics must still be
    byte-identical to what `TableDrivenRGBPartialObsWrapper` steps."""
    from curious_george.envs.obs_bank import TableDrivenRGBPartialObsWrapper

    tables = layout_tables(layouts=rooms_impassable[:2])
    env = gym.make(
        MULTI_ROOM_ID[BASE_ROOM_ID],
        landmarks=list(rooms_impassable[0].landmarks),
        agent_start_pos=None,
        agent_start_dir=None,
        render_mode="rgb_array",
    )
    wrapper = TableDrivenRGBPartialObsWrapper(env, tile_size=1, bank_dir=tmp_path)
    for layout, table in zip(rooms_impassable[:2], tables):
        wrapper.unwrapped.landmarks = list(layout.landmarks)
        wrapper.reset(seed=0)
        assert np.array_equal(table, wrapper._next_state)
    wrapper.close()


def test_walk_matches_the_live_environment(rooms_impassable, table_impassable):
    """Replay one random action sequence through the raw MiniGrid env and the
    table; the pre-action position streams must be identical."""
    env = gym.make(
        MULTI_ROOM_ID[BASE_ROOM_ID],
        landmarks=list(rooms_impassable[0].landmarks),
        agent_start_pos=None,
        agent_start_dir=None,
    )
    env.reset(seed=5)
    base = env.unwrapped
    spawn = (base.agent_pos[0], base.agent_pos[1], base.agent_dir)

    rng = np.random.default_rng(11)
    actions = rng.integers(0, 4, size=(1, 60))
    live = []
    for action in actions[0]:
        live.append(tuple(base.agent_pos))
        env.step(int(action))
    tabled = walk(table_impassable, spawns=np.array([spawn]), actions=actions)
    assert [tuple(p) for p in tabled[0]] == live
    env.close()


# --- distance is measured in actions ----------------------------------------


def test_distance_counts_turns(table_walkable, rooms_walkable):
    """One step ahead is distance 1; the cell directly behind is 3 (two turns
    plus a move). Manhattan distance would call both 1."""
    walkable = rooms_walkable[0].walkable(BASE)
    x, y = next(
        c for c in sorted(walkable)
        if (c[0] + 1, c[1]) in walkable and (c[0] - 1, c[1]) in walkable
    )
    dist = ActionGraph(table_walkable).distance_map(spawn=(x, y, 0))  # facing +x
    assert dist[x, y] == 0
    assert dist[x + 1, y] == 1
    assert dist[x - 1, y] == 3


def test_every_walkable_cell_is_reachable_and_no_other(
    rooms_impassable, table_impassable
):
    """No enclosed pockets in the committed rooms (so the walkable count IS
    the reachable count), and landmark/wall cells are flagged -1."""
    walkable = rooms_impassable[0].walkable(BASE)
    dist = ActionGraph(table_impassable).distance_map(
        spawn=(*next(iter(sorted(walkable))), 0)
    )
    reached = {(x, y) for x in range(dist.shape[0]) for y in range(dist.shape[1]) if dist[x, y] >= 0}
    assert reached == walkable


# --- walkers -----------------------------------------------------------------


def test_turn_left_walker_stays_put(table_walkable, rooms_walkable):
    spawns = spawn_states(
        rooms_walkable[0].walkable(BASE), n=4, rng=np.random.default_rng(0)
    )
    positions = categorical_walk(
        table_walkable, probs=[1, 0, 0, 0], spawns=spawns, steps=32,
        rng=np.random.default_rng(0),
    )
    assert (positions == positions[:, :1]).all()


def test_walkers_never_enter_landmarks(rooms_impassable, table_impassable):
    walkable = rooms_impassable[0].walkable(BASE)
    rng = np.random.default_rng(3)
    spawns = spawn_states(walkable, n=32, rng=rng)
    positions = categorical_walk(
        table_impassable, probs=[0.15, 0.15, 0.6, 0.1], spawns=spawns,
        steps=128, rng=rng,
    )
    assert {tuple(p) for p in positions.reshape(-1, 2)} <= walkable


def test_sweeper_covers_the_walkable_room(rooms_walkable, table_walkable):
    """The positive control does what makes it one: full coverage inside the
    256-step episode (the calibration's cov@256 = 1.000)."""
    walkable = rooms_walkable[0].walkable(BASE)
    graph = ActionGraph(table_walkable)
    for spawn in spawn_states(walkable, n=3, rng=np.random.default_rng(1)):
        positions = sweeper_walk(graph, walkable=walkable, spawn=tuple(spawn), steps=256)
        assert {tuple(p) for p in positions[0]} == walkable


def test_spawns_cover_only_the_walkable_set(rooms_impassable):
    walkable = rooms_impassable[0].walkable(BASE)
    spawns = spawn_states(walkable, n=500, rng=np.random.default_rng(2))
    assert {(x, y) for x, y, _ in spawns} <= walkable
    assert set(spawns[:, 2]) == {0, 1, 2, 3}
