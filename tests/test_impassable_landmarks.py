"""Impassable landmarks: the walkable set and the transition table go per room.

Until 2026-08-27 every room in a pool shared one walkable set and therefore one
transition table, and the device path asserted exactly that. Landmarks were
walkable `Floor`, so they changed what the agent SAW and never where it could
GO. An `Obstacle` breaks that per room, which is the whole point of the change
and the reason several things downstream had to learn that "which cells exist"
is a question about a room, not about a shape.

These pin the three claims the rest of Q3 rests on:
  the walkable set shrinks, per room, by exactly the landmark cells
  the device path holds one transition table PER ROOM and indexes it by stream
  the walkable path is untouched
"""

import pytest
import torch

from curious_george.configs import EnvBackend, EnvCfg
from curious_george.envs.layouts import (
    BASE_ROOM_ID,
    EnvContent,
    EnvShape,
    LandmarkKind,
    Layout,
    RoomSetRules,
    Uniform,
    Vary,
    base_walkable,
    resolve_rooms,
)
from tests.small_config import small_config

SHAPES = ("x", "plus", "block3")


def _content(*, impassable: bool) -> EnvContent:
    return EnvContent(kinds=tuple(LandmarkKind(s, impassable=impassable) for s in SHAPES))


def _rooms(*, impassable: bool, n: int = 3, seed: int = 7):
    return resolve_rooms(
        shape=EnvShape(BASE_ROOM_ID),
        content=_content(impassable=impassable),
        source=Uniform(n=n, seed=seed),
        set_rules=RoomSetRules(varies=frozenset({Vary.POSITION})),
    )


# --- the walkable set is a property of the ROOM, not of the shape ----------


def test_base_walkable_is_the_walls_and_nothing_else():
    """It builds a plain `gym.make(room_id)`, which paints the room's DEFAULT
    landmarks. Reading walkability off that grid gave the right answer only
    while those landmarks were walkable; with impassable ones it would have
    started returning "the room minus the default landmarks" and fed that to
    the whole layout generator."""
    assert len(base_walkable(BASE_ROOM_ID)) == 172


@pytest.mark.parametrize("impassable", [False, True])
def test_a_room_removes_exactly_its_own_landmark_cells(impassable):
    base = base_walkable(BASE_ROOM_ID)
    for room in _rooms(impassable=impassable):
        walkable = room.walkable(base)
        expected = base - room.cells if impassable else base
        assert walkable == expected
        assert room.blocks_movement is impassable


def test_walkable_landmarks_take_nothing_away():
    """The control arm has to be genuinely unchanged, or every comparison
    against it is a comparison of two manipulations."""
    base = base_walkable(BASE_ROOM_ID)
    assert all(room.walkable(base) == base for room in _rooms(impassable=False))


def test_two_rooms_differ_in_which_cells_exist():
    """The premise of everything in Phase 3: with impassable landmarks the
    walkable set is no longer shared, so anything that normalises by it or
    enumerates from it has to ask a room."""
    base = base_walkable(BASE_ROOM_ID)
    sets = {frozenset(r.walkable(base)) for r in _rooms(impassable=True)}
    assert len(sets) > 1


def test_impassability_is_part_of_a_room_key():
    """`key` names cached results. Two rooms with identical landmarks that
    differ only in whether the agent can enter them have different dynamics,
    and a shared key would serve one's results for the other."""
    anchors = _rooms(impassable=False)[0].anchors
    walk, block = (
        Layout(tuple(
            type(lm)(lm.shape, lm.color, lm.anchor, impassable=flag)
            for lm in _rooms(impassable=False)[0].landmarks
        ))
        for flag in (False, True)
    )
    assert walk.anchors == block.anchors == anchors
    assert walk.key != block.key


# --- the device path holds one table per room ------------------------------


def _pool(*, impassable: bool):
    from curious_george.training.setup import setup_envs

    cfg = small_config(
        num_envs=4,
        backend=EnvBackend.DEVICE,
        env=EnvCfg(content=_content(impassable=impassable),
                   source=Uniform(n=3, seed=7),
                   set_rules=RoomSetRules(varies=frozenset({Vary.POSITION}))),
        evals=frozenset({__import__(
            "curious_george.configs", fromlist=["EvalKind"]).EvalKind.SPATIAL_MULTIROOM}),
    )
    return setup_envs(cfg)


def test_the_transition_table_carries_a_layout_axis():
    """It did not, and the device path raised rather than pretend otherwise."""
    pool = _pool(impassable=True)
    assert pool.next_state.shape[0] == pool.n_layouts == 3
    assert pool.next_state.shape[0] == pool.obs_banks.shape[0], (
        "the two tables must be indexed by the same layout axis"
    )


def test_the_per_room_tables_actually_differ():
    """Without this the test above passes on three copies of one table."""
    tables = _pool(impassable=True).next_state
    assert not torch.equal(tables[0], tables[1]) or not torch.equal(tables[1], tables[2])


def test_walkable_rooms_still_share_one_table():
    """The old invariant, now a measured fact rather than an assertion: with
    walkable landmarks the rooms differ in what is SEEN and not in where the
    agent can go."""
    tables = _pool(impassable=False).next_state
    assert torch.equal(tables[0], tables[1]) and torch.equal(tables[1], tables[2])


def test_step_device_uses_the_room_each_stream_is_in():
    """The gather that makes it per-room. Put two streams at the same pose in
    DIFFERENT rooms, step both forward, and they must be able to disagree."""
    pool = _pool(impassable=True)
    base = base_walkable(BASE_ROOM_ID)
    rooms = pool.layouts

    # A cell that is walkable in room 0 and blocked in some other room, so the
    # two streams genuinely have different dynamics from the same pose.
    blocked_elsewhere = [
        (cell, j)
        for j, other in enumerate(rooms[1:], start=1)
        for cell in sorted(rooms[0].walkable(base) & other.cells)
    ]
    if not blocked_elsewhere:
        pytest.skip("this seed's rooms do not overlap that way")
    (cx, cy), other = blocked_elsewhere[0]

    # Approach that cell from a neighbour, facing it.
    from minigrid.core.constants import DIR_TO_VEC

    for d in range(4):
        dx, dy = (int(v) for v in DIR_TO_VEC[d])
        src = (cx - dx, cy - dy)
        if src in rooms[0].walkable(base) and src in rooms[other].walkable(base):
            break
    else:
        pytest.skip("no approach cell walkable in both rooms")

    dev = pool.device
    pool.positions.copy_(torch.tensor([src] * pool.B, device=dev))
    pool.directions.copy_(torch.full((pool.B,), d, dtype=torch.long, device=dev))
    layouts = torch.zeros(pool.B, dtype=torch.long, device=dev)
    layouts[1] = other
    pool.stream_layout.copy_(layouts)

    pool.step_device(actions=torch.full((pool.B,), 2, dtype=torch.long, device=dev))
    moved = pool.positions.cpu().numpy()

    assert tuple(moved[0]) == (cx, cy), "stream 0's room leaves that cell open"
    assert tuple(moved[1]) == src, "stream 1's room has an object there; it must not move"


def test_the_agent_never_starts_inside_an_object():
    """`_gen_grid` places the agent before painting landmarks unless one blocks.
    Checked here too because the reset path the pool uses is not the one the
    fork's own test exercises."""
    pool = _pool(impassable=True)
    for _ in range(15):
        _, positions = pool.reset_all()
        for b, pos in enumerate(positions):
            room = pool.layouts[int(pool.stream_layout[b].item())]
            assert tuple(pos) not in room.cells


# NOT TESTED HERE, deliberately: `_collect_layout_banks` also refuses a layout
# whose table has environment-triggered rewards or terminations, because
# `step_device` returns a constant zero reward and would drop them silently.
# That guard is unreachable through this API - a `Landmark` cannot paint a goal
# or lava - so any test of it would have to fake the table, which would check
# the fake rather than the path. It stays as a defence against a future room
# type. `tests/test_return_is_measured_or_absent.py` covers the same rule at
# the level where it IS reachable.
