"""`Selected` is the same rooms at two affordances, and that is the whole point.

The walkable and impassable pools are DIFFERENT SEQUENCES - `admissible_placements`
takes `content`, and impassable landmarks admit 9,074 placements against
walkable's 19,820 - so selecting rooms by INDEX cannot express "the same rooms,
walkable". Measured: 0 of the 5 selected indices name the same room in both.
`ROOMS_SELECTED` therefore pins the ANCHORS and `Selected` applies the affordance
on top.

These pin the three things that makes the contrast single-variable: the committed
anchors really are the pool's at the named indices, the flip moves no anchor, and
the flip changes no observation.
"""

import numpy as np
import pytest

from curious_george.envs.layouts import (
    BASE_ROOM_ID,
    MULTI_ROOM_ID,
    EnvContent,
    EnvShape,
    LandmarkKind,
    ROOMS_SELECTED,
    RoomRules,
    RoomSetRules,
    Selected,
    Uniform,
    Vary,
    base_walkable,
    resolve_rooms,
    with_affordance,
)

#: Where each committed room came from, in order. The keys are `Layout.key`,
#: which INCLUDES impassability - so these are the impassable pool's keys.
SOURCE_INDICES = (0, 14, 31, 35, 83, 126, 144, 169, 191, 195)


@pytest.fixture(scope="module")
def impassable_pool():
    """The pool the committed anchors were taken from."""
    return resolve_rooms(
        shape=EnvShape(BASE_ROOM_ID),
        content=EnvContent(
            kinds=tuple(LandmarkKind(s, impassable=True) for s in ("x", "plus", "block3"))
        ),
        source=Uniform(n=200, seed=7),
        room_rules=RoomRules(),
        set_rules=RoomSetRules(varies=frozenset({Vary.POSITION})),
    )


def test_committed_rooms_are_the_pool_at_the_named_indices(impassable_pool):
    """The literal table is derivable, so a generator change is CAUGHT here.

    `Committed`'s docstring gives the reason this matters: re-deriving a set
    after a generator change silently yields different rooms while every
    historical result keeps referring to the old ones.
    """
    blocked = with_affordance(ROOMS_SELECTED, impassable=True)
    assert len(blocked) == len(SOURCE_INDICES)
    for room, index in zip(blocked, SOURCE_INDICES):
        assert room.key == impassable_pool[index].key
        assert room.anchors == impassable_pool[index].anchors


def test_the_pools_are_not_index_compatible(impassable_pool):
    """The fact `Selected` exists for. If this ever fails, `Selected` is
    unnecessary and selecting by index would do."""
    walkable_pool = resolve_rooms(
        shape=EnvShape(BASE_ROOM_ID),
        content=EnvContent(
            kinds=tuple(LandmarkKind(s, impassable=False) for s in ("x", "plus", "block3"))
        ),
        source=Uniform(n=200, seed=7),
        room_rules=RoomRules(),
        set_rules=RoomSetRules(varies=frozenset({Vary.POSITION})),
    )
    shared = [
        i for i in SOURCE_INDICES[:5]
        if impassable_pool[i].anchors == walkable_pool[i].anchors
    ]
    assert not shared, (
        f"indices {shared} now name the same room in both pools; selecting by "
        "index may be viable again"
    )


@pytest.mark.parametrize("n", (1, 5, 10))
def test_selected_returns_n_rooms_at_the_requested_affordance(n):
    for impassable in (False, True):
        rooms = resolve_rooms(
            shape=EnvShape(BASE_ROOM_ID), content=EnvContent(),
            source=Selected(n=n, impassable=impassable),
        )
        assert len(rooms) == n
        assert all(r.blocks_movement is impassable for r in rooms)


def test_the_flip_moves_no_anchor():
    walkable = with_affordance(ROOMS_SELECTED, impassable=False)
    blocked = with_affordance(ROOMS_SELECTED, impassable=True)
    for a, b in zip(walkable, blocked):
        assert a.anchors == b.anchors
        assert [(lm.shape, lm.color) for lm in a.landmarks] == \
               [(lm.shape, lm.color) for lm in b.landmarks]
        # The KEY is expected to differ: it encodes impassability on purpose, so
        # one affordance's cached results can never be served for the other.
        assert a.key != b.key


def test_impassable_removes_exactly_its_landmark_cells():
    base = base_walkable(BASE_ROOM_ID)
    for a, b in zip(with_affordance(ROOMS_SELECTED, impassable=False),
                    with_affordance(ROOMS_SELECTED, impassable=True)):
        assert a.walkable(base) == base
        assert b.walkable(base) == base - b.cells


@pytest.mark.parametrize("room_index", (0, 1))
def test_the_flip_changes_no_observation(room_index):
    """The claim the single-variable contrast rests on.

    `LandmarkKind.impassable` says `Obstacle` and `Floor` "render identically at
    every tile size". Checked here where it matters - the agent's own view, over
    every (position, direction) - rather than on the top-down frame, which does
    NOT match at a fixed seed because the agent cannot spawn on an obstacle.
    """
    import gymnasium as gym
    from minigrid.core.grid import Grid

    # HERMETIC. `Grid.tile_cache` is a CLASS-level dict keyed on
    # obj.encode() + (agent_dir, highlight, tile_size), and it survives across
    # tests. Running `pytest tests/test_novel_object.py tests/test_selected_rooms.py`
    # made this assertion report 209 of 688 observations differing, while either
    # file alone reports 0 - so a previous test's cache entries change what this
    # one renders. The underlying claim is INDEPENDENTLY true: rendering a Floor
    # and an Obstacle of the same colour directly, with the cache cleared, gives
    # byte-identical tiles at tile_size 1 and 8. Clearing here makes the test
    # measure the rendering rather than the process's history.
    Grid.tile_cache.clear()

    cells = sorted(base_walkable(BASE_ROOM_ID))

    def views(layout):
        env = gym.make(MULTI_ROOM_ID[BASE_ROOM_ID], landmarks=list(layout.landmarks))
        env.reset(seed=0)
        u = env.unwrapped
        out = {}
        for x, y in cells:
            for d in range(4):
                u.agent_pos, u.agent_dir = (x, y), d
                out[(x, y, d)] = u.get_frame(tile_size=1, agent_pov=True).copy()
        return out

    walkable = views(with_affordance(ROOMS_SELECTED, impassable=False)[room_index])
    blocked = views(with_affordance(ROOMS_SELECTED, impassable=True)[room_index])
    differing = [k for k in walkable if not np.array_equal(walkable[k], blocked[k])]
    assert not differing, f"{len(differing)} of {len(walkable)} observations differ"


def test_n_out_of_range_is_refused():
    with pytest.raises(ValueError, match="must be 1"):
        Selected(n=0)
    with pytest.raises(ValueError, match="must be 1"):
        Selected(n=len(ROOMS_SELECTED) + 1)
