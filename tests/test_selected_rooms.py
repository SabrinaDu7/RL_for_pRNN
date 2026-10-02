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
    Landmark,
    LandmarkKind,
    Layout,
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

    2026-08-30: the committed rooms deliberately carry `triangle3` where the
    source pool carries `x` (the active design changed; anchors and colours
    did not - see SHAPES and docs/invalid-runs.md). `Layout.key` hashes the
    shape, so the key comparison rebuilds each room with the source pool's
    shape: a generator change is still caught bitwise, and only the reviewed
    shape swap is exempted.
    """
    blocked = with_affordance(ROOMS_SELECTED, impassable=True)
    assert len(blocked) == len(SOURCE_INDICES)
    for room, index in zip(blocked, SOURCE_INDICES):
        as_sourced = Layout(tuple(
            Landmark("x" if lm.shape == "triangle3" else lm.shape,
                     lm.color, lm.anchor, impassable=lm.impassable)
            for lm in room.landmarks
        ))
        assert as_sourced.key == impassable_pool[index].key
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
            source=Selected(positions=tuple(range(n)), impassable=impassable),
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


def test_an_empty_or_out_of_range_selection_is_refused():
    with pytest.raises(ValueError, match="positions"):
        Selected(positions=())
    with pytest.raises(ValueError, match="positions"):
        Selected(positions=(len(ROOMS_SELECTED),))


def test_n_is_the_count_of_positions_and_nothing_else():
    """`n` used to be a second field that `positions` silently overrode; the
    launcher set both, so every 8-room run's provenance said n=5 (audit C10)."""
    assert Selected(positions=(0, 1, 2, 3, 5, 6, 7, 8)).n == 8
    assert Selected().n == 5


def test_positions_select_by_position_not_source_index():
    """The CE plan's 8-room set: positions (0,1,2,3,5,6,7,8) drop source
    index 83 (position 4) and take 191 (position 8). Order preserved."""
    from curious_george.envs.layouts import resolve_layouts  # noqa: F401  (import parity)

    picked = Selected(impassable=True, positions=(0, 1, 2, 3, 5, 6, 7, 8))
    chosen = tuple(ROOMS_SELECTED[p] for p in picked.positions)
    assert len(chosen) == 8
    assert ROOMS_SELECTED[4] not in chosen          # source index 83 dropped
    assert chosen[4] == ROOMS_SELECTED[5]           # order preserved past the gap

    with pytest.raises(ValueError, match="positions"):
        Selected(positions=(0, 0))
    with pytest.raises(ValueError, match="positions"):
        Selected(positions=(99,))


# --- the extra landmark per room (2026-10-01) -------------------------------------------

DOT_ANCHORS = ("9,13", "13,2", "2,10", "9,10", "9,10", "2,9", "9,9", "2,13")
EIGHT = (0, 1, 2, 3, 5, 6, 7, 8)


def _resolve(source):
    return resolve_rooms(shape=EnvShape(), content=EnvContent(), source=source,
                         room_rules=RoomRules(), set_rules=RoomSetRules(), indices=None)


def test_extra_anchors_append_one_dot_per_room_and_change_nothing_else():
    plain = _resolve(Selected(positions=EIGHT, impassable=True))
    plus = _resolve(Selected(positions=EIGHT, impassable=True, extra_anchors=DOT_ANCHORS))
    assert len(plus) == len(plain) == 8
    for a, b, text in zip(plain, plus, DOT_ANCHORS):
        assert b.landmarks[:-1] == a.landmarks
        dot = b.landmarks[-1]
        assert dot.shape == "dot" and dot.color == "green" and dot.impassable
        assert dot.anchor == tuple(int(v) for v in text.split(",")) and dot.cells == (dot.anchor,)
        assert dot.anchor not in a.cells


def test_extra_anchors_must_align_with_positions_and_sit_on_free_floor():
    with pytest.raises(ValueError, match="align"):
        Selected(positions=EIGHT, extra_anchors=("9,13",))
    with pytest.raises(ValueError, match="floor|landmarks"):
        _resolve(Selected(positions=(0,), extra_anchors=("0,0",)))
    first = _resolve(Selected(positions=(0,)))[0]
    taken = first.landmarks[0].anchor
    with pytest.raises(ValueError, match="landmarks"):
        _resolve(Selected(positions=(0,), extra_anchors=(f"{taken[0]},{taken[1]}",)))


def test_the_dot_stencil_is_registered_for_configs():
    assert Landmark("dot", "red", (5, 5)).cells == ((5, 5),)


# --- the scattered novel object (2026-10-01) --------------------------------------------

def test_scattered_repeats_each_room_with_the_dot_somewhere_new_and_never_at_the_excluded_cell():
    from curious_george.envs.layouts import Scattered

    source = Scattered(positions=EIGHT, n_placements=16, seed=0, exclude=DOT_ANCHORS)
    rooms = _resolve(source)
    base = _resolve(Selected(positions=EIGHT, impassable=True))
    assert len(rooms) == source.n == 128
    floor = EnvShape().walkable
    for i, room in enumerate(base):
        copies = rooms[i * 16:(i + 1) * 16]
        dots = [c.landmarks[-1] for c in copies]
        assert all(c.landmarks[:-1] == room.landmarks for c in copies)
        assert len({d.anchor for d in dots}) == 16
        held_out = tuple(int(v) for v in DOT_ANCHORS[i].split(","))
        assert held_out not in {d.anchor for d in dots}
        for d in dots:
            assert d.shape == "dot" and d.impassable and d.anchor in floor and d.anchor not in room.cells
            x, y = d.anchor
            assert all((x + dx, y + dy) in floor for dx in (-1, 0, 1) for dy in (-1, 0, 1))
    again = _resolve(Scattered(positions=EIGHT, n_placements=16, seed=0, exclude=DOT_ANCHORS))
    assert [r.landmarks for r in again] == [r.landmarks for r in rooms]
    other = _resolve(Scattered(positions=EIGHT, n_placements=16, seed=1, exclude=DOT_ANCHORS))
    assert [r.landmarks for r in other] != [r.landmarks for r in rooms]
