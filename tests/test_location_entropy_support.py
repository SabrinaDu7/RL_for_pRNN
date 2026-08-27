"""`loc_entropy` is normalised over cells that exist in SOME room, not one room.

Entropy over visited locations is only meaningful against the set of cells the
agent could have been in. That was one set, read off the eval shell, and it was
right only because landmarks were walkable and every room shared it.

Impassable landmarks break that per room, and the visits reaching
`LocationStats` are pooled across streams in DIFFERENT rooms. Masking a pooled
grid with any one room's mask counts cells that were blocked where the agent
actually was and drops cells that were open there. It does not crash - it logs
a series that looks comparable to every earlier run and is not, which is the
worst way for a metric to be wrong.

These pin the support, and pin that the existing series is unbroken.
"""

import numpy as np
import pytest

from curious_george.configs import EnvBackend, EnvCfg, EvalKind
from curious_george.envs.layouts import (
    BASE_ROOM_ID,
    EnvContent,
    LandmarkKind,
    RoomSetRules,
    Uniform,
    Vary,
    base_walkable,
)
from tests.small_config import small_config

SHAPES = ("x", "plus", "block3")


def _algo(*, impassable: bool | None, n_rooms: int = 3, varies=(Vary.POSITION,)):
    """`impassable=None` is the single-room control - no room set at all."""
    from curious_george.training.setup import setup_training

    env = (
        EnvCfg()
        if impassable is None
        else EnvCfg(
            content=EnvContent(
                kinds=tuple(LandmarkKind(s, impassable=impassable) for s in SHAPES)
            ),
            source=Uniform(n=n_rooms, seed=7),
            set_rules=RoomSetRules(varies=frozenset(varies)),
        )
    )
    cfg = small_config(
        num_envs=4,
        backend=EnvBackend.DEVICE,
        env=env,
        evals=frozenset() if impassable is None else frozenset({EvalKind.SPATIAL_MULTIROOM}),
    )
    return setup_training(cfg).algo


def _mask_cells(algo) -> set:
    env = algo.env
    return {
        (x, y)
        for y in range(env.height)
        for x in range(env.width)
        if algo.loc_mask[y * env.width + x]
    }


def test_walkable_rooms_keep_the_historical_support():
    """The bitwise-compatibility claim. Every run before 2026-08-27 has rooms
    that share a walkable set, and their series must not move."""
    assert _mask_cells(_algo(impassable=False)) == base_walkable(BASE_ROOM_ID)


def test_the_single_room_control_is_untouched():
    assert _mask_cells(_algo(impassable=None)) == base_walkable(BASE_ROOM_ID)


def test_impassable_rooms_keep_every_cell_open_in_ANY_room():
    """A cell blocked in one room but open in another is somewhere the agent
    genuinely goes. Dropping it understates the support and inflates the
    entropy - the half of the bug a single room's mask causes by omission.

    Asserted as a RELATIONSHIP, not a cell count. An earlier version of this
    test asserted `union == base`, which was true only while
    `min_anchor_separation` was 6 and kept objects far apart; once they were
    allowed within 3, two cells came to be blocked in all three rooms and the
    union legitimately shrank. The rule is the relationship, not the number.
    """
    algo = _algo(impassable=True)
    base = base_walkable(BASE_ROOM_ID)
    rooms = algo.envs.layouts
    union = set().union(*(r.walkable(base) for r in rooms))

    assert _mask_cells(algo) == union
    assert union <= base
    for room in rooms:
        assert room.walkable(base) < union, (
            "every room hides cells that some other room leaves open, so the "
            "union must be strictly larger than any single room's set"
        )


def test_no_room_alone_would_have_given_the_right_answer():
    """Without this the test above could pass by accident on a pool whose rooms
    happen to agree, and the bug would be untested."""
    algo = _algo(impassable=True)
    base = base_walkable(BASE_ROOM_ID)
    per_room = [r.walkable(base) for r in algo.envs.layouts]
    assert all(set(w) != _mask_cells(algo) for w in per_room)


def test_a_cell_blocked_in_every_room_is_excluded():
    """The other half of the bug, exercised by holding positions fixed so every
    room blocks the same cells. The agent can never stand there, so counting
    them as support understates the entropy for a reason that is not behaviour.
    """
    algo = _algo(impassable=True, varies=())
    base = base_walkable(BASE_ROOM_ID)
    rooms = algo.envs.layouts
    everywhere = set.intersection(*(set(r.cells) for r in rooms))
    assert everywhere, "nothing varies, so all rooms must block the same cells"
    assert not (everywhere & _mask_cells(algo))
    assert _mask_cells(algo) == base - everywhere


def test_the_entropy_actually_uses_that_support():
    """The mask is only worth fixing if `LocationStats` reads it. Feed a uniform
    visit over the support and the entropy must be log2 of its size."""
    algo = _algo(impassable=True)
    cells = sorted(_mask_cells(algo))
    entropy, _ = algo.loc_stats.update(cells)
    assert entropy == pytest.approx(np.log2(len(cells)), rel=1e-9)


# --- the ceiling is a property of the world, quotable without a run ---------


def test_the_ceiling_is_quotable_from_a_config_alone():
    """It has to be readable without training, or a cross-arm comparison means
    re-deriving the reachable count by hand from the config."""
    walkable = EnvCfg()
    assert walkable.reachable_cells == base_walkable(BASE_ROOM_ID)
    assert walkable.loc_entropy_ceiling == pytest.approx(np.log2(172))


def test_objects_lower_the_ceiling_only_where_they_block_every_room():
    """The number that makes an objects arm comparable to its control.

    Varying the positions puts the ceiling near the empty room's, because a
    cell blocked in one room is usually open in another; holding them fixed
    drops it by the full 19 cells. Stated as an ordering plus the exact fixed
    case, because the varying number depends on how much the sampled rooms
    happen to overlap - which is a property of the pool, not a constant.
    """
    varying = EnvCfg(
        content=EnvContent(kinds=tuple(LandmarkKind(s, impassable=True) for s in SHAPES)),
        source=Uniform(n=3, seed=7),
        set_rules=RoomSetRules(varies=frozenset({Vary.POSITION})),
    )
    fixed = EnvCfg(
        content=EnvContent(kinds=tuple(LandmarkKind(s, impassable=True) for s in SHAPES)),
        source=Uniform(n=3, seed=7),
        set_rules=RoomSetRules(varies=frozenset()),
    )
    assert fixed.loc_entropy_ceiling == pytest.approx(np.log2(172 - 19))
    assert fixed.loc_entropy_ceiling < varying.loc_entropy_ceiling
    assert varying.loc_entropy_ceiling <= np.log2(172) + 1e-9
    assert varying.loc_entropy_ceiling == pytest.approx(
        np.log2(len(varying.reachable_cells))
    ), "the ceiling must be log2 of the support it is quoted for"


def test_the_run_and_the_config_agree_on_the_support():
    """`algo._location_mask` and `EnvCfg.reachable_cells` must not drift into
    two answers; they now share `layouts.pooled_walkable`."""
    cfg_env = EnvCfg(
        content=EnvContent(kinds=tuple(LandmarkKind(s, impassable=True) for s in SHAPES)),
        source=Uniform(n=3, seed=7),
        set_rules=RoomSetRules(varies=frozenset({Vary.POSITION})),
    )
    algo = _algo(impassable=True)
    assert _mask_cells(algo) == cfg_env.reachable_cells
