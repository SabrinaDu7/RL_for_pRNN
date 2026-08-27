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

    With objects at DIFFERENT positions, no cell is blocked in every room, so
    the correct support is the whole room. That is the healthy case and worth
    stating outright: each room individually hides 19 cells, and only the union
    puts back the ones the agent demonstrably reaches in some other room.
    """
    algo = _algo(impassable=True)
    base = base_walkable(BASE_ROOM_ID)
    rooms = algo.envs.layouts
    union = set().union(*(r.walkable(base) for r in rooms))
    assert _mask_cells(algo) == union == base
    assert all(r.walkable(base) < base for r in rooms), (
        "each room individually blocks cells; only the union restores them"
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
