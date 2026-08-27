"""The offline checkpoint scorer reads a `Config`, and those reads are checked.

WHY THIS FILE EXISTS. `checkpoint_series.py` survived the 2026-08-26 config
cutover with four reads still in the old spelling - `env.base_room`,
`env.layouts`, `env.eval_rooms_max`, and `.value` on `env_name`, which is now a
plain `str`. Every one of them raises. Nothing caught it because reaching them
meant running `main()`, which needs a run directory full of checkpoints, so no
test imported the module at all.

The fix was to move those reads into `SeriesContext`, which resolves from a
`Config` alone. These tests are what makes that worth having: they fail on the
next field rename instead of at 2 a.m. against a finished cluster run.
"""

import json

import pytest

from curious_george.evaluation.checkpoint_series import (
    ROOMS,
    SOURCES,
    SeriesContext,
    archived,
    run_config,
)
from curious_george.envs.layouts import ROOMS_RUN1, ROOMS_SQUARE


@pytest.mark.parametrize("room", sorted(ROOMS))
@pytest.mark.parametrize("source", SOURCES)
def test_every_config_read_resolves(room, source):
    """The regression this file was written for.

    Each of these attribute reads raised before the fix; the whole point is that
    they are reachable without a checkpoint directory.
    """
    cfg = run_config(room=room, source=source, hiddensize=500)
    ctx = SeriesContext.resolve(cfg, rooms_scored=None)

    assert isinstance(ctx.env_name, str) and ctx.env_name.startswith("MiniGrid-")
    assert ctx.room == ROOMS[room]
    assert 1 <= len(ctx.rooms) <= ctx.n_layouts


def _frozen(room):
    return SeriesContext.resolve(
        run_config(room=room, source="frozen", hiddensize=500), rooms_scored=None
    )


def test_the_committed_set_follows_the_room():
    """Handing the L-room's rooms to a square room is silently wrong, and it is
    what the previous branch-on-room-id could do. `Frozen` makes it impossible."""
    l_ctx, sq_ctx = _frozen("lroom"), _frozen("squareroom")
    assert [r.key for r in l_ctx.rooms] == [r.key for r in ROOMS_RUN1[: len(l_ctx.rooms)]]
    assert [r.key for r in sq_ctx.rooms] == [r.key for r in ROOMS_SQUARE[: len(sq_ctx.rooms)]]
    assert [r.anchors for r in l_ctx.rooms] != [r.anchors for r in sq_ctx.rooms]


def test_the_two_committed_sets_share_their_FIRST_room():
    """A recorded fact with teeth, not an endorsement.

    `--source one` is `indices=(0,)`, so the single-room control is the SAME
    anchor triple in the L-room and the square arm - they differ only in walls.
    Anyone reading a difference between those two arms as an effect of the
    LAYOUT would be wrong. Pinned so that if the sets are ever re-derived, this
    fails and the reading gets revisited.
    """
    assert _frozen("lroom").rooms[0].anchors == _frozen("squareroom").rooms[0].anchors


def test_one_is_the_single_room_control():
    assert len(SeriesContext.resolve(
        run_config(room="lroom", source="one", hiddensize=500), rooms_scored=None).rooms) == 1


def test_rooms_scored_is_a_fixed_prefix_and_defaults_to_the_config():
    """A PREFIX, not a sample: the series is compared across checkpoints, so the
    rooms scored must not move between calls."""
    cfg = run_config(room="lroom", source="uniform", hiddensize=500)
    a = SeriesContext.resolve(cfg, rooms_scored=3)
    b = SeriesContext.resolve(cfg, rooms_scored=3)
    assert [r.key for r in a.rooms] == [r.key for r in b.rooms]
    assert len(a.rooms) == 3

    default = SeriesContext.resolve(cfg, rooms_scored=None)
    assert len(default.rooms) == cfg.eval.rooms_max


def test_meta_is_json_serialisable_and_names_the_run():
    """`checkpoint_curve.json` is written from this. A curve of sRSA against
    step is uninterpretable without knowing which room set produced it."""
    ctx = SeriesContext.resolve(run_config(room="lroom", source="frozen", hiddensize=500),
                                rooms_scored=None)
    blob = json.loads(json.dumps(ctx.meta()))
    assert blob["env_name"] == ctx.env_name
    assert blob["room_id"] == ctx.room
    assert blob["n_rooms_scored"] == len(ctx.rooms)
    assert blob["room_keys"]


def test_archived_orders_by_environment_step(tmp_path):
    """Checkpoints are named by env step, zero-padded so a lexical sort IS a
    numeric sort. Reading them out of order would plot the series backwards."""
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    for step in (10_485_760, 1_048_576, 104_857_600):
        (ckpts / f"predictiveNet_state_step{step:010d}.pt").touch()
    assert [s for s, _ in archived(tmp_path)] == [1_048_576, 10_485_760, 104_857_600]


def test_no_rooms_is_an_error_not_an_empty_series():
    """Scoring zero rooms would print a header and no rows, which reads as a
    finished run that found nothing."""
    from dataclasses import replace

    from curious_george.configs import EnvDefault

    cfg = run_config(room="lroom", source="frozen", hiddensize=500)
    # EnvDefault means "the env's own landmarks", so there is no room SET.
    no_set = replace(cfg, env=replace(cfg.env, source=EnvDefault()),
                     eval=replace(cfg.eval, evals=frozenset()))
    with pytest.raises(ValueError, match="no rooms"):
        SeriesContext.resolve(no_set, rooms_scored=None)
