"""The cached-grid fast reset is the full reset, only cheaper.

The 2026-08-31 speedup replaced per-episode full `env.reset()` with a cached
pre-painted grid plus the same `place_agent()` draw. The claim it rests on -
bitwise-identical RNG consumption - was verified by a manual STATE_SHA A/B at
landing; these tests are that A/B as a permanent gate, plus the audit fixes
around it: cached grids stay immutable, walkable pools fall back to the full
reset, the wrapper's stale bank fingerprint is invalidated, `run.seed`
actually reaches stream 0 (bank collection used to reseed it to a constant),
and a prepare-thread crash surfaces instead of hanging the collector.
"""

import numpy as np
import pytest
import torch

from curious_george.configs import EnvBackend, EnvCfg, EvalKind
from curious_george.envs.layouts import (
    EnvContent,
    LandmarkKind,
    RoomSetRules,
    Uniform,
    Vary,
)
from tests.small_config import small_config

SHAPES = ("x", "plus", "block3")


def _cfg(*, impassable: bool, seed: int = 2):
    return small_config(
        num_envs=4,
        backend=EnvBackend.DEVICE,
        seed=seed,
        env=EnvCfg(
            content=EnvContent(
                kinds=tuple(LandmarkKind(s, impassable=impassable) for s in SHAPES)
            ),
            source=Uniform(n=3, seed=7),
            set_rules=RoomSetRules(varies=frozenset({Vary.POSITION})),
        ),
        evals=frozenset({EvalKind.SPATIAL_MULTIROOM}),
    )


def _pool(*, impassable: bool, seed: int = 2):
    from curious_george.training.setup import setup_envs

    return setup_envs(_cfg(impassable=impassable, seed=seed))


ROUNDS = 6  # several reset rounds so schedule and RNG drift can be observed


def test_fast_reset_is_bitwise_equal_to_full_reset():
    fast = _pool(impassable=True)
    full = _pool(impassable=True)
    full._layout_grids = False  # forces the pre-speedup full-reset path
    for _ in range(ROUNDS):
        f_layouts, f_pos, f_dir = fast._reset_streams()
        e_layouts, e_pos, e_dir = full._reset_streams()
        assert np.array_equal(f_layouts, e_layouts)
        assert np.array_equal(f_pos, e_pos)
        assert np.array_equal(f_dir, e_dir)
    assert isinstance(fast._layout_grids, list), "the fast path never engaged"


def test_fast_reset_does_not_mutate_the_cached_grids():
    pool = _pool(impassable=True)
    pool._reset_streams()  # builds the cache
    before = [g.encode().copy() for g in pool._layout_grids]
    for _ in range(ROUNDS):
        pool._reset_streams()
    for grid, snapshot in zip(pool._layout_grids, before):
        assert np.array_equal(grid.encode(), snapshot)


def test_walkable_layouts_fall_back_to_the_full_reset():
    """Walkable rooms place the agent BEFORE painting (Lroom's historical
    order), so a pre-painted cached grid would move every trajectory."""
    pool = _pool(impassable=False)
    pool._reset_streams()
    assert pool._layout_grids is False


def test_fast_reset_invalidates_the_wrapper_bank_fingerprint():
    """After a fast reset the wrapper's cached bank describes the room of the
    last FULL reset; a stale fingerprint would serve it silently."""
    pool = _pool(impassable=True)
    pool._reset_streams()
    assert all(w._fingerprint is None for w in pool._wrappers)


def test_run_seed_reaches_stream_zero():
    """Bank collection used to call reset(seed=0) on stream 0's LIVE wrapper,
    so after pool construction its np_random held the seed-0 state whatever
    run.seed was (audit 2026-08-31). The room SCHEDULE still varied with
    run.seed (layout_seed), so downstream positions could not see this - the
    gate has to read the generator state itself, not draws through it."""
    state = {
        seed: repr(
            _pool(impassable=True, seed=seed)
            ._wrappers[0].unwrapped._np_random.bit_generator.state
        )
        for seed in (2, 3)
    }
    assert state[2] != state[3], (
        "stream 0's RNG state after construction is identical across run "
        "seeds - bank collection is reseeding the live wrapper again"
    )


def test_prepare_thread_error_surfaces(monkeypatch):
    """The reset schedule is prepared on a worker thread; a crash there must
    raise in the collector at join, not hang or be swallowed."""
    from curious_george.envs.vector import DeviceTableShellPool
    from curious_george.training.setup import setup_training

    def boom(self, *, count):
        raise RuntimeError("prepare_resets exploded on the worker thread")

    monkeypatch.setattr(DeviceTableShellPool, "prepare_resets", boom)
    algo = setup_training(_cfg(impassable=True)).algo
    try:
        with pytest.raises(RuntimeError, match="worker thread"):
            algo.collect_experiences()
    finally:
        algo.envs.close()
