"""The exploration metrics ride the REAL rollout.

`tests/test_exploration_evals.py` gates the metric functions on walkers with
known answers; these gate the WIRING: the pool reports which room each episode
ran in, the collector exports the episode view of its positions, and one
collect lands the scalars in the logs - with the room ids provably the rooms
the streams were actually stepped in.
"""

import dataclasses

import numpy as np
import pytest
import torch

from curious_george.configs import EnvBackend, EnvCfg, EvalKind
from curious_george.envs.layouts import (
    BASE_ROOM_ID,
    EnvContent,
    EnvShape,
    LandmarkKind,
    RoomSetRules,
    Uniform,
    Vary,
    base_walkable,
)
from tests.small_config import small_config

SHAPES = ("x", "plus", "block3")


def _impassable_env() -> EnvCfg:
    return EnvCfg(
        content=EnvContent(
            kinds=tuple(LandmarkKind(s, impassable=True) for s in SHAPES)
        ),
        source=Uniform(n=3, seed=7),
        set_rules=RoomSetRules(varies=frozenset({Vary.POSITION})),
    )


def _multi_room_cfg():
    return small_config(
        backend=EnvBackend.DEVICE,
        env=_impassable_env(),
        evals=frozenset({EvalKind.SPATIAL_MULTIROOM}),
    )


# --- the pool reports the room schedule --------------------------------------


def test_prepare_resets_reports_the_room_schedule():
    """Row 0 is the assignment the rollout enters with; prepared reset i names
    segment i+1's room; the host mirror tracks every overwrite of the live
    `stream_layout`; and the final applied reset is the NEXT rollout's row 0."""
    from curious_george.training.setup import setup_envs

    pool = setup_envs(_multi_room_cfg())
    pool.reset_all()
    entry = pool.stream_layout_host.copy()
    schedule = pool.prepare_resets(count=3)
    assert schedule.shape == (3, pool.B)
    assert np.array_equal(schedule[0], entry)
    for i in range(3):
        pool.apply_prepared_reset(index=i)
        applied = pool.stream_layout.cpu().numpy()
        assert np.array_equal(pool.stream_layout_host, applied)
        if i < 2:
            assert np.array_equal(schedule[i + 1], applied)
    assert np.array_equal(pool.prepare_resets(count=1)[0], applied)


# --- one real collect carries the episode view and its scores ----------------


def test_collect_exports_episode_view_and_scores_it():
    """The wiring gate. The strong claim is the support check: every episode's
    positions lie inside the walkable set of the room its EXPORTED id names.
    A wrong room channel fails this (and `visitation_by_room` inside the
    collect would already have raised)."""
    from curious_george.training.setup import setup_training

    cfg = _multi_room_cfg()
    comps = setup_training(cfg)
    algo = comps.algo
    _, logs = algo.collect_experiences()

    E = cfg.collect.num_envs * cfg.collect.episodes_per_env
    assert algo.positions_episodes.shape == (E, cfg.collect.episode_steps, 2)
    assert algo.segment_layouts.shape == (
        cfg.collect.episodes_per_env,
        cfg.collect.num_envs,
    )

    base = base_walkable(BASE_ROOM_ID)
    rooms = algo.segment_layouts.reshape(-1)
    for e in range(E):
        cells = {tuple(map(int, p)) for p in algo.positions_episodes[e].cpu()}
        assert cells <= comps.envs.layouts[rooms[e]].walkable(base), (
            f"episode {e} left the walkable set of its exported room {rooms[e]}"
        )

    for key in (
        "exploration/coverage",
        "exploration/nauc",
        "exploration/room_entropy_norm",
        "exploration/t50_reached",
        "exploration/t90_reached",
    ):
        assert key in logs, key
    assert 0 < logs["exploration/coverage"] <= 1
    assert 0 <= logs["loc_entropy_norm"] <= 1


def test_non_device_backends_omit_rather_than_fabricate():
    """No episode view off the device backend: the fields are None and the
    exploration keys ABSENT - a missing series cannot be misread."""
    from curious_george.training.setup import setup_training

    algo = setup_training(small_config(backend=EnvBackend.SERIAL_TABLE)).algo
    _, logs = algo.collect_experiences()
    assert algo.positions_episodes is None
    assert algo.segment_layouts is None
    assert not any(k.startswith("exploration/") for k in logs)
    assert "loc_entropy_norm" in logs  # derived from loc_entropy, backend-free


def test_the_logged_scalars_reach_wandb_keys():
    """`build_update_log` forwards exploration keys for EVERY agent - the
    random baselines exist to be compared on exactly these series."""
    from curious_george.training.logging import UpdateStats, build_update_log

    logs = {
        "num_episodes": 1, "entropy": 0.0, "loc_entropy": 5.0,
        "loc_entropy_5": 5.0, "loc_entropy_norm": 0.7,
        "exploration/coverage": 0.4, "exploration/t90_reached": 0.0,
    }
    stats = UpdateStats(num_frames=1, fps=1.0, duration=1, random_agent=True)
    out = build_update_log(logs, stats, mi_policy=None)
    assert out["exploration/coverage"] == 0.4
    assert out["exploration/t90_reached"] == 0.0
    assert out["loc_entropy_norm"] == 0.7
