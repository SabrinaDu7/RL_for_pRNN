"""AsyncShellPool must reproduce the serial list-of-envs transitions exactly:
same construction seeds + same reset chain + deterministic env stepping.
"""

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from pathlib import Path

from curious_george.envs.vector import AsyncShellPool
from curious_george.training.setup import setup_envs

REPO = Path(__file__).resolve().parents[1]
B = 4
T = 30


@pytest.fixture(scope="module")
def cfg():
    with initialize_config_dir(config_dir=str(REPO / "Configs"), version_base=None):
        return compose(
            config_name="main",
            overrides=[
                f"exp.num_envs={B}",
                "exp.async_envs=true",  # the pool is opt-in; this tests it
                "logging.wandb_log=false",
            ],
        )


def _serial_envs(cfg):
    cfg2 = cfg.copy()
    cfg2.exp.async_envs = False
    return setup_envs(cfg2)


def test_pool_matches_serial_transitions(cfg):
    pool = setup_envs(cfg)
    assert isinstance(pool, AsyncShellPool)
    serial = _serial_envs(cfg)

    rng = np.random.default_rng(0)
    try:
        pool_obs, pool_locs = pool.reset_all()
        ser_obs = [e.reset() for e in serial]
        for b in range(B):
            assert np.array_equal(pool_obs[b]["image"], ser_obs[b]["image"])
            assert pool_obs[b]["direction"] == ser_obs[b]["direction"]
            assert np.array_equal(pool_locs[b], np.asarray(serial[b].get_agent_pos()))

        for t in range(T):
            acts = rng.integers(0, 4, size=B)
            pool_obs, pool_rew, _, pool_locs = pool.step(acts)
            for b in range(B):
                s_obs, s_rew, _, _, _ = serial[b].step(acts[b : b + 1])
                assert np.array_equal(pool_obs[b]["image"], s_obs["image"]), f"t={t} b={b}"
                assert pool_obs[b]["direction"] == s_obs["direction"]
                assert pool_rew[b] == s_rew
                assert np.array_equal(pool_locs[b], np.asarray(serial[b].get_agent_pos()))

        # synchronized reset (the seqdur episode cut) continues the same
        # unseeded RNG chain as per-env serial resets
        pool_obs, pool_locs = pool.reset_all()
        ser_obs = [e.reset() for e in serial]
        for b in range(B):
            assert np.array_equal(pool_obs[b]["image"], ser_obs[b]["image"])
            assert np.array_equal(pool_locs[b], np.asarray(serial[b].get_agent_pos()))
    finally:
        pool.close()


def test_pool_static_attrs_and_indexing(cfg):
    pool = setup_envs(cfg)
    try:
        assert len(pool) == B
        assert pool.numHDs == 4 and pool.width > 0 and pool.height > 0
        assert pool[0] is pool.eval_shell
        with pytest.raises(IndexError):
            pool[1]
    finally:
        pool.close()
