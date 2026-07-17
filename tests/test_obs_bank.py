"""The obs bank must be BYTE-EQUAL to live get_frame renders, everywhere the
agent can actually be: random walk + resets, banked env vs plain render env.
"""

import numpy as np
import pytest
from minigrid.wrappers import RGBImgPartialObsWrapper_HD

import gymnasium as gym
from curious_george.envs.obs_bank import BANK_DIR, BankedRGBPartialObsWrapper

ENV_KEY = "MiniGrid-LRoom-v0"
SEED = 3
T = 300


def _raw(seed):
    env = gym.make(ENV_KEY, agent_start_pos=None, agent_start_dir=None,
                   render_mode="rgb_array")
    env.reset(seed=seed)
    return env


@pytest.fixture(scope="module")
def pair():
    banked = BankedRGBPartialObsWrapper(_raw(SEED), tile_size=1)
    live = RGBImgPartialObsWrapper_HD(_raw(SEED), tile_size=1)
    return banked, live


def test_byte_equality_over_random_walk(pair):
    banked, live = pair
    rng = np.random.default_rng(0)
    ob, _ = banked.reset()
    ol, _ = live.reset()
    assert np.array_equal(ob["image"], ol["image"])
    for t in range(T):
        if t % 60 == 59:  # periodic resets, like seqdur episode cuts
            ob, _ = banked.reset()
            ol, _ = live.reset()
        else:
            a = int(rng.integers(0, 3))
            ob = banked.step(a)[0]
            ol = live.step(a)[0]
        assert np.array_equal(ob["image"], ol["image"]), f"t={t}"
        assert ob["direction"] == ol["direction"]
        assert ob["mission"] == ol["mission"]


def test_bank_persisted_and_readonly(pair):
    banked, _ = pair
    fp = banked._grid_fingerprint()
    assert banked._bank_path(fp).exists(), "bank should be saved under data/obs_bank"
    obs, _ = banked.reset()
    with pytest.raises(ValueError):
        obs["image"][0, 0, 0] = 0  # served read-only: mutation must raise
