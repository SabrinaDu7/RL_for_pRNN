"""`envs/palette.py::TILE_VOCABULARY` is the committed measurement, and this
regenerates it: the unique tile values over every Selected-room bank (both
affordances) plus the default single L-room must equal the table exactly -
no missing class, no unseen value, no drifted byte.
"""

import gymnasium as gym
import numpy as np

from curious_george.envs.layouts import (
    BASE_ROOM_ID,
    MULTI_ROOM_ID,
    ROOMS_SELECTED,
    with_affordance,
)
from curious_george.envs.obs_bank import BankedRGBPartialObsWrapper
from curious_george.envs.palette import TILE_VOCABULARY


def test_vocabulary_matches_the_live_banks(tmp_path):
    seen: set[tuple[int, int, int]] = set()

    def absorb(bank: np.ndarray) -> None:
        for row in np.unique(bank.reshape(-1, 3), axis=0):
            seen.add(tuple(int(v) for v in row))

    env = gym.make(
        MULTI_ROOM_ID[BASE_ROOM_ID], landmarks=list(ROOMS_SELECTED[0].landmarks)
    )
    env.reset(seed=0)
    wrapper = BankedRGBPartialObsWrapper(env, tile_size=1)
    for impassable in (False, True):
        for layout in with_affordance(ROOMS_SELECTED, impassable=impassable):
            wrapper.unwrapped.landmarks = list(layout.landmarks)
            env.reset(seed=0)
            wrapper._ensure_bank()
            absorb(np.asarray(wrapper._bank))

    base = gym.make(BASE_ROOM_ID)
    base.reset(seed=0)
    base_wrapper = BankedRGBPartialObsWrapper(base, tile_size=1)
    base_wrapper._ensure_bank()
    absorb(np.asarray(base_wrapper._bank))

    committed = set(TILE_VOCABULARY.values())
    assert seen == committed, (
        f"unseen-in-table: {sorted(seen - committed)}; "
        f"table-not-in-banks: {sorted(committed - seen)}"
    )
