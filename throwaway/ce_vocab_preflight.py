"""CE preflight: enumerate the exact per-tile RGB vocabulary of the obs banks.

Throwaway (results feed a design decision, not a result document): the CE
class alphabet must be an explicit, closed set - this measures it over every
bank the plan's runs will touch. The committed derivation will live with the
CE code; this script exists to check the intuition BEFORE the fork work.

    uv run python throwaway/ce_vocab_preflight.py
"""
import numpy as np
import gymnasium as gym

from curious_george.envs.layouts import (
    BASE_ROOM_ID, MULTI_ROOM_ID, ROOMS_SELECTED, with_affordance,
)
from curious_george.envs.obs_bank import BankedRGBPartialObsWrapper

counts: dict[tuple, int] = {}


def absorb(bank: np.ndarray) -> None:
    flat = bank.reshape(-1, 3)
    uniq, c = np.unique(flat, axis=0, return_counts=True)
    for row, n in zip(uniq, c):
        key = tuple(int(v) for v in row)
        counts[key] = counts.get(key, 0) + int(n)


# every Selected room, both affordances
env = gym.make(MULTI_ROOM_ID[BASE_ROOM_ID], landmarks=list(ROOMS_SELECTED[0].landmarks))
env.reset(seed=0)
w = BankedRGBPartialObsWrapper(env, tile_size=1)
for imp in (False, True):
    for layout in with_affordance(ROOMS_SELECTED, impassable=imp):
        w.unwrapped.landmarks = list(layout.landmarks)
        env.reset(seed=0)
        w._ensure_bank()
        absorb(np.asarray(w._bank))

# the default single L-room (triangle6/plus6/x6)
base = gym.make(BASE_ROOM_ID)
base.reset(seed=0)
wb = BankedRGBPartialObsWrapper(base, tile_size=1)
wb._ensure_bank()
absorb(np.asarray(wb._bank))

total = sum(counts.values())
print(f"{len(counts)} distinct tile RGB values over {total:,} tile observations:")
for rgb, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {str(rgb):18s} {n:>12,}  {100 * n / total:6.2f}%")
