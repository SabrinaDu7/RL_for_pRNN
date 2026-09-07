"""Science bookkeeping observed during rollout collection.

Pure functions + a small persistent-state dataclass, so the rollout loop in
collector.py contains no analysis logic. Nothing here consumes RNG, so the
extraction is bitwise-neutral.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import entropy


@dataclass
class LocationStats:
    """Persistent location-visit state (loc_history is a 5-collect window)."""

    loc_mask: list
    grid_shape: tuple[int, int]
    loc_visits: np.ndarray = field(init=False)
    loc_history: list = field(init=False)

    def __post_init__(self):
        self.loc_visits = np.zeros(self.grid_shape)
        self.loc_history = [np.zeros(np.sum(self.loc_mask))] * 5

    def update(self, locs: list) -> tuple[float, float]:
        """Accumulate this rollout's visits; return (loc_entropy, loc_entropy_5)."""
        # One scatter-add instead of a Python loop over B*T visits (65,536 per
        # production rollout). Integer counts, so the order of accumulation
        # cannot change the result.
        xy = np.asarray(locs, dtype=np.intp).reshape(-1, 2)
        np.add.at(self.loc_visits, (xy[:, 0], xy[:, 1]), 1)
        self.loc_visits = self.loc_visits.flatten("F")[self.loc_mask]
        loc_entropy = entropy(self.loc_visits, base=2)

        self.loc_history.pop(0)
        self.loc_history.append(self.loc_visits)
        loc_entropy_5 = entropy(np.sum(self.loc_history, axis=0), base=2)
        self.loc_visits = np.zeros(self.grid_shape)
        return loc_entropy, loc_entropy_5


def new_joint_probabilities(env, act_dim: int) -> np.ndarray:
    """(HD, x, y, action) accumulator of policy probabilities per state."""
    return np.zeros(
        (getattr(env, "numHDs"), env.width, env.height, act_dim),
        dtype=np.float32,
    )
