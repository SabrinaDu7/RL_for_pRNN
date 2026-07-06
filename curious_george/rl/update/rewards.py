"""Reward terms for the curious agent.

The total reward fed to GAE is
    r[i] = rewards[i] + k_int * int_rewards[i] + k_curious * curious_rewards[i]
(see curious_george.rl.buffer.compute_gae).

Curiosity reward time alignment
-------------------------------
`reward_alignment="legacy"` (default, bitwise-pinned by
tests/golden/golden_v0.pt): the curiosity reward at step i is the pRNN's
error reconstructing obss[i] - the observation the agent saw BEFORE taking
action i, crediting the action with surprise it did not cause.
`"next_obs"` credits action i with the prediction error on the observation it
produced - uniformly for every action, including the last of each episode
(the adapter extends the per-episode predict pass by one zero-action step so
the final observation is a real prediction target; no boundary special case).
"""

import numpy as np
import torch

from curious_george.world_model.adapter import PRNNAdapter


REWARD_ALIGNMENTS = {"legacy": 0, "next_obs": 1}  # name -> prediction target offset


def compute_curious_rewards(
    adapter: PRNNAdapter,
    obss: list,
    actions_np: np.ndarray,
    done_indices: list[int],
    last_observations: list,
    num_frames: int,
    alignment: str = "legacy",
) -> torch.Tensor:
    """Curiosity reward = per-step pRNN observation-prediction MSE.

    alignment="legacy": MSEs[i] is the error on the PRE-action obs.
    alignment="next_obs": MSEs[i] is the error on the obs action i produced
    (see module docstring).
    """
    assert alignment in REWARD_ALIGNMENTS, f"unknown reward_alignment {alignment!r}"
    return adapter.prediction_mses(
        obss=obss,
        actions_np=actions_np,
        done_indices=done_indices,
        last_observations=last_observations,
        num_frames=num_frames,
        target_offset=REWARD_ALIGNMENTS[alignment],
    )
