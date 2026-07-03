"""Reward terms for the curious agent.

The total reward fed to GAE is
    r[i] = rewards[i] + k_int * int_rewards[i] + k_curious * curious_rewards[i]
(see curious_george.rl.buffer.compute_gae).

Curiosity reward time alignment
-------------------------------
`reward_alignment="legacy"` (current and only implemented mode): the curiosity
reward at step i is the pRNN's error reconstructing obss[i] - the observation
the agent saw BEFORE taking action i. This credits the action with surprise it
did not cause. The planned `"next_obs"` mode (refactor plan Phase 3) credits
action i with the prediction error on obss[i+1], the observation the action
produced. Kept legacy-only here so the Phase 2 extraction is bitwise
behavior-preserving against tests/golden/golden_v0.pt.
"""

import numpy as np
import torch

from curious_george.world_model.adapter import PRNNAdapter


def compute_curious_rewards(
    adapter: PRNNAdapter,
    obss: list,
    actions_np: np.ndarray,
    done_indices: list[int],
    last_observations: list,
    num_frames: int,
) -> torch.Tensor:
    """Curiosity reward = per-step pRNN observation-prediction MSE (legacy
    alignment, see module docstring)."""
    return adapter.prediction_mses(
        obss=obss,
        actions_np=actions_np,
        done_indices=done_indices,
        last_observations=last_observations,
        num_frames=num_frames,
    )
