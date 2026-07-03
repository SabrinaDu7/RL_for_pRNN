"""Rollout buffer utilities.

For now this holds the GAE computation as a pure in-place function over the
algo's preallocated [T] tensors. The RolloutBuffer class generalizing these
to [T, B] arrives with parallel-env support (refactor plan Phase 5).
"""

import torch


def compute_gae(
    *,
    advantages: torch.Tensor,
    rewards: torch.Tensor,
    int_rewards: torch.Tensor,
    curious_rewards: torch.Tensor,
    values: torch.Tensor,
    masks: torch.Tensor,
    final_next_value: torch.Tensor,
    final_mask,
    num_frames: int,
    discount: float,
    gae_lambda: float,
    k_int: float,
    k_curious: float,
) -> None:
    """n-step TD generalized advantage estimates, written into `advantages`.

    Index conventions preserved from the original loop: masks[i] is the mask
    recorded BEFORE step i's transition, so next_mask for step i is
    masks[i+1] (or the live post-rollout mask for the last step).
    """
    next_value = final_next_value
    for i in reversed(range(num_frames)):
        next_mask = masks[i + 1] if i < num_frames - 1 else final_mask
        next_value = values[i + 1] if i < num_frames - 1 else next_value
        next_advantage = advantages[i + 1] if i < num_frames - 1 else 0

        reward_term = (
            rewards[i]
            + k_int * int_rewards[i]
            + k_curious * curious_rewards[i]
        )
        delta = (
            reward_term + discount * next_value * next_mask - values[i]
        )
        advantages[i] = (
            delta + discount * gae_lambda * next_advantage * next_mask
        ) # n-step TD generalized advantage estimates
