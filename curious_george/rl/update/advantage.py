"""Rollout buffer utilities.

For now this holds the GAE computation as a pure in-place function over the
algo's preallocated [T] tensors. The RolloutBuffer class generalizing these
to [T, B] arrives with parallel-env support (refactor plan Phase 5).
"""

import numpy as np
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

    The recurrence runs in float32 numpy (identical IEEE ops / op order to the
    historical per-element tensor loop; masks are exactly 0/1 so the one
    reassociated multiply is exact) - per-element indexing of device tensors
    in a Python loop was a measured hotspot.
    """
    rewards_np = rewards.detach().cpu().numpy()
    int_np = int_rewards.detach().cpu().numpy()
    cur_np = curious_rewards.detach().cpu().numpy()
    values_np = values.detach().cpu().numpy()
    masks_np = masks.detach().cpu().numpy()

    next_values = np.empty_like(values_np)
    next_values[:-1] = values_np[1:]
    next_values[-1] = float(final_next_value)
    next_masks = np.empty_like(masks_np)
    next_masks[:-1] = masks_np[1:]
    next_masks[-1] = final_mask

    reward_term = rewards_np + k_int * int_np + k_curious * cur_np
    deltas = reward_term + np.float32(discount) * next_values * next_masks - values_np

    decay = np.float32(discount * gae_lambda) * next_masks
    adv = np.empty_like(deltas)
    next_adv = np.float32(0.0)
    for i in range(num_frames - 1, -1, -1):
        next_adv = deltas[i] + decay[i] * next_adv
        adv[i] = next_adv

    advantages.copy_(torch.from_numpy(adv).to(advantages.device))
