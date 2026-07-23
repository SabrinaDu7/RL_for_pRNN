"""Device-resident generalized advantage estimation."""

import torch


@torch.no_grad()
def compute_gae(
    *,
    rewards: torch.Tensor,
    int_rewards: torch.Tensor,
    curious_rewards: torch.Tensor,
    values: torch.Tensor,
    masks: torch.Tensor,
    final_next_values: torch.Tensor,
    final_masks,
    discount: float,
    gae_lambda: float,
    k_int: float,
    k_curious: float,
) -> torch.Tensor:
    """Return GAE for rollout tensors shaped ``(T, ...)``.

    ``masks[t]`` describes the state before transition ``t``, so transition
    ``t`` uses ``masks[t + 1]`` (and ``final_masks`` at the rollout tail).

    The reverse affine recurrence

    ``adv[t] = delta[t] + decay[t] * adv[t + 1]``

    is an associative prefix scan after reversing time. A Hillis-Steele scan
    evaluates it in ``ceil(log2(T))`` device stages, keeps arbitrary episode
    boundaries, and avoids both a host synchronization and a 256-kernel
    sequential device loop.
    """
    final_values = torch.as_tensor(
        final_next_values, dtype=values.dtype, device=values.device
    )
    final_mask_tensor = torch.as_tensor(
        final_masks, dtype=masks.dtype, device=masks.device
    )
    next_values = torch.cat((values[1:], final_values.unsqueeze(0)), dim=0)
    next_masks = torch.cat((masks[1:], final_mask_tensor.unsqueeze(0)), dim=0)

    reward = rewards + k_int * int_rewards + k_curious * curious_rewards
    deltas = reward + discount * next_values * next_masks - values

    # Each pair represents f(x) = bias + scale*x. Prefix composition of these
    # affine functions is associative:
    #   (b2, s2) o (b1, s1) = (b2 + s2*b1, s2*s1).
    bias = deltas.flip(0)
    scale = (discount * gae_lambda * next_masks).flip(0)
    offset = 1
    while offset < bias.shape[0]:
        bias = torch.cat(
            (
                bias[:offset],
                bias[offset:] + scale[offset:] * bias[:-offset],
            ),
            dim=0,
        )
        scale = torch.cat(
            (
                scale[:offset],
                scale[offset:] * scale[:-offset],
            ),
            dim=0,
        )
        offset *= 2
    return bias.flip(0)
