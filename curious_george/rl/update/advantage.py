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

    The reverse recurrence ``adv[t] = delta[t] + decay[t] * adv[t + 1]`` is run
    SEQUENTIALLY, on device. It is associative, so a Hillis-Steele prefix scan
    would evaluate it in ``ceil(log2(T))`` stages instead of ``T`` - but that
    reassociates the float32 sum and perturbs the result by ~6e-8, which is
    below one ULP yet enough to break the bitwise oracle in
    tests/golden_omt/. The scan was measured at ~1 ms/update against a
    ~1.3 s update (throwaway/ported/docs_legacy/throughput_investigation_2026-07-23.md), so the
    sequential form costs ~0.1% of wall-clock and keeps that gate meaningful.

    Everything stays on device either way; the sequential loop launches T
    small kernels but never synchronizes to the host.
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

    decay = discount * gae_lambda * next_masks
    advantages = torch.empty_like(deltas)
    running = torch.zeros_like(deltas[0])
    for t in range(deltas.shape[0] - 1, -1, -1):
        running = deltas[t] + decay[t] * running
        advantages[t] = running
    return advantages
