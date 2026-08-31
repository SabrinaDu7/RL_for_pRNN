"""Device-resident generalized advantage estimation."""

import torch


class RewardNormalizer:
    """Divide the combined reward by a running std (`train_policy.normalize_reward`).

    Burda et al. (Large-Scale Study of Curiosity-Driven Learning §2.2): the
    curiosity reward is the world model's own loss, which the world model is
    minimising, so the reward scale is non-stationary BY CONSTRUCTION (measured
    here: ~7x decay over a run). A raw-scale reward makes the value target -
    and with it `value_loss_coef`'s meaning - drift over training. Whitened
    advantages fix the policy:entropy ratio; this fixes the CRITIC's target
    scale. Scale only, never centered: subtracting a baseline from the reward
    changes the discounted objective, dividing it does not.

    Welford over reward ELEMENTS (Burda uses the std of discounted reward
    sums; the difference is a constant factor ~1/(1-γλ) which whitened
    advantages absorb, and per-element needs no second recurrence).

    UPDATE-THEN-NORMALIZE per rollout, which makes the gate exact: with a
    fresh normalizer, scaling every reward by c scales the running std by c
    and the normalized reward is IDENTICAL - `k_curious x10` must be a no-op
    with this on, and must not be with it off (tests/test_reward_norm.py).

    Not checkpoint-persisted: a resumed run re-warms the estimate within a few
    rollouts, and the transient is a scale wobble the whitened advantages
    absorb.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update_and_normalize(self, reward: torch.Tensor) -> torch.Tensor:
        n = reward.numel()
        batch_mean = float(reward.mean())
        batch_var = float(reward.var(unbiased=False))
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.m2 += batch_var * n + delta * delta * self.count * n / total
        self.count = total
        return reward / (self.std + self.eps)

    @property
    def std(self) -> float:
        return (self.m2 / self.count) ** 0.5 if self.count else 0.0


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
    count_rewards: torch.Tensor | None = None,
    k_count: float = 0.0,
    reward_normalizer: RewardNormalizer | None = None,
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
    if count_rewards is not None:
        # Branched, not a zero tensor: the default path must stay bitwise
        # identical for the golden gate.
        reward = reward + k_count * count_rewards
    if reward_normalizer is not None:
        # Branched for the same golden-gate reason as count_rewards above.
        reward = reward_normalizer.update_and_normalize(reward)
    deltas = reward + discount * next_values * next_masks - values

    decay = discount * gae_lambda * next_masks
    advantages = torch.empty_like(deltas)
    running = torch.zeros_like(deltas[0])
    for t in range(deltas.shape[0] - 1, -1, -1):
        running = deltas[t] + decay[t] * running
        advantages[t] = running
    return advantages
