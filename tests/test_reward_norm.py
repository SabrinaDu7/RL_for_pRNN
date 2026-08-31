"""`train_policy.normalize_reward`: the three controls from
`docs/claude_logs/rl_tricks_2026-08-29.md` §1, as unit gates on `compute_gae`.

Positive: with the normalizer ON, scaling every reward by a constant is a
NO-OP - if it is not, the normalizer is broken and nothing downstream of it
means anything. Negative: with it OFF the same scaling must move the
advantages, or the reward was not driving them at all. Extreme: an all-zero
reward (k_curious=0 in the real config) must degrade gracefully, never inf.
"""

import torch

from curious_george.rl.update.advantage import RewardNormalizer, compute_gae


def _gae(curious: torch.Tensor, normalizer: RewardNormalizer | None) -> torch.Tensor:
    T, B = curious.shape
    return compute_gae(
        rewards=torch.zeros(T, B),
        int_rewards=torch.zeros(T, B),
        curious_rewards=curious,
        values=torch.linspace(0.1, 0.4, T * B).reshape(T, B),
        masks=torch.ones(T, B),
        final_next_values=torch.full((B,), 0.25),
        final_masks=torch.ones(B),
        discount=0.98,
        gae_lambda=0.95,
        k_int=0.0,
        k_curious=1.0,
        reward_normalizer=normalizer,
    )


def _curious() -> torch.Tensor:
    return torch.rand(16, 8, generator=torch.Generator().manual_seed(7))


def test_scaling_is_a_noop_with_normalization_on():
    curious = _curious()
    a1 = _gae(curious, RewardNormalizer())
    a10 = _gae(curious * 10.0, RewardNormalizer())
    assert torch.allclose(a1, a10, atol=1e-5), (
        "x10 reward moved whitened-reward advantages; the normalizer is broken"
    )


def test_scaling_moves_advantages_with_normalization_off():
    curious = _curious()
    a1 = _gae(curious, None)
    a10 = _gae(curious * 10.0, None)
    assert not torch.allclose(a1, a10, atol=1e-3), (
        "x10 reward changed nothing raw; the reward is not driving advantages"
    )


def test_zero_reward_degrades_gracefully():
    a = _gae(torch.zeros(16, 8), RewardNormalizer())
    assert torch.isfinite(a).all(), "k_curious=0 with normalization on produced non-finite advantages"


def test_running_std_accumulates_across_rollouts():
    norm = RewardNormalizer()
    for _ in range(5):
        norm.update_and_normalize(torch.randn(16, 8, generator=torch.Generator().manual_seed(3)) * 2.0)
    assert abs(norm.std - 2.0) < 0.3, f"running std {norm.std} far from the true 2.0"
