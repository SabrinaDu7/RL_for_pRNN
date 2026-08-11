"""Correctness tests for device-resident generalized advantage estimation."""

import pytest
import torch

from curious_george.rl.update.advantage import compute_gae


def _sequential_reference(
    rewards,
    int_rewards,
    curious_rewards,
    values,
    masks,
    final_values,
    final_masks,
    *,
    discount,
    gae_lambda,
    k_int,
    k_curious,
):
    advantages = torch.empty_like(values)
    next_advantage = torch.zeros_like(final_values)
    for t in range(values.shape[0] - 1, -1, -1):
        next_value = final_values if t == values.shape[0] - 1 else values[t + 1]
        next_mask = final_masks if t == values.shape[0] - 1 else masks[t + 1]
        delta = (
            rewards[t]
            + k_int * int_rewards[t]
            + k_curious * curious_rewards[t]
            + discount * next_value * next_mask
            - values[t]
        )
        next_advantage = (
            delta + discount * gae_lambda * next_mask * next_advantage
        )
        advantages[t] = next_advantage
    return advantages


@pytest.mark.parametrize(("timesteps", "batch"), ((1, 1), (17, 7), (256, 8)))
def test_parallel_scan_matches_sequential_recurrence(timesteps, batch):
    generator = torch.Generator().manual_seed(40 + timesteps)
    shape = (timesteps, batch)
    rewards = torch.randn(shape, generator=generator)
    int_rewards = torch.randn(shape, generator=generator)
    curious_rewards = torch.randn(shape, generator=generator)
    values = torch.randn(shape, generator=generator)
    masks = (torch.rand(shape, generator=generator) > 0.2).float()
    final_values = torch.randn(batch, generator=generator)
    final_masks = (torch.rand(batch, generator=generator) > 0.2).float()
    kwargs = {
        "discount": 0.98,
        "gae_lambda": 0.95,
        "k_int": 0.4,
        "k_curious": 1.3,
    }

    expected = _sequential_reference(
        rewards,
        int_rewards,
        curious_rewards,
        values,
        masks,
        final_values,
        final_masks,
        **kwargs,
    )
    actual = compute_gae(
        rewards=rewards,
        int_rewards=int_rewards,
        curious_rewards=curious_rewards,
        values=values,
        masks=masks,
        final_next_values=final_values,
        final_masks=final_masks,
        **kwargs,
    )

    assert actual.device == values.device
    assert torch.allclose(actual, expected, atol=4e-6, rtol=1e-6)


def test_parallel_scan_handles_long_uninterrupted_trajectory():
    generator = torch.Generator().manual_seed(91)
    shape = (256, 32)
    rewards = torch.randn(shape, generator=generator)
    int_rewards = torch.randn(shape, generator=generator)
    curious_rewards = torch.randn(shape, generator=generator)
    values = torch.randn(shape, generator=generator)
    masks = torch.ones(shape)
    final_values = torch.randn(shape[1], generator=generator)
    final_masks = torch.ones(shape[1])
    kwargs = {
        "discount": 0.99,
        "gae_lambda": 0.95,
        "k_int": 1.0,
        "k_curious": 1.0,
    }

    expected = _sequential_reference(
        rewards,
        int_rewards,
        curious_rewards,
        values,
        masks,
        final_values,
        final_masks,
        **kwargs,
    )
    actual = compute_gae(
        rewards=rewards,
        int_rewards=int_rewards,
        curious_rewards=curious_rewards,
        values=values,
        masks=masks,
        final_next_values=final_values,
        final_masks=final_masks,
        **kwargs,
    )

    # The scan changes FP32 association relative to the serial recurrence.
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-6)
