"""Policy-gradient loss functions.

Every loss has the same signature so the updater is loss-agnostic:

    loss_fn(dist, value, sb, **cfg_kwargs) -> (loss, LossTerms)

where `dist, value` are the model outputs on the minibatch `sb` (a DictList
slice with .action, .log_prob, .advantage, .returnn, .value). Pick a loss by
name via LOSSES / cfg key `rl.loss`.
"""

from dataclasses import dataclass

import torch


@dataclass
class LossTerms:
    """Per-minibatch scalar diagnostics returned next to the loss."""

    policy_entropy_bits: float
    value_mean: float
    policy_loss: float
    value_loss: float


_LOG2 = torch.log(torch.tensor(2.0))


def ppo_clip_loss(
    dist,
    value,
    sb,
    *,
    clip_eps: float,
    entropy_coef: float,
    value_loss_coef: float,
) -> tuple[torch.Tensor, LossTerms]:
    """The historical PPO-clip objective (extracted verbatim from ppo_update)."""
    policy_entropy = dist.entropy().mean()

    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
    surr1 = ratio * sb.advantage
    surr2 = (
        torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        * sb.advantage
    )
    policy_loss = -torch.min(surr1, surr2).mean()

    value_clipped = sb.value + torch.clamp(
        value - sb.value, -clip_eps, clip_eps
    )
    surr1 = (value - sb.returnn).pow(2) # AC_model value estimate - target return
    surr2 = (value_clipped - sb.returnn).pow(2) # ppo clipping
    value_loss = torch.max(surr1, surr2).mean()

    loss = (
        policy_loss
        - entropy_coef * policy_entropy
        + value_loss_coef * value_loss
    )

    terms = LossTerms(
        policy_entropy_bits=policy_entropy.item() / _LOG2.item(),  # nats -> bits
        value_mean=value.mean().item(),
        policy_loss=policy_loss.item(),
        value_loss=value_loss.item(),
    )
    return loss, terms


def a2c_loss(
    dist,
    value,
    sb,
    *,
    entropy_coef: float,
    value_loss_coef: float,
    **_ignored,
) -> tuple[torch.Tensor, LossTerms]:
    """Vanilla advantage actor-critic (no ratio clipping, no value clipping)."""
    policy_entropy = dist.entropy().mean()
    policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean()
    value_loss = (value - sb.returnn).pow(2).mean()

    loss = (
        policy_loss
        - entropy_coef * policy_entropy
        + value_loss_coef * value_loss
    )
    terms = LossTerms(
        policy_entropy_bits=policy_entropy.item() / _LOG2.item(),
        value_mean=value.mean().item(),
        policy_loss=policy_loss.item(),
        value_loss=value_loss.item(),
    )
    return loss, terms


LOSSES = {
    "ppo_clip": ppo_clip_loss,
    "a2c": a2c_loss,
}
