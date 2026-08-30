"""Policy-gradient loss functions.

Every loss has the same signature so `policy.py` is loss-agnostic:

    loss_fn(dist, value, sb, **cfg_kwargs) -> (loss, LossTerms)

where `dist, value` are the model outputs on the minibatch `sb` (a DictList
slice with .action, .log_prob, .advantage, .returnn, .value). Pick a loss by
name via LOSSES / cfg key `rl.loss`.
"""

import math
from dataclasses import dataclass

import torch


@dataclass
class LossTerms:
    """Per-minibatch scalar diagnostics returned next to the loss.

    Fields are detached 0-dim tensors, NOT floats: converting per minibatch
    (.item()) forced a device sync each time; `policy.py` aggregates them and
    syncs once per update.
    """

    policy_entropy_bits: torch.Tensor
    value_mean: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor


_LOG2 = math.log(2.0)


def ppo_clip_loss(
    dist,
    value,
    sb,
    *,
    clip_eps: float,
    entropy_coef: float,
    value_loss_coef: float,
    normalize_advantage: bool = False,
) -> tuple[torch.Tensor, LossTerms]:
    """The historical PPO-clip objective (extracted verbatim from ppo_update).

    `normalize_advantage` whitens the advantage per MINIBATCH to mean 0, std 1.

    WHY IT IS NOT A LEARNING-RATE CHANGE. Adam is invariant to a global rescale
    of the gradient (up to eps), so scaling the policy term alone would do
    nothing. What whitening actually fixes is the RELATIVE WEIGHT of the policy
    term against the entropy and value terms, which Adam cannot absorb: the
    policy gradient scales with |advantage| while `entropy_coef` is a fixed
    additive term, so what governs exploration is the RATIO
    `entropy_coef / |advantage|` - and that denominator moves. Measured in this
    repo, |adv| falls from 0.525 to 0.120 within 30 rollouts, so a "constant"
    coefficient silently gets ~4x stronger as training proceeds. Whitening pins
    the denominator at 1 and makes the coefficient mean one fixed thing.

    That also predicts the SHAPE of the effect: small at `entropy_coef=0`, large
    at 0.01. A single arm cannot show that; a factorial can.
    """
    policy_entropy = dist.entropy().mean()

    advantage = sb.advantage
    if normalize_advantage:
        # eps guards a minibatch whose advantages are all equal - which happens
        # for real at k_curious=0, not only in principle.
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
    surr1 = ratio * advantage
    surr2 = (
        torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        * advantage
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
        policy_entropy_bits=policy_entropy.detach() / _LOG2,  # nats -> bits
        value_mean=value.detach().mean(),
        policy_loss=policy_loss.detach(),
        value_loss=value_loss.detach(),
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
        policy_entropy_bits=policy_entropy.detach() / _LOG2,
        value_mean=value.detach().mean(),
        policy_loss=policy_loss.detach(),
        value_loss=value_loss.detach(),
    )
    return loss, terms


LOSSES = {
    "ppo_clip": ppo_clip_loss,
    "a2c": a2c_loss,
}
