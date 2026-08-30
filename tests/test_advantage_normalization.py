"""Whitening the advantage, and the controls that make it a test.

The policy gradient scales with |advantage| while `entropy_coef` is a fixed
ADDITIVE term, so what governs exploration is `entropy_coef / |advantage|` - and
the denominator moves as the world model minimises its own loss. Whitening pins
it at 1.
"""

import dataclasses

import numpy as np
import pytest
import torch

from curious_george.configs import EnvBackend
from curious_george.rl.update.losses import LOSSES


class _Batch:
    """Just the fields ppo_clip_loss reads."""

    def __init__(self, advantage, action, log_prob, value, returnn):
        self.advantage, self.action, self.log_prob = advantage, action, log_prob
        self.value, self.returnn = value, returnn


def _inputs(scale: float, n: int = 512, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, 4, generator=g)
    dist = torch.distributions.Categorical(logits=logits)
    action = torch.randint(0, 4, (n,), generator=g)
    sb = _Batch(
        advantage=torch.randn(n, generator=g) * scale + 0.3 * scale,
        action=action,
        log_prob=dist.log_prob(action).detach(),
        value=torch.randn(n, generator=g) * scale,
        returnn=torch.randn(n, generator=g) * scale,
    )
    return dist, torch.randn(n, generator=g) * scale, sb


def _policy_grad_norm(scale: float, *, normalize: bool, n: int = 512, seed: int = 0):
    """Gradient of the POLICY term w.r.t. the logits, at a given reward scale.

    `entropy_coef=0` and `value_loss_coef=0`, so this isolates the one term
    whitening acts on. Comparing the whole loss instead is misleading: the value
    term scales with the reward too and whitening does NOT touch it.
    """
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, 4, generator=g).requires_grad_(True)
    dist = torch.distributions.Categorical(logits=logits)
    action = torch.randint(0, 4, (n,), generator=g)
    sb = _Batch(
        advantage=torch.randn(n, generator=g) * scale + 0.3 * scale,
        action=action,
        log_prob=dist.log_prob(action).detach(),
        value=torch.randn(n, generator=g) * scale,
        returnn=torch.randn(n, generator=g) * scale,
    )
    loss, _ = LOSSES["ppo_clip"](
        dist, torch.randn(n, generator=g) * scale, sb,
        clip_eps=0.2, entropy_coef=0.0, value_loss_coef=0.0,
        normalize_advantage=normalize,
    )
    loss.backward()
    return float(logits.grad.norm())


def test_default_is_off_and_bitwise_unchanged():
    """A new field must not move the existing default path."""
    dist, value, sb = _inputs(1.0)
    kw = dict(clip_eps=0.2, entropy_coef=0.01, value_loss_coef=1.0)
    a, _ = LOSSES["ppo_clip"](dist, value, sb, **kw)
    b, _ = LOSSES["ppo_clip"](dist, value, sb, normalize_advantage=False, **kw)
    assert torch.equal(a, b)


def test_whitening_makes_the_policy_gradient_scale_invariant():
    """THE POINT, stated as the thing that actually changes.

    The curiosity reward IS the world model's loss and the world model is
    minimising it, so the advantage scale falls over a run - measured, |adv|
    0.525 -> 0.120 within 30 rollouts. Raw, the policy gradient shrinks with it
    while `entropy_coef` stays fixed, so exploration silently strengthens.
    Whitened, the policy gradient is invariant and the coefficient means one
    fixed thing.
    """
    scales = (1.0, 0.2, 0.05)          # a 20x range, ~ what a run traverses
    raw = [_policy_grad_norm(s, normalize=False) for s in scales]
    white = [_policy_grad_norm(s, normalize=True) for s in scales]
    assert max(raw) / min(raw) > 10, f"raw gradient barely moved: {raw}"
    assert max(white) / min(white) < 1.05, f"whitened gradient still moved: {white}"


def test_whitening_does_NOT_fix_the_value_term():
    """A limitation worth pinning, because it is the argument for the SECOND
    change. Whitening acts on the advantage only; the value loss is
    (value - return)^2, which scales with the reward squared and is untouched.
    So a fixed `value_loss_coef` still means different things over a run."""
    dist, value, sb = _inputs(1.0)
    kw = dict(clip_eps=0.2, entropy_coef=0.0, normalize_advantage=True)
    big, _ = LOSSES["ppo_clip"](dist, value, sb, value_loss_coef=1.0, **kw)
    dist2, value2, sb2 = _inputs(0.2)
    small, _ = LOSSES["ppo_clip"](dist2, value2, sb2, value_loss_coef=1.0, **kw)
    assert abs(float(big)) > 5 * abs(float(small)), (
        "the value term no longer scales with the reward - if this fails, "
        "reward normalization may already be unnecessary"
    )


def test_whitened_advantage_has_unit_scale():
    dist, value, sb = _inputs(0.05)
    adv = (sb.advantage - sb.advantage.mean()) / (sb.advantage.std() + 1e-8)
    assert abs(float(adv.mean())) < 1e-5
    assert abs(float(adv.std()) - 1.0) < 1e-3


def test_a_constant_advantage_does_not_divide_by_zero():
    """The EXTREME control: k_curious=0 makes every advantage equal, so the
    standard deviation vanishes. It must degrade, not produce inf."""
    dist, value, sb = _inputs(1.0)
    sb.advantage = torch.full_like(sb.advantage, 0.7)
    loss, _ = LOSSES["ppo_clip"](
        dist, value, sb, clip_eps=0.2, entropy_coef=0.01,
        value_loss_coef=1.0, normalize_advantage=True,
    )
    assert torch.isfinite(loss), "vanishing advantage std produced a non-finite loss"


def test_the_flag_reaches_the_loss_through_the_config():
    """A field nothing threads is a field that silently does nothing - the
    failure mode `entropy_coef_final` had under CUDA graphs."""
    from curious_george.training.setup import setup_training
    from tests.small_config import small_config

    for want in (False, True):
        cfg = small_config(backend=EnvBackend.SERIAL_TABLE)
        cfg = dataclasses.replace(
            cfg, train_policy=dataclasses.replace(
                cfg.train_policy, normalize_advantage=want))
        assert setup_training(cfg).algo.normalize_advantage is want
