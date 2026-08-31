"""Analytic gates for `evaluation/error_decomposition`: hand-built inputs with
closed-form answers, both losses, plus the bump-mask geometry."""

import torch

from curious_george.envs.palette import TILE_CLASS_NAMES, vocab_tensor
from curious_george.evaluation.error_decomposition import (
    bump_contrast,
    decompose_by_class,
    per_tile_errors,
)

VOCAB = vocab_tensor()
C = len(TILE_CLASS_NAMES)


def _targets(classes: torch.Tensor) -> torch.Tensor:
    return VOCAB[classes].reshape(classes.shape[0], -1)


def test_ce_perfect_prediction_has_zero_error_everywhere():
    classes = torch.randint(0, C, (6, 49), generator=torch.Generator().manual_seed(0))
    logits = torch.full((6, 49, C), -30.0)
    logits.scatter_(2, classes.unsqueeze(-1), 30.0)
    errors, got = per_tile_errors(logits.reshape(6, -1), _targets(classes), ce=True)
    assert torch.equal(got, classes)
    assert float(errors.max()) < 1e-6


def test_mse_error_lands_on_the_wrong_tile_only():
    classes = torch.zeros((1, 49), dtype=torch.long)  # all floor
    pred = _targets(classes).clone()
    wrong = VOCAB[3] - VOCAB[0]  # what predicting blue instead of floor costs
    pred[0, 21:24] = VOCAB[3]    # tile 7 predicted blue
    errors, got = per_tile_errors(pred, _targets(classes), ce=False)
    assert float(errors[0, 7] - float((wrong**2).sum())) < 1e-6
    assert float(errors.sum() - errors[0, 7]) < 1e-9


def test_class_decomposition_shares_sum_to_one():
    classes = torch.randint(0, C, (8, 49), generator=torch.Generator().manual_seed(1))
    logits = torch.randn(8, 49, C, generator=torch.Generator().manual_seed(2))
    errors, got = per_tile_errors(logits.reshape(8, -1), _targets(classes), ce=True)
    d = decompose_by_class(errors, got)
    assert abs(float(d.tile_share.sum()) - 1.0) < 1e-6
    assert abs(float(d.error_share.sum()) - 1.0) < 1e-6
    assert d.class_names == TILE_CLASS_NAMES


def test_bump_contrast_separates_refused_from_free_forwards():
    #      stay  fwd(free) fwd(BUMP) left  fwd(BUMP)
    actions = torch.tensor([3, 2, 2, 0, 2])
    positions = torch.tensor([[1, 1], [1, 1], [2, 1], [2, 1], [2, 1], [2, 1]])
    errors = torch.tensor([0.1, 0.2, 5.0, 0.3, 7.0])
    c = bump_contrast(errors, actions, positions)
    assert (c.n_bumps, c.n_free) == (2, 1)
    assert abs(c.bump_mean_error - 6.0) < 1e-6
    assert abs(c.free_mean_error - 0.2) < 1e-6
    assert c.bump_minus_free > 5
