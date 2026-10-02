"""The weighted pixel MSE keeps the plain MSE's scale and weights only the saturated targets."""

from __future__ import annotations

import pytest
import torch

from curious_george.configs import ArchPrnnCfg, TrainPrnnCfg
from curious_george.models.pixel_weights import WeightedPixelMSE


def test_flags_are_off_by_default():
    assert ArchPrnnCfg().saturated_pixel_weight is None and TrainPrnnCfg().readout_lr_scale is None


def test_weight_one_is_the_plain_mse_and_weights_move_only_saturated_targets():
    target = torch.tensor([[0.298, 0.573, 1.0, 1.0]])
    pred = torch.tensor([[0.398, 0.573, 0.6, 1.0]])
    plain = torch.nn.functional.mse_loss(pred, target)
    assert torch.isclose(WeightedPixelMSE(1.0)(pred, target, None), plain)
    weighted = WeightedPixelMSE(10.0)(pred, target, None)
    # the saturated miss (0.4^2 at weight 10) dominates; the floor miss (0.1^2) is down-weighted by the larger mean weight
    expected = (0.1 ** 2 + 10 * 0.4 ** 2) / (1 + 1 + 10 + 10)
    assert torch.isclose(weighted, torch.tensor(expected))
    assert weighted > plain
    with pytest.raises(ValueError):
        WeightedPixelMSE(0.0)
