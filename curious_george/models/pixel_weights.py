"""A pixel-weighted MSE for the world model's TRAINING loss - and nothing else.

The 2026-10-01 prediction-quality check (questions repository): the targets take four
values, a landmark's saturated colour channel is 1.8% of them, and the trained readout
predicts those at a median of 0.63 where floor and wall sit within 0.05 of their values.
Under a plain MSE the landmark pixels are two percent of the gradient. This weights them
up in the gradient only: the curiosity reward (`models/prnn_adapter.py`'s device pass)
reads the plain per-step error and is untouched, so the policy's reward is the same
quantity as before - what the focal-CE note of 2026-08-31 called "information is
information; only the gradient allocation changes".
"""

from __future__ import annotations

import torch
from torch import nn

SATURATED = 0.9


class WeightedPixelMSE(nn.Module):
    """`predMSE`'s signature `(obs_pred, obs_next, z)`; weight `w` on target values
    >= `threshold`, 1 elsewhere, normalised by the mean weight so the loss's scale
    matches the plain MSE's when every pixel is predicted equally well."""

    def __init__(self, weight: float, threshold: float = SATURATED):
        super().__init__()
        if weight <= 0:
            raise ValueError(f"the saturated-pixel weight must be positive, got {weight}")
        self.weight, self.threshold = float(weight), float(threshold)

    def forward(self, obs_pred, obs_next, z=None):
        w = torch.where(obs_next >= self.threshold, obs_next.new_full((), self.weight), obs_next.new_ones(()))
        return (w * (obs_pred - obs_next) ** 2).sum() / w.sum()


def install_pixel_weights(pN, weight: float) -> WeightedPixelMSE:
    """Replace `pN.loss_fn` (the upstream `predMSE`) for MSE training. Refused under CE."""
    from prnn.utils.lossFuns import predMSE

    if not isinstance(pN.loss_fn, predMSE):
        raise ValueError(f"pixel weights apply to the MSE loss, got {type(pN.loss_fn).__name__}")
    pN.loss_fn = WeightedPixelMSE(weight)
    print(f"pixel weights installed: saturated target channels weighted {weight} in the training loss")
    return pN.loss_fn
