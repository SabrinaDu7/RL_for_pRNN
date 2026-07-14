"""Full-sequence batched pRNN forward vs per-sequence serial forwards.

Verdict on the 'predict(batched=True) permutation bug' (2026-07-14):
- PredictiveNet.predict(batched=True) IS broken: it permutes (B, L, X) to
  3-D (L, X, B), but pRNN.forward's batched path needs 4-D (1, L, X, B)
  (restructure_inputs slices dim1 as time) -> RuntimeError. Keep avoiding it.
- pRNN.forward(batched=True) itself is CORRECT with the 4-D layout: outputs
  match serial forwards to float32 reduction-order noise (~2e-6). Batched
  world-model training builds on the forward directly.
"""

import numpy as np
import pytest
import torch

from prnn.utils.Architectures import thRNN_5win

X, A, H, B, L = 10, 11, 32, 3, 40


def test_forward_batched_4d_matches_serial():
    torch.manual_seed(0)
    net = thRNN_5win(X, A, hidden_size=H)
    net.eval()  # dropout off; zero noise by default

    obs = torch.rand(B, L + 1, X)
    act = torch.rand(B, L, A)

    yb, hb, tb = net(
        obs.permute(1, 2, 0).unsqueeze(0),
        act.permute(1, 2, 0).unsqueeze(0),
        batched=True,
    )
    for b in range(B):
        y, h, t = net(obs[b : b + 1], act[b : b + 1])
        assert torch.allclose(y[0], yb[0, ..., b], atol=1e-5)
        assert torch.allclose(h[0], hb[0, ..., b], atol=1e-5)
        assert torch.equal(t[0], tb[0, ..., b])


def test_predict_batched_input_prep_is_broken():
    """Documents the actual bug: predict()'s 3-D permute crashes the forward.
    If this ever starts passing, prnn fixed predict(batched=True) upstream -
    revisit whether to route through it."""
    from prnn.utils import PredictiveNet  # noqa: F401  (import parity)

    torch.manual_seed(0)
    net = thRNN_5win(X, A, hidden_size=H)
    net.eval()
    obs3d = torch.rand(B, L + 1, X).permute(1, 2, 0)  # predict()'s 3-D layout
    act3d = torch.rand(B, L, A).permute(1, 2, 0)
    with pytest.raises(RuntimeError):
        net(obs3d, act3d, batched=True)
