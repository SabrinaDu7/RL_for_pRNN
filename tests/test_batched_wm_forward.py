"""Full-sequence batched pRNN forward vs per-sequence serial forwards.

Verdict on the 'predict(batched=True) permutation bug', REVISED 2026-07-30:
- pRNN.forward(batched=True) is CORRECT with the 4-D (1, L, X, B) layout:
  outputs match serial forwards to float32 reduction-order noise (~2e-6).
  Batched world-model training builds on the forward directly.
- PredictiveNet.predict(batched=True) is now also CORRECT. It takes 3-D
  (B, L, X) and does the permute-and-unsqueeze to 4-D itself (prnn 383ae24).
  The earlier "keep avoiding it" verdict (2026-07-14) predates that fix.
  Callers must pass 3-D; 4-D raises in clip_mask.

Note the returned h is only reproducible if PredictiveNet.trainNoiseMeanStd
is zeroed or the RNG is seeded: predict injects fresh noise every call, so
two identical calls differ by ~0.4 in h. Dropout is separate and is off in
eval(). Tests here zero the noise.
"""

import numpy as np
import pytest
import torch

from prnn.utils.Architectures import thRNN_5win
from prnn.utils.predictiveNet import PredictiveNet

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


def test_forward_still_requires_4d_batched_layout():
    """predict(batched=True)'s prep was FIXED upstream (prnn 383ae24) to emit
    the 4-D layout; the raw forward still rejects 3-D input, which is what
    made the original bug crash loudly instead of permuting silently."""
    torch.manual_seed(0)
    net = thRNN_5win(X, A, hidden_size=H)
    net.eval()
    obs3d = torch.rand(B, L + 1, X).permute(1, 2, 0)
    act3d = torch.rand(B, L, A).permute(1, 2, 0)
    with pytest.raises(RuntimeError):
        net(obs3d, act3d, batched=True)


def _tiny_pN() -> PredictiveNet:
    """A PredictiveNet wrapping a thRNN_5win, with injected noise disabled."""
    pN = PredictiveNet.__new__(PredictiveNet)
    pN.pRNN = thRNN_5win(X, A, hidden_size=H)
    pN.pRNN.eval()
    pN.hidden_size = H
    pN.trainNoiseMeanStd = (0.0, 0.0)
    return pN


def test_predict_batched_matches_serial():
    """The regression that made the OMT analysis pass 4-D and crash."""
    torch.manual_seed(0)
    pN = _tiny_pN()
    obs = torch.rand(B, L + 1, X)
    act = torch.rand(B, L, A)
    state = torch.zeros(1, 1, H)

    with torch.no_grad():
        _, _, hb = pN.predict(obs, act, state=state.repeat(B, 1, 1),
                              randInit=False, batched=True)
        batched = hb[0].permute(2, 0, 1)  # (1, L, H, B) -> (B, L, H)
        serial = torch.stack([
            pN.predict(obs[i:i + 1], act[i:i + 1], state=state,
                       randInit=False)[2].squeeze(0)
            for i in range(B)
        ])

    assert batched.shape == serial.shape == (B, L, H)
    assert torch.allclose(batched, serial, atol=1e-5)


def test_predict_batched_rejects_4d_input():
    """Passing (B, 1, L, X) - the pre-fix call shape - must not silently work."""
    torch.manual_seed(0)
    pN = _tiny_pN()
    with pytest.raises(RuntimeError):
        pN.predict(torch.rand(B, 1, L + 1, X), torch.rand(B, 1, L, A),
                   batched=True)


def test_predict_injects_noise_unless_disabled():
    """Documents why the probe must seed or zero the noise to be reproducible."""
    torch.manual_seed(0)
    pN = _tiny_pN()
    obs, act = torch.rand(1, L + 1, X), torch.rand(1, L, A)
    state = torch.zeros(1, 1, H)

    with torch.no_grad():
        a = pN.predict(obs, act, state=state, randInit=False)[2]
        b = pN.predict(obs, act, state=state, randInit=False)[2]
    assert torch.equal(a, b), "noise disabled -> deterministic"

    pN.trainNoiseMeanStd = (0.0, 0.05)
    with torch.no_grad():
        c = pN.predict(obs, act, state=state, randInit=False)[2]
        d = pN.predict(obs, act, state=state, randInit=False)[2]
    assert not torch.equal(c, d), "noise enabled -> fresh draw each call"
