"""Where does novel-object information live: the recurrent dynamics, or the readout?

The pRNN predicts observations as obs_pred = sigmoid(W_out @ h). The OMT metric
measures obs_pred; the rate-map analysis measures h. Those came apart - the
pixel-space object effect is clear while h's spatial tuning is unchanged
(median per-unit map correlation ~0.98 across the whole exposure phase) - so
the question is which half of the network carries the object.

The test is a causal swap. Build two chimaeras from the pre-exposure and
post-exposure checkpoints:

    base dynamics    + TRAINED readout
    TRAINED dynamics + base readout

and measure each one's predicted green at the (now absent) object location,
relative to control locations, relative to the baseline net. Whichever
chimaera reproduces the effect is where the information is.

Caveat worth keeping in view: W_out was trained jointly with the dynamics, so
the two halves are not independent and the recovered shares do not sum to 100%.
The transplant works partly BECAUSE h barely moved - "the readout carries it"
and "the dynamics did not change" are two views of one fact, not two
independent confirmations.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from jaxtyping import Float

from prnn.utils import PredictiveNet
from prnn.utils.Shell import FaramaMinigridShell

from scripts.trace_probe import Probe
from tasks.omt.metrics import get_obs_at_loc_fast


def swap_readout(*, dynamics: PredictiveNet, readout: PredictiveNet) -> PredictiveNet:
    """Copy `dynamics`, then take ONLY W_out from `readout`."""
    net = copy.deepcopy(dynamics)
    sd = net.pRNN.state_dict()
    sd["W_out"] = readout.pRNN.state_dict()["W_out"].clone()
    net.pRNN.load_state_dict(sd)
    return net


def relative_weight_change(
    *, base: PredictiveNet, trained: PredictiveNet
) -> dict[str, float]:
    """||trained - base|| / ||base|| per parameter block."""
    sb, st = base.pRNN.state_dict(), trained.pRNN.state_dict()
    return {
        k: float(torch.norm(st[k] - sb[k]) / torch.norm(sb[k]))
        for k in ("W_in", "W", "W_out", "bias")
    }


def predicted_green_at(
    *,
    pN: PredictiveNet,
    probe: Probe,
    env: FaramaMinigridShell,
    loc: tuple[int, int],
    n_trajs: int = 200,
    onset: int = 20,
) -> float:
    """Mean predicted green channel at `loc`, over timesteps where it is in view.

    Runs on the object-ABSENT probe, so this asks what the net predicts where
    the object used to be. Injected noise is disabled for determinism, which
    differs from the OMT eval path (it keeps noise on).
    """
    saved = pN.trainNoiseMeanStd
    pN.trainNoiseMeanStd = (0.0, 0.0)
    try:
        pN.pRNN.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            init = pN.pRNN.rnn.cell.actfun(torch.zeros(n_trajs, 1, pN.hidden_size))
            obs_pred, _, _ = pN.predict(
                probe.obs[:n_trajs], probe.act[:n_trajs],
                state=init, randInit=False, batched=True,
            )
    finally:
        pN.trainNoiseMeanStd = saved

    op: Float[torch.Tensor, "B T X"] = obs_pred[0].permute(2, 0, 1)[:, onset:, :]
    B, T, X = op.shape
    img = env.pred2np(op.reshape(B * T, X).unsqueeze(0))          # (B*T, 7, 7, 3)
    pos = probe.agent_pos[:n_trajs, onset:onset + T, :].reshape(-1, 2)
    hd = probe.agent_dir[:n_trajs, onset:onset + T].reshape(-1)
    locobs, _, _ = get_obs_at_loc_fast(img, list(loc), pos, hd)
    return float("nan") if locobs is None else float(locobs[:, 1].mean())


def object_contrast(
    *,
    pN: PredictiveNet,
    probe: Probe,
    env: FaramaMinigridShell,
    obj_loc: tuple[int, int],
    ctrl_locs: list[list[int]],
    n_trajs: int = 200,
) -> float:
    """Predicted green at the object location minus the mean over control locations."""
    kw = dict(pN=pN, probe=probe, env=env, n_trajs=n_trajs)
    at_obj = predicted_green_at(loc=obj_loc, **kw)
    at_ctrl = float(np.mean([predicted_green_at(loc=tuple(c), **kw) for c in ctrl_locs]))
    return at_obj - at_ctrl


def readout_share(
    *,
    base: PredictiveNet,
    trained: PredictiveNet,
    probe: Probe,
    env: FaramaMinigridShell,
    obj_loc: tuple[int, int],
    ctrl_locs: list[list[int]],
    n_trajs: int = 200,
) -> dict[str, float]:
    """Effect size for the trained net and both chimaeras, plus the readout share."""
    kw = dict(probe=probe, env=env, obj_loc=obj_loc, ctrl_locs=ctrl_locs, n_trajs=n_trajs)
    b = object_contrast(pN=base, **kw)
    full = object_contrast(pN=trained, **kw) - b
    rd = object_contrast(pN=swap_readout(dynamics=base, readout=trained), **kw) - b
    dyn = object_contrast(pN=swap_readout(dynamics=trained, readout=base), **kw) - b
    return {
        "full": full,
        "readout_only": rd,
        "dynamics_only": dyn,
        "readout_share": rd / full if full else float("nan"),
    }
