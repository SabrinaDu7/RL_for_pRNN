"""Spatial-representation metrics: SI, sRSA and sleep-wake distance (SWdist).

The three headline correctness metrics for the pRNN (see docs/refactor_baseline.md):
- sRSA should be HIGH: representational similarity tracks spatial proximity.
- SWdist should be LOW: spontaneous ("sleep") activity stays on the wake manifold.
- SI: per-unit spatial information of the place fields.

Historically this wrapped pN.calculateSpatialRepresentation AND ran a second
wake rollout in compute_sleep_wake_dist (prnn computes SWdist internally but
doesn't return it). Profiling showed ~85% of the eval was those two serial
agent rollouts, so the default path now collects ONE wake rollout and derives
all three metrics from it (same RGA / pynapple calls prnn makes internally).
Estimator-equivalent, not bitwise: sRSA and SWdist now share one wake sample,
and prnn's internal wandb side-logging is replaced by the returned dict
(training/logging.log_spatial forwards it).

The legacy double-rollout path survives behind trainDecoder=True (position
decoding is the one thing prnn's version does that this one doesn't).
"""

import numpy as np
import pynapple as nap
import torch

from prnn.utils import PredictiveNet
from prnn.analysis.representationalGeometryAnalysis import (
    representationalGeometryAnalysis as RGA,
)

from curious_george.world_model.device import on_device


def compute_sleep_wake_dist(
    pN: PredictiveNet,
    env,
    agent,
    *,
    sleepstd: float = 0.03,
    wake_timesteps: int = 5000,
    sleep_timesteps: int = 500,
) -> float:
    """Median cosine distance from each sleep frame to its nearest wake frame.

    Standalone variant that collects its own wake rollout; the training loop
    uses evaluate_spatial_representation, which derives SWdist from the shared
    rollout instead. Expects pN on CPU (numpy interop); wrap with on_device.
    """
    obs, act, state, _ = pN.collectObservationSequence(env, agent, wake_timesteps)
    with torch.no_grad():
        _, _, h = pN.predict(obs, act)
    wake_h = torch.mean(h, dim=0, keepdims=True)[0]
    return _sleep_wake_dist(pN, wake_h.detach().numpy(), sleepstd, sleep_timesteps)


def _sleep_wake_dist(pN, wake_h: np.ndarray, sleepstd: float, sleep_timesteps: int) -> float:
    """SWdist from precomputed wake activity (mirrors prnn predictiveNet.py's
    internal computation: noise-driven spontaneous rollout + RGA distance)."""
    with torch.no_grad():
        _, sleep_h, _ = pN.spontaneous(sleep_timesteps, 0, sleepstd)
    sleep_h = torch.mean(sleep_h, dim=0, keepdims=True)[0]
    swdist, _, _ = RGA.calculateSleepWakeDist(
        wake_h, sleep_h.detach().numpy(), metric="cosine"
    )
    return float(swdist)


def evaluate_spatial_representation(
    pN: PredictiveNet,
    env,
    agent,
    *,
    timesteps: int = 2000,
    trainDecoder: bool = False,
    sleepstd: float = 0.03,
    sleep_timesteps: int = 500,
    onset_transient: int = 20,
    active_time_threshold: int = 200,
    wandb_nameext: str = "",
) -> dict:
    """Run the spatial eval on CPU and return {"sRSA", "SWdist", "SI"}.

    Moves pN (and the agent's AC model, if any) to CPU for the duration and
    restores placement after. See module docstring for the single- vs
    double-rollout story; trainDecoder=True selects the legacy prnn path.
    """
    modules = [pN]
    if hasattr(agent, "acmodel"):
        modules.append(agent.acmodel)

    with on_device(modules, "cpu"):
        if trainDecoder:
            # legacy path: prnn does its own rollout, figures, decoder fit and
            # wandb logging; SWdist needs the second rollout
            _, SI, _, sRSA = pN.calculateSpatialRepresentation(
                env,
                agent,
                timesteps=timesteps,
                trainDecoder=True,
                trainHDDecoder=False,
                saveTrainingData=False,
                bitsec=False,
                calculatesRSA=True,
                sleepstd=sleepstd,
                wandb_nameext=wandb_nameext,
            )
            swdist = compute_sleep_wake_dist(
                pN, env, agent, sleepstd=sleepstd, wake_timesteps=timesteps
            )
            return {"sRSA": sRSA, "SWdist": swdist, "SI": SI}

        # ---- single-rollout path (mirrors prnn predictiveNet.py's
        # calculateSpatialRepresentation, minus decoder/figures) ----
        obs, act, state, _ = pN.collectObservationSequence(
            env, agent, timesteps, discretize=True
        )
        with torch.no_grad():
            _, _, h = pN.predict(obs, act)
        h = torch.mean(h, dim=0, keepdims=True)  # mean over theta windows
        h_np = np.squeeze(h.detach().numpy())

        # SI via pynapple place fields (same bins/thresholds as prnn)
        position = nap.TsdFrame(
            t=np.arange(onset_transient, timesteps),
            d=state["agent_pos"][onset_transient:-1, :],
            columns=("x", "y"),
            time_units="s",
        )
        rates = nap.TsdFrame(
            t=np.arange(onset_transient, h.size(1)),
            d=h_np[onset_transient:, :],
            time_units="s",
        )
        nb_bins_x, nb_bins_y, minmax = env.get_map_bins()
        place_fields, _ = nap.compute_2d_tuning_curves_continuous(
            rates, position, ep=rates.time_support,
            nb_bins=(nb_bins_x, nb_bins_y), minmax=minmax,
        )
        SI = nap.compute_2d_mutual_info(
            place_fields, position, position.time_support, bitssec=False
        )
        num_active = np.sum((h > 0).numpy(), axis=1)
        SI.iloc[(num_active < active_time_threshold).flatten()] = 0

        # sRSA + SWdist from the same wake activity
        wake = {"state": state, "h": h_np}
        (sRSA, _), _, _, _ = RGA.calculateRSA_space(
            RGA, wake, cont=env.continuous, max_dist=env.max_dist
        )
        swdist = _sleep_wake_dist(pN, h_np, sleepstd, sleep_timesteps)

    return {"sRSA": float(sRSA), "SWdist": swdist, "SI": SI}
