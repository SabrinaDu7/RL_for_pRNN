"""Spatial-representation metrics: SI, sRSA and sleep-wake distance (SWdist).

The three headline correctness metrics for the pRNN (see docs/refactor_baseline.md):
- sRSA should be HIGH: representational similarity tracks spatial proximity.
- SWdist should be LOW: spontaneous ("sleep") activity stays on the wake manifold.
- SI: per-unit spatial information of the place fields.

Collection is multi-trajectory (OMT-style): n_trajs rollouts of traj_timesteps
steps each - the SAME length as training trajectories (predNet.seqdur), so the
evaluated hidden-state distribution matches what training produces (a single
long rollout never resets the pRNN state; training does, every seqdur steps).
All trajectories are collected and pooled under ONE CPU device move, and each
metric is computed once on the pooled activity.

Logging: these are pRNN metrics and are logged by prnn, not by this repo
(Sabrina, 2026-07-09). The pooled path computes them here TEMPORARILY and
returns them without logging; moving computation+logging into prnn (on
precomputed activity) is the next refactor phase. The legacy path
(trainDecoder=True -> prnn's calculateSpatialRepresentation) still logs
internally as always.
"""

import numpy as np
import pynapple as nap
import torch
from jaxtyping import Float

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
    pooled rollouts instead. Expects pN on CPU (numpy interop).
    """
    obs, act, state, _ = pN.collectObservationSequence(env, agent, wake_timesteps)
    with torch.no_grad():
        _, _, h = pN.predict(obs, act)
    wake_h = torch.mean(h, dim=0, keepdims=True)[0]
    return _sleep_wake_dist(
        pN, wake_h.detach().numpy(), sleepstd=sleepstd, sleep_timesteps=sleep_timesteps
    )


def _sleep_wake_dist(
    pN: PredictiveNet,
    wake_h: Float[np.ndarray, "N H"],
    *,
    sleepstd: float,
    sleep_timesteps: int,
) -> float:
    """SWdist from precomputed wake activity (mirrors prnn predictiveNet.py's
    internal computation: noise-driven spontaneous rollout + RGA distance)."""
    with torch.no_grad():
        _, sleep_h, _ = pN.spontaneous(sleep_timesteps, 0, sleepstd)
    sleep_h = torch.mean(sleep_h, dim=0, keepdims=True)[0]
    swdist, _, _ = RGA.calculateSleepWakeDist(
        wake_h, sleep_h.detach().numpy(), metric="cosine"
    )
    return float(swdist)


def _pooled_spatial_info(
    env,
    h_pool: Float[np.ndarray, "N H"],
    pos_pool: Float[np.ndarray, "N 2"],
    *,
    active_time_threshold: int,
):
    """Per-unit spatial information on pooled trajectories (prnn's pynapple
    recipe: 2D tuning curves + mutual info, low-activity units zeroed).
    The time axis is fabricated (row index) - pynapple only uses it to align
    rates to positions, and tuning curves are occupancy-binned averages."""
    t = np.arange(len(h_pool))
    position = nap.TsdFrame(t=t, d=pos_pool, columns=("x", "y"), time_units="s")
    rates = nap.TsdFrame(t=t, d=h_pool, time_units="s")
    nb_bins_x, nb_bins_y, minmax = env.get_map_bins()
    place_fields, _ = nap.compute_2d_tuning_curves_continuous(
        rates, position, ep=rates.time_support,
        nb_bins=(nb_bins_x, nb_bins_y), minmax=minmax,
    )
    SI = nap.compute_2d_mutual_info(
        place_fields, position, position.time_support, bitssec=False
    )
    num_active = (h_pool > 0).sum(axis=0)
    SI.iloc[num_active < active_time_threshold] = 0
    return SI


def evaluate_spatial_representation(
    pN: PredictiveNet,
    env,
    agent,
    *,
    n_trajs: int = 8,
    traj_timesteps: int = 256,
    trainDecoder: bool = False,
    legacy_timesteps: int = 15000,
    sleepstd: float = 0.03,
    sleep_timesteps: int = 500,
    onset_transient: int = 20,
    active_time_threshold: int = 200,
    wandb_nameext: str = "",
) -> dict:
    """Run the spatial eval on CPU and return {"sRSA", "SWdist", "SI"}.

    Default path: n_trajs trajectories of traj_timesteps steps (match
    predNet.seqdur so eval statistics match training), pooled, one pass per
    metric. trainDecoder=True selects the legacy prnn path (own rollout of
    legacy_timesteps steps, decoder fit, prnn-internal wandb logging) plus a
    second rollout for the returned SWdist.

    Moves pN (and the agent's AC model, if any) to CPU for the duration and
    restores placement after.
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
                timesteps=legacy_timesteps,
                trainDecoder=True,
                trainHDDecoder=False,
                saveTrainingData=False,
                bitsec=False,
                calculatesRSA=True,
                sleepstd=sleepstd,
                wandb_nameext=wandb_nameext,
            )
            swdist = compute_sleep_wake_dist(
                pN, env, agent, sleepstd=sleepstd, wake_timesteps=legacy_timesteps
            )
            return {"sRSA": sRSA, "SWdist": swdist, "SI": SI}

        # ---- pooled multi-trajectory path ----
        h_rows: list = []
        pos_rows: list = []
        for _ in range(n_trajs):
            obs, act, state, _ = pN.collectObservationSequence(
                env, agent, traj_timesteps, discretize=True
            )
            with torch.no_grad():
                _, _, h = pN.predict(obs, act)
            h_mean: Float[torch.Tensor, "T H"] = torch.mean(h, dim=0)  # theta mean
            # h[t] pairs with agent_pos[t] (RGA convention: pos[:-1] vs h)
            h_rows.append(h_mean[onset_transient:])
            pos_rows.append(state["agent_pos"][onset_transient:-1, :])

        h_pool: Float[np.ndarray, "N H"] = torch.cat(h_rows).detach().numpy()
        pos_pool: Float[np.ndarray, "N 2"] = np.concatenate(pos_rows).astype(
            np.float64, copy=False
        )

        if hasattr(pN, "calculateSpatialMetrics"):
            # prnn owns metric computation AND wandb logging (Sabrina's rule);
            # requires the prnn pin at sdu/prnn-perf-optim or later
            metrics = pN.calculateSpatialMetrics(
                h_pool,
                pos_pool,
                env,
                sleepstd=sleepstd,
                sleep_timesteps=sleep_timesteps,
                active_time_threshold=active_time_threshold,
                wandb_nameext=wandb_nameext,
            )
            return {"sRSA": metrics["sRSA"], "SWdist": metrics["SWdist"], "SI": metrics["SI"]}

        # FALLBACK for older prnn pins (no calculateSpatialMetrics): computed
        # RL-side, NOT logged to wandb. Delete once Phase B is accepted
        # (Sabrina will call it at the start of Phase C).
        SI = _pooled_spatial_info(
            env, h_pool, pos_pool, active_time_threshold=active_time_threshold
        )

        # calculateSpatialDist drops the last position row ([:-1]); append a
        # dummy row so the pooled positions line up 1:1 with h_pool
        wake = {
            "state": {"agent_pos": np.vstack([pos_pool, pos_pool[-1:]])},
            "h": h_pool,
        }
        (sRSA, _), _, _, _ = RGA.calculateRSA_space(
            RGA, wake, cont=env.continuous, max_dist=env.max_dist
        )
        swdist = _sleep_wake_dist(
            pN, h_pool, sleepstd=sleepstd, sleep_timesteps=sleep_timesteps
        )

    return {"sRSA": float(sRSA), "SWdist": swdist, "SI": SI}
