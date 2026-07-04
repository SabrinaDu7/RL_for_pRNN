"""Spatial-representation metrics: sRSA and sleep-wake distance (SWdist).

The two headline correctness metrics for the pRNN (see docs/refactor_baseline.md):
- sRSA should be HIGH: representational similarity tracks spatial proximity.
- SWdist should be LOW: spontaneous ("sleep") activity stays on the wake manifold.

`calculateSpatialRepresentation` computes both but only RETURNS sRSA (SWdist
goes straight to wandb inside prnn). This module returns both, and owns the
CPU placement that used to be done by hand around every callsite (the
computation is numpy-based, so CPU is required).
"""

import numpy as np
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

    Mirrors the computation inside calculateSpatialRepresentation
    (prnn predictiveNet.py:945-953): wake activity from an agent rollout,
    sleep activity from a noise-driven spontaneous rollout; the scalar is the
    FIRST return of RGA.calculateSleepWakeDist (named SleepSimilarity there,
    logged as SWdist).

    Expects pN to already be on CPU (numpy interop); wrap with on_device.
    """
    obs, act, state, _ = pN.collectObservationSequence(env, agent, wake_timesteps)
    with torch.no_grad():
        _, _, h = pN.predict(obs, act)
    wake_h = torch.mean(h, dim=0, keepdims=True)[0]

    _, sleep_h, _ = pN.spontaneous(sleep_timesteps, 0, sleepstd)
    sleep_h = torch.mean(sleep_h, dim=0, keepdims=True)[0]

    swdist, _, _ = RGA.calculateSleepWakeDist(
        wake_h.detach().numpy(), sleep_h.detach().numpy(), metric="cosine"
    )
    return float(swdist)


def evaluate_spatial_representation(
    pN: PredictiveNet,
    env,
    agent,
    *,
    trainDecoder: bool = True,
    sleepstd: float = 0.03,
    wandb_nameext: str = "",
) -> dict:
    """Run the spatial eval on CPU and return {"sRSA", "SWdist", "SI"}.

    Wraps pN.calculateSpatialRepresentation (which also does its own wandb
    logging, including its internal SWdist, when pN.wandb_log is set) and adds
    a directly-returned SWdist scalar. Moves pN (and the agent's AC model, if
    any) to CPU for the duration and restores placement after.
    """
    modules = [pN]
    if hasattr(agent, "acmodel"):
        modules.append(agent.acmodel)

    with on_device(modules, "cpu"):
        _, SI, _, sRSA = pN.calculateSpatialRepresentation(
            env,
            agent,
            trainDecoder=trainDecoder,
            trainHDDecoder=False,
            saveTrainingData=False,
            bitsec=False,
            calculatesRSA=True,
            sleepstd=sleepstd,
            wandb_nameext=wandb_nameext,
        )
        swdist = compute_sleep_wake_dist(pN, env, agent, sleepstd=sleepstd)

    return {"sRSA": sRSA, "SWdist": swdist, "SI": SI}
