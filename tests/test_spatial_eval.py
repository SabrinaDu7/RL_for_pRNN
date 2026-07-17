"""Pooled multi-trajectory spatial eval: shapes, finiteness, and alignment."""

import numpy as np
import pytest
import torch

from prnn.utils import (
    ActionEncodingsEnum,
    MinigridEnvNames,
    PredictiveNet,
    RandomActionAgent,
)
from curious_george import AgentInputType, make_env
from curious_george.evaluation.spatial import evaluate_spatial_representation

SEED = 11


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    env = make_env(
        env_key=MinigridEnvNames.LRoom,
        input_type=AgentInputType.H_PO.value,
        act_enc=ActionEncodingsEnum.SpeedHD.value,
        seed=SEED,
    )
    pN = PredictiveNet(env, hidden_size=32, pRNNtype="thRNN_5win",
                       trainNoiseMeanStd=(0, 0.03), wandb_log=False)
    pN.pRNN.eval()
    agent = RandomActionAgent(env.action_space, np.array([0.15, 0.15, 0.6, 0.1]))
    return env, pN, agent


def test_pooled_eval_returns_finite_metrics(setup):
    env, pN, agent = setup
    metrics = evaluate_spatial_representation(
        pN, env, agent,
        n_trajs=3, traj_timesteps=60,
        onset_transient=5, active_time_threshold=10,
        sleep_timesteps=50,
    )
    assert set(metrics) == {"sRSA", "SWdist", "SI"}
    assert np.isfinite(metrics["sRSA"])
    assert np.isfinite(metrics["SWdist"]) and metrics["SWdist"] >= 0
    assert len(metrics["SI"]) == pN.hidden_size
    assert np.isfinite(np.nanmean(np.asarray(metrics["SI"]["SI"], dtype=float)))


def test_pooled_eval_restores_device_placement(setup):
    env, pN, agent = setup
    before = next(pN.pRNN.parameters()).device
    evaluate_spatial_representation(
        pN, env, agent,
        n_trajs=2, traj_timesteps=40,
        onset_transient=5, active_time_threshold=5, sleep_timesteps=30,
    )
    assert next(pN.pRNN.parameters()).device == before
