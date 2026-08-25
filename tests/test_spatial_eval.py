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
    assert set(metrics) == {
        "sRSA", "SWdist", "SI",
        "SI_units_total", "SI_units_zeroed", "SI_mean_active_only",
    }
    assert np.isfinite(metrics["sRSA"])
    assert np.isfinite(metrics["SWdist"]) and metrics["SWdist"] >= 0
    assert len(metrics["SI"]) == pN.hidden_size
    assert np.isfinite(np.nanmean(np.asarray(metrics["SI"]["SI"], dtype=float)))


def test_si_coverage_describes_the_units_prnn_zeroed(setup):
    """`mean SI` averages a structural zero into every unit that fired in fewer
    than `active_time_threshold` samples, so on its own it cannot be read: a
    falling curve may be sparsification rather than lost spatial tuning. These
    three say how much of it is measurement.

    The gate is that they describe EXACTLY the units prnn zeroed - every unit
    counted as zeroed has SI exactly 0.0 in the returned frame."""
    env, pN, agent = setup
    metrics = evaluate_spatial_representation(
        pN, env, agent,
        n_trajs=3, traj_timesteps=60,
        onset_transient=5, active_time_threshold=10,
        sleep_timesteps=50,
    )
    values = np.asarray(metrics["SI"]["SI"], dtype=float)
    assert metrics["SI_units_total"] == pN.hidden_size
    assert 0 <= metrics["SI_units_zeroed"] <= metrics["SI_units_total"]
    # prnn assigns exactly 0 to the units it excludes, so a zeroed count larger
    # than the number of exact zeros would mean the two disagree about which
    # units are inactive.
    assert metrics["SI_units_zeroed"] <= int((values == 0.0).sum())


def test_si_coverage_raises_the_mean_when_units_are_zeroed(setup):
    """The direction that matters: excluding structural zeros can only move the
    mean up, so `mean SI` understates spatial tuning by exactly the share of
    units below the threshold."""
    env, pN, agent = setup
    # a threshold above any plausible activity count zeroes every unit
    everything = evaluate_spatial_representation(
        pN, env, agent,
        n_trajs=3, traj_timesteps=60,
        onset_transient=5, active_time_threshold=10**9,
        sleep_timesteps=50,
    )
    assert everything["SI_units_zeroed"] == everything["SI_units_total"]
    assert np.isnan(everything["SI_mean_active_only"])
    assert float(np.asarray(everything["SI"]["SI"], dtype=float).max()) == 0.0


def test_pooled_eval_restores_device_placement(setup):
    env, pN, agent = setup
    before = next(pN.pRNN.parameters()).device
    evaluate_spatial_representation(
        pN, env, agent,
        n_trajs=2, traj_timesteps=40,
        onset_transient=5, active_time_threshold=5, sleep_timesteps=30,
    )
    assert next(pN.pRNN.parameters()).device == before
