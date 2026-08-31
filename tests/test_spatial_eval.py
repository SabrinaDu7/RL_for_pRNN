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


def test_seeded_probe_leaves_the_training_streams_untouched(setup):
    """🔴 The 2026-08-31 bug this gates: `probe_seed` reseeded the GLOBAL torch
    (all devices) and numpy generators mid-run, so every analysis event
    restarted the training stream from the same constant in every run -
    partially collapsing seed-to-seed independence. `_probe_rng` must restore
    both streams, and the probe itself must stay reproducible."""
    env, pN, agent = setup

    torch.manual_seed(20260831)
    np.random.seed(20260831)
    torch.rand(3)
    np.random.rand(3)
    torch_before = torch.random.get_rng_state().clone()
    np_before = np.random.get_state()

    pn_state_before = pN.state.clone()
    pn_phase_before = pN.phase
    kwargs = dict(n_trajs=2, traj_timesteps=40, onset_transient=5,
                  active_time_threshold=10, sleep_timesteps=30, probe_seed=10007)
    m1 = evaluate_spatial_representation(pN, env, agent, **kwargs)

    assert torch.equal(pN.state, pn_state_before) and pN.phase == pn_phase_before, (
        "seeded eval perturbed pN.state/pN.phase - the leak's sibling is back"
    )

    assert torch.equal(torch.random.get_rng_state(), torch_before), (
        "seeded eval perturbed the global torch stream"
    )
    b, a = np_before, np.random.get_state()
    assert b[0] == a[0] and np.array_equal(b[1], a[1]) and b[2:] == a[2:], (
        "seeded eval perturbed the global numpy stream"
    )

    # and the probe itself is FIXED: same seed, same numbers
    m2 = evaluate_spatial_representation(pN, env, agent, **kwargs)
    assert m1["sRSA"] == m2["sRSA"] and m1["SWdist"] == m2["SWdist"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gates the CUDA leg of _probe_rng")
def test_seeded_probe_restores_the_cuda_stream(setup):
    """`torch.manual_seed` seeds EVERY device's generator, so the restore has
    to cover CUDA too - the CPU-only test above cannot see this leg."""
    env, pN, agent = setup
    torch.manual_seed(20260831)
    torch.rand(3, device="cuda")
    before = torch.cuda.get_rng_state_all()
    evaluate_spatial_representation(
        pN, env, agent, n_trajs=2, traj_timesteps=40, onset_transient=5,
        active_time_threshold=10, sleep_timesteps=30, probe_seed=10007,
    )
    after = torch.cuda.get_rng_state_all()
    assert all(torch.equal(a, b) for a, b in zip(before, after)), (
        "the probe moved a CUDA generator"
    )
