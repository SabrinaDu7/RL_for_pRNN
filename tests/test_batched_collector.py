"""Smoke + invariant tests for the unified PredictivePPOAlgo at num_envs > 1."""

import numpy as np
import pytest
import torch

from prnn.utils import PredictiveNet, MinigridEnvNames, ActionEncodingsEnum
from curious_george import AgentInputType
import curious_george as cg
from curious_george.rl.algo import PredictivePPOAlgo

B, T, SEQDUR = 2, 32, 16  # num_frames = B*T = 64
SEED = 7


@pytest.fixture(scope="module")
def batched_algo():
    cg.seed(SEED)
    envs = [
        cg.make_env(
            env_key=MinigridEnvNames.LRoom,
            input_type=AgentInputType.H_PO.value,
            act_enc=ActionEncodingsEnum.SpeedHD.value,
            seed=SEED + 10000 + 1000 * i,
        )
        for i in range(B)
    ]
    pN = PredictiveNet(
        envs[0], hidden_size=64, pRNNtype="thRNN_5win",
        trainNoiseMeanStd=(0, 0.05), wandb_log=False,
    )
    pN.env_shell.hd_trans = np.array([-1, 1, 0, 0])
    obs_space, preprocess_obss = cg.get_obss_preprocessor(envs[0].observation_space)
    acmodel = cg.ACModelSR(obs_space, envs[0].action_space, 64, False, True, True)

    return PredictivePPOAlgo(
        envs, acmodel, pN, torch.device("cpu"),
        num_frames=B * T, discount=0.98, lr=3e-4, gae_lambda=0.95,
        entropy_coef=0.0, value_loss_coef=1, max_grad_norm=0.5, adam_eps=1e-8,
        clip_eps=0.2, epochs=4, batch_size=16, preprocess_obss=preprocess_obss,
        train_pN=True, noise_mu=0, noise_std=0.05, prnn_seqdur=SEQDUR,
        intrinsic=False, k_int=1, pastSR=True, curious_agent=True, k_curious=1,
    )


def test_collect_and_update(batched_algo):
    algo = batched_algo
    epochs_before = algo.pN.numTrainingEpochs

    exps, logs = algo.collect_experiences()
    assert exps.action.shape == (B * T,)
    assert exps.SR.shape == (B * T, 64)
    assert torch.isfinite(exps.advantage).all()
    assert torch.isfinite(algo.curious_rewards).all()
    assert (algo.curious_rewards >= 0).all()

    # env-major episode boundaries: every env edge b*T is a boundary, and
    # seqdur forces boundaries inside each env stream
    d = logs and algo.done_indices
    assert d[0] == 0 and d[-1] == B * T
    for b in range(B + 1):
        assert b * T in d
    for b in range(B):
        assert b * T + SEQDUR in d

    logs2 = algo.update_parameters(exps=exps, update_params=True)
    assert np.isfinite(logs2["policy_loss"])
    # pRNN trained once per episode segment
    assert algo.pN.numTrainingEpochs == epochs_before + (len(d) - 1)


def test_second_round_runs(batched_algo):
    exps, logs = batched_algo.collect_experiences()
    assert logs["num_frames"] == B * T
    assert len(logs["locs"]) == B * T
    batched_algo.update_parameters(exps=exps, update_params=True)


def test_joint_dist_counts_all_frames(batched_algo):
    _, logs = batched_algo.collect_experiences()
    # each frame contributes one probability row summing to 1
    assert np.isclose(logs["joint_dist"].sum(), B * T, atol=1e-3)
