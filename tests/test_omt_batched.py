"""Batched eval collection vs the serial path.

With zero pRNN noise, fixed start positions, and pN.state reset to zeros
before each serial trajectory (matching the batched tracker's per-env zero
init), the two collectors must agree to float tolerance. Note this is a
CONTROLLED comparison: the production serial path deliberately carries
pN.state across trajectories (golden_omt pins that quirk), so serial and
batched production runs are equivalent only distributionally.
"""

import numpy as np
import pytest
import torch

from prnn.utils import PredictiveNet, MinigridEnvNames, ActionEncodingsEnum
from curious_george import ACModelSR, AgentInputType, ActorCriticAgent, make_env, seed
from curious_george.evaluation.task import (
    collect_eval_rollouts,
    collect_eval_rollouts_batched,
)
from curious_george.envs.access import base_env

B, T, SEED = 3, 6, 9
START = ([3, 4], 1)  # same fixed start for every trajectory


def _make_env(s=SEED):
    return make_env(
        env_key=MinigridEnvNames.LRoom,
        input_type=AgentInputType.H_PO.value,
        act_enc=ActionEncodingsEnum.SpeedHD.value,
        seed=s,
    )


def _fix_start(env):
    base_env(env).agent_start_pos = np.array(START[0])
    base_env(env).agent_start_dir = START[1]


@pytest.fixture(scope="module")
def setup():
    seed(SEED)
    env = _make_env()
    pN = PredictiveNet(env, hidden_size=32, pRNNtype="thRNN_5win",
                       trainNoiseMeanStd=(0, 0), wandb_log=False)
    pN.env_shell.hd_trans = np.array([-1, 1, 0, 0])
    obs_space = None
    from curious_george import get_obss_preprocessor
    obs_space, _ = get_obss_preprocessor(env.observation_space)
    acmodel = ACModelSR(obs_space, env.action_space, 32, False, True, True)
    agent = ActorCriticAgent(env.action_space, acmodel, pN, torch.device("cpu"),
                             argmax=True, pastSR=True)
    # the real task always collects under eval_mode (dropout off); without it
    # predict() is stochastic and no equivalence can hold
    pN.pRNN.eval()
    acmodel.eval()
    return env, pN, acmodel, agent


def _stats_fn(pN):
    def fn(obs, act, state, render):
        with torch.no_grad():
            zero_state = torch.zeros((1, 1, 32))
            pred, _, _ = pN.predict(obs, act, state=zero_state, randInit=False)
        return {"obs_pred": pred}
    return fn


def test_batched_matches_serial_zero_noise(setup):
    env, pN, acmodel, agent = setup
    eval_modules = [pN, acmodel]

    # serial: reset pN.state to zeros before every trajectory (controlled)
    def before_serial():
        pN.reset_state(randInit=False)
        _fix_start(env)

    serial = collect_eval_rollouts(
        env_eval=env, agent=agent, pN=pN, n_trajs=B, T=T,
        eval_modules=eval_modules, before_each=before_serial,
        traj_stats_fn=_stats_fn(pN),
    )

    envs = [_make_env(SEED + i) for i in range(B)]
    batched = collect_eval_rollouts_batched(
        envs_eval=envs, agent=agent, pN=pN, T=T,
        eval_modules=eval_modules, before_each=_fix_start,
        traj_stats_fn=_stats_fn(pN),
    )

    assert torch.allclose(serial.obs, batched.obs, atol=1e-6)
    assert torch.equal(serial.actions, batched.actions)
    assert np.array_equal(serial.agent_pos, batched.agent_pos)
    assert np.array_equal(serial.agent_dir, batched.agent_dir)
    assert torch.allclose(serial.extras["obs_pred"], batched.extras["obs_pred"], atol=1e-6)


def test_batched_shapes_and_env_guard(setup):
    env, pN, acmodel, agent = setup
    envs = [_make_env(SEED + i) for i in range(B)]
    for e in envs:
        assert e.get_new_obj_pos() is None  # eval envs must be object-free
    batched = collect_eval_rollouts_batched(
        envs_eval=envs, agent=agent, pN=pN, T=T,
        eval_modules=[pN, acmodel], before_each=_fix_start,
        traj_stats_fn=_stats_fn(pN),
    )
    assert batched.obs.shape[0] == B and batched.obs.shape[1] == T + 1
    assert batched.extras["obs_pred"].shape[:2] == (B, T)
    assert torch.isfinite(batched.obs).all()
