"""Gates for the correctness fixes of the 2026-09-05 audit (C4, C5, C6, C15).

Each test names the audit item it pins. C2, C3 and C8 are gated beside the
code they changed (tests/test_configs.py, tests/test_training_schedule.py).
"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from curious_george.configs import EnvBackend
from tests.small_config import small_config


def test_c4_a_failed_wandb_init_reaches_the_world_models_own_logger(tmp_path, monkeypatch):
    """`PredictiveNet` logs through the fork on a flag it was built with, before
    wandb existed. The degrade path flipped only the run context, so the run
    survived `init` and died at the first world-model step - wandb 0.28 raises
    on `log` before `init`."""
    import main_train

    monkeypatch.setenv("RL_STORAGE", str(tmp_path))
    seen: dict = {}

    def failing_init(cfg, run_ctx):
        raise RuntimeError("no network on this compute node")

    def recording_run(cfg, run_ctx, comps):
        seen["wandb_log"] = run_ctx.wandb_log
        seen["pN_wandb_log"] = comps.predictiveNet.wandb_log

    monkeypatch.setattr(main_train, "init_wandb", failing_init)
    monkeypatch.setattr(main_train, "run_training", recording_run)
    cfg = small_config(backend=EnvBackend.SERIAL_TABLE, num_envs=2, rollouts=1)
    main_train.train(replace(cfg, run=replace(cfg.run, wandb=True)))
    assert seen == {"wandb_log": False, "pN_wandb_log": False}


def test_c5_a_frozen_world_model_keeps_the_episode_cut():
    """`train_prnn.train=False` zeroed `prnn_seqdur` - no cuts on the serial
    backend, a refusal on the device backend - against its own comment."""
    from curious_george.training.setup import setup_training

    cfg = small_config(backend=EnvBackend.SERIAL_TABLE, num_envs=2, train_prnn=False)
    comps = setup_training(cfg)
    assert comps.algo.prnn_seqdur == cfg.collect.episode_steps
    assert comps.algo.train_pN is False
    before = {k: v.clone() for k, v in comps.predictiveNet.pRNN.state_dict().items()}
    exps, logs = comps.algo.collect_experiences()
    comps.algo.update_parameters(exps=exps)
    # episodes were cut at episode_steps: one segment per stream
    assert len(logs["num_frames_per_episode"]) == cfg.collect.episodes_per_rollout
    # and the frozen net did not move
    for k, v in comps.predictiveNet.pRNN.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_c6_the_batched_eval_collector_runs():
    """It handed `BatchedSRTrackerShim` the stream COUNT where the shim takes
    the list of initial observations: `len(int)`, a TypeError on first use."""
    from curious_george.evaluation.task import collect_eval_rollouts_batched
    from curious_george.training.setup import setup_training

    comps = setup_training(small_config(backend=EnvBackend.SERIAL_TABLE, num_envs=2))
    T = 5
    rollouts = collect_eval_rollouts_batched(
        envs_eval=list(comps.envs),
        agent=comps.ac_agent,
        pN=comps.predictiveNet,
        T=T,
        eval_modules=[comps.predictiveNet, comps.acmodel],
    )
    assert rollouts.obs.shape[:2] == (2, T + 1)
    assert rollouts.actions.shape[:2] == (2, T)
    assert rollouts.agent_pos.shape == (2, T + 1, 2)
    assert np.isfinite(rollouts.obs).all()


@pytest.mark.parametrize("probe_seed", [None, 7])
def test_c15_the_multi_room_eval_leaves_the_shell_as_it_found_it(probe_seed):
    """The eval rebinds the shell's landmarks room by room; the trajectory
    figure drawn after it showed whichever room was scored last."""
    from prnn.utils import PredictiveNet, RandomActionAgent

    from curious_george.configs import RAND_ACT_PROBA
    from curious_george.envs.layouts import MULTI_ROOM_ID, ROOMS_SELECTED, BASE_ROOM_ID
    from curious_george.envs.factory import make_env
    from curious_george.evaluation.spatial import evaluate_multi_room_representation

    torch.manual_seed(0)
    np.random.seed(0)
    env = make_env(
        env_key=MULTI_ROOM_ID[BASE_ROOM_ID], input_type="pRNN", act_enc="SpeedHD",
        seed=0, landmarks=list(ROOMS_SELECTED[0].landmarks),
    )
    pN = PredictiveNet(env, hidden_size=16, pRNNtype="thRNN_5win",
                       trainNoiseMeanStd=(0, 0.0), wandb_log=False)
    found = list(env.env.unwrapped.landmarks)
    evaluate_multi_room_representation(
        pN, env, RandomActionAgent(env.action_space, np.asarray(RAND_ACT_PROBA)),
        layouts=list(ROOMS_SELECTED[:2]), n_trajs=1, traj_timesteps=30,
        onset_transient=2, active_time_threshold=2, sleep_timesteps=10,
        probe_seed=probe_seed,
    )
    assert list(env.env.unwrapped.landmarks) == found
