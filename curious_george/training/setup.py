"""Construction of everything a training run needs, from the hydra config.

Plain functions returning dataclasses; `setup_training` is the one-call
entry. Construction order matters for run-level reproducibility (torch RNG:
world model before AC model) and is preserved from the historical script.
"""

from dataclasses import dataclass
from typing import Callable

import datetime
import numpy as np
import torch
import torch.nn as nn

from prnn.utils import PredictiveNet, load_pN

from curious_george.utils.common import get_device, seed as seed_everything
from curious_george.envs.factory import make_env
from curious_george.models import ACModel, ACModelSR
from curious_george.rl.algo import PredictivePPOAlgo
from curious_george.rl.collect.format import get_obss_preprocessor
from curious_george.storage import (
    create_folders_if_necessary,
    get_agent,
    get_model_dir,
    get_video_dir,
)
from curious_george.utils.checkpoints import (
    StatusCkptKeys,
    load_statedict_from_acmodel_status,
    status_optimizer_matches,
)
from curious_george.utils.dev_env import get_ckpt_env_vars
from curious_george.utils.enums import AgentType

RAND_ACT_PROBA = np.array([0.15, 0.15, 0.6, 0.1])


@dataclass
class RunContext:
    run_name: str
    model_dir: str
    video_dir: str
    wandb_log: bool


@dataclass
class TrainingComponents:
    envs: list
    predictiveNet: PredictiveNet
    acmodel: nn.Module
    algo: PredictivePPOAlgo
    preprocess_obss: Callable
    obs_space: dict
    status: dict
    pastSR: bool
    random_agent: object
    ac_agent: object

    @property
    def env(self):
        return self.envs[0]


def setup_run(cfg) -> RunContext:
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    agent_type = "curious" if cfg.exp.curious_agent else "rand"
    run_name = f"{cfg.exp.exp_name}_{agent_type}_{date}"

    model_dir = get_model_dir(f"{run_name}/")
    create_folders_if_necessary(model_dir)

    if cfg.logging.video_log_freq != 0:
        video_dir = get_video_dir(f"{run_name}/")
        create_folders_if_necessary(video_dir)
    else:
        video_dir = ""

    print("\n\n\nLOGGING TO: ", model_dir, "\n\n\n")
    return RunContext(
        run_name=run_name,
        model_dir=model_dir,
        video_dir=video_dir,
        wandb_log=cfg.logging.wandb_log,
    )


def setup_env(cfg, seed_offset: int = 0, landmarks: list | None = None):
    start_room = None if cfg.exp.start_rand else cfg.exp.start_room
    # Only forwarded when set: not every registered env takes new_obj_pos, and
    # passing it unconditionally would break the ones that don't.
    obj = cfg.exp.get("new_obj_pos", None)
    extra = {"new_obj_pos": tuple(obj)} if obj else {}
    if landmarks is not None:
        # LEnv_multi has no default room; the pool overwrites this per episode,
        # so it only has to be a valid member of the run's layout set.
        extra["landmarks"] = list(landmarks)
    return make_env(
        **extra,
        env_key=cfg.exp.env_name,
        input_type=cfg.exp.input_type,
        seed=cfg.exp.seed + 10000 + seed_offset,
        act_enc=cfg.predNet.action_encoding,
        open_all_paths=False,  # Only applicable for FourRooms env
        subroom_size=cfg.exp.env_subroom_size,  # Only applicable for FourRooms env
        door_poss=cfg.exp.door_poss,  # Only applicable for FourRooms env
        agent_start_pos=None,
        agent_start_dir=None,
        agent_start_room=start_room,  # Only applicable for FourRooms env
        see_through_walls=cfg.exp.get("see_through_walls", None),
        table_env=(
            cfg.exp.get("table_env", False)
            or cfg.exp.get("device_env", False)
        ),
    )


def setup_envs(cfg) -> list:
    from curious_george.envs.layouts import resolve_layouts

    num_envs = cfg.exp.get("num_envs", 1)
    layouts = resolve_layouts(cfg)
    seed_landmarks = list(layouts[0].landmarks) if layouts else None
    if layouts:
        print(f"multi-room: {len(layouts)} layouts, resampled per episode")

    if cfg.exp.get("device_env", False):
        from curious_george.envs.vector import DeviceTableShellPool

        if num_envs <= 1:
            raise ValueError("exp.device_env requires exp.num_envs > 1")
        training_shells = [
            setup_env(cfg, seed_offset=1000 * i, landmarks=seed_landmarks)
            for i in range(num_envs)
        ]
        # Evaluation must not mutate one of the training reset RNG streams.
        eval_shell = setup_env(cfg, seed_offset=1000 * num_envs, landmarks=seed_landmarks)
        return DeviceTableShellPool(
            training_shells=training_shells,
            eval_shell=eval_shell,
            device=get_device(),
            layouts=layouts,
            layout_seed=int(cfg.exp.seed),
        )
    if layouts:
        raise ValueError(
            "multi-room training requires exp.device_env; the CPU and async "
            "paths hold one grid per stream for the whole run"
        )
    if num_envs > 1 and cfg.exp.get("async_envs", True):
        from curious_george.envs.vector import AsyncShellPool

        # workers use the same per-index seeds as the serial list; the eval
        # shell (analysis/plotting/pRNN services) gets its own stream so eval
        # rollouts no longer perturb training env streams
        eval_shell = setup_env(cfg, seed_offset=1000 * num_envs)
        return AsyncShellPool(cfg, eval_shell)
    return [setup_env(cfg, seed_offset=1000 * i) for i in range(num_envs)]


def load_status(cfg) -> dict:
    if cfg.logging.load_acmodel:
        _, acmodel_status_ckpt = get_ckpt_env_vars()
        return torch.load(acmodel_status_ckpt, map_location=get_device(), weights_only=False)
    return {StatusCkptKeys.NUM_FRAMES.value: 0, StatusCkptKeys.UPDATE.value: 0}


def setup_world_model(cfg, env, wandb_log: bool) -> PredictiveNet:
    predictiveNet = PredictiveNet(
        env,
        hidden_size=cfg.predNet.hiddensize,
        pRNNtype=cfg.predNet.pRNNtype,
        learningRate=cfg.predNet.lr,
        bptttrunc=cfg.predNet.bptttrunc,
        weight_decay=cfg.predNet.weight_decay,
        neuralTimescale=cfg.predNet.ntimescale,
        dropp=cfg.predNet.dropout,
        trainNoiseMeanStd=(cfg.predNet.noisemean, cfg.predNet.noisestd),
        f=cfg.predNet.sparsity,
        wandb_log=wandb_log,
    )
    cfg.predNet.hiddensize = predictiveNet.hidden_size
    predictiveNet.env_shell.hd_trans = np.array([-1, 1, 0, 0])  # TODO: remove later
    # (already the FaramaMinigridShell default; kept for parity)

    if cfg.logging.load_worldmodel:
        prnn_ckpt, _ = get_ckpt_env_vars()
        load_pN(
            model_ckpt_filepath=prnn_ckpt,
            device=get_device(),
            pRNNtype=cfg.predNet.pRNNtype,
            predictive_net=predictiveNet,
        )
        print(f"Existing pRNN model loaded from {prnn_ckpt}")

    return predictiveNet


def setup_acmodel(cfg, env, obs_space, status: dict) -> nn.Module:
    acmodel: nn.Module
    if cfg.exp.pRNN:
        acmodel = ACModelSR(
            obs_space,
            env.action_space,
            cfg.predNet.hiddensize,
            cfg.exp.with_obs,
            cfg.exp.rgb,
            cfg.exp.with_HD,
        )
    else:
        acmodel = ACModel(obs_space, env.action_space, cfg.exp.with_HD, cfg.exp.rgb)

    if StatusCkptKeys.MODEL_STATE.value in status:
        load_statedict_from_acmodel_status(
            receiver=acmodel,
            status=status,
            status_key=StatusCkptKeys.MODEL_STATE,
            device=get_device(),
        )
        print("Existing AC model loaded from status checkpoint")

    acmodel.to(get_device())
    return acmodel


def setup_algo(cfg, envs, acmodel, predictiveNet, preprocess_obss, status: dict,
               device: torch.device | None = None) -> PredictivePPOAlgo:
    device = get_device() if device is None else device
    if cfg.predNet.train:
        assert cfg.predNet.seqdur > 0, "Set an appropriate seqdur"
    else:
        cfg.predNet.seqdur = 0

    pastSR = "prevAct" not in predictiveNet.pRNNtype
    print("pastSR:", pastSR)

    algo = PredictivePPOAlgo(
        envs if len(envs) > 1 else envs[0],
        acmodel,
        predictiveNet,
        device,
        num_frames=cfg.rl.frames,
        discount=cfg.rl.discount,
        lr=cfg.rl.lr,
        gae_lambda=cfg.rl.gae_lambda,
        entropy_coef=cfg.rl.entropy_coef,
        value_loss_coef=cfg.rl.value_loss_coef,
        max_grad_norm=cfg.rl.max_grad_norm,
        adam_eps=cfg.rl.optim_eps,
        clip_eps=cfg.rl.ppo_clip_eps,
        epochs=cfg.rl.ppo_epochs,
        batch_size=cfg.rl.ppo_batch_size,
        preprocess_obss=preprocess_obss,
        train_pN=cfg.predNet.train,
        noise_mu=cfg.predNet.noisemean,
        noise_std=cfg.predNet.noisestd,
        prnn_seqdur=cfg.predNet.seqdur,
        batched_wm=cfg.predNet.get("batched_wm", False),
        cuda_graph=cfg.predNet.get("cuda_graph", False),
        batched_curiosity=cfg.predNet.get("batched_curiosity", False),
        compile_cell=cfg.predNet.get("compile_cell", False),
        wm_segment_stride=cfg.predNet.get("wm_segment_stride", 1),
        wm_pool_group=cfg.predNet.get("wm_pool_group", 0),
        policy_cuda_graph=cfg.rl.get("cuda_graph", False),
        rollout_cuda_graph=cfg.exp.get("rollout_cuda_graph", False),
        intrinsic=cfg.exp.intrinsic,
        k_int=cfg.rl.k_int,
        pastSR=pastSR,
        curious_agent=cfg.exp.curious_agent,
        k_curious=cfg.rl.k_curious,
        reward_alignment=cfg.rl.get("reward_alignment", "legacy"),
        loss=cfg.rl.get("loss", "ppo_clip"),
        adam_betas=cfg.rl.get("optim_betas", [0.9, 0.999]),
    )

    if StatusCkptKeys.OPTIMIZER_STATE.value in status:
        # Guard against a status.pt written before 2026-07-30 by a task run,
        # which stored the pRNN's 4-group RMSprop under this key. Loading that
        # into the AC model's 1-group Adam otherwise fails with an opaque
        # torch error.
        if not status_optimizer_matches(
            algo.optimizer, status[StatusCkptKeys.OPTIMIZER_STATE.value]
        ):
            raise ValueError(
                f"'{StatusCkptKeys.OPTIMIZER_STATE.value}' in this checkpoint does not "
                f"match the AC optimizer ({len(algo.optimizer.param_groups)} param "
                "group(s)). Pre-2026-07-30 task checkpoints stored the pRNN optimizer "
                "under this key; drop it from the status dict before loading."
            )
        load_statedict_from_acmodel_status(
            receiver=algo.optimizer,
            status=status,
            status_key=StatusCkptKeys.OPTIMIZER_STATE,
            device=device,
        )
        print("Optimizer loaded")

    return algo


def setup_training(cfg) -> TrainingComponents:
    """Build the full stack in the historical order (seed -> env -> status ->
    preprocessor -> world model -> AC model -> algo -> analysis agents)."""
    seed_everything(cfg.exp.seed)
    print(f"Device: {get_device()}\n")

    envs = setup_envs(cfg)
    env = envs[0]
    status = load_status(cfg)
    obs_space, preprocess_obss = get_obss_preprocessor(env.observation_space)
    predictiveNet = setup_world_model(cfg, env, cfg.logging.wandb_log)
    acmodel = setup_acmodel(cfg, env, obs_space, status)
    algo = setup_algo(cfg, envs, acmodel, predictiveNet, preprocess_obss, status)

    pastSR = algo.pastSR
    random_agent = get_agent(env=env, rand_act_prob=RAND_ACT_PROBA, agent_Type=AgentType.RANDOM)
    ac_agent = get_agent(
        env=env,
        agent_Type=AgentType.AC,
        prnn=predictiveNet,
        device=get_device(),
        ac_model=acmodel,
        pastSR=pastSR,
    )

    return TrainingComponents(
        envs=envs,
        predictiveNet=predictiveNet,
        acmodel=acmodel,
        algo=algo,
        preprocess_obss=preprocess_obss,
        obs_space=obs_space,
        status=status,
        pastSR=pastSR,
        random_agent=random_agent,
        ac_agent=ac_agent,
    )
