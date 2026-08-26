"""Construction of everything a training run needs, from the typed config.

Plain functions returning dataclasses; `setup_training` is the one-call
entry. Construction order matters for run-level reproducibility (torch RNG:
world model before AC model) and is preserved from the historical script.
"""

from dataclasses import dataclass, replace
from typing import Callable

import datetime
import numpy as np
import torch
import torch.nn as nn

from prnn.utils import PredictiveNet, load_pN

from curious_george.log_and_store import provenance
from curious_george.utils.common import get_device, seed as seed_everything
from curious_george.envs.factory import make_env
from curious_george.configs import EnvBackend, FourRoomsCfg
from curious_george.models.policy import ACModelSR
from curious_george.rl.algo import PredictivePPOAlgo
from curious_george.rl.collect.format import get_obss_preprocessor
from curious_george.log_and_store.storage import (
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
from curious_george.utils.enums import AgentType
from curious_george.training.schedule import TrainingSchedule

RAND_ACT_PROBA = np.array([0.15, 0.15, 0.6, 0.1])


@dataclass
class RunContext:
    run_name: str
    model_dir: str
    video_dir: str
    wandb_log: bool


@dataclass
class TrainingComponents:
    #: The EFFECTIVE config - what was built, not what was asked for. Differs
    #: from the caller's config wherever a constructor overrode a request.
    cfg: object
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
    run_name = f"{cfg.run.exp_name}_{cfg.arch_policy.agent.value}_{date}"

    model_dir = get_model_dir(f"{run_name}/")
    create_folders_if_necessary(model_dir)

    if cfg.run.video_every_episodes != 0:
        video_dir = get_video_dir(f"{run_name}/")
        create_folders_if_necessary(video_dir)
    else:
        video_dir = ""

    # Before this, main_train.py only PRINTED the resolved config, so a finished
    # run carried no record of what produced it and could not be placed on any
    # of the timelines in docs/invalid-runs.md.
    provenance.write(
        model_dir,
        kind="training",
        params={
            "run_name": run_name,
            "config": cfg.to_dict(),
            # The RESOLVED budget beside the config that asked for it, so a run
            # can be placed on a step axis without re-deriving anything.
            "schedule": cfg.schedule.as_dict(),
        },
    )

    print("\n\n\nLOGGING TO: ", model_dir, "\n\n\n")
    return RunContext(
        run_name=run_name,
        model_dir=model_dir,
        video_dir=video_dir,
        wandb_log=cfg.run.wandb,
    )


def setup_env(cfg, seed_offset: int = 0, landmarks: list | None = None):
    # Only forwarded when set: not every registered env takes new_obj_pos, and
    # passing it unconditionally would break the ones that don't.
    obj = getattr(cfg.env, "new_obj_pos", None)
    extra = {"new_obj_pos": tuple(obj)} if obj else {}
    if isinstance(cfg.env, FourRoomsCfg):
        # The ONLY environment whose constructor reads these. MiniGridEnv takes
        # **kwargs and DISCARDS them, so passing them to an L-room was dead
        # config that read as live.
        extra |= {
            "subroom_size": cfg.env.subroom_size,
            "door_poss": list(cfg.env.door_poss),
            "agent_start_room": cfg.env.start_room,
            "open_all_paths": False,
        }
    if landmarks is not None:
        # LEnv_multi has no default room; the pool overwrites this per episode,
        # so it only has to be a valid member of the run's layout set.
        extra["landmarks"] = list(landmarks)
    return make_env(
        **extra,
        env_key=cfg.env.env_name.value,
        input_type=cfg.arch_policy.input_type.value,
        seed=cfg.run.seed + 10000 + seed_offset,
        act_enc=cfg.arch_prnn.action_encoding.value,
        agent_start_pos=None,
        agent_start_dir=None,
        see_through_walls=cfg.env.see_through_walls,
        table_env=cfg.collect.backend.tabled,
    )


def setup_envs(cfg) -> list:
    from curious_george.envs.layouts import resolve_layouts

    num_envs = cfg.collect.num_envs
    layouts = resolve_layouts(cfg)
    seed_landmarks = list(layouts[0].landmarks) if layouts else None
    if layouts:
        print(f"multi-room: {len(layouts)} layouts, resampled per episode")

    if cfg.collect.backend.batched:
        from curious_george.envs.vector import DeviceTableShellPool

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
            layout_seed=cfg.run.seed,
        )
    if num_envs > 1 and cfg.collect.backend in (EnvBackend.ASYNC, EnvBackend.ASYNC_TABLE):
        from curious_george.envs.vector import AsyncShellPool

        # workers use the same per-index seeds as the serial list; the eval
        # shell (analysis/plotting/pRNN services) gets its own stream so eval
        # rollouts no longer perturb training env streams
        eval_shell = setup_env(cfg, seed_offset=1000 * num_envs)
        return AsyncShellPool(cfg, eval_shell)
    return [setup_env(cfg, seed_offset=1000 * i) for i in range(num_envs)]


def load_status(cfg) -> dict:
    if cfg.run.policy_ckpt is not None:
        return torch.load(
            cfg.run.policy_ckpt, map_location=get_device(), weights_only=False
        )
    return {StatusCkptKeys.NUM_FRAMES.value: 0, StatusCkptKeys.UPDATE.value: 0}


def setup_world_model(cfg, env, wandb_log: bool) -> PredictiveNet:
    predictiveNet = PredictiveNet(
        env,
        hidden_size=cfg.arch_prnn.hidden_size,
        pRNNtype=cfg.arch_prnn.prnn_type.value,
        learningRate=cfg.train_prnn.lr,
        bptttrunc=cfg.train_prnn.bptt_trunc,
        weight_decay=cfg.train_prnn.weight_decay,
        neuralTimescale=cfg.arch_prnn.n_timescale,
        dropp=cfg.arch_prnn.dropout,
        trainNoiseMeanStd=(cfg.arch_prnn.noise_mean, cfg.arch_prnn.noise_std),
        f=cfg.arch_prnn.sparsity,
        wandb_log=wandb_log,
    )
    predictiveNet.env_shell.hd_trans = np.array([-1, 1, 0, 0])  # TODO: remove later
    # (already the FaramaMinigridShell default; kept for parity)

    if cfg.run.prnn_ckpt is not None:
        load_pN(
            model_ckpt_filepath=str(cfg.run.prnn_ckpt),
            device=get_device(),
            pRNNtype=cfg.arch_prnn.prnn_type.value,
            predictive_net=predictiveNet,
        )
        print(f"Existing pRNN model loaded from {cfg.run.prnn_ckpt}")

    return predictiveNet


def setup_acmodel(cfg, env, obs_space, status: dict) -> nn.Module:
    acmodel: nn.Module = ACModelSR(
        obs_space,
        env.action_space,
        cfg.arch_prnn.hidden_size,
        cfg.arch_policy.with_obs,
        cfg.arch_policy.rgb,
        cfg.arch_policy.with_head_direction,
    )

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
    schedule = TrainingSchedule.from_config(cfg)
    # A frozen pRNN still needs episode cuts, so segmenting no longer depends on
    # whether it trains - the old code zeroed seqdur here, mutating the config
    # AFTER provenance had already recorded it.
    prnn_seqdur = cfg.collect.episode_steps if cfg.train_prnn.train else 0

    pastSR = "prevAct" not in predictiveNet.pRNNtype
    print("pastSR:", pastSR)

    algo = PredictivePPOAlgo(
        envs if len(envs) > 1 else envs[0],
        acmodel,
        predictiveNet,
        device,
        num_frames=schedule.env_steps_per_rollout,
        discount=cfg.train_policy.discount,
        lr=cfg.train_policy.lr,
        gae_lambda=cfg.train_policy.gae_lambda,
        entropy_coef=cfg.train_policy.entropy_coef,
        value_loss_coef=cfg.train_policy.value_loss_coef,
        max_grad_norm=cfg.train_policy.max_grad_norm,
        adam_eps=cfg.train_policy.optim_eps,
        clip_eps=cfg.train_policy.clip_eps,
        epochs=cfg.train_policy.ppo_epochs,
        batch_size=schedule.ppo_batch_size,
        preprocess_obss=preprocess_obss,
        train_pN=cfg.train_prnn.train,
        noise_mu=cfg.arch_prnn.noise_mean,
        noise_std=cfg.arch_prnn.noise_std,
        prnn_seqdur=prnn_seqdur,
        batched_wm=cfg.train_prnn.batched,
        cuda_graph=cfg.train_prnn.cuda_graph,
        batched_curiosity=cfg.train_prnn.batched_curiosity,
        curiosity_cuda_graph=cfg.train_prnn.curiosity_cuda_graph,
        compile_cell=cfg.train_prnn.compile.adapter_arg,
        # One number, two regimes: pooled AVERAGES the group's gradient, serial
        # takes one episode and drops the rest. The old pair of fields could
        # disagree, and 0 meant "all" - a sentinel whose value depended on the
        # rollout size.
        wm_segment_stride=1 if cfg.train_prnn.batched else cfg.train_prnn.episodes_per_grad_step,
        wm_pool_group=cfg.train_prnn.episodes_per_grad_step if cfg.train_prnn.batched else 0,
        policy_cuda_graph=cfg.train_policy.cuda_graph,
        rollout_cuda_graph=cfg.collect.rollout_cuda_graph,
        intrinsic=cfg.train_policy.intrinsic,
        k_int=cfg.train_policy.k_intrinsic,
        pastSR=pastSR,
        curious_agent=cfg.train_policy.curious,
        k_curious=cfg.train_policy.k_curious,
        reward_alignment=cfg.train_policy.reward_alignment.value,
        loss="ppo_clip",
        adam_betas=list(cfg.train_policy.optim_betas),
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
    seed_everything(cfg.run.seed)
    print(f"Device: {get_device()}\n")

    envs = setup_envs(cfg)
    env = envs[0]
    status = load_status(cfg)
    obs_space, preprocess_obss = get_obss_preprocessor(env.observation_space)
    predictiveNet = setup_world_model(cfg, env, cfg.run.wandb)
    # PredictiveNet may round the requested width. Fold the EFFECTIVE value back
    # in HERE, before provenance or wandb see the config: the old code mutated
    # it after both had already recorded the requested one, so every
    # provenance.json carried a width the run did not use.
    if predictiveNet.hidden_size != cfg.arch_prnn.hidden_size:
        cfg = replace(cfg, arch_prnn=replace(
            cfg.arch_prnn, hidden_size=predictiveNet.hidden_size))
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
        cfg=cfg,
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
