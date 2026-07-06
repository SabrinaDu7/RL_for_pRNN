"""Reusable evaluation-task machinery.

Most tasks follow one shape: load a checkpointed agent + pRNN, let the agent
explore a TRAINING env for a while (agent / world model each optionally
frozen), then freeze everything and collect stats (hidden states, positions,
actions, predictions, ...) from rollouts in an EVAL env.

The two envs are deliberately distinct fields: e.g. the Object Memory Task
trains in `env_train` (novel object present) but evaluates in `env_eval`
(object absent) to probe what the pRNN remembers. Never collapse them.

Construction reuses storage.get_pN / get_SR_acmodel / get_agent and
training.setup.setup_algo; the order matches the historical
ObjectMemoryTask.__init__ exactly (RNG parity, gated by tests/golden_omt).
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
from tqdm import tqdm

import wandb
from prnn.utils import PredictiveNet

from curious_george.rl.algo import PredictivePPOAlgo
from curious_george.rl.collect.format import get_obss_preprocessor
from curious_george.storage import get_agent, get_pN, get_SR_acmodel
from curious_george.training.setup import setup_algo
from curious_george.utils.common import seed as seed_everything
from curious_george.utils.common import synthesize
from curious_george.utils.enums import AgentType
from curious_george.world_model.device import on_device


@dataclass
class FreezeSpec:
    """What stays frozen during the task's training phase.

    world_model=True  -> the pRNN takes no gradient steps (and stays eval()).
    agent=True        -> the AC model takes no gradient steps.
    Everything is frozen during eval collection regardless.
    """

    world_model: bool = False
    agent: bool = False


@dataclass
class TaskComponents:
    env_train: object   # env the agent explores/trains in
    env_eval: object    # env eval rollouts run in (may differ on purpose!)
    pN: PredictiveNet           # the (possibly trained-further) world model
    pN_control: Optional[PredictiveNet]  # frozen pre-task copy, or None
    acmodel: object
    algo: PredictivePPOAlgo
    agent: object
    preprocess_obss: Callable
    obs_space: dict
    agent_type: AgentType
    freeze: FreezeSpec


def setup_task(
    cfg,
    *,
    env_train,
    env_eval,
    prnn_ckpt: str,
    acmodel_ckpt: str,
    agent_type: AgentType,
    device: torch.device,
    freeze: FreezeSpec = FreezeSpec(),
    control_copy: bool = True,
) -> TaskComponents:
    """Build the task stack in the historical order (RNG parity):
    seed -> pN from ckpt (on env_eval) -> deep copy for control ->
    preprocessor -> AC model from ckpt -> algo on env_train -> agent."""
    seed_everything(cfg.exp.seed)

    pN = get_pN(args=cfg, env=env_eval, device=device, pRNN_ckpt=prnn_ckpt)
    pN_control = None
    if control_copy:
        pN_control = pN  # the loaded net becomes the frozen control...
        pN = pN_control.copy()  # ...and its deep copy is the one the task trains
        pN_control.pRNN.eval()
    pN.pRNN.train()

    pN.wandb_log = cfg.logging.wandb_log
    if pN_control is not None:
        pN_control.wandb_log = cfg.logging.wandb_log

    obs_space, preprocess_obss = get_obss_preprocessor(env_train.observation_space)
    acmodel = get_SR_acmodel(
        cfg,
        env_act_space=env_train.action_space,
        obs_space=obs_space,
        acmodel_status_ckpt=acmodel_ckpt,
        device=device,
    )

    status = torch.load(acmodel_ckpt, map_location=device, weights_only=False)
    algo = setup_algo(cfg, [env_train], acmodel, pN, preprocess_obss, status, device=device)
    # NOTE: freezing the world model must NOT zero prnn_seqdur (episode cuts
    # still apply), so it is applied post-construction rather than via
    # cfg.predNet.train.
    if freeze.world_model:
        algo.train_pN = False
        pN.pRNN.eval()

    agent = get_agent(
        env=env_train,
        agent_Type=agent_type,
        ac_model=acmodel,
        prnn=pN,
        device=device,
    )

    return TaskComponents(
        env_train=env_train,
        env_eval=env_eval,
        pN=pN,
        pN_control=pN_control,
        acmodel=acmodel,
        algo=algo,
        agent=agent,
        preprocess_obss=preprocess_obss,
        obs_space=obs_space,
        agent_type=agent_type,
        freeze=freeze,
    )


def train_phase(
    comps: TaskComponents,
    *,
    num_batches: int,
    trajs_per_batch: int,
    seqdur: int,
    saving_interval: int,
    analysis_interval: int,
    wandb_log: bool,
    on_save: Optional[Callable[[int], None]] = None,
    on_analysis: Optional[Callable[[int, int], None]] = None,
) -> None:
    """The exploration/training loop shared by tasks.

    Each batch = one collect_experiences (+ update unless frozen). Task
    specifics (learning-rate scaling, figures, extra logging) go through the
    on_save(index) / on_analysis(index, traj_count) callbacks.
    """
    algo = comps.algo
    for index in tqdm(range(num_batches)):
        traj_count = (index + 1) * trajs_per_batch
        step_count = traj_count * seqdur

        if wandb_log:
            wandb.log({"step_count": step_count})

        if comps.agent_type == AgentType.RANDOM:
            exp_logs = algo.randomAgent_collect_exp_and_update(comps.agent)
        else:
            exps, exp_logs = algo.collect_experiences()
            logs2 = algo.update_parameters(
                exps=exps, update_params=not comps.freeze.agent
            )

            if wandb_log:
                cur_rewards = synthesize(exp_logs["curious_rewards"], abs=True)
                wandb.log({"Train/cur_rewards": cur_rewards["mean"]})
                wandb.log({"Train/cur_rewards": cur_rewards["std"]})
                for key, val in logs2.items():
                    wandb.log({f"Train/{key}": val})

                for key, val in exp_logs.items():
                    if key.startswith("avg_adv") or key.startswith("curious_reward"):
                        wandb.log({f"Train/{key}": val})

        if on_save is not None and index % saving_interval == 0:
            on_save(index)

        if on_analysis is not None and index % analysis_interval == 0:
            with torch.no_grad():
                on_analysis(index, traj_count)


@dataclass
class EvalRollouts:
    """Stats collected from frozen eval rollouts (B trajectories of T steps)."""

    obs: torch.Tensor        # (B, T+1, X) pred-format observations
    actions: torch.Tensor    # (B, T, A) encoded actions
    agent_pos: np.ndarray    # (B, T+1, 2)
    agent_dir: np.ndarray    # (B, T+1)
    renders: Optional[np.ndarray]  # (B, T+1, H, W, C) or None
    extras: dict = field(default_factory=dict)  # task-specific stacked stats


def collect_eval_rollouts(
    *,
    env_eval,
    agent,
    pN: PredictiveNet,
    n_trajs: int,
    T: int,
    eval_modules: list,
    include_render: bool = False,
    before_each: Optional[Callable[[], None]] = None,
    traj_stats_fn: Optional[Callable] = None,
) -> EvalRollouts:
    """Serial eval collection, bitwise-faithful to the historical B-loop.

    Everything runs on CPU (numpy interop) with placement restored on exit.
    Preserved quirk: pN.state is NOT reset between trajectories - the pRNN
    hidden state carries across the whole loop (historical behavior encoded
    in tests/golden_omt).

    traj_stats_fn(obs, act, state, render) -> dict[str, torch.Tensor] adds
    per-trajectory task stats (e.g. OMT's dual-net predictions); returned
    tensors are stacked into extras[key] of shape (B, ...).
    """
    render_size = env_eval.render_size
    _, view_size, C = env_eval.obs_shape
    B, X = n_trajs, None

    with on_device(eval_modules, "cpu"):
        all_obs = torch.zeros((B, T + 1, view_size * view_size * C), dtype=torch.float32)
        all_act = None
        all_agent_pos = np.zeros((B, T + 1, 2), dtype=np.float32)
        all_agent_dir = np.zeros((B, T + 1), dtype=np.int32)
        all_renders = (
            np.zeros((B, T + 1, render_size, render_size, C), dtype=np.uint8)
            if include_render else None
        )
        extras_lists: dict[str, list] = {}

        for n in range(B):
            if before_each is not None:
                before_each()
            obs, act, state, render = pN.collectObservationSequence(
                env=env_eval, agent=agent, tsteps=T, includeRender=include_render,
            )

            if traj_stats_fn is not None:
                for key, val in traj_stats_fn(obs, act, state, render).items():
                    extras_lists.setdefault(key, []).append(val)

            all_obs[n, :, :] = obs
            if all_act is None:
                all_act = torch.zeros((B, *act.shape[1:]), dtype=act.dtype)
            all_act[n] = act
            all_agent_pos[n, :, :] = state["agent_pos"]
            all_agent_dir[n, :] = state["agent_dir"]
            if include_render:
                all_renders[n, :, :, :, :] = np.stack(render, axis=0)

    extras = {k: torch.stack([v.squeeze(0) for v in vals], dim=0)
              for k, vals in extras_lists.items()}
    return EvalRollouts(
        obs=all_obs,
        actions=all_act,
        agent_pos=all_agent_pos,
        agent_dir=all_agent_dir,
        renders=all_renders,
        extras=extras,
    )
