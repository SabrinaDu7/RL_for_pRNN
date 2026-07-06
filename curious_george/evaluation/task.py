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
    envs_train: list    # env(s) the agent explores/trains in (B >= 1)
    env_eval: object    # env eval rollouts run in (may differ on purpose!)

    @property
    def env_train(self):
        return self.envs_train[0]
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
    preprocessor -> AC model from ckpt -> algo on env_train -> agent.

    env_train may be a single shell or a list (parallel train-phase
    collection via the unified algo)."""
    envs_train = env_train if isinstance(env_train, (list, tuple)) else [env_train]
    env_train = envs_train[0]
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
    algo = setup_algo(cfg, list(envs_train), acmodel, pN, preprocess_obss, status, device=device)
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
        envs_train=list(envs_train),
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


def collect_eval_rollouts_batched(
    *,
    envs_eval: list,
    agent,
    pN: PredictiveNet,
    T: int,
    eval_modules: list,
    include_render: bool = False,
    before_each: Optional[Callable[[object], None]] = None,
    traj_stats_fn: Optional[Callable] = None,
) -> EvalRollouts:
    """Batched eval collection: one trajectory per env copy, stepped in
    lockstep with ONE batched AC forward and ONE batched pRNN step per
    timestep (BatchedSRTrackerShim - the direct pN.pRNN batched path, not
    the buggy predict(batched=True)).

    NOT bit-comparable to the serial collector: per-env pRNN states start
    from zeros (no cross-trajectory pN.state carry-over) and RNG order
    differs. Zero-noise equivalence is pinned by tests/test_omt_batched.py.

    Per-trajectory stats still go through the SAME traj_stats_fn as the
    serial path (called with per-env slices), so tasks are mode-agnostic.
    before_each(env) is called per env copy before its reset (e.g. to set
    the start position).
    """
    from curious_george.world_model.adapter import BatchedSRTrackerShim, PRNNAdapter

    assert hasattr(agent, "acmodel"), "batched eval requires an ActorCriticAgent"
    B = len(envs_eval)
    render_size = envs_eval[0].render_size
    _, view_size, C = envs_eval[0].obs_shape

    with on_device(eval_modules, "cpu"):
        device = torch.device("cpu")
        adapter = PRNNAdapter(pN, device, agent.pastSR)
        tracker = BatchedSRTrackerShim(adapter, B)
        preprocess = get_obss_preprocessor(envs_eval[0].observation_space)[1]

        raw_obs = [[] for _ in range(B)]     # per env: T+1 raw obs dicts
        raw_act = [[] for _ in range(B)]     # per env: T action ids
        all_agent_pos = np.zeros((B, T + 1, 2), dtype=np.float32)
        all_agent_dir = np.zeros((B, T + 1), dtype=np.int32)
        all_renders = (
            np.zeros((B, T + 1, render_size, render_size, C), dtype=np.uint8)
            if include_render else None
        )

        obs_b = []
        for b, env in enumerate(envs_eval):
            if before_each is not None:
                before_each(env)
            obs = env.reset()
            obs_b.append(obs)
            raw_obs[b].append(obs)
            all_agent_pos[b, 0] = env.get_agent_pos()
            all_agent_dir[b, 0] = env.get_agent_dir()
            if include_render:
                all_renders[b, 0] = env.render(mode=None)

        sr = tracker.initial_sr()
        for t in range(T):
            preprocessed = preprocess(obs_b, device=device)
            with torch.no_grad():
                dist, _ = agent.acmodel(preprocessed, SR=sr)
            det_np = dist.probs.argmax(dim=-1).cpu().numpy()  # frozen eval: argmax

            pre_obs_b = list(obs_b)
            for b, env in enumerate(envs_eval):
                obs_next = env.step(det_np[b:b + 1])[0]
                raw_act[b].append(det_np[b])
                raw_obs[b].append(obs_next)
                obs_b[b] = obs_next
                all_agent_pos[b, t + 1] = env.get_agent_pos()
                all_agent_dir[b, t + 1] = env.get_agent_dir()
                if include_render:
                    all_renders[b, t + 1] = env.render(mode=None)

            sr = tracker.step(det_np, pre_obs_b, obs_b)

        # format per env (obs -> pred format) and run per-trajectory stats
        all_obs = torch.zeros((B, T + 1, view_size * view_size * C), dtype=torch.float32)
        all_act = None
        extras_lists: dict[str, list] = {}
        for b in range(B):
            obs_f, act_f = pN.env_shell.env2pred(raw_obs[b], np.array(raw_act[b]))
            all_obs[b] = obs_f
            if all_act is None:
                all_act = torch.zeros((B, *act_f.shape[1:]), dtype=act_f.dtype)
            all_act[b] = act_f

            if traj_stats_fn is not None:
                state = {
                    "agent_pos": all_agent_pos[b],
                    "agent_dir": all_agent_dir[b],
                }
                render = list(all_renders[b]) if include_render else False
                for key, val in traj_stats_fn(obs_f, act_f, state, render).items():
                    extras_lists.setdefault(key, []).append(val)

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
