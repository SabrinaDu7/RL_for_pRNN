# Based on the PPO algo from torch-ac library (https://github.com/lcswillems/torch-ac)

"""One algo class for B >= 1 environments.

`PredictivePPOAlgo` wires together:
- rl.collect.collector  - the rollout loop (B=1 is bitwise-identical to the
  historical serial path via SingleSRTracker; B>1 batches the forwards)
- rl.update.updater / losses - loss-agnostic policy updates (rl.loss config)
- rl.update.rewards / advantage - reward terms and GAE
- rl.update.world_model - per-episode pRNN training
- world_model.adapter   - the only rollout-time prnn seam

Pass a single env or a list of envs as the first argument; num_frames is the
TOTAL frames per update across all envs.
"""

import math

import numpy as np
import torch
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from torch_ac.format import default_preprocess_obss

from prnn.utils import PredictiveNet
from curious_george.envs.access import get_subroom_id, grid_shape, subroom_size
from curious_george.rl.collect.collector import (
    CollectorState,
    RolloutConfig,
    collect_rollout,
    get_dist_travelled,
)
from curious_george.rl.collect.diagnostics import LocationStats
from curious_george.rl.update.losses import LOSSES
from curious_george.rl.update.updater import update_policy
from curious_george.rl.update.world_model import train_world_model_on_episodes
from curious_george.world_model.adapter import PRNNAdapter, make_sr_tracker


def compare_trajs(traj1, traj2):
    delta = (traj1 == traj2).cumprod()
    return delta.sum() / len(delta)


class IntrinsicReference:
    """Reference-SR intrinsic reward (historical; B=1 only, off in mainline).

    Preserved quirk: the first error is duplicated in `tail`, so
    int_rewards[0] is always 0.
    """

    def __init__(self, algo: "PredictivePPOAlgo", sr_dim: int):
        self.algo = algo
        self.ref = torch.zeros((1, sr_dim), device=algo.device)
        self.nrefs = 0

    def update_ref(self, activations):
        self.nrefs += 1
        self.ref = self.ref + (activations - self.ref) / self.nrefs

    def update_on_done(self, state: CollectorState, det_np):
        algo = self.algo
        sr = state.sr
        if algo.pastSR:
            preprocessed = algo.preprocess_obss([state.obs_b[0]], device=algo.device)
            with torch.no_grad():
                dist, _ = algo.acmodel(preprocessed, SR=state.sr)
            action = dist.sample()
            sr = algo.adapter.next_sr(action.cpu().numpy(), state.obs_b[0])
        self.update_ref(sr)

    def tail(self, state: CollectorState, SRs: torch.Tensor, last_post_obs) -> torch.Tensor:
        algo = self.algo
        preprocessed = algo.preprocess_obss([state.obs_b[0]], device=algo.device)
        with torch.no_grad():
            dist, _ = algo.acmodel(preprocessed, SR=state.sr)
        action = dist.sample()
        det_np = action.cpu().numpy()
        obs = state.obs_b[0] if algo.pastSR else last_post_obs
        sr = algo.adapter.next_sr(det_np, obs)

        all_SRs = torch.cat((SRs, sr), dim=0).cpu()
        errors = torch.tensor(
            [cosine(s, self.ref.squeeze().cpu()) for s in all_SRs[1:]],
            device=algo.device,
        )
        errors = torch.cat((errors[0][None], errors), dim=0)
        return errors[:-1] - errors[1:]


class PredictivePPOAlgo:
    def __init__(
        self,
        env,
        acmodel,
        predictiveNet: PredictiveNet,
        device: torch.device,
        num_frames=None,
        discount=0.99,
        lr=0.001,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        recurrence=1,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        preprocess_obss=None,
        train_pN=False,
        noise_mu=0,
        noise_std=0.03,
        prnn_seqdur=0,
        intrinsic=False,
        k_int=1,
        pastSR=False,
        curious_agent=False,
        k_curious=1,
        reward_alignment="legacy",
        loss="ppo_clip",
        batched_wm=False,  # appended last: positional callers exist (tests)
        cuda_graph=False,
        batched_curiosity=False,
    ):
        # env may be a single shell, a list of shells (parallel collection),
        # or a batched shell pool (process-parallel or device-resident)
        from curious_george.envs.vector import AsyncShellPool, DeviceTableShellPool

        self.is_async = isinstance(env, AsyncShellPool)
        self.is_device_env = isinstance(env, DeviceTableShellPool)
        self.is_pool = self.is_async or self.is_device_env
        if self.is_pool:
            self.envs = env
            self.env = env.eval_shell  # shell services (encodeAction, grid, ...)
        else:
            self.envs = env if isinstance(env, (list, tuple)) else [env]
            self.env = self.envs[0]
        self.num_envs = len(self.envs)

        self.acmodel = acmodel
        self.pN = predictiveNet
        self.device = device
        self.num_frames = num_frames or 128
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.intrinsic = intrinsic
        self.k_int = k_int
        self.train_pN = train_pN
        self.noise_mu = noise_mu
        self.noise_std = noise_std
        self.prnn_seqdur = prnn_seqdur
        self.batched_wm = batched_wm
        self.cuda_graph = cuda_graph
        self.pastSR = pastSR
        self.curious_agent = curious_agent
        self.k_curious = k_curious
        self.reward_alignment = reward_alignment
        self.loss_name = loss
        assert loss in LOSSES, f"unknown loss {loss!r}; available: {list(LOSSES)}"
        assert pastSR ^ ("Next" in str(self.env.encodeAction))
        assert self.num_frames % self.num_envs == 0, "num_frames must divide by num_envs"
        if self.num_envs > 1:
            assert not intrinsic, "intrinsic rewards not supported with num_envs > 1"
            T = self.num_frames // self.num_envs
            if prnn_seqdur > 0:
                assert T % prnn_seqdur == 0, "per-env T must divide by prnn_seqdur"

        self.adapter = (
            PRNNAdapter(
                self.pN,
                self.device,
                pastSR,
                cuda_graph=cuda_graph,
                batched_curiosity=batched_curiosity,
            )
            if self.pN
            else None
        )
        self._subroom_size = subroom_size(self.env)

        if hasattr(self.env, "loc_mask"):
            self.loc_mask = self.env.loc_mask
        else:
            self.loc_mask = [
                x is None or x.can_overlap() for x in self.env.grid.grid
            ]

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames % self.recurrence == 0

        self.acmodel.to(self.device)
        self.acmodel.train()
        if self.adapter:
            self.adapter.to(self.device)

        # Rollout machinery (env resets + initial SR happen here, in the
        # same order as the historical constructor: reset then init_SR)
        self.tracker = None
        if self.is_pool:
            self._first_obs, first_locs = self.envs.reset_all()
            loc_b = [loc for loc in first_locs]
        else:
            self._first_obs = [e.reset() for e in self.envs]
            loc_b = [self._pos(e) for e in self.envs]
        self.tracker = make_sr_tracker(self.adapter, self.device, self._first_obs)
        self.state = CollectorState(
            obs_b=self._first_obs,
            loc_b=loc_b,
            mask_b=np.ones(self.num_envs, dtype=np.float32),
            sr=self.tracker.initial_sr(),
            init_loc_b=[torch.tensor(loc) for loc in loc_b],
            ep_return=[0.0] * self.num_envs,
            ep_reshaped=[0.0] * self.num_envs,
            ep_frames=[0] * self.num_envs,
        )
        self.loc_stats = LocationStats(self.loc_mask, tuple(grid_shape(self.env)))

        self.intrinsic_ref = (
            IntrinsicReference(self, self.state.sr.shape[-1]) if intrinsic else None
        )

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0

        # analysis code reads these off the algo after each collect
        self.obss: list = []
        self.locs: list = []
        self.subroom_ids: list = []
        self.last_joint_dist = None

    @staticmethod
    def _pos(env):
        return env.get_agent_pos() if hasattr(env, "get_agent_pos") else env.agent_pos

    def agent_pos(self):
        return self._pos(self.env)

    # ------------------------------------------------------------------ #

    def collect_experiences(self, return_joint_distribution=False):
        """Collects rollouts and computes advantages; returns (exps, logs)."""
        result = collect_rollout(
            envs=self.envs,
            acmodel=self.acmodel,
            tracker=self.tracker,
            adapter=self.adapter,
            preprocess_obss=self.preprocess_obss,
            state=self.state,
            cfg=RolloutConfig(
                num_frames=self.num_frames,
                device=self.device,
                prnn_seqdur=self.prnn_seqdur,
                pastSR=self.pastSR,
                curious_agent=self.curious_agent,
                reward_alignment=self.reward_alignment,
                intrinsic=self.intrinsic,
                discount=self.discount,
                gae_lambda=self.gae_lambda,
                k_int=self.k_int,
                k_curious=self.k_curious,
            ),
            loc_stats=self.loc_stats,
            subroom_size_=self._subroom_size,
            intrinsic_ref=self.intrinsic_ref,
        )

        # expose the rollout on the algo for analysis/tasks that read attributes
        self.obss = result.obss
        self.locs = result.locs
        self.subroom_ids = result.subroom_ids
        self.actions = result.actions
        self.values = result.values
        self.rewards = result.rewards
        self.masks = result.masks
        self.SRs = result.SRs
        self.log_probs = result.log_probs
        self.advantages = result.advantages
        self.curious_rewards = result.curious_rewards
        self.int_rewards = result.int_rewards
        self.done_indices = result.done_indices
        self.last_observations = result.last_observations
        self.last_joint_dist = result.joint_dist

        return result.exps, result.logs

    def update_parameters(self, exps, update_params=True):
        # below has to be done so that exps can be batched
        done_indices = exps.done_indices.copy()
        last_observations = exps.last_observations.copy()
        del exps["done_indices"]
        del exps["last_observations"]

        logs, self.batch_num = update_policy(
            self.acmodel,
            self.optimizer,
            exps,
            loss_fn=self.loss_name,
            loss_kwargs=dict(
                clip_eps=self.clip_eps,
                entropy_coef=self.entropy_coef,
                value_loss_coef=self.value_loss_coef,
            ),
            epochs=self.epochs,
            batch_size=self.batch_size,
            recurrence=self.recurrence,
            num_frames=self.num_frames,
            max_grad_norm=self.max_grad_norm,
            batch_num=self.batch_num,
            update_params=update_params,
        )

        if self.train_pN:
            self.adapter.to(self.device)
            train_world_model_on_episodes(
                self.adapter, exps, done_indices, last_observations,
                batched=self.batched_wm,
            )

        return logs.as_dict()

    # ------------------------------------------------------------------ #

    def randomAgent_collect_exp_and_update(self, agent):
        assert self.train_pN, "The only reason to have random actions in algo is to train the pRNN geinus..."
        assert self.num_envs == 1, "random-agent pRNN training is single-env"
        self.adapter.to(self.device)
        numtrials = math.ceil(self.num_frames / self.prnn_seqdur)

        log_curr_seqdurs = []
        subroom_ids = []
        locs = [None] * self.num_frames
        dist_travelled = 0
        for bb in range(numtrials):
            curr_seqdur = min(
                self.prnn_seqdur, self.num_frames - (bb) * self.prnn_seqdur
            )
            log_curr_seqdurs.append(curr_seqdur)
            # Needed if prnn_seqdur is not a perfect divisor of num_frames:
            # the last trial might have < seqdur steps

            obs, act, state, _ = self.pN.collectObservationSequence(
                self.env, agent, curr_seqdur
            )

            # Train
            obs, act = obs.to(self.device), act.to(self.device)
            _, _, _ = self.pN.trainStep(obs, act)
            self.pN.numTrainingEpochs += 1

            # Collect location info
            locs_array = state["agent_pos"][:-1, :] # Shape np.ndarray [seqdur, 2]
            loc_list_current = [tuple(thisloc) for thisloc in locs_array]
            if self._subroom_size is not None:
                subroom_ids = (get_subroom_id(torch.tensor(state["agent_pos"]), self._subroom_size))

            init_pos = state["agent_pos"][0, :]
            final_pos = state["agent_pos"][-1, :]
            dist_travelled = get_dist_travelled(torch.tensor(init_pos).unsqueeze(0), torch.tensor(final_pos).unsqueeze(0)).item()

            startidx = bb * self.prnn_seqdur
            endidx = min(self.num_frames, (bb + 1) * self.prnn_seqdur)
            locs[startidx:endidx] = loc_list_current

        self.locs = locs
        loc_entropy, loc_entropy_5 = self.loc_stats.update(locs)

        policy_entropy = entropy(agent.default_action_probability, base=2)

        return {
            "num_frames": self.num_frames,
            "num_frames_per_episode": log_curr_seqdurs,
            "num_episodes": numtrials,
            "entropy": policy_entropy,
            "loc_entropy": loc_entropy,
            "loc_entropy_5": loc_entropy_5,
            "locs": locs,
            "subroom_ids": subroom_ids,
            "dist_travelled": dist_travelled,
        }
