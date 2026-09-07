# Based on the PPO algo from torch-ac library (https://github.com/lcswillems/torch-ac)

"""One algo class for B >= 1 environments.

`PredictivePPOAlgo` wires together:
- rl.collect.collector  - the rollout loop (B=1 is bitwise-identical to the
  historical serial path via SingleSRTracker; B>1 batches the forwards)
- rl.update.policy / losses - the PPO-clip update
- rl.update.rewards / advantage - reward terms and GAE
- rl.update.world_model - per-episode pRNN training
- models.prnn_adapter   - the rollout-time prnn seam

Pass a single env or a list of envs as the first argument; num_frames is the
TOTAL frames per update across all envs.
"""

import math

import numpy as np
import torch
from scipy.stats import entropy
from torch_ac.format import default_preprocess_obss

from prnn.utils import PredictiveNet
from curious_george.envs.access import grid_shape
from curious_george.rl.collect.collector import (
    CollectorState,
    RolloutConfig,
    collect_rollout,
)
from curious_george.rl.collect.diagnostics import LocationStats
from curious_george.rl.update.losses import ppo_clip_loss
from curious_george.rl.update.policy import update_policy
from curious_george.rl.update.world_model import train_world_model_on_episodes
from curious_george.models.prnn_adapter import PRNNAdapter, make_sr_tracker
from curious_george.utils.timing import timer


class PredictivePPOAlgo:
    def __init__(
        self,
        env,
        acmodel,
        predictiveNet: PredictiveNet,
        device: torch.device,
        *,
        num_frames=None,
        discount=0.99,
        lr=0.001,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        preprocess_obss=None,
        train_pN=False,
        prnn_seqdur=0,
        # NOT 0: preserved verbatim from the `pastSR=False` this replaced.
        # Only direct constructions see it; setup_algo always passes one.
        action_offset: int = 1,
        random_actions: bool = False,
        random_action_probs: tuple[float, ...] | None = None,
        normalize_advantage: bool = False,
        curious_agent=False,
        k_curious=1,
        k_count=0.0,
        normalize_reward=False,
        reward_alignment="legacy",
        batched_wm=False,
        cuda_graph=False,
        batched_curiosity=False,
        compile_cell=False,
        wm_segment_stride=1,
        wm_pool_group=0,
        adam_betas=(0.9, 0.999),
        policy_cuda_graph=False,
        rollout_cuda_graph=False,
        curiosity_cuda_graph=False,
    ):
        # env may be a single shell, a list of shells (serial collection), or
        # the device-resident shell pool
        from curious_george.envs.vector import DeviceTableShellPool

        self.is_device_env = isinstance(env, DeviceTableShellPool)
        if self.is_device_env:
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
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.train_pN = train_pN
        self.prnn_seqdur = prnn_seqdur
        self.batched_wm = batched_wm
        self.wm_segment_stride = wm_segment_stride
        self.wm_pool_group = wm_pool_group
        # train_policy.cuda_graph: replay a captured PPO minibatch step. Built
        # lazily on first update so the acmodel is already on-device and the
        # optimizer state is still empty (the capturable rebuild requires that).
        self.policy_cuda_graph = bool(policy_cuda_graph) and device.type == "cuda"
        self._policy_graph = None
        # collect.rollout_cuda_graph: replay a captured rollout timestep. Built
        # after the tracker exists, below; capture is deferred to the first
        # rollout so the models are on-device and warmed up.
        self.rollout_cuda_graph = bool(rollout_cuda_graph) and device.type == "cuda"
        self._rollout_graph = None
        self.action_offset = action_offset
        self.random_actions = random_actions
        # None = the project default (configs.RAND_ACT_PROBA); resolved by
        # `random_action_probs` at the two sampling sites.
        self.random_action_probs = (
            tuple(random_action_probs) if random_action_probs is not None else None
        )
        self.normalize_advantage = normalize_advantage
        self.curious_agent = curious_agent
        self.k_curious = k_curious
        self.reward_alignment = reward_alignment
        assert self.num_frames % self.num_envs == 0, "num_frames must divide by num_envs"
        if self.num_envs > 1:
            T = self.num_frames // self.num_envs
            if prnn_seqdur > 0:
                assert T % prnn_seqdur == 0, "per-env T must divide by prnn_seqdur"

        self.adapter = (
            PRNNAdapter(
                self.pN,
                self.device,
                action_offset,
                cuda_graph=cuda_graph,
                batched_curiosity=batched_curiosity,
                compile_cell=compile_cell,
                curiosity_cuda_graph=curiosity_cuda_graph,
            )
            if self.pN
            else None
        )

        self.loc_mask = self._location_mask()

        self.acmodel.to(self.device)
        self.acmodel.train()
        if self.adapter:
            self.adapter.to(self.device)

        # Rollout machinery (env resets + initial SR happen here, in the
        # same order as the historical constructor: reset then init_SR)
        self.tracker = None
        if self.is_device_env:
            self._first_obs, first_locs = self.envs.reset_all()
            loc_b = [loc for loc in first_locs]
        else:
            self._first_obs = [e.reset() for e in self.envs]
            loc_b = [self._pos(e) for e in self.envs]
        self.tracker = make_sr_tracker(self.adapter, self.device, self._first_obs)
        if self.is_device_env and self.adapter is not None and action_offset:
            # The device pool's `reset_all` returns None placeholders, so the
            # shim could not build h[0] at construction. Do it here, from the
            # tensors, before the first SR is read.
            images, directions = self.envs.observation_device()
            self.tracker.reset_all_envs(images=images, directions=directions)
        if self.rollout_cuda_graph:
            from curious_george.rl.collect.rollout_graph import GraphRolloutStepper

            if not self.is_device_env:
                raise ValueError(
                    "collect.rollout_cuda_graph captures the device environment "
                    "table's timestep; it requires collect.backend=DEVICE"
                )
            self._rollout_graph = GraphRolloutStepper(
                acmodel=self.acmodel,
                tracker=self.tracker,
                pool=self.envs,
                num_steps=self.num_frames // self.num_envs,
                device=self.device,
                random_actions=self.random_actions,
                random_action_probs=self.random_action_probs,
            )
        self.state = CollectorState(
            obs_b=self._first_obs,
            loc_b=loc_b,
            mask_b=np.ones(self.num_envs, dtype=np.float32),
            sr=self.tracker.initial_sr(),
            ep_return=[0.0] * self.num_envs,
            ep_reshaped=[0.0] * self.num_envs,
            ep_frames=[0] * self.num_envs,
        )
        self.loc_stats = LocationStats(self.loc_mask, tuple(grid_shape(self.env)))
        self.room_supports, self.room_walkable_counts = self._room_geometry()

        self.k_count = float(k_count)
        self.count_bonus = None
        if self.k_count > 0:
            if not self.is_device_env:
                raise ValueError(
                    "train_policy.k_count reads the device positions/directions "
                    "buffers; it requires the DEVICE backend"
                )
            from curious_george.rl.update.rewards import CountBonus

            W, H = grid_shape(self.env)
            self.count_bonus = CountBonus.create(
                n_layouts=self.envs.n_layouts, width=W, height=H, device=self.device
            )

        from curious_george.rl.update.advantage import RewardNormalizer

        self.reward_normalizer = RewardNormalizer() if normalize_reward else None

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        if len(adam_betas) != 2 or not (
            0 <= adam_betas[0] < 1 and 0 <= adam_betas[1] < 1
        ):
            raise ValueError(f"Adam betas must be two values in [0, 1), got {adam_betas}")
        self.optimizer = torch.optim.Adam(
            self.acmodel.parameters(),
            lr,
            betas=tuple(adam_betas),
            eps=adam_eps,
        )

        # analysis code reads these off the algo after each collect
        self.directions = np.empty(0, dtype=np.int64)
        self.locs: list = []
        self.last_joint_dist = None
        self.segment_layouts = None
        self.positions_episodes = None

    def _location_mask(self) -> list:
        """The cells `loc_entropy` is normalised over: everywhere the agent
        could be in ANY room this run trains on.

        Entropy over visited cells only means something against the set of cells
        that exist. That used to be one set, read off the eval shell, because
        landmarks were walkable and every room shared it. Impassable landmarks
        break that per room, and the visits reaching `LocationStats` are pooled
        across streams that are in DIFFERENT rooms - so no single room's mask is
        right for them: it counts cells that were blocked where the agent
        actually was, and drops cells that were open there.

        The union is the one support that is correct for a pooled measure, and
        what makes it matter is NOT the half you would expect. Measured:

          including cells the agent can never reach   changes entropy by 9e-16
          dropping cells the agent DID visit          changes it by 0.19 bits

        Zero-count cells are inert - 0*log0 = 0 - so keeping a blocked cell in
        the support costs nothing. The bug is the other direction: the old mask
        was ONE room's, so in any other room the agent walks on cells that mask
        excludes, and those visits were deleted from the histogram before the
        entropy was taken. The union exists to stop dropping real visits.

        Identical to the old mask whenever the rooms share a walkable set, which
        is every run before 2026-08-27 - so the existing series is unbroken.

        ⚠️ TWO things this still does not fix, both for the reader rather than
        the code. It is a MIXTURE across rooms: `algo.locs` has no room channel,
        so a low count cannot be told apart from "blocked in most rooms". And
        the CEILING moves with the geometry - a perfect explorer scores
        log2(153)=7.26 with objects against log2(172)=7.43 without - so raw
        `loc_entropy` is not comparable between an objects arm and a control.
        That comparison needs normalising by log2(reachable), per room.
        """
        env = self.env
        layouts = getattr(self.envs, "layouts", None)
        if not layouts:
            if hasattr(env, "loc_mask"):
                return env.loc_mask
            from curious_george.envs.access import base_env

            return [x is None or x.can_overlap() for x in base_env(env).grid.grid]

        # The mask is flat with cell (x, y) at index y*width + x, which is the
        # order `LocationStats` masks its Fortran-flattened visit grid in.
        # The union rule lives in layouts.py, so this and EnvCfg.reachable_cells
        # cannot drift into two different answers.
        from curious_george.envs.layouts import pooled_walkable

        reachable = pooled_walkable(self._cells_without_walls(env), layouts)
        return [
            (x, y) in reachable
            for y in range(env.height)
            for x in range(env.width)
        ]

    @staticmethod
    def _cells_without_walls(env) -> frozenset[tuple[int, int]]:
        """What the walls leave open - the `base` every Layout.walkable takes."""
        from curious_george.envs.access import base_env

        grid = base_env(env).grid
        return frozenset(
            (x, y)
            for y in range(env.height)
            for x in range(env.width)
            if getattr(grid.get(x, y), "type", None) != "wall"
        )

    def _room_geometry(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-room walkable supports `(R, W, H)` bool and their cell counts.

        Room r's support is `layouts[r].walkable(base)` - the denominator the
        per-episode exploration metrics normalize by. The POOLED support
        (`loc_mask`) cannot replace these: on the committed room set its
        ceiling is identical across the affordance arms (see
        docs/exploration-evals-2026-08-30.md). With no room set there is one
        room and its support IS the pooled mask.
        """
        W, H = grid_shape(self.env)
        layouts = getattr(self.envs, "layouts", None)
        if layouts:
            base = self._cells_without_walls(self.env)
            supports = torch.zeros((len(layouts), W, H), dtype=torch.bool)
            for r, layout in enumerate(layouts):
                for x, y in layout.walkable(base):
                    supports[r, x, y] = True
        else:
            # loc_mask is Fortran-flat, index y*W + x -> [y, x] -> transpose.
            supports = torch.as_tensor(
                np.asarray(self.loc_mask, dtype=bool).reshape(H, W).T.copy()
            ).unsqueeze(0)
        return supports.to(self.device), supports.long().sum(dim=(1, 2)).to(self.device)

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
                action_offset=self.action_offset,
                random_actions=self.random_actions,
                random_action_probs=self.random_action_probs,
                curious_agent=self.curious_agent,
                reward_alignment=self.reward_alignment,
                discount=self.discount,
                gae_lambda=self.gae_lambda,
                k_curious=self.k_curious,
                k_count=self.k_count,
            ),
            loc_stats=self.loc_stats,
            rollout_graph=self._rollout_graph,
            count_bonus=self.count_bonus,
            reward_normalizer=self.reward_normalizer,
        )

        # expose the rollout on the algo for analysis/tasks that read attributes
        self.directions = result.directions
        self.locs = result.locs
        self.actions = result.actions
        self.values = result.values
        self.rewards = result.rewards
        self.masks = result.masks
        self.SRs = result.SRs
        self.log_probs = result.log_probs
        self.advantages = result.advantages
        self.curious_rewards = result.curious_rewards
        self.done_indices = result.done_indices
        self.last_observations = result.last_observations
        self.last_joint_dist = result.joint_dist
        self.segment_layouts = result.segment_layouts
        self.positions_episodes = result.positions_episodes

        # Exploration metrics from the rollout's episode view (device backend
        # only). Pure reads of already-collected tensors - no RNG, nothing
        # mutated - so the golden gate's pinned stream is untouched.
        if result.positions_episodes is not None:
            from curious_george.evaluation.exploration import (  # local import: avoids cycle
                rollout_summary,
            )

            with timer("collect/exploration"):
                layout_ids = torch.as_tensor(
                    result.segment_layouts.reshape(-1),
                    device=result.positions_episodes.device,
                )
                result.logs.update(rollout_summary(
                    positions=result.positions_episodes,
                    layout_ids=layout_ids,
                    supports=self.room_supports,
                    denominators=self.room_walkable_counts,
                ))
        support_cells = int(np.sum(self.loc_mask))
        if support_cells > 1:
            result.logs["loc_entropy_norm"] = (
                result.logs["loc_entropy"] / math.log2(support_cells)
            )

        return result.exps, result.logs

    def update_parameters(self, exps, update_params=True, update_world_model=None):
        """`update_params` gates the POLICY; `update_world_model` gates the pRNN.

        They used to be one flag, and that conflation is a defect: a random-agent
        BASELINE wants the world model trained on random-walk data while the
        policy takes no steps, and there was no way to say so. Passing
        update_params=False silently trained nothing, and the run scored sRSA
        0.06 - the UNTRAINED score - which looks like a finding rather than a
        bug.

        `None` follows `update_params`, preserving every existing caller
        including `arch_policy.freeze_params`.
        """
        # below has to be done so that exps can be batched
        done_indices = exps.done_indices.copy()
        last_observations = exps.last_observations.copy()
        del exps["done_indices"]
        del exps["last_observations"]

        if self.policy_cuda_graph and self._policy_graph is None and update_params:
            from curious_george.rl.update.policy_graph import GraphPolicyTrainer

            self._policy_graph = GraphPolicyTrainer(
                self.acmodel,
                self.optimizer,
                loss_fn=ppo_clip_loss,
                loss_kwargs=dict(
                    clip_eps=self.clip_eps,
                    entropy_coef=self.entropy_coef,
                    value_loss_coef=self.value_loss_coef,
                    normalize_advantage=self.normalize_advantage,
                ),
                max_grad_norm=self.max_grad_norm,
            )
            # the trainer may have rebuilt the optimizer capturable
            self.optimizer = self._policy_graph.optimizer

        logs = update_policy(
            self.acmodel,
            self.optimizer,
            exps,
            loss_fn=ppo_clip_loss,
            loss_kwargs=dict(
                clip_eps=self.clip_eps,
                entropy_coef=self.entropy_coef,
                value_loss_coef=self.value_loss_coef,
                normalize_advantage=self.normalize_advantage,
            ),
            epochs=self.epochs,
            batch_size=self.batch_size,
            num_frames=self.num_frames,
            max_grad_norm=self.max_grad_norm,
            update_params=update_params,
            graph_trainer=self._policy_graph,
        )

        if update_world_model is None:
            update_world_model = update_params
        if self.train_pN and update_world_model:
            self.adapter.to(self.device)
            train_world_model_on_episodes(
                self.adapter, exps, done_indices, last_observations,
                batched=self.batched_wm,
                segment_stride=self.wm_segment_stride,
                pool_group=self.wm_pool_group,
            )

        return logs.as_dict()

    # ------------------------------------------------------------------ #

    def randomAgent_collect_exp_and_update(self, agent):
        assert self.train_pN, "The only reason to have random actions in algo is to train the pRNN geinus..."
        assert self.num_envs == 1, "random-agent pRNN training is single-env"
        self.adapter.to(self.device)
        numtrials = math.ceil(self.num_frames / self.prnn_seqdur)

        log_curr_seqdurs = []
        locs = [None] * self.num_frames
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
            self.pN.trainStep(obs, act, return_stats=False)
            self.pN.numTrainingEpochs += 1

            # Collect location info
            locs_array = state["agent_pos"][:-1, :] # Shape np.ndarray [seqdur, 2]
            loc_list_current = [tuple(thisloc) for thisloc in locs_array]

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
        }
