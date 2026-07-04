# Based on the PPO algo from torch-ac library (https://github.com/lcswillems/torch-ac)


import torch
import math
import numpy as np
from jaxtyping import Float

from scipy.stats import entropy
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList
from scipy.spatial.distance import cosine

from prnn.utils import PredictiveNet
from curious_george.envs.access import get_subroom_id, subroom_size, grid_shape
from curious_george.common import mean_by_action
from curious_george.world_model.adapter import PRNNAdapter
from curious_george.rl.buffer import compute_gae
from curious_george.rl.rewards import compute_curious_rewards
from curious_george.rl.ppo import ppo_update

def check_large_jump(loc0: tuple, loc1: tuple):
    x0, y0 = loc0
    x1, y1 = loc1
    if (x1 - x0)**2 > 1 or (y1 - y0)**2 > 1:
        return True
    else:
        return False

def compare_trajs(traj1, traj2):
    delta = (traj1 == traj2).cumprod()
    return delta.sum() / len(delta)

def get_dist_travelled(
    start_locs: Float[torch.Tensor, "B 2"],
    end_locs: Float[torch.Tensor, "B 2"]
) -> Float[torch.Tensor, "B"]:
    """
    Calculate L1 distance between start and end locations.

    Since the agent can only move horizontally and vertically in the grid,
    the distance is the sum of absolute differences in x and y coordinates.
    """
    dists = torch.abs(end_locs - start_locs).sum(dim=1)
    return dists


class PredictivePPOAlgo:
    """PPO with pRNN-derived spatial representations and curiosity reward.

    Facade over: PRNNAdapter (all pRNN calls), rl.rewards (curiosity),
    rl.buffer.compute_gae (advantages), rl.ppo.ppo_update (parameter updates).
    Public attributes (obss, locs, actions, values, advantages, masks, ...)
    are preserved for the analysis/task code that reads them.
    """

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
    ):
        # Store parameters
        print("Store parameters")
        self.env = env
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
        self.pastSR = pastSR
        self.curious_agent = curious_agent
        self.k_curious = k_curious
        self.reward_alignment = reward_alignment
        assert pastSR ^ ("Next" in str(env.encodeAction))

        self.adapter = PRNNAdapter(self.pN, self.device, pastSR) if self.pN else None
        self._subroom_size = subroom_size(self.env)

        if hasattr(self.env, "loc_mask"):
            self.loc_mask = self.env.loc_mask
        else:
            self.loc_mask = [x == None or x.can_overlap() for x in env.grid.grid]

        # Control parameters
        print("Control parameters")
        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames % self.recurrence == 0

        # Configure models
        print("Configure acmodel")
        self.acmodel.to(self.device)
        self.acmodel.train()
        if self.adapter:
            # TODO: should be elsewhere if saving the net
            self.adapter.to(self.device)

        self.obs = self.env.reset()
        self.loc = self.agent_pos()
        self.mask = 1
        print("Reset done")

        # Initialize spatial representations (if used)
        self.init_SR()

        # Initialize experiences
        self.init_exp()

        # Initialize log values
        self.init_log()

        # Initialize intrinsic rewards
        if self.intrinsic:
            self.ref = torch.zeros((1, self.SR.shape[-1]), device=self.device)
            self.nrefs = 0
            self.int_rewards = torch.zeros(self.num_frames, device=self.device)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0
        print("All done")

    def collect_experiences(self, return_joint_distribution=False):
        """Collects rollouts and computes advantages.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames, ...).
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        # Joint prob between states and actions. Used in on-policy analysis
        # Count the number of times an specific is taken in that (x, y, HD)
        # In my case, 3D instead of 4D
        joint_probabilities = np.zeros(
            (
                getattr(self.env, "numHDs"),
                self.env.width,
                self.env.height,
                getattr(self.acmodel, "act_dim"),
            ),
            dtype=np.float32,
        )

        # The lists below are only relevant if pRNN is being trained
        logs = {}
        self.done_indices = [0]
        self.last_observations = []
        self.locs = []
        self.subroom_ids = []
        self.obss = []
        obs = None

        dist_travelled = 0
        init_loc = torch.tensor(self.loc)
        for i in range(self.num_frames):
            # Do one agent-environment interaction

            action, dist, value, memory, det_action = self.next_experience()

            if self.prnn_seqdur > 0 and i % self.prnn_seqdur == 0: # First loc of traj
                init_loc = torch.tensor(self.agent_pos())

            # CAREFUL: obs = observation after taking action whereas self.obs is before taking action
            obs, reward, terminated, truncated, _ = self.env.step(det_action)
            loc = self.agent_pos()

            done = terminated or truncated
            if self.prnn_seqdur > 0 and (i + 1) % self.prnn_seqdur == 0:
                done = True

            # DEBUG
            if check_large_jump(self.loc, loc) and i % self.prnn_seqdur != 0:
                print("====== DEBUG START ======")
                print(f"Large jump detected at step {i}: from {self.loc} to {loc}")
                torch.save(self.locs, f"debug_locs{i}.pt")
                print("====== DEBUG END ======")

            # Update spatial representation
            if self.pastSR:
                SR = self.next_SR(det_action, self.obs)
            else:
                SR = self.next_SR(det_action, obs)

            # Update experiences values

            self.obss.append(self.obs)
            self.obs = obs
            self.locs.append(self.loc)
            if self._subroom_size is not None:
                self.subroom_ids.append(get_subroom_id(torch.tensor(self.loc).unsqueeze(0), self._subroom_size).item())
            self.loc = loc

            # SR at step i is the one use to get act[i] (from step i-1 for pastSR)
            self.SRs[i] = self.SR
            self.SR = SR
            self.masks[i] = self.mask
            self.mask = 1 - done
            self.actions[i] = action
            self.values[i] = value
            self.rewards[i] = reward
            self.log_probs[i] = dist.log_prob(action)

            # add counts to joint probs
            hd = self.obss[i]["direction"]
            x, y = self.locs[i]
            act_probs = dist.probs.detach().cpu().numpy().squeeze()
            joint_probabilities[hd, x, y, :] += act_probs

            # Update log values

            self.log_episode_return += reward
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += 1

            if done: # This resets the agent's position to start next trajectory/trial
                if self.intrinsic and reward > 1e-5:
                    if self.pastSR:
                        _, _, _, _, det_action = self.next_experience()
                        SR = self.next_SR(det_action, self.obs)
                    self.update_ref(SR)
                self.log_done_counter += 1
                self.log_return.append(self.log_episode_return)
                self.log_reshaped_return.append(self.log_episode_reshaped_return)
                self.log_num_frames.append(self.log_episode_num_frames)
                if self.adapter:
                    self.adapter.reset_state()
                self.init_SR()
                self.last_observations.append(self.obs)

                dist_travelled = get_dist_travelled(init_loc.unsqueeze(0), torch.tensor(self.loc).unsqueeze(0)).item()
                self.obs = self.env.reset() # Now the agent is in completely new position
                self.loc = self.agent_pos()
                self.log_episode_return = 0
                self.log_episode_reshaped_return = 0
                self.log_episode_num_frames = 0
                self.done_indices.append(i + 1)

        # make sure last obs is included in done indices.
        # these is when each trial ends, for prnn training
        if self.done_indices[-1] != self.num_frames: # i + 1:
            self.done_indices.append(self.num_frames) # (i + 1)
            self.last_observations.append(self.obs)

        # Calculate curious rewards
        actions_preformatted = self.actions.cpu().numpy()
        if self.curious_agent:
            self.curious_rewards = compute_curious_rewards(
                self.adapter,
                obss=self.obss,
                actions_np=actions_preformatted,
                done_indices=self.done_indices,
                last_observations=self.last_observations,
                num_frames=self.num_frames,
                alignment=self.reward_alignment,
            )
            # Separate curious rewards by action type
            curious_by_action = mean_by_action(self.curious_rewards.cpu().numpy(), actions_preformatted)
            logs = {f"curious_reward_{k}": v for k, v in curious_by_action.items()}

        # Calculate intrinsic rewards
        if self.intrinsic:
            _, _, _, _, det_action = self.next_experience()
            if self.pastSR:
                SR = self.next_SR(det_action, self.obs)
            else:
                SR = self.next_SR(det_action, obs)
            # Add SR from the last state
            SRs = torch.cat((self.SRs, SR), dim=0).cpu()
            # Calculate cosine similarity between SRs and reference SR
            # NOTE (preserved quirk): the first error is duplicated below, so
            # int_rewards[0] is always 0.
            errors = torch.tensor(
                [cosine(SR, self.ref.squeeze().cpu()) for SR in SRs[1:]],
                device=self.device,
            )
            errors = torch.cat((errors[0][None], errors), dim=0)
            # Calculate intrinsic rewards
            self.int_rewards = errors[:-1] - errors[1:]

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss([self.obs], device=self.device)
        with torch.no_grad():
            _, next_value = self.acmodel(preprocessed_obs, SR=self.SR)

        compute_gae(
            advantages=self.advantages,
            rewards=self.rewards,
            int_rewards=self.int_rewards,
            curious_rewards=self.curious_rewards,
            values=self.values,
            masks=self.masks,
            final_next_value=next_value,
            final_mask=self.mask,
            num_frames=self.num_frames,
            discount=self.discount,
            gae_lambda=self.gae_lambda,
            k_int=self.k_int,
            k_curious=self.k_curious,
        )

        exps = DictList()
        exps.obs = self.obss
        exps.SR = self.SRs
        exps.action = self.actions
        exps.value = self.values
        exps.reward = self.rewards
        exps.advantage = self.advantages
        exps.returnn = (
            exps.value + exps.advantage
        )  # approximates current and discounted future returns
        exps.log_prob = self.log_probs
        exps.done_indices = self.done_indices
        exps.last_observations = self.last_observations

        # Calculate locations entropy
        for loc in self.locs: # HERE: This is the location sequence you want to plot trajectories
            self.loc_visits[loc] += 1
        self.loc_visits = self.loc_visits.flatten("F")[self.loc_mask]
        loc_entropy = entropy(self.loc_visits, base=2)

        self.loc_history.pop(0)
        self.loc_history.append(self.loc_visits)
        loc_entropy_5 = entropy(np.sum(self.loc_history, axis=0), base=2)
        self.loc_visits = np.zeros(grid_shape(self.env))

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Reset pN state
        if self.adapter:
            self.adapter.reset_state()

        # Log some values

        # Compute average advantages by action type
        adv_by_action = mean_by_action(self.advantages.cpu().numpy(), actions_preformatted)

        new_logs = {
            "return_per_episode": self.log_return,
            "reshaped_return_per_episode": self.log_reshaped_return,
            "num_frames_per_episode": self.log_num_frames,
            "num_frames": self.num_frames,
            "num_episodes": self.log_done_counter,
            "intrinsic_rewards": self.int_rewards.tolist(),
            "curious_rewards": self.curious_rewards.tolist(),
            "values": self.values.tolist(),
            "advantages": self.advantages.tolist(),
            "loc_entropy": loc_entropy,
            "loc_entropy_5": loc_entropy_5,
            "joint_dist": joint_probabilities,
            "locs": self.locs,
            "subroom_ids": self.subroom_ids,
            "dist_travelled": dist_travelled,
            **{f"avg_adv_{k}": v for k, v in adv_by_action.items()},
        }
        logs.update(new_logs)

        # Stash for analysis that reuses this rollout (OnPolicyAnalysis)
        self.last_joint_dist = joint_probabilities

        self.log_return = []
        self.log_reshaped_return = []
        self.log_num_frames = []

        return exps, logs # Everything in exps must be exact same length. Get self.locs out through logs

    def update_parameters(self, exps, update_params=True):
        # below has to be done so that exps can be batched
        done_indices = exps.done_indices.copy()
        last_observations = exps.last_observations.copy()
        del exps["done_indices"]
        del exps["last_observations"]

        logs, self.batch_num = ppo_update(
            self.acmodel,
            self.optimizer,
            exps,
            epochs=self.epochs,
            batch_size=self.batch_size,
            recurrence=self.recurrence,
            num_frames=self.num_frames,
            clip_eps=self.clip_eps,
            entropy_coef=self.entropy_coef,
            value_loss_coef=self.value_loss_coef,
            max_grad_norm=self.max_grad_norm,
            batch_num=self.batch_num,
            update_params=update_params,
        )

        # Update pN

        if self.train_pN:
            self.adapter.to(self.device)
            for idx in range(1, len(done_indices)):
                start_episode = done_indices[idx - 1]
                end_episode = done_indices[idx]
                last_obs = last_observations[idx - 1]
                self.adapter.train_on_episode(
                    exps.obs.image[start_episode:end_episode],
                    exps.obs.direction[start_episode:end_episode],
                    exps.action[start_episode:end_episode].cpu().numpy(),
                    last_obs,
                )

        return logs

    def randomAgent_collect_exp_and_update(self, agent):
        assert self.train_pN, "The only reason to have random actions in algo is to train the pRNN geinus..."
        self.adapter.to(self.device)
        numtrials = math.ceil(self.num_frames / self.prnn_seqdur)

        log_curr_seqdurs = []
        subroom_ids = []
        for bb in range(numtrials):
            curr_seqdur = min(
                self.prnn_seqdur, self.num_frames - (bb) * self.prnn_seqdur
            )
            log_curr_seqdurs.append(curr_seqdur)
            # The above is needed if self.prnn_seqdur is not a perfect divisor of num trials.
            # It implies that the last trial might have <seqdur steps

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
            self.locs[startidx:endidx] = loc_list_current

        for loc in self.locs:
            self.loc_visits[loc] += 1
        self.loc_visits = self.loc_visits.flatten("F")[self.loc_mask]
        loc_entropy = entropy(self.loc_visits, base=2)

        self.loc_history.pop(0)
        self.loc_history.append(self.loc_visits)
        loc_entropy_5 = entropy(np.sum(self.loc_history, axis=0), base=2)
        self.loc_visits = np.zeros([self.env.width, self.env.height])

        policy_entropy = entropy(agent.default_action_probability, base=2)

        return {
            "num_frames": self.num_frames,
            "num_frames_per_episode": log_curr_seqdurs,
            "num_episodes": numtrials,
            "entropy": policy_entropy,
            "loc_entropy": loc_entropy,
            "loc_entropy_5": loc_entropy_5,
            "locs": self.locs,
            "subroom_ids": subroom_ids,
            "dist_travelled": dist_travelled,
        }

    def next_experience(self):
        preprocessed_obs = self.preprocess_obss([self.obs], device=self.device)
        memory = None

        with torch.no_grad():
            dist, value = self.acmodel(preprocessed_obs, SR=self.SR)
        action = dist.sample()  # choose action based on SR from step t-1
        det_action = action.cpu().numpy()

        return action, dist, value, memory, det_action

    def next_SR(self, act, obs):
        if self.adapter:
            return self.adapter.next_sr(act, obs)
        return torch.tensor([], device=self.device).unsqueeze(dim=0)

    def init_SR(self):
        if self.adapter:
            self.SR = self.adapter.init_sr(self.obs)
        else:
            self.SR = torch.tensor([], device=self.device).unsqueeze(dim=0)

    def init_exp(self):
        self.obss = [None] * (self.num_frames)
        self.locs = [None] * (self.num_frames)
        self.mask = 1
        self.masks = torch.zeros(self.num_frames, device=self.device)
        print("Masks done")
        self.actions = torch.zeros(self.num_frames, device=self.device, dtype=torch.int)
        self.values = torch.zeros(self.num_frames, device=self.device)
        self.SRs = torch.zeros((self.num_frames, self.SR.shape[1]), device=self.device)
        print("Values done")
        self.rewards = torch.zeros(self.num_frames, device=self.device)
        self.advantages = torch.zeros(self.num_frames, device=self.device)
        print("Advantages done")
        self.log_probs = torch.zeros(self.num_frames, device=self.device)
        self.int_rewards = torch.zeros(self.num_frames, device=self.device)
        self.curious_rewards = torch.zeros(self.num_frames, device=self.device)
        self.loc_visits = np.zeros([self.env.width, self.env.height])
        self.loc_history = [np.zeros(np.sum(self.loc_mask))] * 5

    def init_log(self):
        print("Initialize log values")
        self.log_episode_return = 0
        self.log_episode_reshaped_return = 0
        self.log_episode_num_frames = 0

        self.log_done_counter = 0
        self.log_return = []
        self.log_reshaped_return = []
        self.log_num_frames = []

    def update_ref(self, activations):
        self.nrefs += 1
        self.ref = self.ref + (activations - self.ref) / self.nrefs

    def agent_pos(self):
        if hasattr(self.env, "get_agent_pos"):
            loc = self.env.get_agent_pos()
        else:
            loc = self.env.agent_pos
        return loc
