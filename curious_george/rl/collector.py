"""Batched (multi-env) experience collection.

`BatchedPredictivePPOAlgo` steps B environments in lockstep: one batched
AC-model forward and one batched pRNN step (BatchedSRTracker) per timestep,
instead of B serial forwards. Environments are plain FaramaMinigridShell
instances stepped in a Python loop (the env step itself is cheap; the win is
the batched network passes).

Used only when num_envs > 1 - the B=1 path is the unchanged
PredictivePPOAlgo.collect_experiences, so the golden fixture and all
old-vs-new comparisons are untouched.

Flat layout: experiences are env-major, index = b*T + t with T = num_frames//B.
Episode segments (for curiosity and pRNN training) never span env boundaries.
GAE runs per env stream on views into the flat tensors.
"""

import numpy as np
import torch

from scipy.stats import entropy
from torch_ac.utils import DictList

from curious_george.common import mean_by_action
from curious_george.envs.access import subroom_size, get_subroom_id
from curious_george.rl.algo import PredictivePPOAlgo, get_dist_travelled
from curious_george.rl.buffer import compute_gae
from curious_george.rl.rewards import compute_curious_rewards


class BatchedPredictivePPOAlgo(PredictivePPOAlgo):
    def __init__(self, envs: list, *args, **kwargs):
        assert len(envs) > 1, "use PredictivePPOAlgo for a single env"
        super().__init__(envs[0], *args, **kwargs)
        assert not self.intrinsic, "intrinsic rewards not supported in batched mode"
        assert self.adapter is not None, "batched mode requires a pRNN"
        assert self.pastSR, "batched mode currently supports pastSR nets only"
        self.envs = envs
        self.B = len(envs)
        assert self.num_frames % self.B == 0, "num_frames must divide by num_envs"
        self.T = self.num_frames // self.B
        if self.prnn_seqdur > 0:
            assert self.T % self.prnn_seqdur == 0, "T must divide by prnn_seqdur"

        self.tracker = self.adapter.make_batched_tracker(self.B)
        self.obs_b = [env.reset() for env in self.envs]
        self.loc_b = [self._pos(env) for env in self.envs]
        self.mask_b = np.ones(self.B, dtype=np.float32)

    @staticmethod
    def _pos(env):
        return env.get_agent_pos() if hasattr(env, "get_agent_pos") else env.agent_pos

    def collect_experiences(self, return_joint_distribution=False):
        B, T = self.B, self.T
        device = self.device

        joint_probabilities = np.zeros(
            (getattr(self.env, "numHDs"), self.env.width, self.env.height,
             getattr(self.acmodel, "act_dim")),
            dtype=np.float32,
        )

        obss = [[None] * T for _ in range(B)]
        locs = [[None] * T for _ in range(B)]
        subrooms = []
        done_indices_b = [[0] for _ in range(B)]
        last_obs_b = [[] for _ in range(B)]

        actions = torch.zeros((T, B), device=device, dtype=torch.int)
        values = torch.zeros((T, B), device=device)
        rewards = torch.zeros((T, B), device=device)
        log_probs = torch.zeros((T, B), device=device)
        masks = torch.zeros((T, B), device=device)
        SRs = torch.zeros((T, B, self.adapter.hidden_size), device=device)

        SR = self.tracker.sr().clone()

        for t in range(T):
            preprocessed = self.preprocess_obss(self.obs_b, device=device)
            with torch.no_grad():
                dist, value = self.acmodel(preprocessed, SR=SR)
            action = dist.sample()                       # (B,)
            act_np = action.cpu().numpy()

            SRs[t] = SR
            masks[t] = torch.as_tensor(self.mask_b, device=device)
            actions[t] = action
            values[t] = value
            log_probs[t] = dist.log_prob(action)

            probs_np = dist.probs.detach().cpu().numpy()
            obs_rows, act_rows = [], []
            for b, env in enumerate(self.envs):
                pre_obs = self.obs_b[b]
                obs_next, reward, terminated, truncated, _ = env.step(np.array([act_np[b]]))
                done = terminated or truncated
                if self.prnn_seqdur > 0 and (t + 1) % self.prnn_seqdur == 0:
                    done = True

                obss[b][t] = pre_obs
                locs[b][t] = self.loc_b[b]
                if self._subroom_size is not None:
                    subrooms.append(get_subroom_id(
                        torch.tensor(self.loc_b[b]).unsqueeze(0), self._subroom_size).item())
                rewards[t, b] = reward

                hd = pre_obs["direction"]
                x, y = locs[b][t]
                joint_probabilities[hd, x, y, :] += probs_np[b]

                # pastSR: SR for step t+1 comes from the PRE-action obs + action
                o_x, a_x = self.pN.env_shell.env2pred([pre_obs, pre_obs], np.array([act_np[b]]))
                obs_rows.append(o_x[:, 0, :])
                act_rows.append(a_x[:, 0, :])

                self.obs_b[b] = obs_next
                self.loc_b[b] = self._pos(env)
                self.mask_b[b] = 1 - done

                if done:
                    done_indices_b[b].append(t + 1)
                    last_obs_b[b].append(self.obs_b[b])
                    self.obs_b[b] = env.reset()
                    self.loc_b[b] = self._pos(env)

            obs_x = torch.cat(obs_rows, dim=0).to(device)
            act_x = torch.cat(act_rows, dim=0).to(device)
            stepped = self.tracker.step(obs_x, act_x)
            # Envs that just finished an episode restart from a zero SR and
            # phase 0 (mirrors init_SR + pN.reset_state on the serial path).
            # Reset strictly AFTER the batched step so the shared phase
            # counter of the other envs is not advanced twice.
            SR = stepped.clone()
            for b in range(B):
                if self.mask_b[b] == 0:
                    self.tracker.reset_env(b)
                    SR[b].zero_()

        for b in range(B):
            if done_indices_b[b][-1] != T:
                done_indices_b[b].append(T)
                last_obs_b[b].append(self.obs_b[b])

        # ---- flatten env-major: index = b*T + t --------------------------
        flat_obss = [obss[b][t] for b in range(B) for t in range(T)]
        flat_locs = [locs[b][t] for b in range(B) for t in range(T)]
        self.obss, self.locs, self.subroom_ids = flat_obss, flat_locs, subrooms

        def flat(x):  # (T, B, ...) -> (B*T, ...)
            return x.permute(1, 0, *range(2, x.dim())).reshape(B * T, *x.shape[2:])

        self.actions = flat(actions)
        self.values = flat(values)
        self.rewards = flat(rewards)
        self.log_probs = flat(log_probs)
        self.masks = flat(masks)
        self.SRs = flat(SRs)
        self.advantages = torch.zeros(B * T, device=device)
        self.curious_rewards = torch.zeros(B * T, device=device)
        self.int_rewards = torch.zeros(B * T, device=device)

        done_indices = []
        last_observations = []
        for b in range(B):
            done_indices.extend(b * T + d for d in done_indices_b[b] if not (b > 0 and d == 0))
            last_observations.extend(last_obs_b[b])
        self.done_indices, self.last_observations = done_indices, last_observations

        actions_np = self.actions.cpu().numpy()
        logs = {}
        if self.curious_agent:
            self.curious_rewards = compute_curious_rewards(
                self.adapter, obss=flat_obss, actions_np=actions_np,
                done_indices=done_indices, last_observations=last_observations,
                num_frames=B * T, alignment=self.reward_alignment,
            )
            curious_by_action = mean_by_action(self.curious_rewards.cpu().numpy(), actions_np)
            logs = {f"curious_reward_{k}": v for k, v in curious_by_action.items()}

        # ---- GAE per env stream ------------------------------------------
        preprocessed = self.preprocess_obss(self.obs_b, device=device)
        with torch.no_grad():
            _, next_values = self.acmodel(preprocessed, SR=self.tracker.sr())
        for b in range(B):
            sl = slice(b * T, (b + 1) * T)
            compute_gae(
                advantages=self.advantages[sl], rewards=self.rewards[sl],
                int_rewards=self.int_rewards[sl], curious_rewards=self.curious_rewards[sl],
                values=self.values[sl], masks=self.masks[sl],
                final_next_value=next_values[b], final_mask=float(self.mask_b[b]),
                num_frames=T, discount=self.discount, gae_lambda=self.gae_lambda,
                k_int=self.k_int, k_curious=self.k_curious,
            )

        exps = DictList()
        exps.obs = flat_obss
        exps.SR = self.SRs
        exps.action = self.actions
        exps.value = self.values
        exps.reward = self.rewards
        exps.advantage = self.advantages
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs
        exps.done_indices = done_indices
        exps.last_observations = last_observations

        for loc in flat_locs:
            self.loc_visits[loc] += 1
        self.loc_visits = self.loc_visits.flatten("F")[self.loc_mask]
        loc_entropy = entropy(self.loc_visits, base=2)
        self.loc_history.pop(0)
        self.loc_history.append(self.loc_visits)
        loc_entropy_5 = entropy(np.sum(self.loc_history, axis=0), base=2)
        self.loc_visits = np.zeros([self.env.width, self.env.height])

        exps.obs = self.preprocess_obss(exps.obs, device=device)
        self.adapter.reset_state()
        self.tracker.reset_all()

        num_episodes = len(done_indices) - 1
        frames_per_episode = [done_indices[i] - done_indices[i - 1] for i in range(1, len(done_indices))]
        returns_per_episode = [
            self.rewards[done_indices[i - 1]:done_indices[i]].sum().item()
            for i in range(1, len(done_indices))
        ]
        adv_by_action = mean_by_action(self.advantages.cpu().numpy(), actions_np)
        logs.update({
            "return_per_episode": returns_per_episode,
            "reshaped_return_per_episode": returns_per_episode,
            "num_frames_per_episode": frames_per_episode,
            "num_frames": B * T,
            "num_episodes": num_episodes,
            "intrinsic_rewards": self.int_rewards.tolist(),
            "curious_rewards": self.curious_rewards.tolist(),
            "values": self.values.tolist(),
            "advantages": self.advantages.tolist(),
            "loc_entropy": loc_entropy,
            "loc_entropy_5": loc_entropy_5,
            "joint_dist": joint_probabilities,
            "locs": flat_locs,
            "subroom_ids": subrooms,
            "dist_travelled": 0,
            **{f"avg_adv_{k}": v for k, v in adv_by_action.items()},
        })
        self.last_joint_dist = joint_probabilities

        return exps, logs
