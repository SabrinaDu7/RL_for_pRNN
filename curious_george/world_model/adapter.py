"""The seam between the RL code and the pRNN world model.

Everything that touches `prnn` at rollout/training time goes through
`PRNNAdapter`. Nothing else in curious_george should call
`pN.env_shell.env2pred`, `pN.predict*`, or `pN.trainStep` directly.

Temporal conventions (confirmed against ../pRNN source):

- All `*_5win` architectures set `predOffset=0`, so `predict()` returns
  `obs_pred[t]` targeting `obs[t]` (the SAME timestep); the "prediction" comes
  from `inMask` zeroing the observation input on 5 of every 6 steps, not from
  a +1 offset. Docstrings in prnn claiming t+1 describe the base-class default
  (`predOffset=1`) which every 5win subclass overrides.
- `pastSR` (== not a `prevAct` architecture): action index t is the action
  taken AFTER observing obs[t], HD comes from the current step (`SpeedHD`),
  and the hidden state aligns to the current/past position. `prevAct`
  architectures shift actions by one and pair with `SpeedNextHD`
  (HD from the next step); their hidden state aligns to the next step.
"""

import numpy as np
import torch

from prnn.utils import PredictiveNet


def infer_past_sr(predictive_net: PredictiveNet) -> bool:
    """pastSR is determined by the architecture family (see module docstring)."""
    return not ("prevAct" in str(predictive_net.pRNN))


def validate_action_encoding(predictive_net: PredictiveNet, env, pastSR: bool) -> None:
    """The env's action encoding must match the architecture's convention:
    pastSR=True pairs with SpeedHD (current HD), pastSR=False with
    SpeedNextHD (next HD). Mismatch silently misaligns SRs by one step.
    """
    assert pastSR ^ ("Next" in str(env.encodeAction)), (
        f"Action encoding {env.encodeAction} inconsistent with "
        f"architecture {type(predictive_net.pRNN).__name__} (pastSR={pastSR})"
    )


class PRNNAdapter:
    def __init__(self, predictive_net: PredictiveNet, device: torch.device, pastSR: bool):
        self.pN = predictive_net
        self.device = device
        self.pastSR = pastSR
        # Theta-cycle nets (thcyc*) roll k+1 windows along dim 0 of predict()'s
        # returns; masked nets (thRNN_5win*) have no .k attribute.
        self.theta = "thcyc" in str(self.pN.pRNN)
        if self.theta:
            self.k = self.pN.pRNN.k + 1

    @property
    def hidden_size(self) -> int:
        return self.pN.hidden_size

    def to(self, device) -> None:
        self.pN.pRNN.to(device)

    def reset_state(self) -> None:
        self.pN.reset_state(device=str(self.device))

    def init_sr(self, obs) -> torch.Tensor:
        """SR before the first action of an episode.

        pastSR nets start from zeros (the SR for step 0 is 'from step -1');
        next-step nets bootstrap from a zero-action prediction on the first obs.
        """
        if self.pastSR:
            return torch.zeros((1, self.pN.hidden_size), device=self.device)

        obs_pN, act_pN = self.pN.env_shell.env2pred([obs, obs], np.array([0]))
        act_pN = torch.zeros_like(act_pN)
        obs_pN, act_pN = obs_pN.to(self.device), act_pN.to(self.device)
        with torch.no_grad():
            SR = self.pN.predict_single(obs_pN[:, :-1, :], act_pN).squeeze(dim=0)
        return SR

    def next_sr(self, act, obs) -> torch.Tensor:
        """SR for step t based on obs and action from step t-1.

        The caller chooses which obs to pass: the pre-action obs (pastSR) or
        the post-action obs (next-step nets).
        """
        if self.theta:
            obs = [obs] * (self.k + 1)
            act = act.repeat(self.k)

            obs_pN, act_pN = self.pN.env_shell.env2pred(obs, act)
            obs_pN, act_pN = obs_pN.to(self.device), act_pN.to(self.device)
            with torch.no_grad():
                SR = self.pN.predict(obs_pN, act_pN)[2][0]
        else:
            obs = [obs, obs]

            obs_pN, act_pN = self.pN.env_shell.env2pred(obs, act)
            obs_pN, act_pN = obs_pN.to(self.device), act_pN.to(self.device)
            with torch.no_grad():
                SR = self.pN.predict_single(obs_pN[:, :-1, :], act_pN).squeeze(dim=0)
        return SR

    def prediction_mses(
        self,
        obss: list,
        actions_np: np.ndarray,
        done_indices: list[int],
        last_observations: list,
        num_frames: int,
    ) -> torch.Tensor:
        """Per-step observation-prediction MSE over the collected rollout,
        computed per episode segment. Used as the curiosity reward.

        ALIGNMENT CONTRACT (legacy, preserved on purpose - see
        docs/refactor_baseline.md flaw #1): with predOffset=0, MSEs[i] is the
        error reconstructing obss[i], the observation BEFORE action i. The
        planned `reward_alignment=next_obs` mode will credit action i with
        MSEs[i+1] instead.
        """
        with torch.no_grad():
            MSEs = torch.zeros(num_frames, device=self.device)

            for idx in range(1, len(done_indices)):
                start_episode, end_episode = done_indices[idx - 1], done_indices[idx]
                last_obs = last_observations[idx - 1]
                acts_now = actions_np[start_episode:end_episode]
                obs_now = obss[start_episode:end_episode] + [last_obs]

                obs_formatted, act_formatted = self.pN.env_shell.env2pred(obs_now, acts_now)
                obs_formatted, act_formatted = obs_formatted.to(self.device), act_formatted.to(self.device)

                obs_pred, obs_next, _ = self.pN.predict(obs_formatted, act_formatted) # obs_next is reformatted version of obs_formatted
                obs_pred, obs_next = obs_pred.squeeze(0), obs_next.squeeze(0)
                MSEs[start_episode:end_episode] = ((obs_pred - obs_next) ** 2).mean(dim=1)

        return MSEs

    def train_on_episode(self, images_tensor, hd_tensor, act_np: np.ndarray, last_obs) -> None:
        """One pRNN gradient step on a single episode segment."""
        obs_for_pN = [
            {"image": images_tensor[i].cpu().numpy(), "direction": hd_tensor[i].item()}
            for i in range(len(images_tensor))
        ]

        obs, act = self.pN.env_shell.env2pred(obs_for_pN + [last_obs], act_np)

        obs = obs.to(self.device)
        act = act.to(self.device)
        _, _, _ = self.pN.trainStep(obs, act)
        self.pN.numTrainingEpochs += 1
