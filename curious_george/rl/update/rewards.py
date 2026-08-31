"""Reward terms for the curious agent, and the count-based control.

The total reward fed to GAE is
    r[i] = rewards[i] + k_int * int_rewards[i] + k_curious * curious_rewards[i]
           + k_count * count_rewards[i]
(see curious_george.rl.update.advantage.compute_gae).

Curiosity reward time alignment
-------------------------------
`reward_alignment="legacy"` (default, bitwise-pinned by
tests/golden/golden_v0.pt): the curiosity reward at step i is the pRNN's
error reconstructing obss[i] - the observation the agent saw BEFORE taking
action i, crediting the action with surprise it did not cause.
`"next_obs"` credits action i with the prediction error on the observation it
produced - uniformly for every action, including the last of each episode
(the adapter extends the per-episode predict pass by one zero-action step so
the final observation is a real prediction target; no boundary special case).
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, Int

from curious_george.models.prnn_adapter import PRNNAdapter


REWARD_ALIGNMENTS = {"legacy": 0, "next_obs": 1}  # name -> prediction target offset


def occurrence_index(states: Int[torch.Tensor, "B T"]) -> Int[torch.Tensor, "B T"]:
    """Per row: how many EARLIER entries equal this one (0 for a first visit).

    Stable sort groups equal states into runs; the position inside a run is the
    occurrence index, scattered back through the sort order. No Python loop.
    """
    B, T = states.shape
    sorted_vals, order = states.sort(dim=1, stable=True)
    steps = torch.arange(T, device=states.device).expand(B, T)
    run_start = steps.clone()
    same_as_prev = sorted_vals[:, 1:] == sorted_vals[:, :-1]
    run_start[:, 1:] = torch.where(same_as_prev, torch.zeros_like(steps[:, 1:]), steps[:, 1:])
    run_start = run_start.cummax(dim=1).values
    occ_sorted = steps - run_start
    return torch.zeros_like(states).scatter_(1, order, occ_sorted)


@dataclass
class CountBonus:
    """Lifetime visit counts over (room, x, y, head direction), and the
    1/sqrt novelty reward they define - curiosity's model-free CONTROL.

    The m-th visit a stream makes to state s within a rollout earns
    ``1 / sqrt(N_pre(s) + m)`` where ``N_pre`` is the lifetime count at rollout
    start. The within-rollout term is not a refinement but what makes the
    signal exist: against a fresh table alone every step of every episode pays
    the identical maximum bonus, a constant reward with zero advantage. Counted
    per stream (each sees the global table plus only its OWN earlier visits),
    the reward is order-independent across the batch, deterministic, and -
    computed after the timestep loop from the recorded buffers, exactly like
    the curiosity pass - never inside a CUDA-graph capture.

    Counts are TRAINING STATE: `state_dict`/`load_state_dict` ride the policy
    checkpoint, or a resumed run would restart novelty from scratch.
    """

    counts: Int[torch.Tensor, "R W H 4"]

    @classmethod
    def create(
        cls, *, n_layouts: int, width: int, height: int, device: torch.device
    ) -> "CountBonus":
        return cls(counts=torch.zeros(
            (n_layouts, width, height, 4), dtype=torch.long, device=device
        ))

    def rewards(
        self,
        *,
        layouts_tb: Int[torch.Tensor, "T B"],
        positions_tb: Int[torch.Tensor, "T B 2"],
        directions_tb: Int[torch.Tensor, "T B"],
    ) -> Float[torch.Tensor, "T B"]:
        """The UNSCALED bonus per step (k_count is applied by compute_gae),
        then the table absorbs this rollout's visits."""
        R, W, H, D = self.counts.shape
        states = (
            (layouts_tb.long() * W + positions_tb[..., 0]) * H + positions_tb[..., 1]
        ) * D + directions_tb.long()  # (T, B)
        flat = self.counts.reshape(-1)
        pre = flat[states]
        m = occurrence_index(states.T).T + 1
        bonus = torch.rsqrt((pre + m).float())
        flat.scatter_add_(0, states.reshape(-1), torch.ones_like(states.reshape(-1)))
        return bonus

    def state_dict(self) -> dict:
        return {"counts": self.counts.cpu()}

    def load_state_dict(self, state: dict) -> None:
        self.counts.copy_(state["counts"].to(self.counts.device))


def compute_curious_rewards(
    adapter: PRNNAdapter,
    obss: list,
    actions_np: np.ndarray,
    done_indices: list[int],
    last_observations: list,
    num_frames: int,
    alignment: str = "legacy",
) -> torch.Tensor:
    """Curiosity reward = per-step pRNN observation-prediction MSE.

    alignment="legacy": MSEs[i] is the error on the PRE-action obs.
    alignment="next_obs": MSEs[i] is the error on the obs action i produced
    (see module docstring).
    """
    assert alignment in REWARD_ALIGNMENTS, f"unknown reward_alignment {alignment!r}"
    return adapter.prediction_mses(
        obss=obss,
        actions_np=actions_np,
        done_indices=done_indices,
        last_observations=last_observations,
        num_frames=num_frames,
        target_offset=REWARD_ALIGNMENTS[alignment],
    )
