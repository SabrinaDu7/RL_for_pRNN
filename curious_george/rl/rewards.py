"""Reward terms for the curious agent.

The total reward fed to GAE is
    r[i] = rewards[i] + k_int * int_rewards[i] + k_curious * curious_rewards[i]
(see curious_george.rl.buffer.compute_gae).

Curiosity reward time alignment
-------------------------------
`reward_alignment="legacy"` (current and only implemented mode): the curiosity
reward at step i is the pRNN's error reconstructing obss[i] - the observation
the agent saw BEFORE taking action i. This credits the action with surprise it
did not cause. The planned `"next_obs"` mode (refactor plan Phase 3) credits
action i with the prediction error on obss[i+1], the observation the action
produced. Kept legacy-only here so the Phase 2 extraction is bitwise
behavior-preserving against tests/golden/golden_v0.pt.
"""

import numpy as np
import torch

from curious_george.world_model.adapter import PRNNAdapter


REWARD_ALIGNMENTS = ("legacy", "next_obs")


def align_to_next_obs(mses: torch.Tensor, done_indices: list[int]) -> torch.Tensor:
    """Shift prediction errors so action i is credited with the error on
    obs[i+1] (the observation the action produced), within each episode.

    The final action of each episode has no in-buffer successor error (the
    prediction targeting the episode's last observation is not produced by
    the per-episode predict pass), so its own error is kept - a one-step
    approximation at episode boundaries, documented here on purpose.
    """
    aligned = mses.clone()
    for idx in range(1, len(done_indices)):
        start, end = done_indices[idx - 1], done_indices[idx]
        if end - start > 1:
            aligned[start:end - 1] = mses[start + 1:end]
        # aligned[end-1] keeps its own (legacy) error
    return aligned


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

    alignment="legacy": MSEs[i] is the error on the PRE-action obs (current
    behavior, see module docstring). alignment="next_obs": errors are shifted
    so action i is credited with the error on the obs it produced.
    """
    assert alignment in REWARD_ALIGNMENTS, f"unknown reward_alignment {alignment!r}"
    mses = adapter.prediction_mses(
        obss=obss,
        actions_np=actions_np,
        done_indices=done_indices,
        last_observations=last_observations,
        num_frames=num_frames,
    )
    if alignment == "next_obs":
        return align_to_next_obs(mses, done_indices)
    return mses
