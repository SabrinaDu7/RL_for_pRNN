"""World-model (pRNN) training schedule: one gradient step per episode segment."""

import numpy as np

from curious_george.world_model.adapter import PRNNAdapter
from curious_george.utils.timing import timer


def train_world_model_on_episodes(
    adapter: PRNNAdapter,
    exps,
    done_indices: list[int],
    last_observations: list,
) -> None:
    """Train the pRNN on each episode segment of the (preprocessed) rollout.

    `exps.obs` must already be the preprocessed DictList (image/direction
    tensors); segments are [done_indices[i-1], done_indices[i]) and never
    span environment boundaries in the flat layout.
    """
    with timer("update/wm_train"):
        for idx in range(1, len(done_indices)):
            start_episode = done_indices[idx - 1]
            end_episode = done_indices[idx]
            last_obs = last_observations[idx - 1]
            adapter.train_on_episode(
                exps.obs.image[start_episode:end_episode],
                exps.obs.direction[start_episode:end_episode],
                exps.action[start_episode:end_episode].cpu().numpy(),
                last_obs,
            )
