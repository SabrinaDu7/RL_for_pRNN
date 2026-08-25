"""World-model (pRNN) training schedule: one gradient step per episode segment."""

import numpy as np

from curious_george.models.prnn_adapter import PRNNAdapter
from curious_george.utils.timing import timer


def train_world_model_on_episodes(
    adapter: PRNNAdapter,
    exps,
    done_indices: list[int],
    last_observations: list,
    *,
    batched: bool = False,
    segment_stride: int = 1,
    pool_group: int = 0,
) -> None:
    """Train the pRNN on each episode segment of the (preprocessed) rollout.

    `exps.obs` must already be the preprocessed DictList (image/direction
    tensors); segments are [done_indices[i-1], done_indices[i]) and never
    span environment boundaries in the flat layout.

    batched=True (predNet.batched_wm): ONE pooled gradient step on all
    segments stacked to (B, L) instead of B sequential steps - an
    optimization-semantics change, curve-gate before defaulting on. Falls
    back to serial when segments are ragged or the encoding isn't SpeedHD.

    segment_stride (predNet.wm_segment_stride) trains on every k-th segment
    instead of all of them, which sets WORLD-MODEL GRADIENT STEPS PER UNIT OF
    EXPERIENCE - the quantity that separates the serial and pooled regimes:

        serial, stride 1   1 step per seqdur          env steps  (256)
        serial, stride k   1 step per k*seqdur        env steps
        pooled             1 step per frames          env steps

    It exists because that ratio turned out to matter. A 2026-08-22 cluster run
    (job 10444495, serial, num_envs=128) trained the world model 8x more per
    environment step than the reference series and its per-room sRSA PEAKED at
    0.7732 by 25.2M steps - 98% of the reference plateau in a third of the
    experience - then FELL to 0.5581 by 83.9M while prediction loss kept
    improving (0.0055 -> 0.0035). Loss down, place code down, is over-training
    the predictor. `stride = frames // seqdur` reproduces the reference's ratio
    while keeping the serial code path, which is the one predNet.cuda_graph
    accelerates.

    ⚠️ Not identical to pooling: a pooled step averages the gradient over all
    segments, a strided serial step uses one. Same step COUNT per experience,
    higher variance. Gate it.
    """
    with timer("update/wm_train"):
        seg_lengths = {
            done_indices[i] - done_indices[i - 1] for i in range(1, len(done_indices))
        }
        if (
            batched
            and adapter.fast_speedhd
            and len(done_indices) > 2
            and len(seg_lengths) == 1
        ):
            adapter.train_on_episodes_batched(
                exps, done_indices, last_observations, group=pool_group
            )
            return
        stride = max(1, int(segment_stride))
        for idx in range(1, len(done_indices), stride):
            start_episode = done_indices[idx - 1]
            end_episode = done_indices[idx]
            last_obs = last_observations[idx - 1]
            adapter.train_on_episode(
                exps.obs.image[start_episode:end_episode],
                exps.obs.direction[start_episode:end_episode],
                exps.action[start_episode:end_episode].cpu().numpy(),
                last_obs,
            )
