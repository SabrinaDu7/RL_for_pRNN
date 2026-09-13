"""`shuffled_minibatches` must partition the rollout - every transition, once.

This is the gate the old `get_batches_starting_indexes` never had. That
function carried `recurrence` machinery which, at the recurrence=1 the repo
has always run, filtered out the index where `(index + 1) % num_frames == 0`
on alternating epochs and then shifted by `recurrence // 2` = 0. Net effect:
one transition silently dropped per odd epoch, and a short final minibatch,
in service of a shift of zero. Nothing failed when it did.
"""

import numpy as np
import pytest

from curious_george.rl.update import shuffled_minibatches


@pytest.mark.parametrize(
    "num_frames,batch_size", [(64, 16), (32768, 256), (32768, 1024), (100, 32)]
)
def test_partitions_every_transition_exactly_once(num_frames, batch_size):
    for _ in range(4):  # the old bug only appeared on alternating calls
        batches = shuffled_minibatches(num_frames=num_frames, batch_size=batch_size)
        assert np.array_equal(np.sort(np.concatenate(batches)), np.arange(num_frames))


@pytest.mark.parametrize(
    "num_frames,batch_size,sizes",
    [(64, 16, [16, 16, 16, 16]), (100, 32, [32, 32, 32, 4])],
)
def test_minibatch_count_and_sizes(num_frames, batch_size, sizes):
    """One gradient step per minibatch, so the count IS the policy step count."""
    batches = shuffled_minibatches(num_frames=num_frames, batch_size=batch_size)
    assert [len(b) for b in batches] == sizes


def test_order_is_shuffled():
    """Minibatches cut across episodes; consecutive transitions must not group."""
    batches = shuffled_minibatches(num_frames=4096, batch_size=256)
    assert not np.array_equal(np.concatenate(batches), np.arange(4096))


# --- the minibatch's contract with the network ------------------------------


@pytest.mark.parametrize("with_HD", [True, False])
def test_the_SR_actor_consumes_the_minibatch_the_indexer_builds(with_HD):
    """`_index_policy_batch` hands the SR actor an obs with `direction` ONLY
    when `acmodel.with_HD`; otherwise it hands over an empty `DictList`.

    `ACModelSR.forward` built the head-direction one-hot ABOVE its `with_HD`
    test, so it read `obs.direction` whatever the flag said and raised
    `KeyError: 'direction'` on that empty obs. It went unseen because the
    default is `with_HD=True` and no run had ever set it False - and it bit at
    CUDA-graph CAPTURE (`rl/update/policy_graph.py`), so every preset that
    captures the policy step - `parity`, `multienv-fast`, everything derived
    from them - could not run the ablation at all. The base `ACModel.forward`
    always had it right; only the override did not.
    """
    import gymnasium as gym
    import torch
    from torch_ac.utils import DictList

    from curious_george.models.policy import ACModelSR
    from curious_george.rl.update.policy import _index_policy_batch

    B, SR_SIZE = 6, 12
    acmodel = ACModelSR(
        obs_space={"image": (7, 7, 3)}, action_space=gym.spaces.Discrete(4),
        SR_size=SR_SIZE, with_CV=False, with_HD=with_HD,
    )
    exps = DictList({
        "SR": torch.randn(B, SR_SIZE),
        "action": torch.zeros(B, dtype=torch.long), "value": torch.zeros(B),
        "advantage": torch.zeros(B), "returnn": torch.zeros(B),
        "log_prob": torch.zeros(B),
    })
    exps.obs = DictList({"direction": torch.zeros(B, dtype=torch.long)})

    sb = _index_policy_batch(exps, torch.arange(B), acmodel)
    # `in`, not `hasattr`: see DictList.__getattr__ - a miss raises KeyError.
    assert ("direction" in sb.obs) is with_HD, (
        "the indexer's obs and the model's expectation disagree"
    )
    dist, value = acmodel(sb.obs, SR=sb.SR)  # KeyError here IS the failure
    assert dist.probs.shape == (B, 4) and value.shape == (B,)
    assert acmodel.embedding_size == SR_SIZE + (4 if with_HD else 0)
