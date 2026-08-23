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
