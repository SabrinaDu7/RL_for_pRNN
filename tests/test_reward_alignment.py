"""Unit tests for curiosity-reward time alignment and world-model conventions.

Uses stubs (no real pRNN/env) so the alignment contract is pinned at the
rewards.py level: which buffer index each action's reward comes from.
"""

import numpy as np
import pytest
import torch

from curious_george.rl.rewards import (
    align_to_next_obs,
    compute_curious_rewards,
)
from curious_george.world_model.adapter import infer_past_sr, validate_action_encoding
from curious_george.world_model.device import on_device, eval_mode


class StubAdapter:
    """prediction_mses returns [0, 1, 2, ...] so indices are legible."""

    def __init__(self, num_frames):
        self.num_frames = num_frames

    def prediction_mses(self, *, obss, actions_np, done_indices, last_observations, num_frames):
        return torch.arange(num_frames, dtype=torch.float32)


def _rewards(alignment, done_indices, num_frames=8):
    return compute_curious_rewards(
        StubAdapter(num_frames),
        obss=[None] * num_frames,
        actions_np=np.zeros(num_frames),
        done_indices=done_indices,
        last_observations=[None] * (len(done_indices) - 1),
        num_frames=num_frames,
        alignment=alignment,
    )


def test_legacy_is_passthrough():
    out = _rewards("legacy", done_indices=[0, 4, 8])
    assert torch.equal(out, torch.arange(8, dtype=torch.float32))


def test_next_obs_shifts_within_episode():
    out = _rewards("next_obs", done_indices=[0, 8])
    # action i credited with error on obs[i+1]; last action keeps its own
    expected = torch.tensor([1, 2, 3, 4, 5, 6, 7, 7], dtype=torch.float32)
    assert torch.equal(out, expected)


def test_next_obs_does_not_cross_episode_boundary():
    out = _rewards("next_obs", done_indices=[0, 4, 8])
    # episodes [0..3] and [4..7]; the shift must not leak error 4 into episode 1
    expected = torch.tensor([1, 2, 3, 3, 5, 6, 7, 7], dtype=torch.float32)
    assert torch.equal(out, expected)


def test_next_obs_single_step_episode():
    out = align_to_next_obs(torch.arange(3, dtype=torch.float32), [0, 1, 3])
    expected = torch.tensor([0, 2, 2], dtype=torch.float32)
    assert torch.equal(out, expected)


def test_unknown_alignment_raises():
    with pytest.raises(AssertionError):
        _rewards("bogus", done_indices=[0, 8])


# ---------------------------------------------------------------------------
# pastSR / action-encoding conventions
# ---------------------------------------------------------------------------

class _StubModule:
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name


class _StubPN:
    def __init__(self, arch_name):
        self.pRNN = _StubModule(arch_name)


class _StubEnv:
    def __init__(self, enc_name):
        self.encodeAction = _StubModule(enc_name)


def test_infer_past_sr():
    assert infer_past_sr(_StubPN("thRNN_5win(...)")) is True
    assert infer_past_sr(_StubPN("thRNN_5win_prevAct(...)")) is False


def test_validate_action_encoding():
    validate_action_encoding(_StubPN("thRNN_5win"), _StubEnv("<function SpeedHD>"), pastSR=True)
    validate_action_encoding(_StubPN("thRNN_5win_prevAct"), _StubEnv("<function SpeedNextHD>"), pastSR=False)
    with pytest.raises(AssertionError):
        validate_action_encoding(_StubPN("thRNN_5win"), _StubEnv("<function SpeedNextHD>"), pastSR=True)


# ---------------------------------------------------------------------------
# device / eval-mode context managers
# ---------------------------------------------------------------------------

def test_on_device_restores():
    m = torch.nn.Linear(2, 2)
    original = next(m.parameters()).device
    with on_device(m, "cpu"):
        assert next(m.parameters()).device.type == "cpu"
    assert next(m.parameters()).device == original


def test_eval_mode_restores_training_and_argmax():
    m1, m2 = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    m1.train()
    m2.eval()

    class A:
        argmax = False

    agent = A()
    with eval_mode([m1, m2], agent=agent):
        assert not m1.training and not m2.training
        assert agent.argmax is True
    assert m1.training and not m2.training
    assert agent.argmax is False


def test_eval_mode_restores_on_exception():
    m = torch.nn.Linear(2, 2)
    m.train()
    with pytest.raises(RuntimeError):
        with eval_mode(m):
            raise RuntimeError("boom")
    assert m.training
