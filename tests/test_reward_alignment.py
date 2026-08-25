"""Unit tests for curiosity-reward time alignment and world-model conventions.

Uses stubs (no real pRNN/env) so the alignment contract is pinned at the
rewards.py level: which buffer index each action's reward comes from.
"""

import numpy as np
import pytest
import torch

from prnn.utils import PredictiveNet, MinigridEnvNames, ActionEncodingsEnum
from curious_george import AgentInputType, make_env
from curious_george.rl.update.rewards import compute_curious_rewards
from curious_george.models.prnn_adapter import (
    PRNNAdapter,
    infer_past_sr,
    validate_action_encoding,
)
from curious_george.models.device import on_device, eval_mode


class StubAdapter:
    """prediction_mses returns [0, 1, ...] (+100 when target_offset=1) so the
    selected alignment is legible from the values."""

    def prediction_mses(self, *, obss, actions_np, done_indices,
                        last_observations, num_frames, target_offset=0):
        return torch.arange(num_frames, dtype=torch.float32) + 100 * target_offset


def _rewards(alignment, done_indices, num_frames=8):
    return compute_curious_rewards(
        StubAdapter(),
        obss=[None] * num_frames,
        actions_np=np.zeros(num_frames),
        done_indices=done_indices,
        last_observations=[None] * (len(done_indices) - 1),
        num_frames=num_frames,
        alignment=alignment,
    )


def test_legacy_selects_offset_0():
    out = _rewards("legacy", done_indices=[0, 4, 8])
    assert torch.equal(out, torch.arange(8, dtype=torch.float32))


def test_next_obs_selects_offset_1():
    out = _rewards("next_obs", done_indices=[0, 8])
    assert torch.equal(out, torch.arange(8, dtype=torch.float32) + 100)


def test_unknown_alignment_raises():
    with pytest.raises(AssertionError):
        _rewards("bogus", done_indices=[0, 8])


# ---------------------------------------------------------------------------
# real-net alignment semantics (zero noise -> deterministic passes)
# ---------------------------------------------------------------------------

L = 10


@pytest.fixture(scope="module")
def episode_stream():
    torch.manual_seed(3)
    np.random.seed(3)
    env = make_env(
        env_key=MinigridEnvNames.LRoom,
        input_type=AgentInputType.H_PO.value,
        act_enc=ActionEncodingsEnum.SpeedHD.value,
        seed=3,
    )
    pN = PredictiveNet(
        env, hidden_size=32, pRNNtype="thRNN_5win",
        trainNoiseMeanStd=(0, 0), wandb_log=False,
    )
    pN.pRNN.eval()
    adapter = PRNNAdapter(pN, torch.device("cpu"), pastSR=True)

    rng = np.random.default_rng(3)
    obs = env.reset()
    obss, acts = [], []
    for _ in range(L):
        a = int(rng.integers(0, 4))
        obss.append(obs)
        acts.append(a)
        obs = env.step(np.array([a]))[0]
    return adapter, obss, np.array(acts), obs  # obs = final (last) observation


def _mses(adapter, obss, acts, done_indices, last_observations, offset):
    torch.manual_seed(11)  # identical draws per pass (zero noise anyway)
    return adapter.prediction_mses(
        obss=obss, actions_np=acts, done_indices=done_indices,
        last_observations=last_observations, num_frames=len(obss),
        target_offset=offset,
    )


def test_next_obs_is_shift_of_legacy_plus_real_final_target(episode_stream):
    adapter, obss, acts, last_obs = episode_stream
    legacy = _mses(adapter, obss, acts, [0, L], [last_obs], 0)
    nxt = _mses(adapter, obss, acts, [0, L], [last_obs], 1)

    # causality: rows 0..L-1 of the extended pass equal the legacy rows,
    # so next_obs[i] == legacy[i+1] for all but the final action
    assert torch.allclose(nxt[:-1], legacy[1:], atol=1e-6)
    # the final action gets a REAL prediction error on last_obs (no duplicate)
    assert torch.isfinite(nxt[-1])
    assert nxt.shape == legacy.shape


def test_next_obs_respects_episode_boundaries(episode_stream):
    adapter, obss, acts, last_obs = episode_stream
    split = 5
    dones = [0, split, L]
    lasts = [obss[split], last_obs]  # first episode's last obs = next pre-action obs
    legacy = _mses(adapter, obss, acts, dones, lasts, 0)
    nxt = _mses(adapter, obss, acts, dones, lasts, 1)

    assert torch.allclose(nxt[0:split - 1], legacy[1:split], atol=1e-6)
    assert torch.allclose(nxt[split:L - 1], legacy[split + 1:L], atol=1e-6)
    assert torch.isfinite(nxt).all()


@pytest.mark.parametrize("offset", (0, 1))
def test_batched_curiosity_matches_serial_without_stochasticity(
    episode_stream, offset
):
    serial, obss, acts, last_obs = episode_stream
    batched = PRNNAdapter(
        serial.pN,
        torch.device("cpu"),
        pastSR=True,
        batched_curiosity=True,
    )
    obss_2 = obss + obss
    acts_2 = np.concatenate([acts, acts])
    dones = [0, L, 2 * L]
    lasts = [last_obs, last_obs]

    expected = _mses(serial, obss_2, acts_2, dones, lasts, offset)
    actual = _mses(batched, obss_2, acts_2, dones, lasts, offset)

    assert torch.allclose(actual, expected, atol=1e-5)


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
        # infer_past_sr keys off pRNNtype since the prnn-new migration
        # (upstream partial(MaskedRNN, ...) factories erased "prevAct" from
        # the class repr).
        self.pRNNtype = arch_name.split("(")[0]


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
