"""Unit tests for curiosity-reward time alignment and world-model conventions.

The contract (rewards.py): action i is rewarded with the prediction error on
the observation it produced. A stub pins which target offset rewards.py asks
the adapter for; the real-net tests pin what that offset MEANS, against the
adapter's unshifted pass (`target_offset=0`, the oracle nothing else reads).
"""

import numpy as np
import pytest
import torch

from prnn.utils import PredictiveNet, MinigridEnvNames, ActionEncodingsEnum
from curious_george import AgentInputType, make_env
from curious_george.rl.update.rewards import compute_curious_rewards
from curious_george.models.prnn_adapter import PRNNAdapter
from curious_george.models.device import on_device, eval_mode


class StubAdapter:
    """prediction_errors returns [0, 1, ...] + 100 * target_offset, so the
    offset rewards.py asked for is legible from the values."""

    def prediction_errors(self, *, obss, actions_np, done_indices,
                        last_observations, num_frames, target_offset=0):
        return torch.arange(num_frames, dtype=torch.float32) + 100 * target_offset


def test_the_reward_is_the_error_on_the_observation_the_action_produced():
    out = compute_curious_rewards(
        StubAdapter(),
        obss=[None] * 8,
        actions_np=np.zeros(8),
        done_indices=[0, 8],
        last_observations=[None],
        num_frames=8,
    )
    assert torch.equal(out, torch.arange(8, dtype=torch.float32) + 100)


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
    adapter = PRNNAdapter(pN, torch.device("cpu"), action_offset=0)

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
    return adapter.prediction_errors(
        obss=obss, actions_np=acts, done_indices=done_indices,
        last_observations=last_observations, num_frames=len(obss),
        target_offset=offset,
    )


def test_the_reward_is_the_unshifted_pass_shifted_plus_a_real_final_target(episode_stream):
    adapter, obss, acts, last_obs = episode_stream
    unshifted = _mses(adapter, obss, acts, [0, L], [last_obs], 0)
    nxt = _mses(adapter, obss, acts, [0, L], [last_obs], 1)

    # causality: rows 0..L-1 of the extended pass equal the unshifted rows,
    # so reward[i] == unshifted[i+1] for all but the final action
    assert torch.allclose(nxt[:-1], unshifted[1:], atol=1e-6)
    # the final action gets a REAL prediction error on last_obs (no duplicate)
    assert torch.isfinite(nxt[-1])
    assert nxt.shape == unshifted.shape


def test_the_reward_respects_episode_boundaries(episode_stream):
    adapter, obss, acts, last_obs = episode_stream
    split = 5
    dones = [0, split, L]
    lasts = [obss[split], last_obs]  # first episode's last obs = next pre-action obs
    unshifted = _mses(adapter, obss, acts, dones, lasts, 0)
    nxt = _mses(adapter, obss, acts, dones, lasts, 1)

    assert torch.allclose(nxt[0:split - 1], unshifted[1:split], atol=1e-6)
    assert torch.allclose(nxt[split:L - 1], unshifted[split + 1:L], atol=1e-6)
    assert torch.isfinite(nxt).all()


@pytest.mark.parametrize("offset", (0, 1))
def test_batched_curiosity_matches_serial_without_stochasticity(
    episode_stream, offset
):
    serial, obss, acts, last_obs = episode_stream
    batched = PRNNAdapter(
        serial.pN,
        torch.device("cpu"),
        action_offset=0,
        batched_curiosity=True,
    )
    obss_2 = obss + obss
    acts_2 = np.concatenate([acts, acts])
    dones = [0, L, 2 * L]
    lasts = [last_obs, last_obs]

    expected = _mses(serial, obss_2, acts_2, dones, lasts, offset)
    actual = _mses(batched, obss_2, acts_2, dones, lasts, offset)

    assert torch.allclose(actual, expected, atol=1e-5)


# `infer_past_sr` and `validate_action_encoding` USED to be tested here, with
# stub architectures. Both were deleted with `pastSR` itself and their tests
# went with them rather than being rewritten against a subject that no longer
# exists:
#   infer_past_sr          derived the circuit from the ARCHITECTURE'S NAME.
#                          `training/setup.py` reads `cfg.arch_prnn.action_offset`
#                          instead, so nothing inferred it any more.
#   validate_action_encoding  asserted `pastSR ^ ("Next" in encodeAction)`, i.e.
#                          that offset 1 REQUIRES SpeedNextHD. That is false for
#                          the route this repo takes - offset 1 runs on plain
#                          SpeedHD with the rows built in `action_rows` - so the
#                          check would have fired on a correct configuration.
#                          It was already unreachable: nothing called it.

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
