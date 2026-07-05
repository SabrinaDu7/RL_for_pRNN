"""BatchedSRTracker must exactly reproduce B serial predict_single streams
(with zero noise; with noise the RNG consumption order differs by design).
"""

import numpy as np
import pytest
import torch

from prnn.utils import PredictiveNet, MinigridEnvNames, ActionEncodingsEnum
from curious_george import AgentInputType
from curious_george import make_env
from curious_george.world_model.adapter import BatchedSRTracker

T = 6
B = 3
SEED = 5


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    env = make_env(
        env_key=MinigridEnvNames.LRoom,
        input_type=AgentInputType.H_PO.value,
        act_enc=ActionEncodingsEnum.SpeedHD.value,
        seed=SEED,
    )
    pN = PredictiveNet(
        env,
        hidden_size=64,
        pRNNtype="thRNN_5win",
        trainNoiseMeanStd=(0, 0),  # zero noise -> exact serial/batched equality
        wandb_log=False,
    )
    pN.pRNN.eval()

    # B independent random obs/act streams, formatted per step like the
    # rollout code does (env2pred on [obs, obs] + one action).
    rng = np.random.default_rng(SEED)
    streams = []
    for _ in range(B):
        rows = []
        obs = env.reset()
        for _ in range(T):
            act = np.array([rng.integers(0, 4)])
            obs_pN, act_pN = pN.env_shell.env2pred([obs, obs], act)
            rows.append((obs_pN[:, :-1, :].clone(), act_pN.clone()))
            obs = env.step(act)[0]
        streams.append(rows)
    return pN, streams


def serial_srs(pN, streams, reset_at=None):
    """SRs from stepping each stream with predict_single (the B=1 path)."""
    out = torch.zeros((T, B, pN.hidden_size))
    for b, rows in enumerate(streams):
        pN.reset_state(randInit=False)
        for t, (obs_x, act_x) in enumerate(rows):
            if reset_at is not None and (t, b) in reset_at:
                pN.reset_state(randInit=False)
            with torch.no_grad():
                sr = pN.predict_single(obs_x, act_x).squeeze(0)
            out[t, b] = sr[0, : pN.hidden_size]
    return out


def batched_srs(pN, streams, reset_at=None):
    tracker = BatchedSRTracker(pN, torch.device("cpu"), B)
    out = torch.zeros((T, B, pN.hidden_size))
    for t in range(T):
        if reset_at is not None:
            for tb, b in reset_at:
                if tb == t:
                    tracker.reset_env(b)
        obs_x = torch.cat([streams[b][t][0][:, 0, :] for b in range(B)], dim=0)
        act_x = torch.cat([streams[b][t][1][:, 0, :] for b in range(B)], dim=0)
        out[t] = tracker.step(obs_x, act_x)
    return out


def test_batched_matches_serial(setup):
    pN, streams = setup
    s = serial_srs(pN, streams)
    b = batched_srs(pN, streams)
    assert torch.allclose(s, b, atol=1e-6), (s - b).abs().max()


def test_batched_matches_serial_with_midstream_reset(setup):
    pN, streams = setup
    reset_at = {(2, 1)}  # env 1 resets before step 2 (phase + state to zero)
    s = serial_srs(pN, streams, reset_at=reset_at)
    b = batched_srs(pN, streams, reset_at=reset_at)
    assert torch.allclose(s, b, atol=1e-6), (s - b).abs().max()


def test_reset_env_only_affects_that_env(setup):
    pN, streams = setup
    base = batched_srs(pN, streams)
    with_reset = batched_srs(pN, streams, reset_at={(3, 0)})
    # env 0 diverges from step 3; envs 1..B-1 identical throughout
    assert not torch.allclose(base[3:, 0], with_reset[3:, 0])
    assert torch.equal(base[:, 1:], with_reset[:, 1:])
