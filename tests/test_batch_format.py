"""The adapter's vectorized SpeedHD formatting must be bitwise-equal to
prnn's env_shell.env2pred (per-dict Python loops + ActionEncodings.SpeedHD).
"""

import numpy as np
import pytest
import torch

from prnn.utils import PredictiveNet, MinigridEnvNames, ActionEncodingsEnum
from curious_george import AgentInputType
from curious_george import make_env
from curious_george.models.prnn_adapter import (
    PRNNAdapter,
    encode_speed_hd_rows,
    flat_obs_rows,
)

SEED = 7
L = 12


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
    pN = PredictiveNet(env, hidden_size=16, pRNNtype="thRNN_5win",
                       trainNoiseMeanStd=(0, 0), wandb_log=False)
    adapter = PRNNAdapter(pN, torch.device("cpu"), action_offset=0)
    assert adapter.fast_speedhd

    rng = np.random.default_rng(SEED)
    obs_dicts = [env.reset()]
    acts = rng.integers(0, 4, size=L)
    for a in acts:
        obs_dicts.append(env.step(np.array([a]))[0])
    return env, adapter, obs_dicts, acts


def test_seq2pred_matches_env2pred(setup):
    env, adapter, obs_dicts, acts = setup
    obs_ref, act_ref = env.env2pred(obs_dicts, acts)
    obs_new, act_new = adapter.seq2pred(obs_dicts, acts)
    assert torch.equal(obs_ref, obs_new)
    assert torch.equal(act_ref, act_new)
    assert act_ref.dtype == act_new.dtype


def test_seq2pred_no_action_flag(setup):
    """act[0] < 0 zeroes the action block for the whole sequence (OneHot flag)."""
    env, adapter, obs_dicts, acts = setup
    acts_neg = acts.copy()
    acts_neg[0] = -1
    obs_ref, act_ref = env.env2pred(obs_dicts, acts_neg)
    obs_new, act_new = adapter.seq2pred(obs_dicts, acts_neg)
    assert torch.equal(obs_ref, obs_new)
    assert torch.equal(act_ref, act_new)
    assert act_new[:, :, : adapter.num_acts].sum() == 0


def test_rows_match_per_env_env2pred(setup):
    """The batched shim rows equal B per-env env2pred([obs, obs], act) calls."""
    env, adapter, obs_dicts, acts = setup
    B = 5
    obs_src = obs_dicts[:B]
    det_np = np.asarray(acts[:B])

    obs_rows_ref, act_rows_ref = [], []
    for b, obs in enumerate(obs_src):
        o_x, a_x = env.env2pred([obs, obs], det_np[b:b + 1])
        obs_rows_ref.append(o_x[:, 0, :])
        act_rows_ref.append(a_x[:, 0, :])
    obs_ref = torch.cat(obs_rows_ref, dim=0)
    act_ref = torch.cat(act_rows_ref, dim=0)

    obs_new = flat_obs_rows(obs_src)
    act_new = encode_speed_hd_rows(
        det_np, [o["direction"] for o in obs_src], adapter.num_acts, adapter.num_hd
    )
    assert torch.equal(obs_ref, obs_new)
    assert torch.equal(act_ref, act_new)
