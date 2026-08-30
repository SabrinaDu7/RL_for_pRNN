"""predNet.batched_wm: pooled-segment world-model training + dispatch rules."""

import numpy as np
import pytest
import torch
from torch_ac.utils import DictList

from prnn.utils import (
    ActionEncodingsEnum,
    MinigridEnvNames,
    PredictiveNet,
    RandomActionAgent,
)
from curious_george import AgentInputType, make_env
from curious_george.rl.update.world_model import train_world_model_on_episodes
from curious_george.models.prnn_adapter import PRNNAdapter

SEED = 5
L = 24  # segment length


@pytest.fixture()
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

    agent = RandomActionAgent(env.action_space, np.array([0.15, 0.15, 0.6, 0.1]))
    obs_dicts, acts = [env.reset()], []
    for _ in range(2 * L):
        a = int(np.random.randint(0, 4))
        acts.append(a)
        obs_dicts.append(env.step(np.array([a]))[0])

    exps = DictList()
    exps.obs = DictList()
    exps.obs.image = torch.stack(
        [torch.tensor(np.asarray(o["image"]), dtype=torch.float) for o in obs_dicts[:-1]]
    )
    exps.obs.direction = torch.tensor([o["direction"] for o in obs_dicts[:-1]])
    exps.action = torch.tensor(acts)
    # two equal segments; each segment's "last obs" is the obs after its end
    done_indices = [0, L, 2 * L]
    last_observations = [obs_dicts[L], obs_dicts[2 * L]]
    return adapter, exps, done_indices, last_observations


def _wsum(pN) -> float:
    return float(sum(p.detach().double().sum() for p in pN.pRNN.parameters()))


def test_batched_takes_one_pooled_step(setup):
    adapter, exps, dones, lasts = setup
    before = _wsum(adapter.pN)
    n0 = len(adapter.pN.TrainingSaver)
    train_world_model_on_episodes(adapter, exps, dones, lasts, batched=True)
    assert _wsum(adapter.pN) != before, "weights should update"
    assert len(adapter.pN.TrainingSaver) == n0 + 1, "ONE pooled trainStep"
    assert np.isfinite(adapter.pN.TrainingSaver["loss"].iloc[-1])


def test_serial_default_takes_per_segment_steps(setup):
    adapter, exps, dones, lasts = setup
    n0 = len(adapter.pN.TrainingSaver)
    train_world_model_on_episodes(adapter, exps, dones, lasts, batched=False)
    assert len(adapter.pN.TrainingSaver) == n0 + 2, "one trainStep per segment"


def test_ragged_segments_fall_back_to_serial(setup):
    adapter, exps, dones, lasts = setup
    ragged = [0, L - 4, 2 * L]  # unequal lengths
    n0 = len(adapter.pN.TrainingSaver)
    train_world_model_on_episodes(adapter, exps, ragged, lasts, batched=True)
    assert len(adapter.pN.TrainingSaver) == n0 + 2, "fallback: per-segment steps"
