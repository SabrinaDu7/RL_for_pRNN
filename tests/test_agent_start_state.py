"""`start_pos` and `start_dir` place the agent, or refuse.

They were declared on `ActorCriticAgent.getObservations` and never read - the
name appeared exactly once in the file, in the signature - so every caller that
passed them silently got a random start and no error. These pin the fix, and
pin that an unstandable cell RAISES rather than putting the agent in a wall.
"""

import numpy as np
import pytest
import torch

from prnn.utils import ActionEncodingsEnum, MinigridEnvNames, PredictiveNet

from curious_george import AgentInputType, make_env
from curious_george.models.policy import ACModelSR
from curious_george.rl.collect.agent import ActorCriticAgent
from curious_george.rl.collect.format import get_obss_preprocessor

HIDDEN = 32


@pytest.fixture(scope="module")
def agent_and_env():
    env = make_env(
        env_key=MinigridEnvNames.LRoom,
        input_type=AgentInputType.H_PO.value,
        act_enc=ActionEncodingsEnum.SpeedHD.value,
        seed=3,
    )
    pN = PredictiveNet(env, hidden_size=HIDDEN, pRNNtype="thRNN_5win",
                       trainNoiseMeanStd=(0, 0), dropp=0.0, wandb_log=False)
    obs_space, _ = get_obss_preprocessor(env.observation_space)
    ac = ACModelSR(obs_space, env.action_space, HIDDEN, False, True, True)
    return ActorCriticAgent(env.action_space, ac, pN, torch.device("cpu"),
                            argmax=True), env


@pytest.mark.parametrize("start_dir", (0, 1, 2, 3))
def test_start_state_is_where_the_rollout_begins(agent_and_env, start_dir):
    agent, env = agent_and_env
    from curious_george.envs.layouts import BASE_ROOM_ID, base_walkable

    start_pos = sorted(base_walkable(BASE_ROOM_ID))[7]
    torch.manual_seed(0)
    _, _, state, _ = agent.getObservations(
        env, 3, start_pos=start_pos, start_dir=start_dir)
    assert tuple(state["agent_pos"][0]) == start_pos
    assert int(np.ravel(state["agent_dir"])[0]) == start_dir


def test_the_observation_matches_the_requested_start(agent_and_env):
    """Not just the recorded position - obs[0] must be the view from THERE.

    Setting `agent_pos` without regenerating the observation leaves obs[0]
    showing wherever `reset()` landed, which is the silent half of the bug.
    """
    agent, env = agent_and_env
    from curious_george.envs.layouts import BASE_ROOM_ID, base_walkable

    cells = sorted(base_walkable(BASE_ROOM_ID))
    seen = {}
    for pos in (cells[0], cells[-1]):
        obs, _, _, _ = agent.getObservations(env, 1, start_pos=pos, start_dir=0)
        seen[pos] = np.asarray(obs[0]["image"]).copy()
    assert not np.array_equal(*seen.values()), (
        "two different start cells produced the same first observation"
    )


def test_a_wall_is_refused_rather_than_occupied(agent_and_env):
    agent, env = agent_and_env
    with pytest.raises(ValueError, match="standable"):
        agent.getObservations(env, 1, start_pos=(0, 0))
    with pytest.raises(ValueError, match="start_dir"):
        agent.getObservations(env, 1, start_dir=9)
