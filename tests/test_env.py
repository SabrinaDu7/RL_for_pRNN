from minigrid.envs import LEnv, LEnv_goal
from prnn.utils import MinigridEnvNames, ActionEncodingsEnum

from curious_george import AgentInputType
from curious_george import make_env

SIZE = 16
ENV_NAME = MinigridEnvNames.LRoom


def test_import_lenv_goal():
    assert LEnv is not None
    assert hasattr(LEnv, "__init__")

def _get_env():
    env = make_env(env_key=ENV_NAME, agent_start_pos=None, input_type = AgentInputType.H_PO, act_enc=ActionEncodingsEnum.SpeedHD)
    return env

def test_create_lroom_goal_env():
    env = _get_env()

    # Basic checks
    assert env is not None
    assert hasattr(env, "reset")
    assert hasattr(env, "step")
    assert hasattr(env, "observation_space")
    assert hasattr(env, "action_space")

    while hasattr(env, "env"):
        env = env.env
    assert isinstance(env, LEnv)


def test_env_reset():
    """Test that the environment can be reset successfully."""
    env = _get_env()

    # Should return observation and info
    result = env.reset()
    assert isinstance(result, dict)
    assert len(result) == 3

    mission, image, direction = result
    assert isinstance(mission, str)  # TODO: Fishy
    assert isinstance(image, str)
    assert isinstance(direction, str)


def test_environment_step():
    """Test that the environment can perform a step."""
    env = _get_env()

    # Take a random action (0 should be valid for most MiniGrid envs)
    action = 0
    result = env.step(action)

    # Should return obs, reward, terminated, truncated, info
    assert isinstance(result, tuple)
    assert len(result) == 5

    obs, reward, terminated, truncated, info = result
    assert obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
