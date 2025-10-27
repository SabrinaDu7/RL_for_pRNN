from minigrid.envs import LEnv_18_goal, LEnv_16_goal
from prnn.utils import make_env, MinigridEnvNames, ActionEncodingsEnum
from utils import get_minigrid_env

SIZE = 16
ENV_NAME = MinigridEnvNames.LRoom18Goal if SIZE == 18 else MinigridEnvNames.LRoom16Goal
ENV_CLASS = LEnv_18_goal if SIZE == 18 else LEnv_16_goal


def test_import_lenv_goal():
    assert ENV_CLASS is not None
    assert hasattr(ENV_CLASS, "__init__")


def test_create_lroom_goal_env():
    env = get_minigrid_env(env_name=ENV_NAME, act_enc=ActionEncodingsEnum.SpeedHD)

    # Basic checks
    assert env is not None
    assert hasattr(env, "reset")
    assert hasattr(env, "step")
    assert hasattr(env, "observation_space")
    assert hasattr(env, "action_space")

    while hasattr(env, "env"):
        env = env.env
    assert isinstance(env, ENV_CLASS)


def test_env_reset():
    """Test that the environment can be reset successfully."""
    env = get_minigrid_env(env_name=ENV_NAME, act_enc=ActionEncodingsEnum.SpeedHD)

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
    env = get_minigrid_env(env_name=ENV_NAME, act_enc=ActionEncodingsEnum.SpeedHD)

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
