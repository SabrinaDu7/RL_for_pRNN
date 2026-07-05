import warnings
from functools import partial

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObservationWrapper, Wrapper
from gymnasium.wrappers.record_video import RecordVideo
from minigrid.wrappers import *

warnings.filterwarnings("ignore", category=UserWarning)

from prnn.utils import ActionEncodingsEnum, MinigridEnvNames
from prnn.utils.Shell import FaramaMinigridShell

from curious_george.enums import AgentInputType

wrappers = {
    "ReseedWrapper": ReseedWrapper,
    "ActionBonus": ActionBonus,
    "StateBonus": StateBonus,
    "ImgObsWrapper": ImgObsWrapper,
    "OneHotPartialObsWrapper": OneHotPartialObsWrapper,
    "RGBImgObsWrapper": RGBImgObsWrapper,
    "RGBImgPartialObsWrapper": RGBImgPartialObsWrapper,
    "RGBImgPartialObsWrapper_HD": RGBImgPartialObsWrapper_HD,
    "FullyObsWrapper": FullyObsWrapper,
    "DictObservationSpaceWrapper": DictObservationSpaceWrapper,
    "FlatObsWrapper": FlatObsWrapper,
    "ViewSizeWrapper": ViewSizeWrapper,
    "DirectionObsWrapper": DirectionObsWrapper,
    "SymbolicObsWrapper": SymbolicObsWrapper,
}


# replace lambda function so i can pickle pNet
def episode_video_trigger(episode, vid_n_episodes):
    return episode % vid_n_episodes == 0


def get_env_name(env):
    """Helper method to get the actual environment class name through the wrapper hierarchy."""
    temp_env = env
    names = []
    while temp_env is not None:
        class_name = type(temp_env).__name__
        names.append(class_name)
        if hasattr(temp_env, "env"):
            temp_env = temp_env.env
        else:
            break
    return " -> ".join(names)


def make_env(
    env_key: str,
    input_type: str,
    agent_start_pos: tuple[int, int] | None = None,
    agent_start_dir: int | None = None,
    seed=0,
    vid_folder="",
    vid_n_episodes=0,
    wrapper=None,
    render_mode="rgb_array",
    act_enc: str | None = None,
    **kwargs,  # e.g., subroom_size and open_all_paths for FourRooms, size for LRoom
):
    assert input_type in AgentInputType
    assert env_key in MinigridEnvNames
    assert act_enc in ActionEncodingsEnum

    env = gym.make(
        env_key,
        agent_start_pos=agent_start_pos,
        agent_start_dir=agent_start_dir,
        render_mode=render_mode,
        **kwargs,
    )

    if input_type == "Visual_FO":
        # Not RGB one here because we want RL agent to have as much info as possible
        env = FullyObsWrapper(env)

    elif "pRNN" in input_type or "PO" in input_type:
        # The same RGB wrapper is used for comparability whenever partial observation is needed
        env = RGBImgPartialObsWrapper_HD(env, tile_size=1)

    else:
        # For the cases without any visual input
        env = HDObsWrapper(env)

    if wrapper:
        env = wrappers[wrapper](env, **kwargs)

    # Below I replaced the lambda function. This allows me to pickle pNet without errors
    if vid_n_episodes:
        # env = RecordVideo(env, video_folder=vid_folder, episode_trigger=lambda x: x%vid_n_episodes == 0)
        trigger_func = partial(episode_video_trigger, vid_n_episodes=vid_n_episodes)
        env = RecordVideo(env, video_folder=vid_folder, episode_trigger=trigger_func)

    env.reset(seed=seed)
    env = FaramaMinigridShell(env, act_enc, env_key)

    # if 'pRNN' in input_type or 'CANN' in input_type or 'Intrinsic' in input_type:
    #     env = FaramaMinigridShell(env, act_enc, env_key)
    # else:
    #     env = ResetWrapper(env)
    #     env.reset(seed=seed)

    return env


class ResetWrapper(Wrapper):
    """
    Wrapper to return a single dictionary of observation, not a tuple with empty second element.
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)[0]


class HDObsWrapper(ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as observation.
    This can be used to have the agent to solve the gridworld in pixel space.
    Including direction information (HD)
    """

    def __init__(self, env: FaramaMinigridShell):
        super().__init__(env)
        HD_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "HD": HD_space}
        )

    def observation(self, obs):
        return {"mission": obs["mission"], "HD": obs["direction"]}
