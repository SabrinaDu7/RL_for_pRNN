"""`make_env`: the one way to build an environment shell."""

import warnings

import gymnasium as gym

from prnn.utils import ActionEncodingsEnum
from prnn.utils.Shell import FaramaMinigridShell

from curious_george.utils.enums import AgentInputType

warnings.filterwarnings("ignore", category=UserWarning)


def make_env(
    env_key: str,
    input_type: str,
    agent_start_pos: tuple[int, int] | None = None,
    agent_start_dir: int | None = None,
    seed=0,
    render_mode="rgb_array",
    act_enc: str | None = None,
    see_through_walls: bool | None = None,
    table_env: bool = False,
    **kwargs,  # e.g. `landmarks` for the -Multi-v0 rooms, `size` for LRoom
):
    """A registered MiniGrid room, wrapped for the pRNN and reset once at `seed`.

    `input_type` names how the agent observes; every member of `AgentInputType`
    is the pRNN's partial RGB view, served from the precomputed bank
    (`envs/obs_bank.py`) rather than rendered per step. The fully-observed and
    direction-only wrappers that used to hang off other members had no caller
    and were deleted 2026-09-06, as were the video recorder and the generic
    `wrapper=` hook.
    """
    assert input_type in AgentInputType
    # gymnasium's registry, not prnn's MinigridEnvNames, is the source of truth
    # for which ids exist: the enum lives in the pinned prnn package, so a new
    # env registered in minigrid would otherwise need a prnn release to be usable.
    if env_key not in gym.registry:
        known = sorted(k for k in gym.registry if k.startswith("MiniGrid-"))
        raise ValueError(f"unknown env id {env_key!r}; registered MiniGrid ids: {known}")
    assert act_enc in ActionEncodingsEnum

    env = gym.make(
        env_key,
        agent_start_pos=agent_start_pos,
        agent_start_dir=agent_start_dir,
        render_mode=render_mode,
        **kwargs,
    )

    if see_through_walls is not None:
        # LEnv hardcodes see_through_walls=True into super().__init__, and
        # gen_obs_grid reads the attribute at observation time, so setting it
        # post-construction is sufficient and avoids touching the env class.
        env.unwrapped.see_through_walls = see_through_walls

    # Same RGB partial obs as RGBImgPartialObsWrapper_HD (byte-equal,
    # tests/test_obs_bank.py) but served from the precomputed bank instead of a
    # per-step get_frame render.
    from curious_george.envs.obs_bank import (
        BankedRGBPartialObsWrapper,
        TableDrivenRGBPartialObsWrapper,
    )

    wrapper_cls = TableDrivenRGBPartialObsWrapper if table_env else BankedRGBPartialObsWrapper
    env = wrapper_cls(env, tile_size=1)

    env.reset(seed=seed)
    return FaramaMinigridShell(env, act_enc, env_key)
