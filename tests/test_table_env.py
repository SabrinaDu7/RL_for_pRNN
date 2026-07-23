"""Differential correctness tests for the table-driven L-room environment."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from minigrid.core.world_object import Key

from curious_george.envs.factory import make_env
from curious_george.envs.obs_bank import (
    BankedRGBPartialObsWrapper,
    TableDrivenRGBPartialObsWrapper,
)
from curious_george.envs.vector import DeviceTableShellPool


ENV_KEYS = (
    "MiniGrid-LRoom-v0",
    "MiniGrid-LRoom_LineGreen-v0",
    "MiniGrid-LRoom_Goal-v0",
)


def _make_wrapper(env_key: str, wrapper_cls, bank_dir, seed: int = 17):
    env = gym.make(
        env_key,
        agent_start_pos=None,
        agent_start_dir=None,
        render_mode="rgb_array",
    )
    wrapped = wrapper_cls(env, tile_size=1, bank_dir=bank_dir)
    wrapped.reset(seed=seed)
    return wrapped


def _set_state(env, x: int, y: int, direction: int, step_count: int) -> None:
    base = env.unwrapped
    base.agent_pos = (x, y)
    base.agent_dir = direction
    base.step_count = step_count
    base.carrying = None


def _assert_observation_equal(reference: dict, fast: dict) -> None:
    assert reference["mission"] == fast["mission"]
    assert reference["direction"] == fast["direction"]
    assert np.array_equal(reference["image"], fast["image"])


@pytest.mark.parametrize("env_key", ENV_KEYS)
def test_every_valid_state_action_matches_minigrid(env_key, tmp_path):
    """Exhaustively compare every occupiable (x, y, dir, action).

    Two step counts cover ordinary and max-step truncating transitions; the
    goal variant also exercises MiniGrid's step-dependent terminal reward.
    """
    reference = _make_wrapper(env_key, BankedRGBPartialObsWrapper, tmp_path)
    fast = _make_wrapper(env_key, TableDrivenRGBPartialObsWrapper, tmp_path)
    base = reference.unwrapped

    for x in range(base.width):
        for y in range(base.height):
            cell = base.grid.get(x, y)
            if cell is not None and not cell.can_overlap():
                continue
            for direction in range(4):
                for action in range(base.action_space.n):
                    for step_count in (0, base.max_steps - 1):
                        _set_state(reference, x, y, direction, step_count)
                        _set_state(fast, x, y, direction, step_count)

                        expected = reference.step(action)
                        actual = fast.step(action)

                        _assert_observation_equal(expected[0], actual[0])
                        assert expected[1:] == actual[1:]
                        assert reference.unwrapped.agent_pos == fast.unwrapped.agent_pos
                        assert reference.unwrapped.agent_dir == fast.unwrapped.agent_dir
                        assert reference.unwrapped.step_count == fast.unwrapped.step_count


@pytest.mark.parametrize("env_key", ENV_KEYS)
def test_seeded_reset_chain_and_rollout_match(env_key, tmp_path):
    reference = _make_wrapper(env_key, BankedRGBPartialObsWrapper, tmp_path)
    fast = _make_wrapper(env_key, TableDrivenRGBPartialObsWrapper, tmp_path)
    rng = np.random.default_rng(123)

    for episode in range(8):
        if episode:
            reference_obs, reference_info = reference.reset()
            fast_obs, fast_info = fast.reset()
            _assert_observation_equal(reference_obs, fast_obs)
            assert reference_info == fast_info
            assert reference.unwrapped.agent_pos == fast.unwrapped.agent_pos
            assert reference.unwrapped.agent_dir == fast.unwrapped.agent_dir

        for _ in range(64):
            action = int(rng.integers(0, 4))
            expected = reference.step(action)
            actual = fast.step(action)
            _assert_observation_equal(expected[0], actual[0])
            assert expected[1:] == actual[1:]


def test_table_env_rejects_mutable_pickable_layout(tmp_path):
    fast = _make_wrapper(
        "MiniGrid-LRoom-v0", TableDrivenRGBPartialObsWrapper, tmp_path
    )
    fast.unwrapped.put_obj(Key("yellow"), 7, 2)
    with pytest.raises(ValueError, match="pickable"):
        fast._build_transition_tables()


def _make_shell(seed: int):
    return make_env(
        "MiniGrid-LRoom-v0",
        "pRNN",
        seed=seed,
        act_enc="SpeedHD",
        table_env=True,
        agent_start_pos=None,
        agent_start_dir=None,
    )


def test_device_table_pool_matches_independent_shells():
    """Compare every batched device result to separately stepped CPU shells.

    This covers reset RNG streams, positions, directions, RGB observations,
    and rewards over multiple synchronized episodes. The transition table
    itself is already exhaustive-tested above.
    """
    B = 7
    seeds = [101 + 1000 * b for b in range(B)]
    references = [_make_shell(seed) for seed in seeds]
    training_shells = [_make_shell(seed) for seed in seeds]
    eval_shell = _make_shell(999_999)
    pool = DeviceTableShellPool(
        training_shells=training_shells,
        eval_shell=eval_shell,
        device=torch.device("cpu"),
    )
    rng = np.random.default_rng(44)

    try:
        for _episode in range(4):
            expected_obss = [shell.reset() for shell in references]
            actual_obss, actual_pos = pool.reset_all()
            for b in range(B):
                _assert_observation_equal(expected_obss[b], actual_obss[b])
                assert tuple(actual_pos[b]) == references[b].get_agent_pos()

            for _ in range(127):
                actions = rng.integers(0, 4, size=B, dtype=np.int64)
                expected = [
                    shell.step(actions[b]) for b, shell in enumerate(references)
                ]
                images, directions, rewards = pool.step_device(
                    actions=torch.from_numpy(actions)
                )
                images_np = images.numpy()
                directions_np = directions.numpy()
                rewards_np = rewards.numpy()

                for b, result in enumerate(expected):
                    _assert_observation_equal(
                        result[0],
                        {
                            "mission": pool.mission,
                            "image": images_np[b],
                            "direction": int(directions_np[b]),
                        },
                    )
                    assert rewards_np[b] == result[1]
                    assert tuple(pool.positions[b].tolist()) == references[
                        b
                    ].get_agent_pos()
    finally:
        pool.close()
        for shell in references:
            shell.env.close()
        eval_shell.env.close()
