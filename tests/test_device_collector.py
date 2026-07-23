"""End-to-end equivalence gate for device-resident rollout collection."""

from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
import pytest
import torch

from curious_george.training.setup import setup_training


REPO = Path(__file__).resolve().parents[1]


def _config(device_env: bool, reward_alignment: str):
    with initialize_config_dir(
        config_dir=str(REPO / "Configs"), version_base=None
    ):
        return compose(
            config_name="main",
            overrides=[
                "logging.wandb_log=false",
                "exp.num_envs=4",
                "exp.async_envs=false",
                "exp.table_env=true",
                f"exp.device_env={str(device_env).lower()}",
                "rl.frames=64",
                "predNet.seqdur=16",
                "predNet.hiddensize=64",
                "predNet.noisestd=0",
                "predNet.dropout=0",
                "predNet.batched_curiosity=true",
                f"rl.reward_alignment={reward_alignment}",
            ],
        )


@pytest.mark.parametrize("reward_alignment", ["legacy", "next_obs"])
def test_device_collector_is_exactly_equal_to_cpu_table_collector(
    reward_alignment: str,
):
    """Same initial weights/RNG must produce the same complete rollout.

    This catches errors outside the transition table itself, including
    pre-vs-post direction alignment in SpeedHD encoding and flatten order.
    """
    reference = setup_training(
        _config(device_env=False, reward_alignment=reward_alignment)
    ).algo
    reference_rng = torch.get_rng_state().clone()
    device = setup_training(
        _config(device_env=True, reward_alignment=reward_alignment)
    ).algo
    device_rng = torch.get_rng_state().clone()

    torch.set_rng_state(reference_rng)
    expected, expected_logs = reference.collect_experiences()
    torch.set_rng_state(device_rng)
    actual, actual_logs = device.collect_experiences()

    for field in (
        "SR",
        "action",
        "value",
        "reward",
        "advantage",
        "returnn",
        "log_prob",
    ):
        assert torch.equal(getattr(expected, field), getattr(actual, field)), field

    assert torch.equal(expected.obs.image, actual.obs.image)
    assert torch.equal(expected.obs.direction, actual.obs.direction)
    assert expected.done_indices == actual.done_indices
    assert reference.locs == device.locs

    for left, right in zip(
        expected.last_observations, actual.last_observations
    ):
        assert left["direction"] == right["direction"]
        assert np.array_equal(left["image"], right["image"])

    for field in ("curious_rewards", "values", "advantages", "joint_dist"):
        assert np.array_equal(expected_logs[field], actual_logs[field]), field

    device.envs.close()
