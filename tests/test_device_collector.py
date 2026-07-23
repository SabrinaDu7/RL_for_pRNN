"""End-to-end equivalence gate for device-resident rollout collection."""

from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
import pytest
import torch

from curious_george.training.setup import setup_training


REPO = Path(__file__).resolve().parents[1]


def _rng_state(device):
    accelerator = None
    if device.type == "cuda":
        accelerator = torch.cuda.get_rng_state(device).clone()
    elif device.type == "mps":
        accelerator = torch.mps.get_rng_state().clone()
    return torch.get_rng_state().clone(), accelerator


def _set_rng_state(device, state):
    cpu, accelerator = state
    torch.set_rng_state(cpu)
    if device.type == "cuda":
        torch.cuda.set_rng_state(accelerator, device)
    elif device.type == "mps":
        torch.mps.set_rng_state(accelerator)


def _config(
    device_env: bool,
    reward_alignment: str = "next_obs",
    extra_overrides: tuple[str, ...] = (),
):
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
                "rl.frames=128",
                "predNet.seqdur=16",
                "predNet.hiddensize=64",
                "predNet.noisestd=0",
                "predNet.dropout=0",
                "predNet.batched_curiosity=true",
                f"rl.reward_alignment={reward_alignment}",
                *extra_overrides,
            ],
        )


def _assert_rollouts_equal(
    reference,
    device,
    expected,
    actual,
    expected_logs,
    actual_logs,
):
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
    assert np.array_equal(reference.directions, device.directions)

    for left, right in zip(
        expected.last_observations, actual.last_observations
    ):
        assert int(left["direction"]) == int(right["direction"])
        right_image = (
            right["image"].cpu().numpy()
            if torch.is_tensor(right["image"])
            else right["image"]
        )
        assert np.array_equal(left["image"], right_image)

    for field in ("curious_rewards", "values", "advantages", "joint_dist"):
        assert np.array_equal(expected_logs[field], actual_logs[field]), field


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
    reference_rng = _rng_state(reference.device)
    device = setup_training(
        _config(device_env=True, reward_alignment=reward_alignment)
    ).algo
    device_rng = _rng_state(device.device)

    try:
        # Two updates catch reset-schedule and RNG drift that a one-shot
        # comparison cannot observe. Each update contains two segments/env.
        for _ in range(2):
            _set_rng_state(reference.device, reference_rng)
            expected, expected_logs = reference.collect_experiences()
            reference_rng = _rng_state(reference.device)

            _set_rng_state(device.device, device_rng)
            actual, actual_logs = device.collect_experiences()
            device_rng = _rng_state(device.device)

            _assert_rollouts_equal(
                reference, device, expected, actual, expected_logs, actual_logs
            )
            assert all(
                left is None or torch.equal(left, right)
                for left, right in zip(reference_rng, device_rng)
            )
    finally:
        device.envs.close()
        for shell in reference.envs:
            shell.env.close()


@pytest.mark.parametrize(
    "extra_overrides",
    (
        ("exp.with_obs=true",),
        ("exp.pRNN=false",),
    ),
)
def test_device_collector_supports_policy_observation_variants(extra_overrides):
    algo = setup_training(
        _config(device_env=True, extra_overrides=extra_overrides)
    ).algo
    try:
        experiences, _ = algo.collect_experiences()
        assert len(experiences.action) == 128
    finally:
        algo.envs.close()


def test_random_init_control_does_not_update_either_model():
    algo = setup_training(_config(device_env=True)).algo
    policy_before = [param.detach().clone() for param in algo.acmodel.parameters()]
    world_before = [param.detach().clone() for param in algo.pN.pRNN.parameters()]
    try:
        experiences, _ = algo.collect_experiences()
        logs = algo.update_parameters(experiences, update_params=False)
        assert np.isfinite(logs["policy_loss"])
        assert all(
            torch.equal(before, after)
            for before, after in zip(policy_before, algo.acmodel.parameters())
        )
        assert all(
            torch.equal(before, after)
            for before, after in zip(world_before, algo.pN.pRNN.parameters())
        )
    finally:
        algo.envs.close()


def test_optimizer_betas_are_configurable():
    algo = setup_training(
        _config(
            device_env=True,
            extra_overrides=("rl.optim_betas=[0.8,0.97]",),
        )
    ).algo
    try:
        assert algo.optimizer.param_groups[0]["betas"] == (0.8, 0.97)
    finally:
        algo.envs.close()
