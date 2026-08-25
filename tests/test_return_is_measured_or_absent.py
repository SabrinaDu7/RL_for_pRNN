"""A logged metric is a measurement or it is not there.

`exp.device_env` uses a backend that rejects environment rewards and
terminations, so extrinsic return is not measurable under it. It used to be
recorded as `0.0` anyway and reach wandb as `return_mean`, where nothing
distinguishes "the agent earned nothing" from "this configuration does not
measure it". Harmless in a goal-less L-room, where the true return is 0 anyway
- and wrong the first time this backend runs `MiniGrid-LRoom_Goal-v0`, which
exists and is exercised by tests/test_table_env.py.

The rule these gate: absent, not zero.
"""

from pathlib import Path

import pytest

SMALL = [
    "logging.wandb_log=false",
    "predNet.seqdur=32",
    "exp.seed=4",
]


def _logs(overrides):
    from hydra import compose, initialize_config_dir

    from curious_george.training.setup import setup_training

    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        cfg = compose(config_name="main", overrides=SMALL + overrides)
    comps = setup_training(cfg)
    _, logs = comps.algo.collect_experiences()
    return cfg, logs


@pytest.fixture(scope="module")
def device_logs():
    return _logs([
        "env=lroom", "run=multienv", "exp.device_env=True",
        "exp.num_envs=4", "rl.frames=128", "rl.ppo_batch_size=32",
    ])


@pytest.fixture(scope="module")
def cpu_logs():
    return _logs([
        "env=lroom", "exp.device_env=False", "exp.num_envs=1",
        "rl.frames=64", "rl.ppo_batch_size=16",
    ])


@pytest.mark.parametrize("key", ["return_per_episode", "reshaped_return_per_episode"])
def test_absent_when_the_backend_cannot_measure_it(device_logs, key):
    _, logs = device_logs
    assert key not in logs, (
        f"{key} present under exp.device_env, which cannot measure extrinsic "
        "reward. A fabricated value here reaches wandb indistinguishable from "
        "a real one."
    )


@pytest.mark.parametrize("key", ["return_per_episode", "reshaped_return_per_episode"])
def test_present_when_the_backend_can_measure_it(cpu_logs, key):
    """The other half: this must not become 'never logged anywhere'."""
    _, logs = cpu_logs
    assert key in logs


@pytest.mark.parametrize("key", ["num_episodes", "num_frames_per_episode"])
def test_the_counts_that_are_real_stay(device_logs, key):
    """`done_counter` and `finished_frames` ARE known on this backend - the
    segment boundaries are synchronized and Python-visible. Only the two return
    fields were fabricated, so only those go."""
    _, logs = device_logs
    assert key in logs


def test_early_stop_refuses_rather_than_never_firing(tmp_path, monkeypatch):
    """`logging.early_stop` reads extrinsic return. With the fabricated zero it
    compared `0.0 > 0.9` every update and silently never fired; now it says so."""
    from hydra import compose, initialize_config_dir

    from curious_george.training.loop import run_training
    from curious_george.training.setup import setup_run, setup_training

    monkeypatch.setenv("RL_STORAGE", str(tmp_path))
    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        cfg = compose(config_name="main", overrides=SMALL + [
            "env=lroom", "run=multienv", "exp.device_env=True",
            "exp.num_envs=4", "rl.frames=128", "rl.ppo_batch_size=32",
            "rl.episodes_total=4", "logging.early_stop=True",
            "logging.analysis_every_steps=0", "logging.plot_every_steps=0",
            "logging.save_every_steps=0", "logging.archive_every_steps=0",
        ])
    assert cfg.logging.early_stop, "override did not take; the test proves nothing"

    with pytest.raises(ValueError, match="exp.device_env does not measure"):
        run_training(cfg, setup_run(cfg), setup_training(cfg))
