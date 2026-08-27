"""A logged metric is a measurement or it is not there.

The DEVICE backend rejects environment rewards and terminations, so extrinsic
return is not measurable under it. It used to be recorded as `0.0` anyway, where
nothing distinguishes "the agent earned nothing" from "this configuration does
not measure it". Harmless in a goal-less L-room, where the true return is 0
anyway - and wrong the first time this backend runs `MiniGrid-LRoom_Goal-v0`,
which exists and is exercised by tests/test_table_env.py.

The rule these gate: absent, not zero.

These are about the COLLECTOR's logs dict, which still carries
`return_per_episode` because `run.early_stop` reads it. The `return_*` wandb
series were pruned on 2026-08-27; that is a separate question from whether the
quantity is measured.
"""


import pytest

SMALL = [
    "logging.wandb_log=false",
    "predNet.seqdur=32",
    "exp.seed=4",
]


def _logs(overrides):
    from curious_george.training.setup import setup_training

    cfg = overrides
    comps = setup_training(cfg)
    _, logs = comps.algo.collect_experiences()
    return cfg, logs


@pytest.fixture(scope="module")
def device_logs():
    from curious_george.configs import EnvBackend
    from tests.small_config import small_config

    return _logs(small_config(num_envs=4, backend=EnvBackend.DEVICE, ppo_batch_size=32))


@pytest.fixture(scope="module")
def cpu_logs():
    from curious_george.configs import EnvBackend
    from tests.small_config import small_config

    return _logs(small_config(num_envs=1, episodes_per_env=4,
                              backend=EnvBackend.SERIAL, ppo_batch_size=16))


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
    """`early_stop` reads extrinsic return. With the fabricated zero it compared
    `0.0 > 0.9` every update and silently never fired; then it raised mid-run.

    Now it cannot be BUILT: the refusal moved from `run_training` into
    `Config.__post_init__`, so a run that could never stop early fails at parse
    time instead of after the allocation is already spent. That is the whole
    point of typing the config - job 10444304 and its kin died minutes in on
    contradictions that were knowable at t=0.
    """
    from curious_george.configs import EnvBackend
    from tests.small_config import small_config

    monkeypatch.setenv("RL_STORAGE", str(tmp_path))

    with pytest.raises(ValueError, match="early_stop needs extrinsic return"):
        small_config(num_envs=4, backend=EnvBackend.DEVICE, early_stop=True)

    # ...and it is allowed on a backend that CAN measure it.
    ok = small_config(num_envs=1, episodes_per_env=4,
                      backend=EnvBackend.SERIAL, early_stop=True)
    assert ok.run.early_stop
