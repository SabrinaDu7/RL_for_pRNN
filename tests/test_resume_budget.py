"""A resumed run's budget is the TOTAL across phases, and says so when it is not.

The two-phase design Q2 needs is: train to completion, then continue from that
checkpoint under a changed environment. `run.prnn_ckpt` / `run.policy_ckpt` make
that a config change rather than new machinery - but the budget does NOT reset.
`training/loop.py` reads the elapsed step count out of the checkpoint and loops
`while num_frames < schedule.total_env_steps`, so a phase-2-sized budget used to
fall through the loop and exit 0: a run directory, a provenance file and a wandb
run, for a phase that never took a gradient step.

That is the worst available failure, because it looks like success. These pin
that it now refuses, and that a correctly-sized budget still runs.
"""

import pytest

from curious_george.training.schedule import TrainingSchedule
from tests.small_config import small_config


def _budget(cfg) -> int:
    return TrainingSchedule.from_config(cfg).total_env_steps


def test_a_phase_two_sized_budget_refuses_instead_of_exiting_clean():
    """The regression. `already` past the budget is what "I set phase 2's own
    step count" produces, and it must not read as a finished run."""
    from curious_george.training.loop import run_training

    cfg = small_config(rollouts=2)
    already = _budget(cfg) * 2

    with pytest.raises(ValueError, match="nothing to train"):
        run_training(cfg, _StubContext(), _StubComponents(num_frames=already))


def test_the_message_names_the_number_to_raise_it_past():
    """An error that does not say what to change costs the same as no error."""
    from curious_george.training.loop import run_training

    cfg = small_config(rollouts=2)
    already = _budget(cfg) + 1

    with pytest.raises(ValueError) as exc:
        run_training(cfg, _StubContext(), _StubComponents(num_frames=already))
    assert f"{already:,}" in str(exc.value)
    assert f"{_budget(cfg):,}" in str(exc.value)


def test_a_fresh_run_is_not_affected():
    """num_frames == 0 must not trip the guard - every run from scratch starts
    there, so a wrong comparison here would break everything."""
    from curious_george.training.loop import run_training

    cfg = small_config(rollouts=2)
    # Reaching the collector is out of scope here; the guard runs before it, so
    # a fresh run must fail LATER and differently, never with "nothing to train".
    with pytest.raises(Exception) as exc:
        run_training(cfg, _StubContext(), _StubComponents(num_frames=0))
    assert "nothing to train" not in str(exc.value)


class _StubContext:
    """Only what `run_training` touches before the guard."""

    run_name = "test"
    model_dir = "/dev/null/"
    video_dir = "/dev/null/"
    wandb_log = False


class _StubComponents:
    """The guard reads the status dict and nothing else. Anything the loop body
    would need is deliberately absent, so a test that got past the guard fails
    loudly rather than quietly training."""

    def __init__(self, *, num_frames: int) -> None:
        from curious_george.utils.checkpoints import StatusCkptKeys

        self.status = {
            StatusCkptKeys.NUM_FRAMES.value: num_frames,
            StatusCkptKeys.UPDATE.value: 0,
        }
        self.algo = None
        self.envs = None
        self.predictiveNet = None
        self.acmodel = None
