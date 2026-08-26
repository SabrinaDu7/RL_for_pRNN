"""TrainingSchedule: the derived counts a run is actually budgeted by.

The gradient-step totals are stated; environment steps, episode counts, rollout
counts and `ppo_batch_size` are derived. These cover the derivation, and in
particular that the two learners' rates move on their own knobs - collapsing
them into one axis is what hid a 4x difference.

`episodes_for_wm_steps` is gone with the Hydra config, and so is the test that
covered it: it inverted "episodes -> gradient steps" through the regime, and
stating `total_grad_steps` directly IS that inverse.
"""

from dataclasses import replace

import pytest

from curious_george.configs import PRESETS, Config, TrainPolicyCfg, TrainPrnnCfg
from curious_george.training.schedule import EntropySchedule, TrainingSchedule
from tests.small_config import small_config


def test_cumulative_gradient_steps_agree_with_the_totals():
    """`gradient_steps_at(total_rollouts)` must equal the budgeted totals.

    These are the numbers a wandb x-axis is read on AND the numbers the startup
    summary promises. If they disagree, one is lying and a plot cannot say which.
    """
    for name, (_, cfg) in PRESETS.items():
        s = cfg.schedule
        at_end = s.gradient_steps_at(s.total_rollouts)
        assert at_end["prnn_grad_steps"] == s.total_prnn_steps, name
        assert at_end["policy_grad_steps"] == s.total_policy_steps, name


def test_the_two_learners_train_at_different_rates():
    """One "gradient steps" axis would hide a 4x difference.

    The pRNN's rate is episode_steps x episodes_per_grad_step; the policy's is
    ppo_batch_size / ppo_epochs and does not involve episode_steps at all.
    """
    cfg = Config(
        collect=replace(small_config().collect, num_envs=8, episodes_per_env=1,
                        episode_steps=256, backend=PRESETS["reference"][1].collect.backend),
        train_prnn=TrainPrnnCfg(total_grad_steps=1000, episodes_per_grad_step=8, batched=True),
        train_policy=TrainPolicyCfg(total_grad_steps=16_000),
    )
    s = cfg.schedule
    assert s.env_steps_per_prnn_step == 256 * 8
    assert s.env_steps_per_policy_step == s.ppo_batch_size // s.ppo_epochs
    assert s.env_steps_per_prnn_step != s.env_steps_per_policy_step


def test_the_policy_budget_moves_only_the_policy_rate():
    """What makes two axes meaningful rather than two names for one number.

    NOT a claim that each rate is independent of every other field: with the
    budgets stated, `episode_steps` and `episodes_per_grad_step` both change
    TOTAL EXPERIENCE, and the policy's rate is experience/budget, so they move
    it too. The clean statement is about the budget itself - doubling the
    policy's budget halves its rate and leaves the pRNN's untouched.
    """
    base = PRESETS["reference"][1]
    denser = replace(base, train_policy=replace(
        base.train_policy, total_grad_steps=2 * base.train_policy.total_grad_steps))

    assert denser.schedule.total_env_steps == base.schedule.total_env_steps
    assert denser.schedule.env_steps_per_prnn_step == base.schedule.env_steps_per_prnn_step
    assert denser.schedule.env_steps_per_policy_step == (
        base.schedule.env_steps_per_policy_step // 2
    )


def test_deriving_the_batch_pins_the_policy_fresh_rate_to_the_budgets():
    """A consequence of DERIVING ppo_batch_size, and the point of doing so.

    fresh rate = ppo_batch_size / ppo_epochs
               = (ppo_epochs * env_steps / total_grad_steps) / ppo_epochs
               = env_steps / total_grad_steps

    So `ppo_epochs` CANNOT move how much new experience backs a policy step -
    it moves the minibatch and the reuse factor instead. Under the old config,
    where the batch was set by hand, changing epochs silently changed the rate.
    Only the two budgets set it now, which is what makes two arms comparable.
    """
    base = PRESETS["reference"][1]
    more_epochs = replace(base, train_policy=replace(base.train_policy, ppo_epochs=8))

    assert more_epochs.schedule.env_steps_per_prnn_step == base.schedule.env_steps_per_prnn_step
    assert more_epochs.schedule.env_steps_per_policy_step == base.schedule.env_steps_per_policy_step
    # ...but the batch and the reuse DO move.
    assert more_epochs.schedule.ppo_batch_size == 2 * base.schedule.ppo_batch_size
    assert more_epochs.schedule.processed_transitions_per_policy_step == (
        2 * base.schedule.processed_transitions_per_policy_step
    )

    # And the fresh rate is exactly experience / budget.
    s = base.schedule
    assert s.env_steps_per_policy_step == s.total_env_steps // s.total_policy_steps


def test_rollout_shape_does_not_move_the_budget():
    """`frames` was never a scientific knob. Widening the rollout 16x leaves
    both gradient-step totals untouched, because they are stated."""
    base = PRESETS["reference"][1]
    wide = replace(base, collect=replace(base.collect, num_envs=128,
                                         backend=PRESETS["ultra"][1].collect.backend))
    assert wide.schedule.env_steps_per_rollout == 16 * base.schedule.env_steps_per_rollout
    assert wide.schedule.total_prnn_steps == base.schedule.total_prnn_steps
    assert wide.schedule.total_policy_steps == base.schedule.total_policy_steps
    assert wide.schedule.total_env_steps == base.schedule.total_env_steps


def test_entropy_ramp_puts_its_resistance_late():
    """A rising coefficient must resist LATE, where collapse is.

    Measured at entropy_coef=0: policy entropy holds near 1.43 at 20M env steps
    and falls to 0.59-1.18 by 60-80M. A constant is the wrong shape for that;
    so is a DECREASING ramp, which would be weakest exactly where the drift is.
    """
    cfg = small_config()
    cfg = replace(cfg, train_policy=replace(
        cfg.train_policy, entropy_coef=0.001, entropy_coef_final=0.01))
    s = EntropySchedule.from_config(cfg)
    total = cfg.total_env_steps

    assert s.at(0) == pytest.approx(0.001)
    assert s.at(total // 2) == pytest.approx(0.0055)
    assert s.at(total) == pytest.approx(0.01)
    assert s.at(10 * total) == pytest.approx(0.01), "must clamp past the end"
    assert s.at(-5) == pytest.approx(0.001), "must clamp before the start"

    const = EntropySchedule.from_config(
        replace(cfg, train_policy=replace(cfg.train_policy, entropy_coef_final=None))
    )
    assert const.at(0) == const.at(total) == pytest.approx(0.001), "None means constant"


def test_schedule_is_built_from_the_config_and_agrees_with_it():
    """`Config.schedule` and the config's own properties are one derivation."""
    for name, (_, cfg) in PRESETS.items():
        assert cfg.schedule.total_env_steps == cfg.total_env_steps, name
        assert cfg.schedule.ppo_batch_size == cfg.ppo_batch_size, name
        assert cfg.schedule.total_rollouts == cfg.total_rollouts, name


def test_as_dict_carries_what_provenance_needs():
    """A run's record must let a reader place it on a step axis without
    re-deriving anything."""
    d = TrainingSchedule.from_config(PRESETS["reference"][1]).as_dict()
    assert d["total_env_steps"] == 20_480_000
    assert d["prnn_grad_steps"] == 80_000
    assert d["policy_grad_steps"] == 320_000
    assert d["ppo_batch_size"] == 256
