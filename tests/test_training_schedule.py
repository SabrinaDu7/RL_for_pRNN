"""TrainingSchedule: the derived counts a run is actually budgeted by.

`rl.episodes_total` is experience, not training. How much training it buys
depends on the world-model regime, so budgeting two arms by episodes silently
gives them different optimisation budgets - which is what these cover.
"""

import pytest

def test_episodes_for_wm_steps_inverts_the_regime():
    """A gradient-step budget must survive a change of wm_pool_group.

    `rl.episodes_total` buys 8x less world-model training at wm_pool_group=8
    than at 1, so budgeting two arms by episodes makes them incomparable. This
    is the inverse: ask for N gradient steps and get the episodes that deliver
    them, whatever the regime.
    """
    from omegaconf import OmegaConf

    from curious_george.training.schedule import TrainingSchedule

    def cfg_for(pool):
        return OmegaConf.create({
            "rl": {"episodes_total": 80_000, "frames": 32768, "ppo_epochs": 4,
                   "ppo_batch_size": 256},
            "predNet": {"seqdur": 256, "batched_wm": True, "wm_pool_group": pool,
                        "wm_segment_stride": 1},
        })

    for pool in (1, 8):
        cfg = cfg_for(pool)
        ep = TrainingSchedule.episodes_for_wm_steps(cfg, 80_000)
        cfg.rl.episodes_total = ep
        got = TrainingSchedule.from_config(cfg).total_world_model_steps
        assert got == 80_000, f"pool={pool}: asked 80,000 steps, schedule gives {got}"

    # and the point of the exercise: g=8 must be told to collect 8x the episodes
    assert TrainingSchedule.episodes_for_wm_steps(cfg_for(8), 80_000) == \
        8 * TrainingSchedule.episodes_for_wm_steps(cfg_for(1), 80_000)


def test_entropy_schedule_ramps_over_environment_steps():
    """A rising coefficient must put its resistance LATE, where collapse is.

    Measured at entropy_coef=0: policy_entropy holds near 1.43 at 20M env steps
    and falls to 0.59-1.18 by 60-80M. A constant is the wrong shape for that;
    so is a DECREASING ramp, which would be weakest exactly where the drift
    happens.
    """
    from omegaconf import OmegaConf

    from curious_george.training.schedule import EntropySchedule

    cfg = OmegaConf.create({
        "rl": {"entropy_coef": 0.001, "entropy_coef_final": 0.01, "episodes_total": 1000},
        "predNet": {"seqdur": 256},
    })
    s = EntropySchedule.from_config(cfg)
    total = 1000 * 256
    assert s.at(0) == pytest.approx(0.001)
    assert s.at(total // 2) == pytest.approx(0.0055)
    assert s.at(total) == pytest.approx(0.01)
    assert s.at(10 * total) == pytest.approx(0.01), "must clamp past the end"
    assert s.at(-5) == pytest.approx(0.001), "must clamp before the start"

    cfg.rl.entropy_coef_final = None
    const = EntropySchedule.from_config(cfg)
    assert const.at(0) == const.at(total) == pytest.approx(0.001), "None means constant"


def _reference_cfg(*, seqdur=256, pool_group=8, ppo_batch_size=2048, ppo_epochs=4):
    """The 2026-08 fast-single reference shape, one knob at a time."""
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "rl": {"episodes_total": 351562, "frames": 65536,
               "ppo_epochs": ppo_epochs, "ppo_batch_size": ppo_batch_size},
        "predNet": {"seqdur": seqdur, "batched_wm": True,
                    "wm_pool_group": pool_group, "wm_segment_stride": 1},
    })


def test_cumulative_gradient_steps_agree_with_the_totals():
    """`gradient_steps_at(total_updates)` must equal the budgeted totals.

    These are the numbers a wandb x-axis is read on and the numbers the startup
    summary promises. If they disagree, one of them is lying and there is no way
    to tell which from a plot.
    """
    from curious_george.training.schedule import TrainingSchedule

    s = TrainingSchedule.from_config(_reference_cfg())
    at_end = s.gradient_steps_at(s.total_updates)
    assert at_end["wm_grad_steps"] == s.total_world_model_steps
    assert at_end["policy_grad_steps"] == s.total_policy_steps


def test_the_two_learners_train_at_different_rates():
    """One "gradient steps" axis would hide a 4x difference.

    The world model's ratio is seqdur * pool_group; the policy's is
    ppo_batch_size / ppo_epochs and does not involve seqdur at all. Conflating
    them is what the wm_pool_group / ppo_batch_size traps were.
    """
    from curious_george.training.schedule import TrainingSchedule

    s = TrainingSchedule.from_config(_reference_cfg())
    assert s.env_steps_per_world_model_step == 256 * 8      # seqdur * pool_group
    assert s.env_steps_per_policy_step == 2048 // 4         # batch / epochs
    assert s.env_steps_per_world_model_step != s.env_steps_per_policy_step


def test_the_policy_rate_ignores_seqdur_and_the_wm_rate_tracks_it():
    """Each ratio moves only on its own knobs - the property that makes two
    axes meaningful rather than two names for one number."""
    from curious_george.training.schedule import TrainingSchedule

    base = TrainingSchedule.from_config(_reference_cfg())
    longer = TrainingSchedule.from_config(_reference_cfg(seqdur=128))
    assert longer.env_steps_per_policy_step == base.env_steps_per_policy_step
    assert longer.env_steps_per_world_model_step != base.env_steps_per_world_model_step

    wider = TrainingSchedule.from_config(_reference_cfg(ppo_batch_size=1024))
    assert wider.env_steps_per_world_model_step == base.env_steps_per_world_model_step
    assert wider.env_steps_per_policy_step != base.env_steps_per_policy_step
