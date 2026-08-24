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
