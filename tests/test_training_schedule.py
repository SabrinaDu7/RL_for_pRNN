"""TrainingSchedule: the derived counts a run is actually budgeted by.

`rl.episodes_total` is experience, not training. How much training it buys
depends on the world-model regime, so budgeting two arms by episodes silently
gives them different optimisation budgets - which is what these cover.
"""

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
