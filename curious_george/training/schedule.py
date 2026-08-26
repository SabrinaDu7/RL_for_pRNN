"""Run length and logging cadence, derived from one ground truth.

The ground truth for a run is **how much experience to collect**: a number of
episodes, each `predNet.seqdur` steps long. Everything else - total
environment steps, how many updates that takes, how many optimizer steps the
world model and the policy actually get - follows from that plus the rollout
shape, and is derived here rather than restated in the config.

This exists because an *update* is not a fixed amount of experience. It is
`rl.frames` steps, which scales with `exp.num_envs`, so any quantity counted
in updates silently rescales when the rollout shape changes. Counting
`performance=ultra` in updates is what turns a 10,000-update run into a
78-update one: identical experience, 1/128th the events.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingSchedule:
    """Every derived count for one run.

    episodes_total:     trajectories to collect over the whole run.
    episode_steps:      steps per trajectory (`predNet.seqdur`).
    frames_per_update:  environment steps collected per update (`rl.frames`).
    ppo_epochs/ppo_batch_size: the policy optimizer's minibatching.
    pooled_world_model: `predNet.batched_wm` - pooled world-model gradient
                        steps instead of one per episode segment.
    pool_group:         `predNet.wm_pool_group` - pooled in groups of g, so
                        segments/g steps per update rather than one. 0 = pool
                        everything into a single step.
    segment_stride:     `predNet.wm_segment_stride` - serial only, train on
                        every k-th segment, so segments/k steps per update.
    """

    episodes_total: int
    episode_steps: int
    frames_per_update: int
    ppo_epochs: int
    ppo_batch_size: int
    pooled_world_model: bool
    pool_group: int = 0
    segment_stride: int = 1

    @staticmethod
    def episodes_for_wm_steps(cfg, wm_steps_total: int) -> int:
        """`rl.episodes_total` that yields `wm_steps_total` world-model steps.

        `episodes_total` is experience, and how much TRAINING that buys depends
        on the regime: at `wm_pool_group=8` it is one gradient step per 8
        episodes, at 1 it is one per episode. So the same `episodes_total`
        trains the world model 8x less at g=8 - two arms budgeted that way are
        not comparable runs, they are one run and a fifth of another.

        Inverts `total_world_model_steps` through the regime rather than
        assuming the pooled case, so it stays correct for serial and for
        `wm_segment_stride` too.
        """
        probe = TrainingSchedule.from_config(cfg)
        per_update = probe.world_model_steps_per_update
        updates = -(-int(wm_steps_total) // max(per_update, 1))
        return updates * probe.episodes_per_update

    @classmethod
    def from_config(cls, cfg) -> "TrainingSchedule":
        return cls(
            episodes_total=int(cfg.rl.episodes_total),
            episode_steps=int(cfg.predNet.seqdur),
            frames_per_update=int(cfg.rl.frames),
            ppo_epochs=int(cfg.rl.ppo_epochs),
            ppo_batch_size=int(cfg.rl.ppo_batch_size),
            pooled_world_model=bool(cfg.predNet.get("batched_wm", False)),
            pool_group=int(cfg.predNet.get("wm_pool_group", 0)),
            segment_stride=int(cfg.predNet.get("wm_segment_stride", 1)),
        )

    @property
    def total_steps(self) -> int:
        """The loop bound: environment steps to collect."""
        return self.episodes_total * self.episode_steps

    @property
    def episodes_per_update(self) -> int:
        return self.frames_per_update // self.episode_steps

    @property
    def total_updates(self) -> int:
        return self.total_steps // self.frames_per_update

    @property
    def world_model_steps_per_update(self) -> int:
        """World-model optimizer steps per update, across all three regimes.

        This number is the run's actual gradient-step budget and the startup
        summary is what the configs tell readers to trust over their own
        comments, so it must track `wm_pool_group` and `wm_segment_stride`. It
        did not when those were added, and printed "1/update, POOLED" for a run
        genuinely taking 16.
        """
        n = self.episodes_per_update
        if self.pooled_world_model:
            g = self.pool_group
            return max(1, -(-n // g)) if g and g > 0 else 1
        return max(1, -(-n // max(1, self.segment_stride)))

    @property
    def policy_steps_per_update(self) -> int:
        return self.ppo_epochs * (self.frames_per_update // self.ppo_batch_size)

    @property
    def total_world_model_steps(self) -> int:
        return self.world_model_steps_per_update * self.total_updates

    @property
    def total_policy_steps(self) -> int:
        return self.policy_steps_per_update * self.total_updates

    @property
    def env_steps_per_world_model_step(self) -> int:
        """The ratio that decides whether the world model is over- or
        under-trained relative to experience (docs/exp_speed_cuda_graph
        2026-08-19 9h). Printing it makes the regime legible at a glance."""
        return self.frames_per_update // max(1, self.world_model_steps_per_update)

    @property
    def env_steps_per_policy_step(self) -> int:
        """The policy's counterpart to `env_steps_per_world_model_step`.

        A DIFFERENT number, on DIFFERENT knobs: the world model's ratio is
        `episode_steps * pool_group` (or `* segment_stride` when serial), the
        policy's is `ppo_batch_size / ppo_epochs` and does not involve seqdur at
        all. In the 2026-08 reference config they are 2048 and 512 - a factor of
        4 - which is why "gradient steps" is never one axis here.
        """
        return self.frames_per_update // max(1, self.policy_steps_per_update)

    def gradient_steps_at(self, update: int) -> dict[str, int]:
        """Cumulative optimizer steps after `update` updates, per learner.

        Logged alongside `frames` so a wandb panel can be read on either axis.
        Environment steps answer "how much experience", gradient steps answer
        "how much training" - and the two come apart exactly when the rollout
        shape changes, which is the failure this module exists to prevent.
        """
        return {
            "wm_grad_steps": self.world_model_steps_per_update * update,
            "policy_grad_steps": self.policy_steps_per_update * update,
        }

    @property
    def _wm_regime(self) -> str:
        if self.pooled_world_model:
            g = self.pool_group
            return f", POOLED in groups of {g}" if g and g > 0 else ", POOLED (all)"
        k = max(1, self.segment_stride)
        return "" if k == 1 else f", SERIAL stride {k}"

    def summary(self) -> str:
        """Printed at startup: optimizer-step budgets are the thing that
        silently collapses when the rollout shape changes, so state them."""
        return "\n".join(
            [
                "training schedule",
                f"  {self.episodes_total} episodes x {self.episode_steps} steps "
                f"= {self.total_steps} environment steps",
                f"  {self.frames_per_update} steps/update "
                f"({self.episodes_per_update} episodes) -> {self.total_updates} updates",
                f"  world-model gradient steps: {self.total_world_model_steps} "
                f"({self.world_model_steps_per_update}/update{self._wm_regime}"
                f") = 1 per {self.env_steps_per_world_model_step} env steps",
                f"  policy gradient steps:      {self.total_policy_steps} "
                f"({self.policy_steps_per_update}/update)"
                f" = 1 per {self.env_steps_per_policy_step} env steps",
            ]
        )


@dataclass(frozen=True)
class EntropySchedule:
    """`rl.entropy_coef` as a function of progress through the run.

    Exists because policy collapse is a LATE phenomenon and a constant is the
    wrong shape for it. Measured at `entropy_coef=0`, policy_entropy holds near
    1.43 at 20M environment steps and falls to 0.59-1.18 by 60-80M, with
    `MI_policy` spiking inversely - the agent stops exploring and the world
    model's input distribution lurches. A coefficient that RISES with progress
    puts the resistance where the drift is.

    The two endpoints are measured, the middle is not: 0.0 drifts to 0.79-1.12
    over a long run, 0.01 pins entropy near its 1.98 maximum and crushes
    MI_policy to 0.015 (the bonus is 0.0198 against an advantage scale of
    0.0369 - over half the learning signal). The target band is the 2026-07
    reference's own: MI 0.20-0.33 at entropy 1.34-1.55.

    `final is None` means constant, which is the historical behaviour.
    """

    start: float
    final: float | None
    total_steps: int

    @classmethod
    def from_config(cls, cfg) -> "EntropySchedule":
        f = cfg.rl.get("entropy_coef_final", None)
        return cls(
            start=float(cfg.rl.entropy_coef),
            final=None if f is None else float(f),
            total_steps=int(cfg.rl.episodes_total) * int(cfg.predNet.seqdur),
        )

    def at(self, step: int) -> float:
        """Linear in ENVIRONMENT STEPS, which is the axis collapse tracks -
        not in updates, which mean different amounts of experience at
        different rollout shapes."""
        if self.final is None:
            return self.start
        frac = min(max(step / max(self.total_steps, 1), 0.0), 1.0)
        return self.start + frac * (self.final - self.start)

    def summary(self) -> str:
        if self.final is None:
            return f"  entropy_coef: {self.start} (constant)"
        return (f"  entropy_coef: {self.start} -> {self.final} "
                f"linearly over {self.total_steps} env steps")


class StepCadence:
    """Fires an event once every `every` environment steps.

    `every <= 0` disables the event, preserving the historical "0 = off"
    meaning of the interval settings. Firing is edge-triggered on elapsed
    steps rather than `update % n`, so the cadence is unchanged by the rollout
    shape - an update larger than `every` fires every update, and an update
    smaller than `every` fires only once enough steps have accumulated.
    """

    def __init__(self, every: int, start_step: int = 0) -> None:
        self.every = int(every)
        # A resumed run starts with num_frames already advanced; without this
        # the first comparison would be against 0 and every event would fire
        # on the first update.
        self._last = int(start_step)

    def fire(self, step: int) -> bool:
        """True if due, recording the firing. Call at most once per update."""
        if self.every <= 0 or step - self._last < self.every:
            return False
        self._last = step
        return True


@dataclass
class TrainingCadence:
    """The interval-triggered sections of the training loop, in steps."""

    log: StepCadence
    plot: StepCadence
    analysis: StepCadence
    save: StepCadence
    archive: StepCadence

    @classmethod
    def from_config(cls, cfg, start_step: int = 0) -> "TrainingCadence":
        log = cfg.logging
        return cls(
            log=StepCadence(log.log_every_steps, start_step),
            plot=StepCadence(log.plot_every_steps, start_step),
            analysis=StepCadence(log.analysis_every_steps, start_step),
            save=StepCadence(log.save_every_steps, start_step),
            archive=StepCadence(log.get("archive_every_steps", 0), start_step),
        )
