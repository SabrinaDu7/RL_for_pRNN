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
    pooled_world_model: `predNet.batched_wm` - one pooled world-model gradient
                        step per update instead of one per episode segment.
    """

    episodes_total: int
    episode_steps: int
    frames_per_update: int
    ppo_epochs: int
    ppo_batch_size: int
    pooled_world_model: bool

    @classmethod
    def from_config(cls, cfg) -> "TrainingSchedule":
        return cls(
            episodes_total=int(cfg.rl.episodes_total),
            episode_steps=int(cfg.predNet.seqdur),
            frames_per_update=int(cfg.rl.frames),
            ppo_epochs=int(cfg.rl.ppo_epochs),
            ppo_batch_size=int(cfg.rl.ppo_batch_size),
            pooled_world_model=bool(cfg.predNet.get("batched_wm", False)),
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
        """Pooling replaces one gradient step per episode segment with one."""
        return 1 if self.pooled_world_model else self.episodes_per_update

    @property
    def policy_steps_per_update(self) -> int:
        return self.ppo_epochs * (self.frames_per_update // self.ppo_batch_size)

    @property
    def total_world_model_steps(self) -> int:
        return self.world_model_steps_per_update * self.total_updates

    @property
    def total_policy_steps(self) -> int:
        return self.policy_steps_per_update * self.total_updates

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
                f"({self.world_model_steps_per_update}/update"
                f"{', POOLED' if self.pooled_world_model else ''})",
                f"  policy gradient steps:      {self.total_policy_steps} "
                f"({self.policy_steps_per_update}/update)",
            ]
        )


class StepCadence:
    """Fires an event once every `every` environment steps.

    `every <= 0` disables the event, preserving the historical "0 = off"
    meaning of the interval settings. Firing is edge-triggered on elapsed
    steps rather than `update % n`, so the cadence is unchanged by the rollout
    shape - an update larger than `every` fires every update, and an update
    smaller than `every` fires only once enough steps have accumulated.
    """

    def __init__(self, every: int) -> None:
        self.every = int(every)
        self._last = 0

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

    @classmethod
    def from_config(cls, cfg) -> "TrainingCadence":
        log = cfg.logging
        return cls(
            log=StepCadence(log.log_every_steps),
            plot=StepCadence(log.plot_every_steps),
            analysis=StepCadence(log.analysis_every_steps),
            save=StepCadence(log.save_every_steps),
        )
