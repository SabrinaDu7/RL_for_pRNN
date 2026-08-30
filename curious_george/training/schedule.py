"""Every derived count for a run, from the two budgets that are ground truth.

`TrainPrnnCfg.total_grad_steps` and `TrainPolicyCfg.total_grad_steps` are stated;
environment steps, episode counts, rollout counts and `ppo_batch_size` are
derived here and nowhere else.

This module exists because a ROLLOUT is not a fixed amount of experience. It is
`num_envs * episodes_per_env` episodes, which scales with the collection shape,
so any quantity counted in rollouts silently rescales when that shape changes -
which is how a 10,000-rollout run became a 78-rollout one with identical
experience. Cadences are therefore in environment steps, and budgets in gradient
steps.

THE TWO LEARNERS TRAIN AT DIFFERENT RATES against the same experience, on
different knobs:

    pRNN    1 step per  episode_steps * episodes_per_prnn_step
    policy  1 step per  ppo_batch_size / ppo_epochs

In the 2026-08 reference those are 2048 and 512 - a factor of 4 - and the
policy's does not involve `episode_steps` at all. That is why there are two
counters and never one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingSchedule:
    """Every derived count for one run.

    Built from a `Config` and holding only plain integers, so it can be printed,
    serialised into provenance, and compared across runs without dragging the
    whole config along.
    """

    total_prnn_steps: int
    total_policy_steps: int
    episode_steps: int
    episodes_per_prnn_step: int
    num_envs: int
    episodes_per_env: int
    ppo_epochs: int

    @classmethod
    def from_config(cls, cfg) -> "TrainingSchedule":
        return cls(
            total_prnn_steps=cfg.train_prnn.total_grad_steps,
            total_policy_steps=cfg.train_policy.total_grad_steps,
            episode_steps=cfg.collect.episode_steps,
            episodes_per_prnn_step=cfg.train_prnn.episodes_per_grad_step,
            num_envs=cfg.collect.num_envs,
            episodes_per_env=cfg.collect.episodes_per_env,
            ppo_epochs=cfg.train_policy.ppo_epochs,
        )

    # -- experience --------------------------------------------------------

    @property
    def total_env_steps(self) -> int:
        """The loop bound. DERIVED from the pRNN budget, not stated."""
        return self.total_prnn_steps * self.env_steps_per_prnn_step

    @property
    def total_episodes(self) -> int:
        return self.total_prnn_steps * self.episodes_per_prnn_step

    # -- the rollout -------------------------------------------------------

    @property
    def episodes_per_rollout(self) -> int:
        return self.num_envs * self.episodes_per_env

    @property
    def env_steps_per_rollout(self) -> int:
        """What `rl.frames` used to be, written by hand in five places."""
        return self.episodes_per_rollout * self.episode_steps

    @property
    def total_rollouts(self) -> int:
        return self.total_env_steps // self.env_steps_per_rollout

    # -- rates: pRNN, one counter, all experience fresh ---------------------

    @property
    def env_steps_per_prnn_step(self) -> int:
        return self.episodes_per_prnn_step * self.episode_steps

    @property
    def prnn_steps_per_rollout(self) -> int:
        return self.episodes_per_rollout // self.episodes_per_prnn_step

    # -- rates: policy, two counters, because it replays -------------------

    @property
    def ppo_batch_size(self) -> int:
        """DERIVED from the two budgets. Scaling it with the rollout is what
        diluted the policy 16x while the world model trained 8x more per
        environment step; stating the step count makes that unrepresentable."""
        return self.ppo_epochs * self.total_env_steps // self.total_policy_steps

    @property
    def env_steps_per_policy_step(self) -> int:
        """FRESH experience behind one step. Same axis as the pRNN's rate."""
        return self.ppo_batch_size // self.ppo_epochs

    @property
    def processed_transitions_per_policy_step(self) -> int:
        """What a minibatch contains, counting the `ppo_epochs` reuse."""
        return self.ppo_batch_size

    @property
    def policy_steps_per_rollout(self) -> int:
        return self.ppo_epochs * self.env_steps_per_rollout // self.ppo_batch_size

    # -- logging -----------------------------------------------------------

    def gradient_steps_at(self, rollout: int) -> dict[str, int]:
        """Cumulative optimizer steps after `rollout` rollouts, per learner.

        Logged beside environment steps so a wandb panel can be read on either
        axis: environment steps answer "how much experience", gradient steps
        answer "how much training", and the two come apart exactly when the
        collection shape changes.
        """
        return {
            "prnn_grad_steps": self.prnn_steps_per_rollout * rollout,
            "policy_grad_steps": self.policy_steps_per_rollout * rollout,
        }

    def as_dict(self) -> dict[str, int]:
        """The resolved budget, for provenance beside the config that asked."""
        return {
            "total_env_steps": self.total_env_steps,
            "total_episodes": self.total_episodes,
            "total_rollouts": self.total_rollouts,
            "env_steps_per_rollout": self.env_steps_per_rollout,
            "prnn_grad_steps": self.total_prnn_steps,
            "policy_grad_steps": self.total_policy_steps,
            "env_steps_per_prnn_step": self.env_steps_per_prnn_step,
            "env_steps_per_policy_step": self.env_steps_per_policy_step,
            "ppo_batch_size": self.ppo_batch_size,
        }

    def summary(self) -> str:
        """Printed at startup. States both budgets AND both rates, because the
        rates are what silently change when the collection shape does."""
        return "\n".join(
            [
                "training schedule",
                f"  {self.total_episodes:,} episodes x {self.episode_steps} steps "
                f"= {self.total_env_steps:,} environment steps",
                f"  rollout: {self.episodes_per_rollout} episodes "
                f"({self.num_envs} envs x {self.episodes_per_env}) "
                f"= {self.env_steps_per_rollout:,} env steps "
                f"-> {self.total_rollouts:,} rollouts",
                f"  pRNN   gradient steps: {self.total_prnn_steps:,} "
                f"({self.prnn_steps_per_rollout}/rollout, "
                f"{self.episodes_per_prnn_step} episodes each)"
                f" = 1 per {self.env_steps_per_prnn_step:,} env steps",
                f"  policy gradient steps: {self.total_policy_steps:,} "
                f"({self.policy_steps_per_rollout}/rollout, "
                f"batch {self.ppo_batch_size:,} x {self.ppo_epochs} epochs)"
                f" = 1 per {self.env_steps_per_policy_step:,} env steps (fresh)",
            ]
        )


@dataclass(frozen=True)
class EntropySchedule:
    """`entropy_coef` as a function of progress through the run.

    Collapse IS a late phenomenon: at `action_offset=1` and a flat 0.001 the
    first excursion below 1.0 bits of policy entropy lands at 80.2% of training
    (75.7% at flat 0.002), and at a flat 0 policy entropy holds near 1.43 at 20M
    environment steps and falls to 0.59-1.18 by 60-80M with mutual information
    spiking inversely.

    🔴 BUT THE RAMP DOES NOT FIX IT, AND MAKES IT WORSE. Measured
    (`mila-off1-e0.001to0.01-s2`, against the flat arms at matched gradient
    steps): a 0.001 -> 0.01 ramp spends **13.3%** of updates below 1.0 bits
    against flat 0.001's 1.9% and flat 0.003's 0.0%, starts collapsing at 35.2%
    of training rather than 80.2%, and has the WORST prediction loss of every
    arm run (0.00590). The reasoning "a rising coefficient puts the resistance
    where the drift is" assumed collapse is driven by the CONTEMPORANEOUS
    coefficient. It is not: at 35% of training the ramp already sits near
    0.0042, above the flat 0.003 that never collapses at all. What matters is
    whether the policy was allowed to sharpen EARLY, which a ramp specifically
    permits.

    Prefer a flat coefficient. 0.003 is the measured knee at `action_offset=1`:
    the lowest value with a 0.0% collapse duty cycle, at a prediction loss
    (0.00458) matching the offset-0 baseline and 3.7x the mutual information of
    a flat 0.01.

    Both endpoints are measured, the middle is not: 0.0 drifts to 0.79-1.12 over
    a long run; 0.01 pins entropy near its 1.98 maximum and crushes mutual
    information to 0.015, the bonus being over half the learning signal.

    `final is None` means constant.
    """

    start: float
    final: float | None
    total_env_steps: int

    @classmethod
    def from_config(cls, cfg) -> "EntropySchedule":
        return cls(
            start=cfg.train_policy.entropy_coef,
            final=cfg.train_policy.entropy_coef_final,
            total_env_steps=TrainingSchedule.from_config(cfg).total_env_steps,
        )

    def at(self, step: int) -> float:
        """Linear in ENVIRONMENT STEPS, which is the axis collapse tracks - not
        in rollouts, which mean different amounts of experience at different
        collection shapes."""
        if self.final is None:
            return self.start
        frac = min(max(step / max(self.total_env_steps, 1), 0.0), 1.0)
        return self.start + frac * (self.final - self.start)

    def summary(self) -> str:
        if self.final is None:
            return f"  entropy_coef: {self.start} (constant)"
        return (
            f"  entropy_coef: {self.start} -> {self.final} "
            f"linearly over {self.total_env_steps:,} env steps"
        )


class StepCadence:
    """Fires an event once every `every` environment steps.

    `every <= 0` disables the event. Firing is edge-triggered on elapsed steps
    rather than `rollout % n`, so the cadence is unchanged by the collection
    shape - a rollout larger than `every` fires every rollout, and a rollout
    smaller than `every` fires only once enough steps have accumulated.
    """

    def __init__(self, every: int, start_step: int = 0) -> None:
        self.every = int(every)
        # A resumed run starts with steps already elapsed; without this the
        # first comparison would be against 0 and every event would fire
        # immediately.
        self._last = int(start_step)

    def fire(self, step: int) -> bool:
        """True if due, recording the firing. Call at most once per rollout."""
        if self.every <= 0 or step - self._last < self.every:
            return False
        self._last = step
        return True


@dataclass
class TrainingCadence:
    """The interval-triggered sections of the training loop, in env steps."""

    log: StepCadence
    plot: StepCadence
    analysis: StepCadence
    save: StepCadence
    archive: StepCadence

    @classmethod
    def from_config(cls, cfg, start_step: int = 0) -> "TrainingCadence":
        return cls(
            log=StepCadence(cfg.eval.log_every_steps, start_step),
            plot=StepCadence(cfg.eval.plot_every_steps, start_step),
            analysis=StepCadence(cfg.eval.analysis_every_steps, start_step),
            save=StepCadence(cfg.run.save_every_steps, start_step),
            archive=StepCadence(cfg.run.archive_every_steps, start_step),
        )
