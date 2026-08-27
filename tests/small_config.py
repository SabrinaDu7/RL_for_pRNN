"""One small, quiet `Config` for tests, so nine of them stop hand-rolling it.

Every test that used to `compose(config_name="main", overrides=[...])` wanted the
same thing: a few environment instances, short episodes, a narrow pRNN, no
wandb, and no cadenced events firing. That was spelled out longhand in each
file, which is how `rl.frames=128` with `predNet.seqdur=16` came to mean
"8 episodes per rollout" in five places without any of them saying so.

The budget is expressed as `rollouts`, because that is what a collector test
actually cares about - how many rounds of collection happen - and everything
else follows from the config's own derivation rather than from arithmetic
repeated per file.
"""

from __future__ import annotations

from dataclasses import replace

from curious_george.configs import (
    ArchPolicyCfg,
    ArchPrnnCfg,
    CollectCfg,
    Config,
    EnvBackend,
    EvalCfg,
    EvalKind,
    EnvCfg,
    RewardAlignment,
    RunCfg,
    TrainPolicyCfg,
    TrainPrnnCfg,
)

__all__ = ["small_config"]


def small_config(
    *,
    # -- collection ---------------------------------------------------------
    num_envs: int = 4,
    episodes_per_env: int = 2,
    episode_steps: int = 16,
    backend: EnvBackend = EnvBackend.DEVICE,
    rollout_cuda_graph: bool = False,
    # -- budget, as rounds of collection ------------------------------------
    rollouts: int = 2,
    ppo_batch_size: int | None = None,
    ppo_epochs: int = 4,
    # -- world model --------------------------------------------------------
    hidden_size: int = 64,
    noise_std: float = 0.0,
    dropout: float = 0.0,
    batched: bool = False,
    episodes_per_grad_step: int = 1,
    batched_curiosity: bool = True,
    prnn_cuda_graph: bool = False,
    train_prnn: bool = True,
    # -- policy -------------------------------------------------------------
    entropy_coef: float = 0.0,
    reward_alignment: RewardAlignment = RewardAlignment.NEXT_OBS,
    policy_cuda_graph: bool = False,
    # -- the rest -----------------------------------------------------------
    env: EnvCfg | None = None,
    seed: int = 2,
    early_stop: bool = False,
    evals: frozenset[EvalKind] = frozenset(),
) -> Config:
    """A `Config` sized for a test.

    `noise_std` and `dropout` default to 0 so a comparison between two paths is
    about the paths and not about two draws from the same distribution.
    `evals` defaults to empty and every cadence to 0, so nothing fires unless a
    test asks for it.
    """
    episodes_per_rollout = num_envs * episodes_per_env
    if episodes_per_rollout % episodes_per_grad_step:
        raise ValueError(
            f"episodes_per_grad_step ({episodes_per_grad_step}) must divide "
            f"episodes_per_rollout ({episodes_per_rollout})"
        )
    total_episodes = rollouts * episodes_per_rollout
    env_steps = total_episodes * episode_steps
    batch = ppo_batch_size if ppo_batch_size is not None else env_steps // rollouts // ppo_epochs

    return Config(
        env=env if env is not None else EnvCfg(),
        collect=CollectCfg(
            num_envs=num_envs,
            episodes_per_env=episodes_per_env,
            episode_steps=episode_steps,
            backend=backend,
            rollout_cuda_graph=rollout_cuda_graph,
        ),
        arch_prnn=ArchPrnnCfg(hidden_size=hidden_size, noise_std=noise_std, dropout=dropout),
        arch_policy=ArchPolicyCfg(),
        train_prnn=TrainPrnnCfg(
            total_grad_steps=total_episodes // episodes_per_grad_step,
            episodes_per_grad_step=episodes_per_grad_step,
            batched=batched,
            batched_curiosity=batched_curiosity,
            cuda_graph=prnn_cuda_graph,
            train=train_prnn,
        ),
        train_policy=TrainPolicyCfg(
            total_grad_steps=ppo_epochs * env_steps // batch,
            ppo_epochs=ppo_epochs,
            entropy_coef=entropy_coef,
            reward_alignment=reward_alignment,
            cuda_graph=policy_cuda_graph,
        ),
        eval=EvalCfg(
            evals=evals,
            log_every_steps=0,
            plot_every_steps=0,
            analysis_every_steps=0,
        ),
        run=RunCfg(
            seed=seed,
            wandb=False,
            save_every_steps=0,
            archive_every_steps=0,
            early_stop=early_stop,
        ),
    )


def with_backend(cfg: Config, backend: EnvBackend) -> Config:
    """The same run on a different stepping backend - the comparison
    `test_device_collector` exists to make."""
    return replace(cfg, collect=replace(cfg.collect, backend=backend))
