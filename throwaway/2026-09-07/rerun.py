"""Re-run a finished run's EXACT config on the current tree, locally.

The fresh-run check the cleanup ends with: the most recent run of each arm,
rebuilt from its own record - `provenance.json` for a run directory,
the wandb config record (the same `Config.to_dict`) for a run only wandb has -
and trained again on `sdu/clean-sept`, seed and budget included.

    uv run python throwaway/2026-09-07/rerun.py <run-dir | entity/project/run-id> <exp_name> <wandb_project> [--dry]
"""

import sys
from dataclasses import replace
from pathlib import Path

from curious_george.configs import Config


def config_of(ident: str) -> Config:
    if Path(ident).is_dir():
        return Config.of_run(ident)
    import wandb

    record = dict(wandb.Api(timeout=60).run(ident).config)
    record.pop("_wandb", None)
    return Config.from_dict(record)


def main() -> None:
    ident, exp_name, project = sys.argv[1:4]
    cfg = config_of(ident)
    cfg = replace(cfg, run=replace(
        cfg.run, exp_name=exp_name, wandb=True, wandb_project=project,
        output_dir=None, prnn_ckpt=None, policy_ckpt=None,
    ))
    print(cfg.schedule.summary())
    print(f"source={cfg.env.source!r} loss={cfg.arch_prnn.loss.value} readout={cfg.arch_prnn.readout.value} "
          f"gamma={cfg.arch_prnn.focal_gamma} seed={cfg.run.seed} entropy={cfg.train_policy.entropy_coef} "
          f"whiten={cfg.train_policy.normalize_advantage} rewnorm={cfg.train_policy.normalize_reward} "
          f"backend={cfg.collect.backend.value} graphs=({cfg.collect.rollout_cuda_graph},{cfg.train_prnn.cuda_graph},{cfg.train_policy.cuda_graph})")
    if "--dry" in sys.argv:
        return
    from main_train import train

    train(cfg)


if __name__ == "__main__":
    main()
