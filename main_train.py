"""Training entry point: a typed config -> setup -> loop.

    uv run python main_train.py reference
    uv run python main_train.py multienv --run.seed 3
    uv run python main_train.py ultra --train-prnn.total-grad-steps 1000

`curious_george/configs.py` owns the schema and the presets; every default has
exactly one home there. Construction lives in `training/setup.py`, the loop in
`training/loop.py`, wandb logging in `training/logging.py`.
"""

import json
import warnings
from dataclasses import replace

import wandb

from curious_george import configs
from curious_george.training.logging import init_wandb
from curious_george.training.loop import run_training
from curious_george.training.setup import setup_run, setup_training

warnings.filterwarnings("ignore", category=UserWarning)


def train(cfg: configs.Config) -> None:
    """Run one training job from an already-built config.

    ORDER MATTERS, and it is the reverse of what this used to do. Everything is
    BUILT before anything is RECORDED, because a constructor may override what
    was requested - PredictiveNet can round the hidden size - and a record of
    the requested value is a record of a run that did not happen. The old order
    wrote provenance and opened wandb first, then mutated the config, so every
    provenance.json carried a width the run never used.

    Safe to do in this order: PredictiveNet's constructor stores `wandb_log`
    and never logs, so nothing needs wandb to exist yet.
    """
    print(json.dumps(cfg.to_dict(), indent=2))

    comps = setup_training(cfg)
    cfg = comps.cfg  # the EFFECTIVE config: what was built, not what was asked

    run_ctx = setup_run(cfg)
    if run_ctx.wandb_log:
        # A logging backend must not be able to kill a training run. Job
        # 10444214 died 61 s in because wandb could not initialise on a compute
        # node, losing the whole allocation for a metrics sink whose numbers
        # are also recoverable offline from the archived checkpoints
        # (curious_george/evaluation/checkpoint_series.py). Degrade, do not abort.
        try:
            init_wandb(cfg, run_ctx)
        except Exception as exc:  # noqa: BLE001 - any backend failure is survivable
            print(
                f"[wandb] init FAILED ({type(exc).__name__}: {exc}). "
                "Continuing WITHOUT wandb; scalars are lost but checkpoints and "
                "the offline spatial scoring are not.",
                flush=True,
            )
            run_ctx = replace(run_ctx, wandb_log=False)

    run_training(cfg, run_ctx, comps)

    if run_ctx.wandb_log:
        wandb.finish()


if __name__ == "__main__":
    train(configs.cli())
