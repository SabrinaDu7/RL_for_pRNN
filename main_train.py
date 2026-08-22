"""Training entry point: hydra config -> setup -> loop.

Replaces trainRL_Adel.py (kept as a thin shim). All construction lives in
curious_george/training/setup.py, the loop in training/loop.py, wandb
logging in training/logging.py.
"""

import warnings
from dataclasses import replace

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from curious_george.training.logging import init_wandb
from curious_george.training.loop import run_training
from curious_george.training.setup import setup_run, setup_training

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(config_path="Configs", config_name="main")
def my_main(cfg: DictConfig):
    my_app(cfg)


def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    run_ctx = setup_run(cfg)
    if run_ctx.wandb_log:
        # A logging backend must not be able to kill a training run. Job
        # 10444214 died 61 s in because wandb could not initialise on a compute
        # node, losing the whole allocation for a metrics sink whose numbers
        # are also recoverable offline from the archived checkpoints
        # (scripts/multienv/checkpoint_curve.py). Degrade, do not abort.
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

    comps = setup_training(cfg)
    run_training(cfg, run_ctx, comps)

    if run_ctx.wandb_log:
        wandb.finish()


if __name__ == "__main__":
    my_main()
