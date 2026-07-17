"""Training entry point: hydra config -> setup -> loop.

Replaces trainRL_Adel.py (kept as a thin shim). All construction lives in
curious_george/training/setup.py, the loop in training/loop.py, wandb
logging in training/logging.py.
"""

import warnings

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
    if cfg.logging.wandb_log:
        init_wandb(cfg, run_ctx)

    comps = setup_training(cfg)
    run_training(cfg, run_ctx, comps)

    if cfg.logging.wandb_log:
        wandb.finish()


if __name__ == "__main__":
    my_main()
