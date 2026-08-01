"""Hydra entry point for the Object-Trace Cells task (stochastic object presence).

Mirrors tasks/omt/main_task.py; see tasks/otc/task.py for why presence is
randomised per episode.
"""

import datetime
import warnings

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import wandb

warnings.filterwarnings("ignore", category=UserWarning)

from prnn.utils import ActionEncodingsEnum, MinigridEnvNames
from prnn.utils.Shell import FaramaMinigridShell

from curious_george import (
    AgentInputType,
    AgentType,
    get_ckpt_env_vars,
    get_model_dir,
    make_env,
)
from tasks.otc.task import ObjectTraceTask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(config_path="../../Configs", config_name="main", version_base=None)
def main(args: DictConfig):
    agent_type = AgentType.RANDOM if args.exp.random_action_agent else AgentType.AC
    date = datetime.datetime.now().strftime("%m%d-%H%M%S")
    run_name = f"{args.exp.exp_name}-otc-p{args.tasks.otc.presence_prob}-{date}"

    if args.logging.wandb_log:
        wandb.init(entity=args.logging.wandb_entity, project=args.logging.wandb_project,
                   name=run_name, group=f"{args.exp.exp_name}-otc")
        wandb.config.update(OmegaConf.to_container(args, resolve=True))
        wandb.define_metric("step_count")
        wandb.define_metric("Analysis/*", step_metric="step_count")

    prnn_ckpt, ac_ckpt = get_ckpt_env_vars(agent_type)
    obj_pos = list(args.tasks.new_obj_loc)

    def build(seed: int, with_object: bool) -> FaramaMinigridShell:
        return make_env(
            env_key=MinigridEnvNames.LRoom,
            new_obj_pos=obj_pos if with_object else None,
            input_type=AgentInputType.H_PO.value,
            act_enc=ActionEncodingsEnum.SpeedHD.value,
            seed=seed,
        )

    num_envs = args.exp.get("num_envs", 1)
    # start object-present; task.train re-randomises presence every batch
    envs_train = [build(args.exp.seed + 10000 + 1000 * i, True) for i in range(num_envs)]
    env_eval = build(args.exp.seed, False)  # probe env never has the object

    print(f"OTC: {num_envs} train envs, presence_prob={args.tasks.otc.presence_prob}, "
          f"object {obj_pos}, device {DEVICE}")

    task = ObjectTraceTask(
        args=args,
        agent_type=agent_type,
        envs_train=envs_train,
        env_eval=env_eval,
        device=DEVICE,
        save_path=get_model_dir(run_name),
        prnn_ckpt=prnn_ckpt,
        acmodel_status_ckpt=ac_ckpt,
        obj_pos=obj_pos,
    )
    task.train(
        num_trajs=args.tasks.training.num_trajs,
        presence_prob=args.tasks.otc.presence_prob,
        saving_interval_trajs=args.tasks.training.saving_interval_trajs,
        lr_trials=args.tasks.training.lr_trials,
        lrgroups=args.tasks.training.lrgroups,
        seed=args.exp.seed,
    )

    if args.logging.wandb_log:
        wandb.finish()


if __name__ == "__main__":
    main()
