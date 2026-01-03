import matplotlib.pyplot as plt
import wandb
import time
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import datetime

warnings.filterwarnings("ignore", category=UserWarning)

from prnn.utils import MinigridEnvNames, ActionEncodingsEnum
from prnn.utils.Shell import FaramaMinigridShell
from tasks.ObjectMemoryTask.define_task import ObjectMemoryTask
from tasks.ObjectMemoryTask.figure import figure_object_learning
from utils import get_wandb_env_vars, get_ckpt_env_vars, get_env_var, AgentType, AgentInputType
from RLutils import make_env
 
# ===== Constants =====
DEVICE = torch.device("cuda")

WANDB_ENTITY, WANDB_PROJECT = get_wandb_env_vars()
RL_STORAGE = get_env_var("RL_STORAGE")

RESULTS_SAVE_FOLDER = "results"
TIME = time.strftime("%m%d-%H%M")

# ===== Helper functions =====
def create_wandb_run(run_name: str, args: DictConfig):
    group_name = "-".join(run_name.rsplit("-", 1)[:-1])
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name,
        group=group_name,
    )
    wandb.config.update(OmegaConf.to_container(args, resolve=True))
    wandb.define_metric("step_count")
    wandb.define_metric("Analysis/*", step_metric="step_count")

    return run

def get_novel_env(room_type: str) -> FaramaMinigridShell:
    if room_type == "line":
        return make_env(env_key=MinigridEnvNames.LRoomLineGreen, input_type=AgentInputType.H_PO.value, act_enc=ActionEncodingsEnum.SpeedHD.value)
    elif room_type == "plus":
        return make_env(env_key=MinigridEnvNames.LRoom, plus_color="green", input_type=AgentInputType.H_PO.value, act_enc=ActionEncodingsEnum.SpeedHD.value)
    elif room_type == "dot":
        # LANDMINE: Hardcoded new object position
        return make_env(env_key=MinigridEnvNames.LRoom, new_obj_pos=[7, 2], input_type=AgentInputType.H_PO.value, act_enc=ActionEncodingsEnum.SpeedHD.value)
    elif room_type == "goal":
        return make_env(env_key=MinigridEnvNames.LRoomGoal, input_type=AgentInputType.H_PO.value, act_enc=ActionEncodingsEnum.SpeedHD.value)
    else:
        raise ValueError(f"Invalid room type: {room_type}")


# ===== Main =====
@hydra.main(config_path="../../Configs", config_name="Conf1_Adel")
def main(args: DictConfig):

    agent_type = AgentType.RANDOM if args.exp.random_action_agent else AgentType.AC
    date = datetime.datetime.now().strftime("%m%d-%H%M%S")
    agent_name = "cur" if args.exp.curious_agent else "rand"
    run_name = f"{args.exp.exp_name}-{agent_name}-{args.tasks.room_type_green}-{date}"

    if args.logging.wandb_log:
        create_wandb_run(run_name, args)
    
    prnn_ckpt, ac_ckpt = get_ckpt_env_vars(agent_type)

    # Get environments
    env_orig_name = MinigridEnvNames.LRoom
    env_orig = make_env(env_key=env_orig_name, input_type=AgentInputType.H_PO.value, act_enc=ActionEncodingsEnum.SpeedHD.value)
    env_novel = get_novel_env(args.tasks.room_type_green)
    assert env_novel.get_goal_loc() is None, "Novel environment cannot have extrinsic goal"
    
    # Run task
    print(f"Running Object Memory Task with {agent_name} agent")
    print(f"Using device: {DEVICE}")
    print(f"Object Learning Figure: {args.tasks.analysis.objLearning_fig}")
    
    omt = ObjectMemoryTask(
        args=args,
        agent_type=agent_type,
        env_orig=env_orig,
        env_novel=env_novel,
        save_path=f"{run_name}",
        prnn_ckpt=prnn_ckpt,
        acmodel_status_ckpt=ac_ckpt,
        device=DEVICE,
    )
    
    omt.trainNovelObject(
        num_trajs=args.tasks.training.num_trajs,
        saving_interval=args.tasks.training.saving_interval,
        analysis_interval=args.tasks.training.analysis_interval,
        lr_trials=args.tasks.training.lr_trials,
        lrgroups=args.tasks.training.lrgroups,
        resetOptimizer=args.tasks.training.resetOptimizer,
        continueTraining=args.tasks.training.continueTraining,
        device=DEVICE,
    )
    
    if args.logging.wandb_log:
        wandb.finish()


if __name__ == "__main__":
    main()