import matplotlib.pyplot as plt
import wandb
import time
import warnings
import hydra
from omegaconf import DictConfig
import torch
import datetime

warnings.filterwarnings("ignore", category=UserWarning)

from prnn.utils import MinigridEnvNames
from tasks.ObjectMemoryTask.define_task import ObjectMemoryTask
from utils import get_wandb_env_vars, get_ckpt_env_vars, get_env_var, AgentType
 
# ===== Constants =====
DEVICE = torch.device("cuda")

WANDB_ENTITY, WANDB_PROJECT = get_wandb_env_vars()
RL_STORAGE = get_env_var("RL_STORAGE")

RESULTS_SAVE_FOLDER = "results"
TIME = time.strftime("%m%d-%H%M")
NET_NAME = f"pN-omt-{TIME}"

# ===== Helper functions =====
def create_wandb_run(agent_type: AgentType):
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"{agent_type}-{TIME}",
    )
    return run

# ===== Main =====
@hydra.main(config_path="../../Configs", config_name="Conf1_Adel")
def main(args: DictConfig):

    agent_type = AgentType.RANDOM if args.exp.random_action_agent else AgentType.AC# Setup
    if args.logging.wandb_log:
        create_wandb_run(agent_type)
    

    prnn_ckpt, ac_ckpt = get_ckpt_env_vars(agent_type)

    env_novel_name = MinigridEnvNames.LRoom16Goal if args.tasks.room_size == 16 else MinigridEnvNames.LRoom18Goal
    env_orig_name = MinigridEnvNames.LRoom16 if args.tasks.room_size == 16 else MinigridEnvNames.LRoom18

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    agent_name = "cur" if args.exp.curious_agent else "rand"
    run_name = f"{args.exp.exp_name}_{agent_name}_{date}"

    print(f"Running Object Memory Task with {agent_name} agent")
    print(f"Using device: {DEVICE}")
    
    omt = ObjectMemoryTask(
        args=args,
        agent_type=agent_type,
        env_novel_name=env_novel_name,
        env_orig_name=env_orig_name,
        save_path=f"{RL_STORAGE}/{run_name}",
        prnn_ckpt=prnn_ckpt,
        acmodel_status_ckpt=ac_ckpt,
        device=DEVICE,
    )
    
    omt.trainNovelObject(
        num_trajs=args.tasks.training.num_trajs,
        saving_interval=args.tasks.training.saving_interval,
        lr_trials=args.tasks.training.lr_trials,
        lrgroups=args.tasks.training.lrgroups,
        resetOptimizer=args.tasks.training.resetOptimizer,
        continueTraining=args.tasks.training.continueTraining,
        device=DEVICE,
        filename=f"{NET_NAME}.pt",
    )
    omt.getTestTrial(timesteps=args.tasks.testing.timesteps)
    objectLearning = omt.quantifyObjectLearning(
        control_location=args.tasks.testing.control_location,
        whichPhase=args.tasks.testing.whichPhase,
    )

    # Display results
    if objectLearning is not None: # In case control or goal locations were never viewed
        print("\nResults:")
        print(f"Goal modulation: {objectLearning['goalmodulation']:.4f}")
        print(f"Control modulation: {objectLearning['ctlmodulation_diffloc']:.4f}")

        # Generate plots
        print("Generating plots...")
        plt.figure(figsize=(12, 8))
        omt.ObjectLearningFigure(netname=NET_NAME, savefolder=RESULTS_SAVE_FOLDER)
        print("Done!")
    
    else:
        print("Object learning results could not be computed due to lack of views of goal or control locations during test trial.")

    if args.logging.wandb_log:
        wandb.finish()


if __name__ == "__main__":
    main()