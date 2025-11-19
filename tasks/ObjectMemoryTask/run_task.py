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
from tasks.ObjectMemoryTask.figure import figure_object_learning
from utils import get_wandb_env_vars, get_ckpt_env_vars, get_env_var, AgentType
 
# ===== Constants =====
DEVICE = torch.device("cuda")

WANDB_ENTITY, WANDB_PROJECT = get_wandb_env_vars()
RL_STORAGE = get_env_var("RL_STORAGE")

RESULTS_SAVE_FOLDER = "results"
TIME = time.strftime("%m%d-%H%M")

# ===== Helper functions =====
def create_wandb_run(run_name: str):
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name,
    )
    return run

# ===== Main =====
@hydra.main(config_path="../../Configs", config_name="Conf1_Adel")
def main(args: DictConfig):

    agent_type = AgentType.RANDOM if args.exp.random_action_agent else AgentType.AC
    date = datetime.datetime.now().strftime("%m%d-%H%M%S")
    agent_name = "cur" if args.exp.curious_agent else "rand"
    run_name = f"{args.exp.exp_name}-{agent_name}-{date}"

    if args.logging.wandb_log:
        create_wandb_run(run_name)
    
    prnn_ckpt, ac_ckpt = get_ckpt_env_vars(agent_type)

    env_novel_name = MinigridEnvNames.LRoom16Goal if args.tasks.room_size == 16 else MinigridEnvNames.LRoom18Goal
    env_orig_name = MinigridEnvNames.LRoom16 if args.tasks.room_size == 16 else MinigridEnvNames.LRoom18

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
        analysis_interval=args.tasks.training.analysis_interval,
        lr_trials=args.tasks.training.lr_trials,
        lrgroups=args.tasks.training.lrgroups,
        resetOptimizer=args.tasks.training.resetOptimizer,
        continueTraining=args.tasks.training.continueTraining,
        device=DEVICE,
    )
    testTrial = omt.getTestTrial(timesteps=omt.trajs_test * omt.seqdur)
    torch.save(testTrial, f"{RL_STORAGE}/{run_name}/testTrial_{args.tasks.training.num_trajs}.pt")

    objectLearning = omt.quantifyObjectLearning(
        control_location=args.tasks.testing.control_location,
        whichPhase=args.tasks.testing.whichPhase,
        traj_count=args.tasks.training.num_trajs,
    )

    # Display results
    if objectLearning is not None: # In case control or goal locations were never viewed
        torch.save(objectLearning, f"{RL_STORAGE}/{run_name}/objectLearning_{args.tasks.training.num_trajs}.pt")
        print(f"Saved object learning results (traj {args.tasks.training.num_trajs}): {RL_STORAGE}/{run_name}/objectLearning_{args.tasks.training.num_trajs}.pt.")
        
        print("\nResults:")
        print(f"Goal modulation: {objectLearning['goalmodulation']:.4f}")
        print(f"Control modulation: {objectLearning['ctlmodulation_diffloc']:.4f}")

        # Generate plots
        print(f"Generating plots in: {RESULTS_SAVE_FOLDER}")
        figure_object_learning(env_name=env_orig_name, run_name=run_name, traj_num=args.tasks.training.num_trajs, save_folder=RESULTS_SAVE_FOLDER)
        print("Done!")
    
    else:
        print("Object learning results could not be computed due to lack of views of goal or control locations during test trial.")

    if args.logging.wandb_log:
        wandb.finish()


if __name__ == "__main__":
    main()