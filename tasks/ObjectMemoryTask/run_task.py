import matplotlib.pyplot as plt
import wandb
import time
import warnings
import hydra
from omegaconf import DictConfig

### Suppress warnings ###
warnings.filterwarnings("ignore", category=UserWarning)

from prnn.utils import MinigridEnvNames
from tasks.ObjectMemoryTask.define_task import ObjectMemoryTask
from utils import get_wandb_env_vars, get_env_var
from RLutils.other import DEVICE

DEVICE = "cpu"
WANDB_ENTITY, WANDB_PROJECT = get_wandb_env_vars()
CKPT_DIR = get_env_var("CKPT_DIR")

RESULTS_SAVE_FOLDER = "results"
TIME = time.strftime("%m%d-%H%M")
NET_NAME = f"pN-omt-{TIME}"

def create_wandb_run():
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"{TIME}",
    )
    return run

@hydra.main(config_path="../../Configs", config_name="Conf1_Adel")
def main(args: DictConfig):
    print("Loading pre-trained network...")
    if args.logging.wandb_log:
        create_wandb_run()
    
    # Step 2: Run the Object Memory Task
    print("Running Object Memory Task...")
    print(f"Using device: {DEVICE}")
    omt = ObjectMemoryTask(
        args=args,
        env_novel_name=MinigridEnvNames.LRoom16Goal,
    )
    omt.trainNovelObject(
        epochs=5,
        num_trials=500,
        sequence_duration=1000,
        lr_trials=2,
        lrgroups=[0, 1, 2],
        resetOptimizer=False,
        continueTraining=False,
        device=DEVICE,
        full_filename=f"{CKPT_DIR}/{NET_NAME}.pt",
    )
    omt.getTestTrial(timesteps=2500)
    objectLearning = omt.quantifyObjectLearning(control_location=[2, 7], whichPhase=0)

    # Step 3: Display results
    print("\nResults:")
    print(f"Goal modulation: {objectLearning['goalmodulation']:.4f}")
    print(f"Control modulation: {objectLearning['ctlmodulation_diffloc']:.4f}")

    # Step 4: Generate plots
    print("Generating plots...")
    plt.figure(figsize=(12, 8))
    omt.ObjectLearningFigure(netname=NET_NAME, savefolder=RESULTS_SAVE_FOLDER)
    print("Done!")

    if args.logging.wandb_log:
        wandb.finish()


if __name__ == "__main__":
    main()