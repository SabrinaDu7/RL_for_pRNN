import matplotlib.pyplot as plt
import wandb
import time
import warnings

### Suppress warnings ###
warnings.filterwarnings("ignore", category=UserWarning)

from prnn.utils.predictiveNet import loadNet
from prnn.analysis.ObjectMemoryTask.define_task import ObjectMemoryTask
from RLutils.other import DEVICE

NET_NAME = "thRNN_5win_16_new_arch"
ENV_NAME = "MiniGrid-LRoom_Goal-16x16-v0"
RESULTS_SAVE_FOLDER = "results"
NETS_SAVE_FOLDER = "nets"
TIME = time.strftime("%m%d-%H%M")

RUN = wandb.init(
    entity="sabrina-du-mila-mila",
    project="curious-george",
)


def main():
    print("Loading pre-trained network...")
    predictiveNet = loadNet(NET_NAME)

    # Step 2: Run the Object Memory Task
    print("Running Object Memory Task...")
    print(f"Using device: {DEVICE}")
    omt = ObjectMemoryTask(
        predictiveNet,
        env_novel_name=ENV_NAME,
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
        full_filename=f"{NETS_SAVE_FOLDER}/{NET_NAME}-{TIME}.pkl",
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


if __name__ == "__main__":
    main()