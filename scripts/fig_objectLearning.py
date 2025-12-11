# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: RL_for_pRNN
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Visualize Object Learning Results

# %%
# %cd /home/sabrina/Documents/experiments/RL_for_pRNN

# %% [markdown]
# ## Imports and Function Definitions
# %%
import sys
import torch
from matplotlib import pyplot as plt

from tasks.ObjectMemoryTask.figure import figure_object_learning, figure_goal_modulation_vs_trajectories
from prnn.utils import MinigridEnvNames, ActionEncodingsEnum
from utils import AgentInputType, get_home_dir_env_var
from RLutils import make_env


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.append(get_home_dir_env_var())

ENVI_ORIG_NAME = MinigridEnvNames.LRoom

# %% [markdown]
# ## Functions

# %%
def get_env_img(env_novel_name: MinigridEnvNames, new_obj_pos = None) -> None:
    env = make_env(env_key=env_novel_name, new_obj_pos=new_obj_pos, input_type=AgentInputType.H_PO.value, act_enc=ActionEncodingsEnum.SpeedHD.value)
    env_img = env.render(mode=None)

    plt.figure()
    plt.imshow(env_img)
    plt.show()


def get_fig_objectLearning(env_orig_name: MinigridEnvNames, run_name: str, num_trajs: int) -> None:
    figure_object_learning(env_name=env_orig_name, 
                       run_name=run_name, 
                       traj_num=num_trajs, 
                       save_folder=run_name,
                       rl_storage=".",
                       show=True)


def print_modulations(run_name: str, num_trajs: int) -> None:
    objectLearning = torch.load(f"{run_name}/objectLearning_{num_trajs}.pt", weights_only=False)
    print(f"\nResults for {run_name}:")
    print(f"Goal modulation for: {objectLearning['goalmodulation']:.4f}")
    print(f"Control modulation: {objectLearning['ctlmodulation_diffloc']:.4f}")


def main(run_name_cur: str, run_name_rand: str, num_trajs: int) -> None:
    print_modulations(run_name=f"../{run_name_cur}", num_trajs=num_trajs)
    get_fig_objectLearning(env_orig_name=ENVI_ORIG_NAME, run_name=f"../{run_name_cur}", num_trajs=num_trajs)

    print_modulations(run_name=f"../{run_name_rand}", num_trajs=num_trajs)
    get_fig_objectLearning(env_orig_name=ENVI_ORIG_NAME, run_name=f"../{run_name_rand}", num_trajs=num_trajs)
    figure_goal_modulation_vs_trajectories(rand_dir=f"../{run_name_rand}", curious_dir=f"../{run_name_cur}", num_datapoints=16, plot_controls=False, transparency=0.3)

# %% [markdown]
# ## Bright Dot Plots
# %%
ENV_NOVEL_NAME = MinigridEnvNames.LRoom
get_env_img(env_novel_name=ENV_NOVEL_NAME, new_obj_pos=[7, 2])

# %%
if __name__ == "__main__":
    NUM_TRAJS=800
    RUN_NAME_CUR= "omt-bright-cur-dot-1126-175158"
    RUN_NAME_RAND = "omt-bright-rand-dot-1126-174909"
    
    main(run_name_cur=RUN_NAME_CUR, run_name_rand=RUN_NAME_RAND, num_trajs=NUM_TRAJS)


# %% [markdown]
# ## Bright Line Plots
# %%
ENV_NOVEL_NAME = MinigridEnvNames.LRoomLineGreen
get_env_img(env_novel_name=ENV_NOVEL_NAME)

# %%
if __name__ == "__main__":
    NUM_TRAJS=200
    RUN_NAME_CUR= "omt-cur-line-1210-190559" # "omt-cur-line-1210-162857"
    RUN_NAME_RAND = "omt-rand-line-1210-190526" # "omt-rand-line-1210-171543"
    
    main(run_name_cur=RUN_NAME_CUR, run_name_rand=RUN_NAME_RAND, num_trajs=NUM_TRAJS)
