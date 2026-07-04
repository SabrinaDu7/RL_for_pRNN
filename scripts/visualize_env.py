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
# # Visualize Minigrid Environment

# %%

# %%
# !pwd

# %% [markdown]
# ## Imports and Function Definitions
# %%
from matplotlib import pyplot as plt

from curious_george import make_env
from utils import AgentInputType

from prnn.utils import (
    MinigridEnvNames,
    ActionEncodingsEnum,
)

from scripts.analysis_OMT import get_walkable_mask

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# %%
from curious_george import get_subroom_id
import torch

def get_env_img(env_key: MinigridEnvNames):
    size = 16
    env = make_env(
        env_key=env_key, 
        input_type=AgentInputType.H_PO.value, 
        act_enc=ActionEncodingsEnum.SpeedHD.value, 
        agent_start_pos=(10, 8),
        agent_start_dir=0,
        subroom_size=8,
        open_all_paths=False,
        size=size,
        # Lwidth=8, 
        # Lheight=6, 
        seed=777,
        door_poss=[1, 2, 3, 4,],
        # goal_pos=[7, 2],
        new_obj_pos=[14, 7]
    )
    mask = get_walkable_mask(env)
    print(mask)
    # Print the indices of NON walkable cells

    # print(type(env.env.unwrapped.subroom_size))
    # id = get_subroom_id(torch.tensor(env.get_agent_pos()).unsqueeze(0), env.env.unwrapped.subroom_size).item()
    # print(f"Subroom ID: {id}")
    print(env.action_space)
    print(env.get_goal_loc())
    print(env.get_new_obj_pos())
    print(env.get_agent_pos())
    env.step(2)
    print(env.get_agent_pos())
    env.reset()
    print(env.get_agent_pos())
    return env.render(mode="human")

# %% [markdown]
# ## Display Environment

# %%
if __name__ == "__main__":
    env_img = get_env_img(env_key=MinigridEnvNames.LRoom)
    fig = plt.figure()
    plt.imshow(env_img)
    plt.show()

# %% [markdown]
# ## Inspect Trajectories in Env

# %%
"""Quick inspection plot for obs, obs_pred, and obs_next from collect_eval_trajectories."""

import numpy as np
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from prnn.utils import MinigridEnvNames, ActionEncodingsEnum
from curious_george import make_env, get_pN, get_agent, get_SR_acmodel, get_obss_preprocessor, seed, DEVICE
from utils import get_ckpt_env_vars, AgentInputType, AgentType

from scripts.analysis_OMT import collect_eval_trajectories, EvalTrajectoryConfig, EvalTrajectories


# %%
def plot_obs_grid(
    result: EvalTrajectories,
    env,
    traj_idx: int = 0,
    timesteps: list[int] | None = None,
    save_path: str = "inspect_trajectories.png",
):
    """Plot obs, obs_pred, and obs_next side by side for selected timesteps.

    Rows: obs (current), obs_next (ground truth next), obs_pred (predicted next)
    Columns: one per timestep

    Args:
        result: Output from collect_eval_trajectories.
        env: The environment shell (for obs_shape).
        traj_idx: Which trajectory to plot.
        timesteps: Which timesteps to show (default: first 4).
        save_path: Where to save the figure.
    """
    _, view_size, C = env.obs_shape
    obs_shape = (view_size, view_size, C)

    if timesteps is None:
        timesteps = [0, 1, 2, 3]

    n_ts = len(timesteps)
    fig, axes = plt.subplots(3, n_ts, figsize=(3 * n_ts, 9))

    row_labels = ["obs[t+1]\n(actual next)", "obs_next[t]\n(ground truth)", "obs_pred[t]\n(pRNN prediction)"]

    for col, t in enumerate(timesteps):
        # obs[t+1]: shape (B, T+1, X) — the actual next observation
        obs_img = result["obs"][traj_idx, t].detach().numpy().reshape(obs_shape)

        # obs_next[t]: shape (B, T, X) — ground truth next obs (should equal obs[t+1])
        obs_next_img = result["obs_next"][traj_idx, t].detach().numpy().reshape(obs_shape)

        # obs_pred[t]: shape (B, T, X) — predicted next obs (should approximate obs[t+1])
        obs_pred_img = result["obs_pred"][traj_idx, t].detach().numpy().reshape(obs_shape)

        for row, img in enumerate([obs_img, obs_next_img, obs_pred_img]):
            ax = axes[row, col]
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(f"t={t}", fontsize=10)
            ax.axis("off")

            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=11)

    fig.suptitle(f"Trajectory {traj_idx}: obs vs predictions", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.show()


# %tb
@hydra.main(config_path="../Configs", config_name="Conf1_Adel")
def main(args: DictConfig):
    seed(55)
    prnn_ckpt, acmodel_status_ckpt = get_ckpt_env_vars(AgentType.AC)
    env = make_env(
        env_key=MinigridEnvNames.LRoom,
        input_type=AgentInputType.H_PO,
        act_enc=ActionEncodingsEnum.SpeedHD,
    )
    obs_space, preprocess_obss = get_obss_preprocessor(env.observation_space)

    pN = get_pN(args=args, env=env, device=DEVICE, pRNN_ckpt=prnn_ckpt)
    ac_model = get_SR_acmodel(
        args,
        env_act_space=env.action_space,
        obs_space=obs_space,
        acmodel_status_ckpt=acmodel_status_ckpt,
        device=DEVICE,
    )

    agent = get_agent(
        env=env, 
        agent_Type=AgentType.AC, 
        ac_model=ac_model, 
        prnn=pN, 
        device=DEVICE
    )

    config = EvalTrajectoryConfig(timesteps=20)
    result = collect_eval_trajectories(pN, agent, env, config)

    plot_obs_grid(result, env, traj_idx=0, timesteps=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])


if __name__ == "__main__":
    import sys
    sys.argv = sys.argv[:1]  # Strip Jupyter kernel args that confuse Hydra
    main()
