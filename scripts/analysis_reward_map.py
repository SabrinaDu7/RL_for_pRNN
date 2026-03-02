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
# # Reward Map Analysis for OMT

# %% [markdown]
# ## Imports and Function Definitions
# %%
"""Compute per-position, per-head-direction reward maps from pRNN prediction error.

Reward at each (hd, x, y) cell is the mean absolute prediction error (MAE)
of the pRNN, averaged over all visits to that cell.  Four collection passes
(one per head direction) ensure every walkable (hd, x, y) cell is visited.
"""

from __future__ import annotations

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from prnn.utils import (
    ActionEncodingsEnum,
    MinigridEnvNames,
)
from RLutils import (
    make_env,
    get_pN,
    get_SR_acmodel,
    get_obss_preprocessor,
    get_agent,
    plot_heatmaps,
    seed,
    DEVICE,
)
from utils import get_ckpt_env_vars, AgentInputType, AgentType

from scripts.analysis_OMT import (
    collect_eval_trajectories,
    get_walkable_mask,
    EvalTrajectoryConfig,
    EvalTrajectories,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# %% [markdown]
# ## Reward Map Computation

# %%
T_STEPS = 256
PRED_OFFSET = 0


def accumulate_reward_maps(
    result: EvalTrajectories,
    env_width: int,
    env_height: int,
    pred_offset: int = PRED_OFFSET,
    reward_sums: torch.Tensor | None = None,
    visit_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate prediction-error rewards into per-(hd, x, y) maps.

    Can be called repeatedly to merge data from multiple collection rounds
    (e.g. fill-in passes) into the same running totals.

    Args:
        result: Output of ``collect_eval_trajectories``.
        env_width: Full grid width (including walls).
        env_height: Full grid height (including walls).
        pred_offset: Number of initial timesteps to skip per trajectory
            (early pRNN predictions are unreliably high).
        reward_sums: Running reward totals to accumulate into, shape ``[4, W, H]``.
            Created as zeros when *None*.
        visit_counts: Running visit counts, shape ``[4, W, H]``.
            Created as zeros when *None*.

    Returns:
        ``(reward_sums, visit_counts)`` — both shape ``[4, W, H]``.
    """
    W = env_width - 2
    H = env_height - 2

    if reward_sums is None:
        reward_sums = torch.zeros(4, W, H)
    if visit_counts is None:
        visit_counts = torch.zeros(4, W, H)

    # Per-timestep MAE across features: (B, T)
    rewards = (result["obs_pred"] - result["obs_next"]).abs().mean(dim=-1)

    B, T = rewards.shape
    if pred_offset >= T:
        return reward_sums, visit_counts

    # Slice away burn-in timesteps
    rewards = rewards[:, pred_offset:]  # (B, T')
    T_prime = rewards.shape[1]

    # Extract positions and directions for valid timesteps
    # State at time t (where the prediction was made)
    all_hd = torch.stack(
        [
            torch.from_numpy(s["agent_dir"][pred_offset : pred_offset + T_prime].astype(np.int64))
            for s in result["states"] # each element in s is the state for ONE traj
        ]
    )  # (B, T')
    
    # Turn minigrid positions into 0-indexed coordinates in the reward map (which excludes walls)
    all_x = torch.stack(
        [
            torch.from_numpy(
                s["agent_pos"][pred_offset : pred_offset + T_prime, 0].astype(np.int64) - 1
            )
            for s in result["states"]
        ]
    )  # (B, T')
    all_y = torch.stack(
        [
            torch.from_numpy(
                s["agent_pos"][pred_offset : pred_offset + T_prime, 1].astype(np.int64) - 1
            )
            for s in result["states"]
        ]
    )  # (B, T')

    # Flat index into [4, W, H] for scatter_add_
    flat_idx = (all_hd * (W * H) + all_x * H + all_y).flatten()

    reward_sums_flat = reward_sums.flatten()
    print(f"{all_hd=}")
    print(f"{all_x=}")
    print(f"{all_y=}")
    print(rewards.flatten())
    reward_sums_flat.scatter_add_(0, flat_idx, rewards.flatten())

    visit_counts_flat = visit_counts.flatten()
    visit_counts_flat.scatter_add_(0, flat_idx, torch.ones_like(rewards).flatten())

    return reward_sums.view(4, W, H), visit_counts.view(4, W, H)


def compute_reward_maps(
    reward_sums: torch.Tensor,
    visit_counts: torch.Tensor,
    walkable_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize accumulated rewards by visit count.

    Args:
        reward_sums: Shape ``[4, W, H]``.
        visit_counts: Shape ``[4, W, H]``.
        walkable_mask: Optional boolean mask of shape ``[W, H]``.
            Wall cells (``False``) are set to ``NaN`` in the output.

    Returns:
        Reward map of shape ``[4, W, H]``. Unvisited and wall cells are ``NaN``.
    """
    reward_maps = reward_sums / visit_counts.clamp(min=1)

    visited = visit_counts > 0
    if walkable_mask is not None:
        visited = visited & walkable_mask.unsqueeze(0)

    visited_vals = reward_maps[visited]
    if visited_vals.numel() > 0:
        rmin, rmax = visited_vals.min(), visited_vals.max()
        if rmax > rmin:
            reward_maps = (reward_maps - rmin) / (rmax - rmin)

    reward_maps[visit_counts == 0] = float("nan")
    if walkable_mask is not None:
        reward_maps[:, ~walkable_mask] = float("nan")

    return reward_maps


def run_reward_map_analysis(args: DictConfig, new_obj_pos: list[int], step: int, withObs: bool = True):
    no_obs_str = "-noObs" if not withObs else ""
    prnn_ckpt = f"/home/sabrina/Documents/experiments/RL_for_pRNN/omt-cur-dot{no_obs_str}-goal{new_obj_pos[0]}{new_obj_pos[1]}/{step}/pN-{step}.pt"
    acmodel_status_ckpt = f"/home/sabrina/Documents/experiments/RL_for_pRNN/omt-cur-dot{no_obs_str}-goal{new_obj_pos[0]}{new_obj_pos[1]}/{step}/status.pt"

    print(f"Loading pRNN checkpoint from {prnn_ckpt}")

    env = make_env(
        env_key=MinigridEnvNames.LRoom,
        input_type=AgentInputType.H_PO if withObs else AgentInputType.H,
        act_enc=ActionEncodingsEnum.SpeedHD,
        new_obj_pos=new_obj_pos,
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
        device=DEVICE,
    )

    # --- 1. Collect trajectories (all positions × 4 directions) ---
    config = EvalTrajectoryConfig(
        timesteps=T_STEPS,
        include_render=False,
        include_hidden_states=False,
    )
    result = collect_eval_trajectories(pN, agent, env, config)
    reward_sums, visit_counts = accumulate_reward_maps(
        result, env.width, env.height, PRED_OFFSET,
    )

    # --- 2. Normalize and visualise ---
    assert reward_sums is not None and visit_counts is not None
    walkable = get_walkable_mask(env)
    reward_maps = compute_reward_maps(reward_sums, visit_counts, walkable)
    fig = plot_heatmaps(reward_maps.detach().numpy(), title="Reward Map (MAE)")
    
    # Save the figure
    save_path = f"outputs/reward_map_loc{new_obj_pos[0]}{new_obj_pos[1]}_s{step}.png"
    fig.write_image(save_path)
    print(f"Saved reward map figure to {save_path}")


@hydra.main(config_path="../Configs", config_name="Conf1_Adel", version_base=None)
def main(args: DictConfig):
    seed(35)
    STEPS = [0, 8, 208, 408, 608, 808, 1008, 1208, 1408, 1608, 1808, 2008, 2208, 2408, 2608, 2808] 
    new_obj_loc = args.tasks.new_obj_loc
    withObs = args.exp.with_obs
    
    for step in STEPS:
        run_reward_map_analysis(args, new_obj_pos=new_obj_loc, step=step, withObs=withObs)


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    end_time = time.time()
    print(f"--- Script finished in {end_time - start_time:.2f} seconds ---")
