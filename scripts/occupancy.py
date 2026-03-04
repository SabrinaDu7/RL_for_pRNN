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
# # Occupancy Heatmaps for wandb data

# %%
# !pwd

# %% [markdown]
# ## Imports and Function Definitions
# %%
import torch
from jaxtyping import Float
from typing import Literal
import numpy as np
from pathlib import Path
from scripts.wandb_data import (
    fetch_occupancy_grids,
    plot_occupancy_average,
    prob_roi,
    plot_prob_roi,
    OccupancyData,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

CACHE_DIR = Path("../outputs/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(agent_type: str, with_obs: bool, target_loc: list[int] | None, entropy_coef: float, ctrl: bool | None, project: str) -> Path:
    loc_str = f"{target_loc[0]}{target_loc[1]}" if target_loc else "none"
    ctrl_str = "none" if ctrl is None else str(ctrl)
    name = f"{project}__{agent_type}__obs{with_obs}__loc{loc_str}__ec{entropy_coef}__ctrl{ctrl_str}.pt"
    return CACHE_DIR / name


def save_occupancy_data(data: OccupancyData, path: Path, config: dict) -> None:
    torch.save({
        "grids": data.grids,
        "target_loc": data.target_loc,
        "run_names": data.run_names,
        "config_values": data.config_values,
        "config_keys": data.config_keys,
        "config": config,
    }, path)


def load_occupancy_data(path: Path) -> OccupancyData:
    d = torch.load(path, weights_only=False)
    return OccupancyData(grids=d["grids"], target_loc=d["target_loc"], run_names=d["run_names"], config_values=d["config_values"], config_keys=d["config_keys"])

# %% [markdown]
# ## Occupancy Heatmaps and Probability of ROI Occupancy
# %%
STEPS = [0, 8, 208, 408, 608, 808, 1008, 1208, 1408, 1608, 1808, 2008, 2208, 2408, 2608, 2808]

def fetch_data(agent_type: Literal["rand", "cur"], with_obs: bool, target_loc: list[int] | None = None, entropy_coef: float = 0.0, ctrl: bool | None = False, project: str = "curious-george-omt", use_cache: bool = True) -> OccupancyData:
    cache_path = _cache_key(agent_type, with_obs, target_loc, entropy_coef, ctrl, project)
    if use_cache and cache_path.exists():
        return load_occupancy_data(cache_path)
    data = fetch_occupancy_grids(
        entity="blake-richards",
        project=project,
        target_loc=target_loc,
        metric="Eval/OPA_Occupancy",
        group=f"omt-{agent_type}-dot",
        filters={"config.tasks.new_obj_loc": target_loc, # or None, [7, 11] or [14, 7] depending on which condition you want to filter for
                "config.rl.entropy_coef": entropy_coef,
                "config.exp.with_obs": with_obs,
                "config.tasks.control": ctrl},
        max_workers=10,
    )
    if use_cache:
        config = {
            "agent_type": agent_type,
            "with_obs": with_obs,
            "target_loc": target_loc,
            "entropy_coef": entropy_coef,
            "ctrl": ctrl,
            "project": project,
        }
        save_occupancy_data(data, cache_path, config)
    return data

def show_occ_heatmaps(agent_type: Literal["rand", "cur"], with_obs: bool, target_loc: list[int] | None = None, entropy_coef: float = 0.0, data: OccupancyData | None = None, ctrl: bool = False):
    obs_naming = "with_obs" if with_obs else "without_obs"
    ctrl_naming = "_ctrl" if ctrl else ""

    if data is None:
        data = fetch_data(agent_type=agent_type, with_obs=with_obs, target_loc=target_loc, entropy_coef=entropy_coef, ctrl=ctrl)
    
    target_loc = [7, 2] if target_loc is None else target_loc   
    avg = {step: np.nanmean(grids, axis=0) for step, grids in data.grids.items()}                                                                             
    fig = plot_occupancy_average(avg_grids=avg, label_steps=STEPS) 
    fig.write_image(f"../outputs/{obs_naming}/occupancy_{target_loc[0]}{target_loc[1]}_ec{entropy_coef}{ctrl_naming}.png")
    fig.show()                                                                                                                           

def show_prob_roi(
    data: list[OccupancyData],
    colors: list[str],
    labels: list[str],
    radius: int = 3,
    use_std: bool = True,
    save_path: str | None = None,
):  
    sorted_steps = [sorted(d.grids.keys()) for d in data]
    min_length = min([len(steps) for steps in sorted_steps])
    label_steps = STEPS[:min_length]

    prob_tensors = [
        prob_roi(d.grids, radius=radius, target_loc=d.target_loc, sorted_steps=sorted_steps[i])
        for i, d in enumerate(data)
    ]
    prob_tensors = [t[:min_length] for t in prob_tensors]
    fig, ax = plot_prob_roi(prob_tensors, label_steps, colors=colors, labels=labels, use_std=use_std, radius=radius)
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax

def main_single(agent: Literal["rand", "cur"], loc: list[int] | None, ec: float, r: int, data: OccupancyData | None = None, ctrl: bool | None = False, project: str = "curious-george-omt"):
    if data is None:
        data = fetch_data(agent_type=agent, with_obs=False, target_loc=loc, entropy_coef=ec, ctrl=ctrl, project=project)
    show_occ_heatmaps(agent_type=agent, with_obs=False, target_loc=loc, entropy_coef=ec, data=data)
    show_prob_roi(data=[data], colors=["blue"], labels=[agent], radius=r, use_std=True)

def main_multiple(target_loc: list[int], radius: int, use_std: bool = True):
    save_path = f"../outputs/without_obs/roi_plot{target_loc[0]}{target_loc[1]}_official.png"
    ec = 0

    data_rand = fetch_data(agent_type="rand", with_obs=False, target_loc=target_loc, entropy_coef=ec, ctrl=False, project="curious-george-ctrl")
    data_cur_exp = fetch_data(agent_type="cur", with_obs=False, target_loc=target_loc, entropy_coef=ec, ctrl=None, project="curious-george-omt")
    data_cur_ctrl = fetch_data(agent_type="cur", with_obs=False, target_loc=target_loc, entropy_coef=ec, ctrl=True, project="curious-george-ctrl")
    data_list = [data_cur_exp, data_cur_ctrl, data_rand]
    labels = ["Curious", "Curious CTRL", "Random"]

    show_prob_roi(data=data_list, colors=["green", "blue", "red"], labels=labels, radius=radius, save_path=save_path, use_std=use_std)

# %% [markdown]
# ## Novel Object
# %%
EC = 0
R = 3

# %%
# Probabilities Plot
if __name__ == "__main__":
    LOC = [7, 11]
    USE_STD = False
    main_multiple(target_loc=LOC, radius=R, use_std=USE_STD)

# %%
# CURIOUS
if __name__ == "__main__":
    AGENT= "cur"
    LOC = [7, 11]
    CTRL = None
    W_OBS = False
    PROJECT = "curious-george-omt"
    
    data = fetch_data(agent_type=AGENT, with_obs=W_OBS, target_loc=LOC, ctrl=CTRL, entropy_coef=EC, project=PROJECT)
    main_single(agent=AGENT, loc=LOC, ec=EC, r=R, ctrl=CTRL, project=PROJECT, data=data)

# %%
# CURIOUS CTRL
if __name__ == "__main__":
    AGENT= "cur"
    LOC = [7, 11]
    CTRL = True
    W_OBS = False
    PROJECT = "curious-george-ctrl"
    
    data = fetch_data(agent_type=AGENT, with_obs=W_OBS, target_loc=LOC, ctrl=CTRL, entropy_coef=EC, project=PROJECT)
    main_single(agent=AGENT, loc=LOC, ec=EC, r=R, ctrl=CTRL, project=PROJECT, data=data)

# %%
# RANDOM
if __name__ == "__main__":
    AGENT = "rand"
    W_OBS = False
    LOC = [7, 11]
    CTRL = False
    PROJECT = "curious-george-ctrl"

    data = fetch_data(agent_type=AGENT, with_obs=W_OBS, target_loc=LOC, ctrl=CTRL, entropy_coef=EC, project=PROJECT)
    main_single(agent=AGENT, loc=LOC, ec=EC, r=R, ctrl=CTRL, project=PROJECT, data=data)
