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

# %%
# !pwd

# %% [markdown]
# ## Imports and Function Definitions
# %%
from typing import Literal
import numpy as np
from scripts.wandb_data import (
    fetch_occupancy_grids,
    plot_occupancy_average,
    prob_roi,
    plot_prob_roi,
    OccupancyData,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# %% [markdown]
# ## Occupancy Heatmaps and Probability of ROI Occupancy
# %%
STEPS = [0, 8, 208, 408, 608, 808, 1008, 1208, 1408, 1608, 1808, 2008, 2208, 2408, 2608, 2808]

def fetch_data(agent_type: Literal["rand", "cur"], with_obs: bool, target_loc: list[int] | None = None, entropy_coef: float = 0.0) -> OccupancyData:
    data = fetch_occupancy_grids(                                                                                                                                
        entity="blake-richards",                                                                                                                                 
        project="curious-george-omt",                                                                                                                            
        metric="Eval/OPA_Occupancy",
        group=f"omt-{agent_type}-dot",  
        filters={"config.tasks.new_obj_loc": target_loc, # or None, [7, 11] or [14, 7] depending on which condition you want to filter for      
                "config.rl.entropy_coef": entropy_coef,
                "config.exp.with_obs": with_obs},                                                                                                                                                   
    )
    return data

def show_occ_heatmaps(agent_type: Literal["rand", "cur"], with_obs: bool, target_loc: list[int] | None = None, entropy_coef: float = 0.0, data: OccupancyData | None = None):
    obs_naming = "with_obs" if with_obs else "without_obs"
    if data is None:
        data = fetch_data(agent_type=agent_type, with_obs=with_obs, target_loc=target_loc, entropy_coef=entropy_coef)
    
    target_loc = [7, 2] if target_loc is None else target_loc   
    avg = {step: np.nanmean(grids, axis=0) for step, grids in data.grids.items()}                                                                             
    fig = plot_occupancy_average(avg_grids=avg, label_steps=STEPS) 
    fig.write_image(f"../outputs/{obs_naming}/occupancy_{target_loc[0]}{target_loc[1]}_ec{entropy_coef}.png")
    fig.show()                                                                                                                           

def show_prob_roi(agent_type: Literal["rand", "cur"], with_obs: bool, target_loc: list[int] | None = None, entropy_coef: float = 0.0, radius: int = 3, data: OccupancyData | None = None):
    obs_naming = "with_obs" if with_obs else "without_obs"
    if data is None:
        data = fetch_data(agent_type=agent_type, with_obs=with_obs, target_loc=target_loc, entropy_coef=entropy_coef)

    sorted_steps = sorted(data.grids.keys())
    res = prob_roi(data.grids, radius=radius, target_loc=target_loc, sorted_steps=sorted_steps)

    target_loc = [7, 2] if target_loc is None else target_loc   
    fig, ax = plot_prob_roi(res, STEPS,) # prob_tensor_rand=res_rand)
    fig.savefig(f"../outputs/{obs_naming}/roi_plot{target_loc[0]}{target_loc[1]}_ec{entropy_coef}_r{radius}.png")    
    fig.show()

def main(agent: Literal["rand", "cur"], loc: list[int] | None, ec: float, r: int, data: OccupancyData | None = None):
    show_occ_heatmaps(agent_type=agent, with_obs=False, target_loc=loc, entropy_coef=ec, data=data)
    show_prob_roi(agent_type=agent, with_obs=False, target_loc=loc, entropy_coef=ec, radius=r, data=data)

# %% [markdown]
# ## Novel Object
# %%
EC = 0.0
R = 3

# %%
# CURIOUS
if __name__ == "__main__":
    AGENT= "cur"
    W_OBS = False
    for loc in [[7, 11], [7, 2], [14, 7]]:
        data = fetch_data(agent_type=AGENT, with_obs=W_OBS, target_loc=loc, entropy_coef=EC)
        main(agent=AGENT, loc=loc, ec=EC, r=R, data=data)

# %%
# RANDOM
if __name__ == "__main__":
    AGEN = "rand"
    W_OBS = True

    for loc in [[7, 11], [7, 2], [14, 7]]:
        data = fetch_data(agent_type=AGENT, with_obs=W_OBS, target_loc=loc, entropy_coef=EC)
        main(agent=AGENT, loc=loc, ec=EC, r=R, data=data)
