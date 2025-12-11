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

from RLutils import make_env
from utils import AgentInputType

from prnn.utils import (
    MinigridEnvNames,
    ActionEncodingsEnum,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# %%
def get_env_img(env_key: MinigridEnvNames):
    env = make_env(
        env_key=env_key, 
        input_type=AgentInputType.H_PO.value, 
        act_enc=ActionEncodingsEnum.SpeedHD.value, 
        agent_start_pos=(4, 1),
        agent_start_dir=2,
        subroom_size=8,
        open_all_paths=False,
        size=16,
        Lwidth=8, 
        Lheight=6, 
        # goal_pos=[7, 2],
        # new_obj_pos=[2, 2]
    )
    
    print(env.action_space)
    print(env.get_goal_loc())
    print(env.get_new_obj_pos())
    print(env.get_agent_pos())
    return env.render(mode=None)

# %% [markdown]
# ## Display Environment

# %%
if __name__ == "__main__":
    env_img = get_env_img(env_key=MinigridEnvNames.LRoomLineGreen)
    fig = plt.figure()
    plt.imshow(env_img)
    plt.show()
