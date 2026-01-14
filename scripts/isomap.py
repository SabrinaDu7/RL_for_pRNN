#!/usr/bin/env python3
"""
Script to generate Isomap visualization with random agent and predictive network.
Based on tasks/ObjectMemoryTask/run_task.py
"""

from pathlib import Path
import torch
import numpy as np
import hydra
from omegaconf import DictConfig

from prnn.utils.figures import IsoMapFigure
from prnn.utils import MinigridEnvNames, ActionEncodingsEnum, AgentInputType
from utils import get_ckpt_env_vars, AgentType
from RLutils import make_env, get_pN, get_agent, get_SR_acmodel, get_obss_preprocessor

# ===== Constants =====
DEVICE = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_isomap(
        args: DictConfig,
        agent_type: AgentType,
        prnn_name: str,
        save_path: Path | None = None,
        timesteps_wake: int = 5000,
        timesteps_sleep: int = 1000,
        usecells: np.ndarray | None = None,):
    """
    Generate Isomap visualization with random agent.
    """

    print(f"Using device: {DEVICE}")

    # Get checkpoint paths
    prnn_cur_ckpt, acmodel_ckpt = get_ckpt_env_vars(agent_type=AgentType.AC)
    prnn_rand_ckpt, _ = get_ckpt_env_vars(agent_type=AgentType.RANDOM)
    prnn_ckpt = prnn_cur_ckpt if prnn_name == "cur" else prnn_rand_ckpt

    # Create environment (standard LRoom without novel objects)
    print("Creating environment...")
    env = make_env(
        env_key=MinigridEnvNames.LRoomLineGreen, 
        new_obj_pos=[7, 2], 
        input_type=AgentInputType.H_PO.value, 
        act_enc=ActionEncodingsEnum.SpeedHD.value)

    # Load predictive network
    print("Loading predictive network...")
    predictiveNet = get_pN(args=args, env=env, device=DEVICE, pRNN_ckpt=prnn_ckpt)
    predictiveNet.pRNN.to(DEVICE)
    predictiveNet.pRNN.eval()

    # Create random agent
    print(f"Creating {agent_type} agent...")
    obs_space, preprocess_obss = get_obss_preprocessor(
        env.observation_space
    )
    ac_model = get_SR_acmodel(
        args,
        env_act_space=env.action_space,
        obs_space=obs_space,
        acmodel_status_ckpt=acmodel_ckpt,
        device=torch.device(DEVICE),
    )
    agent = get_agent(
        env=env, 
        agent_Type=agent_type, 
        prnn=predictiveNet, 
        ac_model=ac_model,
        device=torch.device(DEVICE)
    )
    if agent_type == AgentType.AC:
        agent.acmodel.eval()  # type: ignore[attr-defined]
        agent.argmax = True # type: ignore

    # Generate Isomap figure
    print(f"Generating Isomap with {timesteps_wake} wake steps and {timesteps_sleep} sleep steps...")
    savename = str(save_path.stem) if save_path else None
    savefolder = str(save_path.parent) if save_path else None

    IsoMapFigure(
        predictiveNet=predictiveNet,
        env=env,
        agent=agent,
        noisemag=0,
        noisestd=0.25,
        timesteps_wake=timesteps_wake,
        timesteps_sleep=timesteps_sleep,
        savename=savename,
        savefolder=savefolder,
        usecells=usecells,
    )

    print("Done!")
    if save_path:
        print(f"Figure saved to {save_path}")
    else:
        print("Displaying figure...")
        import matplotlib.pyplot as plt
        plt.show()


@hydra.main(config_path="../Configs", config_name="Conf1_Adel")
def main(args: DictConfig):
    agent_type = AgentType.AC
    prnn_name = "rand"
    agent_name = "cur" if agent_type == AgentType.AC else "rand"

    plot_isomap(
        args,
        agent_type=agent_type,
        prnn_name=prnn_name,
        save_path=Path(f"isomap_{prnn_name}_prnn_{agent_name}_agent.png"),
        timesteps_wake=5000,
        timesteps_sleep=1000,
    )

if __name__ == "__main__":
    main()
