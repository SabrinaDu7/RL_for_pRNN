#!/usr/bin/env python3
"""PCA visualization of pRNN hidden states, colored by novel object in view."""

from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from sklearn.decomposition import PCA

from prnn.utils.figures import saveFig
from prnn.utils import MinigridEnvNames, ActionEncodingsEnum, AgentInputType
from curious_george import get_ckpt_env_vars, AgentType
from curious_george import make_env, get_pN, get_agent, get_SR_acmodel, get_obss_preprocessor
from scripts.analysis_OMT import load_eval_trajectories, EvalTrajectoryConfig
from tasks.omt.metrics import get_view_coords_batch

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

DEVICE = "cpu"


def get_ckpts(agent_type: AgentType, env_type: MinigridEnvNames, omt: bool = False, step: int = 0, obj_pos: list[int] = [7, 11]) -> tuple[str, str | None]:
    if omt:
        prnn_ckpt = f"omt-cur-dot-noObs-goal{obj_pos[0]}{obj_pos[1]}/{step}/pN-{step}.pt"
        acmodel_ckpt = f"omt-cur-dot-noObs-goal{obj_pos[0]}{obj_pos[1]}/{step}/status.pt"
    elif agent_type == AgentType.AC:
        prnn_ckpt, acmodel_ckpt = get_ckpt_env_vars(agent_type=AgentType.AC, env_type=env_type)
    else:
        prnn_ckpt, _ = get_ckpt_env_vars(agent_type=AgentType.RANDOM, env_type=env_type)
        acmodel_ckpt = None
    return prnn_ckpt, acmodel_ckpt


def setup(args: DictConfig, agent_type: AgentType, env_type: MinigridEnvNames, new_obj_pos: list[int] | None = None, omt: bool = False, step: int = 0):
    prnn_ckpt, acmodel_ckpt = get_ckpts(agent_type=agent_type, env_type=env_type, omt=omt, step=step, obj_pos=new_obj_pos if new_obj_pos is not None else [7, 11])
    print(f"Using pRNN checkpoint: {prnn_ckpt}")

    env = make_env(
        env_key=env_type,
        new_obj_pos=new_obj_pos,
        input_type=AgentInputType.H_PO.value,
        act_enc=ActionEncodingsEnum.SpeedHD.value)

    predictiveNet = get_pN(args=args, env=env, device=DEVICE, pRNN_ckpt=prnn_ckpt)
    predictiveNet.pRNN.to(DEVICE)
    predictiveNet.pRNN.eval()

    obs_space, preprocess_obss = get_obss_preprocessor(env.observation_space)
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
    return predictiveNet, ac_model, agent, env


def create_pca_plot(
    savename: str,
    savefolder: str,
    step: int,
    goal: list[int],
    timesteps: int = 256,
    new_obj_pos: list[int] | None = None,
) -> None:
    """Run 2D PCA on hidden states and color by whether the novel object is in view."""
    config = EvalTrajectoryConfig(
        timesteps=timesteps,
        include_hidden_states=True,
        save_path=Path(savefolder) / f"{savename}",
    )

    # TODO: Change this to collect new trajectories if needed, currently just loads existing ones
    print(f"Loading existing trajectories from {savefolder}/{savename}")
    trajectories = load_eval_trajectories(Path(savefolder) / f"data_cur_lroom_step{step}_goal{goal[0]}{goal[1]}" / "trajectories.pt")

    assert trajectories["hidden_states"] is not None
    hidden = trajectories["hidden_states"]          # (B, T', hidden)
    B, T_prime, hidden_size = hidden.shape
    h = hidden.reshape(B * T_prime, hidden_size).detach().numpy()

    states = trajectories["states"]
    pos = np.concatenate([s["agent_pos"][:T_prime] for s in states], axis=0)  # (N, 2)
    dir_ = np.concatenate([s["agent_dir"][:T_prime] for s in states], axis=0)  # (N,)

    print(f"{pos.shape=}, {dir_.shape=}, {h.shape=}")

    if new_obj_pos is not None:
        i_obj, j_obj = new_obj_pos
        vx, vy = get_view_coords_batch(i_obj, j_obj, pos, dir_)
        in_view = ((vx >= 0) & (vx < 7) & (vy >= 0) & (vy < 7)).astype(int)
        print(f"{pos[in_view == 1][:10]=}")
        print(f"{dir_[in_view == 1][:10]=}")
    else:
        in_view = np.zeros(len(h), dtype=int)

    print("Running PCA...")
    Y = PCA(n_components=2).fit_transform(h)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Adjusted for 3 subplots

    axes[0].scatter(Y[:, 0], Y[:, 1], c=in_view, marker=".", s=5, cmap="plasma")
    axes[0].set_title("PCA — novel obj in view")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].scatter(pos[:, 0], pos[:, 1], c=in_view, marker=".", s=5, cmap="plasma")
    axes[1].set_title("Agent positions — novel obj in view")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    if new_obj_pos is not None:
        axes[1].scatter([new_obj_pos[0]], [new_obj_pos[1]], c="red", marker="*", s=100, label="novel obj")
        axes[1].legend(fontsize=8)

    plt.tight_layout()
    saveFig(fig, savename + "_PCA_h", savefolder, filetype="png")


@hydra.main(config_path="../Configs", config_name="Conf1_Adel")
def main(args: DictConfig):
    agent_type = AgentType.AC
    agent_name = "cur" if agent_type == AgentType.AC else "rand"
    env_type = MinigridEnvNames.LRoom
    env_name = "fourroom" if env_type == MinigridEnvNames.FourRoomsObjs else "lroom"
    omt = True
    new_obj_pos = [14, 7]
    step = 1608

    predictiveNet, ac_model, agent, env = setup(
        args=args,
        agent_type=agent_type,
        env_type=env_type,
        new_obj_pos=new_obj_pos,
        omt=omt,
        step=step,
    )

    create_pca_plot(
        step=step,
        goal=new_obj_pos,
        timesteps=256,
        savename=f"pca_h_{agent_name}_agent_{env_name}_step{step}",
        savefolder="outputs",
        new_obj_pos=new_obj_pos,
    )


if __name__ == "__main__":
    main()
