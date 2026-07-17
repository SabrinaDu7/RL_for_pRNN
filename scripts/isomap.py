#!/usr/bin/env python3
"""
Script to generate Isomap visualization with random agent and predictive network.
Based on tasks/omt/main_task.py
"""

from pathlib import Path
from typing import Union
import torch
import numpy as np
import matplotlib.pyplot as plt
import dataclasses
import tyro
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from sklearn import manifold

from prnn.utils.figures import saveFig
from prnn.utils import MinigridEnvNames, ActionEncodingsEnum, AgentInputType, PredictiveNet, RandomActionAgent
from prnn.utils.Shell import FaramaMinigridShell
from curious_george import get_ckpt_env_vars, AgentType
from curious_george import make_env, get_pN, get_agent, get_SR_acmodel, get_obss_preprocessor, ActorCriticAgent
from scripts.analysis_OMT import collect_eval_trajectories, load_eval_trajectories, EvalTrajectoryConfig, LabelFn
from tasks.omt.metrics import get_view_coords_batch, State

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ===== Constants =====
DEVICE = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_novel_obj_in_view_label_fn(obj_pos: list[int], agent_view_size: int = 7) -> LabelFn:
    """Return a label function that marks timesteps where the novel object is in the agent's view.

    Args:
        obj_pos: ``[i, j]`` world coordinates of the novel object.
        agent_view_size: Size of the agent's view window (default 7).

    Returns:
        A ``LabelFn(states, T) -> Tensor[B, T]`` with values 0 (not in view) or 1 (in view).
    """
    i_obj, j_obj = obj_pos

    def label_fn(states: list[State], T: int) -> torch.Tensor:
        B = len(states)
        labels = torch.zeros((B, T), dtype=torch.int64)
        for b, state in enumerate(states):
            pos = state["agent_pos"][:T]   # (T, 2)
            hd  = state["agent_dir"][:T]   # (T,)
            vx, vy = get_view_coords_batch(i_obj, j_obj, pos, hd, agent_view_size)
            in_view = (
                (vx >= 0) & (vx < agent_view_size) &
                (vy >= 0) & (vy < agent_view_size)
            )
            labels[b] = torch.from_numpy(in_view.astype(np.int64))
        return labels

    return label_fn


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
    # Get checkpoint paths
    prnn_ckpt, acmodel_ckpt = get_ckpts(agent_type=agent_type, env_type=env_type, omt=omt, step=step, obj_pos=new_obj_pos if new_obj_pos is not None else [7, 11])
    print(f"Using pRNN checkpoint: {prnn_ckpt}")

    # Create environment (standard LRoom with novel objects)
    print("Creating environment...")
    env = make_env(
        env_key=env_type, 
        new_obj_pos=new_obj_pos, 
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

    return predictiveNet, ac_model, agent, env


def create_isomap(
    pN: PredictiveNet,
    agent: Union[ActorCriticAgent, RandomActionAgent],
    env: FaramaMinigridShell,
    savename: str,
    savefolder: str,
    noisemag: float = 0,
    noisestd: float = 0.25,
    timesteps_wake: int = 256,
    timesteps_sleep: int = 1000,
    max_wake_points: int = 5000,
    usecells: np.ndarray | None = None,
    new_obj_pos: list[int] | None = None,
    label_fn: LabelFn | None = None,
) -> None:
    """Generate Isomap visualization using collect_eval_trajectories for wake data.

    Collects wake trajectories from every (walkable_position, head_direction) pair
    via collect_eval_trajectories, then uses pN.spontaneous for the sleep phase.
    Produces the same 6-subplot figure as IsoMapFigure.

    Note: ``timesteps_wake`` is the number of steps *per trajectory*, not the
    total. Spatial coverage is achieved through B = num_walkable × 4 trajectories,
    so a small value (e.g. 256) is sufficient and keeps collection fast.
    """
    # --- Wake phase ---
    # Check if trajectories exist
    config = EvalTrajectoryConfig(
        timesteps=timesteps_wake,
        include_hidden_states=True,
        save_path=Path(savefolder) / f"{savename}",
    )

    if (Path(savefolder) / f"{savename}").exists():
        print(f"Loading existing trajectories from {savefolder}/{savename}")
        trajectories = load_eval_trajectories(config.save_path / "trajectories.pt")
    else:
        print(f"Creating new trajectories and saving to {savefolder}/{savename}")
        trajectories = collect_eval_trajectories(pN, agent, env, config, label_fn=label_fn)

    assert trajectories["hidden_states"] is not None
    hidden = trajectories["hidden_states"]          # (B, T', hidden)
    B, T_prime, hidden_size = hidden.shape
    h_wake = hidden.reshape(B * T_prime, hidden_size).detach().numpy()

    states = trajectories["states"]
    act = trajectories["act"]                       # (B, T, A)
    whichact = np.argmax(act.detach().numpy().reshape(-1, act.shape[-1]), axis=1)  # (B*T,)

    # Align pos/dir to T' hidden-state timesteps per trajectory
    pos_wake = np.concatenate(
        [s["agent_pos"][:T_prime] for s in states], axis=0
    )  # (B*T', 2)
    dir_wake = np.concatenate(
        [s["agent_dir"][:T_prime] for s in states], axis=0
    )  # (B*T',)

    # Flatten labels (B, T') → (B*T',) if present
    raw_labels = trajectories.get("labels")
    if raw_labels is not None:
        in_view_wake_flat: np.ndarray | None = raw_labels[:, :T_prime].reshape(-1).numpy()
    else:
        in_view_wake_flat = None

    # Subsample wake points to keep Isomap tractable (O(N^3) complexity)
    N_wake_full = len(h_wake)
    if N_wake_full > max_wake_points:
        idx = np.random.choice(N_wake_full, max_wake_points, replace=False)
        idx.sort()
        h_wake = h_wake[idx]
        pos_wake = pos_wake[idx]
        dir_wake = dir_wake[idx]
        whichact = whichact[idx]
        if in_view_wake_flat is not None:
            in_view_wake_flat = in_view_wake_flat[idx]
        print(f"Subsampled wake points: {N_wake_full} → {max_wake_points}")

    in_view_wake = in_view_wake_flat

    inHD = dir_wake[:-1]
    nextHD = dir_wake[1:]

    # --- Sleep phase ---
    _, h_t, _ = pN.spontaneous(timesteps_sleep, noisemag, noisestd)
    h_sleep = np.squeeze(h_t.detach().numpy())      # (timesteps_sleep, hidden)

    # --- Isomap fit ---
    n_neighbors = 50
    n_components = 3
    X = np.concatenate((h_wake, h_sleep))
    method = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, p=1)

    numcells = X.shape[1]
    if usecells is not None:
        X = X[:, usecells]
    print(f"Using {X.shape[1]} of {numcells} cells")

    Y = method.fit_transform(X)

    # --- Plotting ---
    N_wake = len(h_wake)
    color = np.arctan(pos_wake[:, 0] / pos_wake[:, 1])
    SWcolor = np.concatenate((np.ones(N_wake), np.zeros(timesteps_sleep)))

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(3, 3, 1, projection="3d")
    ax.scatter(Y[:N_wake, 0], Y[:N_wake, 1], Y[:N_wake, 2], c=color, marker=".", s=5, cmap="plasma")
    ax.set_title("Position")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticklabels([])

    plt.subplot(4, 4, 3)
    plt.scatter(pos_wake[:, 0], pos_wake[:, 1], c=color, marker=".", s=5, cmap="plasma")
    plt.xticks([])
    plt.yticks([])

    ax = fig.add_subplot(3, 3, 7, projection="3d")
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=SWcolor, marker=".", s=5, cmap="plasma")
    ax.set_title("Sleep-Wake")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticklabels([])

    ax = fig.add_subplot(3, 3, 4, projection="3d")
    ax.scatter(Y[:N_wake, 0], Y[:N_wake, 1], Y[:N_wake, 2], c=whichact[:N_wake], marker=".", s=5, cmap="plasma")
    ax.set_title("Action")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticklabels([])

    ax = fig.add_subplot(3, 3, 5, projection="3d")
    ax.scatter(Y[:N_wake - 1, 0], Y[:N_wake - 1, 1], Y[:N_wake - 1, 2], c=inHD, marker=".", s=5, cmap="plasma")
    ax.set_title("this HD")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticklabels([])

    ax = fig.add_subplot(3, 3, 6, projection="3d")
    ax.scatter(Y[:N_wake - 1, 0], Y[:N_wake - 1, 1], Y[:N_wake - 1, 2], c=nextHD, marker=".", s=5, cmap="plasma")
    ax.set_title("next HD")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticklabels([])

    if in_view_wake is not None:
        ax = fig.add_subplot(3, 3, 8, projection="3d")
        ax.scatter(Y[:N_wake, 0], Y[:N_wake, 1], Y[:N_wake, 2], c=in_view_wake, marker=".", s=5, cmap="plasma")
        ax.set_title("Novel obj in view")
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])

    if savename is not None:
        saveFig(plt.gcf(), savename + "_Isomap", savefolder, filetype="png")


@dataclasses.dataclass
class IsomapConfig:
    new_obj_pos: tuple[int, int] = (7, 11)
    """Position [x, y] of novel object (goal) in the environment"""
    step: int = 1008
    """Training step checkpoint to load"""


def main():
    cfg = tyro.cli(IsomapConfig)
    new_obj_pos = list(cfg.new_obj_pos)
    step = cfg.step

    config_dir = str(Path(__file__).parent.parent / "Configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        args = compose(config_name="Conf1_Adel")

    agent_type = AgentType.AC
    agent_name = "cur" if agent_type == AgentType.AC else "rand"
    env_type = MinigridEnvNames.LRoom
    env_name = "fourroom" if env_type == MinigridEnvNames.FourRoomsObjs else "lroom"
    omt = True

    predictiveNet, ac_model, agent, env = setup(args=args,
                                                agent_type=agent_type,
                                                env_type=env_type,
                                                new_obj_pos=new_obj_pos,
                                                omt=omt,
                                                step=step)

    label_fn = make_novel_obj_in_view_label_fn(new_obj_pos) if new_obj_pos is not None else None

    create_isomap(
        pN=predictiveNet,
        agent=agent,
        env=env,
        noisemag=0,
        noisestd=0.25,
        timesteps_wake=256,
        timesteps_sleep=1000,
        savename=f"data_{agent_name}_{env_name}_step{step}_goal{new_obj_pos[0]}{new_obj_pos[1]}",
        savefolder="outputs",
        new_obj_pos=new_obj_pos,
        label_fn=label_fn,
    )

if __name__ == "__main__":
    main()
