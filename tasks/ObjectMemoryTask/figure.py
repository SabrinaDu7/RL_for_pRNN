from pathlib import Path
import re
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from jaxtyping import Float

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from prnn.utils import PredictiveNet, ActionEncodingsEnum, MinigridEnvNames
from prnn.utils.Shell import FaramaMinigridShell
from utils import get_env_var, AgentInputType
from curious_george import get_pN, make_env, grid_to_pixel_coords

RL_STORAGE = get_env_var("RL_STORAGE")
DEVICE = torch.device("cpu")

# Plotting: Trajectories
def plot_k_trajectories(
        env: FaramaMinigridShell,
        traj_count: int,
        locs_tensor: Float[torch.Tensor, "n_trajectories seqdur 2"],
        k: int | None = None,
        save_full_filename: str | None = None
    ) -> tuple[Figure, Float[np.ndarray, "k seqdur 2"]]:

    env_img = env.render(mode=None)
    plot_scaling_factor = env_img.shape[0] // env.width

    # Determine how many trajectories to plot. Default: 1
    n_trajectories = locs_tensor.shape[0]
    k = min(k, n_trajectories) if k is not None else 1

    fig = plt.figure()
    plt.imshow(env_img)

    # Plot each of the first k trajectories
    cmap = plt.cm.get_cmap('tab10')
    for i in range(k):
        color = cmap(i % 10)

        traj_grid: Float[np.ndarray, "seqdur 2"] = locs_tensor[i, :, :].cpu().numpy()
        traj_pixel = grid_to_pixel_coords(traj_grid, tile_size=plot_scaling_factor)
        traj_x = traj_pixel[:, 0]
        traj_y = traj_pixel[:, 1]

        plt.plot(traj_x, traj_y, ".-", color=color, linewidth=0.5, markersize=3, alpha=0.5)
        plt.plot(traj_x[0], traj_y[0], "o", color="red", markersize=5, alpha=0.8)  # Start point

    plt.xticks([])
    plt.yticks([])
    plt.title(f"Sample Trajectory after training on {traj_count} trajs")

    if save_full_filename is not None:
        plt.savefig(f"{save_full_filename}", dpi=200)

    return fig, locs_tensor[:k, :, :].cpu().numpy()


# Figure: Goal Modulation
def extract_objectlearning_values(key: str, directory: str) -> np.ndarray:
    """
    Extract values for a specific key from all objectLearning dictionaries in a directory.

    Args:
        key: The key to extract from each objectLearning dictionary
             (e.g., "goalmodulation", "traj_count", "inviewtimes")
        directory: Path to directory containing objectLearning_*.pt files

    Returns:
        numpy array of shape (M,) for scalar values or (M, ...) for array values,
        where M is the number of saved objectLearning dictionaries
    """
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all objectLearning files
    pattern = re.compile(r"objectLearning_(\d+)\.pt")
    files = []

    for file in directory_path.glob("objectLearning_*.pt"):
        match = pattern.match(file.name)
        if match:
            traj_count = int(match.group(1))
            files.append((traj_count, file))

    if not files:
        raise FileNotFoundError(f"No objectLearning_*.pt files found in {directory}")

    # Sort by trajectory count
    files.sort(key=lambda x: x[0])

    # Load and extract values
    values = []
    for traj_count, filepath in files:
        obj_learning = torch.load(filepath, weights_only=False)

        if key not in obj_learning:
            raise KeyError(f"Key '{key}' not found in {filepath.name}")

        value = obj_learning[key]

        # Convert to numpy if it's a tensor or other type
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        elif not isinstance(value, np.ndarray):
            # Scalar values (int, float, etc.)
            value = np.array(value)

        values.append(value)

    # Stack into array
    return np.array(values)


def plot_metric_vs_trajectories(
    x_values: list[np.ndarray],
    y_values: list[np.ndarray],
    agent_labels: list[str],
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    save_path: str | None = None,
    run_names: list[str] | None = None,
    control_y_values: list[np.ndarray] | None = None,
    plot_controls: bool = True,
    transparency: float = 0,
) -> Figure:
    """
    Plot metric values vs trajectory count for multiple agent types.

    Args:
        x_values: List of 1D numpy arrays, one per agent (typically trajectory counts)
        y_values: List of 1D numpy arrays, one per agent (typically metric values)
        agent_labels: List of labels for each agent (e.g., ["Random", "Curious"])
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Optional title for the plot
        save_path: Optional path to save the figure
        run_names: Optional list of run/folder names to display at bottom of plot
        control_y_values: Optional list of control metric values to plot as dotted lines
    """

    if len(x_values) != len(y_values) or len(x_values) != len(agent_labels):
        raise ValueError(
            f"Length mismatch: x_values ({len(x_values)}), "
            f"y_values ({len(y_values)}), agent_labels ({len(agent_labels)})"
        )

    if run_names is not None and len(run_names) != len(agent_labels):
        raise ValueError(
            f"Length mismatch: run_names ({len(run_names)}) must match "
            f"agent_labels ({len(agent_labels)})"
        )

    if control_y_values is not None and len(control_y_values) != len(agent_labels):
        raise ValueError(
            f"Length mismatch: control_y_values ({len(control_y_values)}) must match "
            f"agent_labels ({len(agent_labels)})"
        )

    if len(x_values) == 0:
        raise ValueError("Empty input lists")

    # Check lengths and truncate if necessary
    min_len = min(len(x) for x in x_values)
    max_len = max(len(x) for x in x_values)

    if min_len != max_len:
        print(f"Warning: Arrays have different lengths (min={min_len}, max={max_len}). "
              f"Truncating to minimum length {min_len}.")

    # Create plot
    plt.figure(figsize=(10, 6))

    lines = []
    for x, y, label in zip(x_values, y_values, agent_labels):
        line = plt.plot(x[:min_len], y[:min_len], marker='o', label=label, linewidth=1)
        lines.append(line[0])

    # Plot control lines (dotted) with matching colors
    if control_y_values is not None:
        if plot_controls:
            for i, (x, y) in enumerate(zip(x_values, control_y_values)):
                plt.plot(x[:min_len], y[:min_len],
                        linestyle='--',
                        color=lines[i].get_color(),
                        linewidth=1)

        # Plot difference lines (goal - control) with transparency
        for i, (x, y_goal, y_control) in enumerate(zip(x_values, y_values, control_y_values)):
            diff = y_goal[:min_len] - y_control[:min_len]
            plt.plot(x[:min_len], diff,
                     linestyle='-',
                     color=lines[i].get_color(),
                     linewidth=1,
                     alpha=transparency)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    if title is not None:
        plt.title(title, fontsize=14)

    if control_y_values is not None:
        plt.legend(fontsize=8, title="Solid: Goal | Dashed: Control | Transparent: Diff")
    else:
        plt.legend(fontsize=11)

    plt.grid(True, alpha=0.3)

    # Add run names at bottom if provided
    if run_names is not None:
        run_text = "Runs: " + " | ".join([f"{label}: {name}" for label, name in zip(agent_labels, run_names)])
        plt.figtext(0.5, 0.02, run_text, ha='center', fontsize=9, style='italic', wrap=True)
        plt.tight_layout(rect=(0, 0.05, 1, 1))  # Leave space at bottom for text
    else:
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return plt.gcf()


# Figure: Object Learning Panels
def plot_obj_pixel_change_panel(objectLearning: dict) -> None:
    """
    Create a boxplot panel showing pixel changes at goal and control locations.

    Args:
        objectLearning: Dictionary containing 'objectloc_deltaobs' and 'controlloc_deltaobs'
    """
    deltaobs_goal = objectLearning["objectloc_deltaobs"]
    deltaobs_ctl = objectLearning["controlloc_deltaobs"]

    plt.boxplot(
        deltaobs_goal,
        showfliers=False,
        positions=[1.8, 1, 2.2],
    )
    plt.boxplot(
        deltaobs_ctl,
        showfliers=False,
        positions=[4.3, 3.5, 4.7],
    )
    plt.plot(plt.xlim(), [0, 0], "k--")
    plt.plot([3, 3], plt.ylim(), "k:")
    plt.xticks([1, 1.8, 2.2, 3.5, 4.3, 4.7], ["G", "R", "B", "G", "R", "B"])
    plt.ylabel("Change in Predicted Observation", fontsize=6)


def flatten(render: Float[np.ndarray, "B T+1 render_size render_size C"], 
            obs: Float[torch.Tensor, "B T+1 FOV*FOV*C"], 
            obs_pred: Float[torch.Tensor, "B T FOV*FOV*C"], 
            obs_pred_notrain: Float[torch.Tensor, "B T FOV*FOV*C"],
            ) -> tuple[
                Float[np.ndarray, "B*T render_size render_size C"],
                Float[torch.Tensor, "1 B*T FOV*FOV*C"],
                Float[torch.Tensor, "1 B*T FOV*FOV*C"],
                Float[torch.Tensor, "1 B*T FOV*FOV*C"]
            ]:
    
    # Get dimensions
    B = render.shape[0]
    T = obs_pred.shape[1]  # Number of prediction timesteps

    # Flatten batch dimension to match how quantifyObjectLearning flattens them
    # IMPORTANT: Only use first T timesteps of render/obs to align with predictions
    # render: (B, T+1, ...) -> (B, T, ...) -> (B*T, ...)

    render_flat = render[:, :T, ...].reshape(B * T, *render.shape[2:])

    # obs: (B, T+1, X) -> (B, T, X) -> (B*T, X) then process with pred2np
    obs_flat = obs[:, :T, :].reshape(B * T, obs.shape[2]).unsqueeze(dim=0)

    # obs_pred: (B, T, X) -> (B*T, X) then process with pred2np
    obs_pred_flat = obs_pred.reshape(B * T, obs_pred.shape[2]).unsqueeze(dim=0)
    obs_pred_notrain_flat = obs_pred_notrain.reshape(B * T, obs_pred_notrain.shape[2]).unsqueeze(dim=0)

    return render_flat, obs_flat, obs_pred_flat, obs_pred_notrain_flat


def plot_example_obs_sequence_panel(
    objectLearning: dict,
    testTrial: dict,
    pN: PredictiveNet,
    firstrow: int = 3,
    whichview: int = 1
) -> None:
    """
    Create a panel showing example observation sequences.

    Args:
        objectLearning: Dictionary containing 'inviewtimes'
        testTrial: Dictionary containing 'render', 'obs', 'obs_pred', 'obs_pred_control'
        pN: Predictive network with env_shell and plotSequence method
        firstrow: First row index for plotting (default=3)
        whichview: Which view timepoint to center on (default=1)
    """

    # Get arrays from testTrial - shape (B, T+1/T, ...)
    render = testTrial["render"]  # (B, T+1, render_size, render_size, C)
    obs = testTrial["obs"]  # (B, T+1, X)
    obs_pred = testTrial["obs_pred"]  # (B, T, X)
    obs_pred_notrain = testTrial["obs_pred_control"]  # (B, T, X)

    if isinstance(render, np.ndarray):
        render_flat, obs_flat, obs_pred_flat, obs_pred_notrain_flat = flatten(render, obs, obs_pred, obs_pred_notrain)
    else:
        render_flat, obs_flat, obs_pred_flat, obs_pred_notrain_flat = render, obs, obs_pred, obs_pred_notrain

    inviewtimes = objectLearning["inviewtimes"]
    extimes = range(inviewtimes[whichview] - 2, inviewtimes[whichview] + 5)

    # Plot using flattened sequences
    # All sequences now have the same flattened length (B*T) with aligned indices
    pN.plotSequence(render_flat, extimes, firstrow, label="State", show_axis=True)
    pN.plotSequence(pN.env_shell.pred2np(obs_flat), extimes, firstrow + 1, label="True Obs", show_axis=True)
    pN.plotSequence(
        pN.env_shell.pred2np(obs_pred_notrain_flat),
        extimes,
        firstrow + 2,
        label="Pred (Ctrl)",
        show_axis=True
    )
    pN.plotSequence(
        pN.env_shell.pred2np(obs_pred_flat), extimes, firstrow + 3, label="Pred (Exp)", show_axis=True
    )

# Generate Figures
def figure_goal_modulation_vs_trajectories(
        rand_dir: str, 
        curious_dir: str, 
        num_datapoints: int, 
        transparency: float = 0,
        plot_controls: bool = True
        ) -> Figure:
    """
    Generate a plot comparing goal modulation vs trajectory count
    for random and curious agents.
    """

    # Extract goal modulation and trajectory counts for random agent
    seqdur = 256
    # num_datapoints = 25

    # Extract the same for curious agent
    # curious_dir = f"{RL_STORAGE}/omt_curious_25-11-07-16-37-10"
    curious_goalmod = extract_objectlearning_values("goalmodulation", curious_dir)
    curious_traj = extract_objectlearning_values("traj_count", curious_dir)
    curious_ctlmod = extract_objectlearning_values("ctlmodulation_diffloc", curious_dir)

    # rand_dir = f"{RL_STORAGE}/omt_rand_25-11-07-16-42-10"
    rand_goalmod = extract_objectlearning_values("goalmodulation", rand_dir)
    rand_traj = extract_objectlearning_values("traj_count", rand_dir)
    rand_ctlmod = extract_objectlearning_values("ctlmodulation_diffloc", rand_dir)

    # Plot comparison
    fig = plot_metric_vs_trajectories(
        x_values=[rand_traj[:num_datapoints], curious_traj[:num_datapoints]],
        y_values=[rand_goalmod[:num_datapoints], curious_goalmod[:num_datapoints]],
        agent_labels=["Random", "Curious"],
        xlabel="Trajectory Count",
        ylabel="Pixel Change",
        title=f"Goal Modulation Over Training (Seqdur = {seqdur})",
        save_path=None,
        run_names=[rand_dir, curious_dir],
        control_y_values=[rand_ctlmod[:num_datapoints], curious_ctlmod[:num_datapoints]],
        transparency=transparency,
        plot_controls=plot_controls,
    )
    return fig


def figure_object_learning(env_name: MinigridEnvNames | FaramaMinigridShell, 
                           run_name: str, 
                           traj_num: int, 
                           save_folder: str, 
                           testTrial: dict | None = None,
                           objectLearning: dict | None = None,
                           pN: PredictiveNet | None = None,
                           rl_storage: str = RL_STORAGE,
                           config: DictConfig | None = None,
                           show: bool = False,
                           save: bool = False) -> Figure:
    """
    Generate object learning figure panels for a specific run.
    """
    from prnn.utils.general import saveFig

    # Setup
    plt.figure(figsize=(12, 8))
    run_filepath = Path(rl_storage) / Path(run_name)

    # Loading components
    if config is None:
        config = OmegaConf.load(run_filepath / "config.yaml") # pyright: ignore[reportAssignmentType]
        assert config is not None

    if objectLearning is None:
        objectLearning_dict_file = run_filepath / f"objectLearning_{traj_num}.pt"
        object_learning_dict = torch.load(objectLearning_dict_file, weights_only=False)
    else:
        object_learning_dict = objectLearning

    if isinstance(env_name, MinigridEnvNames):
        env = make_env(env_key=env_name.value, input_type=AgentInputType.H_PO.value, act_enc=ActionEncodingsEnum.SpeedHD.value)
    else:
        env = env_name

    if pN is None:
        prnn_ckpt = str(run_filepath / f"pN-{traj_num}.pt")
        predictiveNet = get_pN(args=config, env=env, device=DEVICE, pRNN_ckpt=prnn_ckpt)
    else:
        predictiveNet = pN
    
    if testTrial is None:
        test_trial_file = run_filepath / f"testTrial_{traj_num}.pt"
        test_trial_dict = torch.load(test_trial_file, weights_only=False)
    else:
        test_trial_dict = testTrial
    
    # Plotting
    plt.subplot(4, 3, 1)
    plot_obj_pixel_change_panel(object_learning_dict)

    plot_example_obs_sequence_panel(
        objectLearning=object_learning_dict,
        testTrial=test_trial_dict,
        pN=predictiveNet,
        whichview=1,
    )

    plt.title(f'Object Learning after training on {traj_num} trajectories\nNovel Object Loc: {config.tasks.new_obj_pos}', fontsize=9, loc='right', y=6)
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        saveFig(fig=plt.gcf(), savename="ObjectLearning_" + run_name, savepath=save_folder, filetype="png")
    
    return plt.gcf()
        