import numpy as np
import matplotlib.pyplot as plt
import torch
import re
from pathlib import Path

from utils import get_env_var
RL_STORAGE = get_env_var("RL_STORAGE")


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
    run_names: list[str] | None = None
) -> None:
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

    for x, y, label in zip(x_values, y_values, agent_labels):
        plt.plot(x[:min_len], y[:min_len], marker='o', label=label, linewidth=2)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    if title is not None:
        plt.title(title, fontsize=14)

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

    plt.show()


if __name__ == "__main__":
    # Extract goal modulation and trajectory counts for random agent
    seqdur = 256
    num_datapoints = 25

    rand_dir = f"{RL_STORAGE}/omt_rand_25-11-07-16-42-10"
    rand_goalmod = extract_objectlearning_values("goalmodulation", rand_dir)
    rand_traj = extract_objectlearning_values("traj_count", rand_dir)

    # Extract the same for curious agent
    curious_dir = f"{RL_STORAGE}/omt_curious_25-11-07-16-37-10"
    curious_goalmod = extract_objectlearning_values("goalmodulation", curious_dir)
    curious_traj = extract_objectlearning_values("traj_count", curious_dir)

    # Plot comparison
    plot_metric_vs_trajectories(
        x_values=[rand_traj[:num_datapoints], curious_traj[:num_datapoints]],
        y_values=[rand_goalmod[:num_datapoints], curious_goalmod[:num_datapoints]],
        agent_labels=["Random", "Curious"],
        xlabel="Trajectory Count",
        ylabel="Goal Modulation",
        title=f"Goal Modulation Over Training (Seqdur = {seqdur})",
        save_path=f"results/goalmodulation{num_datapoints}.png",
        run_names=["omt_rand_25-11-07-16-42-10", "omt_curious_25-11-07-16-37-10"]
    )
