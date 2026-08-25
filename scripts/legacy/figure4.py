"""Figure 4: Goal Minus Ctrl Vs. Step Count for Curious (OMT) and Random (ctrl) agents."""
import hashlib
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from curious_george.io.wandb import fetch_run_traces, plot_metric

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

ENTITY = "blake-richards"
METRIC = "Analysis/Goal Minus Ctrl Vs. Step Count"
STEP_KEY = "step_count"
CACHE_DIR = Path("outputs/cache")


def _df_cache_key(project: str, metric: str, step_key: str, filters: dict) -> Path:
    filters_hash = hashlib.md5(json.dumps(filters, sort_keys=True).encode()).hexdigest()[:8]
    name = f"{project}__{metric.replace('/', '_')}__{step_key}__{filters_hash}.pt"
    return CACHE_DIR / name


def save_df(df: pd.DataFrame, path: Path) -> None:
    torch.save({"df": df}, path)


def load_df(path: Path) -> pd.DataFrame:
    return torch.load(path, weights_only=False)["df"]


def fetch_run_traces_cached(
    entity: str,
    project: str,
    metric: str,
    step_key: str,
    filters: dict,
    use_cache: bool = True,
    max_runs: int | None = None,
) -> pd.DataFrame:
    cache_path = _df_cache_key(project, metric, step_key, filters)
    if use_cache and cache_path.exists():
        df = load_df(cache_path)
        return df.iloc[:max_runs] if max_runs is not None else df
    df = fetch_run_traces(entity=entity, project=project, metric=metric, step_key=step_key, filters=filters)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        save_df(df, cache_path)
    return df.iloc[:max_runs] if max_runs is not None else df


def df_to_tensor(df: pd.DataFrame) -> tuple[torch.Tensor, list]:
    """Convert a fetch_run_traces DataFrame to a [n_steps, n_runs] tensor.

    Args:
        df: DataFrame with runs as rows and steps as columns (from fetch_run_traces).

    Returns:
        (tensor, steps): tensor of shape [n_steps, n_runs], sorted step list.
    """
    steps = sorted(df.columns.tolist())
    arr = df[steps].values.T  # [n_steps, n_runs]
    return torch.tensor(arr, dtype=torch.float32), steps


def align_tensors(
    tensors: list[torch.Tensor], steps_list: list[list]
) -> tuple[list[torch.Tensor], list]:
    """Align tensors to their shared step intersection.

    Args:
        tensors: List of [n_steps, n_runs] tensors (steps may differ).
        steps_list: List of step lists corresponding to each tensor.

    Returns:
        (aligned_tensors, common_steps): tensors trimmed to the common steps.
    """
    common_steps = sorted(set.intersection(*[set(s) for s in steps_list]))
    aligned = []
    for tensor, steps in zip(tensors, steps_list):
        idx = [steps.index(s) for s in common_steps]
        aligned.append(tensor[idx, :])
    return aligned, common_steps


if __name__ == "__main__":
    save_dir = Path("outputs/figure4")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Fetch all runs from curious-george-omt (curious agent)
    df_curious = fetch_run_traces_cached(
        entity=ENTITY,
        project="curious-george-omt",
        metric=METRIC,
        step_key=STEP_KEY,
        filters={"config.exp.curious_agent": True,
                 "config.tasks.new_obj_loc": [7, 11],
                 },
        # max_runs=100,
    )

    # Fetch runs from curious-george-ctrl filtered to random (non-curious) agent
    df_random = fetch_run_traces_cached(
        entity=ENTITY,
        project="curious-george-ctrl",
        metric=METRIC,
        step_key=STEP_KEY,
        filters={"config.exp.curious_agent": False},
    )

    tensor_curious, steps_curious = df_to_tensor(df_curious)
    tensor_random, steps_random = df_to_tensor(df_random)

    tensors, common_steps = align_tensors(
        [tensor_curious, tensor_random],
        [steps_curious, steps_random],
    )

    # Hardcoded steps
    STEPS = [8, 208, 408, 608, 808, 1008, 1208, 1408, 1608, 1808, 2008, 2208, 2408, 2608, 2808]

    import numpy as np
    curious_data = tensors[0].numpy()  # [n_steps, n_runs]
    n_valid = np.sum(~np.isnan(curious_data), axis=-1)
    sem = np.nanstd(curious_data, axis=-1) / np.sqrt(n_valid)
    print("\nCurious agent SEM per step:")
    for s, e, n in zip(STEPS, sem, n_valid):
        print(f"  step {s:5d}: SEM = {e:.6f}  (n={n})")

    fig, ax, all_p_vals, all_dfs, _ = plot_metric(
        tensors,
        STEPS,
        colors=["purple", "blue"],
        labels=["Curious", "Random"],
        xlabel="Trajectories taken in novel environment",
        ylabel="Goal Modulation \n(ΔGreen at target - ΔGreen at control locations)",
        use_std=False,
    )

    fig.savefig(save_dir / "goal_minus_ctrl_vs_step_count.png", dpi=300)
    plt.show()
