"""Figure 3: per-metric comparison between Curious and Random agents (mean ± SEM)."""
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from curious_george.io.wandb import fetch_run_traces, plot_metric, add_sig_line
from scripts.legacy.figure4 import CACHE_DIR, save_df, load_df, df_to_tensor, align_tensors

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

ENTITY = "blake-richards"
PROJECT = "curious-george"

CURIOUS_RUNS = [
    "pRNN_LRoom_NoObs_26-02-15-17-33-11",
    "pRNN_curious_26-04-07-22-14-53",
    "pRNN_curious_26-04-07-22-14-54",
    "pRNN_curious_26-04-07-22-29-43",
    "pRNN_curious_26-04-08-08-23-37",

]
RAND_RUNS = [
    "pRNN_rand_25-11-24-12-12-49",
    "pRNN_rand_26-04-08-01-13-18",
    "pRNN_rand_26-04-07-22-15-07",
    "pRNN_rand_26-04-07-22-15-06",
]

METRICS = [
    "mean SI_onPolicy",
    "SWdist_onPolicy",
    "MI_policy_eval",
    "sRSA_onPolicy",
]

METRIC_YLABELS = {
    "mean SI_onPolicy": "Mean Spatial Information",
    "SWdist_onPolicy": "Sleep-Wake Distance",
    "MI_policy_eval": "Mutual Information",
    "sRSA_onPolicy": "Spatial Representation Similarity",
}


def _names_cache_key(project: str, metric: str, step_key: str, run_names: list[str]) -> Path:
    names_hash = hashlib.md5("|".join(sorted(run_names)).encode()).hexdigest()[:8]
    name = f"{project}__{metric.replace('/', '_')}__{step_key}__names_{names_hash}.pt"
    return CACHE_DIR / name


def fetch_run_traces_by_names_cached(
    entity: str,
    project: str,
    metric: str,
    step_key: str,
    run_names: list[str],
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch traces for specific named runs and cache the result.

    Args:
        entity: WandB entity.
        project: WandB project.
        metric: WandB metric key.
        step_key: Step key used as the x-axis.
        run_names: Display names of the runs to fetch.
        use_cache: If True, read/write a local cache file.

    Returns:
        DataFrame with runs as rows and steps as columns.
    """
    cache_path = _names_cache_key(project, metric, step_key, run_names)
    if use_cache and cache_path.exists():
        return load_df(cache_path)

    dfs = [
        fetch_run_traces(
            entity=entity, project=project, metric=metric,
            step_key=step_key, filters={"display_name": name},
        )
        for name in run_names
    ]
    df = pd.concat(dfs, ignore_index=True)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        save_df(df, cache_path)
    return df


if __name__ == "__main__":
    save_dir = Path("outputs/figure3")
    save_dir.mkdir(parents=True, exist_ok=True)

    for metric, ylabel in METRIC_YLABELS.items():
        df_curious = fetch_run_traces_by_names_cached(
            ENTITY, PROJECT, metric, "_step", CURIOUS_RUNS,
        )
        df_random = fetch_run_traces_by_names_cached(
            ENTITY, PROJECT, metric, "_step", RAND_RUNS,
        )

        tensor_curious, steps_curious = df_to_tensor(df_curious)
        tensor_random, steps_random = df_to_tensor(df_random)

        tensors, common_steps = align_tensors(
            [tensor_curious, tensor_random],
            [steps_curious, steps_random],
        )

        fig, ax, p_vals, all_dfs, all_t_stats = plot_metric(
            tensors,
            common_steps,
            colors=["purple", "blue"],
            labels=["Curious", "Random"],
            xlabel="Training step",
            ylabel=ylabel,
            use_std=False,
            xlim_max=117000,
        )

        group_names = ["Curious", "Random"]
        print(f"\n  {metric} — means and SEMs across runs:")
        for name, tensor in zip(group_names, tensors):
            data = tensor.numpy()  # [n_steps, n_runs]
            means = np.nanmean(data, axis=-1)
            sems = np.nanstd(data, axis=-1) / np.sqrt((~np.isnan(data)).sum(axis=-1))
            key = ("Curious", "Random")
            print(f"    {name}: mean={means[-4]:.6f}, SEM={sems[-4]:.6f}, t = {all_t_stats[key][-4]:.4f}, df={all_dfs[key][-4]:.2f}, p={p_vals[key][-4]:.4f}  (step {common_steps[-4]})")
            print(f"    {name}: mean={means[-3]:.6f}, SEM={sems[-3]:.6f}, t = {all_t_stats[key][-3]:.4f}, df={all_dfs[key][-3]:.2f}, p={p_vals[key][-3]:.4f}  (step {common_steps[-3]})")
            print(f"    {name}: mean={means[-2]:.6f}, SEM={sems[-2]:.6f}, t = {all_t_stats[key][-2]:.4f}, df={all_dfs[key][-2]:.2f}, p={p_vals[key][-2]:.4f}  (step {common_steps[-2]})")
            print(f"    {name}: mean={means[-1]:.6f}, SEM={sems[-1]:.6f}, t = {all_t_stats[key][-1]:.4f}, df={all_dfs[key][-1]:.2f}, p={p_vals[key][-1]:.4f}  (step {common_steps[-1]})")

        factor = 0.15 if (metric == "MI_policy_eval" or metric == "SWdist_onPolicy") else 0.05
        ylo, yhi = ax.get_ylim()
        yrange = yhi - ylo
        sig_height = yhi + yrange * factor
        add_sig_line(
            ax, common_steps, p_vals,
            exp_label="Curious",
            control_labels=["Random"],
            control_colors=["black"],
            height=sig_height,
            fontsize=6,
        )
        ax.set_ylim(ylo, yhi + yrange * 0.15)

        safe_name = metric.replace(" ", "_").replace("/", "_")
        fig.savefig(save_dir / f"{safe_name}.png", dpi=300)

    plt.show()
