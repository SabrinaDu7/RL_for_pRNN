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
# # Subroom Percentage Plots for WandB data

# %%
# !pwd

# %% [markdown]
# ## Imports and Function Definitions
# %%
import matplotlib.pyplot as plt
import torch
from typing import Literal, Sequence
from pathlib import Path
import numpy as np
from curious_george.io.wandb import (
    fetch_subroom_counts,
    plot_subroom_percentage,
    save_subroom_data,
    load_subroom_data,
    pairwise_ttest,
    SubroomData,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

CACHE_DIR = Path("../outputs/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(agent_type: str, with_obs: bool, entropy_coef: float, ctrl: bool | None, project: str) -> Path:
    ctrl_str = "none" if ctrl is None else str(ctrl)
    name = f"subroom__{project}__{agent_type}__obs{with_obs}__ec{entropy_coef}__ctrl{ctrl_str}.pt"
    return CACHE_DIR / name


def fetch_data(
    agent_type: Literal["rand", "cur"],
    with_obs: bool,
    entropy_coef: float = 0.0,
    ctrl: bool | None = False,
    project: str = "curious-george-omt",
    last_n: int | None = None,
    use_cache: bool = True,
) -> SubroomData:
    cache_path = _cache_key(agent_type, with_obs, entropy_coef, ctrl, project)
    if use_cache and cache_path.exists():
        return load_subroom_data(str(cache_path))
    data = fetch_subroom_counts(
        entity="blake-richards",
        project=project,
        last_n=last_n,
        metric="subroom_ids",
        group=f"pRNN_fourroom",
        filters={
            "config.rl.entropy_coef": entropy_coef,
            "config.exp.with_obs": with_obs,
            "config.exp.random_action_agent": False if agent_type == "cur" else True,
        },
        max_workers=10,
    )
    if use_cache:
        save_subroom_data(data, str(cache_path))
    return data


def show_subroom_percentage(
    data_list: list[SubroomData],
    labels: Sequence[str],
    colors: Sequence[str] | None = None,
    hatches: Sequence[str] | None = None,
    room_labels: Sequence[str] | None = None,
    step_range: tuple[int, int] | None = (9800, 10000),
    save_path: str | None = None,
) -> tuple:
    """Plot subroom percentage as grouped bars, one bar group per agent condition.

    Args:
        data_list: List of SubroomData instances to compare.
        labels: Legend labels, one per entry in data_list.
        colors: Optional bar colors, one per entry. Defaults to ``_SUBROOM_COLORS``.
        hatches: Optional hatch patterns per entry for color-blind accessibility.
            Defaults to ``_SUBROOM_HATCHES``.
        room_labels: Labels for each room. Defaults to ["Room 1", ..., "Room 4"].
        step_range: Training-step slice to aggregate over.
        save_path: If given, saves the figure to this path.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_subroom_percentage(
        [d.subroom_ids for d in data_list],
        group_labels=labels,
        colors=colors,
        hatches=hatches,
        room_labels=room_labels,
        ax=ax,
        step_range=step_range,
    )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def print_subroom_dfs(
    data_list: list[SubroomData],
    labels: list[str],
    step_range: tuple[int, int] | None = None,
    room_labels: list[str] | None = None,
) -> None:
    """Print Welch's t-test df per room for all group pairs."""
    # Reorder rooms to match plot_subroom_percentage: (1,2,3,4) → (1,3,4,2)
    tensors = [d.subroom_ids[:, :, [0, 2, 3, 1]] for d in data_list]
    n_rooms = tensors[0].shape[-1]
    rooms = room_labels or [f"Room {i + 1}" for i in range(n_rooms)]

    def per_run_means(tensor):
        data = tensor.numpy()
        if step_range is not None:
            data = data[:, step_range[0]:step_range[1], :]
        row_totals = np.nansum(data, axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = np.where(row_totals > 0, data / row_totals * 100, np.nan)
        return np.nanmean(pct, axis=1)  # (n_runs, n_rooms)

    run_means = [per_run_means(t) for t in tensors]

    from itertools import combinations
    for (i, li), (j, lj) in combinations(enumerate(labels), 2):
        pairs = [(run_means[i][:, r], run_means[j][:, r]) for r in range(n_rooms)]
        print(f"\nWelch df — {li} vs {lj}:")
        _, dfs, t_stats = pairwise_ttest(pairs, pair_labels=rooms)
        print(f"  {'Room':<12} {'t':>8} {'df':>8}")
        for room, t, df in zip(rooms, t_stats, dfs):
            print(f"  {room:<12} {t:>8.4f} {df:>8.2f}")


def main_single(
    agent: Literal["rand", "cur"],
    ec: float,
    ctrl: bool | None = False,
    project: str = "curious-george-fourroom",
    last_n: int | None = None,
    use_cache: bool = True,
):
    data = fetch_data(agent_type=agent, with_obs=False, entropy_coef=ec, ctrl=ctrl, project=project, last_n=last_n, use_cache=use_cache)
    ctrl_str = "_ctrl" if ctrl else ("_exp" if ctrl is None else "")
    save_path = f"../outputs/without_obs/subroom_{agent}{ctrl_str}_ec{ec}.png"
    show_subroom_percentage(data_list=[data], labels=[agent], save_path=save_path)


def main_multiple(
    agents: list[Literal["rand", "cur"]],
    ec: float,
    ctrl: bool | None = False,
    project: str = "curious-george-fourroom",
    colors: list[str] | None = None,
    hatches: list[str] | None = None,
    last_n: int | None = None,
    use_cache: bool = True,
    step_range: tuple[int, int] | None = (9000, 10000),
):
    """Plot subroom percentage for multiple agent types on one grouped bar chart.

    Args:
        agents: List of agent types, e.g. ``["rand", "cur"]``.
        ec: Entropy coefficient shared across all agents.
        ctrl: Control flag passed to fetch_data.
        project: WandB project name.
        colors: Optional bar colors, one per agent. Defaults to ``_SUBROOM_COLORS``.
        hatches: Optional hatch patterns per agent for color-blind accessibility.
            Defaults to ``_SUBROOM_HATCHES``.
        last_n: Fetch only the last N runs.
        use_cache: Whether to use cached data.
        step_range: Training-step slice to aggregate over.
    """
    data_list = [
        fetch_data(agent_type=a, with_obs=False, entropy_coef=ec, ctrl=ctrl,
                   project=project, last_n=last_n, use_cache=use_cache)
        for a in agents
    ]
    ctrl_str = "_ctrl" if ctrl else ("_exp" if ctrl is None else "")
    label_str = "_vs_".join(agents)
    save_path = f"../outputs/without_obs/subroom_{label_str}{ctrl_str}_ec{ec}.png"
    show_subroom_percentage(
        data_list=data_list,
        labels=agents,
        colors=colors,
        hatches=hatches,
        step_range=step_range,
        save_path=save_path,
    )
    print_subroom_dfs(data_list, labels=agents, step_range=step_range)


# %% [markdown]
# ## Subroom Percentage Plots
# %%
EC = 0

# %%
# CURIOUS
if __name__ == "__main__":
    AGENT = "cur"
    PROJECT = "curious-george-fourroom"

    data = fetch_data(agent_type=AGENT, with_obs=False, entropy_coef=EC, project=PROJECT)
    main_single(agent=AGENT, ec=EC, project=PROJECT)

# %%
# RANDOM
if __name__ == "__main__":
    AGENT = "rand"
    PROJECT = "curious-george-fourroom"

    data = fetch_data(agent_type=AGENT, with_obs=False, entropy_coef=EC, project=PROJECT)
    main_single(agent=AGENT, ec=EC, project=PROJECT)

# %%
# RANDOM vs CURIOUS
if __name__ == "__main__":
    PROJECT = "curious-george-fourroom"
    main_multiple(agents=["rand", "cur"], ec=EC, project=PROJECT, colors=["tab:red", "tab:blue"])
