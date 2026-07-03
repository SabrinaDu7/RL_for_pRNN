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
# # Four-Room Occupancy Heatmaps (HD-averaged) for specific runs

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scripts.wandb_data import fetch_occupancy_grids, OccupancyData

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

OUTPUT_DIR = Path("./outputs/fourroom")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENTITY = "blake-richards"
PROJECT = "curious-george-fourroom"
METRIC = "OPA_Occupancy_Map"

RUNS = {
    "rand": "pRNN_fourroom_rand_26-03-04-14-56-08",
    "cur":  "pRNN_fourroom_curious_26-03-04-14-52-38",
}


# %%
def fetch_run_data(run_name: str) -> OccupancyData:
    return fetch_occupancy_grids(
        entity=ENTITY,
        project=PROJECT,
        target_loc=None,
        metric=METRIC,
        filters={"display_name": run_name},
        max_workers=10,
    )


def last_step_hd_avg(data: OccupancyData) -> np.ndarray:
    """Return HD-averaged occupancy grid at the last training step.

    Grid shape is (n_runs, 4, H, W); n_runs=1 for a single-run fetch.
    Returns (H, W).
    """
    last_step = max(data.grids.keys())
    return np.nanmean(data.grids[last_step][0], axis=0)  # (4, H, W) -> (H, W)


def plot_hd_averaged(grid: np.ndarray, title: str, save_path: Path, vmax: float) -> None:
    """Plot a single HD-averaged occupancy heatmap."""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(grid, cmap="plasma", vmin=0, vmax=vmax, origin="upper", aspect="equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Occupancy")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved to {save_path}")


# %%
if __name__ == "__main__":
    print("Fetching runs...")
    data = {label: fetch_run_data(run_name) for label, run_name in RUNS.items()}
    grids = {label: last_step_hd_avg(d) for label, d in data.items()}

    for label, grid in grids.items():
        plot_hd_averaged(
            grid,
            title=f"Four-Room Occupancy — {label} agent",
            save_path=OUTPUT_DIR / f"heatmap_{label}.png",
            vmax=grid.max(),
        )
