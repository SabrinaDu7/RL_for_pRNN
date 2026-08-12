"""Regenerate the object-trace-cell figures from cached results.

These were originally produced by throwaway scripts; this module makes them
reproducible from the `.npz`/`.npy` caches under `outputs/trace/` so the
figures in `docs/exp_object_trace_cells_2026-07-30.md` can always be rebuilt.

    uv run python scripts/trace_cell_figures.py           # all
    uv run python scripts/trace_cell_figures.py occupancy # one

Caches consumed (produced by the analyses described in that doc):
    maps_all.npz    baseline + one post-exposure map set per object location
    maps_dense.npz  16-checkpoint map series per location
    curves.npy      within-run object-modulation percentile per checkpoint
    behavior_n3.npy on-policy occupancy/in-view percentiles, 3 seeds x 3 locations
    rewardmap_*.npy per-checkpoint prediction MSE at the object cell
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/trace")
LOCS = [(7, 11), (14, 7), (7, 2)]


def _env():
    from hydra import initialize_config_dir, compose
    from prnn.utils import ActionEncodingsEnum, AgentInputType, MinigridEnvNames
    from curious_george import make_env

    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        compose(config_name="main")
    return make_env(env_key=MinigridEnvNames.LRoom, input_type=AgentInputType.H_PO.value,
                    act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0)


def occupancy() -> None:
    """Probe sampling and the derived valid-bin mask."""
    from scripts.trace import trace_figure as tf

    d = np.load(OUT / "maps_all.npz")
    tf.occupancy_figure(occupancy=d["occ"], valid=d["valid"]).savefig(
        OUT / "fig_occupancy.png", dpi=160, bbox_inches="tight")
    print("wrote fig_occupancy.png")


def tuned_units() -> None:
    """Top-SI place fields under both bin weightings, and the weighting comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.trace import trace_maps as tm, trace_figure as tf

    d = np.load(OUT / "maps_all.npz")
    maps, occ = d["baseline"], d["occ"]
    si_o = tm.spatial_info(maps=maps, occupancy=occ, weighting="occupancy")
    si_u = tm.spatial_info(maps=maps, occupancy=occ, weighting="uniform")

    for si, tag, title in ((si_o, "main_train", "Skaggs (occupancy-weighted) SI"),
                           (si_u, "uniformSI", "uniform-weighted SI")):
        top = np.argsort(-np.nan_to_num(si, nan=-np.inf))[:32]
        tf.unit_panel(maps=maps, units=top, si=si, n_cols=8,
                      title=f"Baseline pRNN place fields, ranked by {title} "
                            "(occupancy-masked)").savefig(
            OUT / f"fig_tuned_units_{tag}.png", dpi=160, bbox_inches="tight")

    ny, nx = maps.shape[1:]
    peaks = [np.unravel_index(np.nanargmax(m), m.shape) if np.isfinite(m).any() else (0, 0)
             for m in maps]
    opeak = np.array([occ[p] for p in peaks])
    f, ax = plt.subplots(figsize=(5.2, 4.4))
    s = ax.scatter(si_o, si_u, c=opeak, s=9, cmap="viridis")
    lim = [0, float(np.nanmax([si_o, si_u])) * 1.05]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Skaggs SI (occupancy-weighted)"); ax.set_ylabel("uniform-weighted SI")
    ax.set_title("SI under two bin weightings, coloured by\nprobe occupancy at the unit's peak bin",
                 fontsize=10)
    f.colorbar(s, ax=ax, label="samples in peak bin")
    f.tight_layout()
    f.savefig(OUT / "fig_si_weighting.png", dpi=160, bbox_inches="tight")
    print("wrote fig_tuned_units_main_train.png, fig_tuned_units_uniformSI.png, fig_si_weighting.png")


def trace_3loc() -> None:
    """Units most modulated at (7,11), across the three object locations."""
    from scripts.trace import trace_maps as tm, trace_figure as tf

    env = _env()
    d = np.load(OUT / "maps_all.npz")
    base = d["baseline"]
    L = (7, 11)
    dm = (tm.object_modulation(maps=d[f"m_{L[0]}_{L[1]}"], obj_xy=L, env=env, radius=2.0)
          - tm.object_modulation(maps=base, obj_xy=L, env=env, radius=2.0))
    units = np.argsort(-np.nan_to_num(dm, nan=-np.inf))[:8]
    tf.trace_panel(
        maps_by_checkpoint=[base] + [d[f"m_{l[0]}_{l[1]}"] for l in LOCS],
        labels=["pre-exposure"] + [f"obj @ {l}" for l in LOCS],
        units=units, obj_xy=L, removal_after=0,
        title=f"Units most modulated at {L} after exposure there.\nWhite + marks {L}. "
              "Same colour scale per row. Columns 2-4 are 3 separate runs.",
    ).savefig(OUT / "fig_trace_3loc.png", dpi=150, bbox_inches="tight")
    print("wrote fig_trace_3loc.png")


def exposure() -> None:
    """The 16-checkpoint exposure timeline, and the within-run percentile curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.trace import trace_maps as tm, trace_figure as tf

    env = _env()
    d = np.load(OUT / "maps_dense.npz")
    steps_by_loc = np.load(OUT / "steps_by_loc.npy", allow_pickle=True).item()
    curves = np.load(OUT / "curves.npy", allow_pickle=True).item()

    L = (14, 7)
    steps = steps_by_loc[L]
    sel = [s for s in steps if s in (0, 200, 600, 1000, 1400, 1800, 2400, 2992)]
    base = d["baseline"]
    dm = (tm.object_modulation(maps=d[f"m_{L[0]}_{L[1]}_{steps[-1]}"], obj_xy=L, env=env, radius=2.0)
          - tm.object_modulation(maps=base, obj_xy=L, env=env, radius=2.0))
    units = np.argsort(-np.nan_to_num(dm, nan=-np.inf))[:8]
    tf.trace_panel(
        maps_by_checkpoint=[base] + [d[f"m_{L[0]}_{L[1]}_{s}"] for s in sel],
        labels=["pre"] + [str(s) for s in sel], units=units, obj_xy=L,
        title=f"Exposure timeline, object at {L} (white +). Columns = trajectories of exposure.\n"
              f"Rows = the 8 units most modulated at {L} by the end. One colour scale per row.",
    ).savefig(OUT / "fig_exposure_timeline_14_7.png", dpi=150, bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for loc, c in curves.items():
        axes[0].plot(c[:, 0], c[:, 1], "o-", ms=4, label=f"obj @ {loc}")
        axes[1].plot(c[:, 0], c[:, 2], "o-", ms=4, label=f"obj @ {loc}")
    for ax, t in zip(axes, ["population mean", "top-25 subpopulation"]):
        ax.axhline(95, color="r", ls="--", lw=1, label="95th pct")
        ax.axhline(50, color="grey", ls=":", lw=1)
        ax.set_ylim(0, 100)
        ax.set_xlabel("trajectories of object exposure")
        ax.set_ylabel("percentile of the object location\namong all 172 walkable cells")
        ax.set_title(t, fontsize=10)
    axes[0].legend(fontsize=7)
    fig.suptitle("Does object-location modulation grow during exposure? (within-run null)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_exposure_timecourse.png", dpi=160, bbox_inches="tight")
    print("wrote fig_exposure_timeline_14_7.png, fig_exposure_timecourse.png")


def behaviour() -> None:
    """On-policy object-directed behaviour, 3 seeds x 3 locations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = np.load(OUT / "behavior_n3.npy", allow_pickle=True)
    cols = {(7, 11): "C0", (14, 7): "C1", (7, 2): "C2"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for i, (k0, kF, t) in enumerate([(2, 3, "occupancy within 2 bins"),
                                     (4, 5, "object location in 7x7 view")]):
        for r in rows:
            loc = r[0] if isinstance(r[0], tuple) else tuple(r[0])
            ax[i].plot([0, 1], [float(r[k0]), float(r[kF])], "o-",
                       color=cols[loc], alpha=0.8, ms=5)
        ax[i].set_xticks([0, 1]); ax[i].set_xticklabels(["traj 0", "final (~2992)"])
        ax[i].set_ylim(0, 100); ax[i].axhline(50, color="grey", ls=":", lw=1)
        ax[i].set_ylabel("percentile among 172 walkable cells")
        ax[i].set_title(t, fontsize=10)
    for loc, c in cols.items():
        ax[0].plot([], [], "o-", color=c, label=f"obj @ {loc}")
    ax[0].legend(fontsize=8)
    fig.suptitle("Object-directed behaviour, 3 seeds x 3 locations, 128 rollouts each",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_behavior_n3.png", dpi=160, bbox_inches="tight")
    print("wrote fig_behavior_n3.png")


ALL = {"occupancy": occupancy, "tuned_units": tuned_units, "trace_3loc": trace_3loc,
       "exposure": exposure, "behaviour": behaviour}

if __name__ == "__main__":
    which = sys.argv[1:] or list(ALL)
    for name in which:
        ALL[name]()
