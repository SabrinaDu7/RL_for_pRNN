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


def trace_3loc(n_units: int = 3) -> None:
    """One block per object location: its OWN units, its OWN marker.

    The previous construction selected units by Δ object-modulation at (7,11)
    only and drew the `+` at (7,11) in every column, including the columns whose
    run had its object at (14,7) or (7,2). Those columns therefore showed units
    chosen for a different location, marked at a different location, and carried
    no information about their own object - the figure could not have shown a
    trace at (14,7) or (7,2) even if one existed.

    Here each of the three blocks selects the units most modulated at ITS
    location and marks ITS location, so every panel is scored against the thing
    it is about. Reading down the marked column of each block is then the
    §4 location control in spatial-tuning form: the diagonal blocks are where a
    trace would appear, the off-diagonal panels are its control.

    Colour scale is shared across a row (one unit, four maps), never across
    rows, so a fading field reads as fading.

    UNITS ARE RANKED BY THE STATISTIC THE NULL TEST ITSELF USES
    `trace_metric.trace_scores`: the change in field gain at the object cell,
    expressed as a percentile among the SAME unit's change at all 172 walkable
    cells. Above 95 is what makes a unit an object cell, so the rows are the
    units the test flagged rather than a separate population.

    The obvious alternatives both fail, measured on this cache:
      - ranking on `object_modulation` (what this figure used to do) is a ratio
        to the unit's own activity, so its variance scales as 1/rate and the top
        fills with the QUIETEST units - all nine former selections sat in the
        bottom 0-28th percentile of rate and changed 0.04-0.31x as much as a
        typical large change;
      - ranking on the raw change in disc rate inverts that bias, selecting the
        LOUDEST units and confounding a whole map dimming with a local change -
        its top picks land at the 4th, 10th and 54th percentile of their own
        within-unit null, i.e. the least object-like units in the population.
    The within-unit percentile is scale-free AND drift-controlled, because every
    comparison happens inside one unit's own distribution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.analysis_OMT import get_walkable_mask, get_walkable_minigrid_positions
    from scripts.trace import trace_figure as tf
    from scripts.trace.trace_metric import OBJECT_CELL_PCTILE, trace_scores

    env = _env()
    d = np.load(OUT / "maps_all.npz")
    base = d["baseline"]
    series = [base] + [d[f"m_{x}_{y}"] for x, y in LOCS]
    labels = ["pre-exposure"] + [f"obj @ {l}" for l in LOCS]
    n_col = len(series)
    cells = [tuple(p.tolist())
             for p in get_walkable_minigrid_positions(get_walkable_mask(env))]

    picks, pcts, dgs = {}, {}, {}
    for L in LOCS:
        s = trace_scores(maps_pre=base, maps_post=d[f"m_{L[0]}_{L[1]}"], env=env,
                         cells=cells, obj_cell=L)
        pct, dg = s["percentile"], s["dg_obj"]
        # percentile first, |dg| as the tie-break: many units saturate at 99
        order = np.lexsort((-np.nan_to_num(np.abs(dg)), -np.nan_to_num(pct, nan=-np.inf)))
        picks[L], pcts[L], dgs[L] = order[:n_units], pct[order[:n_units]], dg[order[:n_units]]
        print(f"  {L}: units {picks[L].tolist()}  percentile {np.round(pcts[L], 0).tolist()}  "
              f"Δfield gain {np.round(dgs[L], 3).tolist()}   "
              f"[{s['n_object_cells']}/{s['n_scored']} units over the "
              f"{OBJECT_CELL_PCTILE:.0f}th percentile = {s['frac']:.3f}, p={s['p_binom']:.3f}]")

    n_row = n_units * len(LOCS)
    fig, axes = plt.subplots(n_row, n_col, figsize=(1.6 * n_col, 1.55 * n_row), squeeze=False)
    for b, L in enumerate(LOCS):
        for i, u in enumerate(picks[L]):
            r = b * n_units + i
            maps_row = [m[u] for m in series]
            vmax = np.nanmax([np.nanmax(m) for m in maps_row])
            vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
            for c, m in enumerate(maps_row):
                ax = axes[r][c]
                tf._draw_map(ax, m, vmax=vmax)
                ax.plot(L[0] - 1, L[1] - 1, "w+", ms=7, mew=1.5)
                if r == 0:
                    ax.set_title(labels[c], fontsize=9, pad=4)
                # The block's OWN run: the only column where a trace could show.
                if c == b + 1:
                    for side in ("top", "bottom", "left", "right"):
                        ax.spines[side].set_visible(True)
                        ax.spines[side].set(color="#C0392B", linewidth=2.0)
            axes[r][0].set_ylabel(f"#{u}\np{pcts[L][i]:.0f}  Δg {dgs[L][i]:+.2f}", fontsize=8,
                                  rotation=0, labelpad=30, va="center")

    fig.suptitle(
        "Object-trace null as spatial tuning. Each block takes ITS own location's top-ranked units and\n"
        "marks ITS own location (white +). Rank = the object cell's change in field gain as a PERCENTILE\n"
        "among that unit's change at all 172 walkable cells - the statistic the null test itself uses\n"
        f"(>{OBJECT_CELL_PCTILE:.0f} = object cell). Row labels give that percentile and the raw Δfield gain.\n"
        "The RED-OUTLINED panel is the block's own run, the only one in which a trace at that + could appear.",
        fontsize=10.5)
    fig.tight_layout(rect=(0.10, 0, 1, 0.945))

    # Block labels and separators are placed from the LAID-OUT axes, not from
    # row arithmetic: tight_layout and the suptitle mean row index and figure
    # fraction are not the same thing (the first attempt drew the separators
    # through the middle of rows).
    for b, L in enumerate(LOCS):
        top = axes[b * n_units][0].get_position().y1
        bot = axes[(b + 1) * n_units - 1][0].get_position().y0
        fig.text(0.045, (top + bot) / 2, f"selected at {L}", ha="center", va="center",
                 fontsize=11, fontweight="bold", rotation=90)
        if b:
            prev = axes[b * n_units - 1][0].get_position().y0
            fig.add_artist(plt.Line2D([0.02, 0.99], [(prev + top) / 2] * 2,
                                      color="k", lw=1.2, ls="--"))
    fig.savefig(OUT / "fig_trace_3loc.png", dpi=150, bbox_inches="tight")
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
