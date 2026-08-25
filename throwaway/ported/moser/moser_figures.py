"""Figures for the Moser object/trace-cell replication.

Consumes outputs/moser/moser_maps.npz (scripts/moser_analysis.py) and the
wandb reference curve, and writes to outputs/moser/.

    uv run python scripts/moser_figures.py          # all
    uv run python scripts/moser_figures.py panel    # one

Figures
  panel    the replication of docs/ref-trace-cells.png: one row per example
           unit, one column per session, occupancy-masked rate maps with the
           object position ringed. Peak rate is printed above each map, as in
           the paper. Heatmaps share one colour scale WITHIN a row so the
           comparison across sessions is honest (a per-panel scale would make
           every session look identical).
  counts   object cells and trace cells per session against the 5% chance rate.
  gain     field gain at the CURRENT object position vs at positions the object
           has already left - the quantity that separates the two cell classes.
  drift    map stability across sessions: is the place code even changing?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/moser")
CHANCE = 0.05


def _load():
    d = np.load(OUT / "moser_maps.npz")
    objs = [None if tuple(o) == (-1, -1) else tuple(int(v) for v in o) for o in d["objects"]]
    return d, objs


def _labels(objs):
    return ["no object" if o is None else f"({o[0]},{o[1]})" for o in objs]


def _ring(ax, pos, env_minmax=(0.5, 14.5), nbins=14, color="white"):
    """Mark the object position on a map drawn with imshow(origin='lower')."""
    if pos is None:
        return
    lo, hi = env_minmax
    x = (pos[0] - lo) / (hi - lo) * nbins - 0.5
    y = (pos[1] - lo) / (hi - lo) * nbins - 0.5
    import matplotlib.pyplot as plt
    ax.add_patch(plt.Circle((x, y), 1.6, fill=False, ec=color, lw=1.8))


def panel(n_units: int = 6) -> None:
    """Rate maps: units x sessions, mirroring docs/ref-trace-cells.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, objs = _load()
    maps, trace_cells, object_cells = d["maps"], d["trace_cells"], d["object_cells"]
    S = maps.shape[0]

    # Prefer units ever called a trace cell, then object cells - but only among
    # units that actually fire. A silent unit has an all-zero map and would
    # otherwise win ties and fill the figure with blank panels.
    peak = np.nanmax(maps, axis=(0, 2, 3))
    alive = peak > np.nanpercentile(peak, 25)
    score = (trace_cells.sum(0) * 100 + object_cells.sum(0)).astype(float)
    score[~alive] = -1
    units = np.argsort(-score)[:n_units]

    fig, axes = plt.subplots(len(units), S, figsize=(1.55 * S, 1.75 * len(units)),
                             squeeze=False)
    labels = _labels(objs)
    for r, u in enumerate(units):
        row = maps[:, u]                       # (S, ny, nx)
        # One scale per row so sessions are comparable, but spanning the row's
        # actual range: anchoring vmin at 0 washes out any unit whose rate is
        # high and near-uniform, which is most of them.
        vmin, vmax = np.nanpercentile(row, 2), np.nanpercentile(row, 99.5)
        if not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, max(float(np.nanmax(row)), 1e-6)
        for k in range(S):
            ax = axes[r][k]
            ax.imshow(row[k], origin="lower", vmin=vmin, vmax=vmax,
                      cmap="jet", interpolation="nearest")
            _ring(ax, objs[k])
            ax.set_xticks([]); ax.set_yticks([])
            peak = np.nanmax(row[k])
            tags = []
            if object_cells[k, u]: tags.append("O")
            if trace_cells[k, u]: tags.append("T")
            ax.set_title(f"{peak:.2f}" + ("  " + "/".join(tags) if tags else ""),
                         fontsize=7, pad=2,
                         color=("crimson" if tags else "black"))
            if r == 0:
                ax.text(0.5, 1.55, labels[k], transform=ax.transAxes,
                        ha="center", fontsize=8)
            if k == 0:
                ax.set_ylabel(f"unit {u}", fontsize=8)
    fig.suptitle("pRNN hidden units across the Moser session sequence\n"
                 "white ring = object position;  O = object cell, T = trace cell;  "
                 "number = peak rate (shared colour scale within a row)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig_moser_panel.png", dpi=170, bbox_inches="tight")
    print("wrote fig_moser_panel.png")


def counts() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, objs = _load()
    oc = d["object_cells"].mean(axis=1)
    tc = d["trace_cells"].mean(axis=1)
    x = np.arange(len(objs))

    fig, ax = plt.subplots(figsize=(1.5 * len(objs) + 2, 4))
    ax.bar(x - 0.19, oc, 0.38, label="object cells (field at current object)", color="#2b6cb0")
    ax.bar(x + 0.19, tc, 0.38, label="trace cells (field at a departed object)", color="#c05621")
    ax.axhline(CHANCE, ls="--", c="k", lw=1,
               label=f"chance = {CHANCE:.0%} (within-unit 95th percentile)")
    ax.set_xticks(x); ax.set_xticklabels(_labels(objs), rotation=30, ha="right")
    ax.set_xlabel("session (object position)")
    ax.set_ylabel("fraction of the 500 hidden units")
    ax.set_title("Object cells and trace cells per session")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_moser_counts.png", dpi=170, bbox_inches="tight")
    print("wrote fig_moser_counts.png")


def gain() -> None:
    """Mean field gain at the current object position vs departed ones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, objs = _load()
    gains, cells = d["gains"], [tuple(c) for c in d["cells"]]
    idx = {c: i for i, c in enumerate(cells)}

    cur, past, other = [], [], []
    for k, o in enumerate(objs):
        g = gains[k]                                    # (n_cells, H)
        prev = [p for p in objs[:k] if p is not None and p != o]
        cur.append(np.nanmean(g[idx[o]]) if o else np.nan)
        past.append(np.nanmean([np.nanmean(g[idx[p]]) for p in prev]) if prev else np.nan)
        used = {o, *prev} - {None}
        rest = [i for c, i in idx.items() if c not in used]
        other.append(np.nanmean(g[rest]))

    x = np.arange(len(objs))
    fig, ax = plt.subplots(figsize=(1.5 * len(objs) + 2, 4))
    ax.plot(x, cur, "o-", c="#2b6cb0", label="at the CURRENT object position")
    ax.plot(x, past, "s-", c="#c05621", label="at positions the object has LEFT")
    ax.plot(x, other, "^--", c="grey", label="all other walkable cells (baseline)")
    ax.set_xticks(x); ax.set_xticklabels(_labels(objs), rotation=30, ha="right")
    ax.set_xlabel("session (object position)")
    ax.set_ylabel("mean field gain  (disc rate / unit mean rate)")
    ax.set_title("Where the population puts its fields, session by session")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_moser_gain.png", dpi=170, bbox_inches="tight")
    print("wrote fig_moser_gain.png")


def drift() -> None:
    """Per-unit map correlation between consecutive sessions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, objs = _load()
    maps = d["maps"]
    rs = []
    for k in range(1, maps.shape[0]):
        a, b = maps[k - 1], maps[k]
        ok = np.isfinite(a) & np.isfinite(b)
        r = []
        for u in range(a.shape[0]):
            m = ok[u]
            if m.sum() > 10:
                r.append(np.corrcoef(a[u][m], b[u][m])[0, 1])
        rs.append(np.array(r))

    fig, ax = plt.subplots(figsize=(1.5 * len(rs) + 2, 4))
    ax.boxplot(rs, labels=[f"{k}->{k+1}" for k in range(len(rs))], showfliers=False)
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(1.0, ls=":", c="grey", lw=1)
    ax.set_xlabel("session transition")
    ax.set_ylabel("per-unit map correlation")
    ax.set_title("Does the place code change between sessions at all?\n"
                 "(r near 1 everywhere would mean the object changed nothing)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_moser_drift.png", dpi=170, bbox_inches="tight")
    print("wrote fig_moser_drift.png")


FIGURES = {"panel": panel, "counts": counts, "gain": gain, "drift": drift}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or list(FIGURES)
    for name in which:
        try:
            FIGURES[name]()
        except Exception as e:  # one bad figure must not lose the rest
            print(f"FAILED {name}: {type(e).__name__}: {e}")
