"""Behaviour through the sequential displacement A -> B -> C -> REMOVED.

Reads `outputs/trace/seq_behavior.npz` (4 phases x 3 locations x 8 seeds,
occupancy/in-view percentiles and raw occupancy) and writes
`outputs/summary/fig_seq_behavior.png`.

Three panels, because the headline and the control disagree and both have to be
visible:
  A  the location-control matrix. Rows are where the object WAS, columns the
     location scored. The diagonal being highest is necessary, not sufficient -
     it is the same reasoning that produced the (14,7) false positive.
  B  the EXCESS: value at L when the object is at L, minus the mean at L when it
     is anywhere else, so pure location bias cancels. Per-seed points shown.
  C  the departure test: does the agent linger where the object last was?

    uv run python scripts/trace/seq_behavior_figure.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

CACHE = Path("outputs/trace/seq_behavior.npz")
OUT = Path("outputs/summary/fig_seq_behavior.png")
PHASES = ["0: object at (7,11)", "1: object at (7,2)", "2: object at (4,7)", "3: REMOVED"]
LOCS = [(7, 11), (7, 2), (4, 7)]
OWN_PHASE = {0: 0, 1: 1, 2: 2}          # location index -> the phase it owns


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(CACHE)
    occ, raw = d["occ"], d["raw"]        # (phase, loc, seed)
    n_seeds = occ.shape[2]
    mean = occ.mean(axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9),
                             gridspec_kw={"width_ratios": [1.25, 1, 0.85], "wspace": 0.52})

    # --- A: location-control matrix -------------------------------------
    ax = axes[0]
    im = ax.imshow(mean, cmap="magma", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(3), [str(l) for l in LOCS])
    ax.set_yticks(range(4), PHASES, fontsize=8.5)
    ax.set_xlabel("location scored")
    ax.set_title("A · occupancy percentile\n(50 = a typical cell of 172)", fontsize=10.5)
    for p in range(4):
        for j in range(3):
            own = OWN_PHASE.get(j) == p
            ax.text(j, p, f"{mean[p, j]:.0f}" + ("\n★" if own else ""),
                    ha="center", va="center", fontsize=10,
                    color="white" if mean[p, j] < 60 else "black",
                    fontweight="bold" if own else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("percentile", fontsize=9)

    # --- B: excess, with the per-seed points ----------------------------
    ax = axes[1]
    for j, loc in enumerate(LOCS):
        p_own = OWN_PHASE[j]
        own = occ[p_own, j, :]
        other = occ[[p for p in range(4) if p != p_own], j, :].mean(axis=0)
        exc = own - other
        t, pv = stats.ttest_rel(own, other)
        ax.scatter(np.full(n_seeds, j) + np.random.default_rng(0).normal(0, .045, n_seeds),
                   exc, s=26, color="#0F5257", alpha=.65, zorder=3,
                   label="individual seeds" if j == 0 else None)
        ax.plot([j - .28, j + .28], [exc.mean()] * 2, color="#C0392B", lw=2.6, zorder=4,
                label="mean" if j == 0 else None)
        ax.annotate(f"{exc.mean():+.1f}\np={pv:.3f}", xy=(j, exc.max()),
                    xytext=(0, 12), textcoords="offset points", ha="center",
                    fontsize=9, color="#C0392B", fontweight="bold")
    ax.axhline(0, color="0.35", lw=1, ls="--", zorder=1, label="no excess (null)")
    ax.set_xticks(range(3), [str(l) for l in LOCS])
    ax.set_ylabel("excess percentile points")
    ax.set_title("B · location control: excess\nat L when object at L, minus when elsewhere",
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(top=ax.get_ylim()[1] + 22)

    # --- C: the departure test ------------------------------------------
    ax = axes[2]
    during, after = raw[2, 2, :], raw[3, 2, :]
    for a, b in zip(during, after):
        ax.plot([0, 1], [a, b], color="0.6", lw=1, marker="o", ms=4, zorder=2)
    ax.plot([0, 1], [during.mean(), after.mean()], color="#C0392B", lw=3, marker="o",
            ms=8, zorder=3, label="mean")
    t, pv = stats.ttest_rel(during, after)
    ax.set_xticks([0, 1], ["object AT (4,7)\n(phase 2)", "object REMOVED\n(phase 3)"],
                  fontsize=9)
    ax.set_ylabel("raw occupancy fraction at (4,7)")
    ax.set_title(f"C · does it linger where the object was?\n"
                 f"{during.mean():.3f} → {after.mean():.3f}  "
                 f"({during.mean()/max(after.mean(),1e-9):.1f}×), "
                 f"{int((after < during).sum())}/{n_seeds} seeds", fontsize=10.5)
    ax.legend(fontsize=8)

    fig.suptitle("Behaviour through the sequential displacement: it follows the PRESENT object,\n"
                 "abandons the departed one — but only ONE location survives its control",
                 fontsize=12.5, y=0.99)
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.075, right=0.985)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
