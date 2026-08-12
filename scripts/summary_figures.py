"""Figures for the 2026-08-12 object/trace-cell result summary.

Every figure here either compares against a baseline/control or shows spatial
tuning; none is a bare time series.

    uv run --no-sync python scripts/summary_figures.py            # all
    uv run --no-sync python scripts/summary_figures.py occlusion  # one

Writes to `outputs/summary/`. Caches consumed:
    outputs/trace/seq4_matrix.npy    (8 seeds, 4 phases, 3 locs) hidden-state object-cell frac
    outputs/trace/seq4_readout.npy   (8 seeds, 4 phases, 3 locs) readout green change
    outputs/summary/wandb_entropy.npz  policy/loc entropy traces (see `fetch_entropy`)

Two figures are hardcoded from documented results because no cache survives:
the occlusion location-control matrices (source cited in-file) and the quadrant
decode (source cited in-file). Both cite the doc and script they came from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

OUT = Path("outputs/summary")
TRACE = Path("outputs/trace")

# Validated categorical/sequential/diverging palette (dataviz skill reference instance).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef5fd", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
)
DIV_BR = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#5598e7", "#cde2fb", "#f0efec", "#f6c3c2", "#e34948", "#8f1f1e"]
)
INK, INK2 = "#0b0b0b", "#52514e"


def _annot_heatmap(
    *,
    ax: plt.Axes,
    data: Float[np.ndarray, "rows cols"],
    row_labels: list[str],
    col_labels: list[str],
    norm: matplotlib.colors.Normalize,
    cmap: matplotlib.colors.Colormap,
    fmt: str = "{:.4f}",
    highlight: list[tuple[int, int]] | None = None,
) -> matplotlib.image.AxesImage:
    """Draw an annotated heatmap; `highlight` cells get a black outline."""
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            rgba = cmap(norm(data[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if lum < 0.55 else INK)
    for i, j in highlight or []:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=2.2))
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def occlusion() -> None:
    """The location-control matrices that killed the scenario-C false positive.

    Numbers are documented results, not caches (no result cache survives):
      left  - docs/exp_trace_cell_scenarios_2026-08-11.md section 2.5 (non-occluded, pre-existing runs)
      right - docs/exp_trace_cell_scenarios_2026-08-11.md section 5 "Result at n=8" (object at (14,7), n=8)
              and docs/compaction.md line 61 (object at (12,7), n=3: 0.0835 vs 0.0441)
    """
    # rows = run's object position, cols = cell scored.
    nonoccl = np.array([[0.0321, 0.0902, 0.0200],
                        [0.0140, 0.1002, 0.0501],
                        [0.0581, 0.0721, 0.0341]])
    nonoccl_rows = ["object at\n(7,11)", "object at\n(14,7)", "object at\n(7,2)"]
    nonoccl_cols = ["(7,11)", "(14,7)", "(7,2)"]

    occl = np.array([[0.0892, 0.0433],
                     [0.0835, 0.0441]])
    occl_rows = ["object at\n(14,7)  n=8", "object at\n(12,7)  n=3"]
    occl_cols = ["(14,7)", "(12,7)"]

    vmax = 0.105
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)  # shared across both panels
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2),
                             gridspec_kw={"width_ratios": [3, 2.15]})
    fig.subplots_adjust(top=0.70, wspace=0.62, right=0.86)

    _annot_heatmap(ax=axes[0], data=nonoccl, row_labels=nonoccl_rows, col_labels=nonoccl_cols,
                   norm=norm, cmap=SEQ_BLUE, highlight=[(0, 0), (1, 1), (2, 2)])
    axes[0].set_title("L-room, no occlusion (n=1 per row)\n"
                      "column (14,7) is highest in EVERY row", fontsize=10.5, color=INK)
    im = _annot_heatmap(ax=axes[1], data=occl, row_labels=occl_rows, col_labels=occl_cols,
                        norm=norm, cmap=SEQ_BLUE, highlight=[(0, 0), (1, 1)])
    axes[1].set_title("Occluded room, scenario C\n"
                      "moving the object does NOT move the peak", fontsize=10.5, color=INK)

    for ax in axes:
        ax.set_xlabel("cell scored", color=INK2)
    axes[0].set_ylabel("run (where the object actually was)", color=INK2)

    cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.04)
    cb.set_label("frac. units called object cells\n(shared scale, both panels)",
                 fontsize=9, color=INK2)
    cb.set_ticks([0.0, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10],
                 labels=["0.00", "0.02", "0.04", "0.05  ← chance", "0.06", "0.08", "0.10"])
    cb.ax.tick_params(labelsize=8)
    cb.ax.axhline(0.05, color=INK, lw=1.6)
    cb.outline.set_visible(False)

    fig.suptitle("Location control: real object coding must MOVE with the object.\n"
                 "Black outline = the cell where the object actually was for that run.",
                 fontsize=12.5, color=INK, y=0.98)
    fig.savefig(OUT / "fig_occlusion_control.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_occlusion_control.png")


def sequential() -> None:
    """Sequential displacement A->B->C->removed: hidden state vs readout, n=8."""
    hid: Float[np.ndarray, "8 4 3"] = np.load(TRACE / "seq4_matrix.npy")
    rdo: Float[np.ndarray, "8 4 3"] = np.load(TRACE / "seq4_readout.npy")
    H, R = hid.mean(0), rdo.mean(0)

    locs = ["(7,11)", "(7,2)", "(4,7)"]
    phases = ["ph0  obj @ (7,11)", "ph1  obj @ (7,2)", "ph2  obj @ (4,7)", "ph3  REMOVED"]
    present = [(0, 0), (1, 1), (2, 2)]  # phase i has the object at locs[i]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    n_h = TwoSlopeNorm(vmin=0.02, vcenter=0.05, vmax=0.08)
    im_h = _annot_heatmap(ax=axes[0], data=H, row_labels=phases, col_labels=locs,
                          norm=n_h, cmap=DIV_BR, highlight=present)
    axes[0].set_title("HIDDEN STATE — fraction of units with a field there\n"
                      "every cell is BELOW the 0.05 chance rate (blue)", fontsize=10.5, color=INK)
    cb_h = fig.colorbar(im_h, ax=axes[0], fraction=0.046, pad=0.03)
    cb_h.set_label("frac. object cells", fontsize=9, color=INK2)
    cb_h.set_ticks([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
                   labels=["0.02", "0.03", "0.04", "0.05  ← chance", "0.06", "0.07", "0.08"])
    cb_h.ax.tick_params(labelsize=8)
    cb_h.ax.axhline(0.05, color=INK, lw=1.8)
    cb_h.outline.set_visible(False)

    lim = float(np.abs(R).max()) * 1.05
    n_r = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    im_r = _annot_heatmap(ax=axes[1], data=R, row_labels=phases, col_labels=locs,
                          norm=n_r, cmap=DIV_BR, highlight=present, fmt="{:+.4f}")
    axes[1].set_title("READOUT — green change at that cell's own view cell\n"
                      "(4,7) is already elevated at ph0/ph1, before the object arrives",
                      fontsize=10.5, color=INK)
    cb_r = fig.colorbar(im_r, ax=axes[1], fraction=0.046, pad=0.03)
    cb_r.set_label("Δ green prediction", fontsize=9, color=INK2)
    cb_r.ax.axhline(0.0, color=INK, lw=1.8)
    cb_r.outline.set_visible(False)

    # Call out the transfer-not-trace cell explicitly.
    axes[1].add_patch(plt.Rectangle((1.55, -0.45), 0.9, 1.9, fill=False, ec=ORANGE, lw=2.6, ls="--"))
    axes[1].text(0.5, -0.26, "dashed box: (4,7) reads +0.0395 BEFORE the object ever arrives "
                             "(t=9.57, p<1e-4),\nvs +0.0445 while it is there — ~89% of the "
                             "signal is generalisation from elsewhere",
                 transform=axes[1].transAxes, ha="center", va="top",
                 fontsize=8.5, color=ORANGE)

    for ax in axes:
        ax.set_xlabel("cell scored", color=INK2)
    axes[0].set_ylabel("training phase", color=INK2)

    fig.suptitle("Sequential displacement, n=8 seeds. Black outline = object present at that "
                 "cell in that phase.\nSame data, two measurements: the hidden state never "
                 "codes the object; the readout does — but non-specifically.",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "fig_seq_hidden_vs_readout.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_seq_hidden_vs_readout.png")


def quadrant() -> None:
    """Quadrant decode from `h`, with vs without the object, square room.

    Source: `scripts/moser_decode_quadrant.py` (prints; no cache written),
    values as recorded in docs/sab_context/goal_2026-08-12.md
    "Why the symmetric room did not bind".
    """
    sessions = [1, 2, 3, 4, 5, 6]
    objects = ["(5,4)", "(10,6)", "(6,11)", "(12,3)", "(3,9)", "(11,12)"]
    with_obj = np.array([0.806, 0.838, 0.830, 0.848, 0.846, 0.828])
    without = np.array([0.805, 0.838, 0.832, 0.845, 0.847, 0.827])

    x = np.arange(len(sessions))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.bar(x - w / 2, with_obj, w, label="object present", color=BLUE)
    ax.bar(x + w / 2, without, w, label="object absent (control)", color=ORANGE)

    for xi, (a, b) in enumerate(zip(with_obj, without)):
        ax.text(xi - w / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=8.5, color=INK)
        ax.text(xi + w / 2, b + 0.012, f"{b:.3f}", ha="center", fontsize=8.5, color=INK)
        ax.text(xi, 0.30, f"Δ {a - b:+.3f}", ha="center", fontsize=8.5, color=INK2)

    ax.axhline(0.25, color="#e34948", ls="--", lw=1.8, zorder=3)
    ax.text(0.5, 0.25, "chance = 0.25", fontsize=9, color="#e34948", ha="center", va="center",
            zorder=4, bbox=dict(fc="white", ec="none", pad=2.0))
    ax.set_xticks(x, [f"s{s}\n{o}" for s, o in zip(sessions, objects)])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("quadrant decode accuracy from h\n(trajectory-level CV)", color=INK2)
    ax.set_xlabel("session (object position)", color=INK2)
    ax.set_title("Symmetric room: the hidden state localises to ~84% with NO object at all.\n"
                 "The object adds nothing — every Δ is within ±0.003.", fontsize=11.5, color=INK)
    ax.legend(frameon=False, loc="upper left", ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e2", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_quadrant_decode.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_quadrant_decode.png")


def trace_null() -> None:
    """Trace-cell counts against the EMPIRICAL null, not the flat 5% line.

    The trace criterion is a cumulative OR over every previously-used object position, so the
    number of chances to score a hit grows with session index and a flat 5% null is wrong.
    `outputs/moser/fig_moser_counts.png` shows the flat-5% version, which is what produced the
    retracted "17% trace cells, p<1e-4" claim.

    observed        - outputs/moser/moser_summary.json (frac_trace), sessions 2-7
    empirical null  - identical criterion on six positions the object NEVER occupied, 400 draws;
                      recorded in docs/sab_context/goal_2026-08-12.md "Trace cells: NULL"
    """
    import json

    summary = json.load(open("outputs/moser/moser_summary.json"))
    sess = [s["session"] for s in summary if s["session"] >= 2]
    observed = np.array([s["frac_trace"] for s in summary if s["session"] >= 2])
    null = np.array([0.037, 0.066, 0.106, 0.121, 0.154, 0.203])
    n_prev = [1, 2, 3, 4, 5, 6]

    x = np.arange(len(sess))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    ax.bar(x - w / 2, observed, w, label="observed trace cells", color=BLUE)
    ax.bar(x + w / 2, null, w, label="empirical null (object never there)", color=ORANGE)
    ax.axhline(0.05, color=INK2, ls="--", lw=1.5, zorder=3)
    ax.text(-0.42, 0.056, "the WRONG null: flat 5%", fontsize=8.5, color=INK2)

    for xi, (o, nu) in enumerate(zip(observed, null)):
        ax.text(xi - w / 2, o + 0.004, f"{o:.3f}", ha="center", fontsize=8.5, color=INK)
        ax.text(xi + w / 2, nu + 0.004, f"{nu:.3f}", ha="center", fontsize=8.5, color=INK)

    ax.set_xticks(x, [f"s{s}\n{k} prior\nposition{'s' if k > 1 else ''}"
                      for s, k in zip(sess, n_prev)])
    ax.set_ylim(0, 0.235)
    ax.set_ylabel("fraction of the 500 hidden units\ncalled trace cells", color=INK2)
    ax.set_xlabel("session (number of previously-used object positions the criterion ORs over)",
                  color=INK2)
    ax.set_title("Trace cells are counting CHANCES, not traces.\n"
                 "The null rises with session index too — and is above the observed count "
                 "at every session.", fontsize=11.5, color=INK)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e2", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_moser_trace_null.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_moser_trace_null.png")


def collapse() -> None:
    """Policy / occupancy entropy per session vs the L-room reference band."""
    d = np.load(OUT / "wandb_entropy.npz")
    ses = [k.split("|")[0] for k in d.files if k.endswith("|policy_entropy|val")]
    moser = sorted(n for n in ses if n.startswith("moser-s"))
    ref = next(n for n in ses if n.startswith("pRNN_curious"))
    objects = ["none", "(5,4)", "(10,6)", "(6,11)", "(12,3)", "(3,9)", "(11,12)", "none"]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True)
    specs = [("policy_entropy", "policy entropy (bits)", 0, 2.15),
             ("loc_entropy", "occupancy entropy (bits)", 3.5, 7.6)]

    for ax, (key, ylab, ylo, yhi) in zip(axes, specs):
        r = d[f"{ref}|{key}|val"]
        lo, hi = np.percentile(r, [10, 90])
        ax.axhspan(lo, hi, color=AQUA, alpha=0.16, zorder=0)
        ax.text(0.996, hi, f"L-room reference, p10–p90  ({lo:.2f}–{hi:.2f})",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=8.5, color="#0e7a55")

        off = 0.0
        for i, n in enumerate(moser):
            v = d[f"{n}|{key}|val"]
            xs = off + np.arange(len(v)) / len(v)
            ax.plot(xs, v, lw=1.0, color=BLUE if i < 4 else ORANGE, alpha=0.55)
            k = max(1, len(v) // 25)
            sm = np.convolve(v, np.ones(k) / k, mode="valid")
            ax.plot(xs[: len(sm)], sm, lw=2.0, color=BLUE if i < 4 else ORANGE)
            if i % 2:
                ax.axvspan(off, off + 1, color="#f4f3f0", zorder=-1)
            off += 1.0

        if key == "policy_entropy":
            ax.axhline(2.0, color=INK2, ls=":", lw=1.4)
            ax.text(0.004, 2.02, "maximum = log2(4) = 2.0 bits", transform=ax.get_yaxis_transform(),
                    fontsize=8.5, color=INK2)
        ax.axvline(4.0, color="#e34948", lw=2.0, ls="--")
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(0, 8)
        ax.set_ylabel(ylab, color=INK2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e2", lw=0.8)
        ax.set_axisbelow(True)

    axes[0].text(4.06, 1.78, "collapse begins (session 4)", fontsize=9.5, color="#e34948")
    axes[0].plot([], [], color=BLUE, lw=2.0, label="sessions 0–3 (healthy)")
    axes[0].plot([], [], color=ORANGE, lw=2.0, label="sessions 4–7 (collapsed)")
    axes[0].fill_between([], [], color=AQUA, alpha=0.16, label="L-room reference band")
    axes[0].legend(frameon=False, loc="lower left", ncols=3, fontsize=9)

    axes[1].set_xticks(np.arange(8) + 0.5,
                       [f"s{i}\n{o}" for i, o in enumerate(objects)])
    axes[1].set_xlabel("session (object position) — each session normalised to its own length",
                       color=INK2)
    fig.suptitle("Square-room policy collapse: the confound that makes sessions 4–6 uninformative.\n"
                 "Objects (12,3), (3,9), (11,12) were presented exactly while the agent had "
                 "stopped exploring.", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig_policy_collapse.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_policy_collapse.png")


def fetch_entropy() -> None:
    """Refresh `outputs/summary/wandb_entropy.npz` from wandb (needs network)."""
    import wandb

    api = wandb.Api()
    runs = {r.name: r for r in api.runs("blake-richards/curious-george",
                                        filters={"display_name": {"$regex": "^moser-s"}})}
    runs.update({r.name: r for r in api.runs(
        "blake-richards/curious-george",
        filters={"display_name": "pRNN_curious_26-07-23-10-06-25"})})
    out: dict[str, np.ndarray] = {}
    for name, r in sorted(runs.items()):
        for k in ("policy_entropy", "loc_entropy", "pRNN loss"):
            h = r.history(keys=[k], samples=5000, pandas=True)
            if len(h) == 0:
                continue
            out[f"{name}|{k}|step"] = h["_step"].to_numpy(float)
            out[f"{name}|{k}|val"] = h[k].to_numpy(float)
    np.savez(OUT / "wandb_entropy.npz", **out)
    print(f"wrote wandb_entropy.npz ({len(out)} arrays)")


ALL = {"occlusion": occlusion, "sequential": sequential, "quadrant": quadrant,
       "trace_null": trace_null, "collapse": collapse, "fetch_entropy": fetch_entropy}
DEFAULT = ["occlusion", "sequential", "quadrant", "trace_null", "collapse"]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or DEFAULT:
        ALL[name]()
