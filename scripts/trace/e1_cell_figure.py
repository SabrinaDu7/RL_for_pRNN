"""E1 as SPATIAL TUNING: the top object / object-vector candidates, per checkpoint.

`e1_multi_anchor.py` reports population fractions. A fraction cannot show you
whether a field recurs at the same offset from each landmark, which is the whole
claim - so this is the Hoydal Fig. 2a view (`docs/ref-objectVector-cells.png`):
the room-frame rate map with every landmark centre ringed in white, so a genuine
object-vector cell reads as one blob at the same displacement from each ring.

Each row is one unit:

    col 0        room-frame rate map, occupancy-masked, landmarks ringed
    cols 1..n    the anchor-centred offset maps that col 0 is cropped into,
                 one per landmark, on a shared scale within the row.
                 An OVC has the SAME bump in all of them; a place cell has a
                 bump in exactly one; an object cell has a bump at the centre
                 of all of them.

⚠️ This figure applies the RATE screen (peak_ratio >= 1.5) but NOT the
spatial-information screen that `ovc_eval.e1` also imposes, so its population
fraction is close to but not bit-identical to that path's. Stated because two
routes reporting "the E1 fraction" must not silently differ.

RANKING (see the module docstring of ovc_metric.py for the maths):
  vector_percentile   the unit's real-anchor vector score against ITS OWN null
                      of random anchor triples. Chance = 5% over the 95th, by
                      construction. This is the ranking.
  radial_score        variance explained by |offset| alone - splits the two
                      classes: high = object cell (fires near ANY anchor, no
                      direction), low = object-VECTOR cell (directional).
  peak_ratio >= 1.5   rate screen, standing in for Hoydal's >= 2 Hz.

    uv run python scripts/trace/e1_cell_figure.py --ckpt <f.pt> --label "3-room L-room"
    uv run python scripts/trace/e1_cell_figure.py --ckpt <f.pt> --env-config squareroom_multi \\
        --layouts rooms --label "3-room square"

Writes outputs/summary/fig_e1_cells_<tag>.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.trace import ovc_metric as om

OUT = Path("outputs/summary")
RADIUS = 4

# Landmarks are drawn as WHAT THEY ARE, not as identical dots. Two reasons, and
# the second is the whole point of the figure: distinct glyphs make the three
# offset maps trivially attributable, and when a room's triad is a ROTATION of
# another's, a genuine object-vector field must rotate with it - which is
# invisible if every landmark is the same white circle.
MARKER = {"x": "X", "plus": "P", "block3": "s", "block2": "s",
          "x6": "X", "plus6": "P", "triangle6": "^"}


def env_landmarks(env):
    """The `Landmark` objects a room actually contains, in SHAPES order.

    Read from the env rather than recovered from the grid by colour: colour
    order is arbitrary and differs between rooms, so a colour-sorted anchor list
    silently relabels which landmark is "first" from room to room - fatal for a
    figure whose point is comparing the same landmark across rooms.
    """
    u = env.env.unwrapped if hasattr(env, "env") else env.unwrapped
    return list(u.landmarks or u._default_landmarks(u.width, u.height))


def rank(*, maps, env, anchors, n_null: int, seed: int = 0, spatial_ok=None) -> dict:
    """Per-unit scores plus the offset maps, ready to plot.

    `spatial_ok` is `ovc_metric.spatial_screen` - pass it to match `ovc_eval.e1`
    exactly. Passing None reports the rate-screen-only population, which is a
    LARGER denominator and therefore a different fraction; both are printed so
    the two can never be confused for each other.
    """
    offsets = om.offset_grid(env=env, anchors=anchors, radius=RADIUS)
    omaps = om.offset_maps(maps=maps, anchors=anchors, offsets=offsets)
    pct, real = om.vector_percentile(maps=maps, env=env, anchors=anchors,
                                     offsets=offsets, n_null=n_null, seed=seed)
    rs, pr = om.radial_score(omaps=omaps, offsets=offsets), om.peak_ratio(omaps=omaps)

    # Where the offset map peaks, pooled over anchors: distance 0-1 is the
    # object-cell class (instructions-objectAndOVC.md 1.2), further out is the
    # displaced field an object-VECTOR cell must have (OVC-1).
    pooled = np.nanmean(omaps, axis=0)                       # (H, n_off)
    peak_off = np.full((maps.shape[0], 2), np.nan)
    for u in range(maps.shape[0]):
        v = pooled[u]
        if np.isfinite(v).any():
            peak_off[u] = offsets[int(np.nanargmax(v))]
    d = np.abs(peak_off).max(axis=1)

    rate_ok = np.isfinite(pct) & (pr >= 1.5)
    screen = rate_ok if spatial_ok is None else (rate_ok & spatial_ok)
    order = np.lexsort((-np.nan_to_num(real), -np.where(screen, np.nan_to_num(pct, nan=-np.inf),
                                                        -np.inf)))
    return {"offsets": offsets, "omaps": omaps, "pct": pct, "vector_score": real,
            "radial": rs, "peak_ratio": pr, "peak_offset": peak_off, "peak_dist": d,
            "screen": screen, "rate_ok": rate_ok, "order": order}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env-config", default=None,
                    choices=(None, "lroom_multi", "squareroom_multi"),
                    help="omit for the plain single-room L-room with its default landmarks")
    ap.add_argument("--layouts", default="rooms", choices=("one", "rooms", "pool"))
    ap.add_argument("--room", type=int, default=0)
    ap.add_argument("--label", required=True, help="what this checkpoint IS, for the title")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-units", type=int, default=3)
    ap.add_argument("--n-dirs", type=int, default=1)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--si-screen", action="store_true",
                    help="also require SI above the trajectory-shuffle null's 95th pct, "
                         "matching ovc_eval.e1's denominator exactly (costs ~50 map builds)")
    ap.add_argument("--gallery", type=int, default=16,
                    help="also write a panel of the top-N units by spatial information "
                         "(0 disables). This is the POSITIVE CONTROL: if these are not clean "
                         "place fields, the object-coding null is a broken measurement, not a bound")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.trace.ovc_eval import activity, maps_from

    if a.env_config:
        from curious_george.envs.layouts import resolve_layouts
        from scripts.multienv.checkpoint_curve import run_config

        cfg = run_config(env_cfg=a.env_config, layouts=a.layouts, hiddensize=500)
        rooms = resolve_layouts(cfg)
        home = rooms[a.room]
        env_name, landmarks = cfg.exp.env_name, home.landmarks
        room_desc = f"room {a.room} ({home.key}): {home.describe()}"
    else:
        env_name, landmarks, room_desc = "MiniGrid-LRoom-v0", None, "default 6x6 landmarks"

    h, pos, env = activity(env_name=env_name, ckpt_dir=a.ckpt, n_dirs=a.n_dirs,
                           landmarks=landmarks)
    maps = maps_from(h=h, pos=pos, env=env)
    lms = env_landmarks(env)
    anchors = [(int(round(l.centroid[0])), int(round(l.centroid[1]))) for l in lms]
    print(f"{a.label}: {env_name}, {room_desc}")
    print(f"  landmarks {[(l.shape, l.color, an) for l, an in zip(lms, anchors)]}"
          f"   probe {h.shape[0]}x{h.shape[1]}")

    sok = None
    if a.si_screen:
        from scripts.trace.ovc_eval import ONSET
        sok = om.spatial_screen(h=h[:, ONSET:, :], pos=pos[:, ONSET:, :], maps=maps, env=env)
        print(f"  spatial screen keeps {int(sok.sum())}/{maps.shape[0]} units")
    r = rank(maps=maps, env=env, anchors=anchors, n_null=a.n_null, spatial_ok=sok)
    units = [u for u in r["order"] if r["screen"][u]][: a.n_units]
    n_a = len(anchors)

    fig, axes = plt.subplots(len(units), n_a + 1,
                             figsize=(2.5 * (n_a + 1), 2.75 * len(units)), squeeze=False)
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad(alpha=0.0)

    span = 2 * RADIUS + 1
    for row, u in enumerate(units):
        m = maps[u]
        vmax = float(np.nanmax(m)) or 1.0
        ax = axes[row][0]
        ax.imshow(np.ma.masked_invalid(m), cmap=cmap, vmin=0, vmax=vmax,
                  interpolation="nearest", origin="upper")
        for lm, (ax_, ay_) in zip(lms, anchors):
            ax.plot(ax_ - 1, ay_ - 1, MARKER.get(lm.shape, "o"), ms=13,
                    mfc=lm.color, mec="white", mew=1.8, zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"#{u}", fontsize=10, rotation=0, labelpad=16, va="center")
        ax.text(0.98, 0.02, f"peak {vmax:.2f}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=8, color="white",
                bbox=dict(fc="black", alpha=0.45, pad=1.4, lw=0))
        cls = "object cell" if r["radial"][u] > 0.5 else "object-VECTOR"
        if row == 0:
            ax.set_title("room-frame rate map\n(landmarks ringed)", fontsize=9.5)

        # the same map, cropped around each landmark
        grid = np.full((n_a, span, span), np.nan)
        for i, (dx, dy) in enumerate(r["offsets"]):
            grid[:, dy + RADIUS, dx + RADIUS] = r["omaps"][:, u, i]
        gmax = float(np.nanmax(grid)) or 1.0
        for k in range(n_a):
            axk = axes[row][k + 1]
            axk.imshow(np.ma.masked_invalid(grid[k]), cmap=cmap, vmin=0, vmax=gmax,
                       interpolation="nearest", origin="upper")
            axk.plot(RADIUS, RADIUS, MARKER.get(lms[k].shape, "o"), ms=12,
                     mfc=lms[k].color, mec="white", mew=1.8, zorder=5)
            axk.set_xticks([]); axk.set_yticks([])
            if row == 0:
                axk.set_title(f"offset map\n{lms[k].shape} / {lms[k].color}", fontsize=9.5)
        axes[row][n_a].text(
            1.06, 0.5,
            f"percentile {r['pct'][u]:.0f}\nvector score {r['vector_score'][u]:+.3f}\n"
            f"radial {r['radial'][u]:.2f}\npeak at d={r['peak_dist'][u]:.0f}\n-> {cls}",
            transform=axes[row][n_a].transAxes, va="center", ha="left", fontsize=8.5)

    n_pass = int((r["screen"] & (r["pct"] > 95)).sum())
    n_scr = int(r["screen"].sum())
    # Print BOTH denominators so the rate-screen-only and ovc_eval-matched
    # fractions can never be mistaken for one another.
    n_rate = int(r["rate_ok"].sum())
    n_rate_pass = int((r["rate_ok"] & (r["pct"] > 95)).sum())
    fig.suptitle(
        f"{a.label} — top {len(units)} object / object-vector CANDIDATES\n"
        f"{env_name}, {room_desc}\n"
        f"Ranked by vector percentile against each unit's own random-anchor null. "
        f"{n_pass}/{n_scr} screened units clear the 95th percentile "
        f"({n_pass / max(n_scr, 1):.1%} vs 5% chance). "
        f"Screen: peak_ratio>=1.5{' + spatial information' if a.si_screen else ' only'}; "
        f"probe {h.shape[0]} trajectories x {h.shape[1]} steps.\n"
        "An object-VECTOR cell shows the same bump at the same displacement in every offset map; "
        "an OBJECT cell shows it at the centre; a place cell in only one.\n"
        "Landmarks are drawn as their own shape and colour, so a rotated triad should rotate the "
        "field with it.",
        fontsize=10.5, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0, 0.87, 0.90))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.tag or a.label.lower().replace(" ", "-").replace("/", "-")
    p = OUT / f"fig_e1_cells_{tag}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    print(f"  rate screen only : {n_rate_pass}/{n_rate} over the 95th pct "
          f"({n_rate_pass / max(n_rate, 1):.3f} vs 0.05)")
    if a.si_screen:
        print(f"  + spatial screen : {n_pass}/{n_scr} over the 95th pct "
              f"({n_pass / max(n_scr, 1):.3f} vs 0.05)   <- matches ovc_eval.e1")
    for u in units:
        print(f"  #{u:>3} pct {r['pct'][u]:>3.0f}  vector {r['vector_score'][u]:+.3f}  "
              f"radial {r['radial'][u]:.2f}  peak d={r['peak_dist'][u]:.0f}")
    print(f"wrote {p}")
    if a.gallery:
        from scripts.trace import trace_maps as tm

        si = tm.spatial_info(maps=maps, occupancy=np.ones(maps.shape[1:]), weighting="uniform")
        top = np.argsort(-np.nan_to_num(si, nan=-np.inf))[: a.gallery]
        n_col = 8
        n_row = int(np.ceil(len(top) / n_col))
        gfig, gax = plt.subplots(n_row, n_col, figsize=(1.75 * n_col, 1.95 * n_row), squeeze=False)
        for ax_ in gax.ravel()[len(top):]:
            ax_.axis("off")
        for ax_, u2 in zip(gax.ravel(), top):
            m2 = maps[u2]
            v2 = float(np.nanmax(m2)) or 1.0
            ax_.imshow(np.ma.masked_invalid(m2), cmap=cmap, vmin=0, vmax=v2,
                       interpolation="nearest", origin="upper")
            for lm, (lx, ly) in zip(lms, anchors):
                ax_.plot(lx - 1, ly - 1, MARKER.get(lm.shape, "o"), ms=8,
                         mfc=lm.color, mec="white", mew=1.2, zorder=5)
            ax_.set_title(f"#{u2}  SI {si[u2]:.2f}", fontsize=7.5, pad=2)
            ax_.set_xticks([]); ax_.set_yticks([])
        gfig.suptitle(
            f"{a.label} — POSITIVE CONTROL: the {len(top)} most spatially informative units\n"
            f"{env_name}, {room_desc}.  Each on its own colour scale; landmarks in their own "
            "shape and colour.\n"
            "Clean, localised place fields here mean the object-coding null below is a real bound "
            "rather than a failed measurement.",
            fontsize=10.5, y=0.995, va="top")
        gfig.tight_layout(rect=(0, 0, 1, 0.90 if n_row > 1 else 0.80))
        gp = OUT / f"fig_e1_gallery_{tag}.png"
        gfig.savefig(gp, dpi=160, bbox_inches="tight")
        print(f"  SI median {np.nanmedian(si):.3f}, top {si[top[0]]:.3f}; wrote {gp}")

    (OUT / f"fig_e1_cells_{tag}.json").write_text(json.dumps(
        {"label": a.label, "ckpt": a.ckpt, "env": env_name, "room": room_desc,
         "anchors": [list(x) for x in anchors],
         "landmarks": [{"shape": l.shape, "color": l.color, "anchor": list(l.anchor)} for l in lms],
         "units": [int(u) for u in units],
         "n_over_95": n_pass, "n_screened": n_scr,
         "frac": n_pass / max(n_scr, 1)}, indent=2, default=float))


if __name__ == "__main__":
    main()
