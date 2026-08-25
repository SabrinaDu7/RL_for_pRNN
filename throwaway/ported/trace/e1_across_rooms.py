"""The same units, the same weights, a DISPLACED triad — does the field follow?

This is the definitional test (Hoydal Fig. 1b trial-2 vs trial-3, and the OVC-3
criterion in `docs/exp_instructions/instructions-objectAndOVC.md`), and for a
multi-room checkpoint it is IN-DISTRIBUTION: the network trained on every room
shown, so a decorrelated map means "no vector code", not "the net is confused".

    rows     the top-ranked object/object-vector candidates, ranked ONCE in the
             home room so the same units appear in every column
    columns  one per room. Same weights, same probe construction, different
             landmark placement. Landmarks drawn as their own shape and colour.

Read across a row:
    field moves WITH the triad      -> anchored to the landmarks (object / OVC)
    field stays at the room location -> a place cell; the agreement seen inside
                                        any single room was coincidence

`ROOMS_RUN1` is a special case worth knowing: its three L-rooms are exact
TRANSLATIONS of one configuration (room 0 -> 1 by (0,3), room 0 -> 2 by (3,0)),
with the shape/colour assignment permuted. So the displacement is clean, and
"follows the configuration" is separable from "follows the identity".

    uv run python scripts/trace/e1_across_rooms.py --ckpt <f.pt> --label "3-room L-room"

Writes outputs/summary/fig_e1_rooms_<tag>.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.trace import ovc_metric as om
from scripts.trace.e1_cell_figure import MARKER, RADIUS, env_landmarks, rank

OUT = Path("outputs/summary")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env-config", default="lroom_multi",
                    choices=("lroom_multi", "squareroom_multi"))
    ap.add_argument("--layouts", default="rooms", choices=("rooms", "pool"))
    ap.add_argument("--rooms", type=int, default=3, help="how many rooms to show")
    ap.add_argument("--home", type=int, default=0, help="room the ranking is computed in")
    ap.add_argument("--label", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-units", type=int, default=3)
    ap.add_argument("--n-dirs", type=int, default=1)
    ap.add_argument("--n-null", type=int, default=100)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from curious_george.envs.layouts import resolve_layouts
    from scripts.multienv.checkpoint_curve import run_config
    from scripts.trace.ovc_eval import activity, maps_from

    cfg = run_config(env_cfg=a.env_config, layouts=a.layouts, hiddensize=500)
    rooms = resolve_layouts(cfg)[: a.rooms]

    per_room = []
    for k, room in enumerate(rooms):
        h, pos, env = activity(env_name=cfg.exp.env_name, ckpt_dir=a.ckpt,
                               n_dirs=a.n_dirs, landmarks=room.landmarks)
        maps = maps_from(h=h, pos=pos, env=env)
        lms = env_landmarks(env)
        anchors = [(int(round(l.centroid[0])), int(round(l.centroid[1]))) for l in lms]
        offsets = om.offset_grid(env=env, anchors=anchors, radius=RADIUS)
        vs = om.vector_score(
            omaps=om.offset_maps(maps=maps, anchors=anchors, offsets=offsets), offsets=offsets)
        per_room.append({"room": room, "maps": maps, "env": env, "lms": lms,
                         "anchors": anchors, "vector_score": vs})
        print(f"  room {k} ({room.key}): {room.describe()}")

    home = per_room[a.home]
    r = rank(maps=home["maps"], env=home["env"], anchors=home["anchors"], n_null=a.n_null)
    units = [u for u in r["order"] if r["screen"][u]][: a.n_units]

    fig, axes = plt.subplots(len(units), len(per_room),
                             figsize=(3.0 * len(per_room), 3.1 * len(units)), squeeze=False)
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad(alpha=0.0)

    for row, u in enumerate(units):
        vmax = max(float(np.nanmax(p["maps"][u])) for p in per_room) or 1.0
        for col, p in enumerate(per_room):
            ax = axes[row][col]
            ax.imshow(np.ma.masked_invalid(p["maps"][u]), cmap=cmap, vmin=0, vmax=vmax,
                      interpolation="nearest", origin="upper")
            for lm, (lx, ly) in zip(p["lms"], p["anchors"]):
                ax.plot(lx - 1, ly - 1, MARKER.get(lm.shape, "o"), ms=14,
                        mfc=lm.color, mec="white", mew=1.9, zorder=5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.98, 0.02, f"vector {p['vector_score'][u]:+.2f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="white",
                    bbox=dict(fc="black", alpha=0.5, pad=1.5, lw=0))
            if row == 0:
                ax.set_title(
                    f"room {col}" + ("\nUNITS RANKED HERE" if col == a.home else "\n(held out)")
                    + f"\nlayout id {p['room'].key}",
                    fontsize=9.5,
                    fontweight="bold" if col == a.home else "normal")
            if col == 0:
                ax.set_ylabel(f"#{u}\npct {r['pct'][u]:.0f}", fontsize=9,
                              rotation=0, labelpad=26, va="center")

    fig.suptitle(
        f"{a.label} — SAME units, SAME weights, DISPLACED landmarks\n"
        f"{cfg.exp.env_name}. Units ranked once in room {a.home}; colour scale shared across a row.\n"
        "If the field moves with the triad it is anchored to the landmarks; if it stays put it is a "
        "place cell\nand the within-room agreement was coincidence. Markers are the landmarks' own "
        "shape and colour.\n"
        "Units are chosen using ONE room and merely displayed in the others - picking the best unit "
        "per room would make every column look good by construction.\n"
        "'layout id' is a content hash of the room's (shape, colour, anchor) triple - an identifier "
        "for traceability, not a quantity.",
        fontsize=11, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.tag or a.label.lower().replace(" ", "-")
    p_out = OUT / f"fig_e1_rooms_{tag}.png"
    fig.savefig(p_out, dpi=160, bbox_inches="tight")

    print(f"\n  {'unit':>5} " + " ".join(f"{'room' + str(k):>10}" for k in range(len(per_room))))
    for u in units:
        print(f"  {u:>5} " + " ".join(f"{p['vector_score'][u]:>+10.3f}" for p in per_room))
    print(f"wrote {p_out}")
    (OUT / f"fig_e1_rooms_{tag}.json").write_text(json.dumps(
        {"label": a.label, "ckpt": a.ckpt, "home": a.home,
         "rooms": [{"key": p["room"].key, "describe": p["room"].describe(),
                    "anchors": [list(x) for x in p["anchors"]]} for p in per_room],
         "units": [int(u) for u in units],
         "vector_score": {int(u): [float(p["vector_score"][u]) for p in per_room] for u in units}},
        indent=2, default=float))


if __name__ == "__main__":
    main()
