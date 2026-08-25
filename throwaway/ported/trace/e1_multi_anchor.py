"""E1 - multi-anchor consistency, with its location control built in.

THE QUESTION. A unit that fires at the same allocentric offset from every
landmark is coding a metric relation. A place cell fires at one room location,
so its offset map differs per anchor. With three landmarks that differ in shape,
colour AND position, agreement across all three is the strongest object-vector
evidence obtainable without moving a landmark.

THE NULL, AND WHY IT IS NOT A TIME SHIFT. Circularly shifting activity against
position destroys ALL spatial structure, so it asks "is this unit spatially
tuned" rather than "is its tuning anchored to the landmarks" - measured in this
repo at negative control 0.130 with injected PLACE fields passing at 0.146.
Instead the map is left exactly as it is and the criterion is re-run at random
walkable triples, WITHIN each unit: these maps carry room-wide common structure
(random triples reach a population p99 vector score of 0.933), so no population
threshold is clearable. `ovc_metric.vector_percentile` owns this.

THE LOCATION CONTROL, WHICH IS THE POINT OF RUNNING IT HERE. A within-unit null
cannot see structure shared ACROSS units at particular room locations - the
failure that produced (14,7), the occlusion gradient, and the retracted OVC
lead. The rule written down after that: score the criterion at anchor positions
where THIS network has no landmark. A multi-room checkpoint gives that for free,
because the other rooms' anchors are landmark-free positions in this room, and
they are geometrically comparable by construction rather than random.

    uv run python scripts/trace/e1_multi_anchor.py --ckpt <file.pt> --room 0

Writes outputs/ovc/e1_multi_anchor_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.trace import ovc_metric as om

OUT = Path("outputs/ovc")


def score_at(*, maps, env, anchors, n_null: int, seed: int = 0) -> dict:
    """Fraction of units whose real-anchor vector score beats their OWN null."""
    offsets = om.offset_grid(env=env, anchors=list(anchors), radius=4)
    if len(offsets) < 8:
        return {"n_offsets": int(len(offsets)), "frac": float("nan"), "note": "too few offsets"}
    pct, real = om.vector_percentile(maps=maps, env=env, anchors=list(anchors),
                                     offsets=offsets, n_null=n_null, seed=seed)
    pr = om.peak_ratio(omaps=om.offset_maps(maps=maps, anchors=list(anchors), offsets=offsets))
    ok = np.isfinite(pct) & (pr >= 1.5)
    return {"n_offsets": int(len(offsets)), "n_screened": int(ok.sum()),
            "frac": float((ok & (pct > 95.0)).sum() / max(int(ok.sum()), 1)),
            "median_vector_score": float(np.nanmedian(real))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env", default="lroom_multi", choices=("lroom_multi", "squareroom_multi"))
    ap.add_argument("--room", type=int, default=0, help="which room the rollouts are collected in")
    ap.add_argument("--n-dirs", type=int, default=1)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    from curious_george.envs.layouts import resolve_layouts
    from scripts.multienv.checkpoint_curve import run_config
    from scripts.trace.ovc_eval import activity, maps_from

    cfg = run_config(env_cfg=a.env, layouts="rooms", hiddensize=500)
    rooms = resolve_layouts(cfg)
    home = rooms[a.room]
    print(f"{cfg.exp.env_name}, rollouts in room {a.room} ({home.key}): {home.describe()}")

    h, pos, env = activity(env_name=cfg.exp.env_name, ckpt_dir=a.ckpt,
                           n_dirs=a.n_dirs, landmarks=home.landmarks)
    maps = maps_from(h=h, pos=pos, env=env)
    print(f"probe {h.shape[0]} trajectories x {h.shape[1]} steps -> maps {maps.shape}")

    rows = []
    for k, r in enumerate(rooms):
        s = score_at(maps=maps, env=env, anchors=r.anchors, n_null=a.n_null)
        s |= {"anchors_of_room": k, "key": r.key, "own": k == a.room,
              "anchors": [list(x) for x in r.anchors]}
        rows.append(s)
        print(f"  anchors of room {k} ({r.key}){'  OWN' if s['own'] else '     '}  "
              f"offsets {s['n_offsets']:>3}  screened {s.get('n_screened', 0):>3}  "
              f"frac>own p95 {s['frac']:.3f}   median vector score {s['median_vector_score']:+.3f}")

    own = [r["frac"] for r in rows if r["own"]]
    other = [r["frac"] for r in rows if not r["own"]]
    print(f"\n  OWN landmarks   {np.mean(own):.3f}")
    print(f"  OTHER positions {np.mean(other):.3f}   (no landmark there in these rollouts)")
    print(f"  chance          0.050")
    print("  -> the signal is landmark-driven only if OWN clearly exceeds OTHER; equal values mean "
          "it is\n     room geometry, which is how the previous OVC lead was retracted.")

    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{a.env}_room{a.room}"
    p = OUT / f"e1_multi_anchor_{tag}.json"
    p.write_text(json.dumps({"ckpt": a.ckpt, "env": cfg.exp.env_name, "home_room": a.room,
                             "n_dirs": a.n_dirs, "n_null": a.n_null, "rows": rows,
                             "own": float(np.mean(own)), "other": float(np.mean(other))},
                            indent=2, default=float))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
