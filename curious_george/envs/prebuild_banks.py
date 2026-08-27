"""Pre-build the observation banks for a room set, and name the rooms.

A bank is built lazily on first use and cached under `data/obs_bank/`, so this
buys nothing a training run would not eventually do itself. What it buys is
TIMING and a RECORD: a cold build is ~0.45 s per room, so a pool pays that at
startup, and a cluster job pays it inside its allocation. Running this first
moves the cost somewhere it can be watched.

The record matters more. Q3 trains on a SUBSET of a pool and evaluates on rooms
it has never seen, so which rooms were which has to be stated rather than
recovered. `EnvCfg.indices` already expresses the split - `indices=(0..9)` is
the training set, `indices=(10..19)` is held out - so it lands in the run's
config and therefore in provenance with no new machinery. This prints the keys
so a results document can quote them.

    uv run python -m curious_george.envs.prebuild_banks --train 10 --held-out 10
    uv run python -m curious_george.envs.prebuild_banks --pool 200 --seed 7
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def build(*, pool: int, seed: int, n_train: int, n_held_out: int, room: str) -> dict:
    import gymnasium as gym

    from curious_george.envs.layouts import (
        MULTI_ROOM_ID,
        EnvContent,
        EnvShape,
        LandmarkKind,
        RoomSetRules,
        Uniform,
        Vary,
        base_walkable,
        resolve_rooms,
        separation_signature,
    )
    from curious_george.envs.obs_bank import TableDrivenRGBPartialObsWrapper

    content = EnvContent(
        kinds=tuple(LandmarkKind(s, impassable=True) for s in ("x", "plus", "block3"))
    )
    source = Uniform(n=pool, seed=seed)
    set_rules = RoomSetRules(varies=frozenset({Vary.POSITION}))
    shape = EnvShape(room)

    rooms = resolve_rooms(shape=shape, content=content, source=source,
                          set_rules=set_rules)
    if len(rooms) < n_train + n_held_out:
        raise SystemExit(
            f"pool of {len(rooms)} is smaller than {n_train} train + "
            f"{n_held_out} held out"
        )

    base = base_walkable(room)
    env = gym.make(MULTI_ROOM_ID[room], landmarks=list(rooms[0].landmarks))
    env.reset(seed=0)
    wrapper = TableDrivenRGBPartialObsWrapper(env, tile_size=1)

    out: dict = {"room": room, "pool": pool, "seed": seed, "sets": {}}
    for name, idx in (("train", range(n_train)),
                      ("held_out", range(n_train, n_train + n_held_out))):
        rows = []
        t0 = time.perf_counter()
        for i in idx:
            layout = rooms[i]
            wrapper.unwrapped.landmarks = list(layout.landmarks)
            env.reset(seed=0)
            wrapper._ensure_bank()
            rows.append({
                "index": i,
                "key": layout.key,
                "separations": list(separation_signature(layout.anchors)),
                "walkable": len(layout.walkable(base)),
                "testable_offsets": layout.n_testable_offsets(
                    walkable=layout.walkable(base)
                ),
                "describe": layout.describe(),
            })
        elapsed = time.perf_counter() - t0
        out["sets"][name] = rows
        print(f"\n{name.upper()}  ({len(rows)} rooms, {elapsed:.1f}s)")
        print(f"  {'idx':>4} {'key':<10} {'separations':<14} {'walkable':>9} {'offsets':>8}")
        for r in rows:
            print(f"  {r['index']:>4} {r['key']:<10} "
                  f"{str(tuple(r['separations'])):<14} {r['walkable']:>9} "
                  f"{r['testable_offsets']:>8}")

    train_keys = {r["key"] for r in out["sets"]["train"]}
    held_keys = {r["key"] for r in out["sets"]["held_out"]}
    if train_keys & held_keys:
        raise SystemExit("train and held-out sets overlap; the split is not a split")
    out["disjoint"] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="MiniGrid-LRoom-v0")
    ap.add_argument("--pool", type=int, default=200,
                    help="rooms drawn from the admissible set; the split indexes into it")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train", type=int, default=10)
    ap.add_argument("--held-out", type=int, default=10)
    ap.add_argument("--out", default=None, help="also write the record as JSON")
    a = ap.parse_args()

    record = build(pool=a.pool, seed=a.seed, n_train=a.train,
                   n_held_out=a.held_out, room=a.room)

    print(f"\nTo train on the first set, and evaluate on rooms never seen:")
    print(f"  train    EnvCfg(source=Uniform(n={a.pool}, seed={a.seed}), "
          f"indices=tuple(range({a.train})))")
    print(f"  held out EnvCfg(source=Uniform(n={a.pool}, seed={a.seed}), "
          f"indices=tuple(range({a.train}, {a.train + a.held_out})))")

    if a.out:
        Path(a.out).write_text(json.dumps(record, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
