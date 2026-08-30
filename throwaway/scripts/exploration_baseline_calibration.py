"""Sizing the exploration evals BEFORE building them: what coverage is reachable
in one 256-step episode, from the transition table alone.

THROWAWAY, and no result may depend on it. It exists to answer one design
question - are T50/T90 measurable at the training episode length, and what is
the dynamic range between a random walker and a near-optimal one - so that the
real implementation in `curious_george/evaluation/` is not designed blind. When
that lands, these numbers get re-derived by the library and pinned in a test;
this file is then dead.

No networks are involved. Every agent here is a policy over the SAME action
table the device backend steps, so the numbers isolate what the GEOMETRY plus an
action distribution buys.

    uv run python throwaway/scripts/exploration_baseline_calibration.py
"""

from collections import deque

import numpy as np
from minigrid.core.constants import DIR_TO_VEC

from curious_george.envs.layouts import (
    EnvContent,
    EnvShape,
    LandmarkKind,
    RoomRules,
    RoomSetRules,
    SHAPES,
    Selected,
    base_walkable,
    pooled_walkable,
    resolve_rooms,
)

T = 256          # the training episode length (CollectCfg.episode_steps)
N_RANDOM = 2000  # episodes per random arm
N_GREEDY = 40    # spawns per greedy arm (deterministic given a spawn)
FORWARD = [0.15, 0.15, 0.6, 0.1]  # storage.RAND_ACT_PROBA
UNIFORM = [0.25] * 4

BASE = base_walkable()


def rooms_for(impassable: bool) -> list:
    return resolve_rooms(
        shape=EnvShape(),
        content=EnvContent(
            kinds=tuple(LandmarkKind(s, impassable=impassable) for s in SHAPES)
        ),
        source=Selected(n=5, impassable=impassable),
        room_rules=RoomRules(),
        set_rules=RoomSetRules(),
        indices=None,
    )


def step_fn(layout, impassable: bool):
    """(x, y, dir, action) -> (x, y, dir), the same machine obs_bank builds."""
    blocked = set(layout.cells) if impassable else set()

    def step(x, y, d, a):
        if a == 0:
            return x, y, (d - 1) % 4
        if a == 1:
            return x, y, (d + 1) % 4
        if a == 2:
            dx, dy = DIR_TO_VEC[d]
            fx, fy = x + int(dx), y + int(dy)
            if (fx, fy) in BASE and (fx, fy) not in blocked:
                return fx, fy, d
        return x, y, d

    return step


def random_curves(step, spawns, probs, *, n: int, seed: int = 0) -> np.ndarray:
    """(n, T) cumulative distinct cells for an i.i.d. action policy."""
    rng = np.random.default_rng(seed)
    starts = [spawns[i] for i in rng.integers(0, len(spawns), size=n)]
    dirs = rng.integers(0, 4, size=n)
    out = np.zeros((n, T), dtype=np.int64)
    for i, (spawn, d) in enumerate(zip(starts, dirs)):
        x, y = spawn
        seen, curve = set(), np.empty(T, dtype=np.int64)
        for t in range(T):
            seen.add((x, y))
            curve[t] = len(seen)
            x, y, d = step(x, y, d, rng.choice(4, p=probs))
        out[i] = curve
    return out


def greedy_curves(step, spawns, *, n: int, seed: int = 0) -> np.ndarray:
    """(n, T) for a sweeper that walks to the nearest unvisited cell.

    The POSITIVE control. Not optimal, but achievable under the same dynamics
    (a turn costs a step), so it bounds what any policy could reach.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((n, T), dtype=np.int64)
    for i in range(n):
        state = (*spawns[rng.integers(0, len(spawns))], int(rng.integers(4)))
        seen, curve = set(), np.empty(T, dtype=np.int64)
        for t in range(T):
            seen.add((state[0], state[1]))
            curve[t] = len(seen)
            prev, queue, target = {state: None}, deque([state]), None
            while queue:
                s = queue.popleft()
                if (s[0], s[1]) not in seen:
                    target = s
                    break
                for a in range(4):
                    ns = step(*s, a)
                    if ns not in prev:
                        prev[ns] = (s, a)
                        queue.append(ns)
            if target is None:
                state = step(*state, 2)
            else:
                path, s = [], target
                while prev[s] is not None:
                    s, a = prev[s]
                    path.append(a)
                state = step(*state, path[-1])
        out[i] = curve
    return out


def row(label: str, curves: np.ndarray, denom: int) -> str:
    frac = curves / denom
    cells = curves[:, -1]
    parts = [
        f"{label:<24}",
        f"{frac[:, -1].mean():.3f}",
        f"{frac.mean():.3f}",  # normalized AUC = mean of the normalized curve
    ]
    for thr in (0.25, 0.5, 0.75, 0.9):
        reached = frac[:, -1] >= thr
        if reached.any():
            t = np.argmax(frac[reached] >= thr, axis=1).mean()
            parts.append(f"{t:5.0f} ({100 * reached.mean():3.0f}%)")
        else:
            parts.append("  cens.    ")
    return "  ".join(parts) + f"   [{cells.mean():5.1f} cells]"


def main() -> None:
    print(f"L-room, one 256-step episode, coverage of the ROOM'S OWN walkable set\n")
    header = (
        f"{'agent':<24}  {'cov@256':>7}  {'nAUC':>5}  "
        f"{'T25':>12}  {'T50':>12}  {'T75':>12}  {'T90':>12}"
    )
    for impassable in (False, True):
        rooms = rooms_for(impassable)
        per_room = [len(r.walkable(BASE)) for r in rooms]
        union = len(pooled_walkable(BASE, rooms))
        print(f"=== impassable={impassable} ===")
        print(f"  per-room walkable {per_room}   union {union}   "
              f"ratio {per_room[0] / union:.4f}")
        layout = rooms[0]
        step = step_fn(layout, impassable)
        spawns = sorted(layout.walkable(BASE))
        denom = len(spawns)
        print(f"  room0 {layout.key}, denominator {denom}")
        print("  " + header)
        print("  " + row("uniform [.25]x4", random_curves(step, spawns, UNIFORM, n=N_RANDOM), denom))
        print("  " + row("forward-weighted", random_curves(step, spawns, FORWARD, n=N_RANDOM), denom))
        print("  " + row("greedy sweeper", greedy_curves(step, spawns, n=N_GREEDY), denom))
        print(f"  combinatorial floor for T90: >= {int(np.ceil(0.9 * denom)) - 1} steps\n")


if __name__ == "__main__":
    main()
