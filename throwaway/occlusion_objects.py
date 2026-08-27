"""What do SOLID landmarks look like to the agent, and why is the background grey?

Throwaway look-see, not a result. Two questions in one figure.

1. THE SHAPES. The standard L-room ships blue `triangle6`, red `plus6` and
   yellow `x6` - 21/20/28 cells on 6x6 footprints (Lroom.py:200-202). Every one
   is `Floor`-derived: the agent walks straight over it and it occludes nothing.
   This paints HALF-SCALE versions of the same three shapes as objects that
   BLOCK, and runs the same actions under both regimes.

2. THE BACKGROUND. In wandb-logged reference observations everything outside the
   room is GREY; in the first version of this figure it was BLACK. Black is
   UNSEEN space. The reference runs with `see_through_walls=True` - the L-room's
   own default, and what `see_through_walls: null` meant in the old config - so
   nothing is ever masked and out-of-room cells render as grey wall. Measured at
   one pose: 0 pure-black pixels with it True, 3,703 of 28,224 with it False,
   and IDENTICAL with and without solid objects, because there the occluder is
   the room's own wall rather than any landmark.

WHAT A SOLID LANDMARK COSTS AN EPISODE: NOTHING. `forward` into a `Wall` leaves
`agent_pos` unchanged and returns `terminated=False, reward=0`
(minigrid_env.py:549-555 terminates on `goal`/`fake_lava`/`lava` only). Episode
length is fixed anyway - the collector forces `done` every `prnn_seqdur` steps
(collector.py:376-378), so a blocked step costs a step and nothing more.

WHY `Wall` AND NOT `Ball`/`Box`. Both are `can_pickup=True`, and this project's
fourth action is MiniGrid's `pickup` wearing the name `stay_put`. A ball
landmark would vanish into `carrying` the first time it fired.

WHY THE GRID IS PAINTED BY HAND. `LandmarkKind.solid` and `.size` are INERT: the
stencil table and the object class both live in the minigrid fork. This is the
look-see that says whether that change is worth making.

    uv run python throwaway/occlusion_objects.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from minigrid.core.world_object import Floor, Wall  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "occlusion_objects.png"

SEED = 0
N_STEPS = 5
TILE = 24  # render size ONLY; the network receives this view at tile_size=1

#: Below the red plus, facing up into it, close enough that five actions reach
#: it. Set after the fact: earlier starts left the agent facing a wall, or one
#: cell short, so nothing was ever blocked - the one thing this exists to show.
START_POS = (10, 6)
START_DIR = 3  # up

#: HALF-SCALE versions of the room's own shapes, centre-anchored, as offsets.
#: The standard set is corner-anchored on 6x6; these keep the shape - a
#: triangle, a cross, a saltire - at roughly a quarter of the area.
SMALL = {
    "triangle": ((-1, -1), (0, -1), (0, 0), (1, -1), (1, 0), (1, 1)),   # 6 cells
    "plus": ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)),                 # 5
    "x": ((-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)),                  # 5
}
#: colour -> which shape sits there, from Lroom.py:200-202.
COLOR_SHAPE = {"blue": "triangle", "red": "plus", "yellow": "x"}

ACTION_NAMES = {0: "turn left", 1: "turn right", 2: "forward", 3: "stay_put/pickup"}
DIR_ARROW = {0: "→", 1: "↓", 2: "←", 3: "↑"}


def standard_landmark_cells(env) -> dict[str, list[tuple[int, int]]]:
    """Where the room's own landmarks are, read off the grid by colour.

    `LEnv` bakes them in and keeps no `landmarks` list, so the grid IS the
    record - which also means this cannot drift from what was actually built.
    """
    u = env.unwrapped
    out: dict[str, list[tuple[int, int]]] = {}
    for x in range(u.width):
        for y in range(u.height):
            cell = u.grid.get(x, y)
            if isinstance(cell, Floor):
                out.setdefault(cell.color, []).append((x, y))
    return out


def build_room(*, solid: bool, see_through_walls: bool):
    """The standard L-room with its landmarks replaced by half-scale ones."""
    env = gym.make("MiniGrid-LRoom-v0")
    env.reset(seed=SEED)
    u = env.unwrapped
    u.see_through_walls = see_through_walls

    painted = {}
    for color, cells in standard_landmark_cells(env).items():
        for c in cells:  # drop the full-size original
            u.grid.set(*c, None)
        xs = [p[0] for p in cells]
        ys = [p[1] for p in cells]
        centre = ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2)
        shape = COLOR_SHAPE[color]
        obj = Wall if solid else Floor
        small = [(centre[0] + dx, centre[1] + dy) for dx, dy in SMALL[shape]]
        for c in small:
            u.grid.set(*c, obj(color))
        painted[shape] = (color, centre, len(small), len(cells))

    u.agent_pos = np.array(START_POS)
    u.agent_dir = START_DIR
    return env, painted


def rollout(env, n_steps: int, seed: int, actions=None):
    """Random actions, recording pose and view. `actions` replays a fixed
    sequence so every arm faces exactly the same poses."""
    rng = np.random.default_rng(seed)
    u = env.unwrapped
    frames = [(np.array(u.agent_pos), u.agent_dir, None,
               u.get_frame(highlight=False, tile_size=TILE),
               u.get_frame(highlight=False, tile_size=TILE, agent_pov=True))]
    taken = []
    for step in range(n_steps):
        action = int(rng.integers(4)) if actions is None else actions[step]
        taken.append(action)
        env.step(action)
        frames.append((np.array(u.agent_pos), u.agent_dir, action,
                       u.get_frame(highlight=False, tile_size=TILE),
                       u.get_frame(highlight=False, tile_size=TILE, agent_pov=True)))
    return frames, taken


def plot(solid_occ, solid_see, walkable, painted, *, path: Path):
    n = len(solid_occ)
    fig, axes = plt.subplots(4, n, figsize=(2.5 * n, 11.0),
                             gridspec_kw={"hspace": 0.26, "wspace": 0.06})
    rows = [
        (0, [f[3] for f in solid_occ], "top-down\n(solid)", "black"),
        (1, [f[4] for f in solid_occ], "SOLID view\nsee_through=False", "darkred"),
        (2, [f[4] for f in solid_see], "SOLID view\nsee_through=True\n(reference AND chosen setting)", "darkorange"),
        (3, [f[4] for f in walkable], "WALKABLE view\nsee_through=True\n(what we train on)", "darkgreen"),
    ]
    for col in range(n):
        pos, direction, action, _, _ = solid_occ[col]
        axes[0, col].set_title(
            ("start" if action is None else ACTION_NAMES[action])
            + f"\n{tuple(int(v) for v in pos)} {DIR_ARROW[direction]}", fontsize=9)
        for row, images, _, _ in rows:
            axes[row, col].imshow(images[col])
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    for row, _, label, colour in rows:
        axes[row, 0].set_ylabel(label, fontsize=9, color=colour)

    shapes = " · ".join(f"{s}:{c} {orig}→{k} cells @{a}"
                             for s, (c, a, k, orig) in painted.items())
    fig.suptitle(
        "Half-scale SOLID landmarks in the standard L-room — the shapes the room "
        "ships (triangle / plus / x)\n"
        f"{shapes}\n"
        "row 2 vs row 3: BLACK is UNSEEN space. see_through_walls=True never masks, "
        "which is why the reference background is grey.",
        fontsize=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


def main() -> None:
    env, painted = build_room(solid=True, see_through_walls=False)
    solid_occ, actions = rollout(env, N_STEPS, SEED)
    solid_see, _ = rollout(build_room(solid=True, see_through_walls=True)[0],
                           N_STEPS, SEED, actions=actions)
    walkable, _ = rollout(build_room(solid=False, see_through_walls=True)[0],
                          N_STEPS, SEED, actions=actions)
    plot(solid_occ, solid_see, walkable, painted, path=OUT)

    def blocked(frames):
        return sum(1 for i in range(1, len(frames))
                   if frames[i][2] == 2 and tuple(frames[i][0]) == tuple(frames[i - 1][0]))

    def black_px(frames):
        return [int((f[4].sum(axis=2) == 0).sum()) for f in frames]

    print(f"  actions: {[ACTION_NAMES[a] for a in actions]}")
    for shape, (color, centre, small, orig) in painted.items():
        print(f"  {shape:9} {color:7} {orig:2d} -> {small} cells, centred {centre}")
    print(f"  forward blocked - SOLID {blocked(solid_occ)}  WALKABLE {blocked(walkable)}")
    print(f"  end position   - SOLID {tuple(int(v) for v in solid_occ[-1][0])}"
          f"  WALKABLE {tuple(int(v) for v in walkable[-1][0])}")
    print(f"  black (unseen) px, see_through=False: {black_px(solid_occ)}")
    print(f"  black (unseen) px, see_through=True : {black_px(solid_see)}")


if __name__ == "__main__":
    main()
