"""What do SOLID landmarks look like to the agent?

Throwaway look-see, not a result. Every landmark this project has trained on is
`Floor`-derived: the agent walks straight over it and it occludes nothing. This
paints the same three shapes, smaller, as objects that BLOCK - both movement and
line of sight - and shows five random steps beside what the agent sees at each.

WHY `Wall` AND NOT `Ball`/`Box`. Both of those are `can_pickup=True`, and this
project's fourth action is MiniGrid's `pickup` under the name `stay_put`
(utils/common.py). A ball landmark would vanish into `carrying` the first time
the agent used it, so the manipulation would erase itself mid-rollout. `Wall`
is `can_overlap=False, can_pickup=False`.

WHY THE GRID IS PAINTED BY HAND. `LandmarkKind.solid` and `.size` exist in the
config and are INERT: the stencil table and the object class both live in the
minigrid fork. Making them real is a fork change; this is the look-see that
says whether it is worth one.

    uv run python throwaway/occlusion_objects.py

Writes throwaway/outputs/occlusion_objects.png.
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
from minigrid.core.world_object import Wall  # noqa: E402

from curious_george.envs.layouts import ROOMS_RUN1  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "occlusion_objects.png"

SEED = 7
N_STEPS = 5
TILE = 24  # render size ONLY; the network receives the same view at tile_size=1

#: Below the red square at (9,3)-(10,4), facing up into it. Deliberate and close
#: enough that the five random actions actually REACH it - at (9,8) the agent ran
#: out of steps one cell short and nothing was ever blocked, which is the one
#: thing this is meant to show. So
#: the agent actually MEETS an object inside five steps.
START_POS = (9, 7)
START_DIR = 3  # up

#: Smaller than the standard stencils, as offsets from the layout's anchor.
#: Standard x/plus/block3 are 5/5/9 cells on a 3x3 footprint; these are 2/3/4
#: on a 2x2. Shape identity is kept - a diagonal, a corner and a square are
#: still three different things in a 7x7 view.
SMALL = {
    "x": ((0, 0), (1, 1)),
    "plus": ((0, 0), (0, 1), (1, 0)),
    "block3": ((0, 0), (0, 1), (1, 0), (1, 1)),
}

ACTION_NAMES = {0: "turn left", 1: "turn right", 2: "forward", 3: "stay_put/pickup"}
DIR_ARROW = {0: "→", 1: "↓", 2: "←", 3: "↑"}


def build_solid_room(layout, *, see_through_walls: bool):
    """The L-room with `layout`'s landmarks replaced by smaller SOLID ones.

    Built through the multi-room id so the anchors come from the established
    layout machinery rather than being invented here; the cells are then erased
    and repainted, because the fork has no solid landmark to ask for.
    """
    env = gym.make("MiniGrid-LRoom-Multi-v0", landmarks=list(layout.landmarks))
    env.reset(seed=SEED)
    u = env.unwrapped
    u.see_through_walls = see_through_walls

    for cell in layout.cells:  # drop the walk-through originals
        u.grid.set(*cell, None)

    painted = {}
    for lm in layout.landmarks:
        cells = [(lm.anchor[0] + dx, lm.anchor[1] + dy) for dx, dy in SMALL[lm.shape]]
        for c in cells:
            u.grid.set(*c, Wall(lm.color))
        painted[lm.shape] = (lm.color, lm.anchor, len(cells))

    # Placed DELIBERATELY, facing the red square from four cells below. A
    # random start put the agent in a corner facing a wall for all five steps,
    # so every view panel was empty grey - a picture of nothing.
    u.agent_pos = np.array(START_POS)
    u.agent_dir = START_DIR
    return env, painted


def build_walkable_room(layout):
    """The SAME layout as it exists today: Floor-derived, walk-through, and it
    occludes nothing. The control this whole look-see is against."""
    env = gym.make("MiniGrid-LRoom-Multi-v0", landmarks=list(layout.landmarks))
    env.reset(seed=SEED)
    u = env.unwrapped
    u.see_through_walls = False
    for cell in layout.cells:
        u.grid.set(*cell, None)
    for lm in layout.landmarks:
        from minigrid.core.world_object import Floor

        for dx, dy in SMALL[lm.shape]:
            u.grid.set(lm.anchor[0] + dx, lm.anchor[1] + dy, Floor(lm.color))
    u.agent_pos = np.array(START_POS)
    u.agent_dir = START_DIR
    return env


def rollout(env, n_steps: int, seed: int, actions=None):
    """Random actions, recording what the agent did and what it saw.

    `actions` replays a fixed sequence so the walkable control faces exactly the
    same poses - otherwise the two arms differ in their random draw and the
    comparison is confounded by the walk, not the objects.
    """
    rng = np.random.default_rng(seed)
    u = env.unwrapped
    frames = [(u.agent_pos, u.agent_dir, None,
               u.get_frame(highlight=False, tile_size=TILE),
               u.get_frame(highlight=False, tile_size=TILE, agent_pov=True))]
    taken = []
    for step in range(n_steps):
        action = int(rng.integers(4)) if actions is None else actions[step]
        taken.append(action)
        env.step(action)
        frames.append((u.agent_pos, u.agent_dir, action,
                       u.get_frame(highlight=False, tile_size=TILE),
                       u.get_frame(highlight=False, tile_size=TILE, agent_pov=True)))
    return frames, taken


def plot(solid, walkable, painted, *, path: Path):
    """Three rows: where the agent is, and what it sees under each regime.

    The walkable arm replays the SAME actions from the SAME start, so the two
    view rows differ only in what the landmarks ARE.
    """
    n = len(solid)
    fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 8.4),
                             gridspec_kw={"hspace": 0.28, "wspace": 0.06})
    for col in range(n):
        pos, direction, action, top, pov = solid[col]
        axes[0, col].imshow(top)
        axes[0, col].set_title(
            ("start" if action is None else ACTION_NAMES[action])
            + f"\n{tuple(int(v) for v in pos)} {DIR_ARROW[direction]}",
            fontsize=9,
        )
        axes[1, col].imshow(pov)
        axes[2, col].imshow(walkable[col][4])
        for row in range(3):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

    axes[0, 0].set_ylabel("top-down\n(solid)", fontsize=10)
    axes[1, 0].set_ylabel("agent view\nSOLID (Wall)", fontsize=10, color="darkred")
    axes[2, 0].set_ylabel("agent view\nWALKABLE (Floor)", fontsize=10, color="darkgreen")

    shapes = " · ".join(f"{s}:{c} {k} cells @{a}" for s, (c, a, k) in painted.items())
    fig.suptitle(
        "Solid landmarks vs the walkable ones this project trains on — same layout, "
        "same start, same actions\n"
        f"{shapes}   ·   smaller than standard (2x2 footprint, was 3x3)   ·   "
        "see_through_walls=False\n"
        "views rendered at tile_size=24; the network receives this same view at "
        "tile_size=1 (7x7 px)",
        fontsize=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")
    return fig


def main() -> None:
    layout = ROOMS_RUN1[0]

    env, painted = build_solid_room(layout, see_through_walls=False)
    solid, actions = rollout(env, N_STEPS, SEED)

    # Same actions, same start: the arms differ only in what a landmark IS.
    walkable, _ = rollout(build_walkable_room(layout), N_STEPS, SEED, actions=actions)

    plot(solid, walkable, painted, path=OUT)

    def blocked(frames):
        return sum(
            1 for i in range(1, len(frames))
            if frames[i][2] == 2 and tuple(frames[i][0]) == tuple(frames[i - 1][0])
        )

    print(f"  actions: {[ACTION_NAMES[a] for a in actions]}")
    print(f"  forward blocked - SOLID: {blocked(solid)}  WALKABLE: {blocked(walkable)}")
    for shape, (color, anchor, ncells) in painted.items():
        print(f"  {shape:7} {color:7} {ncells} cells @ {anchor}")
    end_solid = tuple(int(v) for v in solid[-1][0])
    end_walk = tuple(int(v) for v in walkable[-1][0])
    print(f"  end position - SOLID {end_solid}  WALKABLE {end_walk}"
          + ("  <- the objects changed where it ended up" if end_solid != end_walk else ""))


if __name__ == "__main__":
    main()
