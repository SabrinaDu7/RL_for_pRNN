"""Generate, check, time and plot the landmark layouts for multi-room training.

Answers three questions before any training is launched:

  1. Does the generator produce layouts that satisfy every stated constraint?
     Checked here rather than trusted - each constraint is re-verified on the
     generated pool independently of the code that enforced it.
  2. What does a layout cost? Room construction is not the cost; the
     observation bank is (`width * height * 4` renders per distinct grid), and
     it decides whether a pool of N is affordable.
  3. What do they look like? A layout set is an experimental design, so it has
     to be looked at, not inferred from anchor coordinates.

    uv run python scripts/layout_figures.py --pool 200 --rooms 3

Writes fig_layouts_rooms.png (the alternating-room set), fig_layouts_pool.png
(a sample of the pool) and layouts.json (the frozen design) under outputs/layouts/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from minigrid.envs.Lroom import Landmark

from curious_george.envs.layouts import (
    LANDMARK_COLORS,
    OFFSET_RADIUS,
    SHAPES,
    Layout,
    enumerate_anchor_triples,
    generate_layouts,
    walkable_cells,
)

OUT = Path("outputs/layouts")
ENV_ID = "MiniGrid-LRoom-Multi-v0"
BASE_ENV_ID = "MiniGrid-LRoom-v0"


def base_room():
    """One instance of the unmodified room, to read geometry from."""
    env = gym.make(BASE_ENV_ID)
    env.reset(seed=0)
    return env


def build(layout: Layout):
    env = gym.make(ENV_ID, landmarks=list(layout.landmarks))
    env.reset(seed=0)
    return env


def verify(*, layouts: list[Layout], walkable: frozenset) -> dict:
    """Re-check every constraint on the generated pool, independently."""
    problems: list[str] = []
    for layout in layouts:
        shapes = [lm.shape for lm in layout.landmarks]
        colors = [lm.color for lm in layout.landmarks]
        if sorted(shapes) != sorted(SHAPES):
            problems.append(f"{layout.key}: shapes {shapes} are not the full distinct set")
        if len(set(colors)) != len(colors):
            problems.append(f"{layout.key}: repeated colour in {colors}")
        painted = [c for lm in layout.landmarks for c in lm.cells]
        if len(painted) != len(set(painted)):
            problems.append(f"{layout.key}: landmarks overlap")
        if not set(painted) <= walkable:
            problems.append(f"{layout.key}: a landmark cell is not on walkable floor")
        if len(set(layout.anchors)) != len(layout.anchors):
            problems.append(f"{layout.key}: repeated anchor")

    # the painted cells must survive into the actual grid: put_obj silently
    # drops nothing, but a wall would be overwritten, which would change the room
    sample = layouts[: min(10, len(layouts))]
    for layout in sample:
        env = build(layout)
        grid = env.unwrapped.grid
        for lm in layout.landmarks:
            for x, y in lm.cells:
                cell = grid.get(x, y)
                if cell is None or cell.type != "floor" or cell.color != lm.color:
                    problems.append(
                        f"{layout.key}: cell {(x, y)} is {cell} not {lm.color} floor"
                    )
        if len(walkable_cells(env=env)) != len(walkable):
            problems.append(f"{layout.key}: walkable count changed")

    keys = [lay.key for lay in layouts]
    if len(set(keys)) != len(keys):
        problems.append("duplicate layouts in the pool")
    return {"n": len(layouts), "problems": problems}


def timings(*, layouts: list[Layout], n_bank: int) -> dict:
    """Room construction vs observation-bank BUILD vs bank LOAD.

    Build and load are timed separately and the build is measured only on
    layouts whose bank file does not exist yet - a bank that is already on disk
    takes ~2 ms to load, and reporting that as the build cost would understate
    the one-time price of a pool by two orders of magnitude.
    """
    from curious_george.envs.obs_bank import BankedRGBPartialObsWrapper

    t0 = time.perf_counter()
    for layout in layouts[:50]:
        build(layout)
    make_ms = 1e3 * (time.perf_counter() - t0) / len(layouts[:50])

    def _bank(layout):
        env = gym.make(ENV_ID, landmarks=list(layout.landmarks))
        wrapped = BankedRGBPartialObsWrapper(env, tile_size=1)
        wrapped._ensure_bank_path = None  # noqa - only to keep linters quiet
        t0 = time.perf_counter()
        wrapped.reset(seed=0)
        return 1e3 * (time.perf_counter() - t0), wrapped

    fresh, cached = [], []
    for layout in layouts:
        if len(fresh) >= n_bank and len(cached) >= n_bank:
            break
        env = gym.make(ENV_ID, landmarks=list(layout.landmarks))
        probe = BankedRGBPartialObsWrapper(env, tile_size=1)
        env.reset(seed=0)
        exists = probe._bank_path(probe._grid_fingerprint()).exists()
        if exists and len(cached) >= n_bank:
            continue
        if not exists and len(fresh) >= n_bank:
            continue
        ms, _ = _bank(layout)
        (cached if exists else fresh).append(ms)

    return {
        "gym_make_ms": make_ms,
        "bank_build_ms_median": float(np.median(fresh)) if fresh else float("nan"),
        "bank_load_ms_median": float(np.median(cached)) if cached else float("nan"),
        "n_build_timed": len(fresh),
        "n_load_timed": len(cached),
    }


def plot(*, layouts: list[Layout], walkable: frozenset, path: Path, title: str,
         ncols: int = 5) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows = int(np.ceil(len(layouts) / ncols))
    # 4.9 in per row, not 3.5: the caption under each panel is five lines, and
    # at 3.5 the next row's captions were drawn over the previous row's images.
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.3 * ncols, 4.9 * nrows), squeeze=False,
        gridspec_kw={"hspace": 0.55, "wspace": 0.08},
    )
    for ax in axes.ravel():
        ax.axis("off")
    for ax, layout in zip(axes.ravel(), layouts):
        env = build(layout)
        ax.imshow(env.unwrapped.get_frame(highlight=False, tile_size=16))
        ax.axis("off")
        caption = "\n".join(
            [f"layout {layout.key}"]
            + [f"{lm.shape} · {lm.color} @ {lm.anchor}" for lm in layout.landmarks]
            + [
                f"anchor separation ≥ {layout.min_anchor_separation} cells",
                f"landmark gap ≥ {layout.min_cell_gap()} · "
                f"wall clearance ≥ {layout.min_wall_distance(walkable=walkable)}",
                f"{layout.n_testable_offsets(walkable=walkable)} testable offsets "
                f"(Chebyshev ≤ {OFFSET_RADIUS})",
            ]
        )
        # caption BELOW the image, so nothing can collide with the row above
        ax.text(0.5, -0.04, caption, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, linespacing=1.35)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_shapes(*, path: Path) -> None:
    """What each shape looks like TO THE AGENT, beside the top-down view.

    The top-down render is 16 px per cell; the observation the network receives
    is ONE pixel per cell in a 7x7 view. A pair of shapes can be obvious from
    above and nearly identical in the only view that matters, so both are shown
    at the same scale.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shapes = list(SHAPES)
    fig, axes = plt.subplots(2, len(shapes), figsize=(3.0 * len(shapes), 6.6),
                             squeeze=False, gridspec_kw={"hspace": 0.35})
    centre = (7, 7)
    for col, shape in enumerate(shapes):
        layout = Layout((
            Landmark(shape, "red", centre),
            Landmark(shapes[(col + 1) % len(shapes)], "blue", (3, 12)),
            Landmark(shapes[(col + 2) % len(shapes)], "green", (12, 3)),
        ))
        env = gym.make(ENV_ID, landmarks=list(layout.landmarks))
        env.reset(seed=0)
        u = env.unwrapped
        axes[0][col].imshow(u.get_frame(highlight=False, tile_size=16))
        axes[0][col].set_title(f"{shape} — {len(Landmark(shape, 'red', centre).cells)} cells\n"
                               f"top-down, 16 px per cell", fontsize=9)

        # agent three cells below the landmark centre, facing it (dir 3 = up)
        u.agent_pos = (centre[0], centre[1] + 3)
        u.agent_dir = 3
        view = u.get_frame(highlight=False, tile_size=1, agent_pov=True)
        axes[1][col].imshow(np.repeat(np.repeat(view, 40, axis=0), 40, axis=1),
                            interpolation="nearest")
        axes[1][col].set_title("what the network sees\n7x7 view, 1 px per cell "
                               "(shown 40x)", fontsize=9)
        for r in (0, 1):
            axes[r][col].axis("off")
    fig.suptitle("Landmark shapes at both scales — the bottom row is the only one "
                 "the pRNN ever receives", fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def cross_layout_distance(a: Layout, b: Layout) -> int:
    """How far b's landmarks sit from a's - the smallest anchor-to-anchor move.

    The alternating-room set wants this LARGE: if two rooms put a landmark in
    nearly the same place, a unit can satisfy both with one room-anchored place
    field and the rooms stop being a test of anything.
    """
    return min(
        max(abs(p[0] - q[0]), abs(p[1] - q[1])) for p in a.anchors for q in b.anchors
    )


def pick_rooms(*, layouts: list[Layout], k: int, walkable: frozenset,
               min_config_distance: int = 6) -> list[Layout]:
    """The `k` rooms whose landmark CONFIGURATIONS differ most, then whose
    landmarks are furthest apart.

    Configuration first, and that ordering is the whole point. Maximising
    cross-room landmark distance alone selects rooms that are TRANSLATES of one
    another - measured: it returned three rooms related by (0,3) and (3,0)
    shifts. Translated rooms are the worst possible choice here, because a place
    cell that shifts with the room is then indistinguishable from a cell coding
    an offset from a landmark, which is exactly the dissociation the multi-room
    design exists to make.

    A room's configuration is the set of vectors between its landmarks, taken
    per shape pair so it is label-aware; two rooms are translates iff every one
    of those vectors matches. Distance between configurations is the largest
    disagreement among them, so 0 means "translate".

    Exact, not greedy: thresholds are scanned downward and a valid set is a
    k-clique in the graph of admissible pairs.
    """
    if k != 3:
        raise NotImplementedError("exact selection is implemented for k=3")

    n = len(layouts)
    anchors = np.array([lay.anchors for lay in layouts], dtype=np.int16)     # (n, 3, 2)
    pairs = [(0, 1), (0, 2), (1, 2)]
    config = np.stack([anchors[:, b] - anchors[:, a] for a, b in pairs], axis=1)  # (n,3,2)

    # Chunked: the full (n, n, 3, 3, 2) broadcast is 9 GB at n=7926.
    dist = np.empty((n, n), dtype=np.int8)
    cfg_dist = np.empty((n, n), dtype=np.int8)
    for lo in range(0, n, 256):
        blk = anchors[lo:lo + 256]
        d = np.abs(blk[:, None, :, None, :] - anchors[None, :, None, :, :]).max(-1)
        dist[lo:lo + 256] = d.min(axis=(2, 3)).astype(np.int8)
        c = np.abs(config[lo:lo + 256, None] - config[None, :]).max(-1)   # (blk,n,3)
        cfg_dist[lo:lo + 256] = c.max(axis=2).astype(np.int8)
    np.fill_diagonal(dist, 0)
    np.fill_diagonal(cfg_dist, 0)

    def triangle(adj):
        """Any 3-clique, via packed-bitset common neighbours."""
        np.fill_diagonal(adj, False)
        edges = np.argwhere(np.triu(adj, 1))
        if not len(edges):
            return None
        packed = np.packbits(adj, axis=1)
        both = packed[edges[:, 0]] & packed[edges[:, 1]]
        hit = np.argwhere(both.any(axis=1))
        if not len(hit):
            return None
        e = int(hit[0][0])
        return (int(edges[e][0]), int(edges[e][1]),
                int(np.nonzero(np.unpackbits(both[e])[:n])[0][0]))

    # A floor on configuration distance rather than maximising it: pushing it to
    # its maximum (17, measured) buys nothing scientifically and costs landmark
    # distance, which fell to 1 cell. The floor is the internal anchor
    # separation - two rooms whose landmark geometry differs by at least the
    # distance between landmarks WITHIN a room are different rooms, not
    # perturbations of one - and landmark distance is maximised subject to it.
    floor = min_config_distance
    for t in range(int(dist.max()), 0, -1):
        found = triangle((cfg_dist >= floor) & (dist >= t))
        if found is None:
            continue
        got = int(cfg_dist[np.ix_(list(found), list(found))][np.triu_indices(3, 1)].min())
        print(f"  selection over {n:,} candidates: configuration distance "
              f"≥ {got} (0 would mean translated rooms, floor {floor}), "
              f"smallest cross-room landmark distance {t} cells")
        return [layouts[i] for i in found]
    raise RuntimeError(
        f"no {k} rooms with configuration distance ≥ {floor}; lower it"
    )


def assign_distinct_colors(*, rooms: list[Layout], seed: int) -> list[Layout]:
    """Give each room a colour assignment, avoiding repeats ACROSS rooms.

    Rooms must be told apart from what is visible, so reusing (shape, colour)
    in two rooms throws away free discriminability. Four colours over three
    shapes cannot make every (shape, colour) pair unique across three rooms, so
    this minimises repeats rather than forbidding them.
    """
    rng = np.random.default_rng(seed)
    used: set[tuple[str, str]] = set()
    out = []
    for room in rooms:
        best, best_cost = None, None
        for _ in range(200):
            colors = [LANDMARK_COLORS[i]
                      for i in rng.choice(len(LANDMARK_COLORS), len(SHAPES), replace=False)]
            pairs = set(zip(SHAPES, colors))
            cost = len(pairs & used)
            if best_cost is None or cost < best_cost:
                best, best_cost = colors, cost
            if cost == 0:
                break
        used |= set(zip(SHAPES, best))
        out.append(Layout(tuple(Landmark(sh, c, lm.anchor)
                                for sh, c, lm in zip(SHAPES, best, room.landmarks))))
    return out


def sweep(*, walkable: frozenset, args) -> None:
    """How much wall clearance can we demand before layouts stop existing?

    Clearance and anchor separation pull against each other in a 172-cell room,
    so the honest way to choose them is to see where the feasible region ends
    rather than to pick a number and hope.
    """
    print(f"\n{'wall':>5} {'anchor':>7} | {'admissible':>12} {'x colours':>12} {'enum s':>8}")
    for wall in (1, 2, 3):
        for sep in (5, 6, 7):
            t0 = time.perf_counter()
            triples = enumerate_anchor_triples(
                walkable=walkable, min_cell_gap=args.min_cell_gap,
                min_anchor_separation=sep, min_wall_distance=wall,
                min_testable_offsets=args.min_testable_offsets,
            )
            n_colour = len(LANDMARK_COLORS) * (len(LANDMARK_COLORS) - 1) * (len(LANDMARK_COLORS) - 2)
            print(f"{wall:>5} {sep:>7} | {len(triples):>12,} {len(triples) * n_colour:>12,} "
                  f"{time.perf_counter() - t0:>8.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=200, help="layouts for the random-position run")
    ap.add_argument("--rooms", type=int, default=3, help="rooms for the alternating run")
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--show", type=int, default=10, help="pool layouts to plot")
    ap.add_argument("--n-bank", type=int, default=5, help="banks to build for timing")
    ap.add_argument("--min-cell-gap", type=int, default=2)
    ap.add_argument("--min-anchor-separation", type=int, default=6)
    ap.add_argument("--min-wall-distance", type=int, default=2)
    ap.add_argument("--sweep", action="store_true",
                    help="report the wall-clearance / separation trade-off and stop")
    ap.add_argument("--min-testable-offsets", type=int, default=40)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    walkable = walkable_cells(env=base_room())
    print(f"room: {len(walkable)} walkable cells   shapes {SHAPES}   colours {LANDMARK_COLORS}")

    t0 = time.perf_counter()
    if a.sweep:
        sweep(walkable=walkable, args=a)
        return

    pool = generate_layouts(
        walkable=walkable, n=a.pool, seed=a.seed,
        min_cell_gap=a.min_cell_gap,
        min_anchor_separation=a.min_anchor_separation,
        min_wall_distance=a.min_wall_distance,
        min_testable_offsets=a.min_testable_offsets,
    )
    gen_s = time.perf_counter() - t0
    print(f"\ngenerated {len(pool)} layouts in {gen_s:.2f} s "
          f"({1e3 * gen_s / len(pool):.1f} ms each)")

    check = verify(layouts=pool, walkable=walkable)
    print(f"constraint re-check on all {check['n']}: "
          f"{'PASS' if not check['problems'] else 'FAIL'}")
    for p in check["problems"][:10]:
        print(f"    {p}")

    seps = np.array([lay.min_anchor_separation for lay in pool])
    gaps = np.array([lay.min_cell_gap() for lay in pool])
    offs = np.array([lay.n_testable_offsets(walkable=walkable) for lay in pool])
    print(f"  anchor separation : min {seps.min()}  median {np.median(seps):.0f}  max {seps.max()}")
    print(f"  landmark gap      : min {gaps.min()}  median {np.median(gaps):.0f}  max {gaps.max()}")
    print(f"  testable offsets  : min {offs.min()}  median {np.median(offs):.0f}  max {offs.max()}")

    t = timings(layouts=pool, n_bank=a.n_bank)
    print("\nlatency")
    print(f"  gym.make + reset            : {t['gym_make_ms']:.2f} ms per layout")
    print(f"  observation bank, BUILD     : {t['bank_build_ms_median']:.0f} ms "
          f"(n={t['n_build_timed']}, layouts with no bank on disk)")
    print(f"  observation bank, LOAD      : {t['bank_load_ms_median']:.0f} ms "
          f"(n={t['n_load_timed']}, already built)")
    print(f"  -> pool of {a.pool}: {a.pool * t['bank_build_ms_median'] / 1000:.0f} s "
          f"to build once, then {a.pool * t['bank_load_ms_median'] / 1000:.1f} s to load")

    all_triples = enumerate_anchor_triples(
        walkable=walkable, min_cell_gap=a.min_cell_gap,
        min_anchor_separation=a.min_anchor_separation,
        min_wall_distance=a.min_wall_distance,
        min_testable_offsets=a.min_testable_offsets,
    )
    print(f"\nroom selection searches all {len(all_triples):,} admissible anchor assignments")
    rooms = pick_rooms(
        layouts=[Layout(tuple(Landmark(sh, c, an) for sh, c, an in zip(SHAPES, LANDMARK_COLORS, t)))
                 for t in all_triples],
        k=a.rooms, walkable=walkable,
        min_config_distance=a.min_anchor_separation)
    rooms = assign_distinct_colors(rooms=rooms, seed=a.seed)
    print(f"\nalternating-room set ({a.rooms}):")
    for i, r in enumerate(rooms):
        others = [c for c in rooms if c is not r]
        print(f"  room {i} [{r.key}]  {r.describe()}")
        print(f"      anchor separation {r.min_anchor_separation}, "
              f"gap {r.min_cell_gap()}, "
              f"{r.n_testable_offsets(walkable=walkable)} testable offsets, "
              f"nearest landmark in another room {min(cross_layout_distance(r, o) for o in others)} cells")

    plot_shapes(path=OUT / "fig_layout_shapes.png")
    plot(layouts=rooms, walkable=walkable, path=OUT / "fig_layouts_rooms.png",
         title=f"Run 1 — the {a.rooms} rooms trained on simultaneously "
               f"(shape, colour and position all differ)", ncols=min(a.rooms, 5))
    plot(layouts=pool[: a.show], walkable=walkable, path=OUT / "fig_layouts_pool.png",
         title=f"Run 2 — first {a.show} of the {a.pool}-layout seeded pool "
               f"(seed {a.seed})", ncols=5)

    (OUT / "layouts.json").write_text(json.dumps({
        "seed": a.seed,
        "constraints": {"min_cell_gap": a.min_cell_gap,
                        "min_anchor_separation": a.min_anchor_separation,
                        "min_testable_offsets": a.min_testable_offsets,
                        "offset_radius": OFFSET_RADIUS},
        "timings_ms": t,
        "rooms": [[[lm.shape, lm.color, list(lm.anchor)] for lm in r.landmarks] for r in rooms],
        "pool": [[[lm.shape, lm.color, list(lm.anchor)] for lm in lay.landmarks] for lay in pool],
    }, indent=2))
    print(f"wrote {OUT / 'layouts.json'}")


if __name__ == "__main__":
    main()
