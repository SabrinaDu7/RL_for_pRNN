"""Seeded pools of landmark layouts for multi-room training.

A *layout* is one assignment of (shape, colour, anchor) to each landmark in the
L-room. Multi-room training holds several at once, so that the same
dead-reckoned trajectory maps to a different absolute position depending on
which room a stream is in - the manipulation that makes path integration
insufficient and a visual landmark necessary.

WHY A POOL AND NOT FRESH RANDOM ROOMS
Observations are served from a precomputed bank keyed on the grid fingerprint
(`curious_george/envs/obs_bank.py`), so every distinct layout costs one bank
build - `width * height * 4` renders. Generating a fresh layout per gradient
step would spend more wall-clock rendering than training. A pool is built once,
sampled from thereafter, and - unlike fresh randomness - is a fixed artefact
that analysis can be pointed at afterwards.

THE CONSTRAINTS, AND WHY EACH ONE
  distinct shapes, distinct colours   a unit firing at the same offset from
                                      three landmarks that share neither shape
                                      nor colour is coding a metric relation,
                                      not an identity
  no overlap, a gap between them      overlapping or touching landmarks merge
                                      into one blob in the 7x7 view, and their
                                      anchors stop being distinguishable
  anchors mutually separated          offset maps are read in a window around
                                      each anchor; anchors closer than twice
                                      the window radius give windows that
                                      overlap and therefore correlate for
                                      purely geometric reasons
  clear of the walls                  a landmark against a wall is part of the
                                      boundary, and vector tuning to it cannot
                                      be distinguished from boundary-vector
                                      tuning
  a floor under every cell            a landmark cell on a wall would overwrite
                                      the wall, changing the room
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import permutations

import numpy as np
from minigrid.envs.Lroom import Landmark

# All three are used in every layout, so "distinct shapes" is automatic and the
# set is the design, not a sample space.
SHAPES: tuple[str, ...] = ("x", "plus", "diamond")

# Observations are rendered at tile_size=1, so a cell is ONE pixel and colour
# carries the entire per-cell signal.
#
# Distances below are AS RENDERED, not the nominal palette. `Floor` paints
# `76 + 0.35 * COLORS[c]`, a blend toward grey, so every nominal separation
# reaches the network at 35% of its face value - empty floor is (76, 76, 76),
# not black. Chosen on that basis: within this palette the closest pair is 89
# apart and the closest colour to empty floor is 89.
#
# Excluded, and why - `grey` renders 61 from empty floor, weaker than any pair
# here, and it was visibly indistinguishable in the rendered rooms; `purple` is
# 46 from `blue`; `neon_green` is 21 from `green` and is reserved for
# FloorBright, the OMT novel object.
LANDMARK_COLORS: tuple[str, ...] = ("blue", "green", "red", "yellow")

OFFSET_RADIUS = 4  # the object-vector window; see docs/exp_instructions/instructions-OVC.md


def walkable_cells(*, env) -> frozenset[tuple[int, int]]:
    """World cells the agent can occupy, read from the grid itself."""
    u = env.unwrapped if hasattr(env, "unwrapped") else env
    return frozenset(
        (x, y)
        for y in range(u.height)
        for x in range(u.width)
        if (c := u.grid.get(x, y)) is None or c.can_overlap()
    )


def common_offsets(
    *,
    walkable: frozenset[tuple[int, int]],
    anchors: tuple[tuple[int, int], ...],
    radius: int = OFFSET_RADIUS,
) -> tuple[tuple[int, int], ...]:
    """Offsets landing on a walkable cell for EVERY anchor, sorted.

    The offsets over which an object-vector code can be tested at all: a
    correlation between anchor-centred maps is only defined where every anchor
    has a cell. This is a property of the room and the layout, not a parameter.
    """
    return tuple(
        sorted(
            (dx, dy)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if all((ax + dx, ay + dy) in walkable for ax, ay in anchors)
        )
    )


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _wall_distance(*, cell: tuple[int, int], walkable: frozenset[tuple[int, int]]) -> int:
    """Chebyshev rings outward from `cell` until one contains a non-walkable cell."""
    x, y = cell
    for r in range(1, 8):
        ring = (
            [(x + dx, y + dy) for dx in range(-r, r + 1) for dy in (-r, r)]
            + [(x + dx, y + dy) for dy in range(-r + 1, r) for dx in (-r, r)]
        )
        if any(c not in walkable for c in ring):
            return r
    return 8


@dataclass(frozen=True)
class Layout:
    """One room: the landmarks it contains, in a canonical order."""

    landmarks: tuple[Landmark, ...]

    @property
    def anchors(self) -> tuple[tuple[int, int], ...]:
        return tuple(lm.anchor for lm in self.landmarks)

    @property
    def centroids(self) -> tuple[tuple[float, float], ...]:
        return tuple(lm.centroid for lm in self.landmarks)

    @property
    def cells(self) -> frozenset[tuple[int, int]]:
        return frozenset(c for lm in self.landmarks for c in lm.cells)

    @property
    def min_anchor_separation(self) -> int:
        return min(
            _chebyshev(a, b) for a, b in permutations(self.anchors, 2)
        )

    def min_cell_gap(self) -> int:
        """Smallest Chebyshev distance between cells of two DIFFERENT landmarks."""
        return min(
            _chebyshev(p, q)
            for i, a in enumerate(self.landmarks)
            for b in self.landmarks[i + 1:]
            for p in a.cells
            for q in b.cells
        )

    def min_wall_distance(self, *, walkable: frozenset[tuple[int, int]]) -> int:
        """Smallest Chebyshev distance from any landmark cell to any wall.

        Small here means the landmark is effectively part of the boundary, and
        vector tuning to it cannot be told apart from boundary-vector tuning -
        the first of the known disanalogies in
        docs/exp_instructions/instructions-OVC.md.
        """
        return min(
            _wall_distance(cell=c, walkable=walkable) for c in self.cells
        )

    def n_testable_offsets(self, *, walkable: frozenset[tuple[int, int]]) -> int:
        return len(common_offsets(walkable=walkable, anchors=self.anchors))

    @property
    def key(self) -> str:
        """Short stable id - names bank files, wandb series and cached results."""
        spec = "|".join(
            f"{lm.shape}:{lm.color}:{lm.anchor[0]},{lm.anchor[1]}" for lm in self.landmarks
        )
        return hashlib.sha1(spec.encode()).hexdigest()[:8]

    def describe(self) -> str:
        return "  ".join(
            f"{lm.shape}/{lm.color}@{lm.anchor}" for lm in self.landmarks
        )


def valid_anchors(
    *,
    walkable: frozenset[tuple[int, int]],
    shape: str,
    min_wall_distance: int,
) -> list[tuple[int, int]]:
    """Anchors at which `shape` fits on floor with the required wall clearance."""
    return [
        a
        for a in sorted(walkable)
        if all(c in walkable for c in Landmark(shape, LANDMARK_COLORS[0], a).cells)
        and min(
            _wall_distance(cell=c, walkable=walkable)
            for c in Landmark(shape, LANDMARK_COLORS[0], a).cells
        )
        >= min_wall_distance
    ]


def enumerate_anchor_triples(
    *,
    walkable: frozenset[tuple[int, int]],
    min_cell_gap: int = 2,
    min_anchor_separation: int = 6,
    min_wall_distance: int = 2,
    min_testable_offsets: int = 40,
) -> list[tuple[tuple[int, int], ...]]:
    """EVERY anchor assignment the room admits, one per shape in `SHAPES` order.

    Exhaustive rather than sampled, for two reasons. It makes the size of the
    design space a measured fact instead of a property of a sampler - rejection
    sampling from all walkable cells finds so few layouts at
    `min_anchor_separation=6` that the setting looks infeasible when the room
    actually admits tens of thousands. And a pool drawn uniformly from the
    enumerated set has no sampler bias to argue about later.
    """
    anchors = {
        s: np.array(
            valid_anchors(walkable=walkable, shape=s, min_wall_distance=min_wall_distance),
            dtype=int,
        )
        for s in SHAPES
    }
    if any(len(a) == 0 for a in anchors.values()):
        raise ValueError(f"no anchor has wall clearance {min_wall_distance}")

    grids = np.meshgrid(*[np.arange(len(anchors[s])) for s in SHAPES], indexing="ij")
    picks = [anchors[s][g.ravel()] for s, g in zip(SHAPES, grids)]

    def cheb(a, b):
        return np.maximum(np.abs(a[:, 0] - b[:, 0]), np.abs(a[:, 1] - b[:, 1]))

    sep = np.min(
        [cheb(picks[i], picks[j]) for i in range(len(SHAPES)) for j in range(i + 1, len(SHAPES))],
        axis=0,
    )

    out = []
    for k in np.nonzero(sep >= min_anchor_separation)[0]:
        triple = tuple(tuple(int(v) for v in p[k]) for p in picks)
        layout = Layout(
            tuple(
                Landmark(s, c, a)
                for s, c, a in zip(SHAPES, LANDMARK_COLORS[: len(SHAPES)], triple)
            )
        )
        painted = [c for lm in layout.landmarks for c in lm.cells]
        if len(painted) != len(set(painted)):
            continue
        if layout.min_cell_gap() < min_cell_gap:
            continue
        if layout.n_testable_offsets(walkable=walkable) < min_testable_offsets:
            continue
        out.append(triple)
    return out


def generate_layouts(
    *,
    walkable: frozenset[tuple[int, int]],
    n: int,
    seed: int,
    min_cell_gap: int = 2,
    min_anchor_separation: int = 6,
    min_wall_distance: int = 2,
    min_testable_offsets: int = 40,
) -> list[Layout]:
    """`n` layouts drawn uniformly without replacement from the admissible set.

    Deterministic in `seed`. Raises if the room admits fewer than `n`, rather
    than returning a short pool - a silently smaller pool would change the
    experiment without changing the config.
    """
    triples = enumerate_anchor_triples(
        walkable=walkable,
        min_cell_gap=min_cell_gap,
        min_anchor_separation=min_anchor_separation,
        min_wall_distance=min_wall_distance,
        min_testable_offsets=min_testable_offsets,
    )
    if len(triples) < n:
        raise ValueError(
            f"the room admits {len(triples)} anchor assignments under these "
            f"constraints, fewer than the {n} requested; relax min_cell_gap="
            f"{min_cell_gap}, min_anchor_separation={min_anchor_separation}, "
            f"min_wall_distance={min_wall_distance} or min_testable_offsets="
            f"{min_testable_offsets}"
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(triples), size=n, replace=False)
    out = []
    for idx in chosen:
        colors = [
            LANDMARK_COLORS[i]
            for i in rng.choice(len(LANDMARK_COLORS), len(SHAPES), replace=False)
        ]
        out.append(
            Layout(
                tuple(
                    Landmark(shape, color, anchor)
                    for shape, color, anchor in zip(SHAPES, colors, triples[idx])
                )
            )
        )
    return out
