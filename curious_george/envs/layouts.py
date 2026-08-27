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
from enum import Enum
from typing import Union
from itertools import permutations

import numpy as np
from minigrid.envs.Lroom import Landmark

# All three are used in every layout, so "distinct shapes" is automatic and the
# set is the design, not a sample space.
SHAPES: tuple[str, ...] = ("x", "plus", "block3")

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

OFFSET_RADIUS = 4  # the object-vector window; see throwaway/ported/docs_exp_instructions/instructions-OVC.md


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
        throwaway/ported/docs_exp_instructions/instructions-OVC.md.
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
    dedupe_d4: bool = False,
    span: int = 14,
    stencils: tuple[str, ...] = SHAPES,
) -> list[tuple[tuple[int, int], ...]]:
    """EVERY anchor assignment the room admits, one per stencil in order.

    `stencils` was the module constant `SHAPES`, read at four places inside this
    function. It is an argument so a run can carry a different landmark set -
    different shapes, or solid objects - without editing a global that every
    other layout function also reads.

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
        for s in stencils
    }
    if any(len(a) == 0 for a in anchors.values()):
        raise ValueError(f"no anchor has wall clearance {min_wall_distance}")

    grids = np.meshgrid(*[np.arange(len(anchors[s])) for s in stencils], indexing="ij")
    picks = [anchors[s][g.ravel()] for s, g in zip(stencils, grids)]

    def cheb(a, b):
        return np.maximum(np.abs(a[:, 0] - b[:, 0]), np.abs(a[:, 1] - b[:, 1]))

    sep = np.min(
        [cheb(picks[i], picks[j]) for i in range(len(stencils)) for j in range(i + 1, len(stencils))],
        axis=0,
    )

    out = []
    seen_orbits: set = set()
    for k in np.nonzero(sep >= min_anchor_separation)[0]:
        triple = tuple(tuple(int(v) for v in p[k]) for p in picks)
        layout = Layout(
            tuple(
                Landmark(s, c, a)
                for s, c, a in zip(stencils, LANDMARK_COLORS[: len(stencils)], triple)
            )
        )
        painted = [c for lm in layout.landmarks for c in lm.cells]
        if len(painted) != len(set(painted)):
            continue
        if layout.min_cell_gap() < min_cell_gap:
            continue
        if layout.n_testable_offsets(walkable=walkable) < min_testable_offsets:
            continue
        if dedupe_d4:
            orbit = d4_canonical(triple, span=span)
            if orbit in seen_orbits:
                continue
            seen_orbits.add(orbit)
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
    dedupe_d4: bool = False,
    span: int = 14,
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
        dedupe_d4=dedupe_d4,
        span=span,
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


# The rooms for the alternating-room run, frozen rather than recomputed.
#
# Derived by an exact search over all admissible anchor assignments: rooms whose
# landmark CONFIGURATIONS differ by at least the within-room anchor separation,
# then the largest possible distance between their landmarks. They are committed
# because the search costs minutes and because a training run must not depend on
# a regenerable file under outputs/. Re-derive and check with
#
#     uv run python throwaway/ported/layout_figures.py --pool 500 --rooms 3
#
# which reports configuration distance 12 and cross-room landmark distance 3 -
# the latter being the ceiling over the whole admissible set, not a search
# artifact. The room is only 172 walkable cells.
#
# ⚠️ MEASURED 2026-08-13: THESE THREE ROOMS ARE EXACT TRANSLATIONS OF ONE
# CONFIGURATION. room 0 -> room 1 is a shift by (0, 3); room 0 -> room 2 is a
# shift by (3, 0); all three have separation signature (6, 6, 6), i.e. congruent
# right triangles. The shape/colour assignment is permuted across the shift, so
# identity does vary, but the GEOMETRY does not.
#
# The "configuration distance" floor above therefore does not do what its name
# suggests: it separates the rooms by POSITION, not by internal geometry, which
# is the same failure the square-room selection was later fixed for (see
# ROOMS_SQUARE's `distinct_signatures`). ROOMS_RUN1 predates that fix.
#
# Consequence for interpretation, and it is not small: a network can cover all
# three rooms with ONE map plus a translation, so remapping is never REQUIRED in
# the L-room arm. A remapping index of ~0 there is consistent with a network
# that has bound nothing, AND with one facing a manipulation too weak to force
# it. The square arm does not have this problem. State it with any L-room
# multi-room result.
#
# Verify with:
#     uv run python -c "from curious_george.envs.layouts import ROOMS_RUN1 as R; \
#       print({(dx,dy) for dx in range(-14,15) for dy in range(-14,15) \
#       if {(x+dx,y+dy) for x,y in R[0].anchors} == set(R[1].anchors)})"
ROOMS_RUN1: tuple[Layout, ...] = tuple(
    Layout(tuple(Landmark(shape, color, tuple(anchor)) for shape, color, anchor in spec))
    for spec in (
        (("x", "yellow", (3, 3)), ("plus", "green", (3, 9)), ("block3", "red", (9, 3))),
        (("x", "green", (3, 6)), ("plus", "blue", (9, 6)), ("block3", "yellow", (3, 12))),
        (("x", "blue", (6, 9)), ("plus", "yellow", (6, 3)), ("block3", "green", (12, 3))),
    )
)

# The rooms for the SQUARE alternating-room run, frozen like ROOMS_RUN1.
#
# Derived under the same exact search plus two square-room-specific rules:
#   - D4 dedup. A square has EIGHT symmetries and all three landmark shapes are
#     themselves D4-invariant, so every admissible layout has seven admissible
#     twins that are the same room rotated or reflected. Measured: 32,280
#     admissible assignments collapse to 4,035 orbits, and 100% of layouts share
#     an orbit with another. Without dedup the selection would happily pick two.
#   - Distinct separation signatures. Selecting on distance alone returned three
#     rooms all at signature (6,6,6) - congruent triangles, 7.1% of the set -
#     which tests one configuration moved around rather than three rooms.
#
# Re-derive with:
#     uv run python throwaway/ported/layout_figures.py --room square --pool 500 --rooms 3
ROOMS_SQUARE: tuple = tuple(
    Layout(tuple(Landmark(shape, color, anchor) for shape, color, anchor in spec))
    for spec in (
        (("x", "yellow", (3, 3)), ("plus", "green", (3, 9)), ("block3", "red", (9, 3))),
        (("x", "green", (3, 6)), ("plus", "blue", (9, 10)), ("block3", "yellow", (12, 4))),
        (("x", "blue", (6, 6)), ("plus", "yellow", (4, 12)), ("block3", "green", (12, 12))),
    )
)

#: The `-Multi-v0` variants accept a `landmarks=` argument; the plain ids ship
#: their own landmarks and do not. Which one a run builds follows from whether
#: it specifies a room set at all.
MULTI_ROOM_ID = {
    "MiniGrid-LRoom-v0": "MiniGrid-LRoom-Multi-v0",
    "MiniGrid-SquareRoom-v0": "MiniGrid-SquareRoom-Multi-v0",
}

BASE_ROOM_ID = "MiniGrid-LRoom-v0"       # owns the wall geometry; landmarks never change it
SQUARE_ROOM_ID = "MiniGrid-SquareRoom-v0"

# Rooms a run can be built in. The square room's walls are four-fold symmetric,
# so its landmarks are the ONLY cue to position - which is why it needs the D4
# handling below and the L-room does not.
MULTI_ENV_ID = {
    BASE_ROOM_ID: "MiniGrid-LRoom-Multi-v0",
    SQUARE_ROOM_ID: "MiniGrid-SquareRoom-Multi-v0",
}


def base_walkable(room_id: str = BASE_ROOM_ID) -> frozenset[tuple[int, int]]:
    """Walkable cells of a room, read from a throwaway instance.

    Landmarks are walkable `Floor`, so this is the same set for every layout in
    a room - which is exactly why one transition table serves them all.
    """
    import gymnasium as gym

    env = gym.make(room_id)
    env.reset(seed=0)
    return walkable_cells(env=env)


# --- the dihedral group of the square ---------------------------------------
# A square room has EIGHT symmetries. Two layouts related by one of them are the
# same room: a trajectory through one produces an identical observation sequence
# to the transformed trajectory through the other, and all three landmark shapes
# are themselves D4-invariant (verified: x, plus and block3 each map to
# themselves under rotation and reflection), so a transformed layout is a valid
# layout with the same shapes and colours.
#
# This matters in a specific, silent way. Under a rotation the landmarks move to
# different ABSOLUTE coordinates, so a unit using ONE map, rotated, looks like a
# unit that remapped - the remapping index would read positive for a network
# that did nothing of the kind. Worse, the room-selection metrics actively
# PREFER such pairs: a rotation gives large cross-room landmark distance AND
# large configuration distance while being the same room. That is the translated
# -rooms failure again, in a bigger group.
#
# The L-room's L-shaped wall breaks every one of these symmetries, so this
# applies to the square room only.
def _d4(cell: tuple[int, int], op: int, span: int) -> tuple[int, int]:
    """Image of `cell` under one of the 8 symmetries of a `span`-wide square."""
    x, y = cell
    lo, hi = 1, span                       # interior runs 1..span
    if op & 4:                             # reflect in x first
        x = lo + hi - x
    for _ in range(op & 3):                # then rotate 90 degrees, op&3 times
        x, y = y, lo + hi - x
    return (x, y)


def d4_canonical(anchors: tuple[tuple[int, int], ...], *, span: int) -> tuple:
    """A representative shared by every layout in `anchors`' D4 orbit.

    Two anchor assignments have the same canonical form iff one is a rotation or
    reflection of the other. Anchors keep their shape order (SHAPES), because a
    transformation moves the landmarks without relabelling them.
    """
    return min(
        tuple(_d4(a, op, span) for a in anchors) for op in range(8)
    )


def resolve_layouts(cfg) -> list[Layout] | None:
    """The rooms a run trains on, from its `EnvCfg`.

    A thin adapter over `resolve_rooms`, kept because the training loop and the
    offline scorer both hold a whole config and this is the one place that
    knows which of its fields describe a room set.
    """
    return resolve_rooms(
        shape=cfg.env.shape,
        content=cfg.env.content,
        source=cfg.env.source,
        room_rules=cfg.env.room_rules,
        set_rules=cfg.env.set_rules,
        indices=cfg.env.indices,
    )


# ===========================================================================
# The environment as SHAPE x CONTENT, and the set of rooms a run trains over.
#
# Four layers, and the split is the point:
#
#   EnvShape       the WALLS. Nothing about what is in them.
#   EnvContent     WHAT is in them. No coordinates.
#   RoomRules      WITHIN one room: when is a placement legal?
#   RoomSetRules   BETWEEN rooms: what makes a SET an experiment?
#
# Mixing shape with content is what produces rooms, and it happens in three
# stages because content assignment must be separable from placement. It was
# not: `generate_layouts` drew a fresh colour permutation per room, so colour
# ALWAYS varied and "same objects, different places" could not be expressed.


class Symmetry(str, Enum):
    """The symmetries a room's walls leave intact.

    Not a setting. Two placements related by a rotation of a square room ARE
    the same room, and that is a fact about the geometry - it used to be a
    `dedupe_d4` boolean derived by string-comparing the room id.
    """

    TRIVIAL = "trivial"  # the L-shaped wall breaks all eight
    D4 = "d4"  # square: 4 rotations x 2 reflections


@dataclass(frozen=True)
class EnvShape:
    """The room's walls."""

    room: str = BASE_ROOM_ID

    @property
    def walkable(self) -> frozenset[tuple[int, int]]:
        return base_walkable(self.room)

    @property
    def symmetry(self) -> Symmetry:
        return Symmetry.D4 if self.room == SQUARE_ROOM_ID else Symmetry.TRIVIAL

    @property
    def span(self) -> int:
        """Max interior coordinate, matching `_d4`'s "interior runs 1..span".

        NOT max+1. Computing it that way maps every cell one step outside the
        walkable set - 225 images for a 196-cell room - which would silently
        corrupt the D4 orbits the square-room dedup is built on.
        """
        cells = self.walkable
        return max(max(x for x, _ in cells), max(y for _, y in cells))


@dataclass(frozen=True)
class LandmarkKind:
    """One landmark's identity, independent of where it goes.

    `size` and `solid` are DECLARED BUT INERT: the stencil table and the object
    class both live in the minigrid fork, so a 2-cell landmark or a movement-
    blocking one needs a change there first. They are here because the axis is
    what makes a shape or solid-object experiment a config change rather than a
    refactor - and because a field that does nothing is better than a field
    that silently does something else.
    """

    stencil: str
    size: int = 3
    solid: bool = False


@dataclass(frozen=True)
class EnvContent:
    """What is in the room. No coordinates - placement is the mix."""

    kinds: tuple[LandmarkKind, ...] = tuple(LandmarkKind(s) for s in SHAPES)
    palette: tuple[str, ...] = LANDMARK_COLORS

    @property
    def stencils(self) -> tuple[str, ...]:
        return tuple(k.stencil for k in self.kinds)

    @property
    def n_landmarks(self) -> int:
        return len(self.kinds)

    def __post_init__(self) -> None:
        if not self.kinds:
            raise ValueError("a room needs at least one landmark kind")
        if len(self.palette) < self.n_landmarks:
            raise ValueError(
                f"{self.n_landmarks} landmarks need at least that many colours; "
                f"palette has {len(self.palette)}"
            )


@dataclass(frozen=True)
class RoomRules:
    """WITHIN one room: when is a placement legal?"""

    min_cell_gap: int = 2
    min_anchor_separation: int = 6
    """Between landmarks INSIDE one room. Not to be confused with
    `RoomSetRules.min_configuration_distance`, which is between rooms -
    conflating them is how a set came to be three translations of one room."""
    min_wall_distance: int = 2
    min_testable_offsets: int = 40
    max_coverage: float = 1.0
    """Landmark cells as a fraction of walkable cells - the "% of environment"
    axis.

    INERT by default, deliberately. Measured: the committed designs sit at
    11.0% (L-room, 19 cells of 172) and 9.7% (square, 19 of 196), so a default
    of 0.10 would have silently excluded this project's own L-room design from
    every generated pool. A new constraint must not move the admissible set
    until someone asks it to.
    """


class Vary(str, Enum):
    """What DIFFERS between rooms in a set.

    Anything NOT listed is held constant across the set - and what is held
    constant IS the manipulation. `{POSITION}` is "same objects, different
    places"; `{KIND}` is "same places, different shapes".
    """

    POSITION = "position"
    COLOR = "color"
    KIND = "kind"


@dataclass(frozen=True)
class RoomSetRules:
    """BETWEEN rooms: what makes a set an experiment rather than a bag of rooms?"""

    varies: frozenset[Vary] = frozenset({Vary.POSITION, Vary.COLOR})
    distinct_signatures: bool = True
    """Reject a set whose rooms are congruent. Selecting on distance alone
    returned three rooms all at separation signature (6,6,6) - one configuration
    moved around, which tests nothing about room identity."""
    min_configuration_distance: int | None = None


# --- where a set of rooms comes from --------------------------------------


@dataclass(frozen=True)
class EnvDefault:
    """The landmarks the environment CLASS ships with.

    One room, content not chosen here, and the only source that runs on the
    plain env id rather than the `-Multi-v0` variant that accepts `landmarks=`.
    Nearly every checkpoint in this project trained on this; naming it is what
    makes those runs describable.
    """


@dataclass(frozen=True)
class Committed:
    """Literal rooms, pinned by identity.

    Survives changes to the generator, which is the job: ROOMS_RUN1 was produced
    before the `distinct_signatures` fix, so re-deriving it today yields a
    DIFFERENT set and every historical result would silently refer to rooms it
    never trained on. Not validated against RoomSetRules - it records what ran,
    it does not propose a design.
    """

    rooms: tuple[Layout, ...] = ()
    """Defaulted so the union can build a CLI parser at all - tyro refuses a
    tuple of struct types without one. Empty is not a usable set and raises;
    from a command line you want `Frozen`, which resolves the committed set for
    whatever shape is configured."""


@dataclass(frozen=True)
class Frozen:
    """The committed set FOR THIS SHAPE, resolved by the room.

    `Committed` names literal rooms and cannot come from a command line, and it
    is room-specific: handing the L-room's set to a square room would be silently
    wrong. This picks the right one from `shape`, which is what makes "train on
    the frozen three" expressible as a flag.
    """


@dataclass(frozen=True)
class Curated:
    """`n` rooms chosen to be a good design, under RoomSetRules."""

    n: int = 3
    seed: int = 20260813


@dataclass(frozen=True)
class Uniform:
    """`n` rooms drawn uniformly from the admissible set. No design criteria -
    a broad sample, where degeneracy between any two rooms does not matter."""

    n: int = 500
    seed: int = 20260813


RoomSource = Union[EnvDefault, Frozen, Committed, Curated, Uniform]

#: A placement: one anchor per landmark, in `EnvContent.kinds` order.
AnchorSet = tuple[tuple[int, int], ...]


# --- the mix: shape x content -> rooms, in three stages -------------------
#
# Three and not one because content assignment has to be separable from
# placement. It was not: `generate_layouts` drew a fresh colour permutation
# inside the same loop that chose anchors, so colour ALWAYS varied and "same
# objects, different places" was inexpressible.


def admissible_placements(
    shape: EnvShape, content: EnvContent, rules: RoomRules
) -> list[AnchorSet]:
    """WITHIN-room stage: every placement of `content` that `shape` admits.

    Exhaustive, so the size of the design space is a measured fact rather than
    a property of a sampler. Symmetry dedup is taken from the SHAPE, not passed:
    two placements related by a rotation of a square room are the same room.
    """
    placements = enumerate_anchor_triples(
        walkable=shape.walkable,
        min_cell_gap=rules.min_cell_gap,
        min_anchor_separation=rules.min_anchor_separation,
        min_wall_distance=rules.min_wall_distance,
        min_testable_offsets=rules.min_testable_offsets,
        dedupe_d4=shape.symmetry is Symmetry.D4,
        span=shape.span,
        stencils=content.stencils,
    )
    if rules.max_coverage < 1.0:
        budget = rules.max_coverage * len(shape.walkable)
        placements = [
            p for p in placements
            if len(_layout_of(p, content).cells) <= budget
        ]
    return _drop_relabellings(placements, content)


def _interchangeable_slots(content: EnvContent) -> tuple[tuple[int, ...], ...]:
    """Landmark slots that paint identically: same stencil AND same colour.

    The enumerator assigns stencil `i` to anchor `i`, so with DISTINCT landmarks
    every ordering is a different room and there is nothing to collapse. Give it
    interchangeable ones - which is what an object-vector design wants, three
    objects a vector code must treat alike - and each room comes back once per
    permutation of those slots.

    Conservative by construction: slots differing in either stencil or colour
    are never grouped, so a design that varies colour between rooms is untouched.
    """
    slots: dict[tuple[str, str], list[int]] = {}
    colors = content.palette[: content.n_landmarks]
    for i, key in enumerate(zip(content.stencils, colors)):
        slots.setdefault(key, []).append(i)
    return tuple(tuple(v) for v in slots.values() if len(v) > 1)


def _drop_relabellings(placements: list[AnchorSet], content: EnvContent) -> list[AnchorSet]:
    """One placement per genuinely distinct room.

    Measured on the L-room with three identical objects: 6,894 placements for
    1,149 distinct rooms, exactly 3! per room. Without this, `Uniform(n=500)`
    returned 418 distinct rooms while reporting 500 - so a methods section
    saying "500 rooms" was wrong by 16%, and the run built and held banks for
    duplicates.
    """
    groups = _interchangeable_slots(content)
    if not groups:
        return placements

    def canonical(anchors: AnchorSet) -> AnchorSet:
        out = list(anchors)
        for group in groups:
            for slot, anchor in zip(group, sorted(out[i] for i in group)):
                out[slot] = anchor
        return tuple(out)

    seen: set = set()
    unique = []
    for placement in placements:
        key = canonical(placement)
        if key not in seen:
            seen.add(key)
            unique.append(placement)
    return unique


def _layout_of(anchors: AnchorSet, content: EnvContent,
               colors: tuple[str, ...] | None = None,
               stencils: tuple[str, ...] | None = None) -> Layout:
    cols = colors if colors is not None else content.palette[: content.n_landmarks]
    kinds = stencils if stencils is not None else content.stencils
    return Layout(tuple(Landmark(k, c, a) for k, c, a in zip(kinds, cols, anchors)))


def separation_signature(anchors: AnchorSet) -> tuple[int, ...]:
    """Sorted pairwise Chebyshev distances - a room's internal GEOMETRY.

    Two rooms with the same signature are congruent: the same configuration
    moved around. That is what `RoomSetRules.distinct_signatures` rejects.
    """
    return tuple(sorted(
        _chebyshev(anchors[i], anchors[j])
        for i in range(len(anchors)) for j in range(i + 1, len(anchors))
    ))


def select_rooms(
    placements: list[AnchorSet], source: RoomSource, *,
    set_rules: RoomSetRules,
) -> list[AnchorSet]:
    """BETWEEN-room stage: which placements form the set.

    When POSITION is not varied every room shares one placement, and the set is
    an experiment about something else - `{KIND}` is "same places, different
    shapes".
    """
    if isinstance(source, (EnvDefault, Committed)):
        raise TypeError(f"{type(source).__name__} does not select from a pool")
    n, seed = source.n, source.seed
    rng = np.random.default_rng(seed)

    if Vary.POSITION not in set_rules.varies:
        return [placements[int(rng.integers(len(placements)))]] * n

    if isinstance(source, Uniform):
        if n > len(placements):
            raise ValueError(f"asked for {n} rooms; only {len(placements)} admissible")
        return [placements[i] for i in rng.choice(len(placements), size=n, replace=False)]

    # Curated: distinct geometry, and far apart if asked.
    order = rng.permutation(len(placements))
    chosen: list[AnchorSet] = []
    seen: set = set()
    for i in order:
        cand = placements[int(i)]
        sig = separation_signature(cand)
        if set_rules.distinct_signatures and sig in seen:
            continue
        if set_rules.min_configuration_distance is not None and any(
            min(_chebyshev(a, b) for a, b in zip(cand, other))
            < set_rules.min_configuration_distance
            for other in chosen
        ):
            continue
        chosen.append(cand)
        seen.add(sig)
        if len(chosen) == n:
            return chosen
    raise ValueError(
        f"only {len(chosen)} of {n} rooms satisfy {set_rules}; "
        f"{len(placements)} placements were admissible"
    )


def dress(
    placements: list[AnchorSet], content: EnvContent, *,
    set_rules: RoomSetRules, seed: int,
) -> list[Layout]:
    """BETWEEN-room stage: attach kind and colour to each placement.

    Anything not in `set_rules.varies` is assigned ONCE and reused, which is
    what holds identity constant across the set.
    """
    rng = np.random.default_rng(seed)
    n_lm = content.n_landmarks
    fixed_colors = tuple(content.palette[:n_lm])
    fixed_kinds = content.stencils

    out = []
    for anchors in placements:
        colors = (
            tuple(content.palette[i] for i in rng.choice(len(content.palette), n_lm, replace=False))
            if Vary.COLOR in set_rules.varies else fixed_colors
        )
        kinds = (
            tuple(fixed_kinds[i] for i in rng.permutation(n_lm))
            if Vary.KIND in set_rules.varies else fixed_kinds
        )
        out.append(_layout_of(anchors, content, colors=colors, stencils=kinds))
    return out


def resolve_rooms(
    *, shape: EnvShape, content: EnvContent, source: RoomSource,
    room_rules: RoomRules = RoomRules(),
    set_rules: RoomSetRules = RoomSetRules(),
    indices: tuple[int, ...] | None = None,
) -> list[Layout] | None:
    """The one interpreter: a spec in, the rooms a run trains on out.

    None means "the environment's own landmarks" - the plain env id, no
    `landmarks=` argument - which is what nearly every checkpoint here used.
    """
    if isinstance(source, EnvDefault):
        return None
    if isinstance(source, Committed):
        if not source.rooms:
            raise ValueError(
                "Committed() with no rooms is not a set. Pass rooms explicitly, "
                "or use Frozen() for the committed set of this shape."
            )
        rooms = list(source.rooms)
    elif isinstance(source, Frozen):
        rooms = list(
            ROOMS_SQUARE if shape.room == SQUARE_ROOM_ID else ROOMS_RUN1
        )
    else:
        placements = admissible_placements(shape, content, room_rules)
        seed = source.seed
        rooms = dress(
            select_rooms(placements, source, set_rules=set_rules),
            content, set_rules=set_rules, seed=seed,
        )
    if indices is None:
        return rooms
    if any(i >= len(rooms) for i in indices):
        raise ValueError(f"indices {indices} out of range for {len(rooms)} rooms")
    return [rooms[i] for i in indices]
