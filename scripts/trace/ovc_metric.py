"""Object-vector and object-cell criteria over pRNN rate maps.

Adapts Høydal, Skytøen, Andersson, Moser & Moser (2019, Nature,
doi 10.1038/s41586-019-1077-7) to a gridworld. Rationale, the pipeline being
adapted and the known disanalogies are in
`docs/exp_instructions/instructions-OVC.md`; this module is only the maths.

The one idea worth stating here, because it makes everything else cheap: the
anchor-centred offset map is the ordinary rate map re-indexed,

    R_A(dx, dy) = M[A + (dx, dy)]

so the per-anchor maps are three CROPS of one rate map. An object-vector cell
has the same bump in every crop; a place cell has one bump, so it appears in
exactly one crop. The criterion is therefore a correlation between crops, and
no new map machinery is needed — `scripts/trace/trace_maps.py::occupancy_and_maps`
already produces `M`.

Moser bins rate against (distance, allocentric orientation) at 2 cm x 5°. That
binning is pathological on a 14x14 grid — at distance 1 there are eight
neighbours spread over 360°, so most angular bins are empty and occupancy
varies wildly with radius. The Cartesian offset map carries the same
information with uniform occupancy, and both distance and direction remain
recoverable from it.

All functions are pure over arrays so they can be unit-tested without a
checkpoint (`tests/test_ovc_metric.py`).
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Bool, Float, Int

# World cell (x, y) lives at maps[unit, y - 1, x - 1]: get_map_bins gives 14
# bins over [0.5, 14.5], so bin index is the integer coordinate minus one.
_BIN_OFFSET = 1


def walkable_cells(*, env) -> set[tuple[int, int]]:
    """World cells the agent can occupy."""
    u = env.env.unwrapped if hasattr(env, "env") else env.unwrapped
    return {
        (x, y)
        for y in range(u.height)
        for x in range(u.width)
        if (c := u.grid.get(x, y)) is None or c.can_overlap()
    }


def offset_grid(
    *, env, anchors: list[tuple[int, int]], radius: int = 4
) -> Int[np.ndarray, "n_off 2"]:
    """Offsets landing on a walkable cell for EVERY anchor, sorted.

    Requiring all anchors is what makes the crops comparable: a correlation
    between offset maps is only meaningful over offsets that exist for both.
    The result is a hard property of the room geometry, not a tuning knob — in
    the default L-room with the 6x6 landmarks it is 56 offsets at Chebyshev <= 4.
    """
    walk = walkable_cells(env=env)
    offs = [
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if all((a[0] + dx, a[1] + dy) in walk for a in anchors)
    ]
    return np.asarray(sorted(offs), dtype=int)


def offset_maps(
    *,
    maps: Float[np.ndarray, "H ny nx"],
    anchors: list[tuple[int, int]],
    offsets: Int[np.ndarray, "n_off 2"],
) -> Float[np.ndarray, "n_anchor H n_off"]:
    """`R_A(offset)` for each anchor — the rate map read at anchor + offset."""
    ny, nx = maps.shape[1], maps.shape[2]
    out = np.full((len(anchors), maps.shape[0], len(offsets)), np.nan)
    for a, (ax, ay) in enumerate(anchors):
        for i, (dx, dy) in enumerate(offsets):
            gx, gy = ax + dx - _BIN_OFFSET, ay + dy - _BIN_OFFSET
            if 0 <= gx < nx and 0 <= gy < ny:
                out[a, :, i] = maps[:, gy, gx]
    return out


def _chebyshev(offsets: Int[np.ndarray, "n_off 2"]) -> Int[np.ndarray, "n_off"]:
    return np.abs(offsets).max(axis=1)


def vector_score(
    *,
    omaps: Float[np.ndarray, "n_anchor H n_off"],
    offsets: Int[np.ndarray, "n_off 2"],
    exclude_radius: int = 1,
) -> Float[np.ndarray, "H"]:
    """Mean pairwise correlation between anchors' offset maps, per unit.

    High = the same offset pattern recurs around every anchor, which is the
    object-vector signature. A place cell scores near zero because its single
    field lands in one crop only.

    Offsets within `exclude_radius` of the anchor are dropped: Moser discards
    fields 0-4 cm from the object centre as LEC-style object cells rather than
    vector cells. Here that class is measured separately by `radial_score`
    rather than thrown away.
    """
    keep = _chebyshev(offsets) > exclude_radius
    n_anchor, H, _ = omaps.shape
    out = np.full(H, np.nan)
    for u in range(H):
        rs = []
        for i in range(n_anchor):
            for j in range(i + 1, n_anchor):
                a, b = omaps[i, u, keep], omaps[j, u, keep]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 4:
                    continue
                av, bv = a[ok], b[ok]
                if av.std() < 1e-12 or bv.std() < 1e-12:
                    continue  # a flat crop has no pattern to match
                rs.append(float(np.corrcoef(av, bv)[0, 1]))
        if rs:
            out[u] = float(np.mean(rs))
    return out


def radial_score(
    *,
    omaps: Float[np.ndarray, "n_anchor H n_off"],
    offsets: Int[np.ndarray, "n_off 2"],
) -> Float[np.ndarray, "H"]:
    """Fraction of offset-map variance explained by distance from the anchor alone.

    The object-cell criterion: a cell firing within a radius of any anchor,
    with no directional preference, is fully described by `|offset|`. An
    object-vector cell is directional and so is not.

    Averaged over anchors, pooled across them, so a unit only scores high if
    the radial profile is shared — the same generalisation requirement the
    vector criterion imposes.
    """
    d = np.abs(offsets).max(axis=1)
    H = omaps.shape[1]
    out = np.full(H, np.nan)
    for u in range(H):
        vals = omaps[:, u, :].ravel()
        dist = np.tile(d, omaps.shape[0])
        ok = np.isfinite(vals)
        if ok.sum() < 8:
            continue
        v, dd = vals[ok], dist[ok]
        total = v.var()
        if total < 1e-12:
            continue
        pred = np.empty_like(v)
        for r in np.unique(dd):
            m = dd == r
            pred[m] = v[m].mean()
        out[u] = float(1.0 - ((v - pred) ** 2).mean() / total)
    return out


def peak_ratio(
    *,
    omaps: Float[np.ndarray, "n_anchor H n_off"],
) -> Float[np.ndarray, "H"]:
    """Peak offset-map value over the unit's own mean — a scale-free rate screen.

    Stands in for Moser's >= 2 Hz threshold, which has no meaning for units
    without a physical firing rate.
    """
    H = omaps.shape[1]
    out = np.full(H, np.nan)
    for u in range(H):
        v = omaps[:, u, :][np.isfinite(omaps[:, u, :])]
        if v.size and v.mean() > 1e-12:
            out[u] = float(v.max() / v.mean())
    return out


def inject_vector_field(
    *,
    maps: Float[np.ndarray, "H ny nx"],
    anchors: list[tuple[int, int]],
    offset: tuple[int, int],
    units: Int[np.ndarray, "k"],
    amplitude: float,
    sigma: float = 0.8,
) -> Float[np.ndarray, "H ny nx"]:
    """Positive control: add the SAME bump at `offset` from every anchor.

    Amplitude is relative to each unit's own mean rate, so the control is
    scale-free across units. Returns a copy.
    """
    out = maps.copy()
    ny, nx = maps.shape[1], maps.shape[2]
    gy, gx = np.mgrid[0:ny, 0:nx]
    for u in units:
        scale = amplitude * np.nanmean(maps[u])
        for ax, ay in anchors:
            cx, cy = ax + offset[0] - _BIN_OFFSET, ay + offset[1] - _BIN_OFFSET
            out[u] += scale * np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma**2))
    return out


def inject_place_field(
    *,
    maps: Float[np.ndarray, "H ny nx"],
    cell: tuple[int, int],
    units: Int[np.ndarray, "k"],
    amplitude: float,
    sigma: float = 0.8,
) -> Float[np.ndarray, "H ny nx"]:
    """Specificity control: add ONE bump at a fixed room location.

    This is the control that decides whether a vector-cell claim means
    anything. It is the analogue of Moser's trial 3 — a room-anchored field
    must NOT satisfy the vector criterion, or the criterion cannot tell a
    vector cell from a place cell sitting at a convenient offset.
    """
    out = maps.copy()
    ny, nx = maps.shape[1], maps.shape[2]
    gy, gx = np.mgrid[0:ny, 0:nx]
    cx, cy = cell[0] - _BIN_OFFSET, cell[1] - _BIN_OFFSET
    bump = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma**2))
    for u in units:
        out[u] += amplitude * np.nanmean(maps[u]) * bump
    return out


def classify(
    *,
    omaps: Float[np.ndarray, "n_anchor H n_off"],
    offsets: Int[np.ndarray, "n_off 2"],
    vector_threshold: float,
    radial_threshold: float,
    min_peak_ratio: float = 1.5,
    spatial_ok: Bool[np.ndarray, "H"] | None = None,
    exclude_radius: int = 1,
) -> dict:
    """Apply the screens and both criteria. Thresholds come from the shuffle null.

    `spatial_ok` is the SI screen (SI above the 95th percentile of the
    trajectory-level shuffle, from trace_maps.shuffle_null_si); passing None
    skips it, which is only appropriate for synthetic control data.
    """
    vs = vector_score(omaps=omaps, offsets=offsets, exclude_radius=exclude_radius)
    rs = radial_score(omaps=omaps, offsets=offsets)
    pr = peak_ratio(omaps=omaps)

    screen = np.isfinite(vs) & (pr >= min_peak_ratio)
    if spatial_ok is not None:
        screen &= spatial_ok

    is_ovc = screen & (vs > vector_threshold) & (rs <= radial_threshold)
    is_object = screen & (rs > radial_threshold)
    return {
        "vector_score": vs,
        "radial_score": rs,
        "peak_ratio": pr,
        "screened": screen,
        "is_ovc": is_ovc,
        "is_object_cell": is_object,
        "n_screened": int(screen.sum()),
        "frac_ovc": float(is_ovc.sum() / max(int(screen.sum()), 1)),
        "frac_object_cell": float(is_object.sum() / max(int(screen.sum()), 1)),
    }
