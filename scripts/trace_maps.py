"""Rate maps, occupancy masking and spatial-tuning statistics for pRNN units.

Deliberately independent of prnn's calculateSpatialRepresentation, which
collects its own rollout and zeroes (rather than masks) the SI of low-activity
units - that puts a spurious spike at exactly 0 in any SI distribution and
ranks silent units identically to untuned ones. Here undersampled bins and
inactive units become NaN and are excluded from the statistics, not just from
the picture.

Binning is taken from `env.get_map_bins()` so it matches the rest of the
project (LRoom: 14 x 14 over (0.5, 14.5)).
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Bool, Float
from scipy import sparse

MIN_OCCUPANCY = 20      # samples per bin below which a bin is not estimated
MIN_ACTIVE_SAMPLES = 200  # timesteps with h > 0 below which a unit is silent


def bin_maps(
    *,
    h: Float[np.ndarray, "N H"],
    bin_index: np.ndarray,
    n_bins: int,
    shape: tuple[int, int],
    min_occupancy: int = MIN_OCCUPANCY,
) -> tuple[
    Float[np.ndarray, "H ny nx"], Float[np.ndarray, "ny nx"], Bool[np.ndarray, "ny nx"]
]:
    """Mean activity per bin from precomputed flat bin indices.

    The shared core of occupancy_and_maps (spatial frame) and view_frame_maps
    (object-relative frame) - the only difference between the two is what the
    bin index means.
    """
    occupancy = np.bincount(bin_index, minlength=n_bins).astype(np.float64)
    onehot = sparse.csr_matrix(
        (np.ones(len(bin_index)), (bin_index, np.arange(len(bin_index)))),
        shape=(n_bins, len(bin_index)),
    )
    totals = (onehot @ h).T
    valid = occupancy >= min_occupancy
    with np.errstate(invalid="ignore", divide="ignore"):
        maps = totals / occupancy[None, :]
    maps[:, ~valid] = np.nan
    return maps.reshape(-1, *shape), occupancy.reshape(shape), valid.reshape(shape)


def view_frame_maps(
    *,
    h: Float[np.ndarray, "N H"],
    vx: np.ndarray,
    vy: np.ndarray,
    view_size: int = 7,
    min_occupancy: int = MIN_OCCUPANCY,
) -> tuple[
    Float[np.ndarray, "H v v"], Float[np.ndarray, "v v"], Bool[np.ndarray, "v v"]
]:
    """Tuning to a landmark's position in the agent's egocentric view.

    This is the object-vector frame: instead of "where is the agent in the
    room", the bin is "where is the object relative to the agent". Only
    timesteps with the landmark inside the view contribute - the caller must
    pass the already-filtered rows.
    """
    assert h.shape[0] == vx.shape[0] == vy.shape[0], "h, vx, vy must align"
    return bin_maps(
        h=h,
        bin_index=(vy * view_size + vx).astype(np.int64),
        n_bins=view_size * view_size,
        shape=(view_size, view_size),
        min_occupancy=min_occupancy,
    )


def occupancy_and_maps(
    *,
    h: Float[np.ndarray, "N H"],
    pos: Float[np.ndarray, "N 2"],
    env,
    min_occupancy: int = MIN_OCCUPANCY,
) -> tuple[
    Float[np.ndarray, "H ny nx"], Float[np.ndarray, "ny nx"], Bool[np.ndarray, "ny nx"]
]:
    """Mean activity per spatial bin, with an occupancy count and a valid mask.

    Returns (maps, occupancy, valid). `maps` is NaN wherever `valid` is False,
    so undersampled bins never enter a statistic or a figure. Maps are indexed
    [unit, y, x] - ready for imshow without a transpose.
    """
    nb_x, nb_y, minmax = env.get_map_bins()
    x_edges = np.linspace(minmax[0], minmax[1], nb_x + 1)
    y_edges = np.linspace(minmax[2], minmax[3], nb_y + 1)

    ix = np.clip(np.digitize(pos[:, 0], x_edges) - 1, 0, nb_x - 1)
    iy = np.clip(np.digitize(pos[:, 1], y_edges) - 1, 0, nb_y - 1)
    flat = iy * nb_x + ix

    n_bins = nb_x * nb_y
    occupancy = np.bincount(flat, minlength=n_bins).astype(np.float64)
    # Sum activity per bin as one sparse (n_bins x N) @ (N x H) product. A
    # per-unit bincount loop is ~500x slower, which matters for the shuffles.
    onehot = sparse.csr_matrix(
        (np.ones(len(flat)), (flat, np.arange(len(flat)))), shape=(n_bins, len(flat))
    )
    totals = (onehot @ h).T  # (H, n_bins)

    valid = occupancy >= min_occupancy
    with np.errstate(invalid="ignore", divide="ignore"):
        maps = totals / occupancy[None, :]
    maps[:, ~valid] = np.nan

    return (
        maps.reshape(-1, nb_y, nb_x),
        occupancy.reshape(nb_y, nb_x),
        valid.reshape(nb_y, nb_x),
    )


def spatial_info(
    *,
    maps: Float[np.ndarray, "H ny nx"],
    occupancy: Float[np.ndarray, "ny nx"],
    weighting: str = "occupancy",
) -> Float[np.ndarray, "H"]:
    """Skaggs spatial information in bits per sample, over valid bins only.

    SI = sum_i p_i * (r_i / r) * log2(r_i / r), with p_i the bin weight.
    NaN for units whose mean rate is zero.

    weighting="occupancy" is the standard Skaggs form, p_i proportional to
    dwell time. That is the right estimator for a freely-behaving animal, but
    the random-action probe oversamples corners ~9x (it pushes forward 60% of
    the time and parks against walls), so it inflates the SI of wall- and
    corner-tuned units. weighting="uniform" weights every valid bin equally,
    which removes that bias at the cost of trusting sparsely-sampled bins as
    much as dense ones. Report both; disagreement between them IS the bias.
    """
    valid = np.isfinite(maps[0]) & (occupancy > 0)
    if weighting == "uniform":
        p = np.full(int(valid.sum()), 1.0 / int(valid.sum()))
    elif weighting == "occupancy":
        p = occupancy[valid] / occupancy[valid].sum()
    else:
        raise ValueError(f"weighting must be 'occupancy' or 'uniform', got {weighting!r}")
    r = maps[:, valid]                                   # (H, n_valid)
    mean_rate = (r * p[None, :]).sum(axis=1)             # (H,)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = r / mean_rate[:, None]
        terms = p[None, :] * ratio * np.log2(ratio)
    terms[~np.isfinite(terms)] = 0.0                     # r_i = 0 contributes 0

    si = terms.sum(axis=1)
    si[mean_rate <= 0] = np.nan
    return si


def silent_units(
    *, h: Float[np.ndarray, "N H"], min_active: int = MIN_ACTIVE_SAMPLES
) -> Bool[np.ndarray, "H"]:
    """Units active in fewer than `min_active` samples - reported, not zeroed."""
    return (h > 0).sum(axis=0) < min_active


def shuffle_null_si(
    *,
    h: Float[np.ndarray, "B T H"],
    pos: Float[np.ndarray, "B T 2"],
    env,
    n_shuffles: int = 200,
    min_occupancy: int = MIN_OCCUPANCY,
    seed: int = 0,
) -> Float[np.ndarray, "n_shuffles H"]:
    """Null SI from pairing each trajectory's activity with another's positions.

    Preserves each unit's rate, distribution and within-trajectory
    autocorrelation while destroying the activity-position correspondence.
    Preferred over a circular time shift here because the data is many short
    trajectories with a state reset between them, so wrapping would glue
    unrelated states together.
    """
    rng = np.random.default_rng(seed)
    B = h.shape[0]
    H = h.shape[-1]
    out = np.empty((n_shuffles, H))

    for s in range(n_shuffles):
        perm = rng.permutation(B)
        # derangement: no trajectory keeps its own positions
        fixed = np.nonzero(perm == np.arange(B))[0]
        if fixed.size:
            perm[fixed] = perm[(fixed + 1) % B]
        shuffled_pos = pos[perm].reshape(-1, 2)
        maps, occ, _ = occupancy_and_maps(
            h=h.reshape(-1, H), pos=shuffled_pos, env=env, min_occupancy=min_occupancy
        )
        out[s] = spatial_info(maps=maps, occupancy=occ)

    return out


def split_half_stability(
    *,
    h: Float[np.ndarray, "B T H"],
    pos: Float[np.ndarray, "B T 2"],
    env,
    min_occupancy: int = MIN_OCCUPANCY,
) -> Float[np.ndarray, "H"]:
    """Pearson r between odd- and even-trajectory maps, over commonly valid bins.

    Answers "is this map stable enough to plot and to compare across
    checkpoints", which the shuffle null does not.
    """
    H = h.shape[-1]
    halves = []
    for sel in (slice(0, None, 2), slice(1, None, 2)):
        maps, _, valid = occupancy_and_maps(
            h=h[sel].reshape(-1, H),
            pos=pos[sel].reshape(-1, 2),
            env=env,
            min_occupancy=min_occupancy,
        )
        halves.append((maps, valid))

    (m0, v0), (m1, v1) = halves
    both = v0 & v1
    a = m0[:, both]
    b = m1[:, both]

    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b).sum(axis=1) / denom
    r[~np.isfinite(r)] = np.nan
    return r


def object_modulation(
    *,
    maps: Float[np.ndarray, "H ny nx"],
    obj_xy: tuple[int, int],
    env,
    radius: float = 2.0,
) -> Float[np.ndarray, "H"]:
    """(mean rate near the object - mean rate elsewhere) / their sum, per unit.

    Bounded in [-1, 1]; positive means the unit prefers the object's
    neighbourhood. `obj_xy` is in MiniGrid world coordinates.
    """
    nb_x, nb_y, minmax = env.get_map_bins()
    xs = np.linspace(minmax[0], minmax[1], nb_x + 1)[:-1] + 0.5 * (
        (minmax[1] - minmax[0]) / nb_x
    )
    ys = np.linspace(minmax[2], minmax[3], nb_y + 1)[:-1] + 0.5 * (
        (minmax[3] - minmax[2]) / nb_y
    )
    gx, gy = np.meshgrid(xs, ys)
    near = ((gx - obj_xy[0]) ** 2 + (gy - obj_xy[1]) ** 2) <= radius**2

    finite = np.isfinite(maps)
    inside = np.where(finite & near[None], maps, np.nan)
    outside = np.where(finite & ~near[None], maps, np.nan)

    with np.errstate(invalid="ignore"):
        a = np.nanmean(inside, axis=(1, 2))
        b = np.nanmean(outside, axis=(1, 2))
        mod = (a - b) / (a + b)
    mod[~np.isfinite(mod)] = np.nan
    return mod
