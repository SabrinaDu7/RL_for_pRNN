"""A metric for object / trace-cell coding in the pRNN hidden state.

WHAT A TRACE CELL IS
--------------------
In hippocampus/LEC an object-trace cell has a place field at the location where
an object was, which appears with object experience and persists after removal
(`docs/trace-cells-spatial-tuning.png`). The observable is therefore a **change
in receptive-field strength at one specific location**, relative to the same
unit before exposure.

THE METRIC
----------
For unit `u` and candidate cell `c`, the **object-field gain** is

    g(u, c) = mean rate over a disc of radius r around c   /   mean rate over all valid bins

i.e. how strongly the unit prefers `c` relative to its own overall activity. It
is scale-free, so a unit that simply gets louder everywhere scores zero change.

The **trace score** is the change from pre-exposure to post-exposure:

    dg(u, c) = g_post(u, c) - g_pre(u, c)

WHY A WITHIN-UNIT NULL
----------------------
`dg` at the object alone is not interpretable: maps drift, and some locations
drift more than others. So `dg(u, c)` is computed at **every** walkable cell,
and the unit's score is the object cell's **percentile within its own
distribution**. This controls for (a) global rate changes, (b) the unit's own
map structure, and (c) location-specific drift shared across units — the last
being exactly what produced the spurious (14,7) result in
`exp_object_trace_cells_2026-07-30.md`.

A unit is called an **object cell** when that percentile exceeds 95. Under the
null the expected fraction is 5%, so the population statistic is a binomial
test of "fraction of object cells > 0.05".

VALIDATION IS PART OF THE MODULE
--------------------------------
`validate()` runs two controls, and both must be reported alongside any result:

  negative — score two INDEPENDENT map estimates of the same net (odd vs even
             probe trajectories) against each other. Should give ~5% object
             cells. Scoring a map against itself is degenerate (dg == 0) and
             is not a null.
  positive — inject a synthetic Gaussian field of known amplitude at the object
             location into a known fraction of units, and measure detection.
             This yields a **sensitivity floor**: the smallest field the metric
             can see. Without it, "we found nothing" is not a bound.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Bool, Float

DEFAULT_RADIUS = 2.0
OBJECT_CELL_PCTILE = 95.0


def _disc_masks(
    *, env, cells: list[tuple[int, int]], radius: float
) -> Float[np.ndarray, "n_cells ny nx"]:
    """Boolean disc around each candidate cell, in map coordinates."""
    nb_x, nb_y, minmax = env.get_map_bins()
    xs = np.linspace(minmax[0], minmax[1], nb_x + 1)[:-1] + 0.5 * (minmax[1] - minmax[0]) / nb_x
    ys = np.linspace(minmax[2], minmax[3], nb_y + 1)[:-1] + 0.5 * (minmax[3] - minmax[2]) / nb_y
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([((gx - c[0]) ** 2 + (gy - c[1]) ** 2) <= radius**2 for c in cells])


def field_gain(
    *,
    maps: Float[np.ndarray, "H ny nx"],
    discs: Bool[np.ndarray, "n_cells ny nx"],
) -> Float[np.ndarray, "n_cells H"]:
    """Relative receptive-field strength of each unit at each candidate cell.

    Scale-free: a unit whose rate doubles everywhere scores the same gain.
    NaN map bins (occupancy-masked) are excluded from both numerator and
    denominator.
    """
    finite = np.isfinite(maps)
    safe = np.where(finite, maps, 0.0)
    overall = safe.sum(axis=(1, 2)) / np.maximum(finite.sum(axis=(1, 2)), 1)  # (H,)

    out = np.empty((len(discs), maps.shape[0]))
    for i, d in enumerate(discs):
        n = (finite & d[None]).sum(axis=(1, 2))
        s = np.where(finite & d[None], safe, 0.0).sum(axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = (s / np.maximum(n, 1)) / overall
    out[~np.isfinite(out)] = np.nan
    return out


def trace_scores(
    *,
    maps_pre: Float[np.ndarray, "H ny nx"],
    maps_post: Float[np.ndarray, "H ny nx"],
    env,
    cells: list[tuple[int, int]],
    obj_cell: tuple[int, int],
    radius: float = DEFAULT_RADIUS,
) -> dict:
    """Per-unit trace score, plus the population summary.

    Returns:
        percentile   (H,)  object cell's rank among all cells, per unit
        is_object_cell (H,) percentile > OBJECT_CELL_PCTILE
        frac         float fraction of units called object cells
        p_binom      float P(frac this high | 5% null), one-sided
        dg_obj       (H,)  raw gain change at the object, for effect size
    """
    from scipy import stats

    discs = _disc_masks(env=env, cells=cells, radius=radius)
    dg = field_gain(maps=maps_post, discs=discs) - field_gain(maps=maps_pre, discs=discs)
    idx = cells.index(obj_cell)

    valid = np.isfinite(dg)
    pct = np.full(dg.shape[1], np.nan)
    for u in range(dg.shape[1]):
        col = dg[valid[:, u], u]
        if col.size and np.isfinite(dg[idx, u]):
            pct[u] = 100.0 * np.mean(col < dg[idx, u])

    is_cell = pct > OBJECT_CELL_PCTILE
    n_scored = int(np.isfinite(pct).sum())
    n_cell = int(is_cell.sum())
    frac = n_cell / max(n_scored, 1)
    p = stats.binomtest(n_cell, n_scored, 0.05, alternative="greater").pvalue if n_scored else 1.0
    return {
        "percentile": pct,
        "is_object_cell": is_cell,
        "n_scored": n_scored,
        "n_object_cells": n_cell,
        "frac": frac,
        "p_binom": float(p),
        "dg_obj": dg[idx],
    }


def inject_field(
    *,
    maps: Float[np.ndarray, "H ny nx"],
    env,
    obj_cell: tuple[int, int],
    amplitude: float,
    fraction: float,
    sigma: float = 1.5,
    seed: int = 0,
) -> tuple[Float[np.ndarray, "H ny nx"], np.ndarray]:
    """Add a Gaussian bump at `obj_cell` to a random `fraction` of units.

    `amplitude` is in units of each unit's own mean rate, so it is comparable
    across units. Returns (maps, indices of injected units).
    """
    nb_x, nb_y, minmax = env.get_map_bins()
    xs = np.linspace(minmax[0], minmax[1], nb_x + 1)[:-1] + 0.5 * (minmax[1] - minmax[0]) / nb_x
    ys = np.linspace(minmax[2], minmax[3], nb_y + 1)[:-1] + 0.5 * (minmax[3] - minmax[2]) / nb_y
    gx, gy = np.meshgrid(xs, ys)
    bump = np.exp(-(((gx - obj_cell[0]) ** 2 + (gy - obj_cell[1]) ** 2) / (2 * sigma**2)))

    rng = np.random.default_rng(seed)
    H = maps.shape[0]
    chosen = rng.choice(H, size=max(1, int(round(fraction * H))), replace=False)
    finite = np.isfinite(maps)
    mean_rate = np.where(finite, maps, 0.0).sum(axis=(1, 2)) / np.maximum(
        finite.sum(axis=(1, 2)), 1)

    out = maps.copy()
    out[chosen] = out[chosen] + amplitude * mean_rate[chosen, None, None] * bump[None]
    out[~finite] = np.nan
    return out, chosen


def location_control_matrix(
    *,
    maps_pre: Float[np.ndarray, "H ny nx"],
    maps_post_by_object: dict,
    env,
    cells: list[tuple[int, int]],
    radius: float = DEFAULT_RADIUS,
) -> dict:
    """The decisive control: is the effect at L specific to the object being AT L?

    The within-unit null in `trace_scores` cannot see drift that is shared
    across units at one location — which is exactly what produced the spurious
    (14,7) result. So score every candidate location under every run, and take

        excess(L) = frac(L | object at L) - mean over M != L of frac(L | object at M)

    A real object effect is positive; pure location drift cancels.
    """
    locs = list(maps_post_by_object)
    mat = np.zeros((len(locs), len(locs)))          # [run, scored location]
    for i, run_loc in enumerate(locs):
        for j, score_loc in enumerate(locs):
            mat[i, j] = trace_scores(
                maps_pre=maps_pre, maps_post=maps_post_by_object[run_loc], env=env,
                cells=cells, obj_cell=score_loc, radius=radius)["frac"]
    excess = {}
    for j, L in enumerate(locs):
        off = [mat[i, j] for i in range(len(locs)) if i != j]
        excess[L] = float(mat[j, j] - np.mean(off))
    return {"locs": locs, "matrix": mat, "excess": excess}


def validate(
    *,
    maps_pre: Float[np.ndarray, "H ny nx"],
    env,
    cells: list[tuple[int, int]],
    obj_cell: tuple[int, int],
    amplitudes: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.25),
    maps_pre_split: tuple | None = None,
    fraction: float = 0.1,
    radius: float = DEFAULT_RADIUS,
) -> dict:
    """Negative control + sensitivity curve. Report both with every result."""
    if maps_pre_split is None:
        neg = {"frac": float("nan"), "p_binom": float("nan")}  # no independent estimate given
    else:
        neg = trace_scores(maps_pre=maps_pre_split[0], maps_post=maps_pre_split[1], env=env,
                           cells=cells, obj_cell=obj_cell, radius=radius)
    rows = []
    for a in amplitudes:
        inj, chosen = inject_field(maps=maps_pre, env=env, obj_cell=obj_cell,
                                   amplitude=a, fraction=fraction)
        r = trace_scores(maps_pre=maps_pre, maps_post=inj, env=env, cells=cells,
                         obj_cell=obj_cell, radius=radius)
        recall = float(r["is_object_cell"][chosen].mean())
        rows.append({"amplitude": a, "frac": r["frac"], "recall": recall,
                     "p_binom": r["p_binom"]})
    return {"negative_frac": neg["frac"], "negative_p": neg["p_binom"], "sensitivity": rows}
