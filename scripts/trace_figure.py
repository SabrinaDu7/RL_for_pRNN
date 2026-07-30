"""Rate-map panels in the style of docs/trace-cells-spatial-tuning.png.

Occupancy-masked maps: bins the agent did not sample enough are left blank
(white) rather than drawn as a colour, matching how rodent rate maps are
published. One row per unit; one column per checkpoint when a series is given.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Bool, Float
from matplotlib.figure import Figure


def _draw_map(
    ax,
    rate_map: Float[np.ndarray, "ny nx"],
    *,
    vmax: float,
    cmap: str = "jet",
) -> None:
    """One occupancy-masked rate map; invalid bins stay background-coloured."""
    masked = np.ma.masked_invalid(rate_map)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(alpha=0.0)
    ax.imshow(masked, origin="upper", interpolation="nearest", cmap=cm, vmin=0, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def unit_panel(
    *,
    maps: Float[np.ndarray, "H ny nx"],
    units: np.ndarray,
    si: Float[np.ndarray, "H"] | None = None,
    n_cols: int = 8,
    obj_xy: tuple[int, int] | None = None,
    title: str = "",
) -> Figure:
    """Grid of rate maps for the given units, each on its own colour scale."""
    n = len(units)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.62 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[n:]:
        ax.axis("off")

    for ax, u in zip(axes, units):
        m = maps[u]
        vmax = np.nanmax(m)
        _draw_map(ax, m, vmax=vmax if np.isfinite(vmax) and vmax > 0 else 1.0)
        label = f"#{u}"
        if si is not None:
            label += f"  SI {si[u]:.2f}"
        ax.set_title(label, fontsize=7, pad=2)
        if obj_xy is not None:
            ax.plot(obj_xy[0] - 1, obj_xy[1] - 1, "w+", markersize=6, markeredgewidth=1.2)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    return fig


def trace_panel(
    *,
    maps_by_checkpoint: list[Float[np.ndarray, "H ny nx"]],
    labels: list[str],
    units: np.ndarray,
    obj_xy: tuple[int, int] | None = None,
    removal_after: int | None = None,
    title: str = "",
) -> Figure:
    """The trace figure: rows = units, columns = checkpoints.

    Each unit gets ONE colour scale across all its columns, so a fading field
    reads as fading rather than being renormalised away at every step.
    `removal_after` is the index of the last column with the object present;
    a divider is drawn after it.
    """
    n_rows, n_cols = len(units), len(maps_by_checkpoint)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.35 * n_cols, 1.45 * n_rows), squeeze=False
    )

    for r, u in enumerate(units):
        series = [m[u] for m in maps_by_checkpoint]
        vmax = np.nanmax([np.nanmax(s) for s in series])
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        for c, m in enumerate(series):
            ax = axes[r][c]
            _draw_map(ax, m, vmax=vmax)
            if obj_xy is not None:
                ax.plot(obj_xy[0] - 1, obj_xy[1] - 1, "w+", markersize=5,
                        markeredgewidth=1.0)
            if r == 0:
                ax.set_title(labels[c], fontsize=8, pad=3)
            if c == 0:
                ax.set_ylabel(f"#{u}", fontsize=8, rotation=0, labelpad=14,
                              va="center")

    if removal_after is not None and 0 <= removal_after < n_cols - 1:
        x = (removal_after + 1) / n_cols
        fig.add_artist(plt.Line2D([x, x], [0.02, 0.96], color="k", lw=1.5, ls="--"))

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if title else 1))
    return fig


def occupancy_figure(
    *,
    occupancy: Float[np.ndarray, "ny nx"],
    valid: Bool[np.ndarray, "ny nx"],
) -> Figure:
    """Sanity panel: raw sample counts and the mask derived from them."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 3.2))
    im = a.imshow(occupancy, origin="upper", cmap="viridis")
    a.set_title(
        f"probe occupancy  min {occupancy[valid].min():.0f} / "
        f"med {np.median(occupancy[valid]):.0f} / max {occupancy.max():.0f}",
        fontsize=8,
    )
    fig.colorbar(im, ax=a, fraction=0.046)
    # Excluded bins are drawn BLANK here, matching the rate-map panels, so the
    # two figures never disagree about which colour means "no data".
    b.imshow(np.where(valid, 1.0, np.nan), origin="upper", cmap="Greys", vmin=0, vmax=2)
    b.set_title(f"valid bins (grey): {valid.sum()}/{valid.size}", fontsize=9)
    for ax in (a, b):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig
