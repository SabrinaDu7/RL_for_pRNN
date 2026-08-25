"""The occupancy figure must draw the measurement, not be it.

`occupancy_counts` was extracted from inside `get_occupancy_fig`, which built
the array and threw it away into a plotly figure - so the only way to analyse
occupancy afterwards was to scrape the figure's `z` back out of wandb. A number
recovered from a picture. These pin that the array and the picture cannot
disagree, and that the two index conventions are what the docstrings say.
"""

import numpy as np
import plotly.graph_objects as go

from curious_george import get_occupancy_fig, occupancy_counts


class _Env:
    width, height = 16, 16


class _Algo:
    """Only what occupancy_counts reads: env extent, locs, directions."""

    env = _Env()

    def __init__(self, locs, directions):
        self.locs, self.directions = locs, directions


def test_counts_land_where_the_agent_was():
    algo = _Algo(locs=[(1, 1), (1, 1), (3, 5)], directions=[0, 0, 2])
    occ = occupancy_counts(algo, 3)
    assert occ.shape == (4, 14, 14)
    assert occ.sum() == 3
    assert occ[0, 0, 0] == 2, "MiniGrid (1,1) is index [0, 0], and it was visited twice"
    assert occ[2, 2, 4] == 1, "MiniGrid (3,5) at head-direction 2 is [2, 2, 4]"


def test_indexing_is_x_then_y():
    """`[hd, x, y]`, x horizontal - matching get_walkable_mask and MiniGrid's
    grid.get(x, y). An asymmetric visit settles it."""
    occ = occupancy_counts(_Algo(locs=[(3, 9)], directions=[1]), 1)
    assert occ[1, 2, 8] == 1
    assert occ[1, 8, 2] == 0


def test_timesteps_bounds_what_is_counted():
    algo = _Algo(locs=[(1, 1), (2, 2), (3, 3)], directions=[0, 1, 2])
    assert occupancy_counts(algo, 2).sum() == 2


def test_the_figure_draws_exactly_these_counts():
    """The gate that matters: the picture is downstream of the array, so a
    figure and an analysis reading the same run cannot disagree.

    The figure transposes for display, which is the convention anyone reading
    the logged plotly JSON inherits - so this also pins that `z` is [y, x]."""
    algo = _Algo(locs=[(1, 1), (1, 1), (3, 5), (7, 2)], directions=[0, 0, 2, 1])
    occ = occupancy_counts(algo, 4)
    fig = get_occupancy_fig(algo, 4)
    assert isinstance(fig, go.Figure)
    for hd in range(4):
        drawn = np.asarray(fig.data[hd].z)
        assert np.array_equal(drawn, occ[hd].T), f"head-direction {hd}: figure != counts"
