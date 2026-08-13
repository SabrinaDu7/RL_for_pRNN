"""The object-vector criterion must separate vector cells from place cells.

These run on synthetic maps, so the criterion is gated by pytest rather than
only by its own controls at analysis time. The specificity test is the one that
matters: a room-anchored place field must NOT satisfy the vector criterion, or
the metric cannot distinguish a vector cell from a place cell that happens to
sit at a convenient offset from one anchor.
"""

import numpy as np
import pytest

from scripts.trace.ovc_metric import (
    classify,
    inject_place_field,
    inject_vector_field,
    offset_maps,
    peak_ratio,
    radial_score,
    vector_score,
)

ANCHORS = [(4, 4), (11, 4), (4, 11)]
NY = NX = 14


def _offsets(radius=4):
    return np.asarray(
        sorted((dx, dy) for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1)), dtype=int
    )


def _base_maps(n_units=6, seed=0):
    """Low-amplitude noise: no structure, so any score comes from what we inject."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.4, 0.6, size=(n_units, NY, NX))


def test_injected_vector_field_scores_high():
    offs = _offsets()
    maps = inject_vector_field(
        maps=_base_maps(), anchors=ANCHORS, offset=(2, 2),
        units=np.arange(6), amplitude=6.0,
    )
    vs = vector_score(omaps=offset_maps(maps=maps, anchors=ANCHORS, offsets=offs),
                      offsets=offs)
    assert np.nanmin(vs) > 0.5, f"vector field should correlate across anchors, got {vs}"


def test_place_field_does_not_score_as_vector():
    """The specificity control - the analogue of Moser's trial 3."""
    offs = _offsets()
    maps = inject_place_field(
        maps=_base_maps(), cell=(6, 6), units=np.arange(6), amplitude=6.0,
    )
    vs = vector_score(omaps=offset_maps(maps=maps, anchors=ANCHORS, offsets=offs),
                      offsets=offs)
    assert np.nanmax(vs) < 0.3, f"a single room-anchored field must not look like a vector cell, got {vs}"


def test_vector_beats_place_on_the_same_base():
    """Direct contrast, so the test does not depend on absolute thresholds."""
    offs = _offsets()
    base = _base_maps()
    vec = offset_maps(maps=inject_vector_field(maps=base, anchors=ANCHORS, offset=(2, 2),
                                               units=np.arange(6), amplitude=6.0),
                      anchors=ANCHORS, offsets=offs)
    plc = offset_maps(maps=inject_place_field(maps=base, cell=(6, 6),
                                              units=np.arange(6), amplitude=6.0),
                      anchors=ANCHORS, offsets=offs)
    assert np.nanmean(vector_score(omaps=vec, offsets=offs)) > \
           np.nanmean(vector_score(omaps=plc, offsets=offs)) + 0.3


def test_radial_field_is_called_an_object_cell_not_a_vector_cell():
    """A ring around every anchor is directionless: high radial, and it is the
    class Moser discards as LEC-style rather than vector."""
    offs = _offsets()
    maps = _base_maps()
    gy, gx = np.mgrid[0:NY, 0:NX]
    for u in range(maps.shape[0]):
        for ax, ay in ANCHORS:
            d = np.sqrt((gx - (ax - 1)) ** 2 + (gy - (ay - 1)) ** 2)
            maps[u] += 4.0 * np.exp(-((d - 2.0) ** 2) / (2 * 0.7**2))
    om = offset_maps(maps=maps, anchors=ANCHORS, offsets=offs)
    assert np.nanmean(radial_score(omaps=om, offsets=offs)) > 0.5


def test_directional_field_is_not_radially_explained():
    offs = _offsets()
    maps = inject_vector_field(maps=_base_maps(), anchors=ANCHORS, offset=(3, 0),
                               units=np.arange(6), amplitude=6.0)
    om = offset_maps(maps=maps, anchors=ANCHORS, offsets=offs)
    rs, vs = radial_score(omaps=om, offsets=offs), vector_score(omaps=om, offsets=offs)
    assert np.nanmean(vs) > np.nanmean(rs), "a one-sided bump is directional, not radial"


def test_flat_unit_is_rejected_by_the_rate_screen():
    offs = _offsets()
    maps = np.full((3, NY, NX), 0.5)
    om = offset_maps(maps=maps, anchors=ANCHORS, offsets=offs)
    assert np.all(peak_ratio(omaps=om) < 1.5)
    out = classify(omaps=om, offsets=offs, vector_threshold=0.4, radial_threshold=0.5)
    assert out["n_screened"] == 0, "a constant unit has no field and must not be classified"


def test_offset_maps_read_the_right_cells():
    """Guards the world->bin convention: maps[unit, y-1, x-1]."""
    maps = np.zeros((1, NY, NX))
    maps[0, 6 - 1, 7 - 1] = 9.0                      # world cell (7, 6)
    offs = np.asarray([[3, 2]], dtype=int)           # (4,4) + (3,2) = (7,6)
    om = offset_maps(maps=maps, anchors=[(4, 4)], offsets=offs)
    assert om[0, 0, 0] == pytest.approx(9.0)


def test_nan_bins_do_not_crash_and_are_excluded():
    offs = _offsets()
    maps = inject_vector_field(maps=_base_maps(), anchors=ANCHORS, offset=(2, 2),
                               units=np.arange(6), amplitude=6.0)
    maps[:, 0, :] = np.nan                            # an unvisited row
    om = offset_maps(maps=maps, anchors=ANCHORS, offsets=offs)
    vs = vector_score(omaps=om, offsets=offs)
    assert np.isfinite(vs).all() and np.nanmin(vs) > 0.4
