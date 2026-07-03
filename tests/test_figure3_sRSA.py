"""Tests for figure3_sRSA.py."""
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt

from scripts.figure3_sRSA import (
    plot_sRSA,
    plot_sRSA_independent,
    _bootstrap_ci,
    _welch_ttest_per_step,
    METRIC,
    YLABEL,
)
from scripts.figure4 import df_to_tensor, align_tensors
from scripts.wandb_data import plot_metric


def _make_df(n_runs: int = 3, n_steps: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = list(range(0, n_steps * 1000, 1000))
    return pd.DataFrame(rng.random((n_runs, n_steps)), columns=steps)


def test_metric_constant():
    assert METRIC == "sRSA_onPolicy"


def test_ylabel_constant():
    assert YLABEL == "Spatial Representation Similarity"


def test_plot_sRSA_runs_with_mock(tmp_path):
    """plot_sRSA fetches data, calls plot_metric, and saves the figure."""
    df_curious = _make_df(n_runs=3, seed=0)
    df_random = _make_df(n_runs=3, seed=1)

    with (
        patch("scripts.figure3_sRSA.fetch_run_traces_by_names_cached",
              side_effect=[df_curious, df_random]) as mock_fetch,
    ):
        plot_sRSA(
            entity="ent",
            project="proj",
            curious_runs=["run_c1", "run_c2", "run_c3"],
            rand_runs=["run_r1", "run_r2", "run_r3"],
            use_cache=False,
            save_dir=tmp_path,
        )
        assert mock_fetch.call_count == 2

    saved = tmp_path / "sRSA_onPolicy.png"
    assert saved.exists()
    plt.close("all")


# ---------------------------------------------------------------------------
# _bootstrap_ci
# ---------------------------------------------------------------------------

def test_bootstrap_ci_shape():
    data = np.random.default_rng(0).random((10, 5))
    lo, hi = _bootstrap_ci(data, n_bootstrap=50)
    assert lo.shape == (10,)
    assert hi.shape == (10,)


def test_bootstrap_ci_non_negative():
    """Lower and upper errors should be >= 0 (they are distances from the mean)."""
    data = np.random.default_rng(1).random((8, 6))
    lo, hi = _bootstrap_ci(data, n_bootstrap=100)
    assert np.all(lo >= 0)
    assert np.all(hi >= 0)


def test_bootstrap_ci_wider_for_more_variance():
    """Higher-variance data should produce wider CIs."""
    rng = np.random.default_rng(2)
    low_var  = rng.random((5, 20)) * 0.01
    high_var = rng.random((5, 20)) * 10.0
    lo_lv, hi_lv = _bootstrap_ci(low_var,  n_bootstrap=200)
    lo_hv, hi_hv = _bootstrap_ci(high_var, n_bootstrap=200)
    assert np.all(lo_hv > lo_lv)
    assert np.all(hi_hv > hi_lv)


# ---------------------------------------------------------------------------
# _welch_ttest_per_step
# ---------------------------------------------------------------------------

def test_welch_ttest_lengths():
    rng = np.random.default_rng(1)
    a = rng.random((6, 5))
    b = rng.random((6, 4))
    steps = list(range(6))
    p_vals, dfs, t_stats = _welch_ttest_per_step(a, b, steps)
    assert len(p_vals) == len(dfs) == len(t_stats) == 6


def test_welch_ttest_identical_data_p_is_one():
    """Same data in both groups → p-value should be 1."""
    data = np.random.default_rng(2).random((3, 5))
    p_vals, _, _ = _welch_ttest_per_step(data, data.copy(), list(range(3)))
    for p in p_vals:
        assert abs(p - 1.0) < 1e-10


def test_welch_ttest_clearly_different_groups():
    """Well-separated groups → p-values should all be small."""
    rng = np.random.default_rng(3)
    a = rng.random((4, 10))           # values in [0, 1]
    b = rng.random((4, 10)) + 100.0   # shifted far away
    p_vals, _, _ = _welch_ttest_per_step(a, b, list(range(4)))
    assert all(p < 0.001 for p in p_vals)


# ---------------------------------------------------------------------------
# plot_sRSA_independent
# ---------------------------------------------------------------------------

def test_plot_sRSA_independent_saves_file(tmp_path):
    df_curious = _make_df(n_runs=3, seed=0)
    df_random  = _make_df(n_runs=3, seed=1)

    with patch("scripts.figure3_sRSA.fetch_run_traces_by_names_cached",
               side_effect=[df_curious, df_random]):
        plot_sRSA_independent(use_cache=False, save_dir=tmp_path)

    assert (tmp_path / "sRSA_onPolicy_independent.png").exists()
    plt.close("all")


def test_plot_sRSA_independent_has_two_labeled_lines(tmp_path):
    df_curious = _make_df(n_runs=4, seed=20)
    df_random  = _make_df(n_runs=4, seed=21)

    figs_before = set(plt.get_fignums())
    with patch("scripts.figure3_sRSA.fetch_run_traces_by_names_cached",
               side_effect=[df_curious, df_random]):
        plot_sRSA_independent(use_cache=False, save_dir=tmp_path)

    new_fig_num = (set(plt.get_fignums()) - figs_before).pop()
    ax = plt.figure(new_fig_num).axes[0]
    labeled = [l for l in ax.get_lines() if l.get_label() in ("Curious", "Random")]
    assert len(labeled) == 2
    plt.close("all")


