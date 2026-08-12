"""Tests for figure3.py helper functions."""
import hashlib
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from unittest.mock import patch

from scripts.legacy.figure3 import _names_cache_key, fetch_run_traces_by_names_cached
from scripts.legacy.figure4 import df_to_tensor, align_tensors


# ---------------------------------------------------------------------------
# _names_cache_key
# ---------------------------------------------------------------------------

def test_names_cache_key_is_deterministic():
    p1 = _names_cache_key("proj", "metric/A", "_step", ["run1", "run2"])
    p2 = _names_cache_key("proj", "metric/A", "_step", ["run1", "run2"])
    assert p1 == p2


def test_names_cache_key_order_independent():
    p1 = _names_cache_key("proj", "metric", "_step", ["run1", "run2"])
    p2 = _names_cache_key("proj", "metric", "_step", ["run2", "run1"])
    assert p1 == p2


def test_names_cache_key_differs_by_runs():
    p1 = _names_cache_key("proj", "metric", "_step", ["run1"])
    p2 = _names_cache_key("proj", "metric", "_step", ["run2"])
    assert p1 != p2


def test_names_cache_key_differs_by_metric():
    p1 = _names_cache_key("proj", "metric_A", "_step", ["run1"])
    p2 = _names_cache_key("proj", "metric_B", "_step", ["run1"])
    assert p1 != p2


# ---------------------------------------------------------------------------
# fetch_run_traces_by_names_cached
# ---------------------------------------------------------------------------

def _make_df(n_steps: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = list(range(0, n_steps * 100, 100))
    return pd.DataFrame(rng.random((1, n_steps)), columns=steps)


def test_fetch_by_names_concatenates_runs():
    run_dfs = [_make_df(seed=i) for i in range(3)]

    def mock_fetch(**kwargs):
        name = kwargs["filters"]["display_name"]
        idx = ["run0", "run1", "run2"].index(name)
        return run_dfs[idx]

    with patch("scripts.legacy.figure3.fetch_run_traces", side_effect=mock_fetch):
        result = fetch_run_traces_by_names_cached(
            "entity", "proj", "metric", "_step",
            run_names=["run0", "run1", "run2"],
            use_cache=False,
        )

    assert len(result) == 3


def test_fetch_by_names_uses_cache(tmp_path):
    from scripts.legacy.figure4 import save_df
    from scripts.legacy.figure3 import CACHE_DIR

    df = _make_df(seed=42)
    cache_path = _names_cache_key("proj", "metric", "_step", ["cached_run"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_df(df, cache_path)

    try:
        with patch("scripts.legacy.figure3.fetch_run_traces") as mock_fetch:
            result = fetch_run_traces_by_names_cached(
                "entity", "proj", "metric", "_step",
                run_names=["cached_run"],
                use_cache=True,
            )
            mock_fetch.assert_not_called()
        pd.testing.assert_frame_equal(result, df)
    finally:
        cache_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Integration: fetch -> df_to_tensor -> align_tensors
# ---------------------------------------------------------------------------

def test_pipeline_two_groups():
    """Two groups of runs produce aligned tensors usable by plot_metric."""
    import matplotlib.pyplot as plt
    from scripts.legacy.wandb_data import plot_metric

    rng = np.random.default_rng(0)
    steps = [0, 100, 200]

    def make_group_df(n_runs: int, seed: int) -> pd.DataFrame:
        r = np.random.default_rng(seed)
        return pd.DataFrame(r.random((n_runs, len(steps))), columns=steps)

    df_a = make_group_df(5, seed=0)
    df_b = make_group_df(4, seed=1)

    t_a, s_a = df_to_tensor(df_a)
    t_b, s_b = df_to_tensor(df_b)
    tensors, common = align_tensors([t_a, t_b], [s_a, s_b])

    assert common == steps
    assert tensors[0].shape == (3, 5)
    assert tensors[1].shape == (3, 4)

    fig, ax, *_ = plot_metric(
        tensors, common,
        colors=["purple", "blue"],
        labels=["Curious", "Random"],
        xlabel="Step", ylabel="Value",
        use_std=False,
    )
    assert fig is not None
    plt.close(fig)
