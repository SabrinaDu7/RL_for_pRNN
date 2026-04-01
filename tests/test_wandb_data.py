"""Tests for scripts.wandb_data module."""

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import json
import os

import numpy as np
import plotly.graph_objects as go

import torch

from scripts.wandb_data import (
    OccupancyData,
    SubroomData,
    _build_filters,
    _download_and_extract,
    _draw_significance_bracket,
    _extract_heatmap_grids,
    _fetch_history,
    _fetch_plotly_file,
    _histogram_to_subroom_counts,
    _resolve_config_value,
    _scan_occupancy_refs,
    _pval_stars,
    _sig_label,
    _subroom_pct_array,
    _subroom_percentages,
    _unwrap_wandb_config,
    bootstrap_ci,
    fetch_occupancy_grids,
    fetch_run_traces,
    fetch_subroom_counts,
    load_subroom_data,
    pairwise_ttest,
    plot_occupancy_average,
    plot_metric,
    plot_subroom_percentage,
    plot_traces,
    prob_roi,
    runs_per_step,
    save_subroom_data,
)


# ---------------------------------------------------------------------------
# _unwrap_wandb_config
# ---------------------------------------------------------------------------

def test_unwrap_wandb_config_strips_value_envelope():
    raw = {"exp": {"value": {"seed": 42}}, "rl": {"value": {"lr": 0.001}}}
    assert _unwrap_wandb_config(raw) == {"exp": {"seed": 42}, "rl": {"lr": 0.001}}


def test_unwrap_wandb_config_passes_through_plain_dicts():
    plain = {"exp": {"seed": 42}, "rl": {"lr": 0.001}}
    assert _unwrap_wandb_config(plain) == plain


def test_unwrap_wandb_config_skips_wandb_internal():
    """_wandb key has {"value": ...} but also other keys — should not unwrap."""
    raw = {"_wandb": {"value": {"cli_version": "0.22"}, "extra": True},
           "exp": {"value": {"seed": 1}}}
    result = _unwrap_wandb_config(raw)
    assert result["_wandb"] == raw["_wandb"]  # not unwrapped
    assert result["exp"] == {"seed": 1}       # unwrapped


# ---------------------------------------------------------------------------
# _resolve_config_value
# ---------------------------------------------------------------------------

def test_resolve_config_value_flat():
    config = {"seed": 42, "lr": 0.001}
    assert _resolve_config_value(config, "seed") == 42


def test_resolve_config_value_flat_dotted_key():
    """Wandb flattens Hydra configs so 'exp.seed' is a top-level key."""
    config = {"exp.seed": 42, "rl.lr": 0.001}
    assert _resolve_config_value(config, "exp.seed") == 42
    assert _resolve_config_value(config, "rl.lr") == 0.001


def test_resolve_config_value_wandb_envelope():
    """Config from wandb API with {"value": ...} wrapping."""
    config = {
        "exp": {"value": {"seed": 42, "curious_agent": True}},
        "tasks": {"value": {"testing": {"start_low_bound": [1, 1]}}},
    }
    assert _resolve_config_value(config, "exp.seed") == 42
    assert _resolve_config_value(config, "exp.curious_agent") is True
    assert _resolve_config_value(config, "tasks.testing.start_low_bound") == (1, 1)


def test_resolve_config_value_nested():
    config = {"exp": {"seed": 42, "env_name": "LRoom"}, "rl": {"lr": 0.001}}
    assert _resolve_config_value(config, "exp.seed") == 42
    assert _resolve_config_value(config, "rl.lr") == 0.001


def test_resolve_config_value_deep():
    config = {"a": {"b": {"c": {"d": 99}}}}
    assert _resolve_config_value(config, "a.b.c.d") == 99


def test_resolve_config_value_missing_key():
    config = {"exp": {"seed": 42}}
    with pytest.raises(KeyError, match="not found"):
        _resolve_config_value(config, "exp.nonexistent")


def test_resolve_config_value_missing_intermediate():
    config = {"exp": {"seed": 42}}
    with pytest.raises(KeyError, match="not found"):
        _resolve_config_value(config, "exp.nested.seed")


def test_resolve_config_value_non_dict_intermediate():
    config = {"exp": {"seed": 42}}
    with pytest.raises(KeyError):
        _resolve_config_value(config, "exp.seed.something")


# ---------------------------------------------------------------------------
# _build_filters
# ---------------------------------------------------------------------------

def test_build_filters_none_none():
    assert _build_filters(None, None) is None


def test_build_filters_group_only():
    assert _build_filters(None, "my_group") == {"group": "my_group"}


def test_build_filters_filters_only():
    f = {"config.exp.seed": 42}
    assert _build_filters(f, None) is f


def test_build_filters_both():
    f = {"config.exp.seed": 42}
    result = _build_filters(f, "my_group")
    assert result == {"$and": [f, {"group": "my_group"}]}


# ---------------------------------------------------------------------------
# fetch_run_traces (mocked)
# ---------------------------------------------------------------------------

@dataclass
class MockFile:
    """Stand-in for a wandb File object."""
    name: str
    _content: bytes

    def download(self, root: str = ".", replace: bool = False) -> str:
        path = os.path.join(root, self.name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(self._content)
        return path


@dataclass
class MockRun:
    """Lightweight stand-in for a wandb Run."""
    name: str
    config: dict
    _history: list[dict] = field(default_factory=list)
    _files: dict[str, MockFile] = field(default_factory=dict)

    def scan_history(
        self,
        keys: list[str] | None = None,
        page_size: int = 1000,
        min_step: int | None = None,
        max_step: int | None = None,
    ):
        if keys is None:
            return iter(self._history)
        return iter(
            {k: row[k] for k in keys if k in row} for row in self._history
        )

    def history(
        self,
        keys: list[str] | None = None,
        samples: int = 500,
    ) -> pd.DataFrame:
        if keys is None:
            rows = self._history[:samples]
        else:
            rows = [
                {k: row[k] for k in keys if k in row}
                for row in self._history[:samples]
            ]
        return pd.DataFrame(rows)

    def file(self, name: str) -> MockFile:
        return self._files[name]


def _make_mock_api(runs: list[MockRun]):
    """Return a patched wandb.Api whose .runs() yields the given MockRun list."""
    mock_api_instance = MagicMock()
    mock_api_instance.runs.return_value = runs
    return mock_api_instance


# ---------------------------------------------------------------------------
# _fetch_history
# ---------------------------------------------------------------------------

def test_fetch_history_scan():
    """samples=None uses scan_history (exact)."""
    run = MockRun(
        name="r", config={},
        _history=[{"_step": 0, "m": 1.0}, {"_step": 1, "m": 2.0}],
    )
    s = _fetch_history(run, step_key="_step", metric="m", samples=None)
    assert list(s.index) == [0, 1]
    assert list(s.values) == [1.0, 2.0]


def test_fetch_history_sampled():
    """samples=int uses history() (sampled)."""
    run = MockRun(
        name="r", config={},
        _history=[{"_step": 0, "m": 1.0}, {"_step": 1, "m": 2.0}, {"_step": 2, "m": 3.0}],
    )
    s = _fetch_history(run, step_key="_step", metric="m", samples=2)
    # MockRun.history slices to [:samples]
    assert list(s.index) == [0, 1]
    assert list(s.values) == [1.0, 2.0]


# ---------------------------------------------------------------------------
# fetch_run_traces (mocked)
# ---------------------------------------------------------------------------

def test_fetch_run_traces_basic_structure():
    mock_runs = [
        MockRun(
            name="run_A",
            config={"exp": {"seed": 1}},
            _history=[
                {"_step": 0, "loss": 1.0},
                {"_step": 1, "loss": 0.5},
                {"_step": 2, "loss": 0.3},
            ],
        ),
        MockRun(
            name="run_B",
            config={"exp": {"seed": 2}},
            _history=[
                {"_step": 0, "loss": 0.9},
                {"_step": 2, "loss": 0.4},
            ],
        ),
    ]
    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api(mock_runs)):
        df = fetch_run_traces(
            entity="test",
            project="test",
            metric="loss",
        )

    assert df.shape[0] == 2
    assert sorted(df.columns.tolist()) == [0, 1, 2]
    assert df.index.name == "run_name"
    assert df.loc["run_A", 0] == 1.0
    assert df.loc["run_A", 2] == 0.3
    assert df.loc["run_B", 0] == 0.9
    assert pd.isna(df.loc["run_B", 1])


def test_fetch_run_traces_multiindex():
    mock_runs = [
        MockRun(
            name="run_A",
            config={"exp": {"seed": 1}, "rl": {"lr": 0.001}},
            _history=[{"_step": 0, "m": 10.0}],
        ),
        MockRun(
            name="run_B",
            config={"exp": {"seed": 2}, "rl": {"lr": 0.01}},
            _history=[{"_step": 0, "m": 20.0}],
        ),
    ]
    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api(mock_runs)):
        df = fetch_run_traces(
            entity="test",
            project="test",
            metric="m",
            config_keys=["exp.seed", "rl.lr"],
        )

    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["run_name", "exp.seed", "rl.lr"]
    assert df.loc[("run_A", 1, 0.001), 0] == 10.0
    assert df.loc[("run_B", 2, 0.01), 0] == 20.0


def test_fetch_run_traces_no_runs_raises():
    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api([])):
        with pytest.raises(ValueError, match="No runs found"):
            fetch_run_traces(
                entity="test",
                project="test",
                metric="loss",
            )


def test_fetch_run_traces_columns_sorted():
    mock_runs = [
        MockRun(
            name="run_A",
            config={},
            _history=[
                {"_step": 5, "v": 1.0},
                {"_step": 0, "v": 2.0},
                {"_step": 3, "v": 3.0},
            ],
        ),
    ]
    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api(mock_runs)):
        df = fetch_run_traces(
            entity="test",
            project="test",
            metric="v",
        )

    assert df.columns.tolist() == [0, 3, 5]


def test_fetch_run_traces_custom_step_key():
    mock_runs = [
        MockRun(
            name="run_A",
            config={},
            _history=[
                {"step_count": 100, "goal_mod": 0.5},
                {"step_count": 200, "goal_mod": 0.8},
            ],
        ),
        MockRun(
            name="run_B",
            config={},
            _history=[
                {"step_count": 100, "goal_mod": 0.3},
                {"step_count": 300, "goal_mod": 0.9},
            ],
        ),
    ]
    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api(mock_runs)):
        df = fetch_run_traces(
            entity="test",
            project="test",
            metric="goal_mod",
            step_key="step_count",
        )

    assert df.columns.tolist() == [100, 200, 300]
    assert df.loc["run_A", 100] == 0.5
    assert df.loc["run_A", 200] == 0.8
    assert pd.isna(df.loc["run_A", 300])
    assert df.loc["run_B", 300] == 0.9


def test_fetch_run_traces_flat_dotted_config():
    """Wandb flattens Hydra configs: 'exp.seed' is a top-level key."""
    mock_runs = [
        MockRun(
            name="run_flat",
            config={"exp.seed": 7, "rl.lr": 0.01},
            _history=[{"_step": 0, "loss": 1.0}],
        ),
    ]
    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api(mock_runs)):
        df = fetch_run_traces(
            entity="test",
            project="test",
            metric="loss",
            config_keys=["exp.seed"],
        )

    assert isinstance(df.index, pd.MultiIndex)
    assert df.loc[("run_flat", 7), 0] == 1.0


def test_fetch_run_traces_filters_passed_through():
    mock_api = _make_mock_api([
        MockRun(name="r", config={}, _history=[{"_step": 0, "x": 1.0}]),
    ])
    with patch("scripts.wandb_data.wandb.Api", return_value=mock_api):
        fetch_run_traces(
            entity="ent",
            project="proj",
            metric="x",
            filters={"config.exp.seed": 5},
            group="grp",
        )

    mock_api.runs.assert_called_once_with(
        path="ent/proj",
        filters={"$and": [{"config.exp.seed": 5}, {"group": "grp"}]},
    )


def test_fetch_run_traces_with_samples():
    mock_runs = [
        MockRun(
            name="run_A",
            config={},
            _history=[
                {"_step": 0, "v": 1.0},
                {"_step": 1, "v": 2.0},
                {"_step": 2, "v": 3.0},
            ],
        ),
    ]
    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api(mock_runs)):
        df = fetch_run_traces(
            entity="test",
            project="test",
            metric="v",
            samples=2,
        )

    # MockRun.history slices to [:samples], so only steps 0 and 1
    assert df.columns.tolist() == [0, 1]
    assert df.loc["run_A", 0] == 1.0
    assert df.loc["run_A", 1] == 2.0


# ---------------------------------------------------------------------------
# plot_traces
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt


def _make_trace_df():
    """Build a small multi-indexed DataFrame for plot tests."""
    index = pd.MultiIndex.from_tuples(
        [
            ("r1", "A", 1),
            ("r2", "A", 1),
            ("r3", "B", 2),
            ("r4", "B", 2),
        ],
        names=["run_name", "group", "seed"],
    )
    data = [
        [1.0, 2.0, 3.0],
        [1.5, 2.5, 3.5],
        [4.0, 5.0, 6.0],
        [4.5, 5.5, 6.5],
    ]
    return pd.DataFrame(data, index=index, columns=[0, 1, 2])


def test_plot_traces_returns_axes():
    df = _make_trace_df()
    ax = plot_traces(df, group_keys=["group"], colors=["red", "blue"])
    assert isinstance(ax, plt.Axes)
    # 2 groups -> 2 lines
    assert len(ax.lines) == 2
    plt.close("all")


def test_plot_traces_multi_group_keys():
    df = _make_trace_df()
    ax = plot_traces(df, group_keys=["group", "seed"], colors=["red", "blue"])
    assert len(ax.lines) == 2
    plt.close("all")


def test_plot_traces_uses_provided_ax():
    df = _make_trace_df()
    fig, provided_ax = plt.subplots()
    returned_ax = plot_traces(df, group_keys=["group"], colors=["red", "blue"], ax=provided_ax)
    assert returned_ax is provided_ax
    plt.close("all")


def test_plot_traces_wrong_color_count():
    df = _make_trace_df()
    with pytest.raises(ValueError, match="Expected 2 colors"):
        plot_traces(df, group_keys=["group"], colors=["red"])
    plt.close("all")


def test_plot_traces_missing_group_key():
    df = _make_trace_df()
    with pytest.raises(ValueError, match="not found in DataFrame index"):
        plot_traces(df, group_keys=["nonexistent"], colors=["red"])
    plt.close("all")


def test_plot_traces_no_multiindex():
    df = pd.DataFrame(
        [[1.0, 2.0]], index=pd.Index(["r1"], name="run_name"), columns=[0, 1],
    )
    with pytest.raises(ValueError, match="must have a MultiIndex"):
        plot_traces(df, group_keys=["group"], colors=["red"])
    plt.close("all")


# ---------------------------------------------------------------------------
# _extract_heatmap_grids
# ---------------------------------------------------------------------------

def _make_plotly_json(grids: list[np.ndarray]) -> dict:
    """Build a minimal Plotly JSON dict with heatmap traces from z-arrays."""
    return {
        "data": [
            {"type": "heatmap", "z": g.tolist()} for g in grids
        ],
        "layout": {},
    }


def test_extract_heatmap_grids_basic():
    g0 = np.arange(6).reshape(2, 3).astype(float)
    g1 = np.arange(6, 12).reshape(2, 3).astype(float)
    plotly_json = _make_plotly_json([g0, g1])

    result = _extract_heatmap_grids(plotly_json)

    assert result.shape == (2, 2, 3)
    np.testing.assert_array_equal(result[0], g0)
    np.testing.assert_array_equal(result[1], g1)


def test_extract_heatmap_grids_no_heatmaps():
    plotly_json = {"data": [{"type": "scatter", "x": [1], "y": [2]}]}
    with pytest.raises(ValueError, match="No heatmap traces"):
        _extract_heatmap_grids(plotly_json)


def test_extract_heatmap_grids_mixed_trace_types():
    """Only heatmap traces are extracted; other trace types are ignored."""
    g0 = np.ones((3, 3))
    plotly_json = {
        "data": [
            {"type": "scatter", "x": [1], "y": [2]},
            {"type": "heatmap", "z": g0.tolist()},
            {"type": "bar", "x": [1], "y": [2]},
        ],
        "layout": {},
    }
    result = _extract_heatmap_grids(plotly_json)
    assert result.shape == (1, 3, 3)
    np.testing.assert_array_equal(result[0], g0)


def test_extract_heatmap_grids_roundtrip():
    """Build a Plotly figure exactly like get_occupancy_fig and verify roundtrip."""
    from plotly.subplots import make_subplots

    occ = np.random.rand(4, 14, 14)
    fig = make_subplots(rows=1, cols=4)
    for hd in range(4):
        fig.add_trace(
            go.Heatmap(z=occ[hd].T, showscale=False),
            row=1, col=hd + 1,
        )

    # Serialize to JSON dict (same format WandB stores)
    plotly_json = json.loads(fig.to_json())
    result = _extract_heatmap_grids(plotly_json)

    assert result.shape == (4, 14, 14)
    for hd in range(4):
        np.testing.assert_array_almost_equal(result[hd], occ[hd].T)


# ---------------------------------------------------------------------------
# _fetch_plotly_file
# ---------------------------------------------------------------------------

def test_fetch_plotly_file(tmp_path):
    expected = {"data": [{"type": "heatmap", "z": [[1, 2], [3, 4]]}]}
    media_path = "media/plotly/test.plotly.json"

    run = MockRun(
        name="r",
        config={},
        _files={media_path: MockFile(media_path, json.dumps(expected).encode())},
    )

    result = _fetch_plotly_file(run, media_path, str(tmp_path))
    assert result == expected


# ---------------------------------------------------------------------------
# _scan_occupancy_refs / _download_and_extract
# ---------------------------------------------------------------------------

def test_scan_occupancy_refs_collects_refs():
    media_ref_1 = {"_type": "plotly-file", "path": "media/plotly/a.json"}
    media_ref_2 = {"_type": "plotly-file", "path": "media/plotly/b.json"}
    run = MockRun(
        name="r",
        config={},
        _history=[
            {"_step": 10, "occ": media_ref_1},
            {"_step": 20, "occ": media_ref_2},
        ],
    )

    refs = _scan_occupancy_refs(run, metric="occ", step_key="_step")

    assert len(refs) == 2
    assert refs[0] == (10, media_ref_1)
    assert refs[1] == (20, media_ref_2)


def test_scan_occupancy_refs_skips_missing_metric():
    run = MockRun(
        name="r",
        config={},
        _history=[
            {"_step": 0, "other_metric": 1.0},
            {"_step": 1},
        ],
    )

    refs = _scan_occupancy_refs(run, metric="occ", step_key="_step")
    assert refs == []


def test_download_and_extract_file_reference(tmp_path):
    """Handles WandB file-reference dicts."""
    grid = np.ones((4, 3, 3))
    plotly_json = _make_plotly_json([grid[hd] for hd in range(4)])
    media_path = "media/plotly/occ.plotly.json"

    run = MockRun(
        name="r",
        config={},
        _files={media_path: MockFile(media_path, json.dumps(plotly_json).encode())},
    )

    result = _download_and_extract(
        run, {"_type": "plotly-file", "path": media_path}, str(tmp_path),
    )

    assert result is not None
    np.testing.assert_array_equal(result, grid)


def test_download_and_extract_inline_data():
    """Handles inline Plotly JSON data."""
    grid = np.arange(9).reshape(1, 3, 3).astype(float)
    plotly_json = _make_plotly_json([grid[0]])

    run = MockRun(name="r", config={})
    result = _download_and_extract(run, plotly_json, "/unused")

    assert result is not None
    np.testing.assert_array_equal(result[0], grid[0])


def test_download_and_extract_unknown_format():
    """Returns None for unrecognised media references."""
    run = MockRun(name="r", config={})
    result = _download_and_extract(run, {"_type": "unknown"}, "/unused")
    assert result is None


# ---------------------------------------------------------------------------
# fetch_occupancy_grids (mocked)
# ---------------------------------------------------------------------------

def _make_occ_mock_run(
    name: str,
    config: dict,
    step_grids: list[tuple[int, np.ndarray]],
) -> MockRun:
    """Build a MockRun with occupancy Plotly files at given steps."""
    history = []
    files = {}
    for step, grid in step_grids:
        plotly_json = _make_plotly_json([grid[hd] for hd in range(grid.shape[0])])
        media_path = f"media/plotly/occ_{step}.plotly.json"
        history.append({
            "_step": step,
            "Eval/OPA_Occupancy": {"_type": "plotly-file", "path": media_path},
        })
        files[media_path] = MockFile(media_path, json.dumps(plotly_json).encode())
    return MockRun(name=name, config=config, _history=history, _files=files)


def test_fetch_occupancy_grids_single_run():
    grid = np.random.rand(4, 3, 3)
    mock_runs = [_make_occ_mock_run("r1", {}, [(10, grid)])]

    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api(mock_runs)):
        data = fetch_occupancy_grids(entity="e", project="p")

    assert isinstance(data, OccupancyData)
    assert list(data.grids.keys()) == [10]
    assert data.grids[10].shape == (1, 4, 3, 3)
    np.testing.assert_array_almost_equal(data.grids[10][0], grid)
    assert data.run_names == ["r1"]


def test_fetch_occupancy_grids_multiple_runs():
    g1 = np.ones((4, 3, 3))
    g2 = np.ones((4, 3, 3)) * 2
    mock_runs = [
        _make_occ_mock_run("r1", {}, [(10, g1)]),
        _make_occ_mock_run("r2", {}, [(10, g2)]),
    ]

    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api(mock_runs)):
        data = fetch_occupancy_grids(entity="e", project="p")

    assert data.grids[10].shape == (2, 4, 3, 3)
    # Average should be 1.5
    avg = np.nanmean(data.grids[10], axis=0)
    np.testing.assert_array_almost_equal(avg, np.ones((4, 3, 3)) * 1.5)


def test_fetch_occupancy_grids_mismatched_steps():
    """Runs with different steps produce NaN-padded arrays."""
    g1 = np.ones((4, 2, 2))
    g2 = np.ones((4, 2, 2)) * 3
    mock_runs = [
        _make_occ_mock_run("r1", {}, [(10, g1)]),
        _make_occ_mock_run("r2", {}, [(10, g2), (20, g2)]),
    ]

    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api(mock_runs)):
        data = fetch_occupancy_grids(entity="e", project="p")

    assert sorted(data.grids.keys()) == [10, 20]
    # Step 10: both runs have data
    assert not np.any(np.isnan(data.grids[10]))
    # Step 20: only run r2 has data, r1 should be NaN
    assert np.all(np.isnan(data.grids[20][0]))  # r1 missing
    np.testing.assert_array_almost_equal(data.grids[20][1], g2)  # r2 present


def test_fetch_occupancy_grids_no_runs_raises():
    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api([])):
        with pytest.raises(ValueError, match="No runs found"):
            fetch_occupancy_grids(entity="e", project="p")


def test_fetch_occupancy_grids_no_data_raises():
    """Runs exist but none have occupancy data."""
    mock_runs = [MockRun(name="r1", config={}, _history=[])]
    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api(mock_runs)):
        with pytest.raises(ValueError, match="No occupancy data"):
            fetch_occupancy_grids(entity="e", project="p")


def test_fetch_occupancy_grids_with_config_keys():
    g = np.zeros((4, 2, 2))
    mock_runs = [
        _make_occ_mock_run("r1", {"exp": {"seed": 1}}, [(5, g)]),
    ]

    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api(mock_runs)):
        data = fetch_occupancy_grids(
            entity="e", project="p", config_keys=["exp.seed"],
        )

    assert data.config_keys == ["exp.seed"]
    assert data.config_values == [(1,)]


def test_fetch_occupancy_grids_filters_passed_through():
    mock_api = _make_mock_api([
        _make_occ_mock_run("r", {}, [(0, np.zeros((4, 2, 2)))]),
    ])
    with patch("scripts.wandb_data.wandb.Api", return_value=mock_api):
        fetch_occupancy_grids(
            entity="e",
            project="p",
            filters={"config.exp.seed": 5},
            group="grp",
        )

    mock_api.runs.assert_called_once_with(
        path="e/p",
        filters={"$and": [{"config.exp.seed": 5}, {"group": "grp"}]},
    )


# ---------------------------------------------------------------------------
# plot_occupancy_average
# ---------------------------------------------------------------------------

def test_plot_occupancy_average_returns_figure():
    avg = {
        10: np.random.rand(4, 3, 3),
        20: np.random.rand(4, 3, 3),
    }
    fig = plot_occupancy_average(avg)

    assert isinstance(fig, go.Figure)
    # 2 steps x 4 HDs = 8 heatmap traces
    heatmaps = [t for t in fig.data if t.type == "heatmap"]
    assert len(heatmaps) == 8


def test_plot_occupancy_average_subset_steps():
    avg = {
        10: np.random.rand(4, 3, 3),
        20: np.random.rand(4, 3, 3),
        30: np.random.rand(4, 3, 3),
    }
    fig = plot_occupancy_average(avg, steps=[10, 30])

    heatmaps = [t for t in fig.data if t.type == "heatmap"]
    # 2 steps x 4 HDs = 8
    assert len(heatmaps) == 8


# ---------------------------------------------------------------------------
# runs_per_step
# ---------------------------------------------------------------------------

def test_runs_per_step_all_same():
    """All runs log at the same steps."""
    entries = [[(0, "a"), (1, "b")], [(0, "c"), (1, "d")]]
    assert runs_per_step(entries) == {0: 2, 1: 2}


def test_runs_per_step_partial_overlap():
    """Runs with partially overlapping steps."""
    entries = [
        [(1, "a"), (2, "b")],
        [(2, "c"), (3, "d")],
    ]
    assert runs_per_step(entries) == {1: 1, 2: 2, 3: 1}


def test_runs_per_step_no_overlap():
    """Runs with completely disjoint steps."""
    entries = [[(0, "a")], [(5, "b")]]
    assert runs_per_step(entries) == {0: 1, 5: 1}


def test_runs_per_step_empty_list():
    """No runs at all."""
    assert runs_per_step([]) == {}



# NOTE: save_occupancy / load_occupancy tests removed — functions not yet in source.


# ---------------------------------------------------------------------------
# _histogram_to_subroom_counts
# ---------------------------------------------------------------------------

def test_histogram_to_subroom_counts_perfect_bins():
    """Bins perfectly aligned with integer subroom IDs."""
    # Simulate wandb.Histogram output for data [0, 0, 1, 2, 3, 3]
    # With bins exactly at integer boundaries
    hist = {
        "bins": [-0.5, 0.5, 1.5, 2.5, 3.5],
        "values": [2, 1, 1, 2],
    }
    counts = _histogram_to_subroom_counts(hist, n_subrooms=4)
    np.testing.assert_array_equal(counts, [2, 1, 1, 2])


def test_histogram_to_subroom_counts_fine_bins():
    """Many fine bins (like wandb's default 64) correctly summed per subroom."""
    # Create 64 bins spanning 0 to 3, put all mass in the bin around subroom 2
    edges = np.linspace(0, 3, 65)
    values = np.zeros(64)
    # Bins whose center is closest to 2.0
    centers = (edges[:-1] + edges[1:]) / 2
    for i, c in enumerate(centers):
        if round(c) == 2:
            values[i] = 5.0
    hist = {"bins": edges.tolist(), "values": values.tolist()}
    counts = _histogram_to_subroom_counts(hist, n_subrooms=4)
    assert counts[2] == values.sum()
    assert counts[0] == 0
    assert counts[1] == 0
    assert counts[3] == 0


def test_histogram_to_subroom_counts_out_of_range():
    """Bins outside [0, n_subrooms) are ignored."""
    hist = {
        "bins": [-1.5, -0.5, 0.5, 1.5],
        "values": [10, 5, 3],
    }
    counts = _histogram_to_subroom_counts(hist, n_subrooms=4)
    # center -1.0 -> round=-1 -> out of range, ignored
    # center 0.0 -> subroom 0
    # center 1.0 -> subroom 1
    np.testing.assert_array_equal(counts, [5, 3, 0, 0])


# ---------------------------------------------------------------------------
# fetch_subroom_counts (mocked)
# ---------------------------------------------------------------------------

import torch


def _make_hist_dict(subroom_counts: list[float]) -> dict:
    """Build a minimal wandb Histogram dict from per-subroom counts.

    Simulates wandb.Histogram output for actual 1-indexed subroom IDs (1–4).
    Bins use 1-indexed boundaries [0.5, 1.5, 2.5, 3.5, 4.5] to match real data,
    so that fetch_subroom_counts (which shifts bins by -1 internally) maps
    rooms 1–4 to output indices 0–3.
    """
    n = len(subroom_counts)
    bins = [i + 0.5 for i in range(n + 1)]
    return {"_type": "histogram", "bins": bins, "values": subroom_counts}


def test_fetch_subroom_counts_basic():
    """Single run, exactly last_n entries."""
    n_steps = 5
    history = []
    for t in range(n_steps):
        history.append({
            "_step": t,
            "subroom_ids": _make_hist_dict([t, t + 1, t + 2, t + 3]),
        })
    mock_runs = [MockRun(name="r1", config={}, _history=history)]

    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api(mock_runs)):
        result = fetch_subroom_counts(
            entity="e", project="p", last_n=5,
        )

    assert isinstance(result, SubroomData)
    assert result.subroom_ids.shape == (1, 5, 4)
    # Check first timestep
    np.testing.assert_array_almost_equal(
        result.subroom_ids[0, 0].numpy(), [0, 1, 2, 3],
    )


def test_fetch_subroom_counts_truncates_to_last_n():
    """More entries than last_n → keeps only the last last_n."""
    history = [
        {"_step": t, "subroom_ids": _make_hist_dict([t, 0, 0, 0])}
        for t in range(10)
    ]
    mock_runs = [MockRun(name="r1", config={}, _history=history)]

    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api(mock_runs)):
        result = fetch_subroom_counts(entity="e", project="p", last_n=3)

    assert result.subroom_ids.shape == (1, 3, 4)
    # Should have steps 7, 8, 9 (the last 3)
    np.testing.assert_array_almost_equal(result.subroom_ids[0, 0, 0].item(), 7)
    np.testing.assert_array_almost_equal(result.subroom_ids[0, 1, 0].item(), 8)
    np.testing.assert_array_almost_equal(result.subroom_ids[0, 2, 0].item(), 9)


def test_fetch_subroom_counts_pads_short_runs():
    """Runs with fewer than last_n entries are front-padded with NaN."""
    history = [
        {"_step": 0, "subroom_ids": _make_hist_dict([1, 2, 3, 4])},
    ]
    mock_runs = [MockRun(name="r1", config={}, _history=history)]

    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api(mock_runs)):
        result = fetch_subroom_counts(entity="e", project="p", last_n=3)

    assert result.subroom_ids.shape == (1, 3, 4)
    # First 2 rows should be NaN (padding)
    assert torch.isnan(result.subroom_ids[0, 0]).all()
    assert torch.isnan(result.subroom_ids[0, 1]).all()
    # Last row has actual data
    np.testing.assert_array_almost_equal(result.subroom_ids[0, 2].numpy(), [1, 2, 3, 4])


def test_fetch_subroom_counts_multiple_runs():
    """Multiple runs stacked along dim 0."""
    h1 = [{"_step": 0, "subroom_ids": _make_hist_dict([1, 0, 0, 0])}]
    h2 = [{"_step": 0, "subroom_ids": _make_hist_dict([0, 0, 0, 2])}]
    mock_runs = [
        MockRun(name="r1", config={}, _history=h1),
        MockRun(name="r2", config={}, _history=h2),
    ]

    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api(mock_runs)):
        result = fetch_subroom_counts(entity="e", project="p", last_n=1)

    assert result.subroom_ids.shape == (2, 1, 4)
    np.testing.assert_array_almost_equal(result.subroom_ids[0, 0].numpy(), [1, 0, 0, 0])
    np.testing.assert_array_almost_equal(result.subroom_ids[1, 0].numpy(), [0, 0, 0, 2])


def test_fetch_subroom_counts_no_runs_raises():
    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api([])):
        with pytest.raises(ValueError, match="No runs found"):
            fetch_subroom_counts(entity="e", project="p")


def test_fetch_subroom_counts_page_size_forwarded():
    """page_size is forwarded to scan_history."""
    history = [{"_step": 0, "subroom_ids": _make_hist_dict([1, 0, 0, 0])}]
    mock_run = MockRun(name="r1", config={}, _history=history)
    mock_runs = [mock_run]

    with patch("scripts.wandb_data.wandb.Api", return_value=_make_mock_api(mock_runs)):
        with patch.object(mock_run, "scan_history", wraps=mock_run.scan_history) as spy:
            fetch_subroom_counts(entity="e", project="p", last_n=1, page_size=5000)
            _, kwargs = spy.call_args
            assert kwargs.get("page_size") == 5000


# ---------------------------------------------------------------------------
# plot_subroom_percentage
# ---------------------------------------------------------------------------

def test_plot_subroom_percentage_returns_axes():
    # 2 runs, 3 timesteps, 4 rooms — uniform visits
    counts = torch.tensor([
        [[10, 10, 10, 10], [20, 20, 20, 20], [5, 5, 5, 5]],
        [[8, 8, 8, 8], [12, 12, 12, 12], [6, 6, 6, 6]],
    ], dtype=torch.float)
    ax = plot_subroom_percentage([counts])
    assert isinstance(ax, plt.Axes)
    # 4 bars
    assert len(ax.patches) == 4
    plt.close("all")


def test_plot_subroom_percentage_values():
    """All visits in room 0 → 100% room 0, 0% elsewhere."""
    counts = torch.tensor([
        [[10, 0, 0, 0], [10, 0, 0, 0]],
    ], dtype=torch.float)
    ax = plot_subroom_percentage([counts])
    heights = [p.get_height() for p in ax.patches]
    assert heights[0] == pytest.approx(100.0)
    for h in heights[1:]:
        assert h == pytest.approx(0.0)
    plt.close("all")


def test_plot_subroom_percentage_custom_labels():
    counts = torch.ones((1, 2, 3))
    ax = plot_subroom_percentage([counts], room_labels=["A", "B", "C"])
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["A", "B", "C"]
    plt.close("all")


def test_plot_subroom_percentage_step_range():
    """step_range restricts aggregation to the specified slice of steps."""
    # Steps 0-1: all visits in room 0. Step 2: all visits in room 1.
    counts = torch.tensor([
        [[10, 0, 0, 0], [10, 0, 0, 0], [0, 10, 0, 0]],
    ], dtype=torch.float)
    # With step_range=(0, 2) we should get 100% room 0
    ax = plot_subroom_percentage([counts], step_range=(0, 2))
    heights = [p.get_height() for p in ax.patches]
    assert heights[0] == pytest.approx(100.0)
    for h in heights[1:]:
        assert h == pytest.approx(0.0)
    plt.close("all")

    # With step_range=(2, 3) we should get 100% room 1
    ax = plot_subroom_percentage([counts], step_range=(2, 3))
    heights = [p.get_height() for p in ax.patches]
    assert heights[1] == pytest.approx(100.0)
    assert heights[0] == pytest.approx(0.0)
    plt.close("all")


def test_plot_subroom_percentage_se_errorbars():
    """SE is computed per-sample (run, step), not globally."""
    # 1 run, 2 steps: step 0 → room 0 gets 75%, step 1 → room 0 gets 25%
    counts = torch.tensor([
        [[3, 1, 0, 0], [1, 3, 0, 0]],
    ], dtype=torch.float)
    ax = plot_subroom_percentage([counts])
    # Mean for room 0: (75 + 25) / 2 = 50%
    heights = [p.get_height() for p in ax.patches]
    assert heights[0] == pytest.approx(50.0)
    # SE for room 0: std([75, 25]) / sqrt(2) = 25 / sqrt(2)
    expected_se = 25.0 / np.sqrt(2)
    mean_pct, se_pct = _subroom_percentages(counts)
    assert se_pct[0] == pytest.approx(expected_se, rel=1e-5)
    plt.close("all")


def test_plot_subroom_percentage_nan_room0_still_valid():
    """Rows where room 0 is NaN but other rooms have data are still counted as valid."""
    # 1 run, 1 step: room 0 is NaN, rooms 1-3 have counts
    counts = torch.tensor([
        [[float("nan"), 5, 5, 0]],
    ], dtype=torch.float)
    ax = plot_subroom_percentage([counts])
    heights = [p.get_height() for p in ax.patches]
    # Room 1 and 2 each get 50%, room 0 and 3 get 0/NaN
    assert heights[1] == pytest.approx(50.0)
    assert heights[2] == pytest.approx(50.0)
    plt.close("all")


def test_plot_subroom_percentage_nan_rows_excluded():
    """NaN rows (front-padding) are excluded from SE computation."""
    # 1 run, 3 steps: first is NaN padding, then two real steps
    counts = torch.tensor([
        [[float("nan"), float("nan"), float("nan"), float("nan")],
         [10, 0, 0, 0],
         [10, 0, 0, 0]],
    ], dtype=torch.float)
    ax = plot_subroom_percentage([counts])
    heights = [p.get_height() for p in ax.patches]
    # Both valid samples: room 0 = 100%, so mean = 100% and SE = 0
    assert heights[0] == pytest.approx(100.0)
    plt.close("all")


def test_plot_subroom_percentage_multigroup_bar_count():
    """Two groups → 2 bars per room (8 patches for 4 rooms)."""
    counts_a = torch.tensor([[[10, 0, 0, 0], [10, 0, 0, 0]]], dtype=torch.float)
    counts_b = torch.tensor([[[0, 10, 0, 0], [0, 10, 0, 0]]], dtype=torch.float)
    ax = plot_subroom_percentage([counts_a, counts_b])
    assert len(ax.patches) == 8
    plt.close("all")


def test_plot_subroom_percentage_multigroup_legend():
    """group_labels appear in the legend."""
    counts_a = torch.ones((1, 1, 4))
    counts_b = torch.ones((1, 1, 4))
    ax = plot_subroom_percentage([counts_a, counts_b], group_labels=["random", "curious"])
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_texts == ["random", "curious"]
    plt.close("all")


def test_plot_subroom_percentage_multigroup_colors():
    """Custom colors are applied to each group's bars."""
    counts_a = torch.ones((1, 1, 4))
    counts_b = torch.ones((1, 1, 4))
    ax = plot_subroom_percentage([counts_a, counts_b], colors=["red", "blue"])
    bar_colors = [p.get_facecolor() for p in ax.patches]
    # First 4 patches belong to group 0 (red), next 4 to group 1 (blue)
    import matplotlib.colors as mcolors
    red = mcolors.to_rgba("red")
    blue = mcolors.to_rgba("blue")
    for fc in bar_colors[:4]:
        assert tuple(fc) == pytest.approx(red)
    for fc in bar_colors[4:]:
        assert tuple(fc) == pytest.approx(blue)
    plt.close("all")


def test_subroom_percentages_helper():
    """_subroom_percentages returns correct mean and SE."""
    counts = torch.tensor([[[3, 1, 0, 0], [1, 3, 0, 0]]], dtype=torch.float)
    mean_pct, se_pct = _subroom_percentages(counts)
    assert mean_pct[0] == pytest.approx(50.0)
    assert se_pct[0] == pytest.approx(25.0 / np.sqrt(2))


# ---------------------------------------------------------------------------
# _sig_label
# ---------------------------------------------------------------------------


def test_sig_label_thresholds():
    assert _sig_label(0.05) == "*"
    assert _sig_label(0.01) == "**"
    assert _sig_label(0.001) == "***"
    assert _sig_label(0.049) == "**"   # < 0.05 → **
    assert _sig_label(0.009) == "***"  # < 0.01 → ***


# ---------------------------------------------------------------------------
# _pval_stars
# ---------------------------------------------------------------------------


def test_pval_stars_thresholds():
    assert _pval_stars(0.0001) == "***"
    assert _pval_stars(0.0009) == "***"
    assert _pval_stars(0.001) == "**"   # 0.001 is not < 0.001, but is < 0.01
    assert _pval_stars(0.005) == "**"
    assert _pval_stars(0.01) == "*"     # 0.01 is not < 0.01, but is < 0.05
    assert _pval_stars(0.03) == "*"
    assert _pval_stars(0.05) is None    # boundary: not < 0.05
    assert _pval_stars(0.1) is None


def test_plot_subroom_percentage_star_annotation():
    """Bracket text reflects the actual p-value significance level."""
    # Two groups with n=30 runs each: group a all in room 0, group b all in room 1.
    # p-value for room 0 and room 1 will be very small → expect *** brackets.
    rng = np.random.default_rng(0)
    n_runs = 30
    counts_a = torch.zeros((n_runs, 1, 2), dtype=torch.float)
    counts_b = torch.zeros((n_runs, 1, 2), dtype=torch.float)
    counts_a[:, :, 0] = 10.0
    counts_b[:, :, 1] = 10.0

    ax = plot_subroom_percentage([counts_a, counts_b], group_labels=["a", "b"])
    texts = [t.get_text() for t in ax.texts]
    assert any(t in {"*", "**", "***"} for t in texts), f"Expected star annotation, got: {texts}"
    plt.close("all")


# ---------------------------------------------------------------------------
# _draw_significance_bracket
# ---------------------------------------------------------------------------


def test_draw_significance_bracket_adds_lines_and_text():
    _, ax = plt.subplots()
    top = _draw_significance_bracket(
        ax, x1=0.0, x2=1.0, y_bracket=50.0, text="**",
        y1_bottom=30.0, y2_bottom=40.0,
    )
    # One Line2D for the bracket
    assert len(ax.lines) == 1
    # One Text annotation
    assert any(t.get_text() == "**" for t in ax.texts)
    # Returns y_bracket
    assert top == pytest.approx(50.0)
    plt.close("all")


def test_draw_significance_bracket_vertical_lines_reach_bar_tops():
    """The bracket's vertical lines start at y1_bottom and y2_bottom."""
    _, ax = plt.subplots()
    _draw_significance_bracket(
        ax, x1=0.0, x2=1.0, y_bracket=60.0, text="*",
        y1_bottom=20.0, y2_bottom=35.0,
    )
    ydata = ax.lines[0].get_ydata()
    assert ydata[0] == pytest.approx(20.0)  # left bar top
    assert ydata[3] == pytest.approx(35.0)  # right bar top
    plt.close("all")


def test_draw_significance_bracket_stacking():
    """Calling twice with an offset y_bracket stacks brackets."""
    _, ax = plt.subplots()
    top1 = _draw_significance_bracket(ax, 0.0, 1.0, y_bracket=50.0, text="*",
                                       y1_bottom=30.0, y2_bottom=30.0)
    top2 = _draw_significance_bracket(ax, 0.0, 1.0, y_bracket=top1 + 4.0, text="**",
                                       y1_bottom=30.0, y2_bottom=30.0)
    assert top2 > top1
    plt.close("all")


# ---------------------------------------------------------------------------
# pairwise_ttest
# ---------------------------------------------------------------------------


def test_pairwise_ttest_returns_p_values(capsys):
    """Returns one p-value per pair."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    p_vals = pairwise_ttest([(a, b)], alpha=0.01)
    assert len(p_vals) == 1
    assert 0.0 <= p_vals[0] <= 1.0


def test_pairwise_ttest_significant():
    """Clearly separated distributions yield p < alpha."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 50)
    b = rng.normal(100, 1, 50)
    p_vals = pairwise_ttest([(a, b)], alpha=0.01)
    assert p_vals[0] < 0.01


def test_pairwise_ttest_not_significant():
    """Identical distributions yield p > alpha."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 10)
    b = a.copy()
    # ttest_ind on identical arrays → p=1 (or NaN); just check it doesn't crash
    p_vals = pairwise_ttest([(a, b)], alpha=0.01)
    assert len(p_vals) == 1


def test_pairwise_ttest_multiple_pairs():
    """One p-value returned per pair."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    c = np.array([100.0, 101.0, 102.0])
    p_vals = pairwise_ttest([(a, b), (a, c)], alpha=0.05)
    assert len(p_vals) == 2


def test_pairwise_ttest_pair_labels_printed(capsys):
    """pair_labels appear in printed output."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    pairwise_ttest([(a, b)], alpha=0.05, pair_labels=["rand vs cur"])
    out = capsys.readouterr().out
    assert "rand vs cur" in out


def test_subroom_pct_array_shape():
    """_subroom_pct_array returns shape (n_runs, n_rooms)."""
    counts = torch.ones((3, 5, 4))  # 3 runs, 5 steps, 4 rooms
    result = _subroom_pct_array(counts)
    assert result.shape == (3, 4)


def test_subroom_pct_array_uniform():
    """Uniform visits → 25% per room for each run."""
    counts = torch.ones((2, 3, 4))
    result = _subroom_pct_array(counts)
    assert result == pytest.approx(25.0)


def test_plot_subroom_percentage_ttest_printed(capsys):
    """With two groups, per-subroom t-test is printed."""
    counts_a = torch.tensor([[[10, 0, 0, 0]] * 5], dtype=torch.float)  # all room 0
    counts_b = torch.tensor([[[0, 10, 0, 0]] * 5], dtype=torch.float)  # all room 1
    plot_subroom_percentage(
        [counts_a, counts_b],
        group_labels=["rand", "cur"],
    )
    out = capsys.readouterr().out
    assert "Welch's t-test" in out
    assert "Room 1" in out
    plt.close("all")


# ---------------------------------------------------------------------------
# prob_roi
# ---------------------------------------------------------------------------
#
# Grid format (from the full pipeline):
#   get_occupancy_fig builds occ[hd, x-1, y-1], shape (4, W-2, H-2).
#   It logs go.Heatmap(z=occ[hd].T), which transposes to (H-2, W-2).
#   fetch_occupancy_grids reads z back as-is, stacks → (4, H-2, W-2).
#   data.grids[step] has shape (n_runs, 4, H-2, W-2).
#
# So grid[run, hd, i, j] corresponds to MiniGrid position (x=j+1, y=i+1).
# target_loc = (x, y) in MiniGrid 1-based coordinates.


def _make_grid_items(occ_array: np.ndarray) -> dict:
    """Wrap a (1, 4, H, W) occupancy array as a single-step grid_items dict."""
    return {0: occ_array}


def _grid_for_minigrid_pos(mg_x: int, mg_y: int, room_w: int, room_h: int) -> np.ndarray:
    """
    Return a (1, 4, H-2, W-2) grid with all occupancy at MiniGrid (mg_x, mg_y).
    Grid indices: row = mg_y - 1, col = mg_x - 1.
    """
    grid = np.zeros((1, 4, room_h - 2, room_w - 2))
    grid[0, :, mg_y - 1, mg_x - 1] = 1.0
    return grid


def test_prob_roi_full_probability_at_target():
    """All occupancy at target location → probability ≈ 1."""
    mg_x, mg_y = 7, 2
    room_w, room_h = 16, 16
    grid = _grid_for_minigrid_pos(mg_x, mg_y, room_w, room_h)
    result = prob_roi(
        grid_items=_make_grid_items(grid),
        radius=1,
        target_loc=[mg_x, mg_y],
        sorted_steps=[0],
    )
    assert result.shape == (1, 1)
    assert result[0, 0].item() == pytest.approx(1.0)


def test_prob_roi_zero_probability_wrong_location():
    """All occupancy at target, but ROI centred elsewhere → probability ≈ 0."""
    mg_x, mg_y = 7, 2
    room_w, room_h = 16, 16
    grid = _grid_for_minigrid_pos(mg_x, mg_y, room_w, room_h)
    # Wrong location far from the actual object
    wrong_loc = [mg_x + 8, mg_y + 8]
    result = prob_roi(
        grid_items=_make_grid_items(grid),
        radius=1,
        target_loc=wrong_loc,
        sorted_steps=[0],
    )
    assert result[0, 0].item() == pytest.approx(0.0)


def test_prob_roi_axis_not_swapped():
    """
    Verify x and y are not swapped.
    Place occupancy at (mg_x=3, mg_y=10) — asymmetric so a swap is detectable.
    ROI at (3, 10) should give prob ≈ 1; ROI at swapped (10, 3) should give 0.
    """
    mg_x, mg_y = 3, 10
    room_w, room_h = 16, 16
    grid = _grid_for_minigrid_pos(mg_x, mg_y, room_w, room_h)

    result_correct = prob_roi(
        grid_items=_make_grid_items(grid),
        radius=0.5,
        target_loc=[mg_x, mg_y],
        sorted_steps=[0],
    )
    result_swapped = prob_roi(
        grid_items=_make_grid_items(grid),
        radius=0.5,
        target_loc=[mg_y, mg_x],  # intentionally swapped
        sorted_steps=[0],
    )

    assert result_correct[0, 0].item() == pytest.approx(1.0)
    assert result_swapped[0, 0].item() == pytest.approx(0.0)


def test_prob_roi_off_by_one_not_present():
    """
    MiniGrid position (mg_x, mg_y) maps to grid index (mg_y-1, mg_x-1).
    If the -1 offset were missing, (mg_x, mg_y)=(2,2) would look for index (2,2)
    instead of (1,1), causing a miss. Verify the correct index is hit.
    """
    mg_x, mg_y = 2, 2  # grid index should be (1, 1), not (2, 2)
    room_w, room_h = 16, 16
    grid = _grid_for_minigrid_pos(mg_x, mg_y, room_w, room_h)

    result = prob_roi(
        grid_items=_make_grid_items(grid),
        radius=0.5,
        target_loc=[mg_x, mg_y],
        sorted_steps=[0],
    )
    assert result[0, 0].item() == pytest.approx(1.0)


def test_prob_roi_multiple_steps():
    """Output shape is (n_trajs, n_steps) and values are correct per step."""
    mg_x, mg_y = 5, 5
    room_w, room_h = 12, 12
    grid_on = _grid_for_minigrid_pos(mg_x, mg_y, room_w, room_h)   # all mass at target
    grid_off = _grid_for_minigrid_pos(mg_x + 4, mg_y, room_w, room_h)  # all mass elsewhere

    grid_items = {0: grid_on, 100: grid_off}
    result = prob_roi(
        grid_items=grid_items,
        radius=1,
        target_loc=[mg_x, mg_y],
        sorted_steps=[0, 100],
    )
    assert result.shape == (1, 2)
    assert result[0, 0].item() == pytest.approx(1.0)  # step 0: on target
    assert result[0, 1].item() == pytest.approx(0.0)  # step 100: off target


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

def test_bootstrap_ci_output_shape():
    """Returns two arrays of shape [n_train_steps]."""
    rng = np.random.default_rng(0)
    data = rng.random((5, 20))
    lower_err, upper_err = bootstrap_ci(data, n_bootstrap=200)
    assert lower_err.shape == (5,)
    assert upper_err.shape == (5,)


def test_bootstrap_ci_non_negative_errors():
    """Error bars are non-negative (distances from mean)."""
    rng = np.random.default_rng(1)
    data = rng.random((4, 30))
    lower_err, upper_err = bootstrap_ci(data, n_bootstrap=200)
    assert np.all(lower_err >= 0)
    assert np.all(upper_err >= 0)


def test_bootstrap_ci_wider_with_more_variance():
    """Higher-variance data produces wider CI than lower-variance data."""
    rng = np.random.default_rng(2)
    low_var = rng.normal(loc=0.5, scale=0.01, size=(3, 50))
    high_var = rng.normal(loc=0.5, scale=1.0, size=(3, 50))
    lo_l, lo_u = bootstrap_ci(low_var, n_bootstrap=500)
    hi_l, hi_u = bootstrap_ci(high_var, n_bootstrap=500)
    assert (lo_l + lo_u).mean() < (hi_l + hi_u).mean()


# ---------------------------------------------------------------------------
# plot_metric (smoke tests)
# ---------------------------------------------------------------------------

def _make_metric_tensor(n_steps: int = 3, n_runs: int = 10, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.random((n_steps, n_runs)).astype(np.float32))


def test_plot_metric_bootstrap_returns_figure():
    t = _make_metric_tensor()
    fig, ax = plot_metric(
        tensors=[t],
        steps=[0, 1, 2],
        colors=["blue"],
        labels=["agent"],
        xlabel="Steps",
        ylabel="Value",
        use_std=False,
        n_bootstrap=100,
    )
    import matplotlib.pyplot as plt
    assert fig is not None
    plt.close(fig)


def test_plot_metric_std_returns_figure():
    t = _make_metric_tensor()
    fig, ax = plot_metric(
        tensors=[t],
        steps=[0, 1, 2],
        colors=["red"],
        labels=["agent"],
        xlabel="Steps",
        ylabel="Value",
        use_std=True,
    )
    import matplotlib.pyplot as plt
    assert fig is not None
    plt.close(fig)


def test_plot_metric_bootstrap_ci_returns_figure():
    t = _make_metric_tensor()
    fig, ax = plot_metric(
        tensors=[t],
        steps=[0, 1, 2],
        colors=["green"],
        labels=["agent"],
        xlabel="Steps",
        ylabel="Value",
        use_std=False,
    )
    import matplotlib.pyplot as plt
    assert fig is not None
    plt.close(fig)


def test_plot_metric_ttest_significant(capsys):
    """Two clearly separated distributions should yield a significant t-test result."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    t1 = torch.from_numpy(rng.normal(0.0, 0.1, size=(3, 30)).astype(np.float32))
    t2 = torch.from_numpy(rng.normal(1.0, 0.1, size=(3, 30)).astype(np.float32))
    fig, ax = plot_metric(
        tensors=[t1, t2],
        steps=[0, 1, 2],
        colors=["blue", "red"],
        labels=["low", "high"],
        xlabel="Steps",
        ylabel="Value",
        use_std=True,
        alpha=0.01,
    )
    plt.close(fig)
    captured = capsys.readouterr()
    assert "significant" in captured.out
    assert "not significant" not in captured.out


def test_plot_metric_ttest_not_significant(capsys):
    """Two nearly identical distributions should yield a non-significant t-test result."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    data = rng.normal(0.5, 0.5, size=(3, 30)).astype(np.float32)
    t1 = torch.from_numpy(data.copy())
    t2 = torch.from_numpy(data.copy())
    fig, ax = plot_metric(
        tensors=[t1, t2],
        steps=[0, 1, 2],
        colors=["blue", "red"],
        labels=["a", "b"],
        xlabel="Steps",
        ylabel="Value",
        use_std=True,
        alpha=0.01,
    )
    plt.close(fig)
    captured = capsys.readouterr()
    assert "not significant" in captured.out


# ---------------------------------------------------------------------------
# save_subroom_data / load_subroom_data
# ---------------------------------------------------------------------------


def test_save_load_subroom_data_roundtrip(tmp_path):
    """Saved SubroomData is loaded back identically."""
    data = SubroomData(
        subroom_ids=torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]),
        run_names=["r1"],
        config_values=None,
        config_keys=[],
    )
    path = str(tmp_path / "subroom.pt")
    save_subroom_data(data, path)
    loaded = load_subroom_data(path)

    torch.testing.assert_close(loaded.subroom_ids, data.subroom_ids)
    assert loaded.run_names == data.run_names
    assert loaded.config_values == data.config_values
    assert loaded.config_keys == data.config_keys


def test_save_load_subroom_data_with_config(tmp_path):
    """Config values and keys survive the roundtrip."""
    data = SubroomData(
        subroom_ids=torch.ones(2, 3, 4),
        run_names=["r1", "r2"],
        config_values=[(42,), (7,)],
        config_keys=["exp.seed"],
    )
    path = str(tmp_path / "subroom_cfg.pt")
    save_subroom_data(data, path)
    loaded = load_subroom_data(path)

    assert loaded.config_keys == ["exp.seed"]
    assert loaded.config_values == [(42,), (7,)]
    assert loaded.run_names == ["r1", "r2"]


def test_save_load_subroom_data_preserves_nan(tmp_path):
    """NaN padding survives the roundtrip."""
    ids = torch.full((1, 3, 4), float("nan"))
    ids[0, 2] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    data = SubroomData(subroom_ids=ids, run_names=["r"], config_values=None, config_keys=[])
    path = str(tmp_path / "subroom_nan.pt")
    save_subroom_data(data, path)
    loaded = load_subroom_data(path)

    assert torch.isnan(loaded.subroom_ids[0, 0]).all()
    assert torch.isnan(loaded.subroom_ids[0, 1]).all()
    torch.testing.assert_close(loaded.subroom_ids[0, 2], ids[0, 2])
