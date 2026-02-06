"""Tests for scripts.wandb_data module."""

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import json
import os

import numpy as np
import plotly.graph_objects as go

from scripts.wandb_data import (
    OccupancyData,
    _build_filters,
    _download_and_extract,
    _extract_heatmap_grids,
    _fetch_history,
    _fetch_plotly_file,
    _resolve_config_value,
    _scan_occupancy_refs,
    _unwrap_wandb_config,
    fetch_occupancy_grids,
    fetch_run_traces,
    load_occupancy,
    plot_occupancy_average,
    plot_traces,
    runs_per_step,
    save_occupancy,
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


# ---------------------------------------------------------------------------
# save_occupancy / load_occupancy
# ---------------------------------------------------------------------------

def _make_occupancy_data() -> OccupancyData:
    """Build a small OccupancyData for save/load tests."""
    rng = np.random.default_rng(42)
    g10 = rng.random((3, 4, 5, 5))        # 3 runs, 4 HDs, 5x5 grid
    g20 = rng.random((3, 4, 5, 5))
    g20[2] = np.nan                         # run 2 missing at step 20
    return OccupancyData(
        grids={10: g10, 20: g20},
        run_names=["run_a", "run_b", "run_c"],
        config_values=[(1,), (2,), (3,)],
        config_keys=["exp.seed"],
    )


def test_save_load_roundtrip(tmp_path):
    """Save then load produces identical OccupancyData."""
    original = _make_occupancy_data()
    save_occupancy(original, "test_occ", output_dir=tmp_path)
    loaded = load_occupancy("test_occ", output_dir=tmp_path)

    assert loaded.run_names == original.run_names
    assert loaded.config_keys == original.config_keys
    assert loaded.config_values == original.config_values
    assert loaded.hd_labels == original.hd_labels
    assert sorted(loaded.grids.keys()) == sorted(original.grids.keys())
    for step in original.grids:
        np.testing.assert_array_almost_equal(
            loaded.grids[step], original.grids[step],
        )


def test_save_load_no_config(tmp_path):
    """Roundtrip works when config_values is None."""
    data = OccupancyData(
        grids={0: np.ones((1, 4, 3, 3))},
        run_names=["r1"],
        config_values=None,
        config_keys=[],
    )
    save_occupancy(data, "no_cfg", output_dir=tmp_path)
    loaded = load_occupancy("no_cfg", output_dir=tmp_path)

    assert loaded.config_values is None
    assert loaded.config_keys == []
    np.testing.assert_array_equal(loaded.grids[0], data.grids[0])


def test_save_creates_files(tmp_path):
    """Save creates grids.parquet and metadata.json."""
    data = _make_occupancy_data()
    out = save_occupancy(data, "check_files", output_dir=tmp_path)

    assert (out / "grids.parquet").exists()
    assert (out / "metadata.json").exists()
