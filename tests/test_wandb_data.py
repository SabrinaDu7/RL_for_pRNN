"""Tests for scripts.wandb_data module."""

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scripts.wandb_data import (
    _build_filters,
    _fetch_history,
    _resolve_config_value,
    _unwrap_wandb_config,
    fetch_run_traces,
    plot_traces,
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
class MockRun:
    """Lightweight stand-in for a wandb Run."""
    name: str
    config: dict
    _history: list[dict] = field(default_factory=list)

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
