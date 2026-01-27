"""Utilities for fetching WandB run traces into pandas DataFrames."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import wandb

# Fetching from wandb functions
def _unwrap_wandb_config(config: dict) -> dict:
    """Unwrap wandb's {"key": {"value": ...}} config envelope.

    Wandb wraps each config section in a {"value": <actual>} dict.
    This strips that wrapper so keys resolve normally.
    """
    unwrapped = {}
    for k, v in config.items():
        if isinstance(v, dict) and "value" in v and len(v) == 1:
            unwrapped[k] = v["value"]
        else:
            unwrapped[k] = v
    return unwrapped

def ensure_dict(config: str | dict) -> dict:
    """Convert a string representation of a dictionary to an actual dictionary.

    Args:
        config: A string representation of a dictionary or an actual dictionary.

    Returns:
        A dictionary.
    """
    if isinstance(config, str):
        import json
        config = json.loads(config)

    assert isinstance(config, dict)
    return config


def _resolve_config_value(config: dict, dotted_key: str) -> Any:
    """Resolve a dot-separated key from a wandb config dict.

    Handles wandb's {"key": {"value": ...}} envelope format,
    flat configs (dotted keys at the top level), and plain nested dicts.

    Args:
        config: The config dict from wandb run.config.
        dotted_key: A dot-separated path like "exp.seed" or "rl.lr".

    Returns:
        The value at that path.

    Raises:
        KeyError: If the key path does not exist in the config.
    """
    config = ensure_dict(config)
    config = _unwrap_wandb_config(config)

    # Flat lookup first (wandb sometimes stores dotted keys at the top level)
    if dotted_key in config:
        value = config[dotted_key]
    else:
        # Nested traversal fallback
        keys = dotted_key.split(".")
        current = config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                raise KeyError(
                    f"Config key '{dotted_key}' not found: "
                    f"failed at segment '{key}'"
                )
            current = current[key]
        value = current

    # Convert lists to tuples so values are hashable for MultiIndex
    if isinstance(value, list):
        value = tuple(value)
    return value


def _build_filters(
    filters: dict | None,
    group: str | None,
) -> dict | None:
    """Merge group filter with user-provided filters.

    Args:
        filters: User-provided MongoDB-style filters dict, or None.
        group: Optional group name to filter by.

    Returns:
        Merged filters dict, or None if both inputs are None.
    """
    if group is None and filters is None:
        return None
    if group is None:
        return filters
    group_filter = {"group": group}
    if filters is None:
        return group_filter
    return {"$and": [filters, group_filter]}


def _fetch_history(run, step_key: str, metric: str, samples: int | None) -> pd.Series:
    """Fetch a single metric trace from a wandb run.

    Args:
        run: A wandb Run object.
        step_key: The key used as the x-axis (e.g., "_step", "step_count").
        metric: The metric key to fetch.
        samples: If None, use scan_history (exact, slower). If an int, use
            history(samples=N) (sampled, faster).

    Returns:
        pd.Series with step values as index and metric values as data.
    """
    steps = []
    values = []

    if samples is None:
        rows = run.scan_history(keys=[step_key, metric])
    else:
        rows = run.history(keys=[step_key, metric], samples=samples).to_dict("records")

    for row in rows:
        if step_key in row and metric in row:
            steps.append(row[step_key])
            values.append(row[metric])

    return pd.Series(data=values, index=steps, dtype=float)


def fetch_run_traces(
    entity: str,
    project: str,
    metric: str,
    step_key: str = "_step",
    config_keys: list[str] | None = None,
    filters: dict | None = None,
    group: str | None = None,
    samples: int | None = None,
) -> pd.DataFrame:
    """Fetch metric traces from WandB runs into a multi-indexed DataFrame.

    Each run produces a "metric vs step" curve. The function fetches all
    matching runs and assembles them into a single DataFrame where columns
    are timesteps and rows are runs.

    Args:
        entity: WandB entity (team or user).
        project: WandB project name.
        metric: The metric key to fetch (e.g., "return_mean", "policy_loss").
        step_key: The key used as the x-axis / timestep. Defaults to "_step"
            (wandb's auto-increment). Use "step_count" for OMT runs that
            define a custom step metric.
        config_keys: Optional list of dot-separated config keys to include in
            the MultiIndex (e.g., ["exp.seed", "rl.lr"]). These are resolved
            from the nested Hydra config stored in WandB.
        filters: Optional WandB API filters dict (MongoDB query format).
            Example: {"config.exp.curious_agent": True}
        group: Optional WandB group name to filter by. Convenience shorthand
            that is merged into filters.
        samples: If None (default), use scan_history for exact data. If an int,
            use history(samples=N) for faster but potentially sampled data.

    Returns:
        pd.DataFrame with:
            - Columns: step values (sorted, union of all runs' steps)
            - Rows: one per run
            - Index: pd.MultiIndex with levels ["run_name"] + config_keys
              (or just ["run_name"] if config_keys is None/empty)
            - Values: the metric value at each step (NaN where missing)

    Raises:
        ValueError: If no runs match the provided filters.
    """
    if config_keys is None:
        config_keys = []

    merged_filters = _build_filters(filters, group)

    api = wandb.Api()
    runs = api.runs(path=f"{entity}/{project}", filters=merged_filters)

    runs_list = list(runs)
    if not runs_list:
        raise ValueError(
            f"No runs found for entity='{entity}', project='{project}' "
            f"with filters={merged_filters}"
        )

    def _process_run(run):
        """Extract index tuple and metric series from a single run."""
        config_values = [
            _resolve_config_value(run.config, key) for key in config_keys
        ]
        index_tuple = (run.name, *config_values)
        series = _fetch_history(run, step_key, metric, samples)
        return index_tuple, series

    # Fetch histories in parallel (network I/O bound)
    results = [None] * len(runs_list)
    with ThreadPoolExecutor(max_workers=min(8, len(runs_list))) as pool:
        futures = {
            pool.submit(_process_run, run): i
            for i, run in enumerate(runs_list)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result() # type: ignore

    index_tuples = [r[0] for r in results] # type: ignore
    all_series = [r[1] for r in results] # type: ignore

    index_names = ["run_name"] + config_keys
    if len(index_names) == 1:
        index = pd.Index([t[0] for t in index_tuples], name="run_name")
    else:
        index = pd.MultiIndex.from_tuples(index_tuples, names=index_names)

    df = pd.DataFrame(all_series, index=index)
    df = df.sort_index(axis=1)

    return df

# Plotting functions

def plot_traces(
    df: pd.DataFrame,
    group_keys: list[str],
    colors: list[str],
    use_se: bool = True,
    ax: Axes | None = None,
    ylabel: str = "",
    xlabel: str = "Step",
) -> Axes:
    """Plot mean +/- spread for each group defined by config key combinations.

    Args:
        df: DataFrame from fetch_run_traces with a MultiIndex containing
            the requested group_keys.
        group_keys: Config key names to group by (must be levels in df.index).
            Each unique combination of values across these keys becomes one curve.
        colors: List of colors, one per group combination. Length must match
            the number of unique groups.
        use_se: If True, shade mean +/- SEM. If False, shade mean +/- SD.
        ax: Optional matplotlib Axes to plot on. If None, creates a new figure.
        ylabel: Label for the y-axis.
        xlabel: Label for the x-axis.

    Returns:
        The matplotlib Axes with the plot.

    Raises:
        ValueError: If group_keys are not in the DataFrame index, or if
            the number of colors doesn't match the number of groups.
    """
    import matplotlib.pyplot as plt

    # Validate that group_keys are present in the index
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            f"DataFrame must have a MultiIndex to group by {group_keys}. "
            f"Got index with name='{df.index.name}'."
        )
    missing = [k for k in group_keys if k not in df.index.names]
    if missing:
        raise ValueError(
            f"group_keys {missing} not found in DataFrame index levels "
            f"{list(df.index.names)}."
        )

    # Use scalar level when single key to avoid FutureWarning
    level = group_keys[0] if len(group_keys) == 1 else group_keys
    grouped = df.groupby(level=level)
    n_groups = grouped.ngroups

    if len(colors) != n_groups:
        raise ValueError(
            f"Expected {n_groups} colors (one per group), got {len(colors)}."
        )

    if ax is None:
        _, ax = plt.subplots()

    steps = df.columns.values

    for (group_label, group_df), color in zip(grouped, colors):
        mean = group_df.mean()
        std = group_df.std()
        n = group_df.count()  # non-NaN count per step
        spread = std / n.pow(0.5) if use_se else std

        label = str(group_label)
        ax.plot(steps, mean, color=color, label=label)
        ax.fill_between(steps, mean - spread, mean + spread, color=color, alpha=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    return ax


if __name__ == "__main__":
    df = fetch_run_traces(
        entity="blake-richards",
        project="curious-george-omt",
        metric="Analysis/Avg Distance Travelled",
        step_key="step_count",
        config_keys=["tasks.testing.start_low_bound", "exp.curious_agent"],
        samples=20, # Goal minus ctrl only has 15 datapoints per trace
    )

    ax = plot_traces(                                                                           
      df,                          # DataFrame from fetch_run_traces                     
      group_keys=["tasks.testing.start_low_bound", "exp.curious_agent"],  # config keys to group by                           
      colors=["red", "blue", "green", "orange", "purple", "brown"],   # one color per group combination                   
      use_se=True,                   # True=SEM shading, False=SD shading                
      ax=None,                       # optional existing Axes                            
      ylabel="Goal Modulation",                                                          
      xlabel="Step",                                                                     
    ) 
    plt.savefig("plot.png")
