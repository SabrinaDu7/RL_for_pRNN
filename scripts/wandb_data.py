"""Utilities for fetching WandB run traces into pandas DataFrames."""
import base64
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

    api = wandb.Api(timeout=29)
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


# ---------------------------------------------------------------------------
# Occupancy heatmap fetching
# ---------------------------------------------------------------------------


def _decode_plotly_z(z) -> np.ndarray:
    """Decode a Plotly heatmap ``z`` field into a numpy array.

    Handles two serialization formats:

    * **List-of-lists** (older Plotly / plain JSON): ``[[1, 2], [3, 4]]``
    * **Binary dict** (Plotly v6+): ``{"dtype": "f8", "bdata": "...", "shape": "3, 3"}``

    Args:
        z: The ``z`` value from a Plotly heatmap trace dict.

    Returns:
        2-D numpy array of floats.
    """
    if isinstance(z, dict) and "bdata" in z:
        raw = base64.b64decode(z["bdata"])
        dtype = np.dtype(z["dtype"])
        shape = tuple(int(s) for s in z["shape"].split(","))
        return np.frombuffer(raw, dtype=dtype).reshape(shape).astype(float)
    return np.array(z, dtype=float)


def _extract_heatmap_grids(plotly_json: dict) -> np.ndarray:
    """Extract z-arrays from all Heatmap traces in a Plotly JSON dict.

    Args:
        plotly_json: A deserialized Plotly JSON object with a ``"data"`` key
            containing trace dicts.

    Returns:
        np.ndarray of shape ``(n_traces, H, W)`` containing the z values
        from each Heatmap trace, in order.

    Raises:
        ValueError: If no heatmap traces are found.
    """
    grids = []
    for trace in plotly_json["data"]:
        if trace.get("type") == "heatmap":
            grids.append(_decode_plotly_z(trace["z"]))
    if not grids:
        raise ValueError("No heatmap traces found in Plotly JSON")
    return np.stack(grids, axis=0)


def _fetch_plotly_file(run, media_path: str, tmp_dir: str) -> dict:
    """Download and parse a Plotly JSON file from a WandB run.

    Args:
        run: A wandb Run object (from the public API).
        media_path: The path to the plotly JSON file within the run
            (e.g., ``"media/plotly/OPA_Occupancy_2360_abc.plotly.json"``).
        tmp_dir: Temporary directory to download the file into.

    Returns:
        Parsed JSON dict (the Plotly figure data).
    """
    run.file(media_path).download(root=tmp_dir, replace=True)
    full_path = os.path.join(tmp_dir, media_path)
    with open(full_path) as f:
        return json.load(f)


def _scan_occupancy_refs(
    run,
    metric: str,
    step_key: str,
) -> list[tuple[int, dict]]:
    """Scan a run's history to collect Plotly media references.

    This is the fast first phase — no file downloads, just collecting
    the ``(step, media_ref)`` pairs from ``scan_history``.

    Args:
        run: A wandb Run object.
        metric: The metric key (e.g., ``"Eval/OPA_Occupancy"``).
        step_key: The step key to use for indexing.

    Returns:
        List of ``(step_value, media_ref_dict)`` pairs.
    """
    refs: list[tuple[int, dict]] = []
    for row in run.scan_history(keys=[step_key, metric]):
        if metric not in row or step_key not in row:
            continue
        media_ref = row[metric]
        if isinstance(media_ref, dict):
            refs.append((int(row[step_key]), media_ref))
    return refs


def _download_and_extract(
    run,
    media_ref: dict,
    tmp_dir: str,
) -> np.ndarray | None:
    """Download a single Plotly file and extract its heatmap grids.

    Args:
        run: A wandb Run object (needed for ``run.file()``).
        media_ref: The media reference dict from ``scan_history``.
        tmp_dir: Temporary directory for downloads.

    Returns:
        Array of shape ``(n_traces, H, W)``, or ``None`` if the
        reference format is not recognised.
    """
    if "path" in media_ref:
        plotly_json = _fetch_plotly_file(run, media_ref["path"], tmp_dir)
    elif "data" in media_ref:
        plotly_json = media_ref
    else:
        return None
    return _extract_heatmap_grids(plotly_json)


@dataclass
class OccupancyData:
    """Container for fetched occupancy grid data across runs."""

    grids: dict[int, np.ndarray]
    """Mapping from step value to array of shape ``(n_runs, 4, H, W)``."""

    run_names: list[str]
    """Run names in the order they appear along axis 0 of each grid array."""

    config_values: list[tuple] | None
    """Config values per run (one tuple per run), or None if no config_keys."""

    config_keys: list[str]
    """The config keys that were requested."""

    hd_labels: list[str] = field(
        default_factory=lambda: ["→", "↓", "←", "↑"]
    )
    """Head-direction labels matching axis 1 of each grid array."""


def fetch_occupancy_grids(
    entity: str,
    project: str,
    metric: str = "Eval/OPA_Occupancy",
    step_key: str = "_step",
    config_keys: list[str] | None = None,
    filters: dict | None = None,
    group: str | None = None,
    max_workers: int = 32,
) -> OccupancyData:
    """Fetch occupancy heatmap grids from WandB runs and stack across runs.

    Downloads Plotly JSON files for *metric* from all matching runs,
    extracts the heatmap z-arrays, and stacks them by step.

    Uses a two-phase parallel strategy for speed:

    1. **Scan** all run histories in parallel to collect Plotly file refs.
    2. **Download** all files in parallel with up to *max_workers* threads.

    Args:
        entity: WandB entity (team or user).
        project: WandB project name.
        metric: The metric key for the logged Plotly occupancy figure.
        step_key: The step key used as the x-axis. Defaults to ``"_step"``.
            Note that ``Eval/*`` metrics use WandB's auto-incrementing
            ``_step``, not ``step_count``.
        config_keys: Optional list of dot-separated config keys to resolve
            per run (e.g., ``["exp.seed"]``).
        filters: Optional WandB API filters dict (MongoDB query format).
        group: Optional WandB group name to filter by.
        max_workers: Maximum number of parallel threads for scanning and
            downloading.  Defaults to 32 (I/O-bound work).

    Returns:
        An :class:`OccupancyData` instance whose ``grids`` maps each step
        value to an array of shape ``(n_runs, 4, H, W)``.  Runs missing
        data at a given step are filled with ``NaN``.

    Raises:
        ValueError: If no runs match the provided filters, or if no
            occupancy data is found in any run.
    """
    from tqdm import tqdm

    if config_keys is None:
        config_keys = []

    merged_filters = _build_filters(filters, group)

    api = wandb.Api(timeout=69)
    runs = api.runs(path=f"{entity}/{project}", filters=merged_filters)
    runs_list = list(runs)
    if not runs_list:
        raise ValueError(
            f"No runs found for entity='{entity}', project='{project}' "
            f"with filters={merged_filters}"
        )

    n_runs = len(runs_list)
    print(f"Found {n_runs} runs matching filters.")
    workers = min(max_workers, n_runs)

    # -- Phase 1: scan histories to collect file refs + config values ------
    def _scan_run(run):
        config_vals = tuple(
            _resolve_config_value(run.config, key) for key in config_keys
        )
        refs = _scan_occupancy_refs(run, metric, step_key)
        return run.name, config_vals, refs

    scan_results: list = [None] * n_runs
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_scan_run, run): i
            for i, run in enumerate(runs_list)
        }
        for future in tqdm(
            as_completed(futures), total=n_runs, desc="Scanning runs"
        ):
            scan_results[futures[future]] = future.result()

    run_names = [r[0] for r in scan_results]
    config_values = [r[1] for r in scan_results] if config_keys else None
    # per_run_refs[i] = [(step, media_ref), ...]
    per_run_refs = [r[2] for r in scan_results]

    total_files = sum(len(refs) for refs in per_run_refs)
    if total_files == 0:
        raise ValueError("No occupancy data found in any run.")

    # -- Phase 2: download and extract all files in parallel ---------------
    # Build a flat list of download tasks: (run_idx, step, run, media_ref)
    download_tasks = []
    for run_idx, refs in enumerate(per_run_refs):
        run = runs_list[run_idx]
        for step, media_ref in refs:
            download_tasks.append((run_idx, step, run, media_ref))

    # results_map[run_idx] = [(step, grid), ...]
    results_map: dict[int, list[tuple[int, np.ndarray]]] = {
        i: [] for i in range(n_runs)
    }
    sample_grid = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        def _do_download(task):
            run_idx, step, run, media_ref = task
            # Use run-specific subdirectory to avoid path collisions
            run_tmp = os.path.join(tmp_dir, str(run_idx))
            os.makedirs(run_tmp, exist_ok=True)
            grid = _download_and_extract(run, media_ref, run_tmp)
            return run_idx, step, grid

        dl_workers = min(max_workers, len(download_tasks))
        with ThreadPoolExecutor(max_workers=dl_workers) as pool:
            futures = [pool.submit(_do_download, t) for t in download_tasks]
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Downloading heatmaps",
            ):
                run_idx, step, grid = future.result()
                if grid is not None:
                    results_map[run_idx].append((step, grid))
                    if sample_grid is None:
                        sample_grid = grid

    if sample_grid is None:
        raise ValueError("No occupancy data found in any run.")

    # -- Phase 3: assemble into per-step arrays ----------------------------
    all_entries = [results_map[i] for i in range(n_runs)]
    all_steps = set()
    for entries in all_entries:
        all_steps.update({step for step, _ in entries})
    all_steps = sorted(all_steps)

    all_steps_new = {all_steps[-1]}  # Always include the last step
    for i in range(len(all_steps) - 1):
        if all_steps[i + 1] - all_steps[i] >= 10:
            all_steps_new.add(all_steps[i])
    
    print(f"Original steps: {all_steps}")
    print(f"Filtered steps: {sorted(all_steps_new)}")

    grids: dict[int, np.ndarray] = {}
    for step in all_steps_new:
        step_array = np.full((n_runs, *sample_grid.shape), np.nan)
        for run_idx, entries in enumerate(all_entries):
            for s, g in entries:
                if s == step:
                    step_array[run_idx] = g
                    break
        grids[step] = step_array

    return OccupancyData(
        grids=grids,
        run_names=run_names,
        config_values=config_values,
        config_keys=config_keys,
    )


def runs_per_step(
    all_entries: list[list[tuple]],
) -> dict[int, int]:
    """Count how many runs have data at each step.

    Args:
        all_entries: Per-run list of ``(step, value)`` tuples, as built by
            :func:`fetch_occupancy_grids`.

    Returns:
        Dictionary ``{step: n_runs}`` sorted by step.
    """
    from collections import Counter

    counts = Counter(step for entries in all_entries for step, _ in entries)
    return dict(sorted(counts.items()))


def plot_occupancy_average(
    avg_grids: dict[int, np.ndarray],
    steps: list[int] | None = None,
    scale: str = "viridis",
    title: str = "Average Occupancy",
) -> go.Figure:
    """Plot cross-run average occupancy heatmaps.

    Creates a grid of heatmaps: one row per requested step, 4 columns
    for head directions.

    Args:
        avg_grids: Dict mapping step to ``np.ndarray`` of shape
            ``(4, H, W)`` (one grid per head direction).
        steps: Which steps to plot. If ``None``, plots all steps.
        scale: Plotly colorscale name.
        title: Figure title.

    Returns:
        A ``plotly.graph_objects.Figure`` with the heatmap grid.
    """
    if steps is None:
        steps = sorted(avg_grids.keys())

    hd_labels = ["→", "↓", "←", "↑"]
    n_steps = len(steps)

    fig = make_subplots(
        rows=n_steps,
        cols=4,
        horizontal_spacing=0.02,
        vertical_spacing=min(0.05, 0.15 / max(n_steps - 1, 1)),
        column_titles=[f"HD {i}: {lbl}" for i, lbl in enumerate(hd_labels)],
        row_titles=[f"Step {s}" for s in steps],
    )

    for row_idx, step in enumerate(steps):
        grid = avg_grids[step]  # (4, H, W)
        for hd in range(4):
            fig.add_trace(
                go.Heatmap(z=grid[hd], showscale=False, colorscale=scale),
                row=row_idx + 1,
                col=hd + 1,
            )

    fig.update_xaxes(showticklabels=False, constrain="domain")
    fig.update_yaxes(showticklabels=False, autorange="reversed", constrain="domain")
    # Anchor each y-axis to its own x-axis so aspect ratios are independent.
    for row_idx in range(n_steps):
        for col_idx in range(4):
            ax_id = row_idx * 4 + col_idx + 1
            suffix = "" if ax_id == 1 else str(ax_id)
            fig.update_layout(
                **{f"yaxis{suffix}": dict(scaleanchor=f"x{suffix}", scaleratio=1)}
            )
    fig.update_layout(
        height=300 * n_steps,
        width=1400,
        title=title,
        title_x=0.5,
    )

    return fig


if __name__ == "__main__":

    data = fetch_occupancy_grids(                                                                                                                                
      entity="blake-richards",                                                                                                                                 
      project="curious-george-omt",                                                                                                                            
      metric="Eval/OPA_Occupancy",
      group="omt-cur-dot",  
      filters={"config.tasks.new_obj_loc": [7, 11]},   # or [7, 11] or [14, 7] depending on which condition you want to filter for                                                                                                                 
      # step_key defaults to "_step" (correct for Eval/* metrics)                                            
    )                                                                                                                                                            
                                                                                                                                                                                                                                                                   
    avg = {step: np.nanmean(grids, axis=0) for step, grids in data.grids.items()}                                                                             
    fig = plot_occupancy_average(avg)                                                                                                                            
    fig.write_image("occupancy_711.png")

    df = fetch_run_traces(
        entity="blake-richards",
        project="curious-george-omt",
        metric="Analysis/Avg Distance Travelled",
        step_key="step_count",
        config_keys=["tasks.testing.start_low_bound", "exp.curious_agent"],
        samples=15, # Goal minus ctrl only has 15 datapoints per trace
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
