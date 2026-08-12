"""Utilities for fetching WandB run traces into pandas DataFrames."""
import base64
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Sequence
import torch

from matplotlib.axes import Axes
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wandb
from jaxtyping import Float

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

    series = pd.Series(data=values, index=steps, dtype=float)
    if series.index.duplicated().any():
        series = series[~series.index.duplicated(keep="last")]
    return series


def fetch_run_traces(
    entity: str,
    project: str,
    metric: str,
    step_key: str = "_step",
    config_keys: list[str] | None = None,
    filters: dict | None = None,
    group: str | None = None,
    samples: int | None = None,
    max_runs: int | None = None,
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
        max_runs: If set, truncate to the first N runs returned by the API.

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

    api = wandb.Api(timeout=59)
    runs = api.runs(path=f"{entity}/{project}", filters=merged_filters)

    runs_list = list(runs)
    if max_runs is not None:
        runs_list = runs_list[:max_runs]
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
    labels: list[str] | None = None,
    ylabel: str = "",
    xlabel: str = "Step",
    figsize: tuple[float, float] = (10, 6),
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
        _, ax = plt.subplots(figsize=figsize)

    steps = df.columns.values

    for i, ((group_label, group_df), color) in enumerate(zip(grouped, colors)):
        mean = group_df.mean()
        std = group_df.std()
        n = group_df.count()  # non-NaN count per step
        spread = std / n.pow(0.5) if use_se else std

        if labels is not None:
            label = labels[i]
        else:
            label = str(group_label)
        ax.plot(steps, mean, color=color, label=label)
        ax.fill_between(steps, mean - spread, mean + spread, color=color, alpha=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.6)
    # ax.legend()

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

    target_loc: list[int] | None
    """Target location for this occupancy data."""

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
    target_loc: list[int] | None,
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
    THRESH = 20
    all_entries = [results_map[i] for i in range(n_runs)]
    all_steps = set()
    for entries in all_entries:
        all_steps.update({step for step, _ in entries})
    all_steps = sorted(all_steps)

    all_steps_new = {all_steps[-1]}  # Always include the last step
    for i in range(len(all_steps) - 1):
        if all_steps[i + 1] - all_steps[i] >= THRESH:
            all_steps_new.add(all_steps[i])

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
        target_loc=target_loc,
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
    scale: str = "plasma",
    title: str = "Average Occupancy",
    label_steps: list[int] | None = None,
    collapse_hd: bool = False,
) -> go.Figure:
    """Plot cross-run average occupancy heatmaps.

    Creates a grid of heatmaps: one row per requested step, either 4 columns
    for head directions or 1 column when ``collapse_hd=True``.

    Args:
        avg_grids: Dict mapping step to ``np.ndarray`` of shape
            ``(4, H, W)`` (one grid per head direction).
        steps: Which steps to plot. If ``None``, plots all steps.
        scale: Plotly colorscale name.
        title: Figure title.
        collapse_hd: If ``True``, average across all 4 head directions and
            plot a single column instead of 4.

    Returns:
        A ``plotly.graph_objects.Figure`` with the heatmap grid.
    """
    if steps is None:
        steps = sorted(avg_grids.keys())

    n_steps = len(steps)

    if collapse_hd:
        n_cols = 1
        col_titles = ["All HD (mean)"]
    else:
        n_cols = 4
        hd_labels = ["→", "↓", "←", "↑"]
        col_titles = [f"HD {i}: {lbl}" for i, lbl in enumerate(hd_labels)]

    fig = make_subplots(
        rows=n_steps,
        cols=n_cols,
        horizontal_spacing=0.02,
        vertical_spacing=min(0.05, 0.15 / max(n_steps - 1, 1)),
        column_titles=col_titles,
        row_titles=[f"Step {s}" for s in (label_steps if label_steps is not None else steps)],
    )

    global_max = max(np.nanmax(avg_grids[s]) for s in steps)
    for row_idx, step in enumerate(steps):
        grid = avg_grids[step]  # (4, H, W)
        if collapse_hd:
            grids_to_plot = [np.nanmean(grid, axis=0)]
        else:
            grids_to_plot = [grid[hd] for hd in range(4)]

        for col_idx, g in enumerate(grids_to_plot):
            show_scale = row_idx == len(steps) - 1 and col_idx == len(grids_to_plot) - 1
            fig.add_trace(
                go.Heatmap(z=g, showscale=show_scale, colorscale=scale, zmin=0, zmax=global_max),
                row=row_idx + 1,
                col=col_idx + 1,
            )

    fig.update_xaxes(showticklabels=False, constrain="domain")
    fig.update_yaxes(showticklabels=False, autorange="reversed", constrain="domain")
    # Anchor each y-axis to its own x-axis so aspect ratios are independent.
    for row_idx in range(n_steps):
        for col_idx in range(n_cols):
            ax_id = row_idx * n_cols + col_idx + 1
            suffix = "" if ax_id == 1 else str(ax_id)
            fig.update_layout(
                **{f"yaxis{suffix}": dict(scaleanchor=f"x{suffix}", scaleratio=1)} #type: ignore
            )
    fig.update_layout(
        height=300 * n_steps,
        width=1400 if not collapse_hd else 400,
        title=title,
        title_x=0.5,
    )

    return fig

def prob_roi(grid_items, radius, target_loc, sorted_steps) -> Float[torch.Tensor, "n_train_steps n_runs"]:
    """
    Computes the probability of occupancy within a radius R of a target location.
    P = occupancy(x, y) / total_occupancy
    
    Args:
        grid_items: Dictionary mapping steps to numpy arrays of shape (n_runs, 4, H, W).
                    Matches the output of data.grids.items().
        radius: Float/Int defining the L2 distance (radius) for the ROI.
        target_loc: Tuple (x, y) in MiniGrid coordinates (x = horizontal, y = vertical).
                    MiniGrid uses 1-based coordinates starting from (1, 1).

    Returns:
        torch.Tensor: Shape [n_train_steps, n_runs] containing the occupancy probability.
    """

    # Get metadata from the first entry
    first_step_data = grid_items[sorted_steps[0]]
    n_runs, num_hd, height, width = first_step_data.shape
    n_train_steps = len(sorted_steps)

    # The grids are fetched from WandB Plotly figures where z = occ[hd].T.
    # The original occ has shape (4, W-2, H-2) indexed as occ[hd, x-1, y-1].
    # After .T and re-fetch, shape is (4, H-2, W-2) indexed as grid[hd, y-1, x-1].
    # So: dim 2 (height) = MiniGrid y - 1 (rows), dim 3 (width) = MiniGrid x - 1 (cols).
    #
    # target_loc = (x, y) in MiniGrid coords → grid indices are (y-1, x-1).
    yy, xx = np.ogrid[:height, :width]  # yy = y-1 row indices, xx = x-1 col indices
    dist_from_center = np.sqrt(
        (yy - (target_loc[1] - 1))**2 + (xx - (target_loc[0] - 1))**2
    )
    roi_mask = dist_from_center <= radius  # Boolean mask of shape (H, W)
    
    # Initialize the output tensor
    prob_tensor = torch.zeros((n_train_steps, n_runs))
    
    for step_idx, step in enumerate(sorted_steps):
        # Array shape: (n_runs, 4, H, W)
        grids = grid_items[step]
        
        # 1. Sum across head directions to get total occupancy per cell
        # Result shape: (n_runs, H, W)
        total_occupancy_per_cell = np.nansum(grids, axis=1)
        
        # 2. Sum the occupancy within the ROI
        # Using boolean indexing over the last two dimensions
        # Result shape: (n_runs,)
        roi_occupancy = total_occupancy_per_cell[:, roi_mask].sum(axis=1)
        
        # 3. Sum total occupancy across the entire grid to normalize
        # Result shape: (n_runs,)
        total_grid_occupancy = total_occupancy_per_cell.reshape(n_runs, -1).sum(axis=1)
        
        # 4. Compute probability (avoid division by zero if a trajectory is empty)
        # We use np.divide to handle NaNs/Zeros gracefully
        occupancy_prob = np.divide(
            roi_occupancy, 
            total_grid_occupancy, 
            out=np.zeros_like(roi_occupancy), 
            where=total_grid_occupancy != 0
        )
        
        prob_tensor[step_idx] = torch.from_numpy(occupancy_prob)
    
    print("Computed occupancy probabilities for ROI:")
    print(prob_tensor.shape)
    return prob_tensor

def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes a bootstrap confidence interval for the mean along axis 0.

    Args:
        data: Array of shape [n_train_steps, n_runs]
        n_bootstrap: Number of bootstrap resamples
        ci: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_error, upper_error) arrays of shape [n_train_steps],
        where errors are distances from the mean (for use with ax.errorbar's yerr).
    """
    n_runs = data.shape[1]
    boot_means = np.stack([
        np.nanmean(data[:, np.random.randint(0, n_runs, size=n_runs)], axis=1)
        for _ in range(n_bootstrap)
    ], axis=1)  # [n_train_steps, n_bootstrap]

    alpha = (1 - ci) / 2
    lower = np.nanpercentile(boot_means, 100 * alpha, axis=1)
    upper = np.nanpercentile(boot_means, 100 * (1 - alpha), axis=1)
    mean = np.nanmean(data, axis=1)
    return mean - lower, upper - mean


def _sig_label(alpha: float) -> str:
    """Return a significance star string for a given alpha level.

    * for alpha >= 0.05, ** for alpha >= 0.01, *** for alpha >= 0.001.
    """
    if alpha >= 0.05:
        return "*"
    if alpha >= 0.01:
        return "**"
    return "***"


def _pval_stars(p_val: float) -> str | None:
    """Return significance stars based on p-value, or None if not significant (p >= 0.05)."""
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return None


def _draw_significance_bracket(
    ax: "Axes",
    x1: float,
    x2: float,
    y_bracket: float,
    text: str,
    y1_bottom: float,
    y2_bottom: float,
    color: str = "black",
    fontsize: int = 10,
) -> float:
    """Draw a significance bracket whose vertical lines reach down to each bar top.

    Args:
        ax: Axes to draw on.
        x1, x2: Horizontal centres of the two bars.
        y_bracket: Height of the horizontal connecting line.
        text: Annotation text, e.g. ``"**"``.
        y1_bottom: Top of the left bar (where the left vertical line starts).
        y2_bottom: Top of the right bar (where the right vertical line starts).
        color: Line and text colour.
        fontsize: Font size for the annotation text.

    Returns:
        ``y_bracket`` (useful for stacking further brackets above this one).
    """
    ax.plot([x1, x1, x2, x2], [y1_bottom, y_bracket, y_bracket, y2_bottom],
            color=color, linewidth=0.75)
    ax.text((x1 + x2) / 2, y_bracket, text, ha="center", va="bottom",
            color=color, fontsize=fontsize)
    return y_bracket


def pairwise_ttest(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    pair_labels: list[str] | None = None,
    alpha: float = 0.05,
) -> tuple[list[float], list[float], list[float]]:
    """Run Welch's t-test for each (a, b) pair and return p-values, degrees of freedom, and t-statistics.

    Args:
        pairs: List of ``(a, b)`` where ``a`` and ``b`` are 1D arrays of samples.
        pair_labels: Optional label per pair for printed output.
        alpha: Significance level used for labelling printed output.

    Returns:
        Tuple of (p_vals, dfs, t_stats): lists of p-values, Welch–Satterthwaite degrees of
        freedom, and t-statistics, one entry per pair.
    """
    from scipy import stats

    def compute_sig(p_val: float) -> float | str:
        if p_val < 0.001:
            return 0.001
        elif p_val < 0.01:
            return 0.01
        elif p_val < 0.05:
            return 0.05
        else:
            return "ns"

    def welch_df(a: np.ndarray, b: np.ndarray) -> float:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        n1, n2 = len(a), len(b)
        v1 = np.var(a, ddof=1)
        v2 = np.var(b, ddof=1)
        num = (v1 / n1 + v2 / n2) ** 2
        denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        return float(num / denom) if denom != 0 else float("nan")

    p_vals, dfs, t_stats = [], [], []
    for idx, (a, b) in enumerate(pairs):
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        t_stat, p_val = float(t_stat), float(p_val)
        df = welch_df(a, b)
        p_vals.append(p_val)
        dfs.append(df)
        t_stats.append(t_stat)
        label = pair_labels[idx] if pair_labels is not None else f"pair {idx}"
        sig_level = compute_sig(p_val)
        sig = _sig_label(sig_level) if isinstance(sig_level, float) else sig_level
        print(f"  {label}: t={t_stat:.4f}, df={df:.2f}, p={p_val:.4f} → {sig}")
    return p_vals, dfs, t_stats


def plot_metric(tensors: list[Float[torch.Tensor, "n_train_steps n_runs"]], steps: list, colors: list, labels: list[str], xlabel: str, ylabel: str, use_std: bool = True, n_bootstrap: int = 1000, alpha=0.01, xlim_max: int | None = None):
    """
    Plots the mean of a metric with a 95% Confidence Interval for multiple agents.

    Args:
        tensors: List of tensors, each of shape [n_train_steps, n_runs]
        steps: List of step values (x-axis)
        colors: List of colors, one per tensor
        labels: List of legend labels, one per tensor
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
        use_std: If True, shade with std; if False, use bootstrap 95% CI
        n_bootstrap: Number of bootstrap resamples (only used when use_std=False)
        alpha: Significance level for Welch's t-test (default 0.01)
        xlim_max: If set, clips the plotted line and x-axis to this value so the
            line ends cleanly at the last datapoint within the range.
    """
    from itertools import combinations

    if len(tensors) != len(colors):
        raise ValueError(f"tensors and colors must have the same length, got {len(tensors)} and {len(colors)}")
    if len(tensors) != len(labels):
        raise ValueError(f"tensors and labels must have the same length, got {len(tensors)} and {len(labels)}")

    ref_shape = tensors[0].shape
    for i, t in enumerate(tensors[1:], start=1):
        if t.shape[0] != ref_shape[0]:
            raise ValueError(f"All tensors must have the same shape. tensors[0] has shape {ref_shape}, but tensors[{i}] has shape {t.shape}")

    fig, ax = plt.subplots(figsize=(10, 6))

    all_data = []
    all_means = []
    all_yerrs = []
    for tensor, color, label in zip(tensors, colors, labels):
        data = tensor.numpy()
        all_data.append(data)
        mean = np.nanmean(data, axis=-1)

        if use_std:
            yerr = np.nanstd(data, axis=-1)
        else:
            yerr = np.stack(bootstrap_ci(data, n_bootstrap=n_bootstrap), axis=0)  # [2, n_steps]

        all_means.append(mean)
        all_yerrs.append(yerr)

        print(f"Plotting {label}:")
        print("     Final mean:", mean[-1])
        print("     Final yerr:", yerr[-1])
        light_color = (*mcolors.to_rgb(color), 0.25)

        if xlim_max is not None:
            idx = [i for i, s in enumerate(steps) if s <= xlim_max]
            plot_steps = [steps[i] for i in idx]
            plot_mean = mean[idx]
            plot_yerr = yerr[:, idx] if yerr.ndim == 2 else yerr[idx]
        else:
            plot_steps, plot_mean, plot_yerr = steps, mean, yerr

        ax.plot(plot_steps, plot_mean, color=color, label=label, linewidth=2,
                marker="o", markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.5)
        ax.errorbar(plot_steps, plot_mean, yerr=plot_yerr,
                    fmt="none", ecolor=light_color, elinewidth=1.5, capsize=3)

    # Welch's t-test (unequal variances) on all steps for all pairs
    label_pairs = [
        (label_i, label_j)
        for (_, label_i), (_, label_j) in combinations(enumerate(labels), 2)
    ]
    idx_pairs = list(combinations(range(len(labels)), 2))
    print(f"\nWelch's t-test (two-sided, α={alpha}) across all steps:")
    all_p_vals: dict[tuple[str, str], list[float]] = {}
    all_dfs: dict[tuple[str, str], list[float]] = {}
    all_t_stats: dict[tuple[str, str], list[float]] = {}
    for (i, j), (label_i, label_j) in zip(idx_pairs, label_pairs):
        pairs = [(all_data[i][s, :], all_data[j][s, :]) for s in range(len(steps))]
        pair_labels = [f"step {steps[s]}" for s in range(len(steps))]
        print(f"\n  {label_i} vs {label_j}:")
        p_vals_pair, dfs_pair, t_stats_pair = pairwise_ttest(pairs, pair_labels=pair_labels, alpha=alpha)
        all_p_vals[(label_i, label_j)] = p_vals_pair
        all_dfs[(label_i, label_j)] = dfs_pair
        all_t_stats[(label_i, label_j)] = t_stats_pair

    # Formatting
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    if xlim_max is not None:
        ax.set_xlim(right=xlim_max)
        visible_idx = [i for i, s in enumerate(steps) if s <= xlim_max]
        if visible_idx:
            visible_ys = np.concatenate([m[visible_idx] for m in all_means])
            ymin, ymax = float(np.nanmin(visible_ys)), float(np.nanmax(visible_ys))
            pad = max((ymax - ymin) * 0.1, 1e-6)
            ax.set_ylim(ymin - pad, ymax + pad)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.legend()

    plt.tight_layout()
    return fig, ax, all_p_vals, all_dfs, all_t_stats


def add_sig_line(
    ax,
    steps: list,
    p_vals: dict[tuple[str, str], list[float]],
    exp_label: str,
    control_labels: list[str],
    control_colors: list[str],
    height: float,
    alpha: float = 0.05,
    row_spacing: float = 0.0,
    fontsize: int = 9,
) -> None:
    """Draw significance annotations on an existing plot.

    For each control condition, draws a thin grey horizontal line at ``height``
    (offset vertically by ``row_spacing`` per additional control) and places
    significance stars at steps where the t-test p-value < alpha.
    Stars are colored the same as the corresponding control condition.

    Args:
        ax: Axes to annotate.
        steps: x-axis step values (same list passed to plot_metric).
        p_vals: Dict returned by plot_metric: {(label_i, label_j): [p_val_per_step]}.
        exp_label: Label of the experimental condition.
        control_labels: Labels of control conditions, in order.
        control_colors: Colors matching control_labels (same order).
        height: y-position for the first (or only) row of annotations.
        alpha: Significance threshold (default 0.05).
        row_spacing: Extra y offset between rows when there are multiple controls.
        fontsize: Font size for the star annotations (default 9).
    """
    for k, (ctrl_label, ctrl_color) in enumerate(zip(control_labels, control_colors)):
        y = height + k * row_spacing
        pv = p_vals.get((exp_label, ctrl_label)) or p_vals.get((ctrl_label, exp_label))
        if pv is None:
            raise KeyError(f"No p-values found for pair ({exp_label}, {ctrl_label})")

        sig_steps = [steps[s] for s, p in enumerate(pv) if p < alpha]
        if not sig_steps:
            continue

        ax.hlines(y, min(sig_steps), max(sig_steps), colors="grey", linewidth=0.8, zorder=3)

        for s, p in enumerate(pv):
            if p >= alpha:
                continue
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*"
            ax.text(steps[s], y, star, ha="center", va="bottom",
                    color=ctrl_color, fontsize=fontsize, zorder=4)


# ---------------------------------------------------------------------------
# Subroom count fetching
# ---------------------------------------------------------------------------


@dataclass
class SubroomData:
    """Container for fetched subroom visit count data across runs."""

    subroom_ids: "Float[torch.Tensor, 'n_runs n_training_steps 4']"
    """Shape (n_runs, n_training_steps, 4): visit counts per room per training step."""

    run_names: list[str]
    """Run names in the order they appear along axis 0."""

    config_values: list[tuple] | None
    """Config values per run (one tuple per run), or None if no config_keys."""

    config_keys: list[str]
    """The config keys that were requested."""


def _histogram_to_subroom_counts(hist: dict, n_subrooms: int = 4) -> np.ndarray:
    """Convert a wandb histogram dict to per-subroom visit counts.

    Bin centers are rounded to the nearest integer to determine which subroom
    each bin belongs to. Bins whose rounded center is outside [0, n_subrooms)
    are ignored.

    Args:
        hist: A dict with ``"bins"`` (n+1 edges) and ``"values"`` (n counts).
        n_subrooms: Number of subrooms (size of the output array).

    Returns:
        np.ndarray of shape ``(n_subrooms,)`` with visit counts per subroom.
    """
    bins = np.array(hist["bins"])
    values = np.array(hist["values"])
    centers = (bins[:-1] + bins[1:]) / 2
    counts = np.zeros(n_subrooms)
    for center, val in zip(centers, values):
        idx = round(center)
        if 0 <= idx < n_subrooms:
            counts[idx] += val
    return counts


def fetch_subroom_counts(
    entity: str,
    project: str,
    last_n: int | None = None,
    metric: str = "subroom_ids",
    step_key: str = "_step",
    config_keys: list[str] | None = None,
    filters: dict | None = None,
    group: str | None = None,
    max_workers: int = 32,
    page_size: int = 10000,
) -> SubroomData:
    """Fetch subroom visit count data from WandB runs.

    ``subroom_ids`` is logged as ``wandb.Histogram`` of per-frame room IDs
    (1–4 in MiniGrid convention). This function converts those histograms into
    per-room counts and aligns them across runs.

    Args:
        entity: WandB entity (team or user).
        project: WandB project name.
        last_n: If given, keep only the last ``last_n`` logged steps per run,
            front-padding with NaN for runs with fewer entries. If ``None``,
            uses the maximum number of steps across all runs.
        metric: WandB metric key for the histogram. Defaults to ``"subroom_ids"``.
        step_key: Step key used as the x-axis. Defaults to ``"_step"``.
        config_keys: Optional dot-separated config keys to resolve per run.
        filters: Optional WandB API filters dict (MongoDB query format).
        group: Optional WandB group name to filter by.
        max_workers: Maximum parallel threads for scanning run histories.
        page_size: Number of rows fetched per API call in ``scan_history``.
            Larger values reduce round-trips. Defaults to 10000.

    Returns:
        A :class:`SubroomData` with ``subroom_ids`` of shape
        ``(n_runs, n_training_steps, 4)``.

    Raises:
        ValueError: If no runs match the provided filters or no subroom data
            is found in any run.
    """
    from tqdm import tqdm

    if config_keys is None:
        config_keys = []

    merged_filters = _build_filters(filters, group)
    api = wandb.Api(timeout=59)
    runs = api.runs(path=f"{entity}/{project}", filters=merged_filters)
    runs_list = list(runs)
    if not runs_list:
        raise ValueError(
            f"No runs found for entity='{entity}', project='{project}' "
            f"with filters={merged_filters}"
        )

    n_runs = len(runs_list)
    workers = min(max_workers, n_runs)

    def _scan_run(run):
        config_vals = tuple(
            _resolve_config_value(run.config, key) for key in config_keys
        )
        entries: list[tuple[int, np.ndarray]] = []
        for row in run.scan_history(keys=[step_key, metric], page_size=page_size):
            if metric not in row or step_key not in row:
                continue
            hist = row[metric]
            if not isinstance(hist, dict) or "bins" not in hist:
                continue
            # Actual subroom IDs are 1-indexed (1–4); shift bins by -1 so that
            # _histogram_to_subroom_counts maps them to 0-indexed (0–3).
            shifted_hist = {
                "bins": [b - 1 for b in hist["bins"]],
                "values": hist["values"],
            }
            counts = _histogram_to_subroom_counts(shifted_hist, n_subrooms=4)
            entries.append((int(row[step_key]), counts))
        return run.name, config_vals, entries

    scan_results: list = [None] * n_runs
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scan_run, run): i for i, run in enumerate(runs_list)}
        for future in tqdm(as_completed(futures), total=n_runs, desc="Scanning runs"):
            scan_results[futures[future]] = future.result()

    run_names = [r[0] for r in scan_results]
    config_values = [r[1] for r in scan_results] if config_keys else None
    per_run_entries: list[list[tuple[int, np.ndarray]]] = [r[2] for r in scan_results]

    # Determine the number of time steps to keep
    max_entries = max((len(e) for e in per_run_entries), default=0)
    n_steps = last_n if last_n is not None else max_entries

    # Build (n_runs, n_steps, 4) tensor, front-padding short runs with NaN
    result = torch.full((n_runs, n_steps, 4), float("nan"))
    for run_idx, entries in enumerate(per_run_entries):
        if not entries:
            continue
        # Sort by step, then take the last n_steps
        entries_sorted = sorted(entries, key=lambda x: x[0])
        entries_to_use = entries_sorted[-n_steps:] if n_steps else entries_sorted
        counts_array = np.stack([c for _, c in entries_to_use], axis=0)  # (T, 4)
        t = counts_array.shape[0]
        result[run_idx, n_steps - t :] = torch.from_numpy(counts_array.astype(np.float32))

    return SubroomData(
        subroom_ids=result,
        run_names=run_names,
        config_values=config_values,
        config_keys=config_keys,
    )


def save_subroom_data(data: SubroomData, path: str) -> None:
    """Save a :class:`SubroomData` instance to a file.

    Args:
        data: The SubroomData to save.
        path: Destination file path (conventionally ``.pt``).
    """
    torch.save(
        {
            "subroom_ids": data.subroom_ids,
            "run_names": data.run_names,
            "config_values": data.config_values,
            "config_keys": data.config_keys,
        },
        path,
    )


def load_subroom_data(path: str) -> SubroomData:
    """Load a :class:`SubroomData` instance from a file saved by :func:`save_subroom_data`.

    Args:
        path: Path to the saved file.

    Returns:
        The loaded :class:`SubroomData`.
    """
    d = torch.load(path, weights_only=False)
    return SubroomData(
        subroom_ids=d["subroom_ids"],
        run_names=d["run_names"],
        config_values=d["config_values"],
        config_keys=d["config_keys"],
    )


_SUBROOM_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]


def _subroom_pct_array(
    counts: "Float[torch.Tensor, 'n_runs n_training_steps n_rooms']",
    step_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return per-run mean subroom percentages, shape ``(n_runs, n_rooms)``.

    For each run, visit counts are normalised to percentages at each step, then
    averaged over steps (NaN rows excluded).
    """
    data = counts.numpy() if isinstance(counts, torch.Tensor) else np.asarray(counts, dtype=float)
    if step_range is not None:
        data = data[:, step_range[0]:step_range[1], :]
    row_totals = np.nansum(data, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(row_totals > 0, data / row_totals * 100, np.nan)
    return np.nanmean(pct, axis=1)  # (n_runs, n_rooms)


def _subroom_percentages(
    counts: "Float[torch.Tensor, 'n_runs n_training_steps n_rooms']",
    step_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and SE of subroom percentages across (run, step) samples.

    Each ``(run, training_step)`` pair is one sample. Returns mean and SE per room.
    """
    data = counts.numpy() if isinstance(counts, torch.Tensor) else np.asarray(counts, dtype=float)
    if step_range is not None:
        data = data[:, step_range[0]:step_range[1], :]

    n_rooms = data.shape[-1]
    row_totals = np.nansum(data, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(row_totals > 0, data / row_totals * 100, np.nan)

    flat = pct.reshape(-1, n_rooms)
    n_valid = int((~np.isnan(flat).all(axis=-1)).sum())
    if n_valid == 0:
        return np.zeros(n_rooms), np.zeros(n_rooms)
    mean_pct = np.nanmean(flat, axis=0)
    se_pct = np.nanstd(flat, axis=0) / np.sqrt(n_valid) if n_valid > 1 else np.zeros(n_rooms)
    return mean_pct, se_pct


_SUBROOM_HATCHES = ["/", "", ".", "x", "\\", "o", "+"]


def plot_subroom_percentage(
    counts_list: "list[Float[torch.Tensor, 'n_runs n_training_steps n_rooms']]",
    group_labels: "Sequence[str | None] | None" = None,
    colors: "Sequence[str] | None" = None,
    hatches: "Sequence[str] | None" = None,
    room_labels: "Sequence[str] | None" = None,
    ax: "Axes | None" = None,
    step_range: tuple[int, int] | None = None,
    alpha: float = 0.05,
    reorder_rooms: bool = True,
) -> "Axes":
    """Plot mean percentage of time spent in each subroom, one bar group per entry.

    Each ``(run, training_step)`` pair is treated as a single sample for mean/SE.
    Multiple groups (e.g. random vs. curious agent) are plotted as side-by-side
    grouped bars in different colors.

    Args:
        counts_list: List of tensors, each of shape
            ``(n_runs, n_training_steps, n_rooms)``.
        group_labels: Legend labels for each group, e.g. ``["random", "curious"]``.
            ``None`` entries suppress the label for that group; if all are ``None``
            no legend is shown.
        colors: Bar colors per group. Defaults to ``_SUBROOM_COLORS``.
        hatches: Hatch patterns per group for accessibility (e.g. ``["", "/", "."]``).
            Defaults to ``_SUBROOM_HATCHES``.
        room_labels: X-axis labels for each room. Defaults to
            ``["Room 1", ..., "Room N"]``.
        ax: Optional matplotlib Axes to plot on. Creates a new figure if ``None``.
        step_range: Optional ``(start, end)`` slice of training steps (exclusive end).
        alpha: Significance level for per-subroom Welch's t-test (printed when
            more than one group is provided).

    Returns:
        The matplotlib Axes with the bar chart.
    """
    counts_list = [tensor[:, :, [0, 2, 3, 1]] for tensor in counts_list]  # Reorder from (1,2,3,4) to (1,3,4,2)
    n_groups = len(counts_list)
    first = counts_list[0]
    n_rooms = (first.numpy() if isinstance(first, torch.Tensor) else np.asarray(first)).shape[-1]

    resolved_labels: list[str | None] = list(group_labels) if group_labels is not None else [None] * n_groups
    resolved_colors: list[str] = list(colors) if colors is not None else [_SUBROOM_COLORS[i % len(_SUBROOM_COLORS)] for i in range(n_groups)]
    resolved_hatches: list[str] = list(hatches) if hatches is not None else [_SUBROOM_HATCHES[i % len(_SUBROOM_HATCHES)] for i in range(n_groups)]
    resolved_rooms: list[str] = list(room_labels) if room_labels is not None else [f"Room {i + 1}" for i in range(n_rooms)]

    if ax is None:
        _, ax = plt.subplots()

    x = np.arange(n_rooms)
    width = 0.8 / n_groups

    # Collect per-group stats so we can position brackets after plotting
    group_stats: list[tuple[np.ndarray, np.ndarray]] = []
    for i, (counts, label, color, hatch) in enumerate(zip(counts_list, resolved_labels, resolved_colors, resolved_hatches)):
        mean_pct, se_pct = _subroom_percentages(counts, step_range=step_range)
        print(f"{label or f'Group {i}'} mean percentages: {mean_pct}, SE: {se_pct}")
        group_stats.append((mean_pct, se_pct))
        offset = (i - n_groups / 2 + 0.5) * width
        ax.bar(x + offset, mean_pct, width, yerr=se_pct, label=label, color=color,
               hatch=hatch, capsize=4, error_kw=dict(elinewidth=2, ecolor="black"))

    ax.set_xticks(x)
    ax.set_xticklabels(resolved_rooms)
    ax.set_ylabel("% time in subroom")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    if any(l is not None for l in resolved_labels):
        ax.legend()

    # Per-subroom Welch's t-test and significance brackets for all group pairs
    if n_groups > 1:
        from itertools import combinations as _combinations
        run_means = [_subroom_pct_array(c, step_range=step_range) for c in counts_list]
        # bracket_top[r] tracks how high we've stacked brackets for room r
        bracket_top = np.array([
            max(group_stats[k][0][r] + group_stats[k][1][r] for k in range(n_groups))
            for r in range(n_rooms)
        ]) + 2.0

        for (i, label_i), (j, label_j) in _combinations(enumerate(resolved_labels), 2):
            lbl_i = label_i or f"group {i}"
            lbl_j = label_j or f"group {j}"
            print(f"\nWelch's t-test per subroom ({lbl_i} vs {lbl_j}, α={alpha}):")
            pairs = [(run_means[i][:, r], run_means[j][:, r]) for r in range(n_rooms)]
            p_vals, _dfs, _ = pairwise_ttest(pairs, pair_labels=resolved_rooms)

            offset_i = (i - n_groups / 2 + 0.5) * width
            offset_j = (j - n_groups / 2 + 0.5) * width
            for r, p_val in enumerate(p_vals):
                stars = _pval_stars(p_val)
                if stars is not None:
                    mean_i, se_i = group_stats[i]
                    mean_j, se_j = group_stats[j]

                    _draw_significance_bracket(
                        ax,
                        x1=x[r] + offset_i,
                        x2=x[r] + offset_j,
                        y_bracket=bracket_top[r],
                        text=stars,
                        y1_bottom=mean_i[r] + se_i[r] + 1,
                        y2_bottom=mean_j[r] + se_j[r] + 1,
                    )
                    bracket_top[r] += 4.0

    return ax
