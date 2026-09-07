"""Reading finished runs back out of wandb: the occupancy grids the questions repo consumes.

What is left of a 1,488-line analysis module (traces, bootstrap CIs, t-tests,
subroom percentages, significance brackets) whose only consumer outside this
file was `fetch_occupancy_grids` (../experiment-curiousgeorge, Q1). The rest
had no caller anywhere and was deleted 2026-09-06 (audit 2026-09-05, §3).
"""
import base64
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
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
    for row in _history_rows(run, metric=metric, step_key=step_key):
        if metric not in row or step_key not in row:
            continue
        media_ref = row[metric]
        if isinstance(media_ref, dict):
            refs.append((int(row[step_key]), media_ref))
    return refs


def _history_rows(run, *, metric: str, step_key: str) -> list[dict]:
    """History rows for `metric`, via `scan_history` when it works.

    `scan_history` is the streaming reader and the right default, but it raises
    `Step column '_step' not found in schema` on runs whose media were logged
    through their own `wandb.log()` calls rather than the update log - which is
    every OMT run, and is recorded in this project's methodology notes.
    `run.history()` reads the same rows through a different endpoint and works
    there, at the cost of being capped (10,000 samples) and materialising in
    memory.

    The cap does NOT bite here: this reads media REFERENCES for an eval that
    fires a handful of times per run, not a dense scalar series. It would bite
    on a training curve, and a caller reading one should say so.

    `scan_history` also fails SILENTLY on some runs: it returns the full row
    count with the requested key simply absent from every row, which reads as
    success and yields an empty series downstream. Measured on
    fast-single-e0.001-...-19-30-36: 87,761 rows, no `frames` key, while
    `history()` returned the series. So the result is judged by whether the
    metric came back, not by the absence of an exception.
    """
    try:
        rows = list(run.scan_history(keys=[step_key, metric]))
        if any(row.get(metric) is not None for row in rows):
            return rows
    except Exception:  # noqa: BLE001 - any backend refusal, not just the known one
        pass

    frame = run.history(keys=[metric], pandas=True)
    if frame is None or len(frame) == 0:
        return []
    if step_key not in frame.columns:
        frame = frame.reset_index().rename(columns={"index": step_key})
    return [row for row in frame.to_dict("records") if isinstance(row.get(metric), dict)]


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


