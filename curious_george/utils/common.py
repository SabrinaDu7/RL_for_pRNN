import os
import random
import numpy as np
import torch
import collections


# CG_DEVICE=cpu|cuda overrides the default. NOTE: at B=1 / hidden 500 the CPU
# is ~2x faster than CUDA end-to-end (throwaway/ported/docs_legacy/perf_baseline.md). DEVICE binds at
# import time, so control is via env var, not the hydra config (the unused
# `hardware:` config block was removed 2026-07-09 with Sabrina's approval).
#
# SINGLE SOURCE OF TRUTH. Code that needs to know where work runs must call
# `get_device()` / `on_cuda()` - never `torch.cuda.is_available()`, which
# answers a DIFFERENT question. On a GPU box running CG_DEVICE=cpu,
# is_available() is True while on_cuda() is False; branching on the former
# is what mislabelled perf records and made CPU runs pay spurious CUDA syncs.
def _resolve_device() -> torch.device:
    """Resolve the device this process runs on from CG_DEVICE (default: cuda
    if present). Requesting cuda when it is unavailable is a hard error - a
    silent CPU fallback would make a 'GPU run' secretly a CPU run."""
    requested = os.environ.get("CG_DEVICE")
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CG_DEVICE={requested!r} requested but torch.cuda.is_available() "
            "is False - refusing to fall back to CPU silently."
        )
    return device


DEVICE = _resolve_device()


def get_device() -> torch.device:
    """The device this process runs on. **Prefer this over importing DEVICE.**

    `from ...utils.common import DEVICE` copies the object into the importing
    module at import time, so it cannot observe a later reassignment of
    `common.DEVICE`; worse, using it as a default argument
    (`def f(device=DEVICE)`) freezes the value at def time. `get_device()`
    reads the module global on every call, so runtime device switching works.
    `DEVICE` is retained only as the import-time default and for backwards
    compatibility.
    """
    return DEVICE


def on_cuda() -> bool:
    """True iff work actually runs on CUDA. NOT the same as
    `torch.cuda.is_available()` - see the module note above.

    Reads the module global (like `get_device()`), so it tracks a runtime
    reassignment of `common.DEVICE`.
    """
    return DEVICE.type == "cuda"


def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # seeds every visible CUDA device regardless of DEVICE: this is about
    # reproducibility, not about where work runs, and tests that explicitly
    # use CUDA under CG_DEVICE=cpu still need their CUDA RNG seeded.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize(array, signs=False, abs=False):
    d = collections.OrderedDict()
    d["mean"] = np.nanmean(array)
    d["std"] = np.nanstd(array)
    d["min"] = np.nanmin(array)
    d["max"] = np.nanmax(array)
    if signs:
        # valid_idxs = ~np.isnan(array)
        # array = array[valid_idxs]
        array = np.array(array)
        d["pos"] = np.sum(np.sign(array)[array > 0])
        d["neg"] = np.sum(np.abs(np.sign(array)[array < 0]))
        d["ratio"] = d["pos"] / d["neg"] if d["neg"] != 0 else np.nan
    if abs:
        d["abs_mean"] = np.nanmean(np.abs(array))
    return d

def mean_by_action(values: np.ndarray, actions: np.ndarray) -> dict:
    """Compute mean values separated by action type.

    Args:
        values: Array of values (e.g., advantages, rewards)
        actions: Array of action indices (0=turn_left, 1=turn_right, 2=forward, 3=stay_put)

    Returns:
        Dict with keys "turn_left", "turn_right", "forward", "stay_put" and mean values
    """
    action_names = ["turn_left", "turn_right", "forward", "stay_put"]
    result = {}
    for i, name in enumerate(action_names):
        filtered = values[actions == i]
        result[name] = float(np.mean(filtered)) if len(filtered) > 0 else 0.0
    return result


def grid_to_pixel_coords(grid_coords: np.ndarray, tile_size: int = 32) -> np.ndarray:
    """
    Convert FaramaMinigrid grid coordinates to pixel coordinates for plotting on rendered image.

    In FaramaMinigrid environments, grid coordinates are integers (e.g., 0-15 for a 16x16 grid).
    When rendered, each grid cell becomes a tile_size x tile_size pixel block (default 32x32).
    This function converts grid coordinates to pixel coordinates centered in each tile.

    Args:
        grid_coords: Array of shape (n, 2) or (2,) with (x, y) grid coordinates
        tile_size: Size of each grid cell in pixels (default 32 for standard rendering)

    Returns:
        Array of shape (n, 2) or (2,) with (x, y) pixel coordinates centered in each tile

    Examples:
        >>> grid_coords = np.array([[0, 0], [1, 1], [15, 15]])
        >>> pixel_coords = grid_to_pixel_coords(grid_coords, tile_size=32)
        >>> pixel_coords
        array([[ 16.,  16.],
               [ 48.,  48.],
               [496., 496.]])
    """
    # Convert to pixel coordinates and center within tile
    # Adding 0.5 centers the point in the middle of each tile
    pixel_coords = (grid_coords + 0.5) * tile_size
    return pixel_coords