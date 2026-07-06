import random
import numpy as np
import torch
import collections


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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