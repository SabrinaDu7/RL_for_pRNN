"""Accessors for reaching into the wrapped MiniGrid environment.

Single home for wrapper reach-ins so the rest of the code never spells
`env.env.env...`.
"""

import torch
from jaxtyping import Integer
from minigrid.envs.Lroom import LEnv
from minigrid.wrappers import RGBImgPartialObsWrapper_HD
from prnn.utils.Shell import FaramaMinigridShell


def base_env(shell: FaramaMinigridShell) -> LEnv:
    """The raw MiniGrid env under the FaramaMinigridShell and all gym wrappers.

    Replaces hand-spelled peels like `env.env.env.env.env` and
    `env.env.unwrapped` (the shell itself is not a gym wrapper, so
    `.unwrapped` must be reached through its `.env` attribute).
    """
    env_wrapper: RGBImgPartialObsWrapper_HD = shell.env
    return env_wrapper.unwrapped


def grid_shape(shell: FaramaMinigridShell) -> tuple[int, int]:
    """(width, height) of the underlying grid."""
    grid = base_env(shell).grid
    return grid.width, grid.height


# --- walkable geometry -------------------------------------------------------
# Promoted out of the OMT analysis 2026-08-25: ten call sites across the probe,
# the task code and the figure scripts, so this is env geometry the library
# owns, not analysis.

def get_walkable_mask(env: FaramaMinigridShell) -> Integer[torch.Tensor, "W H"]:
    """Return a boolean mask of walkable positions, shape ``[W, H]``.

    A cell is walkable if it is empty (``None``) or the object there
    supports overlap (e.g. ``Floor``, ``Goal``).  Wall cells are not walkable.

    **Indexing convention**: ``mask[x, y]`` where *x* is the horizontal
    position and *y* is the vertical position (0-indexed, walls excluded).
    This matches the ``(x, y)`` position convention used throughout the
    codebase and in Minigrid (``grid.get(x, y)``).

    Note: dim-0 = width (horizontal), dim-1 = height (vertical).  This is
    transposed relative to standard row-major matrix layout.  To visualise
    correctly with ``imshow``, use ``imshow(mask.T)``.
    """
    grid = env.env.unwrapped.grid
    W = env.width - 2   # horizontal extent (walls excluded)
    H = env.height - 2  # vertical extent (walls excluded)
    mask = torch.zeros(W, H, dtype=torch.bool)
    for x in range(W):
        for y in range(H):
            cell = grid.get(x + 1, y + 1)  # +1 to skip wall row/col
            if cell is None or cell.can_overlap():
                mask[x, y] = True
    return mask


def get_walkable_minigrid_positions(walkable_mask: Integer[torch.Tensor, "W H"]) -> Integer[torch.Tensor, "N 2"]:
    """Return Minigrid positions for walkable cells, shape ``[N, 2]``.

    Each row is an ``(x, y)`` position in Minigrid coordinates (1-indexed),
    where *x* = horizontal, *y* = vertical.

    Args:
        walkable_mask: Boolean mask of shape ``[W, H]`` (see
            :func:`get_walkable_mask` for indexing convention).

    Returns:
        Int tensor of shape ``[N, 2]`` with ``(x, y)`` Minigrid positions.
    """
    # nonzero returns (dim0_idx, dim1_idx) = (x, y) because dim0 = W
    mask_positions = walkable_mask.nonzero(as_tuple=False)  # (N, 2)
    return mask_positions + 1  # 0-indexed → Minigrid 1-indexed
