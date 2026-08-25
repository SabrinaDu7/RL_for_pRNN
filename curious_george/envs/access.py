"""Accessors for reaching into the wrapped MiniGrid environment.

Single home for wrapper reach-ins so the rest of the code never spells
`env.env.env...` (see refactor plan Phase 3 for the full accessor set).
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


def subroom_size(shell: FaramaMinigridShell) -> int | None:
    """FourRooms subroom size, or None for envs without subrooms."""
    return getattr(base_env(shell), "subroom_size", None)


def get_new_obj_pos(shell: FaramaMinigridShell) -> list[int] | None:
    """Goal position of goal-bearing envs (e.g. LRoomGoal)."""
    return base_env(shell).new_obj_pos


def get_subroom_id(
    agent_pos: Integer[torch.Tensor, "T 2"], subroom_size: int
) -> Integer[torch.Tensor, "B"]:
    """Helper method to get the subroom ID based on agent position and subroom size."""

    col = (agent_pos[:, 0] > subroom_size).long()
    row = (agent_pos[:, 1] > subroom_size).long()

    return row * 2 + col + 1


# ---------------------------------------------------------------------------
# Visualization helpers (state / observation / prediction / hidden-state images)
# ---------------------------------------------------------------------------

ACTION_NAMES = ("left", "right", "forward", "stay")


def render_env(shell) -> "np.ndarray":
    """Current RGB frame of the full environment (agent pos/dir included)."""
    return shell.render(mode=None)


def obs_image(obs: dict, upscale: int = 24):
    """Raw observation dict -> upscaled RGB array in [0, 1] for imshow."""
    import numpy as np

    img = np.asarray(obs["image"], dtype=float)
    if img.max() > 1.0:
        img = img / 255.0
    return np.kron(img, np.ones((upscale, upscale, 1)))


def pred_image(shell, row: torch.Tensor, upscale: int = 24):
    """One prediction/target row (X,) -> upscaled RGB array in [0, 1].

    Uses the shell's own pred2np decoding (expects a (phase, T, X) tensor).
    """
    import numpy as np

    img = shell.pred2np(row.detach().cpu()[None, None, :])[0]
    img = np.clip(np.asarray(img, dtype=float), 0.0, 1.0)
    return np.kron(img, np.ones((upscale, upscale, 1)))


def hidden_image(h: torch.Tensor, width: int = 25):
    """Hidden-state / SR vector -> 2D heatmap array (pad to a width-column grid)."""
    import numpy as np

    v = h.detach().cpu().flatten().numpy()
    pad = (-len(v)) % width
    v = np.pad(v, (0, pad), constant_values=np.nan)
    return v.reshape(-1, width)


# --- walkable geometry -------------------------------------------------------
# Promoted out of throwaway/ported/analysis_OMT.py 2026-08-25: ten call sites across the
# probe, the task code and the figure scripts, so this is env geometry the
# library owns, not analysis.

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
