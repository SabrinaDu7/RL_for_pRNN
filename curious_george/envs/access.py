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
