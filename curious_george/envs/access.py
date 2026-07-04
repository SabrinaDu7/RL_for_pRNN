"""Accessors for reaching into the wrapped MiniGrid environment.

Single home for wrapper reach-ins so the rest of the code never spells
`env.env.env...` (see refactor plan Phase 3 for the full accessor set).
"""

import torch
from jaxtyping import Integer


def base_env(shell):
    """The raw MiniGrid env under the FaramaMinigridShell and all gym wrappers.

    Replaces hand-spelled peels like `env.env.env.env.env` and
    `env.env.unwrapped` (the shell itself is not a gym wrapper, so
    `.unwrapped` must be reached through its `.env` attribute).
    """
    return shell.env.unwrapped


def grid_shape(shell) -> tuple[int, int]:
    """(width, height) of the underlying grid."""
    grid = base_env(shell).grid
    return grid.width, grid.height


def subroom_size(shell) -> int | None:
    """FourRooms subroom size, or None for envs without subrooms."""
    return getattr(base_env(shell), "subroom_size", None)


def get_goal_loc(shell) -> list[int]:
    """Goal position of goal-bearing envs (e.g. LRoomGoal)."""
    return base_env(shell).goal_pos


def get_subroom_id(agent_pos: Integer[torch.Tensor, "T 2"], subroom_size: int) -> Integer[torch.Tensor, "B"]:
    """Helper method to get the subroom ID based on agent position and subroom size."""

    col = (agent_pos[:, 0] > subroom_size).long()
    row = (agent_pos[:, 1] > subroom_size).long()

    return row * 2 + col + 1
