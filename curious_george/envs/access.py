"""Accessors for reaching into the wrapped MiniGrid environment.

Single home for wrapper reach-ins so the rest of the code never spells
`env.env.env...` (see refactor plan Phase 3 for the full accessor set).
"""

import torch
from jaxtyping import Integer


def get_subroom_id(agent_pos: Integer[torch.Tensor, "T 2"], subroom_size: int) -> Integer[torch.Tensor, "B"]:
    """Helper method to get the subroom ID based on agent position and subroom size."""

    col = (agent_pos[:, 0] > subroom_size).long()
    row = (agent_pos[:, 1] > subroom_size).long()

    return row * 2 + col + 1
