"""Standalone functions for collecting evaluation trajectories from RL agents.

Provides a modular alternative to the trajectory collection logic in
ObjectMemoryTask.getTestTrial, decoupled from the training loop.

COORDINATE SYSTEM
All code works with (0, 0) as the top left, first walkwable position of the env.
Minigrid uses (1, 1) as the top left corner, so we ONLY change coordinates when interfacing with minigrid.

"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypedDict, Union

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Integer
from tqdm import tqdm

from prnn.utils import PredictiveNet, RandomActionAgent
from prnn.utils.Shell import FaramaMinigridShell
from curious_george import ActorCriticAgent

from tasks.omt.metrics import State

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

# Type alias for a label function: given a list of State dicts (one per trajectory),
# returns an integer label tensor of shape (B, T).
LabelFn = Callable[[list[State], int], torch.Tensor]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class EvalTrajectories(TypedDict):
    obs: Float[torch.Tensor, "B T+1 X"] # T steps + last observation
    obs_pred: Float[torch.Tensor, "B T X"]
    obs_next: Float[torch.Tensor, "B T X"]
    act: Float[torch.Tensor, "B T A"]
    states: list[State]
    hidden_states: Float[torch.Tensor, "B T hidden"] | None
    labels: torch.Tensor | None  # (B, T) integer labels, or None
    renders: np.ndarray | None
    config: EvalTrajectoryConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvalTrajectoryConfig:
    """Configuration for collecting evaluation trajectories."""

    save_path: Path
    timesteps: int
    include_render: bool = False
    include_hidden_states: bool = False
    ckpt_step: int | None = None # ckpt_step
    goal_loc: list[int] | None = None # [x, y]


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


@contextmanager
def eval_mode(pN: PredictiveNet, agent: Union[ActorCriticAgent, RandomActionAgent]):
    """Temporarily set pRNN and agent to evaluation mode.

    Restores original training state and argmax setting on exit.
    """
    original_pN_training = pN.pRNN.training
    original_argmax = None
    original_acmodel_training = None

    pN.pRNN.eval()
    if hasattr(agent, "acmodel"):
        assert isinstance(agent, ActorCriticAgent)
        original_acmodel_training = agent.acmodel.training
        original_argmax = agent.argmax
        agent.acmodel.eval()
        agent.argmax = True

    try:
        yield
    finally:
        if original_pN_training:
            pN.pRNN.train()
        if hasattr(agent, "acmodel"):
            if original_acmodel_training:
                assert isinstance(agent, ActorCriticAgent)
                agent.acmodel.train()
            agent.argmax = original_argmax  # type: ignore[assignment]


@contextmanager
def on_device(pN: PredictiveNet, device: torch.device):
    """Temporarily move pRNN to *device*, restoring the original device on exit."""
    original_device = next(pN.pRNN.parameters()).device
    pN.pRNN.to(device)
    try:
        yield
    finally:
        pN.pRNN.to(original_device)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
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


def collect_eval_trajectories(
    pN: PredictiveNet,
    agent: Union[ActorCriticAgent, RandomActionAgent],
    env: FaramaMinigridShell,
    config: EvalTrajectoryConfig,
    label_fn: LabelFn | None = None,
) -> EvalTrajectories:
    """Collect evaluation trajectories from an agent in a MiniGrid environment.

    Runs one trajectory per (walkable_position, head_direction) pair, giving
    ``num_walkable * 4`` trajectories of ``config.timesteps`` steps each.

    The pRNN is moved to CPU during collection (required for numpy operations
    inside the agent) and restored to its original device afterward.

    Args:
        pN: Predictive network wrapper.
        agent: Agent to collect trajectories from.
        env: MiniGrid environment wrapped in FaramaMinigridShell.
        config: Collection parameters.
        label_fn: Optional hook ``(states, T) -> Tensor[B, T]`` that assigns
            an integer label to each (trajectory, timestep) pair.  For example,
            ``novel_obj_in_view`` sets 1 when the object is visible and 0
            otherwise.  When *None*, ``labels`` in the result is *None*.

    Returns:
        EvalTrajectories dict with collected data.
    """
    T = config.timesteps
    _, view_size, C = env.obs_shape
    X = view_size * view_size * C
    A = env.getActSize()

    # Build starting conditions: every (walkable_position, head_direction) pair
    walkable_mask = get_walkable_mask(env)
    start_positions = get_walkable_minigrid_positions(walkable_mask)
    num_walkable = len(start_positions)
    # One trajectory per (position, direction) combination
    start_conditions = [
        (tuple(start_positions[i].tolist()), d)
        for i in range(num_walkable)
        for d in range(4)
    ]
    B = len(start_conditions)  # num_walkable * 4

    # Pre-allocate storage
    all_obs = torch.zeros((B, T + 1, X), dtype=torch.float32)
    all_act = torch.zeros((B, T, A), dtype=torch.float32)
    all_states: list[State] = []

    all_renders: np.ndarray | None = None
    if config.include_render:
        render_size = env.render_size
        all_renders = np.zeros(
            (B, T + 1, render_size, render_size, C), dtype=np.uint8
        )

    with eval_mode(pN, agent), on_device(pN, torch.device("cpu")):
        # Phase 1: Sequential env collection
        for n in tqdm(range(B), desc="Collecting Trajectories"):
            pos, direction = start_conditions[n]
            env.env.unwrapped.agent_start_pos = pos
            env.env.unwrapped.agent_start_dir = direction

            obs, act, state, render = pN.collectObservationSequence(
                env=env,
                agent=agent,
                tsteps=T,
                includeRender=config.include_render,
            )

            all_obs[n] = obs.squeeze(0)
            all_act[n] = act.squeeze(0)
            all_states.append(state)

            if config.include_render and all_renders is not None and render:
                all_renders[n] = np.stack(render, axis=0)

        # Phase 2: Batched pRNN prediction.
        # predict(batched=True) takes 3-D (B, L, X) and permutes internally to
        # (1, L, X, B); passing 4-D raises in clip_mask. Verified against the
        # serial path on 2026-07-30: with the injected noise disabled the two
        # agree to 6e-6.
        with torch.no_grad():
            obs_pred, obs_next, h_t = pN.predict(
                all_obs,  # (B, T + 1, obs_size)
                all_act,  # (B, T, act_size)
                batched=True,
                randInit=True,         # handles init_state internally
            )

            # Outputs are (1, T', features, B) → (B, T', features)
            all_obs_pred = obs_pred[0].permute(2, 0, 1)
            all_obs_next = obs_next[0].permute(2, 0, 1)

            all_hidden: torch.Tensor | None = None
            if config.include_hidden_states:
                all_hidden = h_t[0].permute(2, 0, 1)

    all_labels: torch.Tensor | None = None
    if label_fn is not None:
        all_labels = label_fn(all_states, T)  # (B, T)

    result: EvalTrajectories = {
        "obs": all_obs,
        "obs_pred": all_obs_pred,
        "obs_next": all_obs_next,
        "act": all_act,
        "states": all_states, # Contains agent_pos and agent_dir
        "hidden_states": all_hidden,
        "labels": all_labels,
        "renders": all_renders,
        "config" : config
    }

    if config.save_path is not None:
        save_eval_trajectories(result, config.save_path)

    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_eval_trajectories(data: EvalTrajectories, path: Path) -> Path:
    """Save evaluation trajectories to disk.

    Writes the full tensor data as a ``.pt`` file and a summary parquet with
    agent positions and directions for quick inspection.

    Args:
        data: The collected trajectories.
        path: Directory to save into (created if it doesn't exist).

    Returns:
        The directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Full tensor data
    save_data = {k: v for k, v in data.items() if k != "renders" and v is not None}
    torch.save(save_data, path / "trajectories.pt")

    # Summary parquet (tabular state data)
    states: list[State] = data["states"]
    records = []
    for b, state in enumerate(states):
        T_plus_1 = state["agent_pos"].shape[0]
        for t in range(T_plus_1):
            records.append(
                {
                    "traj_id": b,
                    "timestep": t,
                    "pos_x": float(state["agent_pos"][t, 0]),
                    "pos_y": float(state["agent_pos"][t, 1]),
                    "direction": int(state["agent_dir"][t]),
                }
            )
    pd.DataFrame(records).to_parquet(path / "summary.parquet", index=False)

    return path


def load_eval_trajectories(path: Path) -> dict:
    """Load trajectories saved by save_eval_trajectories.

    Accepts the directory save_eval_trajectories wrote into (harmonized with
    its interface) or a direct path to the ``.pt`` file.

    Args:
        path: Save directory, or full path to the .pt file inside it.

    Returns:
        Dict with the saved tensor data.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "trajectories.pt"
    return torch.load(path, weights_only=False)

