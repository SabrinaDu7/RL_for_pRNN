"""On-policy behavioural probe: is the agent actually going to the novel object?

Complements the rate-map analysis. Rolls the TRAINED actor-critic policy in the
object-PRESENT environment at each checkpoint and asks, per checkpoint, whether
the object's location is visited / viewed more than the room's other cells.

The statistic is a within-run null: compute the behavioural measure for EVERY
walkable cell from the same trajectories, then report the percentile rank of the
actual object location. That controls for the L-room's structurally unequal
accessibility without pooling across seeds - pooling can manufacture apparent
clustering if object locations happen to sit in high-traffic regions.

Two measures, because they answer different questions (see the module docstring
of trace_maps for the occupancy caveat):
  occupancy  - timesteps spent within `radius` of the cell
  in-view    - timesteps with the cell inside the agent's 7x7 egocentric view,
               which is what actually drives prediction error, since the object
               is a non-blocking floor tile visible from most of the room.
"""

from __future__ import annotations

import numpy as np
import torch
from jaxtyping import Float, Int

from prnn.utils import PredictiveNet
from prnn.utils.Shell import FaramaMinigridShell

from curious_george import ActorCriticAgent
from curious_george.envs.access import base_env
from curious_george.world_model.device import eval_mode, on_device
from tasks.omt.metrics import get_view_coords_batch


def collect_policy_rollouts(
    *,
    pN: PredictiveNet,
    agent: ActorCriticAgent,
    env: FaramaMinigridShell,
    n_trajs: int = 32,
    n_steps: int = 256,
    seed: int = 0,
) -> tuple[Int[np.ndarray, "B T+1 2"], Int[np.ndarray, "B T+1"]]:
    """Positions and head directions from `n_trajs` on-policy rollouts.

    Start positions are randomised per trajectory (matching the OMT eval's
    `tasks.testing.start_random=True`) from a fixed seed, so every checkpoint
    faces the same start distribution.
    """
    rng = np.random.default_rng(seed)
    pos_rows: list[np.ndarray] = []
    dir_rows: list[np.ndarray] = []

    with eval_mode([pN], agent=agent), on_device([pN, agent.acmodel], "cpu"):
        for _ in range(n_trajs):
            base_env(env).agent_start_pos = None  # place_agent() picks at random
            base_env(env).agent_start_dir = int(rng.integers(0, 4))
            _, _, state, _ = pN.collectObservationSequence(
                env=env, agent=agent, tsteps=n_steps, discretize=True
            )
            pos_rows.append(np.asarray(state["agent_pos"]))
            dir_rows.append(np.asarray(state["agent_dir"]))

    return (
        np.stack(pos_rows).astype(np.int32),
        np.stack(dir_rows).astype(np.int32),
    )


def occupancy_near(
    *,
    pos: Int[np.ndarray, "B T 2"],
    cell: tuple[int, int],
    radius: float = 2.0,
) -> float:
    """Fraction of timesteps spent within `radius` of `cell`."""
    d2 = (pos[..., 0] - cell[0]) ** 2 + (pos[..., 1] - cell[1]) ** 2
    return float((d2 <= radius**2).mean())


def in_view_fraction(
    *,
    pos: Int[np.ndarray, "B T 2"],
    hd: Int[np.ndarray, "B T"],
    cell: tuple[int, int],
    view_size: int = 7,
) -> float:
    """Fraction of timesteps with `cell` inside the agent's egocentric view."""
    flat_pos = pos.reshape(-1, 2)
    flat_hd = hd.reshape(-1)
    vx, vy = get_view_coords_batch(cell[0], cell[1], flat_pos, flat_hd, view_size)
    inside = (vx >= 0) & (vx < view_size) & (vy >= 0) & (vy < view_size)
    return float(inside.mean())


def within_run_percentile(
    *,
    pos: Int[np.ndarray, "B T 2"],
    hd: Int[np.ndarray, "B T"],
    cells: list[tuple[int, int]],
    target: tuple[int, int],
    radius: float = 2.0,
) -> dict:
    """Percentile rank of `target` among `cells` for both behavioural measures.

    Returns the two percentiles plus the raw values, so a high percentile that
    rests on a tiny absolute value is visible rather than hidden.
    """
    occ = np.array([occupancy_near(pos=pos, cell=c, radius=radius) for c in cells])
    view = np.array([in_view_fraction(pos=pos, hd=hd, cell=c) for c in cells])
    i = cells.index(target)
    return {
        "occ_pct": 100.0 * float(np.mean(occ < occ[i])),
        "view_pct": 100.0 * float(np.mean(view < view[i])),
        "occ_value": float(occ[i]),
        "view_value": float(view[i]),
        "occ_median": float(np.median(occ)),
        "view_median": float(np.median(view)),
    }
