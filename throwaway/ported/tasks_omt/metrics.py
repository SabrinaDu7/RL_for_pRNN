"""OMT metrics: view-coordinate projection and the object-learning quantifier.

Pure functions (no task state). The headline metric asks: does the trained
pRNN predict extra green at the novel-object location (when it is in view)
relative to the untrained control net and relative to control locations?
"""

from typing import Optional, TypedDict

import numpy as np
import torch
from jaxtyping import Float

from curious_george import get_dist_travelled


class State(TypedDict):
    agent_pos: Float[np.ndarray, "T+1 2"]
    agent_dir: Float[np.ndarray, "T+1"]
    SRs: Optional[Float[torch.Tensor, "(T+1) * hidden_size"]] # Not present with RandomActionAgent


# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]


def get_view_coords(i, j, pos, HD, agent_view_size=7):
    ax, ay = pos
    dx, dy = DIR_TO_VEC[HD]
    rx, ry = -dy, dx

    # Compute the absolute coordinates of the top-left view corner
    sz = agent_view_size
    hs = agent_view_size // 2
    tx = ax + (dx * (sz - 1)) - (rx * hs)
    ty = ay + (dy * (sz - 1)) - (ry * hs)

    lx = i - tx
    ly = j - ty

    # Project the coordinates of the object relative to the top-left
    # corner onto the agent's own coordinate system
    vx = rx * lx + ry * ly
    vy = -(dx * lx + dy * ly)

    return int(vx), int(vy)


def get_obs_at_loc(obs, goal_loc, pos, HD):
    i, j = goal_loc

    locobs = []
    viewtimes = []
    viewcoords = []
    for tt in range(obs.shape[0]):
        # Get egocentric coordinates of the goal/control loc
        # Check that these coordinates are in the agent's 7x7 view
        vx, vy = get_view_coords(i, j, pos[tt, :], HD[tt])
        if (vx >= 0) & (vx < 7) & (vy >= 0) & (vy < 7):
            locobs.append(obs[tt, vy, vx, :])
            viewtimes.append(tt)
            viewcoords.append([vx, vy])

    if locobs == []:
        # No views of the location were found
        return None, None, None

    locobs = np.stack(locobs, axis=0)
    viewtimes = np.stack(viewtimes, axis=0)
    viewcoords = np.stack(viewcoords, axis=0)
    return locobs, viewtimes, viewcoords


# Pre-built array for vectorized direction lookups
_DIR_TO_VEC_ARRAY = np.array(DIR_TO_VEC, dtype=np.int32)  # (4, 2)


def get_view_coords_batch(
    i: int, j: int, pos: np.ndarray, HD: np.ndarray, agent_view_size: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized version of get_view_coords over T timesteps.

    Args:
        i, j: World-coordinate location to project.
        pos:  (T, 2) agent positions.
        HD:   (T,)  agent head directions in [0, 3].

    Returns:
        vx, vy: (T,) int arrays of egocentric view coordinates.
    """
    dxdy = _DIR_TO_VEC_ARRAY[HD]          # (T, 2)
    dx, dy = dxdy[:, 0], dxdy[:, 1]
    rx, ry = -dy, dx

    sz, hs = agent_view_size, agent_view_size // 2
    ax, ay = pos[:, 0], pos[:, 1]

    tx = ax + dx * (sz - 1) - rx * hs
    ty = ay + dy * (sz - 1) - ry * hs

    lx = i - tx
    ly = j - ty

    vx = (rx * lx + ry * ly).astype(int)
    vy = (-(dx * lx + dy * ly)).astype(int)
    return vx, vy


def get_obs_at_loc_fast(
    obs: np.ndarray, goal_loc: list[int], pos: np.ndarray, HD: np.ndarray, return_obs: bool = True
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Vectorized version of get_obs_at_loc.

    Computes view coordinates for all timesteps at once instead of looping.

    Args:
        obs:      (T, H, W, C) predicted observations.
        goal_loc: [i, j] world coordinate of location of interest.
        pos:      (T_pos, 2) agent positions (only first T rows are used).
        HD:       (T_pos,)  agent head directions (only first T rows are used).

    Returns:
        locobs, viewtimes, viewcoords — same semantics as get_obs_at_loc.
    """
    i, j = goal_loc
    T = obs.shape[0]

    vx, vy = get_view_coords_batch(i, j, pos[:T], HD[:T])

    in_view = (vx >= 0) & (vx < 7) & (vy >= 0) & (vy < 7)
    viewtimes = np.where(in_view)[0]

    if viewtimes.size == 0:
        return None, None, None

    vx_v, vy_v = vx[viewtimes], vy[viewtimes]
    locobs = obs[viewtimes, vy_v, vx_v, :] if return_obs else None
    viewcoords = np.stack([vx_v, vy_v], axis=1)
    return locobs, viewtimes, viewcoords


def quantify_object_learning(
    test_trial: dict,
    *,
    env_shell,
    new_obj_pos: list[int],
    ctrl_locs: list[list[int]],
    whichPhase: int,
    traj_count: int,
) -> dict | None:
    """Predicted-pixel modulation at the (absent) novel-object location.

    test_trial: dict from the eval rollouts (obs_pred / obs_pred_control /
    agent_pos / agent_dir); env_shell decodes prediction rows to images.
    """
    start_pos = test_trial["agent_pos"][:, 0, :]
    end_pos = test_trial["agent_pos"][:, -1, :]
    avg_dist = torch.mean(get_dist_travelled(
        start_locs=torch.tensor(start_pos, dtype=torch.float32),
        end_locs=torch.tensor(end_pos, dtype=torch.float32)
    )) # Average distance travelled across all test trajectories

    pos = test_trial["agent_pos"][:, whichPhase:, :]
    HD = test_trial["agent_dir"][:, whichPhase:]
    obs_pred = test_trial["obs_pred"][:, whichPhase:, :]
    obs_pred_notrain = test_trial["obs_pred_control"][:, whichPhase:, :]

    B, T, X = obs_pred.shape
    obs_pred = obs_pred.reshape(B * T, X).unsqueeze(dim=0)
    obs_pred_notrain = obs_pred_notrain.reshape(B * T, X).unsqueeze(dim=0)
    pos = pos.reshape(B * (T + 1), 2)
    HD = HD.reshape(B * (T + 1))

    # Get predicted pixel values at the object/control location for trained and control pRNNs
    obs_np = env_shell.pred2np(obs_pred)
    obs_notrain_np = env_shell.pred2np(obs_pred_notrain)

    locobs, inviewtimes, viewcoords = get_obs_at_loc_fast(obs_np, new_obj_pos, pos, HD)
    locobs_notrain, inviewtimes, viewcoords = get_obs_at_loc_fast(obs_notrain_np, new_obj_pos, pos, HD)

    if locobs is None:
        print("No views of the goal or control location were found during the test trial.")
        return None

    objectloc_deltaobs = locobs - locobs_notrain
    goalmodulation = np.mean(objectloc_deltaobs[:, 1])
    ctlmodulation_diffcolor = np.mean(
        np.concatenate((objectloc_deltaobs[:, 0], objectloc_deltaobs[:, 2]))
    )

    ctl_mods = []
    for control_location in ctrl_locs:
        conobs, _, _ = get_obs_at_loc_fast(obs_np, control_location, pos, HD)
        conobs_notrain, _, _ = get_obs_at_loc_fast(obs_notrain_np, control_location, pos, HD)

        if conobs is not None:
            controlloc_deltaobs = conobs - conobs_notrain
            ctlmodulation_diffloc = np.mean(controlloc_deltaobs[:, 1]) # TODO: Do we just want to control for change in green at cntrl locs?
            ctl_mods.append(ctlmodulation_diffloc)

    ctlmodulation_diffloc = np.mean(ctl_mods)

    return {
        "inviewtimes": inviewtimes,
        "viewcoords": viewcoords,
        "objectloc_obs": locobs,
        "controlloc_obs": conobs,
        "objectloc_obs_controlNet": locobs_notrain,
        "controlloc_obs_controlNet": conobs_notrain,
        "objectloc_deltaobs": objectloc_deltaobs,
        "controlloc_deltaobs": controlloc_deltaobs,
        "goalmodulation": goalmodulation,
        "ctlmodulation_diffcolor": ctlmodulation_diffcolor,
        "ctlmodulation_diffloc": ctlmodulation_diffloc,
        "traj_count": traj_count,
        "avg_dist": avg_dist.item(),
    }
