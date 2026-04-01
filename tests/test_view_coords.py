"""Tests for vectorized view-coordinate helpers.

Verifies that get_view_coords_batch and get_obs_at_loc_fast produce
identical outputs to the original scalar implementations.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from tasks.ObjectMemoryTask.define_task import (
    get_view_coords,
    get_view_coords_batch,
    get_obs_at_loc,
    get_obs_at_loc_fast,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _random_pos_hd(T: int, grid_size: int = 9):
    pos = RNG.integers(1, grid_size, size=(T, 2)).astype(np.float32)
    hd = RNG.integers(0, 4, size=(T,)).astype(np.int32)
    return pos, hd


def _random_obs(T: int, H: int = 7, W: int = 7, C: int = 3):
    return RNG.random((T, H, W, C)).astype(np.float32)


# ---------------------------------------------------------------------------
# get_view_coords_batch
# ---------------------------------------------------------------------------


class TestGetViewCoordsBatch:
    """get_view_coords_batch must agree with get_view_coords on every timestep."""

    @pytest.mark.parametrize("T", [1, 10, 100])
    @pytest.mark.parametrize("goal_loc", [[3, 3], [1, 1], [8, 5]])
    def test_matches_scalar(self, T, goal_loc):
        pos, hd = _random_pos_hd(T)
        i, j = goal_loc

        vx_batch, vy_batch = get_view_coords_batch(i, j, pos, hd)

        for t in range(T):
            vx_s, vy_s = get_view_coords(i, j, pos[t], hd[t])
            assert vx_batch[t] == vx_s, f"vx mismatch at t={t}"
            assert vy_batch[t] == vy_s, f"vy mismatch at t={t}"

    def test_output_shapes(self):
        T = 20
        pos, hd = _random_pos_hd(T)
        vx, vy = get_view_coords_batch(3, 3, pos, hd)
        assert vx.shape == (T,)
        assert vy.shape == (T,)

    def test_all_directions(self):
        """One timestep per direction; validate exhaustively."""
        pos = np.array([[5, 5]] * 4, dtype=np.float32)
        hd = np.array([0, 1, 2, 3], dtype=np.int32)
        vx, vy = get_view_coords_batch(5, 5, pos, hd)
        for t in range(4):
            exp_vx, exp_vy = get_view_coords(5, 5, pos[t], hd[t])
            assert vx[t] == exp_vx
            assert vy[t] == exp_vy

    def test_single_timestep(self):
        pos = np.array([[4, 6]], dtype=np.float32)
        hd = np.array([2], dtype=np.int32)
        vx, vy = get_view_coords_batch(3, 3, pos, hd)
        exp_vx, exp_vy = get_view_coords(3, 3, pos[0], hd[0])
        assert vx[0] == exp_vx
        assert vy[0] == exp_vy


# ---------------------------------------------------------------------------
# get_obs_at_loc_fast
# ---------------------------------------------------------------------------


class TestGetObsAtLocFast:
    """get_obs_at_loc_fast must agree with get_obs_at_loc on all outputs."""

    def _compare(self, obs, goal_loc, pos, hd):
        slow = get_obs_at_loc(obs, goal_loc, pos, hd)
        fast = get_obs_at_loc_fast(obs, goal_loc, pos, hd)

        if slow[0] is None:
            assert fast[0] is None
            assert fast[1] is None
            assert fast[2] is None
        else:
            np.testing.assert_array_equal(fast[0], slow[0], err_msg="locobs mismatch")
            np.testing.assert_array_equal(fast[1], slow[1], err_msg="viewtimes mismatch")
            np.testing.assert_array_equal(fast[2], slow[2], err_msg="viewcoords mismatch")

    @pytest.mark.parametrize("T", [10, 50, 200])
    def test_matches_slow(self, T):
        pos, hd = _random_pos_hd(T)
        obs = _random_obs(T)
        self._compare(obs, [3, 3], pos, hd)

    def test_returns_none_when_no_views(self):
        """Goal at a position never in view should return (None, None, None)."""
        T = 50
        # Agent always at (1, 1) facing right (dir=0); goal at (0, 0) is behind
        pos = np.ones((T, 2), dtype=np.float32)
        hd = np.zeros(T, dtype=np.int32)
        obs = _random_obs(T)
        slow = get_obs_at_loc(obs, [0, 0], pos, hd)
        fast = get_obs_at_loc_fast(obs, [0, 0], pos, hd)
        assert slow[0] is None
        assert fast[0] is None

    def test_locobs_values_correct(self):
        """Pixel values at in-view timesteps must match obs array exactly."""
        T = 30
        pos, hd = _random_pos_hd(T, grid_size=5)
        obs = _random_obs(T)
        locobs_s, vt_s, vc_s = get_obs_at_loc(obs, [3, 3], pos, hd)
        locobs_f, vt_f, vc_f = get_obs_at_loc_fast(obs, [3, 3], pos, hd)

        if locobs_s is None:
            assert locobs_f is None
        else:
            np.testing.assert_array_almost_equal(locobs_f, locobs_s)

    def test_pos_longer_than_obs(self):
        """pos may have more rows than obs (T+1 vs T); fast version handles this."""
        T = 20
        pos, hd = _random_pos_hd(T + 1)   # one extra row, like B*(T+1)
        obs = _random_obs(T)               # only T rows
        self._compare(obs, [3, 3], pos, hd)

    @pytest.mark.parametrize("goal_loc", [[2, 4], [6, 1], [4, 4]])
    def test_multiple_goal_locs(self, goal_loc):
        T = 40
        pos, hd = _random_pos_hd(T)
        obs = _random_obs(T)
        self._compare(obs, goal_loc, pos, hd)


# ---------------------------------------------------------------------------
# Saved trajectory fixture
# ---------------------------------------------------------------------------

TRAJ_PATH = Path("outputs/data_cur_lroom_step1608_goal711/trajectories.pt")

# goal_loc=[7,11] encoded in the directory name (goal711); config.goal_loc is None in this file
_GOAL_LOC = [7, 11]


@pytest.fixture(scope="module")
def trajectories():
    return torch.load(TRAJ_PATH, weights_only=False)


@pytest.mark.skipif(not TRAJ_PATH.exists(), reason="trajectories.pt not found")
class TestSavedTrajectories:
    """Validate the shape and dtype of a saved EvalTrajectories file."""

    def test_keys_present(self, trajectories):
        for key in ("obs", "obs_pred", "obs_next", "act", "states", "hidden_states", "config"):
            assert key in trajectories, f"missing key: {key}"

    def test_batch_size_consistent(self, trajectories):
        B = trajectories["obs"].shape[0]
        assert trajectories["obs_pred"].shape[0] == B
        assert trajectories["obs_next"].shape[0] == B
        assert trajectories["act"].shape[0] == B
        assert len(trajectories["states"]) == B
        if trajectories["hidden_states"] is not None:
            assert trajectories["hidden_states"].shape[0] == B

    def test_time_dims_consistent(self, trajectories):
        T_plus_1 = trajectories["obs"].shape[1]   # B x T+1 x X
        T = T_plus_1 - 1
        assert trajectories["obs_pred"].shape[1] == T
        assert trajectories["obs_next"].shape[1] == T
        assert trajectories["act"].shape[1] == T
        if trajectories["hidden_states"] is not None:
            assert trajectories["hidden_states"].shape[1] == T

    def test_obs_feature_dim_consistent(self, trajectories):
        X = trajectories["obs"].shape[2]
        assert trajectories["obs_pred"].shape[2] == X
        assert trajectories["obs_next"].shape[2] == X

    def test_dtypes(self, trajectories):
        for key in ("obs", "obs_pred", "obs_next", "act"):
            assert trajectories[key].dtype == torch.float32, f"{key} wrong dtype"

    def test_states_inner_shapes(self, trajectories):
        T_plus_1 = trajectories["obs"].shape[1]
        for b, state in enumerate(trajectories["states"]):
            assert state["agent_pos"].shape == (T_plus_1, 2), f"traj {b}: agent_pos shape"
            assert state["agent_dir"].shape == (T_plus_1,), f"traj {b}: agent_dir shape"


# ---------------------------------------------------------------------------
# Consistency tests: scalar vs vectorized, using real saved trajectories
# ---------------------------------------------------------------------------

# Indices of trajectories to test (a small representative sample)
_SAMPLE_IDXS = [0, 1, 50, 100, 300, 687]


def _traj_obs_pos_hd(trajectories, b):
    """Return (obs, pos, hd) numpy arrays for trajectory b."""
    obs = trajectories["obs"][b].numpy().reshape(-1, 7, 7, 3)  # (T+1, 7, 7, 3)
    state = trajectories["states"][b]
    pos = state["agent_pos"].astype(np.float32)   # (T+1, 2)
    hd = state["agent_dir"].astype(np.int32)       # (T+1,)
    return obs, pos, hd


@pytest.mark.skipif(not TRAJ_PATH.exists(), reason="trajectories.pt not found")
class TestViewCoordsBatchOnRealTrajectories:
    """get_view_coords_batch must agree with get_view_coords on saved trajectories."""

    @pytest.mark.parametrize("b", _SAMPLE_IDXS)
    def test_matches_scalar(self, trajectories, b):
        obs, pos, hd = _traj_obs_pos_hd(trajectories, b)
        i, j = _GOAL_LOC
        T = obs.shape[0]

        vx_batch, vy_batch = get_view_coords_batch(i, j, pos, hd)

        for t in range(T):
            vx_s, vy_s = get_view_coords(i, j, pos[t], hd[t])
            assert vx_batch[t] == vx_s, f"traj {b}, t={t}: vx mismatch"
            assert vy_batch[t] == vy_s, f"traj {b}, t={t}: vy mismatch"


@pytest.mark.skipif(not TRAJ_PATH.exists(), reason="trajectories.pt not found")
class TestObsAtLocFastOnRealTrajectories:
    """get_obs_at_loc_fast must agree with get_obs_at_loc on saved trajectories."""

    @pytest.mark.parametrize("b", _SAMPLE_IDXS)
    def test_matches_slow(self, trajectories, b):
        obs, pos, hd = _traj_obs_pos_hd(trajectories, b)

        slow = get_obs_at_loc(obs, _GOAL_LOC, pos, hd)
        fast = get_obs_at_loc_fast(obs, _GOAL_LOC, pos, hd)

        if slow[0] is None:
            assert fast[0] is None and fast[1] is None and fast[2] is None
        else:
            np.testing.assert_array_equal(fast[0], slow[0], err_msg=f"traj {b}: locobs mismatch")
            np.testing.assert_array_equal(fast[1], slow[1], err_msg=f"traj {b}: viewtimes mismatch")
            np.testing.assert_array_equal(fast[2], slow[2], err_msg=f"traj {b}: viewcoords mismatch")
