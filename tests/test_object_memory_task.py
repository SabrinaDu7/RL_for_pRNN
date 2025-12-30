import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock

from tasks.ObjectMemoryTask.define_task import get_obs_at_loc, ObjectMemoryTask


class TestQuantifyObjectLearning:
    """Test the quantifyObjectLearning method with multiple control locations."""

    def create_mock_object_memory_task(self):
        """Create a mock ObjectMemoryTask instance with necessary attributes."""
        mock_task = MagicMock()

        # Create mock test trial data
        # Simulate a trajectory with 100 timesteps
        timesteps = 100
        mock_task.testTrial = {
            "state": {
                "agent_pos": np.random.randint(0, 10, size=(timesteps, 2)),
                "agent_dir": np.random.randint(0, 4, size=timesteps),
            },
            "obs_pred": torch.randn(timesteps, 7, 7, 3),  # Mock predictions
            "obs_pred_control": torch.randn(timesteps, 7, 7, 3),  # Mock control predictions
        }

        # Mock the env_shell and pRNN methods
        mock_task.pN_post = MagicMock()
        mock_task.pN_post.env_shell.pred2np = MagicMock(
            side_effect=lambda pred, whichPhase: pred.numpy() if isinstance(pred, torch.Tensor) else pred
        )

        mock_task.new_obj_pos = [5, 5]

        return mock_task

    def test_single_valid_control_location(self):
        """Test with a single valid control location."""
        from tasks.ObjectMemoryTask.define_task import ObjectMemoryTask

        mock_task = self.create_mock_object_memory_task()

        # Mock get_obs_at_loc to return consistent valid observations
        with patch('tasks.ObjectMemoryTask.define_task.get_obs_at_loc') as mock_get_obs:
            # Setup return values
            # Object location returns valid observations
            obj_obs = np.random.rand(10, 3)  # 10 views, RGB
            obj_obs_control = np.random.rand(10, 3)

            # Control location returns valid observations
            ctrl_obs = np.random.rand(5, 3)  # 5 views, RGB
            ctrl_obs_control = np.random.rand(5, 3)

            mock_get_obs.side_effect = [
                (obj_obs, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), np.array([[0, 0]] * 10)),  # Object location trained
                (obj_obs_control, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), np.array([[0, 0]] * 10)),  # Object location control
                (ctrl_obs, np.array([1, 2, 3, 4, 5]), np.array([[0, 0]] * 5)),  # Control location 1 trained
                (ctrl_obs_control, np.array([1, 2, 3, 4, 5]), np.array([[0, 0]] * 5)),  # Control location 1 control
            ]

            # Call the actual method from ObjectMemoryTask
            result = ObjectMemoryTask.quantifyObjectLearning(
                mock_task,
                control_location=[[4, 7]],
                whichPhase=0,
                traj_count=100
            )

            assert result is not None
            assert "controlloc_obs" in result
            assert "controlloc_deltaobs" in result
            assert result["controlloc_obs"].shape == (5, 3)  # Should be the control observations

    def test_multiple_valid_control_locations(self):
        """Test with multiple valid control locations - observations should be concatenated."""
        from tasks.ObjectMemoryTask.define_task import ObjectMemoryTask

        mock_task = self.create_mock_object_memory_task()

        with patch('tasks.ObjectMemoryTask.define_task.get_obs_at_loc') as mock_get_obs:
            # Object location returns valid observations
            obj_obs = np.random.rand(10, 3)
            obj_obs_control = np.random.rand(10, 3)

            # Control location 1 has 5 views
            ctrl_obs_1 = np.random.rand(5, 3)
            ctrl_obs_control_1 = np.random.rand(5, 3)

            # Control location 2 has 7 views
            ctrl_obs_2 = np.random.rand(7, 3)
            ctrl_obs_control_2 = np.random.rand(7, 3)

            mock_get_obs.side_effect = [
                (obj_obs, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), np.array([[0, 0]] * 10)),
                (obj_obs_control, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), np.array([[0, 0]] * 10)),
                (ctrl_obs_1, np.array([1, 2, 3, 4, 5]), np.array([[0, 0]] * 5)),
                (ctrl_obs_control_1, np.array([1, 2, 3, 4, 5]), np.array([[0, 0]] * 5)),
                (ctrl_obs_2, np.array([1, 2, 3, 4, 5, 6, 7]), np.array([[0, 0]] * 7)),
                (ctrl_obs_control_2, np.array([1, 2, 3, 4, 5, 6, 7]), np.array([[0, 0]] * 7)),
            ]

            result = ObjectMemoryTask.quantifyObjectLearning(
                mock_task,
                control_location=[[4, 7], [3, 6]],
                whichPhase=0,
                traj_count=100
            )

            assert result is not None
            # Observations should be concatenated: 5 + 7 = 12 total views
            assert result["controlloc_obs"].shape == (12, 3)
            assert result["controlloc_deltaobs"].shape == (12, 3)

    def test_some_invalid_control_locations(self):
        """Test with some invalid control locations - should skip None and use valid ones."""
        from tasks.ObjectMemoryTask.define_task import ObjectMemoryTask

        mock_task = self.create_mock_object_memory_task()

        with patch('tasks.ObjectMemoryTask.define_task.get_obs_at_loc') as mock_get_obs:
            obj_obs = np.random.rand(10, 3)
            obj_obs_control = np.random.rand(10, 3)

            # Control location 2 has valid observations
            ctrl_obs_2 = np.random.rand(7, 3)
            ctrl_obs_control_2 = np.random.rand(7, 3)

            mock_get_obs.side_effect = [
                (obj_obs, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), np.array([[0, 0]] * 10)),
                (obj_obs_control, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), np.array([[0, 0]] * 10)),
                (None, None, None),  # Control location 1 is invalid
                (None, None, None),  # Control location 1 control is also None
                (ctrl_obs_2, np.array([1, 2, 3, 4, 5, 6, 7]), np.array([[0, 0]] * 7)),
                (ctrl_obs_control_2, np.array([1, 2, 3, 4, 5, 6, 7]), np.array([[0, 0]] * 7)),
            ]

            result = ObjectMemoryTask.quantifyObjectLearning(
                mock_task,
                control_location=[[4, 7], [3, 6]],
                whichPhase=0,
                traj_count=100
            )

            assert result is not None
            # Should only have observations from control location 2
            assert result["controlloc_obs"].shape == (7, 3)

    def test_all_invalid_control_locations(self):
        """Test with all invalid control locations - should return None."""
        from tasks.ObjectMemoryTask.define_task import ObjectMemoryTask

        mock_task = self.create_mock_object_memory_task()

        with patch('tasks.ObjectMemoryTask.define_task.get_obs_at_loc') as mock_get_obs:
            obj_obs = np.random.rand(10, 3)
            obj_obs_control = np.random.rand(10, 3)

            mock_get_obs.side_effect = [
                (obj_obs, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), np.array([[0, 0]] * 10)),
                (obj_obs_control, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), np.array([[0, 0]] * 10)),
                (None, None, None),  # Control location 1 invalid
                (None, None, None),
                (None, None, None),  # Control location 2 invalid
                (None, None, None),
            ]

            result = ObjectMemoryTask.quantifyObjectLearning(
                mock_task,
                control_location=[[4, 7], [3, 6]],
                whichPhase=0,
                traj_count=100
            )

            # Should return None when no control locations have valid observations
            assert result is None

    def test_invalid_object_location(self):
        """Test when object location is invalid but control locations are valid."""
        from tasks.ObjectMemoryTask.define_task import ObjectMemoryTask

        mock_task = self.create_mock_object_memory_task()

        with patch('tasks.ObjectMemoryTask.define_task.get_obs_at_loc') as mock_get_obs:
            # Control location has valid observations
            ctrl_obs = np.random.rand(5, 3)
            ctrl_obs_control = np.random.rand(5, 3)

            mock_get_obs.side_effect = [
                (None, None, None),  # Object location invalid
                (None, None, None),  # Object location control invalid
                (ctrl_obs, np.array([1, 2, 3, 4, 5]), np.array([[0, 0]] * 5)),
                (ctrl_obs_control, np.array([1, 2, 3, 4, 5]), np.array([[0, 0]] * 5)),
            ]

            result = ObjectMemoryTask.quantifyObjectLearning(
                mock_task,
                control_location=[[4, 7]],
                whichPhase=0,
                traj_count=100
            )

            # Should return None when object location is invalid
            assert result is None

    def test_averaging_correctness(self):
        """Test that the averaging of control observations is mathematically correct."""
        from tasks.ObjectMemoryTask.define_task import ObjectMemoryTask

        mock_task = self.create_mock_object_memory_task()

        with patch('tasks.ObjectMemoryTask.define_task.get_obs_at_loc') as mock_get_obs:
            obj_obs = np.random.rand(10, 3)
            obj_obs_control = np.random.rand(10, 3)

            # Create known control observations for testing
            ctrl_obs_1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 views
            ctrl_obs_control_1 = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])

            ctrl_obs_2 = np.array([[7.0, 8.0, 9.0]])  # 1 view
            ctrl_obs_control_2 = np.array([[3.5, 4.0, 4.5]])

            mock_get_obs.side_effect = [
                (obj_obs, np.arange(10), np.array([[0, 0]] * 10)),
                (obj_obs_control, np.arange(10), np.array([[0, 0]] * 10)),
                (ctrl_obs_1, np.array([1, 2]), np.array([[0, 0]] * 2)),
                (ctrl_obs_control_1, np.array([1, 2]), np.array([[0, 0]] * 2)),
                (ctrl_obs_2, np.array([1]), np.array([[0, 0]])),
                (ctrl_obs_control_2, np.array([1]), np.array([[0, 0]])),
            ]

            result = ObjectMemoryTask.quantifyObjectLearning(
                mock_task,
                control_location=[[4, 7], [3, 6]],
                whichPhase=0,
                traj_count=100
            )

            assert result is not None

            # Expected concatenated observations
            expected_ctrl_obs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
            expected_ctrl_obs_control = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0], [3.5, 4.0, 4.5]])

            np.testing.assert_array_almost_equal(result["controlloc_obs"], expected_ctrl_obs)
            np.testing.assert_array_almost_equal(result["controlloc_obs_controlNet"], expected_ctrl_obs_control)

            # Check delta calculations
            expected_delta = expected_ctrl_obs - expected_ctrl_obs_control
            np.testing.assert_array_almost_equal(result["controlloc_deltaobs"], expected_delta)


class TestGetTestTrial:
    """Test the getTestTrial method with multiple trajectories."""

    def test_single_trajectory(self):
        """Test getTestTrial with n_trajs=1."""
        # Create mock ObjectMemoryTask (without spec to allow flexible mocking)
        mock_omt = MagicMock()
        mock_omt.seqdur = 256
        mock_omt.args = MagicMock()
        mock_omt.args.tasks.testing.whichPhase = 0
        mock_omt.agent = MagicMock()
        mock_omt.env_orig = MagicMock()

        # Mock the pRNNs
        mock_omt.pN_post = MagicMock()
        mock_omt.pN_control = MagicMock()

        # Mock pRNN parameters for device detection
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_omt.pN_post.pRNN.parameters = MagicMock(return_value=iter([mock_param]))

        # Mock collectObservationSequence to return consistent data
        obs = [{"image": np.random.rand(7, 7, 3)} for _ in range(257)]  # seqdur+1
        act = np.random.randint(0, 4, size=256)  # seqdur actions
        state = {
            "agent_pos": np.random.randint(0, 10, size=(257, 2)),
            "agent_dir": np.random.randint(0, 4, size=257),
            "SRs": np.random.randn(257, 128),
        }
        render = [np.random.rand(64, 64, 3) for _ in range(257)]

        mock_omt.pN_post.collectObservationSequence = MagicMock(return_value=(obs, act, state, render))

        # Mock predict methods
        obs_pred = torch.randn(1, 256, 64)  # (batch, time, latent)
        mock_omt.pN_post.predict = MagicMock(return_value=(obs_pred, None, None))
        mock_omt.pN_control.predict = MagicMock(return_value=(obs_pred, None, None))

        # Call the actual method
        result = ObjectMemoryTask.getTestTrial(mock_omt, n_trajs=1)

        # Verify the method was called once
        assert mock_omt.pN_post.collectObservationSequence.call_count == 1

        # Verify returned structure
        assert "obs" in result
        assert "obs_pred" in result
        assert "obs_pred_control" in result
        assert "state" in result
        assert "render" in result

    def test_multiple_trajectories_concatenation(self):
        """Test that getTestTrial correctly concatenates multiple trajectories."""
        # Create mock ObjectMemoryTask
        mock_omt = MagicMock()
        mock_omt.seqdur = 100
        mock_omt.args = MagicMock()
        mock_omt.args.tasks.testing.whichPhase = 10  # Skip first 10 timesteps
        mock_omt.agent = MagicMock()
        mock_omt.env_orig = MagicMock()

        # Mock the pRNNs
        mock_omt.pN_post = MagicMock()
        mock_omt.pN_control = MagicMock()

        # Mock pRNN parameters
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_omt.pN_post.pRNN.parameters = MagicMock(return_value=iter([mock_param]))

        # Track calls to collectObservationSequence
        call_count = [0]

        def mock_collect_obs(*args, **kwargs):
            call_count[0] += 1
            obs = [{"image": np.random.rand(7, 7, 3)} for _ in range(101)]  # seqdur+1
            act = np.random.randint(0, 4, size=100)
            state = {
                "agent_pos": np.random.randint(0, 10, size=(101, 2)),
                "agent_dir": np.random.randint(0, 4, size=101),
                "SRs": np.random.randn(101, 128),
            }
            render = [np.random.rand(64, 64, 3) for _ in range(101)]
            return obs, act, state, render

        mock_omt.pN_post.collectObservationSequence = MagicMock(side_effect=mock_collect_obs)

        # Mock predict methods
        def mock_predict(obs, act):
            # Return predictions matching the sliced data
            timesteps = len(act)
            return torch.randn(1, timesteps, 64), None, None

        mock_omt.pN_post.predict = MagicMock(side_effect=mock_predict)
        mock_omt.pN_control.predict = MagicMock(side_effect=mock_predict)

        # Call with n_trajs=3
        result = ObjectMemoryTask.getTestTrial(mock_omt, n_trajs=3)

        # Verify collectObservationSequence was called 3 times
        assert call_count[0] == 3

        # Verify concatenation shapes
        # Each trajectory: seqdur+1=101 timesteps, whichPhase=10 -> 91 timesteps per traj
        # Total: 3 * 91 = 273 timesteps
        expected_timesteps = 3 * (101 - 10)

        assert len(result["obs"]) == expected_timesteps
        assert result["state"]["agent_pos"].shape[0] == expected_timesteps
        assert result["state"]["agent_dir"].shape[0] == expected_timesteps
        assert result["state"]["SRs"].shape[0] == expected_timesteps
        assert len(result["render"]) == expected_timesteps

        # Predictions: 3 trajectories * 90 timesteps (seqdur - whichPhase)
        expected_pred_timesteps = 3 * (100 - 10)
        assert result["obs_pred"].shape == (1, expected_pred_timesteps, 64)

    def test_whichPhase_slicing_applied_per_trajectory(self):
        """Test that whichPhase is applied to each trajectory individually."""
        mock_omt = MagicMock()
        mock_omt.seqdur = 50
        mock_omt.args = MagicMock()
        mock_omt.args.tasks.testing.whichPhase = 5
        mock_omt.agent = MagicMock()
        mock_omt.env_orig = MagicMock()

        # Mock the pRNNs
        mock_omt.pN_post = MagicMock()
        mock_omt.pN_control = MagicMock()

        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_omt.pN_post.pRNN.parameters = MagicMock(return_value=iter([mock_param]))

        # Create deterministic data for verification
        def mock_collect_obs(*args, **kwargs):
            obs = list(range(51))  # 0, 1, 2, ..., 50
            act = np.arange(50)  # 0, 1, 2, ..., 49
            state = {
                "agent_pos": np.arange(51).reshape(-1, 1).repeat(2, axis=1),
                "agent_dir": np.arange(51),
                "SRs": np.arange(51).reshape(-1, 1).repeat(128, axis=1),
            }
            render = list(range(51))
            return obs, act, state, render

        mock_omt.pN_post.collectObservationSequence = MagicMock(side_effect=mock_collect_obs)

        def mock_predict(obs, act):
            timesteps = len(act)
            return torch.randn(1, timesteps, 64), None, None

        mock_omt.pN_post.predict = MagicMock(side_effect=mock_predict)
        mock_omt.pN_control.predict = MagicMock(side_effect=mock_predict)

        # Call with n_trajs=2
        result = ObjectMemoryTask.getTestTrial(mock_omt, n_trajs=2)

        # Each trajectory: [0, 1, 2, ..., 50], skip first 5 -> [5, 6, ..., 50] = 46 elements
        # Two trajectories: 2 * 46 = 92 elements
        assert len(result["obs"]) == 92

        # First trajectory should start at 5, second also at 5 (not 51+5=56)
        assert result["obs"][0] == 5  # First element from first trajectory
        assert result["obs"][46] == 5  # First element from second trajectory

    def test_concatenated_data_works_with_quantifyObjectLearning(self):
        """Integration test: verify concatenated data works with quantifyObjectLearning."""
        mock_omt = MagicMock()
        mock_omt.seqdur = 50
        mock_omt.args = MagicMock()
        mock_omt.args.tasks.testing.whichPhase = 0
        mock_omt.new_obj_pos = [5, 5]
        mock_omt.agent = MagicMock()
        mock_omt.env_orig = MagicMock()

        # Mock the pRNNs and env_shell
        mock_omt.pN_post = MagicMock()
        mock_omt.pN_control = MagicMock()

        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_omt.pN_post.pRNN.parameters = MagicMock(return_value=iter([mock_param]))

        # Create realistic data
        def mock_collect_obs(*args, **kwargs):
            obs = [{"image": np.random.rand(7, 7, 3)} for _ in range(51)]
            act = np.random.randint(0, 4, size=50)
            state = {
                "agent_pos": np.random.randint(0, 10, size=(51, 2)),
                "agent_dir": np.random.randint(0, 4, size=51),
                "SRs": np.random.randn(51, 128),
            }
            render = [np.random.rand(64, 64, 3) for _ in range(51)]
            return obs, act, state, render

        mock_omt.pN_post.collectObservationSequence = MagicMock(side_effect=mock_collect_obs)

        def mock_predict(obs, act):
            timesteps = len(act)
            return torch.randn(1, timesteps, 64), None, None

        mock_omt.pN_post.predict = MagicMock(side_effect=mock_predict)
        mock_omt.pN_control.predict = MagicMock(side_effect=mock_predict)

        # Mock pred2np
        def mock_pred2np(obs_pred, whichPhase):
            # Return numpy array with expected shape (T, H, W, 3)
            T = obs_pred.shape[1]
            return np.random.rand(T, 64, 64, 3)

        mock_omt.pN_post.env_shell.pred2np = MagicMock(side_effect=mock_pred2np)

        # Run getTestTrial
        testTrial = ObjectMemoryTask.getTestTrial(mock_omt, n_trajs=2)
        mock_omt.testTrial = testTrial

        # Verify the data structure is compatible
        assert "state" in testTrial
        assert "agent_pos" in testTrial["state"]
        assert "agent_dir" in testTrial["state"]
        assert testTrial["state"]["agent_pos"].ndim == 2
        assert testTrial["state"]["agent_dir"].ndim == 1
