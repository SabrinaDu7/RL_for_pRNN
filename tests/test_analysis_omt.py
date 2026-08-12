import numpy as np
import pytest
import torch
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

from prnn.utils import (
    PredictiveNet,
    ActionEncodingsEnum,
    MinigridEnvNames,
    pRNNtypes,
    RandomActionAgent,
)
from prnn.utils.Shell import FaramaMinigridShell

from curious_george import make_env, get_pN, get_SR_acmodel, get_obss_preprocessor, ActorCriticAgent, seed, DEVICE
from curious_george import get_ckpt_env_vars, AgentInputType, AgentType

from scripts.analysis_OMT import (
    EvalTrajectoryConfig,
    EvalTrajectories,
    LabelFn,
    eval_mode,
    on_device,
    collect_eval_trajectories,
    save_eval_trajectories,
    load_eval_trajectories,
    get_walkable_mask,
    get_walkable_minigrid_positions as get_walkable_positions,
)
from scripts.legacy.isomap import make_novel_obj_in_view_label_fn


# ---------------------------------------------------------------------------
# Helpers (reused from test_ckpts.py patterns)
# ---------------------------------------------------------------------------

PRNN_CKPT, ACMODEL_STATUS_CKPT = get_ckpt_env_vars(AgentType.AC)
PRNN_RAND_CKPT, ACMODEL_STATUS_RAND_CKPT = get_ckpt_env_vars(AgentType.RANDOM)


def _get_pRNN(prnn_ckpt: str, device: torch.device, env: FaramaMinigridShell | None = None) -> PredictiveNet:
    @dataclass
    class PredNetArgs:
        hiddensize: int = 500
        pRNNtype: str = pRNNtypes.masked.value
        lr: float = 3e-3
        bptttrunc: float = 1e8
        weight_decay: float = 3e-3
        ntimescale: int = 2
        dropout: float = 0.15
        noisemean: float = 0
        noisestd: float = 0.05
        sparsity: float = 0.5

    @dataclass
    class LoggingArgs:
        wandb_log: bool = False

    @dataclass
    class Args:
        predNet: PredNetArgs = PredNetArgs()
        logging: LoggingArgs = LoggingArgs()

    if env is None:
        env = make_env(
            env_key=MinigridEnvNames.LRoom,
            agent_start_pos=None,
            input_type=AgentInputType.H_PO,
            act_enc=ActionEncodingsEnum.SpeedHD,
        )
    return get_pN(args=Args(), env=env, device=device, pRNN_ckpt=prnn_ckpt)


def _get_env() -> FaramaMinigridShell:
    return make_env(
        env_key=MinigridEnvNames.LRoom,
        input_type=AgentInputType.H_PO,
        act_enc=ActionEncodingsEnum.SpeedHD,
    )


# ---------------------------------------------------------------------------
# Fast unit tests
# ---------------------------------------------------------------------------


class TestEvalTrajectoryConfig:
    def test_defaults(self):
        # save_path became a required field (no default) in analysis_OMT.py
        cfg = EvalTrajectoryConfig(save_path=None, timesteps=100)
        assert cfg.timesteps == 100
        assert cfg.include_render is False
        assert cfg.include_hidden_states is False
        assert cfg.save_path is None

    def test_all_fields(self):
        cfg = EvalTrajectoryConfig(
            timesteps=50,
            include_render=True,
            include_hidden_states=True,
            save_path=Path("/tmp/test"),
        )
        assert cfg.include_render is True
        assert cfg.include_hidden_states is True


class TestEvalModeContextManager:
    def test_restores_actor_critic(self):
        """eval_mode should set eval+argmax and restore train+original argmax."""
        mock_pN = MagicMock()
        mock_pN.pRNN.training = True

        mock_agent = MagicMock()
        mock_agent.__class__ = ActorCriticAgent  # eval_mode isinstance-asserts
        mock_agent.acmodel.training = True
        mock_agent.argmax = False

        with eval_mode(mock_pN, mock_agent):
            mock_pN.pRNN.eval.assert_called_once()
            mock_agent.acmodel.eval.assert_called_once()
            assert mock_agent.argmax is True

        # After exit: should restore
        mock_pN.pRNN.train.assert_called_once()
        mock_agent.acmodel.train.assert_called_once()
        assert mock_agent.argmax is False

    def test_works_with_random_agent(self):
        """eval_mode should not crash when agent has no acmodel."""
        mock_pN = MagicMock()
        mock_pN.pRNN.training = True

        mock_agent = MagicMock(spec=RandomActionAgent)
        # RandomActionAgent has no acmodel attribute
        del mock_agent.acmodel
        del mock_agent.argmax

        with eval_mode(mock_pN, mock_agent):
            mock_pN.pRNN.eval.assert_called_once()

        mock_pN.pRNN.train.assert_called_once()

    def test_restores_on_exception(self):
        """eval_mode should restore state even if body raises."""
        mock_pN = MagicMock()
        mock_pN.pRNN.training = True

        mock_agent = MagicMock()
        mock_agent.__class__ = ActorCriticAgent  # eval_mode isinstance-asserts
        mock_agent.acmodel.training = True
        mock_agent.argmax = False

        with pytest.raises(ValueError):
            with eval_mode(mock_pN, mock_agent):
                raise ValueError("test error")

        mock_pN.pRNN.train.assert_called_once()
        mock_agent.acmodel.train.assert_called_once()
        assert mock_agent.argmax is False


class TestOnDeviceContextManager:
    def test_restores_device(self):
        """on_device should move pRNN and restore original device."""
        linear = torch.nn.Linear(10, 10)
        mock_pN = MagicMock()
        mock_pN.pRNN = linear

        original_device = next(linear.parameters()).device
        assert original_device == torch.device("cpu")

        with on_device(mock_pN, torch.device("cpu")):
            assert next(mock_pN.pRNN.parameters()).device == torch.device("cpu")

        assert next(mock_pN.pRNN.parameters()).device == original_device


class TestSaveLoadRoundtrip:
    def test_roundtrip(self, tmp_path):
        """save then load should preserve tensor shapes and values."""
        B, T, X, A = 2, 5, 10, 3
        data: EvalTrajectories = {
            "obs": torch.randn(B, T + 1, X),
            "obs_pred": torch.randn(B, T, X),
            "obs_next": torch.randn(B, T, X),
            "act": torch.randn(B, T, A),
            "states": [
                {
                    "agent_pos": np.random.randn(T + 1, 2).astype(np.float32),
                    "agent_dir": np.random.randint(0, 4, T + 1).astype(np.int32),
                }
                for _ in range(B)
            ],
            "hidden_states": None,
            "labels": None,
            "renders": None,
        }

        save_eval_trajectories(data, tmp_path / "out")

        assert (tmp_path / "out" / "trajectories.pt").exists()
        assert (tmp_path / "out" / "summary.parquet").exists()

        loaded = load_eval_trajectories(tmp_path / "out")  # dir, like save
        assert torch.allclose(loaded["obs"], data["obs"])
        assert torch.allclose(loaded["obs_pred"], data["obs_pred"])
        assert torch.allclose(loaded["obs_next"], data["obs_next"])
        assert torch.allclose(loaded["act"], data["act"])
        for b in range(B):
            np.testing.assert_array_equal(
                loaded["states"][b]["agent_pos"], data["states"][b]["agent_pos"]
            )
            np.testing.assert_array_equal(
                loaded["states"][b]["agent_dir"], data["states"][b]["agent_dir"]
            )

    def test_save_creates_parquet_with_correct_columns(self, tmp_path):
        """Parquet summary should have expected columns."""
        import pandas as pd

        B, T, X, A = 2, 3, 5, 2
        data: EvalTrajectories = {
            "obs": torch.randn(B, T + 1, X),
            "obs_pred": torch.randn(B, T, X),
            "obs_next": torch.randn(B, T, X),
            "act": torch.randn(B, T, A),
            "states": [
                {
                    "agent_pos": np.random.randn(T + 1, 2).astype(np.float32),
                    "agent_dir": np.random.randint(0, 4, T + 1).astype(np.int32),
                }
                for _ in range(B)
            ],
            "hidden_states": None,
            "labels": None,
            "renders": None,
        }

        save_eval_trajectories(data, tmp_path / "out")

        df = pd.read_parquet(tmp_path / "out" / "summary.parquet")
        assert set(df.columns) == {"traj_id", "timestep", "pos_x", "pos_y", "direction"}
        assert len(df) == B * (T + 1)


class TestGetWalkablePositions:
    def test_offset_by_one(self):
        """Returned positions should be mask indices + 1."""
        mask = torch.zeros(3, 3, dtype=torch.bool)
        mask[0, 0] = True
        mask[2, 1] = True

        positions = get_walkable_positions(mask)

        assert positions.shape == (2, 2)
        assert positions.dtype == torch.long
        expected = torch.tensor([[1, 1], [3, 2]])
        assert torch.equal(positions, expected)

    def test_all_walkable(self):
        """All-True mask should return all positions offset by 1."""
        mask = torch.ones(2, 2, dtype=torch.bool)
        positions = get_walkable_positions(mask)

        assert positions.shape == (4, 2)
        assert positions.min().item() == 1

    def test_no_walkable(self):
        """All-False mask should return an empty tensor."""
        mask = torch.zeros(3, 3, dtype=torch.bool)
        positions = get_walkable_positions(mask)

        assert positions.shape == (0, 2)


# ---------------------------------------------------------------------------
# Slow integration tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCollectEvalTrajectories:
    def test_shapes_random_agent(self):
        """Collect trajectories with random agent, verify all tensor shapes."""
        seed(42)
        env = _get_env()
        pN = _get_pRNN(prnn_ckpt=PRNN_RAND_CKPT, device=DEVICE, env=env)

        action_prob = np.array([0.15, 0.15, 0.6, 0.1])
        agent = RandomActionAgent(env.action_space, action_prob)

        T = 10
        config = EvalTrajectoryConfig(save_path=None, timesteps=T)
        result = collect_eval_trajectories(pN, agent, env, config)

        _, view_size, C = env.obs_shape
        X = view_size * view_size * C
        A = env.getActSize()

        # B = num_walkable * 4 (one trajectory per position × direction)
        num_walkable = int(get_walkable_mask(env).sum())
        B = num_walkable * 4

        assert result["obs"].shape == (B, T + 1, X)
        assert result["obs_pred"].shape[0] == B
        assert result["obs_pred"].shape[-1] == X
        assert result["obs_next"].shape[0] == B
        assert result["obs_next"].shape[-1] == X
        assert result["act"].shape == (B, T, A)
        assert len(result["states"]) == B
        assert result["states"][0]["agent_pos"].shape == (T + 1, 2)
        assert result["states"][0]["agent_dir"].shape == (T + 1,)
        assert result["hidden_states"] is None
        assert result["renders"] is None

    def test_with_hidden_states(self):
        """include_hidden_states=True should populate hidden_states tensor."""
        seed(42)
        env = _get_env()
        pN = _get_pRNN(prnn_ckpt=PRNN_RAND_CKPT, device=DEVICE, env=env)

        action_prob = np.array([0.15, 0.15, 0.6, 0.1])
        agent = RandomActionAgent(env.action_space, action_prob)

        T = 10
        num_walkable = int(get_walkable_mask(env).sum())
        B = num_walkable * 4

        config = EvalTrajectoryConfig(
            save_path=None, timesteps=T, include_hidden_states=True
        )
        result = collect_eval_trajectories(pN, agent, env, config)

        assert result["hidden_states"] is not None
        assert result["hidden_states"].shape[0] == B
        assert result["hidden_states"].shape[-1] == pN.hidden_size

    def test_with_save(self, tmp_path):
        """save_path should produce .pt and .parquet files."""
        seed(42)
        env = _get_env()
        pN = _get_pRNN(prnn_ckpt=PRNN_RAND_CKPT, device=DEVICE, env=env)

        action_prob = np.array([0.15, 0.15, 0.6, 0.1])
        agent = RandomActionAgent(env.action_space, action_prob)

        save_dir = tmp_path / "eval_out"
        config = EvalTrajectoryConfig(
            timesteps=10, save_path=save_dir
        )
        collect_eval_trajectories(pN, agent, env, config)

        assert (save_dir / "trajectories.pt").exists()
        assert (save_dir / "summary.parquet").exists()

    def test_covers_all_positions_and_directions(self):
        """Every (walkable_position, direction) pair should have a trajectory."""
        seed(42)
        env = _get_env()
        pN = _get_pRNN(prnn_ckpt=PRNN_RAND_CKPT, device=DEVICE, env=env)

        action_prob = np.array([0.15, 0.15, 0.6, 0.1])
        agent = RandomActionAgent(env.action_space, action_prob)

        config = EvalTrajectoryConfig(timesteps=5)
        result = collect_eval_trajectories(pN, agent, env, config)

        num_walkable = int(get_walkable_mask(env).sum())
        assert len(result["states"]) == num_walkable * 4

        # Check that each trajectory starts at a valid position with no jumps
        for state in result["states"]:
            positions = state["agent_pos"]
            deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            assert np.all(deltas <= np.sqrt(2)), "Agent position jumped unexpectedly"


# ---------------------------------------------------------------------------
# Tests for label_fn hook
# ---------------------------------------------------------------------------


class TestMakeNovelObjInViewLabelFn:
    """Unit tests for make_novel_obj_in_view_label_fn (no env / pRNN needed)."""

    def _make_states(self, positions: list[np.ndarray], directions: list[np.ndarray]) -> list:
        """Build minimal State dicts from position/direction arrays."""
        return [
            {"agent_pos": pos, "agent_dir": hd}
            for pos, hd in zip(positions, directions)
        ]

    def test_returns_zeros_when_out_of_view(self):
        """All-zeros label when object is behind the agent."""
        # Agent at (5, 5) facing right (dir=0), object at (5, 0) far left
        B, T = 1, 3
        pos = np.tile(np.array([[5, 5]]), (T, 1)).astype(np.float32)
        hd = np.zeros(T, dtype=np.int32)
        states = self._make_states([pos], [hd])

        fn = make_novel_obj_in_view_label_fn(obj_pos=[5, 0])
        labels = fn(states, T)

        assert labels.shape == (B, T)
        assert labels.sum().item() == 0

    def test_returns_ones_when_in_view(self):
        """Object directly in front should produce label=1 at every step."""
        # Agent at (2, 2) facing right (dir=0, forward = +x).
        # With view_size=7, the agent sees columns 2..8 ahead.
        # Object at (5, 2) is 3 cells ahead and centered — should be in view.
        B, T = 1, 4
        pos = np.tile(np.array([[2, 2]]), (T, 1)).astype(np.float32)
        hd = np.zeros(T, dtype=np.int32)  # dir=0 → facing right
        states = self._make_states([pos], [hd])

        fn = make_novel_obj_in_view_label_fn(obj_pos=[5, 2])
        labels = fn(states, T)

        assert labels.shape == (B, T)
        assert labels.sum().item() == T  # always in view

    def test_shape_multi_trajectory(self):
        """Output shape is (B, T) for B > 1."""
        B, T = 3, 6
        states = self._make_states(
            [np.tile(np.array([[1, 1]]), (T, 1)).astype(np.float32)] * B,
            [np.zeros(T, dtype=np.int32)] * B,
        )
        fn = make_novel_obj_in_view_label_fn(obj_pos=[3, 1])
        labels = fn(states, T)
        assert labels.shape == (B, T)
        assert labels.dtype == torch.int64

    def test_values_are_binary(self):
        """Labels must be 0 or 1 only."""
        B, T = 2, 8
        pos = np.random.randint(1, 10, size=(T, 2)).astype(np.float32)
        hd = np.random.randint(0, 4, size=T).astype(np.int32)
        states = self._make_states([pos, pos], [hd, hd])

        fn = make_novel_obj_in_view_label_fn(obj_pos=[5, 5])
        labels = fn(states, T)

        unique_vals = labels.unique().tolist()
        assert all(v in (0, 1) for v in unique_vals)


class TestCollectEvalTrajectoriesLabelFn:
    """Integration tests for label_fn hook in collect_eval_trajectories."""

    @pytest.mark.slow
    def test_label_fn_shape(self):
        """label_fn result should be stored as (B, T) in 'labels'."""
        seed(42)
        env = _get_env()
        pN = _get_pRNN(prnn_ckpt=PRNN_RAND_CKPT, device=DEVICE, env=env)
        agent = RandomActionAgent(env.action_space, np.array([0.15, 0.15, 0.6, 0.1]))

        T = 10
        num_walkable = int(get_walkable_mask(env).sum())
        B = num_walkable * 4

        def constant_label_fn(states, T):
            return torch.ones((len(states), T), dtype=torch.int64)

        config = EvalTrajectoryConfig(timesteps=T)
        result = collect_eval_trajectories(pN, agent, env, config, label_fn=constant_label_fn)

        assert result["labels"] is not None
        assert result["labels"].shape == (B, T)
        assert result["labels"].sum().item() == B * T  # all ones

    @pytest.mark.slow
    def test_no_label_fn_gives_none(self):
        """Without label_fn, 'labels' key should be None."""
        seed(42)
        env = _get_env()
        pN = _get_pRNN(prnn_ckpt=PRNN_RAND_CKPT, device=DEVICE, env=env)
        agent = RandomActionAgent(env.action_space, np.array([0.15, 0.15, 0.6, 0.1]))

        config = EvalTrajectoryConfig(timesteps=5)
        result = collect_eval_trajectories(pN, agent, env, config)

        assert result["labels"] is None
