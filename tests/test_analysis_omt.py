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

from RLutils import make_env, get_pN, get_SR_acmodel, get_obss_preprocessor, ActorCriticAgent, seed, DEVICE
from utils import get_ckpt_env_vars, AgentInputType, AgentType

from scripts.analysis_OMT import (
    EvalTrajectoryConfig,
    EvalTrajectories,
    eval_mode,
    on_device,
    collect_eval_trajectories,
    save_eval_trajectories,
    load_eval_trajectories,
    get_walkable_mask,
    get_walkable_positions,
)


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
        cfg = EvalTrajectoryConfig(timesteps=100)
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
            "renders": None,
        }

        save_eval_trajectories(data, tmp_path / "out")

        assert (tmp_path / "out" / "trajectories.pt").exists()
        assert (tmp_path / "out" / "summary.parquet").exists()

        loaded = load_eval_trajectories(tmp_path / "out")
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
        config = EvalTrajectoryConfig(timesteps=T)
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
            timesteps=T, include_hidden_states=True
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
