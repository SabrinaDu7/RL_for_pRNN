"""The 2026-09-24 training variants: the unit clamp holds its units through the cell's
forward, and the bump penalty pays only for a forward that did not move."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from curious_george.configs import TrainPolicyCfg
from curious_george.envs.vector import FORWARD_ACTION, blocked_forward_rewards
from curious_george.models.prnn_adapter import FORWARD_IDX
from curious_george.models.unit_clamp import clamp_hook, load_clamp, resolve_clamp_path


class _Cell(torch.nn.Module):
    """The cell's contract: `(hy, (hy,))`, the same tensor twice."""

    def forward(self, x):
        hy = torch.relu(x)
        return hy, (hy,)


def test_the_hook_holds_the_units_in_both_outputs_and_keeps_autograd():
    cell = _Cell()
    units = torch.tensor([1, 3]); values = torch.tensor([0.25, 0.75])
    cell.register_buffer("clamp_units", units); cell.register_buffer("clamp_values", values)
    cell.register_forward_hook(clamp_hook(cell))
    x = torch.randn(5, 6, requires_grad=True)
    hy, (state,) = cell(x)
    assert torch.equal(hy, state)
    assert torch.allclose(hy[:, units], values.expand(5, -1))
    untouched = [i for i in range(6) if i not in (1, 3)]
    assert torch.equal(hy[:, untouched], torch.relu(x)[:, untouched])
    hy.sum().backward()  # the clone keeps ReLU's saved output intact
    assert x.grad is not None and torch.all(x.grad[:, units] == 0)


def test_load_clamp_refuses_bad_files(tmp_path):
    good = tmp_path / "good.npz"
    np.savez(good, units=np.array([4, 7]), values=np.array([0.1, 0.2]))
    units, values = load_clamp(good)
    assert units.tolist() == [4, 7] and values.dtype == np.float32
    bad = tmp_path / "bad.npz"
    np.savez(bad, units=np.array([4, 4]), values=np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="repeated"):
        load_clamp(bad)
    shape = tmp_path / "shape.npz"
    np.savez(shape, units=np.array([4, 7]), values=np.array([0.1]))
    with pytest.raises(ValueError, match="shape"):
        load_clamp(shape)


def test_bump_penalty_pays_only_for_a_forward_that_did_not_move():
    assert FORWARD_ACTION == FORWARD_IDX
    positions = torch.tensor([[2, 2], [2, 2], [2, 2], [5, 5]])
    next_rows = torch.tensor([[2, 2, 0], [3, 2, 0], [2, 2, 1], [5, 5, 3]])
    actions = torch.tensor([FORWARD_ACTION, FORWARD_ACTION, 1, FORWARD_ACTION])
    rewards = blocked_forward_rewards(next_rows, positions, actions, 0.5)
    assert rewards.tolist() == [-0.5, 0.0, 0.0, -0.5]
    assert rewards.dtype == torch.float32


def test_a_negative_bump_penalty_is_refused():
    with pytest.raises(ValueError, match="magnitude"):
        TrainPolicyCfg(bump_penalty=-0.1)
    assert TrainPolicyCfg(bump_penalty=0.01).bump_penalty == 0.01


def test_a_missing_clamp_file_falls_back_to_cg_clamp_dir(tmp_path, monkeypatch):
    here = tmp_path / "here"; there = tmp_path / "there"; there.mkdir()
    np.savez(there / "set.npz", units=np.array([1]), values=np.array([0.5]))
    monkeypatch.delenv("CG_CLAMP_DIR", raising=False)
    with pytest.raises(FileNotFoundError):
        resolve_clamp_path(here / "set.npz")
    monkeypatch.setenv("CG_CLAMP_DIR", str(there))
    assert resolve_clamp_path(here / "set.npz") == there / "set.npz"
    assert load_clamp(here / "set.npz")[0].tolist() == [1]
