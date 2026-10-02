"""The readout norm wraps the pixel readout, keeps the Linear's tensors and gives the norm
its own optimizer group; off by default, so untouched networks keep their keys."""

from __future__ import annotations

import torch
from torch import nn

from curious_george.configs import ArchPrnnCfg
from curious_george.models.readout_norm import GROUP_NAME, has_readout_norm, install_readout_norm


class _RNN(nn.Module):
    def __init__(self, hidden: int, out: int):
        super().__init__()
        proj = nn.Linear(hidden, out, bias=False)
        proj.bias = nn.Parameter(torch.zeros(out))
        self.outlayer = nn.Sequential(proj, nn.Sigmoid())
        self.out_proj = proj
        self.W_out = proj.weight
        self.b_out = proj.bias


class _Net:
    def __init__(self, hidden: int = 16, out: int = 6):
        self.hidden_size = hidden
        self.learningRate = 3e-3
        self.pRNN = _RNN(hidden, out)
        self.optimizer = torch.optim.RMSprop([
            {"params": [self.pRNN.W_out], "name": "OutputWeights", "lr": 1e-4, "weight_decay": 1e-7},
            {"params": [self.pRNN.b_out], "name": "OutputBias", "lr": 3e-3, "weight_decay": 0.0},
        ])


def test_off_by_default():
    assert ArchPrnnCfg().readout_norm is False


def test_install_wraps_the_readout_and_registers_the_norm():
    net = _Net()
    before = dict(net.pRNN.state_dict())
    norm = install_readout_norm(net)
    assert has_readout_norm(net)
    assert net.pRNN.outlayer[0] is norm and net.pRNN.outlayer[1][0] is net.pRNN.out_proj
    assert net.pRNN.W_out is net.pRNN.out_proj.weight and net.pRNN.b_out is net.pRNN.out_proj.bias
    keys = set(net.pRNN.state_dict())
    assert {"outlayer.0.weight", "outlayer.0.bias", "outlayer.1.0.weight", "outlayer.1.0.bias", "W_out", "b_out"} <= keys
    assert torch.equal(net.pRNN.state_dict()["outlayer.1.0.weight"], before["outlayer.0.weight"])
    group = [g for g in net.optimizer.param_groups if g.get("name") == GROUP_NAME]
    assert len(group) == 1 and len(group[0]["params"]) == 2 and group[0]["weight_decay"] == 0.0 and group[0]["lr"] == 3e-3
    h = torch.randn(5, 16) * 10 + 3
    y = net.pRNN.outlayer(h)
    assert y.shape == (5, 6) and torch.all(y > 0) and torch.all(y < 1)
    # the norm's parameters train through the readout
    y.sum().backward()
    assert norm.weight.grad is not None and norm.bias.grad is not None


def test_second_install_is_refused():
    net = _Net()
    install_readout_norm(net)
    try:
        install_readout_norm(net)
    except RuntimeError as e:
        assert "already" in str(e)
    else:
        raise AssertionError("a second install must be refused")
