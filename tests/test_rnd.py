import pytest
import torch
from RLutils.rnd import RandomFeatureNetwork


def test_rnd_network_frozen():
    """Test that RND network weights are frozen."""
    net = RandomFeatureNetwork(input_dim=64, output_dim=32)
    for p in net.parameters():
        assert not p.requires_grad


def test_rnd_network_output_shape():
    """Test RND network output dimensions."""
    net = RandomFeatureNetwork(input_dim=64, output_dim=32)
    x = torch.randn(10, 64)
    out = net(x)
    assert out.shape == (10, 32)


def test_rnd_network_deterministic():
    """Test that same input produces same output (no randomness in forward)."""
    net = RandomFeatureNetwork(input_dim=64, output_dim=32)
    x = torch.randn(5, 64)
    out1 = net(x)
    out2 = net(x)
    assert torch.allclose(out1, out2)
