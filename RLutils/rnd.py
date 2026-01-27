"""Random Network Distillation for curiosity-driven exploration."""

import torch
import torch.nn as nn


class RandomFeatureNetwork(nn.Module):
    """Fixed random network for RND-style curiosity rewards.

    Projects observations into a random feature space where prediction
    error provides a novelty signal. Weights are frozen after initialization.

    Args:
        input_dim: Dimension of input observations
        output_dim: Dimension of random feature space (default: 128)
    """

    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        # Freeze weights - this network should never be trained
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
