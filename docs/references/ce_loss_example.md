# Example python script for CE loss of a predictor accomplishing similar function as pRNN

```
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor


def logits_to_probs(logits: Tensor) -> NDArray[np.float32]:
    """Convert model logits to softmax probabilities."""
    assert logits.dim() in (3, 4)
    return F.softmax(logits, dim=-3).cpu().numpy().astype(np.float32)


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def init_orthogonal(module: nn.Module, gain: float = math.sqrt(2)) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            bias = m.bias
            if bias is not None:
                nn.init.zeros_(bias)
```
