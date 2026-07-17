from enum import Enum
from typing import Any
import torch
import torch.nn as nn

# Data classes
ACMODEL_STATUS = dict[str, Any] # Will follow StatusCkptKeys
class StatusCkptKeys(str, Enum):
    """String enums for checkpoint dictionary keys."""
    NUM_FRAMES = 'num_frames'
    UPDATE = 'update'
    MODEL_STATE = 'model_state'
    OPTIMIZER_STATE = 'optimizer_state'

def load_statedict_from_acmodel_status(
    receiver,
    status: ACMODEL_STATUS,
    status_key: StatusCkptKeys,
    device: torch.device,
):
    if status_key in status:
        receiver.load_state_dict(status[status_key.value])
        if isinstance(receiver, nn.Module):
            receiver.to(device)
    else:
        raise KeyError(f"Status key '{status_key}' not found in checkpoint.")
