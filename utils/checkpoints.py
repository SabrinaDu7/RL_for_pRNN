from enum import Enum
from typing import Any, Optional
import torch
from RLutils import DEVICE

# Data classes
ACMODEL_STATUS = dict[str, Any] # Will follow StatusCkptKeys
class StatusCkptKeys(str, Enum):
    """String enums for checkpoint dictionary keys."""
    NUM_FRAMES = 'num_frames'
    UPDATE = 'update'
    MODEL_STATE = 'model_state'
    OPTIMIZER_STATE = 'optimizer_state'


# Functions
def load_acmodel_status(
    acmodel_status_ckpt: str, # Path to the checkpoint file (.pt)
    device: Optional[torch.device | str] = DEVICE
) -> ACMODEL_STATUS:
    """Load AC model checkpoint status from disk.

    Returns:
        Dictionary containing:
            - num_frames: Total number of training frames
            - update: Number of training updates
            - model_state: Model state_dict (if present)
            - optimizer_state: Optimizer state_dict (if present)
    """
    
    status: ACMODEL_STATUS = torch.load(
        acmodel_status_ckpt,
        map_location=device,
        weights_only=False
    )
    
    return status


def load_statedict_from_acmodel_status(
    receiver,
    status: ACMODEL_STATUS,
    status_key: StatusCkptKeys,
):
    if status_key in status:
        receiver.load_state_dict(status[status_key.value])
    else:
        raise KeyError(f"Status key '{status_key}' not found in checkpoint.")
