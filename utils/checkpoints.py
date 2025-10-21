from enum import Enum
from typing import Any, Optional, Mapping
import torch
from RLutils import DEVICE, ACModel

# Data classes
class StateCkptKeys(str, Enum):
    """String enums for checkpoint dictionary keys."""
    NUM_FRAMES = 'num_frames'
    UPDATE = 'update'
    MODEL_STATE = 'model_state'
    OPTIMIZER_STATE = 'optimizer_state'


ACMODEL_STATUS = dict[str, Any]


# Functions
def load_acmodel_status(
    acmodel_status_ckpt: str,
    device: Optional[torch.device | str] = DEVICE
) -> ACMODEL_STATUS:
    """Load AC model checkpoint status from disk.
    
    Args:
        acmodel_status_ckpt: Path to the checkpoint file (.pt)
        device: Device to load tensors to (e.g., 'cpu', 'cuda', or torch.device). 
    
    Returns:
        Dictionary containing:
            - num_frames: Total number of training frames
            - update: Number of training updates
            - model_state: Model state_dict (if present)
            - optimizer_state: Optimizer state_dict (if present)
    
    Example:
        >>> status = load_acmodel_status("/path/to/status.pt")
        >>> if StateCkptKeys.MODEL_STATE in status:
        ...     acmodel.load_state_dict(status[StateCkptKeys.MODEL_STATE])
        >>> if StateCkptKeys.OPTIMIZER_STATE in status:
        ...     optimizer.load_state_dict(status[StateCkptKeys.OPTIMIZER_STATE])
    """
    
    status: ACMODEL_STATUS = torch.load(
        acmodel_status_ckpt,
        map_location=device,
        weights_only=False
    )
    
    return status


def load_acmodel(
    acmodel: ACModel,
    acmodel_status_ckpt: str,
    device: Optional[torch.device | str] = DEVICE
):
    """Load ACModel into acmodel."""

    status = load_acmodel_status(acmodel_status_ckpt, device=device)

    if StateCkptKeys.MODEL_STATE in status:
        acmodel.load_state_dict(status[StateCkptKeys.MODEL_STATE])
    
    raise KeyError(
        f"Checkpoint at '{acmodel_status_ckpt}' does not contain a model state dict. "
        f"Available keys: {list(status.keys())}"
    )

def load_acmodel_optimizer(
    optimizer: torch.optim.Optimizer,
    acmodel_status_ckpt: str,
    device: Optional[torch.device | str] = DEVICE
):
    """Load ACModel into acmodel."""

    status = load_acmodel_status(acmodel_status_ckpt, device=device)

    if StateCkptKeys.OPTIMIZER_STATE in status:
        optimizer.load_state_dict(status[StateCkptKeys.OPTIMIZER_STATE])
    
    raise KeyError(
        f"Checkpoint at '{acmodel_status_ckpt}' does not contain a optimizer state dict. "
        f"Available keys: {list(status.keys())}"
    )
