from enum import Enum
from typing import Any
import torch
import torch.nn as nn

# Data classes
ACMODEL_STATUS = dict[str, Any] # Will follow StatusCkptKeys
class StatusCkptKeys(str, Enum):
    """String enums for checkpoint dictionary keys.

    OPTIMIZER_STATE is the AC model's optimizer (a 1-group Adam) in EVERY
    writer. The pRNN's own optimizer (a 4-group RMSprop) has its own key -
    they are not interchangeable, and setup_algo loads OPTIMIZER_STATE into
    the AC Adam, so writing the pRNN's state under that key produces a
    param-group-count error on reload. See load_prnn_optimizer_state.
    """
    NUM_FRAMES = 'num_frames'
    UPDATE = 'update'
    MODEL_STATE = 'model_state'
    OPTIMIZER_STATE = 'optimizer_state'
    #: Lifetime visit counts of the count-bonus agent (train_policy.k_count).
    #: Training state: without it a resumed run restarts novelty from zero.
    COUNT_VISITS = 'count_visits'
    PRNN_OPTIMIZER_STATE = 'prnn_optimizer_state'

def status_optimizer_matches(receiver, state: dict) -> bool:
    """True when `state` has the same number of param groups as `receiver`.

    Cheap guard against loading the wrong optimizer's state: the AC Adam has
    one param group, the pRNN RMSprop has four.
    """
    return len(state.get("param_groups", [])) == len(receiver.param_groups)


def load_prnn_optimizer_state(
    pn_optimizer, status: ACMODEL_STATUS, *, strict: bool = False
) -> bool:
    """Restore the pRNN optimizer from a status dict; return whether it loaded.

    Reads PRNN_OPTIMIZER_STATE. Also accepts OPTIMIZER_STATE when its shape
    matches the pRNN optimizer, which is how checkpoints written before
    2026-07-30 stored it (see StatusCkptKeys).
    """
    state = status.get(StatusCkptKeys.PRNN_OPTIMIZER_STATE.value)
    if state is None:
        legacy = status.get(StatusCkptKeys.OPTIMIZER_STATE.value)
        if legacy is not None and status_optimizer_matches(pn_optimizer, legacy):
            state = legacy  # pre-2026-07-30 layout
    if state is None:
        if strict:
            raise KeyError(
                f"No '{StatusCkptKeys.PRNN_OPTIMIZER_STATE.value}' in status checkpoint."
            )
        return False
    pn_optimizer.load_state_dict(state)
    return True


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
