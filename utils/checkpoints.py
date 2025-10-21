from enum import Enum

class StateCkptKeys(str, Enum):
    """String enums for checkpoint dictionary keys."""
    NUM_FRAMES = 'num_frames'
    UPDATE = 'update'
    MODEL_STATE = 'model_state'
    OPTIMIZER_STATE = 'optimizer_state'
