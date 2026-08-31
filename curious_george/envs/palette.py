"""The tile vocabulary: every RGB value a 7x7 tile_size=1 observation can hold.

The CE prediction loss (`arch_prnn.loss = CE`) classifies each tile over this
alphabet, so it is a COMMITTED CONSTANT, never derived per run: a single room
carries only three of the four landmark colours, so a run-local derivation
would renumber the classes between configs and make checkpoints incomparable.

Measured over every Selected-room bank (both affordances) plus the default
single L-room - 1,053,696 tile observations, closed at these values
(2026-08-31, minigrid 22ef960). `tests/test_palette.py` regenerates the
measurement from live banks and diffs it against this table; `predCE` asserts
at runtime that no observation falls outside it. The OMT novel object
(FloorBright neon_green) is deliberately NOT here - it never appears in
training rooms, and an OMT-style env reaching a CE loss should fail loudly
until the vocabulary is extended on purpose.
"""

import torch

#: name -> the exact uint8 RGB a bank observation holds for that tile kind.
TILE_VOCABULARY: dict[str, tuple[int, int, int]] = {
    "floor": (76, 76, 76),
    "wall": (146, 146, 146),
    "agent": (135, 76, 76),
    "blue": (76, 76, 255),
    "green": (76, 255, 76),
    "red": (255, 76, 76),
    "yellow": (255, 255, 76),
}

TILE_CLASS_NAMES: tuple[str, ...] = tuple(TILE_VOCABULARY)


def vocab_tensor() -> torch.Tensor:
    """(C, 3) float32 in [0, 1], the exact dtype/scaling the pRNN's
    observation rows use (`flat_obs_rows`: uint8 tensor -> float32 / 255)."""
    return (
        torch.tensor(list(TILE_VOCABULARY.values()), dtype=torch.float32) / 255
    )


def class_render() -> torch.Tensor:
    """(C, 3) uint8 - class index -> displayable RGB, for argmax renders."""
    return torch.tensor(list(TILE_VOCABULARY.values()), dtype=torch.uint8)
