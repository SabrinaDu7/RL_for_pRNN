"""Prediction-error decomposition: by tile class, and by bump events.

ANALYSIS ONLY - takes tensors a probe already produced (the adapter's
`episode_prediction_rows` or any (pred, target) pair in observation rows),
collects nothing, and returns small typed results. Two questions it answers:

- WHICH TILES carry the error? Under pale-MSE the landmark share was buried
  (the 2026-08-30 rendering line's whole motivation); this measures the share
  directly, per `envs/palette.py` class.
- Does the model know AFFORDANCE? In an impassable room a refused `forward`
  (the agent bumps an object or wall and the observation stays put) is
  informative exactly when the model has NOT bound the room's layout; the
  bump-vs-free error contrast measures that binding.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int

from curious_george.envs.palette import TILE_CLASS_NAMES, vocab_tensor


@dataclass(frozen=True)
class ClassErrors:
    """Per-tile-class error shares over a probe. Shares sum to 1 (up to nan
    classes that never appear)."""

    class_names: tuple[str, ...]
    tile_share: Float[torch.Tensor, "C"]     # fraction of tiles of each class
    error_share: Float[torch.Tensor, "C"]    # fraction of TOTAL error carried
    mean_error: Float[torch.Tensor, "C"]     # mean per-tile error within class
    total_error: float


def per_tile_errors(
    pred_rows: Float[torch.Tensor, "T X"],
    target_rows: Float[torch.Tensor, "T 147"],
    *,
    ce: bool,
) -> tuple[Float[torch.Tensor, "T 49"], Int[torch.Tensor, "T 49"]]:
    """(per-tile error, per-tile target class) for either loss.

    MSE: summed squared pixel error per tile. CE: per-tile surprisal in nats
    (pred_rows are logits, X = 49 * C). Targets are exact-vocab lookups either
    way - a target outside the palette asserts loudly.
    """
    vocab = vocab_tensor().to(target_rows.device)
    n_classes, n_channels = vocab.shape
    n_tiles = target_rows.shape[-1] // n_channels
    pixels = target_rows.reshape(-1, n_tiles, n_channels)
    dist = (pixels.unsqueeze(-2) - vocab).abs().sum(-1)
    mindist, classes = dist.min(-1)
    assert float(mindist.max()) < 1e-3, "target tile outside the committed vocabulary"

    if ce:
        logits = pred_rows.reshape(-1, n_tiles, n_classes)
        logp = torch.log_softmax(logits, dim=-1)
        errors = -logp.gather(-1, classes.unsqueeze(-1)).squeeze(-1)
    else:
        pred = pred_rows.reshape(-1, n_tiles, n_channels)
        errors = ((pred - pixels) ** 2).sum(-1)
    return errors, classes


def decompose_by_class(
    errors: Float[torch.Tensor, "T 49"],
    classes: Int[torch.Tensor, "T 49"],
) -> ClassErrors:
    n = len(TILE_CLASS_NAMES)
    tile_share = torch.zeros(n)
    error_share = torch.zeros(n)
    mean_error = torch.full((n,), float("nan"))
    total = float(errors.sum())
    for c in range(n):
        mask = classes == c
        count = int(mask.sum())
        tile_share[c] = count / classes.numel()
        if count:
            class_error = float(errors[mask].sum())
            error_share[c] = class_error / total if total else 0.0
            mean_error[c] = class_error / count
    return ClassErrors(
        class_names=TILE_CLASS_NAMES,
        tile_share=tile_share,
        error_share=error_share,
        mean_error=mean_error,
        total_error=total,
    )


@dataclass(frozen=True)
class BumpContrast:
    """Prediction error at refused-forward steps vs free moves.

    `bump_minus_free > 0` means bumps still surprise the model - it has not
    bound the room's affordance; near 0 (with enough bumps) means it has."""

    n_bumps: int
    n_free: int
    bump_mean_error: float
    free_mean_error: float

    @property
    def bump_minus_free(self) -> float:
        return self.bump_mean_error - self.free_mean_error


FORWARD = 2  # the SpeedHD forward action index (prnn_adapter.FORWARD_IDX)


def bump_contrast(
    step_errors: Float[torch.Tensor, "T"],
    actions: Int[torch.Tensor, "T"],
    positions: Int[torch.Tensor, "T+1 2"],
) -> BumpContrast:
    """`step_errors[t]` must be the error attributed to action t (the
    `next_obs` reward alignment); positions are pre-action, so action t moves
    positions[t] -> positions[t+1]."""
    moved = (positions[1:] != positions[:-1]).any(dim=-1)
    fwd = actions == FORWARD
    bumps = fwd & ~moved
    free = fwd & moved
    return BumpContrast(
        n_bumps=int(bumps.sum()),
        n_free=int(free.sum()),
        bump_mean_error=float(step_errors[bumps].mean()) if bumps.any() else float("nan"),
        free_mean_error=float(step_errors[free].mean()) if free.any() else float("nan"),
    )
