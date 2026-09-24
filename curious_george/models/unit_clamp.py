"""Hold a set of pRNN hidden units at fixed values through every recurrent step.

The ablate-and-retrain arm of the 2026-09-24 exploration (questions repository,
`docs/claude_logs/sessions/2026-09-24-overnight-ovc-function.md`): resume a finished run
with its object-vector cells held at their mean activation and keep training, so the
policy and the world model adapt to a state without that code.

One forward hook on the recurrent cell covers every path that steps the pRNN - the
batched rollout tracker (`BatchedSRTracker._run_cell` calls `pN.pRNN.rnn(...)`), the
curiosity forward, the world-model gradient step (`pN.trainStep` -> `pRNN.forward` ->
`thetaRNNLayer` -> the cell per timestep) and the evaluations. The cell returns
`(hy, (hy,))` - the emitted output and the carried state are the SAME tensor - so the hook
returns a clamped copy in both slots. A copy, not an in-place write: ReLU's backward reads
its output, and an in-place edit of it breaks autograd.

Capture-safe: the indices and values live on the device as buffers of `pN.pRNN`, so the
hook does no host-device copy and no data-dependent branch, and a CUDA-graph capture
records its kernels like any other. `torch.compile` of the layer (`train_prnn.compile
= "layer"`) is the one untested interaction; the resume launcher passes `--train-prnn.compile OFF`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

CLAMP_UNITS = "clamp_units"
CLAMP_VALUES = "clamp_values"


def load_clamp(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """`units` (int64 indices) and `values` (one float per unit) from an .npz."""
    with np.load(path) as file:
        units = np.asarray(file["units"], dtype=np.int64)
        values = np.asarray(file["values"], dtype=np.float32)
    if units.ndim != 1 or values.shape != units.shape:
        raise ValueError(f"{path}: `units` must be 1-D and `values` its shape, got {units.shape} / {values.shape}")
    if len(np.unique(units)) != len(units):
        raise ValueError(f"{path}: repeated unit indices")
    return units, values


def clamp_hook(owner: torch.nn.Module, units_name: str = CLAMP_UNITS, values_name: str = CLAMP_VALUES):
    """The forward hook: the cell's `(hy, (hy,))` with the units overwritten by the values.

    The buffers are read from `owner` at call time, not captured at install: `on_device`
    moves the module between CPU (serial evaluation) and CUDA, and buffers follow it."""

    def hook(_cell, _inputs, output):
        hy, _state = output
        units = getattr(owner, units_name)
        values = getattr(owner, values_name)
        out = hy.clone()
        out[..., units] = values.to(device=out.device, dtype=out.dtype)
        return out, (out,)

    return hook


def install_unit_clamp(pN, path: Path | str, device: torch.device) -> torch.utils.hooks.RemovableHandle:
    """Register the clamp buffers on `pN.pRNN` and the hook on its cell; returns the handle."""
    units, values = load_clamp(path)
    hidden = pN.hidden_size
    if units.min() < 0 or units.max() >= hidden:
        raise ValueError(f"clamp indices must lie in [0, {hidden}), got [{units.min()}, {units.max()}]")
    rnn = pN.pRNN
    rnn.register_buffer(CLAMP_UNITS, torch.as_tensor(units, device=device), persistent=False)
    rnn.register_buffer(CLAMP_VALUES, torch.as_tensor(values, device=device), persistent=False)
    handle = rnn.rnn.cell.register_forward_hook(clamp_hook(rnn))
    print(f"unit clamp installed: {len(units)} of {hidden} hidden units held at fixed values ({path})")
    return handle
