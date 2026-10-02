"""A LayerNorm on the hidden state before the pixel readout, and nowhere else.

The 2026-10-01 prediction-quality check (questions repository, Q18's session log and the
discussion that followed) found the trained readout under-shooting the saturated colour
channels of landmarks by 0.23 even when the observation is the network's input - not
uncertainty, a readout that cannot reach them. The 2026-08-30 bias note already measured
why `h` is a poor basis for the readout: its mean moves every timestep and half of it is
exactly zero. This normalises what the readout sees and leaves the recurrent dynamics,
the state the policy reads and every analysis of `h` untouched.

Installed by wrapping the existing `outlayer` - `Sequential(LayerNorm, outlayer)` - so the
Linear keeps its tensors: the `W_out` / `b_out` aliases, their optimizer groups and the
checkpoint keys of the Linear all stay valid (the Linear's keys move from `outlayer.0.*`
to `outlayer.1.0.*`; a checkpoint saved by a normalised network carries them there, and
only such a checkpoint may be loaded into one). The norm's two parameters get their own
optimizer group at the UNSCALED learning rate, as the readout bias does, with no weight
decay. Must run before `load_pN` and before the capturable optimizer rebuild, which
preserves parameter groups.
"""

from __future__ import annotations

from torch import nn

GROUP_NAME = "ReadoutNorm"


def install_readout_norm(pN) -> nn.LayerNorm:
    """Wrap `pN.pRNN.outlayer` in a LayerNorm over the hidden state and register its
    parameters with the optimizer. Idempotent: a second call is refused."""
    rnn = pN.pRNN
    if isinstance(rnn.outlayer, nn.Sequential) and len(rnn.outlayer) == 2 and isinstance(rnn.outlayer[0], nn.LayerNorm):
        raise RuntimeError("readout norm already installed")
    hidden = pN.hidden_size
    norm = nn.LayerNorm(hidden)
    device = next(rnn.parameters()).device
    norm.to(device)
    rnn.outlayer = nn.Sequential(norm, rnn.outlayer)
    pN.optimizer.add_param_group({
        "params": list(norm.parameters()),
        "name": GROUP_NAME,
        "lr": pN.learningRate,
        "weight_decay": 0.0,
    })
    print(f"readout norm installed: LayerNorm({hidden}) before the pixel readout, group {GROUP_NAME!r} at lr {pN.learningRate}")
    return norm


def has_readout_norm(pN) -> bool:
    out = pN.pRNN.outlayer
    return isinstance(out, nn.Sequential) and len(out) == 2 and isinstance(out[0], nn.LayerNorm)


__all__ = ["GROUP_NAME", "has_readout_norm", "install_readout_norm"]
