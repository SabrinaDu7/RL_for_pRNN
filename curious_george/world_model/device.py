"""Device / eval-mode context managers.

Generalized from scripts/analysis_OMT.py. These replace the ad-hoc
`.to("cpu")` / `.to(DEVICE)` toggling that used to be scattered across the
training and task code: callers that genuinely need modules on a device
(e.g. numpy-based analysis needs CPU) say so locally and the original
placement is always restored.

Accepts plain nn.Modules OR PredictiveNet-like objects (anything with a
`.pRNN` module). For a PredictiveNet, the recurrent hidden state
(`pN.state`) is moved together with the weights - moving only the module
leaves a stale-device state tensor behind and predict_single crashes.
"""

from contextlib import contextmanager

import torch
import torch.nn as nn


def _as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


def _module_of(target) -> nn.Module:
    return target.pRNN if hasattr(target, "pRNN") else target


def _move(target, device) -> None:
    _module_of(target).to(device)
    if hasattr(target, "state") and isinstance(target.state, torch.Tensor):
        target.state = target.state.to(device)


@contextmanager
def on_device(targets, device: torch.device | str):
    """Temporarily move module(s)/PredictiveNet(s) to `device`; restore on exit.

    The restore is ADDRESS-PRESERVING: the original parameter storages are held
    alive for the duration and the values are copied back into them, so every
    parameter ends at the `data_ptr()` it started with.

    That is not a micro-optimization, it is what makes this composable with
    `predNet.cuda_graph`. `Module.to()` REPLACES `param.data`, so a plain
    cuda->cpu->cuda round trip ends on a fresh allocation and leaves every
    captured CUDA graph writing to freed memory - a use-after-free that killed
    two cluster runs (`obs.direction`==184, 7 updates after an analysis event).
    The fingerprint in `_GraphWMTrainer` detects that and re-captures, but
    re-capturing into the memory the move just fragmented was itself implicated
    in a residual crash, so detection alone forced graphed runs to skip the
    spatial eval and the sample-trajectory figure entirely.

    Holding the original storages also means the accelerator memory is never
    released and re-acquired, so there is no fragmentation to re-capture into.
    The cost is the parameters staying resident while the copy runs on the other
    device - ~1.6 MB for the h=500 pRNN.

    `_GraphWMTrainer._invalidate_if_moved` is deliberately KEPT: any other path
    that reassigns `param.data` still has to be caught, and this only makes THIS
    path safe.
    """
    targets = _as_list(targets)
    originals = [next(_module_of(t).parameters()).device for t in targets]
    storages = [[p.data for p in _module_of(t).parameters()] for t in targets]
    for t in targets:
        _move(t, device)
    try:
        yield
    finally:
        for t, original, saved in zip(targets, originals, storages):
            _move(t, original)
            with torch.no_grad():
                for p, home in zip(_module_of(t).parameters(), saved):
                    if p.data.data_ptr() != home.data_ptr():
                        home.copy_(p.data)
                        p.data = home


@contextmanager
def eval_mode(targets, agent=None):
    """Temporarily set module(s) to eval() (and an ActorCriticAgent to argmax).

    Restores original training flags and the agent's argmax setting on exit.
    """
    modules = [_module_of(t) for t in _as_list(targets)]
    originals = [m.training for m in modules]
    original_argmax = None

    for m in modules:
        m.eval()
    if agent is not None and hasattr(agent, "argmax"):
        original_argmax = agent.argmax
        agent.argmax = True

    try:
        yield
    finally:
        for m, original in zip(modules, originals):
            if original:
                m.train()
        if agent is not None and hasattr(agent, "argmax"):
            agent.argmax = original_argmax
