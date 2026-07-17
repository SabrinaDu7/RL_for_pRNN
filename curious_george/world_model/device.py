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
    """Temporarily move module(s)/PredictiveNet(s) to `device`; restore on exit."""
    targets = _as_list(targets)
    originals = [next(_module_of(t).parameters()).device for t in targets]
    for t in targets:
        _move(t, device)
    try:
        yield
    finally:
        for t, original in zip(targets, originals):
            _move(t, original)


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
