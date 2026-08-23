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

    The restore is ADDRESS-PRESERVING for `param.data`, `param.grad` AND every
    registered BUFFER:
    the original storages are held alive for the duration and the values are
    copied back into them, so every parameter and every gradient ends at the
    `data_ptr()` it started with.

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

    `.grad` matters as much as `.data` and is easy to miss, because missing it
    fails SILENTLY rather than crashing. `Module._apply` moves gradients too, so
    restoring only `.data` leaves a captured graph writing its gradients to, and
    its captured optimizer step reading them from, the OLD gradient storage,
    while `param.grad` points somewhere else entirely. The forward pass still
    reads the correct `.data`, so the prediction loss keeps falling normally and
    nothing looks wrong - but the optimizer consumes stranded memory and the
    representation degrades. Measured: a graphed run whose spatial evals ran
    every ~200k steps held sRSA at the untrained floor (~0.02) for a whole
    20.48M-step run while its eager control reached 0.52, with the pRNN loss
    trajectories of the two IDENTICAL to three decimal places.

    `_GraphWMTrainer._invalidate_if_moved` is deliberately KEPT: any other path
    that reassigns `param.data` still has to be caught, and this only makes THIS
    path safe.
    """
    targets = _as_list(targets)
    originals = [next(_module_of(t).parameters()).device for t in targets]

    # `.data` and `.grad` need OPPOSITE treatment, because `Module._apply`
    # treats them differently: it REPLACES `param.data` with a new tensor (so
    # holding a reference keeps the original storage alive and at its address),
    # but it MUTATES `param.grad` IN PLACE (same Python object, swapped
    # storage - so a reference follows the gradient to the other device and
    # preserves nothing). Gradients are therefore DETACHED from the module
    # before the move, which is what keeps `_apply` away from them, and put
    # back afterwards. The eval this wraps is inference and neither reads nor
    # writes gradients.
    saved = []
    for t in targets:
        m = _module_of(t)
        rows = []
        for p in m.parameters():
            rows.append((p.data, p.grad))
            p.grad = None
        # BUFFERS TOO. `thRNN_5win` multiplies its inputs, actions and targets
        # by the registered phase masks `inMask_f` / `actMask_f` / `outMask_f`
        # (Architectures.clip_mask), and `Module._apply` relocates buffers
        # exactly as it relocates `param.data`. Restoring only the parameters
        # left a captured graph multiplying by a STALE mask address: the loss
        # dropped ~6x on identical data because the inputs were being scaled by
        # whatever now occupied that memory, and the run trained on nothing.
        bufs = [(name, b) for name, b in m.named_buffers() if b is not None]
        saved.append((rows, bufs))

    for t in targets:
        _move(t, device)
    try:
        yield
    finally:
        for t, original, (rows, bufs) in zip(targets, originals, saved):
            m = _module_of(t)
            _move(t, original)
            with torch.no_grad():
                for p, (home, grad_home) in zip(m.parameters(), rows):
                    if p.data.data_ptr() != home.data_ptr():
                        home.copy_(p.data)
                        p.data = home
                    p.grad = grad_home
                current = dict(m.named_buffers())
                for name, home in bufs:
                    cur = current.get(name)
                    if cur is None or cur.data_ptr() == home.data_ptr():
                        continue
                    home.copy_(cur)
                    owner, _, leaf = name.rpartition(".")
                    (m.get_submodule(owner) if owner else m)._buffers[leaf] = home


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
