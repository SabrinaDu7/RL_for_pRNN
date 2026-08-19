"""predNet.cuda_graph must skip the model-moving periodic diagnostics.

A captured CUDA graph writes to the parameter addresses it saw at capture
time; the periodic plotSampleTrajectory + spatial eval move the pRNN off
device (`on_device([pN, ...], "cpu")`), and the allocation churn inside the
eval fragments memory so a re-captured graph's pool aliases live tensors - a
use-after-free that crashed cluster runs (reproduced 2026-07-22: crash at
~update 427 with the swaps, clean 1200 updates with them skipped). This guard
is the fix; no GPU needed, so it runs on CPU CI where the long soak cannot.

The original guard (5753a7e) warned once via a module-global flag and had a
third test for it. That global is gone: `run_training` evaluates the guard
ONCE before the update loop and reuses the result, so warn-once is structural
rather than stateful. The flag was also process-global, so a second run in the
same interpreter (tests, sweeps) silently skipped the warning - the reason the
original test needed a `_reset_warn()` helper.
"""

from omegaconf import OmegaConf

import curious_george.training.loop as loop


def test_guard_skips_under_cuda_graph():
    cfg = OmegaConf.create({"predNet": {"cuda_graph": True}})
    assert loop._skip_model_move_diag(cfg) is True


def test_guard_allows_when_cuda_graph_off():
    for pred in ({"cuda_graph": False}, {}):  # off, and key absent
        cfg = OmegaConf.create({"predNet": pred})
        assert loop._skip_model_move_diag(cfg) is False
