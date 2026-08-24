"""Helpers shared by every CUDA-graph capture in the tree.

One home for the two process-global toggles a capture has to flip, so the
reason each exists is written once rather than re-derived at each call site.
"""

from contextlib import contextmanager

import torch


@contextmanager
def no_dist_validation():
    """Distribution arg-checking off, restored afterwards.

    `torch.distributions.Categorical` validates its arguments by default,
    which runs `if not valid.all()` on a device tensor. That is a HOST SYNC,
    and a host sync during capture is illegal - it is the single reason a
    region containing a policy forward would not capture at all.

    `Distribution.set_default_validate_args` is process-global, so this is
    scoped as tightly as possible: capture only, never the eager path a
    non-graphed run takes. Replay executes no Python, so nothing is skipped at
    run time that would otherwise have run. Measured cost of the check in
    eager: none (3.558 ms validated vs 3.752 unvalidated, i.e. noise).
    """
    prev = torch.distributions.Distribution._validate_args
    torch.distributions.Distribution.set_default_validate_args(False)
    try:
        yield
    finally:
        torch.distributions.Distribution.set_default_validate_args(prev)
