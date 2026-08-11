"""Lightweight stage timers for profiling the training loop.

Usage:
    from curious_george.utils.timing import timer

    with timer("collect/env_step"):
        ...

Disabled by default (near-zero overhead: one truthiness check). Enable with
`timer.enabled = True` (the perf benchmark does this) or the env var
CG_TIMING=1. With `timer.sync_device = True`, each stage boundary synchronizes
the selected CUDA/MPS device so work is attributed to the stage that queued
it. Only enable this for profiling because it serializes accelerator work.
"""

import os
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

from curious_george.utils.common import get_device


class StageTimer:
    def __init__(self):
        self.enabled = os.environ.get("CG_TIMING", "") == "1"
        self.sync_device = False
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    @staticmethod
    def _synchronize_accelerator() -> None:
        device_type = get_device().type
        if device_type == "cuda":
            torch.cuda.synchronize()
        elif device_type == "mps":
            torch.mps.synchronize()

    @contextmanager
    def __call__(self, name: str):
        if not self.enabled:
            yield
            return
        # Synchronization is profiling-only. It makes attribution precise but
        # perturbs throughput, especially around tiny recurrent operations.
        if self.sync_device and get_device().type in {"cuda", "mps"}:
            self._synchronize_accelerator()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_device and get_device().type in {"cuda", "mps"}:
                self._synchronize_accelerator()
            self.totals[name] += time.perf_counter() - t0
            self.counts[name] += 1

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()

    def report(self) -> dict[str, dict]:
        """{stage: {total_s, calls, mean_ms}} sorted by total desc."""
        return {
            k: {
                "total_s": round(self.totals[k], 4),
                "calls": self.counts[k],
                "mean_ms": round(1000 * self.totals[k] / max(self.counts[k], 1), 4),
            }
            for k in sorted(self.totals, key=self.totals.get, reverse=True)
        }

    def pretty(self) -> str:
        rows = [f"{'stage':<32}{'total_s':>10}{'calls':>8}{'mean_ms':>10}"]
        for k, v in self.report().items():
            rows.append(f"{k:<32}{v['total_s']:>10.3f}{v['calls']:>8}{v['mean_ms']:>10.3f}")
        return "\n".join(rows)


timer = StageTimer()
