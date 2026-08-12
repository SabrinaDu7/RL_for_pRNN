"""Compare pooled vs serial world-model training on LOSS PER GRADIENT STEP.

The question the batch sweep exists to answer: with `predNet.batched_wm`, one
update is one world-model gradient step, so `exp.num_envs` sets the whole
gradient-step budget (see curious_george/training/schedule.py). Pooling buys a
larger, lower-variance gradient but proportionally fewer steps. Which wins?

Wall-clock is deliberately NOT the axis here - `trainStep` is flat in batch on
this GPU, so per-second comparisons flatter large batches for a reason that has
nothing to do with learning. The reference curve is the finished serial B=8 run
on wandb, on its true `trial` (gradient step) axis.

    uv run python tests/perf/results/../compare_batch_learning.py \
        --sweep tests/perf/results/batch_learning_sweep.json

Prints a table; writes nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REF_RUN = "pRNN_curious_26-07-23-10-06-25"
CHECKPOINTS = (100, 200, 400, 800, 1600, 3200)


def _at(losses: np.ndarray, step: int, window: int = 50) -> float | None:
    """Loss averaged over the `window` gradient steps ending at `step`."""
    if len(losses) < step:
        return None
    return float(losses[max(0, step - window):step].mean())


def _reference(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    trial, loss = path / "ref_trial.npy", path / "ref_loss.npy"
    if not (trial.exists() and loss.exists()):
        return None
    return np.load(trial), np.load(loss)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="tests/perf/results/batch_learning_sweep.json")
    ap.add_argument("--ref-dir", default=None,
                    help="directory holding ref_trial.npy / ref_loss.npy from wandb")
    args = ap.parse_args()

    data = json.loads(Path(args.sweep).read_text())
    arms = data["arms"]

    print(f"device={data.get('device')}  gpu={data.get('gpu')}  seqdur={data.get('seqdur')}\n")
    header = f"{'arm':<14} {'WM batch':>9} {'grad steps':>11} " + \
             "".join(f"{f'@{c}':>10}" for c in CHECKPOINTS)
    print("LOSS PER GRADIENT STEP  (mean of the 50 steps ending at each mark)")
    print(header)
    print("-" * len(header))

    for a in arms:
        losses = np.asarray(a["losses"], dtype=float)
        cells = "".join(
            f"{v:>10.5f}" if (v := _at(losses, c)) is not None else f"{'-':>10}"
            for c in CHECKPOINTS
        )
        print(f"{a['arm']:<14} {a['wm_batch']:>9} {a['grad_steps']:>11} {cells}")

    ref = _reference(Path(args.ref_dir)) if args.ref_dir else None
    if ref is not None:
        trial, loss = ref
        cells = ""
        for c in CHECKPOINTS:
            m = trial <= c
            cells += f"{loss[m][-50:].mean():>10.5f}" if m.sum() >= 5 else f"{'-':>10}"
        print(f"{'wandb serial':<14} {1:>9} {int(trial.max()) + 1:>11} {cells}")
        print(f"\n  reference run: {REF_RUN} (serial B=8, full run)")
        print(f"  its final loss after {int(trial.max()) + 1} gradient steps: "
              f"{loss[-50:].mean():.6f}")

    print("\nCOST PER ARM")
    print(f"{'arm':<14} {'s/update':>10} {'grad steps/s':>14} {'FPS':>10} {'total s':>10}")
    for a in arms:
        print(f"{a['arm']:<14} {a['s_per_update_median']:>10.3f} "
              f"{a['grad_steps_per_sec']:>14.2f} {a['fps']:>10.0f} {a['total_s']:>10.1f}")


if __name__ == "__main__":
    main()
