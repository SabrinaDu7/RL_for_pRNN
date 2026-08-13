"""Does the current stack reproduce the reference run's learning curve?

Overlays the pRNN loss of a new run against `pRNN_curious_26-07-23-10-06-25`,
the finished run every checkpoint and result in this project descends from,
on a GRADIENT-STEP axis. That axis matters: the reference logs one point per
`trainStep`, so its wandb `trial` counter is the gradient step, and comparing
on wall-clock or on frames would flatter or punish a config for reasons that
have nothing to do with learning.

Both runs are serial world-model training (one gradient step per episode
segment), so the curves are directly comparable. The new run additionally uses
the device-resident environment, which is trajectory-exact
(tests/test_device_collector.py), so any divergence is a real regression rather
than an expected difference.

    uv run python scripts/verify_figure.py --run verify-lroom-0812
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REFERENCE = "pRNN_curious_26-07-23-10-06-25"
PROJECT = "blake-richards/curious-george"
OUT = Path("outputs/moser")


def curve(run_name: str):
    """(gradient step, pRNN loss) for a run, newest wandb match wins."""
    import wandb

    api = wandb.Api()
    try:
        run = api.run(f"{PROJECT}/{run_name}")
    except Exception:
        matches = [r for r in api.runs(PROJECT) if r.name == run_name or run_name in r.name]
        if not matches:
            raise SystemExit(f"no wandb run matching {run_name!r} in {PROJECT}")
        run = sorted(matches, key=lambda r: r.created_at)[-1]
    df = run.history(keys=["trial", "pRNN loss"], samples=100000, pandas=True)
    df = df.dropna(subset=["pRNN loss"])
    return (df["trial"].to_numpy(dtype=float),
            df["pRNN loss"].to_numpy(dtype=float), run.name)


def smooth(y, w=50):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="wandb name of the new run")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    xr, yr, nr = curve(REFERENCE)
    xn, yn, nn = curve(args.run)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, logy in zip(axes, (False, True)):
        ax.plot(xr[: len(smooth(yr))], smooth(yr), lw=1.6, c="#444",
                label=f"reference: {nr}")
        ax.plot(xn[: len(smooth(yn))], smooth(yn), lw=1.6, c="#c0392b",
                label=f"new: {nn}")
        ax.set_xlabel("world-model gradient step")
        ax.set_ylabel("pRNN loss (50-step moving average)")
        if logy:
            ax.set_yscale("log"); ax.set_title("log scale")
        else:
            ax.set_xlim(0, min(xn.max(), xr.max()) * 1.02)
            ax.set_title("matched range")
        ax.legend(fontsize=8)
    fig.suptitle("Verification: does the current stack reproduce the reference learning curve?")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_verify_loss.png", dpi=170, bbox_inches="tight")
    print("wrote fig_verify_loss.png")

    print(f"\n{'grad step':>10} {'reference':>12} {'new':>12} {'ratio':>8}")
    for t in (500, 1000, 2500, 5000, 10000, 12000):
        mr, mn = xr <= t, xn <= t
        if mr.sum() < 5 or mn.sum() < 5:
            continue
        a, b = yr[mr][-50:].mean(), yn[mn][-50:].mean()
        print(f"{t:>10} {a:>12.6f} {b:>12.6f} {b / a:>8.3f}")


if __name__ == "__main__":
    main()
