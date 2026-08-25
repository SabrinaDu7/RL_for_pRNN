"""THROWAWAY: analysis of the orthogonal-pRNN-init grid. Reads only; collects nothing.

Input:  the JSON files written by throwaway/orthogonal_init_early_training.py,
        one per (arm, seed), named <arm>_s<seed>.json.
Output: a text table on stdout and, with --fig, one figure.

The comparison axis is the WORLD-MODEL GRADIENT STEP, not wall-clock and not
the update: every arm runs the identical schedule (128 world-model steps and
512 policy steps per update), so step index is matched by construction.

The verdict rule is fixed before looking: an arm is called a difference only if
its across-seed range does not overlap the baseline's across-seed range at the
same gradient step. Anything else is inside seed spread.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WM_CHECKPOINTS = (1_000, 2_000, 4_000, 8_000, 12_800)
WINDOW = 256  # trailing gradient steps averaged at each checkpoint


@dataclass(frozen=True)
class ArmRuns:
    """Every seed of one arm, already aligned on the gradient-step axis."""

    arm: str
    seeds: tuple[int, ...]
    wm_loss: np.ndarray  # (n_seeds, n_gradient_steps)
    grad_W: np.ndarray  # (n_seeds, n_updates)
    grad_W_in: np.ndarray
    grad_W_out: np.ndarray
    entropy: np.ndarray  # (n_seeds, n_updates)
    policy_loss: np.ndarray
    value_loss: np.ndarray
    ac_grad_norm: np.ndarray
    sRSA: np.ndarray  # (n_seeds,)
    SWdist: np.ndarray
    mean_SI: np.ndarray
    reinit: list[str]
    spectrum_before: dict
    spectrum_after: dict


def load(indir: Path) -> dict[str, ArmRuns]:
    runs: dict[str, list[dict]] = {}
    for f in sorted(indir.glob("*_s[0-9].json")):
        r = json.loads(f.read_text())
        runs.setdefault(r["arm"], []).append(r)

    out: dict[str, ArmRuns] = {}
    for arm, rs in runs.items():
        rs.sort(key=lambda r: r["seed"])
        n_wm = min(len(r["wm_loss_per_gradient_step"]) for r in rs)
        n_up = min(len(r["per_update"]["entropy"]) for r in rs)
        col = lambda key, n: np.array(  # noqa: E731
            [r["per_update"][key][:n] for r in rs], dtype=float
        )
        gcol = lambda key, n: np.array(  # noqa: E731
            [r["prnn_grad_norm_per_update"][key][:n] for r in rs], dtype=float
        )
        out[arm] = ArmRuns(
            arm=arm,
            seeds=tuple(r["seed"] for r in rs),
            wm_loss=np.array([r["wm_loss_per_gradient_step"][:n_wm] for r in rs], dtype=float),
            grad_W=gcol("W", n_up),
            grad_W_in=gcol("W_in", n_up),
            grad_W_out=gcol("W_out", n_up),
            entropy=col("entropy", n_up),
            policy_loss=col("policy_loss", n_up),
            value_loss=col("value_loss", n_up),
            ac_grad_norm=col("grad_norm", n_up),
            sRSA=np.array([r["spatial"]["sRSA"] for r in rs]),
            SWdist=np.array([r["spatial"]["SWdist"] for r in rs]),
            mean_SI=np.array([r["spatial"]["mean_SI"] for r in rs]),
            reinit=rs[0]["reinit"],
            spectrum_before=rs[0]["spectrum_before"],
            spectrum_after=rs[0]["spectrum_after"],
        )
    return out


def at_step(wm_loss: np.ndarray, step: int) -> np.ndarray:
    """Per-seed mean world-model loss over the WINDOW steps ending at `step`."""
    hi = min(step, wm_loss.shape[1])
    return wm_loss[:, max(0, hi - WINDOW) : hi].mean(axis=1)


def fmt(v: np.ndarray) -> str:
    return f"{v.mean():.5f} [{v.min():.5f},{v.max():.5f}]"


def separated(a: np.ndarray, b: np.ndarray) -> bool:
    """True when the two across-seed RANGES do not overlap."""
    return a.max() < b.min() or b.max() < a.min()


def report(arms: dict[str, ArmRuns]) -> None:
    base = arms["baseline"]
    order = ["baseline"] + [a for a in arms if a != "baseline"]

    print("=" * 100)
    print("WHAT WAS RE-INITIALISED, and what it did to the singular values (seed 1)")
    print("=" * 100)
    for a in order:
        r = arms[a]
        print(f"\n{a}: {r.reinit or 'nothing (untouched)'}")
        for w in ("W_in", "W", "W_out"):
            b, f_ = r.spectrum_before[w], r.spectrum_after[w]
            sr = (
                f" spectral_radius {b['spectral_radius']:.4f}->{f_['spectral_radius']:.4f}"
                if b["spectral_radius"] is not None
                else ""
            )
            print(
                f"  {w:>5} {tuple(b['shape'])}: singular mean "
                f"{b['singular_mean']:.4f}->{f_['singular_mean']:.4f}  sd "
                f"{b['singular_std']:.4f}->{f_['singular_std']:.4f}  range "
                f"[{b['singular_min']:.4f},{b['singular_max']:.4f}]->"
                f"[{f_['singular_min']:.4f},{f_['singular_max']:.4f}]{sr}"
            )

    print("\n" + "=" * 100)
    print(f"WORLD-MODEL LOSS at matched gradient steps (mean over {WINDOW} trailing steps)")
    print(f"cell = mean over seeds [min,max];  n = {[len(arms[a].seeds) for a in order]}")
    print("=" * 100)
    head = "arm".ljust(12) + "".join(f"{s:>26}" for s in WM_CHECKPOINTS)
    print(head)
    for a in order:
        row = a.ljust(12)
        for s in WM_CHECKPOINTS:
            row += f"{fmt(at_step(arms[a].wm_loss, s)):>26}"
        print(row)
    print("\nseparated from baseline (across-seed ranges disjoint)?")
    for a in order:
        if a == "baseline":
            continue
        marks = []
        for s in WM_CHECKPOINTS:
            x, y = at_step(arms[a].wm_loss, s), at_step(base.wm_loss, s)
            if not separated(x, y):
                marks.append("-")
            else:
                marks.append("BETTER" if x.mean() < y.mean() else "WORSE")
        print(f"  {a.ljust(12)} " + " ".join(f"{s}:{m}" for s, m in zip(WM_CHECKPOINTS, marks)))

    print("\n" + "=" * 100)
    print("pRNN GRADIENT NORM (last world-model step of each update), mean over update windows")
    print("=" * 100)
    windows = [(0, 10), (10, 30), (30, 60), (60, 100)]
    for wname, arr_name in [("W", "grad_W"), ("W_in", "grad_W_in"), ("W_out", "grad_W_out")]:
        print(f"\n|grad {wname}|")
        print("arm".ljust(12) + "".join(f"{f'upd {lo}-{hi}':>26}" for lo, hi in windows))
        for a in order:
            arr = getattr(arms[a], arr_name)
            row = a.ljust(12)
            for lo, hi in windows:
                row += f"{fmt(arr[:, lo : min(hi, arr.shape[1])].mean(axis=1)):>26}"
            print(row)

    print("\n" + "=" * 100)
    print("POLICY (UpdateLogs), mean over update windows")
    print("=" * 100)
    for label, arr_name in [
        ("entropy", "entropy"),
        ("policy_loss", "policy_loss"),
        ("value_loss", "value_loss"),
        ("ac grad_norm", "ac_grad_norm"),
    ]:
        print(f"\n{label}")
        print("arm".ljust(12) + "".join(f"{f'upd {lo}-{hi}':>26}" for lo, hi in windows))
        for a in order:
            arr = getattr(arms[a], arr_name)
            row = a.ljust(12)
            for lo, hi in windows:
                row += f"{fmt(arr[:, lo : min(hi, arr.shape[1])].mean(axis=1)):>26}"
            print(row)

    print("\n" + "=" * 100)
    print("SPATIAL REPRESENTATION at the end of the run (fixed probe seed, so eval noise is shared)")
    print("=" * 100)
    print("arm".ljust(12) + f"{'sRSA':>26}{'SWdist':>26}{'mean SI':>26}")
    for a in order:
        r = arms[a]
        print(a.ljust(12) + f"{fmt(r.sRSA):>26}{fmt(r.SWdist):>26}{fmt(r.mean_SI):>26}")
    print("\nseparated from baseline?")
    for a in order:
        if a == "baseline":
            continue
        r = arms[a]
        flags = [
            f"{n}:{'SEPARATED' if separated(getattr(r, n), getattr(base, n)) else '-'}"
            for n in ("sRSA", "SWdist", "mean_SI")
        ]
        print(f"  {a.ljust(12)} " + "  ".join(flags))


def figure(arms: dict[str, ArmRuns], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["baseline"] + [a for a in arms if a != "baseline"]
    colors = dict(zip(order, plt.cm.tab10.colors))
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

    smooth = 256
    ax = axes[0]
    for a in order:
        y = arms[a].wm_loss
        n = (y.shape[1] // smooth) * smooth
        yb = y[:, :n].reshape(y.shape[0], -1, smooth).mean(axis=2)
        x = np.arange(1, yb.shape[1] + 1) * smooth
        ax.plot(x, yb.mean(axis=0), label=f"{a} (n={len(arms[a].seeds)})", color=colors[a])
        ax.fill_between(x, yb.min(axis=0), yb.max(axis=0), alpha=0.18, color=colors[a])
    ax.set_xlabel("world-model gradient step")
    ax.set_ylabel(f"pRNN prediction loss (mean of {smooth} consecutive steps)")
    ax.set_title("CHANGED VARIABLE: which pRNN weight is orthogonally initialised\n"
                 "band = min-max across seeds")
    ax.set_yscale("log")
    ax.legend()

    ax = axes[1]
    for a in order:
        y = arms[a].grad_W
        n = (y.shape[1] // 5) * 5
        yb = y[:, :n].reshape(y.shape[0], -1, 5).mean(axis=2)
        x = np.arange(1, yb.shape[1] + 1) * 5
        ax.plot(x, yb.mean(axis=0), label=a, color=colors[a])
        ax.fill_between(x, yb.min(axis=0), yb.max(axis=0), alpha=0.18, color=colors[a])
    ax.set_xlabel("update (128 world-model gradient steps each)")
    ax.set_ylabel("|grad W| of the last world-model step in the update")
    ax.set_title("gradient reaching the RECURRENT matrix\n(the channel orthogonal init is meant to fix)")
    ax.set_yscale("log")
    ax.legend()

    ax = axes[2]
    for i, a in enumerate(order):
        v = arms[a].sRSA
        ax.bar(i, v.mean(), width=0.6, color=colors[a], label=a)
        ax.plot([i] * len(v), v, "k.", ms=9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("sRSA (on-policy, fixed probe)")
    ax.set_title("spatial representation at the end of the run\ndots = individual seeds")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nwrote {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True)
    p.add_argument("--fig", default=None)
    args = p.parse_args()
    arms = load(Path(args.indir))
    if not arms:
        raise SystemExit(f"no *_s<seed>.json under {args.indir}")
    report(arms)
    if args.fig:
        figure(arms, Path(args.fig))


if __name__ == "__main__":
    main()
