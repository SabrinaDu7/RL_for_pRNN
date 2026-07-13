"""Compare learning curves across the Phase C num_envs sweep (bsweep-B*-s*).

Fetches wandb runs whose group starts with `bsweep-B`, bins each metric over
frames, and reports per-B mean +/- across-seed std per bin, plus a simple
overlap verdict (B>1 bins within B=1's across-seed band). Optionally saves a
figure.

Usage:
    uv run python scripts/analysis_bsweep.py [--prefix bsweep-B] [--fig out.png]
"""

import argparse
import re

import numpy as np
import wandb

METRICS = [
    "cur_reward_mean",
    "cur_reward_max",
    "policy_loss",
    "value_loss",
    "policy_entropy",
    "loc_entropy",
    "advantages_max",
    "advantages_min",
]
NBINS = 12


def fetch(prefix: str, entity: str, project: str) -> dict:
    """{(B, seed): history DataFrame} for finished/running sweep runs."""
    api = wandb.Api(timeout=59)
    runs = api.runs(f"{entity}/{project}")
    out = {}
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)-s(\d+)_")
    for r in runs:
        m = pat.match(r.name)
        if not m:
            continue
        B, seed = int(m.group(1)), int(m.group(2))
        h = r.history(keys=["frames"] + METRICS, samples=4000, pandas=True)
        if len(h):
            out[(B, seed)] = h.sort_values("frames")
            print(f"  fetched {r.name}: {len(h)} rows, frames <= {h['frames'].max():.0f}")
    return out


def binned(h, metric: str, edges: np.ndarray) -> np.ndarray:
    vals = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        rows = h[(h["frames"] >= edges[i]) & (h["frames"] < edges[i + 1])][metric]
        if len(rows):
            vals[i] = rows.mean()
    return vals


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="bsweep-B")
    p.add_argument("--entity", default="blake-richards")
    p.add_argument("--project", default="curious-george")
    p.add_argument("--fig", default=None)
    args = p.parse_args()

    hs = fetch(args.prefix, args.entity, args.project)
    if not hs:
        print("no runs found")
        return
    Bs = sorted({b for b, _ in hs})
    fmax = min(h["frames"].max() for h in hs.values())  # common horizon
    edges = np.linspace(0, fmax, NBINS + 1)
    print(f"\ncommon horizon: {fmax:.0f} frames, {NBINS} bins; B values: {Bs}")

    curves = {}  # (B, metric) -> (mean, std) arrays over bins
    for metric in METRICS:
        print(f"\n== {metric}")
        for B in Bs:
            per_seed = np.array(
                [binned(h, metric, edges) for (b, _), h in sorted(hs.items()) if b == B]
            )
            mean, std = np.nanmean(per_seed, axis=0), np.nanstd(per_seed, axis=0)
            curves[(B, metric)] = (mean, std)
            print(f"  B={B}: " + " ".join(f"{m:+.3g}" for m in mean))
        if 1 in Bs:
            m1, s1 = curves[(1, metric)]
            for B in Bs:
                if B == 1:
                    continue
                mB, _ = curves[(B, metric)]
                inside = np.abs(mB - m1) <= 2 * np.maximum(s1, 1e-12)
                frac = np.nanmean(inside.astype(float))
                print(f"  B={B} within 2sd of B=1 band: {frac:.0%} of bins")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ncols = 2
        nrows = (len(METRICS) + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3 * nrows))
        x = (edges[:-1] + edges[1:]) / 2
        for ax, metric in zip(axes.flat, METRICS):
            for B in Bs:
                mean, std = curves[(B, metric)]
                ax.plot(x, mean, label=f"B={B}")
                ax.fill_between(x, mean - std, mean + std, alpha=0.2)
            ax.set_title(metric)
            ax.set_xlabel("frames")
        axes.flat[0].legend()
        fig.tight_layout()
        fig.savefig(args.fig, dpi=150)
        print(f"\nwrote {args.fig}")


if __name__ == "__main__":
    main()
