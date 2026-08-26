"""Compare a run's logged metrics against a reference run, on matched env steps.

The question this answers is the only one that decides whether an optimization
was free: did the learning change? Wall-clock lives in the perf tooling; this
reads what wandb actually recorded and puts the two curves side by side.

Matching is on **environment steps**, never on raw `_step` or `trial`. Those
count updates and trajectories, which mean different amounts of experience at
different rollout shapes - a num_envs=1 reference and a num_envs=128 arm are
128x apart on `_step` and identical on env steps.

Getting that axis is not as simple as reading `frames` alongside the metric.
`sRSA_onPolicy`, `SWdist_onPolicy`, `mean SI_onPolicy` and `pRNN loss` are
logged BY PRNN in its own `wandb.log()` calls (see
`curious_george/training/logging.py::log_spatial`), on different wandb steps
from the update log that carries `frames`. Asking for both keys in one row
therefore returns NOTHING for exactly the metrics this comparison is about.
So each metric is fetched against `_step`, and `_step` is converted to env
steps through the run's own (`_step`, `frames`) series.

Because the metrics are noisy (an untrained net scores sRSA 0.062 +/- 0.040,
and SWdist has a measured 27.5% coefficient of variation), a single matched
point proves nothing. This reports, per metric, the reference's own
adjacent-sample spread as the error band, and flags a difference only when it
exceeds that band.

    uv run python -m curious_george.check.wandb_compare \\
        --reference pRNN_curious_26-07-08-16-04-37 \\
        --run <run-id-or-name> [--metric sRSA_onPolicy ...]

Writes $RL_STORAGE/summary/wandb_compare[_<tag>].json and prints a table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import wandb

from curious_george.log_and_store.storage import get_storage_dir

ENTITY, PROJECT = "blake-richards", "curious-george"
STEP_KEY = "frames"

# The metrics that decide whether a speed change was free. Off-policy keys are
# included but a reference with offpolicy_prnn_eval=false simply will not have
# them, which the report states rather than silently skipping.
DEFAULT_METRICS = (
    "sRSA_onPolicy", "SWdist_onPolicy", "mean SI_onPolicy",
    "sRSA_offPolicy", "SWdist_offPolicy",
    "pRNN loss", "MI_policy", "loc_entropy", "policy_entropy",
    "return_mean", "cur_reward_mean",
)


def resolve(api, ident: str):
    """A wandb run from either its id or its display name."""
    try:
        return api.run(f"{ENTITY}/{PROJECT}/{ident}")
    except Exception:
        matches = list(api.runs(ENTITY + "/" + PROJECT,
                                filters={"displayName": ident}))
        if not matches:
            raise SystemExit(f"no run with id or display name {ident!r}")
        if len(matches) > 1:
            raise SystemExit(
                f"{ident!r} matches {len(matches)} runs: "
                + ", ".join(r.id for r in matches)
            )
        return matches[0]


def _rows(run, metric: str, step_key: str = "_step"):
    """History rows for (STEP_KEY, metric), exact if the backend allows it.

    `scan_history(keys=...)` is the exact path but raises "Step column '_step'
    not found in schema" on older runs whose history lacks the column the
    keyed-scan endpoint requires. `history()` uses a different endpoint and
    works there, at the cost of sampling - so which one served a number is
    recorded in the output rather than left ambiguous.
    """
    try:
        return list(run.scan_history(keys=[step_key, metric])), "scan_history"
    except Exception:
        df = run.history(keys=[step_key, metric], samples=100_000)
        return df.to_dict("records"), "history(sampled)"


def _series(run, metric: str, step_key: str) -> tuple[np.ndarray, np.ndarray, str]:
    """(step, value) for one metric against `step_key`, NaNs dropped, sorted."""
    xs, ys = [], []
    rows, source = _rows(run, metric, step_key)
    for row in rows:
        x, y = row.get(step_key), row.get(metric)
        if x is None or y is None:
            continue
        try:
            y = float(y)
        except (TypeError, ValueError):
            continue  # media rows (figures) share these keys
        if np.isfinite(y):
            xs.append(float(x)); ys.append(y)
    if not xs:
        return np.array([]), np.array([]), source
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ys)[order], source


def env_step_axis(run):
    """The run's (`_step` -> env steps) mapping, from its own update log."""
    xs, ys, _ = _series(run, STEP_KEY, "_step")
    if xs.size == 0:
        raise SystemExit(f"run {run.id} logs no `{STEP_KEY}`; cannot build an env-step axis")
    return xs, ys


def trace(run, metric: str, axis) -> tuple[np.ndarray, np.ndarray, str]:
    """(env_steps, values, source) - the metric on the shared env-step axis."""
    ax_step, ax_frames = axis
    xs, ys, source = _series(run, metric, "_step")
    if xs.size == 0:
        return xs, ys, source
    return np.interp(xs, ax_step, ax_frames), ys, source


def band(values: np.ndarray) -> float:
    """The reference's own adjacent-sample spread: std of successive diffs.

    This is the scale at which the metric moves for reasons that are not the
    intervention, so a difference smaller than it is not evidence of anything.
    """
    return float(np.std(np.diff(values))) if values.size > 2 else float("nan")


def at_steps(xs: np.ndarray, ys: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Nearest recorded value at each target env step (no interpolation - these
    are measurements, not a continuous function)."""
    return np.array([ys[int(np.argmin(np.abs(xs - t)))] if xs.size else np.nan
                     for t in targets])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="run id or display name")
    ap.add_argument("--run", required=True, help="run id or display name")
    ap.add_argument("--metric", action="append", default=None)
    ap.add_argument("--points", type=int, default=6, help="matched env-step checkpoints")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    api = wandb.Api(timeout=59)
    ref, new = resolve(api, args.reference), resolve(api, args.run)
    metrics = tuple(args.metric) if args.metric else DEFAULT_METRICS
    print(f"reference : {ref.name}  ({ref.id})  state={ref.state}")
    print(f"run       : {new.name}  ({new.id})  state={new.state}\n")
    ref_axis, new_axis = env_step_axis(ref), env_step_axis(new)
    print(f"env-step axis: reference to {ref_axis[1].max():,.0f}, "
          f"run to {new_axis[1].max():,.0f}\n")

    rows, missing = [], []
    for m in metrics:
        rx, ry, rsrc = trace(ref, m, ref_axis)
        nx, ny, nsrc = trace(new, m, new_axis)
        if ry.size == 0 or ny.size == 0:
            missing.append((m, "reference" if ry.size == 0 else "run"))
            continue
        hi = min(rx.max(), nx.max())
        targets = np.linspace(hi / args.points, hi, args.points)
        rv, nv = at_steps(rx, ry, targets), at_steps(nx, ny, targets)
        b = band(ry)
        rows.append(dict(metric=m, band=b, source={"reference": rsrc, "run": nsrc},
                         steps=targets.tolist(),
                         reference=rv.tolist(), run=nv.tolist(),
                         delta=(nv - rv).tolist(),
                         within_band=[bool(abs(d) <= b) if np.isfinite(b) else None
                                      for d in (nv - rv)]))

    w = max(len(r["metric"]) for r in rows) if rows else 12
    for r in rows:
        print(f"{r['metric']:<{w}}  band +/-{r['band']:.4f}")
        print(f"{'  env step':<{w}}  " + "".join(f"{s:>12,.0f}" for s in r["steps"]))
        print(f"{'  reference':<{w}}  " + "".join(f"{v:>12.4f}" for v in r["reference"]))
        print(f"{'  run':<{w}}  " + "".join(f"{v:>12.4f}" for v in r["run"]))
        print(f"{'  delta':<{w}}  " + "".join(
            f"{d:>11.4f}{'' if ok else '*'}" for d, ok in zip(r["delta"], r["within_band"])))
        print()
    if missing:
        print("not logged by one side (cannot be compared):")
        for m, where in missing:
            print(f"  {m}  - absent from the {where}")
    n_out = sum(not ok for r in rows for ok in r["within_band"] if ok is not None)
    print(f"\n* = outside the reference's own adjacent-sample band. {n_out} of "
          f"{sum(len(r['within_band']) for r in rows)} matched points.")

    # RL_STORAGE, not a path literal: a hardcoded run-output path works on
    # the machine that wrote it and lands in $SLURM_TMPDIR on the cluster,
    # which is deleted when the job ends.
    out = Path(get_storage_dir()) / "summary"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"wandb_compare{'_' + args.tag if args.tag else ''}.json"
    path.write_text(json.dumps(
        {"reference": {"id": ref.id, "name": ref.name},
         "run": {"id": new.id, "name": new.name},
         "step_key": STEP_KEY, "rows": rows,
         "missing": [{"metric": m, "absent_from": w} for m, w in missing]},
        indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
