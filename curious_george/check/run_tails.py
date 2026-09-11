"""Tail statistics for finished runs, and the exploration-collapse duty cycle.

Two questions `wandb_compare` does not answer, because neither is a comparison
between two runs:

**What is this run's value for a metric?** Not its final logged point.
`sRSA_onPolicy` and `multiroom/mean_room_sRSA` are logged a couple of dozen
times per run with an adjacent-sample spread of 0.05-0.11, so a single endpoint
is inside the noise - reading one produced three wrong conclusions in the first
version of `docs/action-offset-ab-2026-08-29.md`, which marks the rule in red.
The estimator is a TAIL MEAN, reported beside the band it has to clear.

**Did the policy collapse?** `arch_prnn.action_offset=1` hands the policy `h[t]`
instead of `h[t-1]` - a strictly better basis for choosing an action - and in
the raw-advantage era that let it condition sharply enough to stop exploring.
The signature is not an endpoint either: it is a TRANSIENT with a duty cycle,
measured as the fraction of logged updates with `policy_entropy` below
`COLLAPSE_BITS`. The A/B measured 1.9-21.1% for offset 1 at `entropy_coef=0.001`
against 0.0% for every offset-0 run; medians barely separated, which is why the
duty cycle and not the median is the statistic.

    uv run python -m curious_george.check.run_tails --run <name> [--run <name> ...]
    uv run python -m curious_george.check.run_tails --prefix mx-impassable-n8-s2-
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass

import numpy as np

from curious_george.configs import RunCfg

#: policy_entropy below this counts as collapsed. The A/B's threshold, kept
#: identical so its numbers and these are the same measurement.
COLLAPSE_BITS: float = 1.0

#: log2 of the four-action space - what a uniform policy scores, for scale.
UNIFORM_ENTROPY_BITS: float = 2.0

#: `run.history()` returns at most this many rows however many are asked for
#: (measured: `samples=100000` on a 45,375-step run returns 10,000).
HISTORY_CAP: int = 10_000

#: Tail window as a FRACTION of the logged series, not a row count. A row count
#: is density-dependent - "the last 500 rows" is the last 500 gradient steps on
#: a dense series and the last ~2,270 on a 10,000-row subsample of the same run,
#: which are different questions. A fraction is the same window either way, and
#: the subsample is uniform, so the fractional tail mean is unbiased.
TAIL_FRACTION: float = 0.0114

#: The analysis-cadence metric the multi-room arm is judged on, and the dense
#: per-update metrics. Separate because they are logged at different cadences
#: and so need different tail lengths.
SPARSE_METRIC: str = "multiroom/mean_room_sRSA"
DENSE_METRICS: tuple[str, ...] = ("pRNN loss", "policy_entropy", "loc_entropy")


@dataclass(frozen=True)
class RunTails:
    """One finished run, summarised by estimators that survive their own noise.

    `srsa_band` is the run's own adjacent-sample standard deviation - the same
    band `wandb_compare` flags against. A difference between two runs smaller
    than either band is not a difference.
    """

    run: str
    srsa_tail: float
    srsa_band: float
    loss_tail: float
    policy_entropy_min: float
    policy_entropy_median: float
    collapse_duty_cycle: float
    """Fraction of logged updates with `policy_entropy` < `COLLAPSE_BITS`.
    Unbiased on a uniform subsample, so it does not depend on `dense`."""
    loc_entropy_min: float
    dense: bool = True
    """False when a dense series was unavailable and the 10,000-row subsample
    was read instead. The tail means and the duty cycle are still right - the
    window is a FRACTION and the subsample is uniform - but the two MINIMA are
    then a lower bound on how extreme the run actually got, because a subsample
    can only miss extremes. Rows print `~` in that case."""

    def row(self) -> str:
        def f(x: float, w: int, p: int, suffix: str = "") -> str:
            return f"{'n/a':>{w}}" if np.isnan(x) else f"{x:{w}.{p}f}{suffix}"

        mark = "" if self.dense else "~"
        return (f"{self.run + mark:44s} {f(self.srsa_tail, 8, 3)} {f(self.srsa_band, 6, 3)} "
                f"{f(self.loss_tail, 10, 5)} {f(self.policy_entropy_min, 7, 3)} "
                f"{f(self.policy_entropy_median, 7, 3)} "
                f"{f(100 * self.collapse_duty_cycle, 6, 1, '%')} "
                f"{f(self.loc_entropy_min, 8, 3)}")

    @staticmethod
    def header() -> str:
        return (f"{'run (~ = read from the 10k subsample)':44s} {'sRSA t':>8s} {'band':>6s} {'loss t':>10s} "
                f"{'pe MIN':>7s} {'pe MED':>7s} {'%<1.0':>7s} {'loc MIN':>8s}")


def series(run, metric: str, *, attempts: int = 3) -> tuple[np.ndarray, bool]:
    """(values in logged order, whether every logged row is present).

    `scan_history` streams EVERY row and is the right answer when it works. It
    refuses with `Step column '_step' not found in schema`, and it can return
    rows with the key absent - both recorded in
    `log_and_store/wandb.py::_history_rows`. MEASURED here: the refusal is
    deterministic per (run, metric) over minutes, not flaky per call - the same
    twelve runs all served dense history once and then refused for a stable
    subset across two later attempts with backoff between. So retrying is worth
    a couple of tries and no more.

    `run.history()` is then the fallback, and its result is a SUBSAMPLE: it is
    capped at 10,000 rows whatever `samples` asks for, so a 45,375-step run
    comes back at roughly every 4.5th row. That is why this reports the flag
    instead of hiding the difference - see `RunTails` for what each statistic
    does with it.
    """
    for attempt in range(attempts):
        if attempt:
            time.sleep(2.0 * 2 ** (attempt - 1))
        try:
            rows = list(run.scan_history(keys=[metric], page_size=2000))
        except Exception:  # noqa: BLE001 - any backend refusal
            continue
        vals = [r.get(metric) for r in rows]
        if any(v is not None for v in vals):
            v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
            return v, True
    try:
        frame = run.history(keys=[metric], samples=HISTORY_CAP, pandas=True)
    except Exception:  # noqa: BLE001
        return np.array([], dtype=float), False
    if frame is None or len(frame) == 0 or metric not in getattr(frame, "columns", []):
        return np.array([], dtype=float), False
    v = frame[metric].to_numpy(dtype=float)
    return v[np.isfinite(v)], False


def tails_of(run, *, sparse_points: int = 3) -> RunTails:
    """Summarise one wandb run object. `run.name` identifies it."""
    def tail(v: np.ndarray, n: int) -> float:
        return float(v[-n:].mean()) if v.size else float("nan")

    def frac_tail(v: np.ndarray) -> float:
        return tail(v, max(1, round(TAIL_FRACTION * v.size))) if v.size else float("nan")

    srsa, _ = series(run, SPARSE_METRIC)
    loss, loss_dense = series(run, "pRNN loss")
    pe, pe_dense = series(run, "policy_entropy")
    loc, loc_dense = series(run, "loc_entropy")
    return RunTails(
        run=run.name.split("_curious")[0],
        srsa_tail=tail(srsa, sparse_points),
        # The estimator's own error bar: how far consecutive samples move.
        srsa_band=float(np.std(np.diff(srsa))) if srsa.size > 2 else float("nan"),
        loss_tail=frac_tail(loss),
        policy_entropy_min=float(pe.min()) if pe.size else float("nan"),
        policy_entropy_median=float(np.median(pe)) if pe.size else float("nan"),
        collapse_duty_cycle=float((pe < COLLAPSE_BITS).mean()) if pe.size else float("nan"),
        loc_entropy_min=float(loc.min()) if loc.size else float("nan"),
        dense=bool(loss_dense and pe_dense and loc_dense),
    )


def main(argv: "list[str] | None" = None) -> None:
    import json

    import wandb

    d = RunCfg()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], help="run id or display name")
    ap.add_argument("--prefix", default=None, help="every run whose name starts with this")
    ap.add_argument("--limit", type=int, default=12, help="cap on --prefix matches")
    ap.add_argument("--entity", default=d.wandb_entity)
    ap.add_argument("--project", default=d.wandb_project)
    ap.add_argument("--json", default=None, help="also write the rows here")
    a = ap.parse_args(argv)
    if not a.run and not a.prefix:
        ap.error("give --run (repeatable) or --prefix")

    api = wandb.Api()
    path = f"{a.entity}/{a.project}"
    runs = [api.run(f"{path}/{n}") for n in a.run]
    if a.prefix:
        runs += [r for r in api.runs(path, order="-created_at", per_page=40)
                 if r.name.startswith(a.prefix)][:a.limit]

    out = sorted((tails_of(r) for r in runs), key=lambda t: t.run)
    print(RunTails.header())
    print("-" * len(RunTails.header()))
    for t in out:
        print(t.row())
    print(f"\npolicy_entropy ceiling (uniform over four actions) = "
          f"{UNIFORM_ENTROPY_BITS:.3f} bits; collapse threshold {COLLAPSE_BITS} bits, "
          f"as in docs/action-offset-ab-2026-08-29.md")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump([asdict(t) for t in out], fh, indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
