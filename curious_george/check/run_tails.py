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
    """Fraction of logged updates with `policy_entropy` < `COLLAPSE_BITS`."""
    loc_entropy_min: float

    def row(self) -> str:
        def f(x: float, w: int, p: int, suffix: str = "") -> str:
            return f"{'n/a':>{w}}" if np.isnan(x) else f"{x:{w}.{p}f}{suffix}"

        return (f"{self.run:38s} {f(self.srsa_tail, 8, 3)} {f(self.srsa_band, 6, 3)} "
                f"{f(self.loss_tail, 10, 5)} {f(self.policy_entropy_min, 7, 3)} "
                f"{f(self.policy_entropy_median, 7, 3)} "
                f"{f(100 * self.collapse_duty_cycle, 6, 1, '%')} "
                f"{f(self.loc_entropy_min, 8, 3)}")

    @staticmethod
    def header() -> str:
        return (f"{'run':38s} {'sRSA t':>8s} {'band':>6s} {'loss t':>10s} "
                f"{'pe MIN':>7s} {'pe MED':>7s} {'%<1.0':>7s} {'loc MIN':>8s}")


def series(run, metric: str, *, attempts: int = 6) -> np.ndarray:
    """A metric's values in logged order, DENSE, or empty if they cannot be had.

    `scan_history` is the streaming reader and the only one that returns every
    row. It fails two ways, both recorded in
    `log_and_store/wandb.py::_history_rows`: it raises `Step column '_step' not
    found in schema`, and it can return rows with the key simply absent. Both
    are INTERMITTENT on this backend - the same run and metric raised on one
    call and returned 43,936 rows on the next - so this retries.

    🔴 It does NOT fall back to `run.history()`. That endpoint returns a fixed
    SUBSAMPLE spanning the whole run (500 rows by default), so the "last 500"
    of it is the whole-run mean, not a tail: measured on `focal5mlp-off0`,
    0.04261 against the true 0.01047. A tail of a subsample is not a tail, and
    a wrong number that looks right is worse than no number. Empty here becomes
    NaN in `RunTails`, which prints as `n/a`.
    """
    for attempt in range(attempts):
        if attempt:
            # Exponential: the refusals cluster when the API is being hit hard,
            # and a fixed 2 s retry just adds to the hammering.
            time.sleep(2.0 * 2 ** (attempt - 1))
        try:
            rows = list(run.scan_history(keys=[metric], page_size=2000))
        except Exception:  # noqa: BLE001 - any backend refusal; it is transient
            continue
        vals = [r.get(metric) for r in rows]
        if any(v is not None for v in vals):
            return np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    return np.array([], dtype=float)


def tails_of(run, *, sparse_points: int = 3, dense_points: int = 500) -> RunTails:
    """Summarise one wandb run object. `run.name` identifies it."""
    def tail(v: np.ndarray, n: int) -> float:
        return float(v[-n:].mean()) if v.size else float("nan")

    srsa = series(run, SPARSE_METRIC)
    loss = series(run, "pRNN loss")
    pe = series(run, "policy_entropy")
    loc = series(run, "loc_entropy")
    return RunTails(
        run=run.name.split("_curious")[0],
        srsa_tail=tail(srsa, sparse_points),
        # The estimator's own error bar: how far consecutive samples move.
        srsa_band=float(np.std(np.diff(srsa))) if srsa.size > 2 else float("nan"),
        loss_tail=tail(loss, dense_points),
        policy_entropy_min=float(pe.min()) if pe.size else float("nan"),
        policy_entropy_median=float(np.median(pe)) if pe.size else float("nan"),
        collapse_duty_cycle=float((pe < COLLAPSE_BITS).mean()) if pe.size else float("nan"),
        loc_entropy_min=float(loc.min()) if loc.size else float("nan"),
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
