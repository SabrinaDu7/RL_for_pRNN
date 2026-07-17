"""Compare two benchmark.py JSON outputs.

Exact-math refactors must match within tight tolerance (default rtol 1e-5);
RNG-order changes are compared loosely (--loose: report deltas, fail only on
NaN/inf or gross divergence >50% on loss means).

Usage:
    uv run python tests/perf/compare_metrics.py baseline.json new.json [--loose]
"""

import argparse
import json
import math
import sys


def rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline")
    p.add_argument("new")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--loose", action="store_true")
    args = p.parse_args()

    base = json.load(open(args.baseline))
    new = json.load(open(args.new))

    print(f"FPS: {base['fps']} -> {new['fps']}  "
          f"({new['fps'] / max(base['fps'], 1e-9):.2f}x)")

    failed = []
    for key, bvals in base["metrics"].items():
        nvals = new["metrics"].get(key, [])
        if len(bvals) != len(nvals):
            print(f"  {key}: length {len(bvals)} -> {len(nvals)} (skipping elementwise)")
            continue
        if not bvals:
            continue
        worst = max((rel_diff(b, n) for b, n in zip(bvals, nvals)), default=0.0)
        bad_num = any(math.isnan(n) or math.isinf(n) for n in nvals)
        mean_b = sum(bvals) / len(bvals)
        mean_n = sum(nvals) / len(nvals)
        status = "OK"
        if bad_num:
            status = "FAIL(nan/inf)"
            failed.append(key)
        elif args.loose:
            if key in ("policy_loss", "value_loss", "prnn_loss") and rel_diff(mean_b, mean_n) > 0.5:
                status = "FAIL(mean>50%)"
                failed.append(key)
        elif worst > args.rtol:
            status = f"FAIL(worst rel {worst:.2e})"
            failed.append(key)
        print(f"  {key:<18} mean {mean_b:+.5g} -> {mean_n:+.5g}  worst-rel {worst:.2e}  {status}")

    print("\nstage timings (total_s):")
    stages = set(base["timings"]) | set(new["timings"])
    for s in sorted(stages):
        b = base["timings"].get(s, {}).get("total_s", 0.0)
        n = new["timings"].get(s, {}).get("total_s", 0.0)
        print(f"  {s:<32}{b:>9.3f} -> {n:>9.3f}")

    if failed:
        print(f"\nFAILED: {failed}")
        sys.exit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
