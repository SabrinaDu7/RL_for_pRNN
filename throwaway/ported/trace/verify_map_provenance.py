"""Which run produced each cached map set in `outputs/trace/maps_all.npz`?

`maps_all.npz` holds `baseline` and one post-exposure map set per object
location (`m_7_11`, `m_14_7`, `m_7_2`), and Figures 0.2, 1.2 and 2.x are built
from it and from the run directories under `outputs/mila_omt_dense/`. But
**neither the npz nor those run directories records which object location a run
was trained at**: the runs have no config on disk, and no wandb record survives
for them (checked 2026-08-13 against `blake-richards/curious-george` and
`curious-george-otc` - the only run anywhere dated 2026-07-30 is
`prnn_curious_26-07-30-10-37-02`). So the columns of those figures are labelled
from memory, which is not a source.

This recovers the mapping from the data itself. Every cached map set was built
by replaying a checkpoint through the fixed object-absent probe; replaying the
same checkpoint through the same probe reproduces it. So for each candidate run
we replay its final checkpoint and correlate the resulting maps against each
cached set. The run that GENERATED a set matches it near-exactly; a different
run matches at the exposure-stability level instead, which this project has
already measured at r ~ 0.98 - so the discriminator is the gap between ~1.000
and ~0.98, and the script reports the margin rather than just the winner, since
a small margin means the answer is not established.

    uv run python scripts/trace/verify_map_provenance.py                 # the 3 runs the figures use
    uv run python scripts/trace/verify_map_provenance.py --all           # all 9 dense runs
    uv run python scripts/trace/verify_map_provenance.py --runs outputs/mila_omt/*

The pipeline here is asserted, not assumed: `--check-pipeline` replays the known
baseline checkpoint and correlates it against the cached `baseline`. That must
be ~1.0 before any negative result below means anything, because a pipeline
mismatch and a wrong-run answer look identical in the correlations.

Writes outputs/trace/map_provenance.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path("outputs/trace")
NPZ = OUT / "maps_all.npz"
DENSE = Path("outputs/mila_omt_dense")
ONSET, HID = 20, 500

# The three the figures actually consume: `otc_figures.CONDITIONS`
# "[2,2,2] fixedpos" and the "[2,2,2] normal" column of `otc_figures.maps()`.
FIGURE_RUNS = ["omt-cur-dot-0730-165325", "omt-cur-dot-0730-172405",
               "omt-cur-dot-0730-175326"]

# A match this far below 1.0 means the replay did not reproduce the cache at
# all, so the whole comparison is uninformative rather than negative.
REPRODUCTION_FLOOR = 0.995


def checkpoint_at(run: Path, step: int | None) -> str:
    """The run's pRNN state at `step`, or at its largest step when None.

    The step has to be selectable: `maps_all.npz` holds one map set per location
    at the end of training, while `maps_dense.npz` holds a per-step series, and
    a cached set can only be reproduced from the checkpoint it was built from.
    """
    from curious_george import resolve_prnn_ckpt

    steps = [int(d.name) for d in run.iterdir() if d.is_dir() and d.name.isdigit()]
    if not steps:
        raise SystemExit(f"{run} has no numbered step directories")
    if step is not None and step not in steps:
        raise SystemExit(f"{run} has no step {step}; has {sorted(steps)}")
    return resolve_prnn_ckpt(f"{run}/{max(steps) if step is None else step}")


def replayed_maps(*, ckpt: str, args, env, probe) -> np.ndarray:
    from curious_george import get_pN
    from scripts.trace import trace_maps as tm, trace_probe as tp

    pN = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=ckpt)
    pN.wandb_log = False
    h = tp.replay_checkpoint(pN=pN, probe=probe)[:, ONSET:, :].numpy()
    pos = probe.agent_pos[:, ONSET:probe.n_steps, :].astype(np.float64)
    return tm.occupancy_and_maps(h=h.reshape(-1, HID), pos=pos.reshape(-1, 2), env=env)[0]


def correlate(a: np.ndarray, b: np.ndarray) -> float:
    """Median per-unit correlation over bins valid in BOTH map sets."""
    rs = []
    for u in range(a.shape[0]):
        x, y = a[u].ravel(), b[u].ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 8 or x[ok].std() < 1e-12 or y[ok].std() < 1e-12:
            continue
        rs.append(float(np.corrcoef(x[ok], y[ok])[0, 1]))
    return float(np.median(rs)) if rs else float("nan")


BASE_CKPT = "outputs/ckpts/pRNN_curious_26-07-23-10-06-25/predictiveNet_state.pt"

# Recovered by the map route above; used as the POSITIVE CONTROL for the
# readout route, which is the only route left for runs no cached map came from.
KNOWN_DENSE = {"omt-cur-dot-0730-165325": (7, 11),
               "omt-cur-dot-0730-165916": (7, 2),
               "omt-cur-dot-0730-165922": (14, 7)}
CANDIDATES = [(7, 11), (14, 7), (7, 2)]


def readout_route(*, runs, step, args, env, probe, n_trajs: int = 200) -> list[dict]:
    """Identify a run's object location from its READOUT, not its rate maps.

    The map route can only identify runs a cached map set was built from. Every
    other run needs a signal that exposure itself leaves, and §0 of
    `docs/results/result-summary-2026-08-12.md` says where that lives: exposure
    writes the object into `W_out`. It also says the write is ~89%
    POSITION-INDEPENDENT, so only the ~11% residual can discriminate - which is
    exactly why this is reported against the runs whose answer is already known.
    If those three are not classified correctly, the route cannot be trusted for
    the rest, and that is the finding rather than a failure to report.

    Score is the excess predicted green at each candidate over the pre-exposure
    baseline at the SAME cell, so a cell that is simply greener for everyone
    cancels - the location control this project has had to learn three times.
    """
    from curious_george import get_pN
    from scripts.trace.trace_readout_test import predicted_green_at

    base_pN = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=BASE_CKPT)
    base_pN.wandb_log = False
    base = {L: predicted_green_at(pN=base_pN, probe=probe, env=env, loc=L, n_trajs=n_trajs)
            for L in CANDIDATES}

    rows = []
    for run in runs:
        pN = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=checkpoint_at(run, step))
        pN.wandb_log = False
        exc = {L: predicted_green_at(pN=pN, probe=probe, env=env, loc=L, n_trajs=n_trajs) - base[L]
               for L in CANDIDATES}
        order = sorted(CANDIDATES, key=lambda L: -exc[L])
        rows.append({"run": run.name, "excess": {str(k): v for k, v in exc.items()},
                     "predicted": order[0], "margin": exc[order[0]] - exc[order[1]],
                     "known": KNOWN_DENSE.get(run.name)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="score all 9 dense runs, not just the 3 used")
    ap.add_argument("--runs", nargs="*", default=None, help="explicit run directories")
    ap.add_argument("--cache", default="maps_all.npz", help="cached map archive under outputs/trace/")
    ap.add_argument("--step", type=int, default=None,
                    help="replay this checkpoint step and score only cached sets tagged with it")
    ap.add_argument("--readout", action="store_true",
                    help="identify by W_out's object signal instead of by rate maps; "
                         "reports the known-answer runs as its positive control")
    a = ap.parse_args()

    from hydra import compose, initialize_config_dir
    from prnn.utils import ActionEncodingsEnum, AgentInputType, MinigridEnvNames

    from curious_george import make_env
    from scripts.trace.trace_probe import load_probe

    cached = np.load(OUT / a.cache)
    targets = [k for k in cached.files
               if k.startswith("m_")
               and (a.step is None or k.endswith(f"_{a.step}"))]
    if not targets:
        raise SystemExit(f"{a.cache} has no map sets"
                         + (f" tagged step {a.step}" if a.step is not None else ""))
    if a.runs:
        runs = [Path(r) for r in a.runs]
    else:
        runs = sorted(DENSE.iterdir()) if a.all else [DENSE / r for r in FIGURE_RUNS]
    runs = [r for r in runs if r.is_dir()]

    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        args = compose(config_name="main")
    env = make_env(env_key=MinigridEnvNames.LRoom, input_type=AgentInputType.H_PO.value,
                   act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0)
    probe = load_probe(OUT / "probe_lroom_noobj")

    # Assert the pipeline BEFORE interpreting any negative: a wrong-run answer
    # and a wrong-pipeline answer are indistinguishable in the correlations.
    r_pipe = correlate(replayed_maps(ckpt=BASE_CKPT, args=args, env=env, probe=probe),
                       cached["baseline"])
    print(f"pipeline check: baseline checkpoint vs cached 'baseline'  r = {r_pipe:.5f}"
          f"{'' if r_pipe >= REPRODUCTION_FLOOR else '   PIPELINE MISMATCH - stop'}\n")
    if r_pipe < REPRODUCTION_FLOOR:
        raise SystemExit("the map pipeline does not reproduce the cache; nothing below is meaningful")

    if a.readout:
        rows = readout_route(runs=runs, step=a.step, args=args, env=env, probe=probe)
        print(f"{'run':>26} " + " ".join(f"{str(L):>10}" for L in CANDIDATES)
              + f" {'predicted':>11} {'margin':>9} {'known':>9} {'':>5}")
        for r in rows:
            hit = "" if r["known"] is None else ("OK" if tuple(r["known"]) == r["predicted"] else "WRONG")
            print(f"{r['run']:>26} "
                  + " ".join(f"{r['excess'][str(L)]:>+10.4f}" for L in CANDIDATES)
                  + f" {str(r['predicted']):>11} {r['margin']:>+9.4f} "
                  f"{str(r['known'] or '-'):>9} {hit:>5}")
        checked = [r for r in rows if r["known"] is not None]
        n_ok = sum(tuple(r["known"]) == r["predicted"] for r in checked)
        print(f"\npositive control: {n_ok}/{len(checked)} runs with a known answer classified "
              f"correctly")
        if checked and n_ok < len(checked):
            print("-> the readout route does NOT recover known answers, so its predictions for the\n"
                  "   remaining runs are not evidence. Their object locations stay unrecovered.")
        (OUT / "map_provenance_readout.json").write_text(json.dumps(rows, indent=2, default=float))
        print(f"wrote {OUT / 'map_provenance_readout.json'}")
        return

    print(f"{len(runs)} runs x {len(targets)} cached map sets, "
          f"probe {probe.n_trajs} trajectories x {probe.n_steps} steps\n")
    print(f"{'run':>26} " + " ".join(f"{t:>10}" for t in targets)
          + f" {'baseline':>10} {'best':>10} {'margin':>8}")

    rows = []
    for run in runs:
        maps = replayed_maps(ckpt=checkpoint_at(run, a.step), args=args, env=env, probe=probe)
        rs = {t: correlate(maps, cached[t]) for t in targets}
        rs_base = correlate(maps, cached["baseline"])
        order = sorted(rs, key=lambda k: -rs[k])
        best, margin = order[0], rs[order[0]] - rs[order[1]]
        print(f"{run.name:>26} " + " ".join(f"{rs[t]:>10.5f}" for t in targets)
              + f" {rs_base:>10.5f} {best:>10} {margin:>+8.5f}"
              + ("" if rs[best] >= REPRODUCTION_FLOOR else "   NO REPRODUCTION"))
        rows.append({"run": run.name, "r": rs, "r_baseline": rs_base,
                     "best": best, "margin": margin,
                     "reproduced": bool(rs[best] >= REPRODUCTION_FLOOR)})

    (OUT / "map_provenance.json").write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nwrote {OUT / 'map_provenance.json'}")
    if not any(r["reproduced"] for r in rows):
        print(f"\nNo run reproduces any cached map set (all best matches < {REPRODUCTION_FLOOR}).\n"
              "The pipeline check above passed, so this is a CONFIRMED NEGATIVE, not an\n"
              "inconclusive test: none of these runs produced these maps. The ~0.97 values are\n"
              "the exposure-stability level this project measured as r ~ 0.98, i.e. what any two\n"
              "L-room checkpoints score against each other.")


if __name__ == "__main__":
    main()
