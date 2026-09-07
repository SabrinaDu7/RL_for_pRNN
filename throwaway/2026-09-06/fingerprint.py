"""Bitwise fingerprint of a short training run, for gating the 2026-09 cleanup.

Runs a preset for two rollouts with one spatial-eval event between them, then
one extra collection, and writes SHA-256 digests of every tensor that matters
(weights, optimizer moments, the rollout buffers, the RNG streams) plus the
printed eval numbers. Two runs of the same code must agree byte for byte; a
cleanup step that changes any digest on a production case is not a cleanup.

    uv run python throwaway/2026-09-06/fingerprint.py --label baseline --cases prod
    uv run python throwaway/2026-09-06/fingerprint.py --label step3 --cases all
    uv run python throwaway/2026-09-06/fingerprint.py --diff baseline step3

Throwaway by design: nothing in the repo depends on it.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).parent / "fp"

_COMMON = [
    "--run.no-wandb", "--run.exp-name", "fp",
    "--run.save-every-steps", "0", "--run.archive-every-steps", "0",
    "--eval.plot-every-steps", "0", "--eval.log-every-steps", "0",
]
_TWO_ROLLOUTS_POOLED = [
    "--train-prnn.total-grad-steps", "64", "--train-policy.total-grad-steps", "256",
    "--eval.evals", "SPATIAL_MULTIROOM", "--eval.analysis-every-steps", "65536",
]
CASES: dict[str, list[str]] = {
    # the impassable-objects multienv production arm (multienv.sh + the CE flags)
    "prod": [
        "multienv-fast", *_COMMON, *_TWO_ROLLOUTS_POOLED,
        "--arch-prnn.loss", "CE", "--arch-prnn.focal-gamma", "5", "--arch-prnn.readout", "MLP",
        "--train-policy.normalize-reward",
        "env.source:selected", "--env.source.n", "5", "--env.source.impassable",
        "--env.source.positions", "0", "1", "2", "3", "5", "6", "7", "8",
    ],
    # the walkable multienv arm, MSE (the preset as it ships)
    "walkable": ["multienv-fast", *_COMMON, *_TWO_ROLLOUTS_POOLED],
    # the accelerated single L-room (the A/B shape), MSE
    "parity": [
        "parity", *_COMMON,
        "--train-prnn.total-grad-steps", "64", "--train-policy.total-grad-steps", "256",
        "--eval.evals", "SPATIAL_ONPOLICY", "--eval.analysis-every-steps", "65536",
    ],
    # the serial single L-room baseline, MSE, walkable default landmarks
    "reference": [
        "reference", *_COMMON,
        "--train-prnn.total-grad-steps", "16", "--train-policy.total-grad-steps", "64",
        "--eval.evals", "SPATIAL_ONPOLICY", "--eval.analysis-every-steps", "2048",
    ],
}


def _digest(t) -> str:
    if isinstance(t, torch.Tensor):
        a = t.detach().cpu().contiguous()
        return f"{tuple(a.shape)}:{a.dtype}:" + hashlib.sha256(a.numpy().tobytes()).hexdigest()[:16]
    if isinstance(t, np.ndarray):
        return f"{t.shape}:{t.dtype}:" + hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest()[:16]
    if isinstance(t, (list, tuple)):
        return hashlib.sha256(repr(t).encode()).hexdigest()[:16]
    return repr(t)


def _state_digests(state: dict) -> dict:
    return {k: _digest(v) for k, v in state.items()}


def _optimizer_digests(opt) -> dict:
    out = {}
    sd = opt.state_dict()
    for i, st in sd["state"].items():
        for k, v in st.items():
            out[f"{i}.{k}"] = _digest(v) if isinstance(v, torch.Tensor) else repr(v)
    out["param_groups"] = repr([{k: v for k, v in g.items() if k != "params"} for g in sd["param_groups"]])
    return out


def run_case(name: str) -> dict:
    from curious_george import configs
    from curious_george.training.loop import run_training
    from curious_george.training.setup import setup_run, setup_training

    cfg = configs.cli(CASES[name])
    t0 = time.perf_counter()
    comps = setup_training(cfg)
    cfg = comps.cfg
    run_ctx = setup_run(cfg)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_training(cfg, run_ctx, comps)
        exps, logs = comps.algo.collect_experiences()
    printed = buf.getvalue()
    eval_lines = [ln.strip() for ln in printed.splitlines() if "sRSA" in ln]

    algo = comps.algo
    fp: dict = {
        "case": name,
        "argv": CASES[name],
        "seconds": round(time.perf_counter() - t0, 1),
        "eval_lines": eval_lines,
        "prnn_state": _state_digests(comps.predictiveNet.pRNN.state_dict()),
        "acmodel_state": _state_digests(comps.acmodel.state_dict()),
        "ac_optimizer": _optimizer_digests(algo.optimizer),
        "prnn_optimizer": _optimizer_digests(comps.predictiveNet.optimizer),
        "rollout": {
            k: _digest(getattr(exps, k)) for k in ("SR", "action", "value", "reward", "advantage", "returnn", "log_prob")
        },
        "rollout_obs": {k: _digest(getattr(exps.obs, k)) for k in ("image", "direction") if hasattr(exps.obs, k)},
        "algo_views": {
            k: _digest(getattr(algo, k)) for k in ("curious_rewards", "masks", "SRs", "directions")
        },
        "positions_episodes": _digest(algo.positions_episodes) if algo.positions_episodes is not None else None,
        "segment_layouts": _digest(algo.segment_layouts) if algo.segment_layouts is not None else None,
        "locs": _digest(list(algo.locs)),
        "log_scalars": {
            k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else _digest(v))
            for k, v in sorted(logs.items())
            if k not in ("locs", "subroom_ids", "joint_dist")
        },
        "joint_dist": _digest(logs["joint_dist"]),
        "rng": {
            "torch_cpu": _digest(torch.random.get_rng_state()),
            "torch_cuda": [_digest(s) for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
            "numpy": _digest(np.random.get_state()[1]),
        },
        "pN_state": _digest(comps.predictiveNet.state) if isinstance(comps.predictiveNet.state, torch.Tensor) else repr(comps.predictiveNet.state),
        "pN_phase": comps.predictiveNet.phase,
        "torch": torch.__version__,
    }
    if hasattr(comps.envs, "positions"):
        fp["pool"] = {
            "positions": _digest(comps.envs.positions),
            "directions": _digest(comps.envs.directions),
            "stream_layout": _digest(comps.envs.stream_layout),
            "layout_episodes": _digest(comps.envs.layout_episodes),
        }
    if hasattr(comps.envs, "close"):
        comps.envs.close()
    return fp


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def diff(a: str, b: str, cases: list[str]) -> int:
    bad = 0
    for case in cases:
        pa, pb = OUT / a / f"{case}.json", OUT / b / f"{case}.json"
        if not pa.exists() or not pb.exists():
            print(f"{case}: missing {'A' if not pa.exists() else 'B'}")
            bad += 1
            continue
        fa, fb = _flatten(json.loads(pa.read_text())), _flatten(json.loads(pb.read_text()))
        skip = ("seconds", "argv")
        keys = sorted((set(fa) | set(fb)) - {k for k in set(fa) | set(fb) if k.split(".")[0] in skip})
        moved = [k for k in keys if fa.get(k, "<absent>") != fb.get(k, "<absent>")]
        if moved:
            bad += 1
            print(f"{case}: {len(moved)} of {len(keys)} leaves differ")
            for k in moved[:40]:
                print(f"    {k}: {fa.get(k, '<absent>')!r} -> {fb.get(k, '<absent>')!r}")
        else:
            print(f"{case}: IDENTICAL ({len(keys)} leaves)")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--cases", default="all")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    cases = list(CASES) if a.cases == "all" else a.cases.split(",")
    if a.diff:
        sys.exit(diff(*a.diff, cases))
    assert a.label, "--label is required to run"
    for r in range(a.repeat):
        label = a.label if a.repeat == 1 else f"{a.label}-r{r + 1}"
        (OUT / label).mkdir(parents=True, exist_ok=True)
        for case in cases:
            fp = run_case(case)
            (OUT / label / f"{case}.json").write_text(json.dumps(fp, indent=1))
            print(f"[{label}] {case}: {fp['seconds']} s; eval: {fp['eval_lines']}", flush=True)


if __name__ == "__main__":
    main()
