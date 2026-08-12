"""Object cells and trace cells across the Moser session sequence.

Consumes the checkpoints written by `scripts/moser_sessions.py` and asks, for
each session, which pRNN hidden units carry a field at the object's CURRENT
position (object cells) and which carry one at a position the object has since
LEFT (trace cells).

Definitions follow Tsao, Moser & Moser (2013, doi 10.1016/j.cub.2013.01.036),
which reports the two as independent populations - trace cells "generally did
not respond to the object when it was present". So:

  object cell at session k   field at P_k
  trace cell at session k    field at some earlier P_j (j<k, P_j != P_k)
                             AND no field at P_k
                             AND no field at P_j back in session 0, before any
                             object had ever appeared (else it is just a
                             pre-existing place field)

"Field at c" means the unit's `field_gain` at c (mean rate in a radius-2 disc
divided by the unit's own mean rate, scripts/trace_metric.py) ranks above the
95th percentile of that SAME unit's gains over all walkable cells. The
within-unit percentile is what makes 5% the chance rate, so the population
test is binomial against 0.05 - the same validated metric used throughout this
project (negative control 0.0601, p=0.174; positive control 100% recall).

The probe is identical across sessions by construction: the object is a
non-blocking floor tile, so a fixed action sequence visits exactly the same
positions whether or not it is present. Only the observations differ.

    uv run --no-sync python scripts/moser_analysis.py --manifest outputs/moser_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

RADIUS = 2.0
PCTILE = 95.0
ONSET = 20
OUT = Path("outputs/moser")


def _setup(env_name: str, obj):
    from hydra import initialize_config_dir, compose
    from prnn.utils import ActionEncodingsEnum, AgentInputType
    from curious_george import make_env

    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        args = compose(config_name="main")
    kw = {"new_obj_pos": tuple(obj)} if obj else {}
    env = make_env(env_key=env_name, input_type=AgentInputType.H_PO.value,
                   act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0, **kw)
    return args, env


def _build_probe(*, pN, env, env_name: str, n_dirs: int, n_steps: int = 256):
    """One trajectory per (walkable cell, head direction) for `n_dirs` directions.

    Same construction as trace_probe.build_probe, but built at the resolution
    actually needed instead of building all four directions and discarding
    three quarters - the collection loop is the expensive part here.
    """
    from curious_george import get_agent, AgentType
    from curious_george.world_model.device import eval_mode, on_device
    from scripts.analysis_OMT import get_walkable_mask, get_walkable_minigrid_positions
    from scripts.trace_probe import Probe, PROBE_SEED

    torch.manual_seed(PROBE_SEED)
    np.random.seed(PROBE_SEED)
    agent = get_agent(env, AgentType.RANDOM)
    positions = get_walkable_minigrid_positions(get_walkable_mask(env))
    starts = [(tuple(p.tolist()), d) for p in positions for d in range(n_dirs)]

    obs_rows, act_rows, pos_rows, dir_rows = [], [], [], []
    with eval_mode([pN]), on_device([pN], "cpu"):
        for pos, direction in starts:
            env.env.unwrapped.agent_start_pos = pos
            env.env.unwrapped.agent_start_dir = direction
            obs, act, state, _ = pN.collectObservationSequence(
                env=env, agent=agent, tsteps=n_steps, discretize=True
            )
            obs_rows.append(obs.squeeze(0)); act_rows.append(act.squeeze(0))
            pos_rows.append(np.asarray(state["agent_pos"]))
            dir_rows.append(np.asarray(state["agent_dir"]))
    return Probe(obs=torch.stack(obs_rows), act=torch.stack(act_rows),
                 agent_pos=np.stack(pos_rows).astype(np.int32),
                 agent_dir=np.stack(dir_rows).astype(np.int32),
                 seed=PROBE_SEED, env_name=env_name)


def session_maps(*, ckpt_dir: str, env_name: str, obj, n_dirs: int = 2):
    """Occupancy-masked rate maps for one session's checkpoint in its own env."""
    from curious_george import get_pN
    from scripts import trace_maps as tm, trace_probe as tp

    args, env = _setup(env_name, obj)
    pN = get_pN(args=args, env=env, device="cpu",
                pRNN_ckpt=str(Path(ckpt_dir) / "predictiveNet_state.pt"))
    pN.wandb_log = False

    probe = _build_probe(pN=pN, env=env, env_name=env_name, n_dirs=n_dirs)
    h = tp.replay_checkpoint(pN=pN, probe=probe).detach().numpy()
    pos = probe.agent_pos[:, : h.shape[1], :]
    h_rows = h[:, ONSET:, :].reshape(-1, h.shape[-1])
    pos_rows = pos[:, ONSET:, :].reshape(-1, 2).astype(np.float64)
    maps, occ, valid = tm.occupancy_and_maps(h=h_rows, pos=pos_rows, env=env)
    return maps, occ, valid, env


def analyse(manifest_path: str, n_dirs: int = 2) -> dict:
    from scripts.trace_metric import _disc_masks, field_gain
    from scripts.analysis_OMT import get_walkable_mask

    sessions = json.loads(Path(manifest_path).read_text())
    sessions = [s for s in sessions if s.get("run_dir")]
    env_name = "MiniGrid-SquareRoom-v0"

    _, env0 = _setup(env_name, None)
    mask = get_walkable_mask(env0).numpy()
    cells = [(int(x), int(y)) for x in range(mask.shape[0]) for y in range(mask.shape[1])
             if mask[x, y]]
    discs = _disc_masks(env=env0, cells=cells, radius=RADIUS)

    obj_positions = [tuple(s["object"]) if s["object"] else None for s in sessions]
    all_maps, gains, pcts = [], [], []

    for s, obj in zip(sessions, obj_positions):
        print(f"  session {s['session']}: object={obj}  {s['run_dir']}", flush=True)
        maps, occ, valid, _ = session_maps(ckpt_dir=s["run_dir"], env_name=env_name,
                                           obj=obj, n_dirs=n_dirs)
        g = field_gain(maps=maps, discs=discs)          # (n_cells, H)
        # Within-unit percentile: rank of each cell among that unit's own cells.
        p = np.full_like(g, np.nan)
        for u in range(g.shape[1]):
            col = g[:, u]
            ok = np.isfinite(col)
            if ok.sum() > 1:
                p[ok, u] = 100.0 * (col[ok][:, None] > col[ok][None, :]).mean(axis=1)
        all_maps.append(maps); gains.append(g); pcts.append(p)

    gains = np.stack(gains)      # (S, n_cells, H)
    pcts = np.stack(pcts)
    idx = {c: i for i, c in enumerate(cells)}

    H = gains.shape[2]
    has_field = pcts > PCTILE                       # (S, n_cells, H)
    baseline = has_field[0]                         # session 0: no object ever yet

    object_cells, trace_cells, detail = [], [], []
    for k, obj in enumerate(obj_positions):
        cur = has_field[k, idx[obj]] if obj else np.zeros(H, bool)
        prev = [p for p in obj_positions[:k] if p is not None and p != obj]
        # a trace needs a field now that was NOT there before any object existed
        tr = np.zeros(H, bool)
        for p in prev:
            tr |= has_field[k, idx[p]] & ~baseline[idx[p]]
        tr &= ~cur                                   # Moser: trace cells do not fire at the present object
        object_cells.append(cur); trace_cells.append(tr)
        detail.append({"session": k, "object": obj,
                       "n_object_cells": int(cur.sum()), "frac_object": float(cur.mean()),
                       "n_trace_cells": int(tr.sum()), "frac_trace": float(tr.mean()),
                       "prev_positions": prev})

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "moser_maps.npz",
                        maps=np.stack(all_maps), gains=gains, pcts=pcts,
                        cells=np.array(cells),
                        objects=np.array([o if o else (-1, -1) for o in obj_positions]),
                        object_cells=np.stack(object_cells), trace_cells=np.stack(trace_cells))
    (OUT / "moser_summary.json").write_text(json.dumps(detail, indent=2))
    return {"detail": detail, "objects": obj_positions, "cells": cells}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/moser_manifest.json")
    ap.add_argument("--n-dirs", type=int, default=2)
    args = ap.parse_args()
    res = analyse(args.manifest, args.n_dirs)
    print(f"\n{'session':>8} {'object':>10} {'object cells':>14} {'trace cells':>13}")
    for d in res["detail"]:
        o = "none" if d["object"] is None else f"{d['object'][0]},{d['object'][1]}"
        print(f"{d['session']:>8} {o:>10} "
              f"{d['n_object_cells']:>6} ({d['frac_object']:>5.1%}) "
              f"{d['n_trace_cells']:>5} ({d['frac_trace']:>5.1%})")
    print(f"\nchance rate is 5.0% by construction (within-unit 95th percentile)")
    print(f"wrote {OUT}/moser_maps.npz and moser_summary.json")


if __name__ == "__main__":
    main()
