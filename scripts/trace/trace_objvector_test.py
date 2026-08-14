"""Object-vector coding test + the >100% readout-share anomaly check.

Part 1: does exposure change tuning in the OBJECT-RELATIVE frame (the object's
position in the agent's 7x7 view) even though it leaves the allocentric place
code alone? Control: the same analysis for tasks.testing.ctrl_locs, which are
not objects, so any generic "landmark in view" tuning shows up there too.

Part 2: is the >100% readout share a gain mismatch? Compare ||h|| between the
baseline and trained nets on the same probe.
"""
import numpy as np
import torch
from pathlib import Path

from hydra import initialize_config_dir, compose
from prnn.utils import MinigridEnvNames, ActionEncodingsEnum, AgentInputType
from curious_george import make_env, get_pN, resolve_prnn_ckpt
from scripts.trace.trace_probe import load_probe, replay_checkpoint, PROBE_SEED
from scripts.trace import trace_maps as tm
from tasks.omt.metrics import get_view_coords_batch

RUNS = {(7, 11): "outputs/mila_omt_dense/omt-cur-dot-0730-165325",
        (14, 7): "outputs/mila_omt_dense/omt-cur-dot-0730-165922",
        (7, 2):  "outputs/mila_omt_dense/omt-cur-dot-0730-165916"}
CTRL = [[2, 5], [4, 7], [11, 3], [13, 5]]
BASE = "outputs/ckpts/pRNN_curious_26-07-23-10-06-25"
ONSET, H = 20, 500
# Resolved from the working directory, not from one machine's home: an
# absolute path here silently breaks for every other checkout.
CFG = str(Path("Configs").resolve())

with initialize_config_dir(config_dir=CFG, version_base=None):
    args = compose(config_name="main")


def view_si(h_rows, probe, cell, T):
    """SI of each unit in the object-relative frame, for one landmark cell."""
    pos = probe.agent_pos[:, ONSET:ONSET + T, :].reshape(-1, 2)
    hd = probe.agent_dir[:, ONSET:ONSET + T].reshape(-1)
    vx, vy = get_view_coords_batch(cell[0], cell[1], pos, hd)
    m = (vx >= 0) & (vx < 7) & (vy >= 0) & (vy < 7)
    maps, occ, _ = tm.view_frame_maps(h=h_rows[m], vx=vx[m], vy=vy[m])
    return tm.spatial_info(maps=maps, occupancy=occ), maps, int(m.sum())


print("=" * 74)
print("PART 1  object-vector frame: SI change from baseline (median over units)")
print("=" * 74)
for LOC, RUN in RUNS.items():
    env = make_env(env_key=MinigridEnvNames.LRoom, new_obj_pos=list(LOC),
                   input_type=AgentInputType.H_PO.value,
                   act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0)
    probe = load_probe(f"outputs/trace/probe_lroom_obj{LOC[0]}_{LOC[1]}")
    step = max(int(s.name) for s in Path(RUN).iterdir() if s.is_dir())

    hs = {}
    for tag, ck in (("base", f"{BASE}/predictiveNet_state.pt"),
                    ("trained", resolve_prnn_ckpt(f"{RUN}/{step}"))):
        pN = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=ck)
        hh = replay_checkpoint(pN=pN, probe=probe)
        hs[tag] = hh[:, ONSET:, :].numpy().reshape(-1, H)
        T = hh.shape[1] - ONSET

    print(f"\nobject at {LOC}  (run {Path(RUN).name}, step {step})")
    print(f"  {'landmark':>12} {'SI base':>9} {'SI trained':>11} {'delta':>9} {'n in-view':>10}")
    si_b, _, n = view_si(hs["base"], probe, LOC, T)
    si_t, mp_t, _ = view_si(hs["trained"], probe, LOC, T)
    d_obj = float(np.nanmedian(si_t - si_b))
    print(f"  {'OBJECT':>12} {np.nanmedian(si_b):>9.4f} {np.nanmedian(si_t):>11.4f} "
          f"{d_obj:>+9.4f} {n:>10}")
    dc = []
    for c in CTRL:
        sb, _, nc = view_si(hs["base"], probe, tuple(c), T)
        st, _, _ = view_si(hs["trained"], probe, tuple(c), T)
        dc.append(float(np.nanmedian(st - sb)))
        print(f"  {str(tuple(c)):>12} {np.nanmedian(sb):>9.4f} {np.nanmedian(st):>11.4f} "
              f"{dc[-1]:>+9.4f} {nc:>10}")
    print(f"  -> OBJECT minus control mean: {d_obj - float(np.mean(dc)):+.4f}")
    np.savez_compressed(f"outputs/trace/objvector_{LOC[0]}_{LOC[1]}.npz",
                        si_base=si_b, si_trained=si_t, maps_trained=mp_t)

print()
print("=" * 74)
print("PART 2  is the >100% readout share a gain mismatch?")
print("=" * 74)
env = make_env(env_key=MinigridEnvNames.LRoom, input_type=AgentInputType.H_PO.value,
               act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0)
probe0 = load_probe("outputs/trace/probe_lroom_noobj")
pb = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=f"{BASE}/predictiveNet_state.pt")
hb = replay_checkpoint(pN=pb, probe=probe0)[:, ONSET:, :].numpy().reshape(-1, H)
print(f"{'run':>10} {'||h|| base':>11} {'||h|| trained':>14} {'ratio':>7} "
      f"{'||W_out|| ratio':>16}")
for LOC, RUN in RUNS.items():
    step = max(int(s.name) for s in Path(RUN).iterdir() if s.is_dir())
    pt = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=resolve_prnn_ckpt(f"{RUN}/{step}"))
    ht = replay_checkpoint(pN=pt, probe=probe0)[:, ONSET:, :].numpy().reshape(-1, H)
    nb, nt = float(np.linalg.norm(hb, axis=1).mean()), float(np.linalg.norm(ht, axis=1).mean())
    wb = float(torch.norm(pb.pRNN.state_dict()["W_out"]))
    wt = float(torch.norm(pt.pRNN.state_dict()["W_out"]))
    print(f"{str(LOC):>10} {nb:>11.3f} {nt:>14.3f} {nt/nb:>7.3f} {wt/wb:>16.3f}")
print("DONE", flush=True)
