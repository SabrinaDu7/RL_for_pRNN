"""Gain-corrected readout swap.

||h|| is ~9% smaller in the trained nets than in the baseline, and ||W_out|| is
~2% larger - so a raw W_out transplant onto baseline dynamics OVER-drives the
prediction, and the reverse transplant UNDER-drives it. That biases both
chimaeras in the direction of "the readout carries it", which is exactly the
conclusion drawn from them. Rescale W_out by the measured ||h|| ratio so each
chimaera is driven at the gain its readout was trained for, and re-measure.
"""
import copy
from pathlib import Path

import numpy as np
import torch
from hydra import initialize_config_dir, compose
from prnn.utils import MinigridEnvNames, ActionEncodingsEnum, AgentInputType
from curious_george import make_env, get_pN, resolve_prnn_ckpt
from scripts.trace_probe import load_probe, replay_checkpoint
from scripts.trace_readout_test import object_contrast

RUNS = {(7, 11): ["165325", "172405", "175326"],
        (14, 7): ["165922", "172845", "175805"],
        (7, 2):  ["165916", "173016", "180402"]}
CTRL = [[2, 5], [4, 7], [11, 3], [13, 5]]
BASE = "outputs/ckpts/pRNN_curious_26-07-23-10-06-25/predictiveNet_state.pt"
ONSET, H = 20, 500
CFG = "/home/sabrina/Documents/experiments/RL_for_pRNN/Configs"

with initialize_config_dir(config_dir=CFG, version_base=None):
    args = compose(config_name="main")
env = make_env(env_key=MinigridEnvNames.LRoom, input_type=AgentInputType.H_PO.value,
               act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0)
probe = load_probe("outputs/trace/probe_lroom_noobj")


def mean_h_norm(pN):
    h = replay_checkpoint(pN=pN, probe=probe)[:, ONSET:, :].numpy().reshape(-1, H)
    return float(np.linalg.norm(h, axis=1).mean())


def swap_scaled(dynamics, readout, scale):
    net = copy.deepcopy(dynamics)
    sd = net.pRNN.state_dict()
    sd["W_out"] = readout.pRNN.state_dict()["W_out"].clone() * scale
    net.pRNN.load_state_dict(sd)
    return net


base = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=BASE)
nb = mean_h_norm(base)
print(f"||h|| baseline = {nb:.3f}\n")
print(f"{'object':>8} {'run':>8} | {'full':>8} {'readout*':>9} {'dyn*':>8} | {'share*':>7}")
print("-" * 60)
rows = []
for loc, tags in RUNS.items():
    for tag in tags:
        d = next(p for p in Path("outputs/mila_omt_dense").iterdir() if tag in p.name)
        step = max(int(s.name) for s in d.iterdir() if s.is_dir())
        tr = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=resolve_prnn_ckpt(f"{d}/{step}"))
        nt = mean_h_norm(tr)
        r = nt / nb                       # trained h is smaller by this factor
        kw = dict(probe=probe, env=env, obj_loc=loc, ctrl_locs=CTRL, n_trajs=200)
        b = object_contrast(pN=base, **kw)
        full = object_contrast(pN=tr, **kw) - b
        # trained W_out onto LARGER baseline h -> scale it DOWN by r
        rd = object_contrast(pN=swap_scaled(base, tr, r), **kw) - b
        # baseline W_out onto SMALLER trained h -> scale it UP by 1/r
        dy = object_contrast(pN=swap_scaled(tr, base, 1.0 / r), **kw) - b
        rows.append((loc, tag, full, rd, dy, rd / full if full else np.nan, r))
        print(f"{str(loc):>8} {tag:>8} | {full:>+8.4f} {rd:>+9.4f} {dy:>+8.4f} | "
              f"{100*rd/full:>6.0f}%")
a = np.array([[x[2], x[3], x[4]] for x in rows])
print(f"\nfull {a[:,0].mean():+.4f} +/- {a[:,0].std():.4f}")
print(f"readout* {a[:,1].mean():+.4f} +/- {a[:,1].std():.4f}")
print(f"dynamics* {a[:,2].mean():+.4f} +/- {a[:,2].std():.4f}")
print(f"readout* > dynamics* in {int((a[:,1] > a[:,2]).sum())}/9 runs")
np.save("outputs/trace/readout_gaincorrected.npy", np.array(rows, dtype=object), allow_pickle=True)
print("DONE", flush=True)
