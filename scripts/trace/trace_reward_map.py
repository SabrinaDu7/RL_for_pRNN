"""Curiosity-reward map per checkpoint, for the two remaining object locations.

Prediction MSE is the quantity compute_curious_rewards is built from
(curious_george/rl/update/rewards.py). Measured in the object-PRESENT env on a
probe whose trajectories are identical to the object-absent one (the floor tile
cannot alter the path for a fixed action sequence).
"""
import numpy as np
import torch
from pathlib import Path

from hydra import initialize_config_dir, compose
from prnn.utils import MinigridEnvNames, ActionEncodingsEnum, AgentInputType
from curious_george import make_env, get_pN, resolve_prnn_ckpt
from scripts.trace.trace_probe import build_probe, save_probe, load_probe, PROBE_SEED
from scripts.trace import trace_maps as tm
from scripts.analysis_OMT import get_walkable_mask, get_walkable_minigrid_positions

RUNS = {(14, 7): "outputs/mila_omt_dense/omt-cur-dot-0730-165922",
        (7, 2):  "outputs/mila_omt_dense/omt-cur-dot-0730-165916"}
ONSET, BS = 20, 128
CFG = "/home/sabrina/Documents/experiments/RL_for_pRNN/Configs"


def mse_per_step(pN, probe):
    saved = pN.trainNoiseMeanStd
    pN.trainNoiseMeanStd = (0.0, 0.0)
    try:
        pN.pRNN.eval()
        out = []
        with torch.no_grad():
            for lo in range(0, probe.n_trajs, BS):
                hi = min(lo + BS, probe.n_trajs)
                torch.manual_seed(PROBE_SEED + lo)
                init = pN.pRNN.rnn.cell.actfun(torch.zeros(hi - lo, 1, pN.hidden_size))
                op, on, _ = pN.predict(probe.obs[lo:hi], probe.act[lo:hi],
                                       state=init, randInit=False, batched=True)
                out.append(((op[0].permute(2, 0, 1) - on[0].permute(2, 0, 1)) ** 2).mean(-1))
    finally:
        pN.trainNoiseMeanStd = saved
    return torch.cat(out).numpy()


with initialize_config_dir(config_dir=CFG, version_base=None):
    args = compose(config_name="main")

for LOC, RUN in RUNS.items():
    env = make_env(env_key=MinigridEnvNames.LRoom, new_obj_pos=list(LOC),
                   input_type=AgentInputType.H_PO.value,
                   act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0)
    pdir = f"outputs/trace/probe_lroom_obj{LOC[0]}_{LOC[1]}"
    if not Path(f"{pdir}/probe.pt").exists():
        seed_pN = get_pN(args=args, env=env, device="cpu",
                         pRNN_ckpt="outputs/ckpts/pRNN_curious_26-07-23-10-06-25/predictiveNet_state.pt")
        save_probe(build_probe(pN=seed_pN, env=env, n_steps=256, seed=PROBE_SEED), pdir)
    probe = load_probe(pdir)

    cells = [tuple(p.tolist()) for p in get_walkable_minigrid_positions(get_walkable_mask(env))]
    pos = probe.agent_pos[:, ONSET:probe.n_steps, :].astype(np.float64).reshape(-1, 2)
    steps = sorted(int(s.name) for s in Path(RUN).iterdir() if s.is_dir())

    print(f"\n=== object at {LOC}, run {Path(RUN).name}", flush=True)
    print(f"{'trajs':>6} | {'MSE @ obj':>10} {'room median':>12} {'ratio':>7} | {'%ile':>6}", flush=True)
    rows = []
    for s in steps:
        pN = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=resolve_prnn_ckpt(f"{RUN}/{s}"))
        e = mse_per_step(pN, probe)[:, ONSET:]
        maps, _, _ = tm.occupancy_and_maps(h=e.reshape(-1, 1), pos=pos, env=env)
        vals = np.array([maps[0][c[1] - 1, c[0] - 1] for c in cells])
        at, med = float(vals[cells.index(LOC)]), float(np.nanmedian(vals))
        pct = 100.0 * float(np.mean(vals[np.isfinite(vals)] < at))
        rows.append((s, at, med, at / med, pct))
        print(f"{s:>6} | {at:>10.5f} {med:>12.5f} {at/med:>7.2f} | {pct:>6.1f}", flush=True)
    np.save(f"outputs/trace/rewardmap_{LOC[0]}_{LOC[1]}.npy", np.array(rows))
print("DONE", flush=True)
